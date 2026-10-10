"""Reproduce and extend the two-patient worked example in Table 1.

The manuscript gives the length of each record, each patient's event count,
and a constant forecast equal to that patient's event rate. It does not give
the calendar-day labels or the order of the event days. Since the forecast is
constant within patient, the metrics below depend only on these grouped counts;
the script deliberately does not invent a day-by-day outcome order.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "open_data" / "results" / "v10"
NB_THRESHOLD = 0.37


# Table 1's prose specifies 30 days per patient, with 20 and 2 attack days.
# The forecast is each patient's own attack rate; 0.67 and 0.07 are the
# manuscript's two-decimal display values for 20/30 and 2/30.
records = pd.DataFrame(
    [
        {"patient": "A", "days": 30, "events": 20},
        {"patient": "B", "days": 30, "events": 2},
    ]
)
records["non_events"] = records["days"] - records["events"]
records["forecast"] = records["events"] / records["days"]
records["event_rate"] = records["events"] / records["days"]


def auc_from_groups(grouped: pd.DataFrame) -> float:
    """AUC with half credit for ties, using score-group event counts."""
    positive_total = int(grouped["events"].sum())
    negative_total = int(grouped["non_events"].sum())
    if positive_total == 0 or negative_total == 0:
        return float("nan")

    concordant = 0.0
    negatives_at_higher_scores = 0
    for row in grouped.sort_values("forecast", ascending=False).itertuples():
        positives = int(row.events)
        negatives = int(row.non_events)
        negatives_at_lower_scores = negative_total - negatives_at_higher_scores - negatives
        concordant += positives * (negatives_at_lower_scores + 0.5 * negatives)
        negatives_at_higher_scores += negatives
    return concordant / (positive_total * negative_total)


def average_precision_from_groups(grouped: pd.DataFrame) -> float:
    """Non-interpolated average precision, accumulated at distinct scores."""
    positive_total = int(grouped["events"].sum())
    if positive_total == 0:
        return float("nan")

    true_positives = 0
    false_positives = 0
    average_precision = 0.0
    for row in grouped.sort_values("forecast", ascending=False).itertuples():
        previous_recall = true_positives / positive_total
        true_positives += int(row.events)
        false_positives += int(row.non_events)
        recall = true_positives / positive_total
        precision = true_positives / (true_positives + false_positives)
        average_precision += (recall - previous_recall) * precision
    return average_precision


def random_order_ap_chance(days: int, events: int) -> float | None:
    """Expected AP over all no-tie random orderings with fixed event count."""
    if days == 0 or events == 0:
        return None
    ranks = np.arange(1, days + 1, dtype=float)
    if days == 1:
        return 1.0
    expected_precision = (1.0 + (ranks - 1.0) * (events - 1.0) / (days - 1.0)) / ranks
    return float(np.mean(expected_precision))


def logistic_slope(grouped: pd.DataFrame) -> float:
    """Fit pooled logistic calibration intercept and slope to grouped data."""
    x = np.column_stack(
        [np.ones(len(grouped)), np.log(grouped["forecast"].to_numpy() / (1 - grouped["forecast"].to_numpy()))]
    )
    n = grouped["days"].to_numpy(dtype=float)
    y = grouped["events"].to_numpy(dtype=float)
    beta = np.array([0.0, 1.0])

    for _ in range(100):
        eta = x @ beta
        fitted = 1.0 / (1.0 + np.exp(-eta))
        weights = n * fitted * (1.0 - fitted)
        score = x.T @ (y - n * fitted)
        information = x.T @ (x * weights[:, None])
        step = np.linalg.solve(information, score)
        beta += step
        if np.max(np.abs(step)) < 1e-12:
            break
    return float(beta[1])


def defined_mean(values: list[float | None]) -> float | None:
    defined = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(defined)) if defined else None


def score_one_patient(row: pd.Series) -> dict[str, float | None]:
    one = pd.DataFrame([row])
    n = int(row["days"])
    events = int(row["events"])
    non_events = int(row["non_events"])
    rate = float(row["event_rate"])

    # The manuscript identifies the top forecast level (shown as 0.67) as the
    # alarm group. The population-rate threshold in this example selects the
    # same group; net benefit below uses the specified t=0.37 exactly.
    alarm = bool(row["forecast"] == records["forecast"].max())
    tp = events if alarm else 0
    fp = non_events if alarm else 0
    fn = events - tp
    tn = non_events - fp
    sensitivity = tp / events if events else None
    specificity = tn / non_events if non_events else None
    ppv = tp / (tp + fp) if tp + fp else None
    npv = tn / (tn + fn) if tn + fn else None

    model_brier = (
        events * (1.0 - float(row["forecast"])) ** 2
        + non_events * float(row["forecast"]) ** 2
    ) / n
    own_rate_brier = rate * (1.0 - rate)
    brier_skill = 1.0 - model_brier / own_rate_brier if own_rate_brier > 0 else None

    model_net_benefit = (tp - fp * NB_THRESHOLD / (1.0 - NB_THRESHOLD)) / n
    best_own_rate_policy = (
        rate - (1.0 - rate) * NB_THRESHOLD / (1.0 - NB_THRESHOLD)
        if rate >= NB_THRESHOLD
        else 0.0
    )
    incremental_net_benefit = model_net_benefit - best_own_rate_policy

    # The slope within a patient is not estimable because the forecast is
    # constant for that patient. AUC/AP remain defined because each patient
    # has both outcomes; ties receive half credit for AUC.
    return {
        "AUC": auc_from_groups(one),
        "AP": average_precision_from_groups(one),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "Net benefit": incremental_net_benefit,
        "Brier skill": brier_skill,
        "Calibration slope": None,
    }


patient_scores = {
    row.patient: score_one_patient(pd.Series(row._asdict()))
    for row in records.itertuples(index=False)
}

pooled_n = int(records["days"].sum())
pooled_events = int(records["events"].sum())
pooled_non_events = int(records["non_events"].sum())
population_rate = pooled_events / pooled_n
pooled_forecast_brier = float(
    np.mean(
        [
            (
                int(row.events) * (1.0 - float(row.forecast)) ** 2
                + int(row.non_events) * float(row.forecast) ** 2
            )
            / int(row.days)
            for row in records.itertuples(index=False)
        ]
    )
)
population_rate_brier = population_rate * (1.0 - population_rate)
pooled_brier_skill = 1.0 - pooled_forecast_brier / population_rate_brier

pooled_tp = int(
    records.loc[records["forecast"] >= NB_THRESHOLD, "events"].sum()
)
pooled_fp = int(
    records.loc[records["forecast"] >= NB_THRESHOLD, "non_events"].sum()
)
pooled_fn = pooled_events - pooled_tp
pooled_tn = pooled_non_events - pooled_fp
pooled_sensitivity = pooled_tp / pooled_events
pooled_specificity = pooled_tn / pooled_non_events
pooled_ppv = pooled_tp / (pooled_tp + pooled_fp)
pooled_npv = pooled_tn / (pooled_tn + pooled_fn)
pooled_net_benefit = (
    pooled_tp - pooled_fp * NB_THRESHOLD / (1.0 - NB_THRESHOLD)
) / pooled_n
pooled_auc = auc_from_groups(records)
pooled_ap = average_precision_from_groups(records)
pooled_slope = logistic_slope(records)
ap_chance_levels = {
    row.patient: random_order_ap_chance(int(row.days), int(row.events))
    for row in records.itertuples(index=False)
}


# Reproduce the values printed in the v7 Table 1 at their printed precision.
assert records["days"].tolist() == [30, 30]
assert records["events"].tolist() == [20, 2]
assert round(float(records.loc[0, "forecast"]), 2) == 0.67
assert round(float(records.loc[1, "forecast"]), 2) == 0.07
assert round(population_rate, 2) == 0.37
assert round(pooled_auc, 2) == 0.82
assert all(round(float(patient_scores[p]["AUC"]), 2) == 0.50 for p in ("A", "B"))
assert round(pooled_sensitivity * 100, 0) == 91
assert round(pooled_specificity * 100, 0) == 74
assert round(float(patient_scores["A"]["Sensitivity"]) * 100, 0) == 100
assert round(float(patient_scores["B"]["Sensitivity"]) * 100, 0) == 0
assert round(float(patient_scores["A"]["Specificity"]) * 100, 0) == 0
assert round(float(patient_scores["B"]["Specificity"]) * 100, 0) == 100
assert round(float(patient_scores["A"]["Sensitivity"]) + float(patient_scores["A"]["Specificity"]) - 1, 2) == 0
assert round(float(patient_scores["B"]["Sensitivity"]) + float(patient_scores["B"]["Specificity"]) - 1, 2) == 0
assert round(pooled_ppv, 2) == 0.67
assert round(float(patient_scores["A"]["PPV"]), 2) == 0.67
assert patient_scores["B"]["PPV"] is None
assert round(pooled_brier_skill * 100, 0) == 39
assert round(float(patient_scores["A"]["Brier skill"]), 2) == 0.00
assert round(float(patient_scores["B"]["Brier skill"]), 2) == 0.00
assert round(pooled_slope, 2) == 1.00
assert patient_scores["A"]["Calibration slope"] is None
assert patient_scores["B"]["Calibration slope"] is None

# Checks for the added quantities.
assert round(pooled_ap, 3) == 0.639
assert round(float(patient_scores["A"]["AP"]), 3) == 0.667
assert round(float(patient_scores["B"]["AP"]), 3) == 0.067
assert round(pooled_npv, 2) == 0.93
assert patient_scores["A"]["NPV"] is None
assert round(float(patient_scores["B"]["NPV"]), 2) == 0.93
assert round(pooled_net_benefit, 3) == 0.235
assert round(float(patient_scores["A"]["Net benefit"]), 3) == 0.000
assert round(float(patient_scores["B"]["Net benefit"]), 3) == 0.000


def row_values(metric: str, pooled_value: float) -> dict[str, float | None]:
    per_patient = [patient_scores[p][metric] for p in ("A", "B")]
    return {
        "patient_A": per_patient[0],
        "patient_B": per_patient[1],
        "per_patient_mean": defined_mean(per_patient),
        "pooled": pooled_value,
    }


metric_values: dict[str, dict[str, float | None]] = {
    "AUC": row_values("AUC", pooled_auc),
    "AP": row_values("AP", pooled_ap),
    "Sensitivity": row_values("Sensitivity", pooled_sensitivity),
    "Specificity": row_values("Specificity", pooled_specificity),
    "PPV": row_values("PPV", pooled_ppv),
    "NPV": row_values("NPV", pooled_npv),
    "Net benefit": row_values("Net benefit", pooled_net_benefit),
    "Brier skill": row_values("Brier skill", pooled_brier_skill),
    "Calibration slope": row_values("Calibration slope", pooled_slope),
}

ppv_excess = {
    "patient_A": float(patient_scores["A"]["PPV"]) - float(records.loc[0, "event_rate"]),
    "patient_B": None,
    "per_patient_mean": float(patient_scores["A"]["PPV"]) - float(records.loc[0, "event_rate"]),
    "pooled": pooled_ppv - population_rate,
}
npv_excess = {
    "patient_A": None,
    "patient_B": float(patient_scores["B"]["NPV"]) - (1.0 - float(records.loc[1, "event_rate"])),
    "per_patient_mean": float(patient_scores["B"]["NPV"]) - (1.0 - float(records.loc[1, "event_rate"])),
    "pooled": pooled_npv - (1.0 - population_rate),
}

metric_specs = {
    "AUC": {"digits": 2, "unit": ""},
    "AP": {"digits": 3, "unit": ""},
    "Sensitivity": {"digits": 0, "unit": "%", "scale": 100},
    "Specificity": {"digits": 0, "unit": "%", "scale": 100},
    "PPV": {"digits": 2, "unit": "", "excess": ppv_excess, "excess_name": "event_rate"},
    "NPV": {"digits": 2, "unit": "", "excess": npv_excess, "excess_name": "non_event_rate"},
    "Net benefit": {"digits": 3, "unit": ""},
    "Brier skill": {"digits": 0, "unit": "%", "scale": 100},
    "Calibration slope": {"digits": 2, "unit": ""},
}

columns = ["patient_A", "patient_B", "per_patient_mean", "pooled"]
column_names = {
    "patient_A": "patient A",
    "patient_B": "patient B",
    "per_patient_mean": "per-patient mean",
    "pooled": "pooled",
}


def rounded_metric_value(metric: str, column: str) -> float | str:
    value = metric_values[metric][column]
    if value is None or not np.isfinite(value):
        return "n/a"
    spec = metric_specs[metric]
    scaled = value * spec.get("scale", 1)
    rounded = round(float(scaled), spec["digits"])
    return 0.0 if rounded == 0 else rounded


json_metrics: list[dict[str, object]] = []
markdown_lines = [
    "# Table 1. One forecast, two patients",
    "",
    "| Metric | patient A | patient B | per-patient mean | pooled |",
    "| --- | ---: | ---: | ---: | ---: |",
]

for metric in metric_values:
    spec = metric_specs[metric]
    json_row: dict[str, object] = {"metric": metric, "unit": spec["unit"] or "value"}
    markdown_cells: list[str] = []
    for column in columns:
        value = rounded_metric_value(metric, column)
        if value == "n/a":
            json_row[column] = "n/a"
            markdown_cells.append("n/a")
            continue

        if metric in ("PPV", "NPV"):
            excess_value = ppv_excess[column] if metric == "PPV" else npv_excess[column]
            rounded_excess = round(float(excess_value), 2) if excess_value is not None else None
            json_row[column] = {"value": value, "excess_over_reference": rounded_excess}
            markdown_cells.append(f"{value:.2f} (excess {rounded_excess:+.2f})")
        elif spec["unit"] == "%":
            json_row[column] = value
            markdown_cells.append(f"{value:.0f}%")
        else:
            json_row[column] = value
            markdown_cells.append(f"{value:.{spec['digits']}f}")

    markdown_lines.append(f"| {metric} | " + " | ".join(markdown_cells) + " |")
    json_metrics.append(json_row)

undefined_note = (
    "*n/a:* PPV is undefined for patient B because no days are alarmed; NPV is undefined for patient A "
    "because no days are unalarmed. Per-patient calibration slopes and their mean are undefined because each "
    "patient's forecast is constant, so there is no within-patient forecast variation."
)
records_note = (
    "*Record detail:* the manuscript reports 30 days with 20 events for A and 30 days with 2 events for B, "
    "and forecasts equal to each patient's own rate (2/3 and 2/30; displayed as 0.67 and 0.07). It does not "
    "report event-day labels or their order. The reported grouped counts are sufficient because forecasts are constant "
    "within patient; no day order has been imputed."
)
methods_note = (
    "*Definitions:* AP is non-interpolated average precision at distinct forecast levels; its expected chance "
    "references for random orderings without ties are 0.701 for A and 0.163 for B. PPV/NPV cells show "
    "the value and its excess over the patient's own event/non-event rate (pooled: over the population rate). "
    "Per-patient net benefit is the model's net benefit minus the best policy supported by that patient's own "
    "event rate; pooled net benefit is standard net benefit at t = 0.37. Per-patient Brier skill uses each own "
    "rate as reference; pooled Brier skill uses the population rate. The population event rate is 0.37 when "
    "rounded as in the manuscript. The high forecast group (shown as 0.67) is the alarm group; threshold t = 0.37 "
    "selects the same days for net benefit. Youden's index is 0 in each patient, as printed in Table 1."
)
markdown_lines.extend(["", undefined_note, "", records_note, "", methods_note, ""])
markdown = "\n".join(markdown_lines)

plain_footnotes = [
    "n/a: PPV is undefined for patient B because no days are alarmed; NPV is undefined for patient A because no days are unalarmed. Per-patient calibration slopes and their mean are undefined because each patient's forecast is constant, so there is no within-patient forecast variation.",
    "Record detail: the manuscript reports 30 days with 20 events for A and 30 days with 2 events for B, and forecasts equal to each patient's own rate (2/3 and 2/30; displayed as 0.67 and 0.07). It does not report event-day labels or their order. The grouped counts are sufficient because forecasts are constant within patient; no day order has been imputed.",
    "Definitions: AP is non-interpolated average precision at distinct forecast levels; its expected chance references for random orderings without ties are 0.701 for A and 0.163 for B (per-patient mean 0.432). PPV/NPV cells show the value and its excess over the patient's own event/non-event rate (pooled: over the population rate). Per-patient net benefit is the model's net benefit minus the best policy supported by that patient's own event rate; pooled net benefit is standard net benefit at t = 0.37. Per-patient Brier skill uses each own rate as reference; pooled Brier skill uses the population rate. The population event rate is 0.37 when rounded as in the manuscript. The high forecast group (shown as 0.67) is the alarm group; threshold t = 0.37 selects the same days for net benefit. Youden's index is 0 in each patient, as printed in Table 1.",
]

json_payload = {
    "title": "Table 1. One forecast, two patients",
    "threshold_for_net_benefit": NB_THRESHOLD,
    "record_summary": [
        {
            "patient": row.patient,
            "days": int(row.days),
            "events": int(row.events),
            "non_events": int(row.non_events),
            "forecast_probability": round(float(row.forecast), 6),
            "forecast_as_printed": round(float(row.forecast), 2),
        }
        for row in records.itertuples(index=False)
    ],
    "population_event_rate": round(population_rate, 6),
    "ap_patient_chance_references": {
        patient: round(float(value), 3) for patient, value in ap_chance_levels.items()
    },
    "metrics": json_metrics,
    "footnotes": plain_footnotes,
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "table1.md").write_text(markdown, encoding="utf-8")
(OUTPUT_DIR / "table1.json").write_text(
    json.dumps(json_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)

print(f"Wrote {OUTPUT_DIR / 'table1.md'}")
print(f"Wrote {OUTPUT_DIR / 'table1.json'}")
