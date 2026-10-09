"""Patient-cluster information audit for pooled repeated-forecast AUCs.

Run from the repository root with:
    python open_data/cases_information_audit.py

This script reuses the exported test forecasts. It does not refit the models.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import cases_predictive_values as predictive_values
import cases_same_patients as same_patients
import realtime_analysis


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results" / "information_audit"
OUT_JSON = OUT_DIR / "information_audit.json"
OUT_MD = OUT_DIR / "information_audit.md"
N_BOOT = 2000
SEED = 20261009
MIN_PATIENTS_PER_TIME = 5
TOL = 1e-9

COHORT_SETUP = {
    "REPLACE-BG": {
        "label": "REPLACE-BG",
        "same_patients_key": "REPLACE-BG",
        "clocks": {
            "night_since_enrollment": "Night since enrollment",
        },
    },
    "CTN": {
        "label": "CTN-0094",
        "same_patients_key": "CTN-0094",
        "clocks": {
            "study_week": "Study week since randomization",
        },
    },
    "CES": {
        "label": "College Experience",
        "same_patients_key": "College Experience",
        "clocks": {
            "follow_up_week": "Week since first complete report",
            "calendar_date": "Calendar date",
        },
    },
}


def finite_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    """Return the shared tie-aware AUC helper's value, or None if undefined."""
    value = same_patients.auc(np.asarray(y, dtype=int), np.asarray(score, dtype=float))
    return float(value) if np.isfinite(value) else None


def patient_auc_summary(frame: pd.DataFrame, score_col: str) -> tuple[float | None, int, list[dict]]:
    """Equal-weight patient AUC among patients observed with both outcomes."""
    values = []
    details = []
    for patient, group in frame.groupby("patient", sort=False):
        y = group["y"].to_numpy(dtype=int)
        if np.unique(y).size < 2:
            continue
        value = finite_auc(y, group[score_col].to_numpy(dtype=float))
        if value is not None:
            values.append(value)
            details.append({"patient": int(patient), "auc": value})
    return (float(np.mean(values)) if values else None, len(values), details)


def training_time_rates() -> dict:
    """Event rate per time point among the TRAINING patients (first 60% of each cohort's seeded split),
    used as the prospective time-only forecast for test patients. Time points with fewer than
    MIN_PATIENTS_PER_TIME training patients fall back to the overall training rate."""
    root = realtime_analysis.ROOT
    out = {}

    def first60(ids, seed, fractional=False):
        ids = np.array(sorted(pd.unique(ids)))
        rng_ = np.random.default_rng(seed); rng_.shuffle(ids); n = len(ids)
        a = int(round(.6 * n)) if fractional else int(.6 * n)
        return set(ids[:a])

    def rates(train, col):
        g = train.groupby(col).agg(rate=("y", "mean"), n=("pid", "nunique"))
        overall = float(train["y"].mean())
        return {k: (float(r) if n >= MIN_PATIENTS_PER_TIME else overall) for k, (r, n) in g[["rate", "n"]].iterrows()}, overall

    p = pd.read_csv(root / 'replace_bg_panel.csv'); p.pid = p.pid.astype(str); p.y = p.y.astype(int)
    p = p.loc[~p.model_runin.astype(bool)].copy(); tr = p.loc[p.pid.isin(first60(p.pid, 2026, True))].copy()
    tr["t"] = pd.to_numeric(tr.day)
    out["REPLACE-BG"] = {"night_since_enrollment": rates(tr, "t")}

    p = pd.read_csv(root / 'ctn0094_weekly_panel.csv')
    tr = p.loc[p.who.isin(first60(p.who, 20261006)) & p.event.notna()].copy()
    tr["y"] = tr.event.astype(int); tr["pid"] = tr.who; tr["t"] = tr.study_week
    out["CTN"] = {"study_week": rates(tr, "t")}

    raw = pd.read_csv(root / 'ces_openesm' / '0040_nepal_ts.tsv', sep='	', parse_dates=['date'])
    phq = ['nervous', 'worrying', 'depressed', 'anhedonia']; raw = raw.loc[raw[phq].notna().all(axis=1)].copy()
    raw['event'] = (raw[phq].sum(axis=1) >= 3).astype(int)
    raw = raw.sort_values(['id', 'date'], kind='mergesort').reset_index(drop=True)
    panel = []
    for pid, g in raw.groupby('id', sort=False):
        g = g.reset_index(drop=True); n = len(g)
        if n < 60: continue
        ix = np.arange(20, n - 1); q = g.iloc[ix][['date']].copy().reset_index(drop=True)
        q['pid'] = pid; q['y'] = g.event.to_numpy(int)[ix + 1]; panel.append(q)
    p = pd.concat(panel, ignore_index=True)
    ids = np.array(sorted(p.pid.unique()), dtype=object); rng_ = np.random.default_rng(20261006); rng_.shuffle(ids)
    tr = p.loc[p.pid.isin(set(ids[:int(.6 * len(ids))]))].copy()
    first = raw.groupby('id').date.min()
    tr['date'] = pd.to_datetime(tr.date).dt.normalize()
    tr['follow_up_week'] = (tr.date - tr.pid.map(first)).dt.days.astype(int) // 7
    out["CES"] = {"follow_up_week": rates(tr, "follow_up_week"), "calendar_date": rates(tr, "date")}
    return out


def time_metrics(frame: pd.DataFrame, time_col: str) -> dict:
    """Compute A0,time over all rows and its eligible per-time-point AUC."""
    grouped = frame.groupby(time_col, sort=True, dropna=False)
    frame = frame.copy()
    if "time_rate" not in frame.columns:
        raise ValueError("time_rate (training-cohort rate per time point) must be supplied")
    pooled_auc = finite_auc(frame["y"].to_numpy(dtype=int), frame["time_rate"].to_numpy(dtype=float))

    details = []
    eligible_scores = []
    eligible_rows = 0
    for time_value, group in frame.groupby(time_col, sort=True, dropna=False):
        n_patients = int(group["patient"].nunique())
        y = group["y"].to_numpy(dtype=int)
        n_events = int(y.sum())
        n_nonevents = int(len(y) - n_events)
        if n_patients < MIN_PATIENTS_PER_TIME or n_events == 0 or n_nonevents == 0:
            continue
        event_rate = float(group["time_rate"].iloc[0])
        value = finite_auc(y, group["time_rate"].to_numpy(dtype=float))
        if value is None:
            continue
        eligible_scores.append(value)
        eligible_rows += len(group)
        details.append({
            "time": str(time_value),
            "patients": n_patients,
            "observations": int(len(group)),
            "events": n_events,
            "event_rate": event_rate,
            "auc": value,
        })

    per_timepoint_auc = float(np.mean(eligible_scores)) if eligible_scores else None
    if per_timepoint_auc is not None and not np.isclose(per_timepoint_auc, 0.5, atol=1e-12, rtol=0):
        raise AssertionError(f"A0,time per-time-point AUC must be 0.5; got {per_timepoint_auc}")
    return {
        "pooled_auc": pooled_auc,
        "per_timepoint_auc": per_timepoint_auc,
        "eligible_timepoints": details,
        "n_eligible_timepoints": len(details),
        "n_eligible_observations": int(eligible_rows),
        "n_all_observations": int(len(frame)),
    }


def compute_metrics(frame: pd.DataFrame, clock_columns: dict[str, str], threshold: float) -> dict:
    """Recompute all requested quantities from one original or bootstrap sample."""
    pp = predictive_values.per_patient(frame, "gb", threshold)
    rate_by_patient = pp["rate"].to_dict()
    a0_score = frame["patient"].map(rate_by_patient).to_numpy(dtype=float)

    gb = frame["gb"].to_numpy(dtype=float)
    patient_mean = frame.groupby("patient", sort=False)["gb"].transform("mean").to_numpy(dtype=float)
    cohort_mean = float(gb.mean())
    who_score = patient_mean
    # Probability-scale centering; AUC uses these as rank scores even if shifted values
    # fall outside [0, 1]. The within-patient ordering is unchanged by each shift.
    when_score = gb - patient_mean + cohort_mean

    scored = frame.copy()
    scored["a0"] = a0_score
    scored["who"] = who_score
    scored["when"] = when_score

    gb_patient_auc, n_patient_auc, gb_patient_details = patient_auc_summary(scored, "gb")
    when_patient_auc, n_when_patient_auc, when_patient_details = patient_auc_summary(scored, "when")
    if n_patient_auc != n_when_patient_auc:
        raise AssertionError("GB and when-part patient AUC eligibility differs")
    if gb_patient_details:
        difference = max(abs(a["auc"] - b["auc"])
                         for a, b in zip(gb_patient_details, when_patient_details))
        if difference > TOL:
            raise AssertionError(
                f"When-part patient AUC must preserve GB ranks on the probability scale; max difference {difference}"
            )

    result = {
        "a0_pooled_auc": finite_auc(scored["y"].to_numpy(dtype=int), scored["a0"].to_numpy(dtype=float)),
        "gb_pooled_auc": finite_auc(scored["y"].to_numpy(dtype=int), scored["gb"].to_numpy(dtype=float)),
        "who_pooled_auc": finite_auc(scored["y"].to_numpy(dtype=int), scored["who"].to_numpy(dtype=float)),
        "when_pooled_auc": finite_auc(scored["y"].to_numpy(dtype=int), scored["when"].to_numpy(dtype=float)),
        "when_per_patient_auc": when_patient_auc,
        "gb_per_patient_auc": gb_patient_auc,
        "n_patients_with_both_outcomes": n_patient_auc,
        "share_reproduced": None,
        "a0_time": {},
    }
    if result["gb_pooled_auc"] is not None and result["who_pooled_auc"] is not None:
        denominator = result["gb_pooled_auc"] - 0.5
        if abs(denominator) > 1e-15:
            result["share_reproduced"] = (result["who_pooled_auc"] - 0.5) / denominator

    for key, column in clock_columns.items():
        result["a0_time"][key] = time_metrics(scored[["patient", "y", column, column + "_rate"]].rename(
            columns={column: "time", column + "_rate": "time_rate"}
        ), "time")
    return result


def metric_entry(estimate: float | None, draws: list[float | None]) -> dict:
    valid = np.asarray([v for v in draws if v is not None and np.isfinite(v)], dtype=float)
    ci = [float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))] if len(valid) else [None, None]
    return {"estimate": estimate, "ci95": ci, "valid_resamples": int(len(valid))}


def clock_draw_value(metrics: dict, key: str, clock: str) -> float | None:
    value = metrics["a0_time"][clock][key]
    return None if value is None else float(value)


def bootstrap_metrics(frame: pd.DataFrame, clock_columns: dict[str, str], threshold: float,
                      point: dict, rng: np.random.Generator) -> dict:
    """Resample whole patients, relabel duplicate draws, and recompute every metric."""
    patient_values = pd.unique(frame["patient"])
    row_groups = [np.flatnonzero(frame["patient"].to_numpy() == value) for value in patient_values]
    draw_names = [
        "a0_pooled_auc", "gb_pooled_auc", "who_pooled_auc", "when_pooled_auc",
        "when_per_patient_auc", "share_reproduced",
    ]
    for clock in clock_columns:
        draw_names.extend([f"a0_time:{clock}:pooled_auc", f"a0_time:{clock}:per_timepoint_auc"])
    draws = {name: [] for name in draw_names}

    for _ in range(N_BOOT):
        chosen = rng.integers(0, len(patient_values), size=len(patient_values))
        index = np.concatenate([row_groups[j] for j in chosen])
        sampled = frame.iloc[index].copy().reset_index(drop=True)
        sampled["patient"] = np.concatenate([
            np.full(len(row_groups[j]), copy_id, dtype=int) for copy_id, j in enumerate(chosen)
        ])
        values = compute_metrics(sampled, clock_columns, threshold)
        if values["when_per_patient_auc"] is not None and values["gb_per_patient_auc"] is not None:
            if not np.isclose(values["when_per_patient_auc"], values["gb_per_patient_auc"],
                              atol=TOL, rtol=0):
                raise AssertionError("Bootstrap when-part and GB per-patient AUCs differ")
        for name in draw_names[:6]:
            draws[name].append(values[name])
        for clock in clock_columns:
            draws[f"a0_time:{clock}:pooled_auc"].append(clock_draw_value(values, "pooled_auc", clock))
            draws[f"a0_time:{clock}:per_timepoint_auc"].append(
                clock_draw_value(values, "per_timepoint_auc", clock)
            )

    metrics = {
        "A0": metric_entry(point["a0_pooled_auc"], draws["a0_pooled_auc"]),
        "GB_pooled": metric_entry(point["gb_pooled_auc"], draws["gb_pooled_auc"]),
        "who_part_pooled": metric_entry(point["who_pooled_auc"], draws["who_pooled_auc"]),
        "when_part_pooled": metric_entry(point["when_pooled_auc"], draws["when_pooled_auc"]),
        "when_part_per_patient": metric_entry(point["when_per_patient_auc"], draws["when_per_patient_auc"]),
        "share_reproduced": metric_entry(point["share_reproduced"], draws["share_reproduced"]),
        "A0_time": {},
    }
    for clock in clock_columns:
        metrics["A0_time"][clock] = {
            "pooled": metric_entry(
                point["a0_time"][clock]["pooled_auc"], draws[f"a0_time:{clock}:pooled_auc"]
            ),
            "per_timepoint": metric_entry(
                point["a0_time"][clock]["per_timepoint_auc"], draws[f"a0_time:{clock}:per_timepoint_auc"]
            ),
            "n_eligible_timepoints": point["a0_time"][clock]["n_eligible_timepoints"],
            "n_eligible_observations": point["a0_time"][clock]["n_eligible_observations"],
            "n_all_observations": point["a0_time"][clock]["n_all_observations"],
            "eligible_timepoints": point["a0_time"][clock]["eligible_timepoints"],
        }
    metrics["n_patients_with_both_outcomes"] = point["n_patients_with_both_outcomes"]
    metrics["gb_per_patient_auc"] = point["gb_per_patient_auc"]
    return metrics


def format_entry(entry: dict) -> str:
    estimate = entry["estimate"]
    low, high = entry["ci95"]
    if estimate is None or low is None or high is None:
        return "NA"
    return f"{estimate:.3f} ({low:.3f} to {high:.3f})"


def build_markdown(results: dict) -> str:
    lines = [
        "# Information audit of pooled versus within-patient forecast AUC",
        "",
        "Entries are point estimates with 95% percentile intervals from 2,000 whole-patient resamples (seed 20261009).",
        "A0 is recomputed as each test patient's observed event rate across that patient's scored period. A0,time is a time-only forecast: the TRAINING patients' event rate at each time point (overall training rate where fewer than five training patients), the same for every test patient at that time; its pooled AUC covers all scored test observations; its per-time-point AUC averages only points with both outcomes and at least five patients.",
        "The who part is each patient's mean GB probability. The when part is the GB probability minus that patient's mean plus the row-weighted cohort mean, all on the probability scale. Per-patient AUCs use patients with both outcomes.",
        "For College Experience, A0,time is shown for both follow-up week and calendar date.",
        "",
        "| Cohort | A0 | A0,time | GB pooled | who part pooled | when part pooled | when part per-patient | share reproduced |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, result in results.items():
        time_cells = []
        for clock, clock_result in result["A0_time"].items():
            clock_label = result["clock_labels"][clock]
            time_cells.append(
                f"{clock_label}: pooled {format_entry(clock_result['pooled'])}; "
                f"per-time-point {format_entry(clock_result['per_timepoint'])}"
            )
        share = result["share_reproduced"]
        share_text = format_entry(share)
        lines.append(
            f"| {label} | {format_entry(result['A0'])} | {'<br>'.join(time_cells)} | "
            f"{format_entry(result['GB_pooled'])} | {format_entry(result['who_part_pooled'])} | "
            f"{format_entry(result['when_part_pooled'])} | {format_entry(result['when_part_per_patient'])} | "
            f"{share_text} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    cohorts = realtime_analysis.prepare()
    global TIME_RATES
    TIME_RATES = training_time_rates()
    rng = np.random.default_rng(SEED)
    output = {
        "method": {
            "bootstrap_resamples": N_BOOT,
            "bootstrap_seed": SEED,
            "patient_resampling": "whole test patients with replacement; duplicate draws relabelled as distinct bootstrap patients",
            "timepoint_eligibility": "at least five distinct test-patient copies and both outcomes",
            "A0": "recomputed patient event rate over the scored test period",
            "A0_time_pooled_population": "all scored test observations; forecast = training-cohort event rate per time point (fixed across resamples)",
            "per_patient_auc_eligibility": "patients with both outcomes",
            "centering_scale": "probability; cohort mean is the row-weighted mean of test GB forecasts",
            "share_reproduced": "(AUC_who - 0.5) / (AUC_GB - 0.5)",
        },
        "cohorts": {},
    }
    markdown_results = {}

    for cohort_key, (panel, _, _, _) in cohorts.items():
        setup = COHORT_SETUP[cohort_key]
        cohort_meta = same_patients.COHORTS[setup["same_patients_key"]]
        reference_column = cohort_meta[2]
        forecasts = realtime_analysis.load_forecast(cohort_key)
        alignment = realtime_analysis.align(cohort_key, panel, forecasts)
        if not (alignment["patient_rows_equal"] and alignment["y_sequence_equal"]):
            raise RuntimeError(f"Forecast/panel alignment failed for {cohort_key}: {alignment}")
        if reference_column not in forecasts.columns:
            raise ValueError(f"Missing existing A0 reference column {reference_column!r} for {cohort_key}")
        if "gradient_boosting" not in forecasts.columns or "threshold" not in forecasts.columns:
            raise ValueError(f"Missing GB forecast or threshold in {cohort_key} forecast export")

        frame = forecasts[["patient", "y", "gradient_boosting", "threshold"]].copy()
        frame = frame.rename(columns={"gradient_boosting": "gb"})
        frame["patient"] = frame["patient"].astype(int)
        frame["y"] = frame["y"].astype(int)
        if not set(frame["y"].unique()).issubset({0, 1}):
            raise ValueError(f"Outcomes are not binary for {cohort_key}")
        if not np.isfinite(frame["gb"].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite GB forecasts for {cohort_key}")
        threshold_values = frame["threshold"].to_numpy(dtype=float)
        if not np.allclose(threshold_values, threshold_values[0]):
            raise ValueError(f"Threshold varies within {cohort_key}")
        threshold = float(threshold_values[0])

        clock_columns = {}
        clock_labels = {}
        if cohort_key == "REPLACE-BG":
            clock_columns["night_since_enrollment"] = "night_since_enrollment"
            clock_labels["night_since_enrollment"] = setup["clocks"]["night_since_enrollment"]
            frame["night_since_enrollment"] = pd.to_numeric(panel["clock"], errors="raise").to_numpy()
        elif cohort_key == "CTN":
            clock_columns["study_week"] = "study_week"
            clock_labels["study_week"] = setup["clocks"]["study_week"]
            frame["study_week"] = pd.to_numeric(panel["clock"], errors="raise").to_numpy()
        else:
            clock_columns["follow_up_week"] = "follow_up_week"
            clock_columns["calendar_date"] = "calendar_date"
            clock_labels.update(setup["clocks"])
            frame["follow_up_week"] = pd.to_numeric(panel["week"], errors="raise").to_numpy()
            frame["calendar_date"] = pd.to_datetime(panel["clock2"]).to_numpy()

        for clock, column in clock_columns.items():
            mapping, overall = TIME_RATES[cohort_key][clock]
            keys = frame[column]
            if clock == "calendar_date":
                keys = pd.to_datetime(keys)
            frame[column + "_rate"] = keys.map(mapping).fillna(overall).astype(float).to_numpy()
        point = compute_metrics(frame, clock_columns, threshold)
        summary = bootstrap_metrics(frame, clock_columns, threshold, point, rng)
        summary.update({
            "label": setup["label"],
            "n_test_patients": int(frame["patient"].nunique()),
            "n_scored_observations": int(len(frame)),
            "alignment": alignment,
            "existing_A0_reference_column": reference_column,
            "clock_labels": clock_labels,
        })
        output["cohorts"][setup["label"]] = summary
        markdown_results[setup["label"]] = summary

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    table = build_markdown(markdown_results)
    OUT_MD.write_text(table, encoding="utf-8")
    print(table, end="")


if __name__ == "__main__":
    main()
