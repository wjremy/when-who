#!/usr/bin/env python3
"""Review analyses B13-B15 on the exported v7.1 test forecasts (no refit)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OPEN_DATA = Path(__file__).resolve().parent
ROOT = OPEN_DATA.parent
OUT_DIR = OPEN_DATA / "results" / "review_v71"
OUT_JSON = OUT_DIR / "review_v71.json"
OUT_MD = OUT_DIR / "review_v71.md"
SEED = 20261009
N_BOOT = 2000
N_PERM = 2000
N_CAL_SIM = 1000
CAL_TOLERANCE = 0.05

COHORTS = [
    {
        "label": "REPLACE-BG",
        "file": "forecasts_replace_bg.csv",
        "updated_column": "expanding_rate",
        "table_key": "replace_bg",
        "paper_calibration_share_percent": 93,
    },
    {
        "label": "CTN-0094",
        "file": "forecasts_ctn_weekly_missing_excluded.csv",
        "updated_column": "updating_rate",
        "table_key": "ctn_weekly_missing_excluded",
        "paper_calibration_share_percent": 46,
    },
    {
        "label": "College Experience",
        "file": "forecasts_ces.csv",
        "updated_column": "expanding_rate",
        "table_key": "ces",
        "paper_calibration_share_percent": 84,
    },
]

# Use the repository's common AUC implementation and its existing S4
# moment-corrected normal-approximation benchmark. The latter subtracts the
# day-weighted q(1-q)/(n-1) binomial sampling component from the observed,
# day-weighted variance of patient rates, truncates at zero, and maps the
# corrected SD to A0 with the normal approximation.
sys.path.insert(0, str(ROOT / "code" / "withinpatient" / "src"))
sys.path.insert(0, str(OPEN_DATA))
from withinpatient import auc, within_patient_auc  # noqa: E402
from ctn0094_analysis import benchmark as _existing_population_benchmark  # noqa: E402


def patient_groups(patient: np.ndarray) -> list[np.ndarray]:
    """Return row indexes in first-seen patient order."""
    return [np.flatnonzero(patient == key) for key in pd.unique(patient)]


def retrospective_a0(y: np.ndarray, patient: np.ndarray) -> float:
    """Pooled AUC of each patient's observed rate over their scored days."""
    score = np.empty(len(y), dtype=float)
    for ix in patient_groups(patient):
        score[ix] = float(np.mean(y[ix]))
    return float(auc(y, score))


def population_a0_moment_corrected(y: np.ndarray, patient: np.ndarray) -> float:
    """S4 moment-corrected population A0 from the existing CTN helper."""
    result = _existing_population_benchmark(y, patient, {})
    return float(result["normal_approximation"]["auc_sampling_corrected_sd"])


def observed_calibration_share(
    y: np.ndarray, forecast: np.ndarray, patient: np.ndarray
) -> float:
    """Share of patients within 0.05 of their mean forecast, as in Table 4."""
    close = []
    for ix in patient_groups(patient):
        close.append(abs(float(np.mean(y[ix])) - float(np.mean(forecast[ix]))) <= CAL_TOLERANCE)
    return float(np.mean(close))


def percentile_interval(values: list[float]) -> list[float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise RuntimeError("No finite bootstrap/permutation values were available")
    return [float(x) for x in np.percentile(finite, [2.5, 97.5])]


def metric_with_interval(estimate: float, values: list[float]) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return {
        "estimate": float(estimate),
        "ci95": percentile_interval(finite.tolist()),
        "valid_resamples": int(len(finite)),
    }


def b13_bootstrap(
    y: np.ndarray,
    patient: np.ndarray,
    forecasts: dict[str, np.ndarray],
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Paired whole-patient bootstrap for AUCs, A0, and GB-minus-A0."""
    groups = patient_groups(patient)
    point_a0 = retrospective_a0(y, patient)
    point = {
        "gb_pooled_auc": float(auc(y, forecasts["gradient_boosting"])),
        "retrospective_a0": point_a0,
        "gb_minus_retrospective_a0": float(
            auc(y, forecasts["gradient_boosting"]) - point_a0
        ),
        "population_a0_moment_corrected": population_a0_moment_corrected(y, patient),
        "runin_rate_auc": float(auc(y, forecasts["runin_rate"])),
        "updated_rate_auc": float(auc(y, forecasts["updated_rate"])),
    }
    draws = {name: [] for name in point}

    for _ in range(n_boot):
        chosen = rng.integers(0, len(groups), size=len(groups))
        row_ix = np.concatenate([groups[j] for j in chosen])
        # Distinct labels preserve duplicate sampled patients as separate clusters.
        bootstrap_patient = np.concatenate(
            [np.full(len(groups[j]), draw_id, dtype=int) for draw_id, j in enumerate(chosen)]
        )
        yb = y[row_ix]
        gb_auc = float(auc(yb, forecasts["gradient_boosting"][row_ix]))
        a0 = retrospective_a0(yb, bootstrap_patient)
        values = {
            "gb_pooled_auc": gb_auc,
            "retrospective_a0": a0,
            "gb_minus_retrospective_a0": gb_auc - a0,
            "population_a0_moment_corrected": population_a0_moment_corrected(
                yb, bootstrap_patient
            ),
            "runin_rate_auc": float(auc(yb, forecasts["runin_rate"][row_ix])),
            "updated_rate_auc": float(auc(yb, forecasts["updated_rate"][row_ix])),
        }
        for name, value in values.items():
            if np.isfinite(value):
                draws[name].append(float(value))

    return {
        name: metric_with_interval(point[name], draws[name])
        for name in point
    }


def b14_permutation(
    y: np.ndarray,
    patient: np.ndarray,
    forecasts: dict[str, np.ndarray],
    n_perm: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Within-patient forecast permutation test, keeping outcomes fixed."""
    model_names = [name for name in ("gradient_boosting", "logistic_regression") if name in forecasts]
    observed = {name: float(auc(y, forecasts[name])) for name in model_names}
    null = {name: [] for name in model_names}
    groups = patient_groups(patient)

    for _ in range(n_perm):
        permuted = {name: np.empty(len(y), dtype=float) for name in model_names}
        for ix in groups:
            order = rng.permutation(len(ix))
            for name in model_names:
                # Share the same within-patient day permutation for both models.
                permuted[name][ix] = forecasts[name][ix[order]]
        for name in model_names:
            null[name].append(float(auc(y, permuted[name])))

    result: dict[str, Any] = {}
    for name in model_names:
        vals = np.asarray(null[name], dtype=float)
        result[name] = {
            "observed_pooled_auc": observed[name],
            "null_mean": float(np.mean(vals)),
            "null_percentiles_2_5_97_5": [
                float(x) for x in np.percentile(vals, [2.5, 97.5])
            ],
            "one_sided_p_value": float((np.sum(vals >= observed[name]) + 1) / (n_perm + 1)),
            "permutations": int(n_perm),
        }
    return result


def b15_noise_reference(
    y: np.ndarray,
    patient: np.ndarray,
    forecast: np.ndarray,
    n_sim: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Simulate the Table 4 calibration share under y* ~ Bernoulli(forecast)."""
    groups = patient_groups(patient)
    patient_mean_forecasts = np.asarray([np.mean(forecast[ix]) for ix in groups], dtype=float)
    observed = observed_calibration_share(y, forecast, patient)

    # A boolean simulation matrix is modest for these test sets and keeps the
    # outcome generation explicit: each scored day is drawn independently.
    simulated_y = rng.random((n_sim, len(y))) < forecast[np.newaxis, :]
    shares = np.zeros(n_sim, dtype=float)
    for group_id, ix in enumerate(groups):
        simulated_rates = simulated_y[:, ix].mean(axis=1)
        shares += np.abs(simulated_rates - patient_mean_forecasts[group_id]) <= CAL_TOLERANCE
    shares /= len(groups)
    return {
        "observed_share": observed,
        "perfect_calibration_expected_share_mean": float(np.mean(shares)),
        "perfect_calibration_expected_share_percentiles_2_5_97_5": [
            float(x) for x in np.percentile(shares, [2.5, 97.5])
        ],
        "simulations": int(n_sim),
        "patients": int(len(groups)),
    }


def read_reference_results() -> tuple[dict[str, Any], dict[str, Any]]:
    table_path = OPEN_DATA / "cases_tables.json"
    audit_path = OPEN_DATA / "results" / "information_audit" / "information_audit.json"
    with table_path.open(encoding="utf-8") as stream:
        table = json.load(stream)
    with audit_path.open(encoding="utf-8") as stream:
        audit = json.load(stream)
    return table, audit


def check_reproduction(
    cohort: dict[str, Any],
    patient: np.ndarray,
    y: np.ndarray,
    forecasts: dict[str, np.ndarray],
    table: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Assert reproduction of Table 4 pooled/per-patient AUC and existing A0."""
    gb_auc = float(auc(y, forecasts["gradient_boosting"]))
    patient_auc = float(
        within_patient_auc(patient, y, forecasts["gradient_boosting"])["value"]
    )
    a0 = retrospective_a0(y, patient)
    published = table[cohort["table_key"]]["forecasts"]["gradient_boosting"]
    expected_a0 = float(audit["cohorts"][cohort["label"]]["A0"]["estimate"])
    comparisons = {
        "pooled_auc": (gb_auc, float(published["pooled_auc"])),
        "per_patient_auc": (patient_auc, float(published["within_patient_auc"])),
        "retrospective_a0": (a0, expected_a0),
    }
    for metric, (computed, expected) in comparisons.items():
        if not np.isclose(computed, expected, rtol=0.0, atol=1e-12):
            raise AssertionError(
                f"{cohort['label']} {metric} does not reproduce: "
                f"computed {computed:.15g}, expected {expected:.15g}"
            )
    share = observed_calibration_share(y, forecasts["gradient_boosting"], patient)
    rounded_percent = int(math.floor(100 * share + 0.5))
    expected_percent = int(cohort["paper_calibration_share_percent"])
    if rounded_percent != expected_percent:
        raise AssertionError(
            f"{cohort['label']} calibration share rounds to {rounded_percent}%, "
            f"not the Table 4 value {expected_percent}%"
        )
    return {
        metric: {"computed": float(values[0]), "existing": float(values[1]), "passed": True}
        for metric, values in comparisons.items()
    } | {
        "table4_calibration_share": {
            "computed": share,
            "rounded_percent": rounded_percent,
            "published_percent": expected_percent,
            "passed": True,
        }
    }


def load_cohort(cohort: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    frame = pd.read_csv(OPEN_DATA / cohort["file"])
    required = {
        "patient", "y", "runin_rate", cohort["updated_column"],
        "gradient_boosting",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{cohort['file']} is missing columns: {sorted(missing)}")
    if "logistic_regression" in frame.columns:
        required.add("logistic_regression")
    if frame.empty:
        raise ValueError(f"{cohort['file']} contains no test forecasts")

    patient = frame["patient"].to_numpy()
    y_float = frame["y"].to_numpy(dtype=float)
    if not np.isfinite(y_float).all() or not np.isin(y_float, [0.0, 1.0]).all():
        raise ValueError(f"{cohort['file']} has missing or non-binary outcomes")
    y = y_float.astype(int)
    forecasts = {
        "gradient_boosting": frame["gradient_boosting"].to_numpy(dtype=float),
        "runin_rate": frame["runin_rate"].to_numpy(dtype=float),
        "updated_rate": frame[cohort["updated_column"]].to_numpy(dtype=float),
    }
    if "logistic_regression" in frame.columns:
        forecasts["logistic_regression"] = frame["logistic_regression"].to_numpy(dtype=float)
    for name, values in forecasts.items():
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"{cohort['file']} has invalid probabilities in {name}")
    if pd.isna(patient).any():
        raise ValueError(f"{cohort['file']} has missing patient identifiers")
    return patient, y, forecasts


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    return value


def format_interval(entry: dict[str, Any]) -> str:
    lo, hi = entry["ci95"]
    return f"{entry['estimate']:.3f} ({lo:.3f}, {hi:.3f})"


def build_markdown(results: dict[str, Any]) -> str:
    labels = [cohort["label"] for cohort in COHORTS]
    lines = [
        "# v7.1 review analyses B13–B15",
        "",
        "All analyses use the exported held-out test forecasts; no model is refit. The fixed seed is 20261009.",
        "",
        "## B13. Pooled AUC and patient-rate references",
        "",
        "Entries are estimates with percentile 95% intervals from paired whole-patient bootstrap resamples (2,000 by default). A0 is the pooled AUC of each patient's observed scored-period rate. Population A0 applies the S4 binomial-noise moment correction and normal approximation.",
        "",
        "| Measure | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(labels),
    ]
    b13_rows = [
        ("GB pooled AUC", "gb_pooled_auc"),
        ("Retrospective A0", "retrospective_a0"),
        ("GB pooled AUC − retrospective A0", "gb_minus_retrospective_a0"),
        ("Population A0, moment corrected", "population_a0_moment_corrected"),
        ("Run-in rate pooled AUC", "runin_rate_auc"),
        ("Updated rate pooled AUC", "updated_rate_auc"),
    ]
    for title, key in b13_rows:
        lines.append("| " + title + " | " + " | ".join(
            format_interval(results["B13"][label][key]) for label in labels
        ) + " |")
    lines.extend([
        "",
        "This shows how pooled GB discrimination compares with the hindsight rate benchmark, the noise-corrected population benchmark, and patient-specific rates available when forecasts are issued.",
        "",
        "## B14. Within-patient forecast permutation test",
        "",
        "Outcomes stay fixed while each patient's forecasts are permuted across that patient's scored days. Cells show observed pooled AUC; null mean (2.5th, 97.5th percentiles); and one-sided plus-one p-value.",
        "",
        "| Model | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(labels),
    ])
    model_names = ["gradient_boosting"]
    if any("logistic_regression" in results["B14"][label] for label in labels):
        model_names.append("logistic_regression")
    model_labels = {"gradient_boosting": "GB", "logistic_regression": "LR"}
    for model in model_names:
        cells = []
        for label in labels:
            value = results["B14"][label][model]
            lo, hi = value["null_percentiles_2_5_97_5"]
            cells.append(
                f"{value['observed_pooled_auc']:.3f}; {value['null_mean']:.3f} "
                f"({lo:.3f}, {hi:.3f}); " + ("p<0.001" if value['one_sided_p_value'] < 0.001 else f"p={value['one_sided_p_value']:.3f}")
            )
        lines.append("| " + model_labels[model] + " | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "This tests whether the fitted forecasts' pooled AUC exceeds the within-patient no-order reference while preserving each patient's outcomes and forecast distribution.",
        "",
        "## B15. Calibration-in-the-large noise reference",
        "",
        "Shares are proportions of patients. Simulated intervals are the 2.5th and 97.5th percentiles across 1,000 outcome draws from the exported forecasts.",
        "",
        "| Share within 0.05 of mean GB forecast | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(labels),
    ])
    observed_cells = [f"{results['B15'][label]['observed_share']:.3f}" for label in labels]
    lines.append("| Observed | " + " | ".join(observed_cells) + " |")
    expected_cells = []
    for label in labels:
        value = results["B15"][label]
        lo, hi = value["perfect_calibration_expected_share_percentiles_2_5_97_5"]
        expected_cells.append(
            f"{value['perfect_calibration_expected_share_mean']:.3f} "
            f"({lo:.3f}, {hi:.3f})"
        )
    lines.append("| Expected if perfectly calibrated | " + " | ".join(expected_cells) + " |")
    lines.extend([
        "",
        "This shows how much patient-level calibration-in-the-large agreement the observed record lengths can produce through binomial noise when forecasts are perfectly calibrated.",
        "",
        "## Reproduction checks",
        "",
        "The script asserts the exported-data pooled AUC, per-patient AUC, and retrospective A0 against the existing results, and checks that the calibration shares round to the Table 4 percentages (93%, 46%, 84%).",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true",
        help="use 50 patient bootstrap resamples, permutations, and calibration simulations",
    )
    args = parser.parse_args()
    n_boot = 50 if args.quick else N_BOOT
    n_perm = 50 if args.quick else N_PERM
    n_cal_sim = 50 if args.quick else N_CAL_SIM
    table, audit = read_reference_results()
    rng = np.random.default_rng(SEED)
    results: dict[str, Any] = {
        "method": {
            "source": "exported held-out test forecast CSVs; no refit",
            "seed": SEED,
            "quick_mode": bool(args.quick),
            "bootstrap_resamples": n_boot,
            "permutations": n_perm,
            "calibration_simulations": n_cal_sim,
            "bootstrap_unit": "whole test patients sampled with replacement; repeated copies relabelled",
            "permutation": "outcomes fixed; each patient's forecasts permuted over that patient's scored days; same permutation applied to GB and LR",
            "permutation_p_value": "(count(null pooled AUC >= observed pooled AUC) + 1) / (B + 1)",
            "population_a0": "existing ctn0094_analysis.benchmark S4 correction: day-weighted variance of observed patient rates minus the day-weighted q(1-q)/(n-1) sampling component, truncated at zero; normal approximation using corrected SD",
            "runin_rate_columns": "runin_rate in each forecast export",
            "updated_rate_columns": {
                "REPLACE-BG": "expanding_rate",
                "CTN-0094": "updating_rate",
                "College Experience": "expanding_rate",
            },
            "calibration_share": "patient's observed scored-period rate within 0.05 inclusive of the patient's mean GB forecast",
            "calibration_simulation": "independent Bernoulli draw per scored day with probability equal to exported GB forecast",
        },
        "B13": {},
        "B14": {},
        "B15": {},
        "reproduction_checks": {},
    }

    for cohort in COHORTS:
        patient, y, forecasts = load_cohort(cohort)
        results["reproduction_checks"][cohort["label"]] = check_reproduction(
            cohort, patient, y, forecasts, table, audit
        )
        results["B13"][cohort["label"]] = b13_bootstrap(
            y,
            patient,
            forecasts,
            n_boot,
            rng,
        )
        results["B14"][cohort["label"]] = b14_permutation(
            y,
            patient,
            forecasts,
            n_perm,
            rng,
        )
        results["B15"][cohort["label"]] = b15_noise_reference(
            y,
            patient,
            forecasts["gradient_boosting"],
            n_cal_sim,
            rng,
        )
        results["B15"][cohort["label"]]["patients"] = int(len(patient_groups(patient)))
        results["B15"][cohort["label"]]["scored_days"] = int(len(y))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(clean(results), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown = build_markdown(results)
    OUT_MD.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
