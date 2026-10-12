#!/usr/bin/env python3
"""Build CTN-0094 A1 (who as a column) and A6 (test-patient selection) results.

Run from the repository root with Wesley's Python environment:
    python open_data/v13_ctn_who.py

The fitted CTN model is refitted through the existing model-fixed helper so
that exported rows can be aligned to their calendar study weeks exactly.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OPEN_DATA = ROOT / "open_data"
INPUT = OPEN_DATA / "forecasts_ctn_weekly_missing_excluded.csv"
OUTPUT_DIR = ROOT / "results" / "v13"
A1_OUTPUT = OUTPUT_DIR / "A1_who_column.csv"
A6_OUTPUT = OUTPUT_DIR / "A6_selection.csv"


def _load_modules():
    if str(OPEN_DATA) not in sys.path:
        sys.path.insert(0, str(OPEN_DATA))
    import main_table_v10 as main_table

    table4, np, pd, _, _ = main_table._load_dependencies()
    from wheneval import auc

    return main_table, table4, np, pd, auc


def _selection_rows(frame, pd) -> list[dict[str, Any]]:
    """Summarise event and scored-week counts for eligible and ineligible people."""
    required = {"patient", "t", "y"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"forecast export is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("forecast export has no scored rows")

    patient_rows = []
    for patient, rows in frame.groupby("patient", sort=False, dropna=False):
        outcomes = rows.y.to_numpy(dtype=int)
        if not set(outcomes).issubset({0, 1}):
            raise ValueError(f"patient {patient!r} has a non-binary outcome")
        if rows.t.isna().any() or rows.t.nunique() != len(rows):
            raise ValueError(f"patient {patient!r} has missing or duplicate scored-week indices")
        n_events = int(outcomes.sum())
        n_rows = len(rows)
        patient_rows.append({
            "patient": patient,
            "n_events": n_events,
            "n_event_free": n_rows - n_events,
            "event_rate": float(n_events / n_rows),
            "scored_weeks": int(n_rows),
        })

    patients = pd.DataFrame(patient_rows)
    both = (patients.n_events.gt(0) & patients.n_event_free.gt(0))
    groups = (
        ("both_outcomes", patients.loc[both]),
        ("all_events", patients.loc[~both & patients.n_events.gt(0)]),
        ("no_events", patients.loc[~both & patients.n_events.eq(0)]),
    )
    records = []
    for group_name, group in groups:
        n_patients = int(len(group))
        for measure, unit in (("event_rate", "proportion"), ("scored_weeks", "weeks")):
            if n_patients:
                median, q1, q3 = group[measure].quantile([0.5, 0.25, 0.75]).tolist()
            else:
                median = q1 = q3 = None
            records.append({
                "patient_group": group_name,
                "n_patients": n_patients,
                "measure": measure,
                "median": median,
                "q1": q1,
                "q3": q3,
                "unit": unit,
            })
    return records


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_selection(frame, pd) -> None:
    rows = _selection_rows(frame, pd)
    _write_csv(
        A6_OUTPUT,
        rows,
        ["patient_group", "n_patients", "measure", "median", "q1", "q3", "unit"],
    )


def _pair_type_counts(patient, study_week, y) -> dict[str, int]:
    """Count all pooled event/event-free pairs in the three disjoint classes."""
    event_mask = y == 1
    free_mask = y == 0
    total_pairs = int(event_mask.sum()) * int(free_mask.sum())

    event_by_patient: dict[Any, int] = {}
    free_by_patient: dict[Any, int] = {}
    event_by_week: dict[Any, int] = {}
    free_by_week: dict[Any, int] = {}
    event_by_patient_week: dict[tuple[Any, Any], int] = {}
    free_by_patient_week: dict[tuple[Any, Any], int] = {}
    for pid, week, outcome in zip(patient, study_week, y):
        if outcome == 1:
            event_by_patient[pid] = event_by_patient.get(pid, 0) + 1
            event_by_week[week] = event_by_week.get(week, 0) + 1
            key = (pid, week)
            event_by_patient_week[key] = event_by_patient_week.get(key, 0) + 1
        else:
            free_by_patient[pid] = free_by_patient.get(pid, 0) + 1
            free_by_week[week] = free_by_week.get(week, 0) + 1
            key = (pid, week)
            free_by_patient_week[key] = free_by_patient_week.get(key, 0) + 1

    same_patient = sum(
        count * free_by_patient.get(pid, 0) for pid, count in event_by_patient.items()
    )
    same_week_all = sum(
        count * free_by_week.get(week, 0) for week, count in event_by_week.items()
    )
    same_week_same_patient = sum(
        count * free_by_patient_week.get(key, 0)
        for key, count in event_by_patient_week.items()
    )
    same_week_different_patients = same_week_all - same_week_same_patient
    other = total_pairs - same_patient - same_week_different_patients
    counts = {
        "same_patient": int(same_patient),
        "same_study_week_different_patients": int(same_week_different_patients),
        "other": int(other),
    }
    if min(counts.values()) < 0 or sum(counts.values()) != total_pairs:
        raise AssertionError("pair categories do not partition the pooled pairs")
    return {**counts, "total_pairs": total_pairs}


def _patient_auc_mean(patient, y, score, auc, np, pd) -> tuple[float | None, int]:
    values = []
    for key in pd.unique(patient):
        positions = np.flatnonzero(patient == key)
        value = float(auc(y[positions], score[positions]))
        if value == value:  # NaN means this patient has only one outcome class.
            values.append(value)
    return (float(sum(values) / len(values)) if values else None, len(values))


def _week_auc_mean(study_week, y, score, auc, np, pd) -> tuple[float | None, int]:
    values = []
    for key in pd.unique(study_week):
        positions = np.flatnonzero(study_week == key)
        if len(set(y[positions].tolist())) < 2:
            continue
        value = float(auc(y[positions], score[positions]))
        if value == value:
            values.append(value)
    return (float(sum(values) / len(values)) if values else None, len(values))


def _prepare_aligned_rows(frame, pm, main_table, table4, np, pd):
    config = next(
        c for c in table4.COHORTS if c["permutation_key"] == "ctn0094"
    )
    fitted_bundle = pm.fit_ctn()

    # Table 4's alignment retains the identity of each model-fixed test row,
    # including the calendar study week that is absent from the CSV export.
    bundle_order, file_order = table4.align_forecast_rows(frame, fitted_bundle, config)
    context = main_table._load_context(pm, table4, config, np, pd)
    context_order, context_file_order = main_table._align_context(context, frame, np, pd)
    if not np.array_equal(bundle_order, context_order):
        raise AssertionError("Table 4 and main-table test-row alignments disagree")
    if not np.array_equal(file_order, context_file_order):
        raise AssertionError("Table 4 and main-table export-row alignments disagree")

    aligned = frame.iloc[file_order].copy().reset_index(drop=True)
    aligned["who"] = np.asarray(fitted_bundle["pid"])[bundle_order]
    aligned["study_week"] = (
        fitted_bundle["test"].iloc[bundle_order]["study_week"].to_numpy()
    )
    if aligned.study_week.isna().any():
        raise AssertionError("aligned calendar study weeks contain missing values")
    if aligned.duplicated(["who", "study_week"]).any():
        raise AssertionError("CTN test rows contain duplicate patient-study-week pairs")

    if not np.allclose(
        context["updated_observed"][context_order],
        aligned[config["updated_rate_column"]].to_numpy(float),
        rtol=0,
        atol=1e-10,
    ):
        raise AssertionError("main-table observed rate does not match the aligned export")
    return aligned, context, context_order, config


def _a1_rows(aligned, context, context_order, config, pm, table4, np, pd, auc):
    patient = aligned.who.to_numpy()
    study_week = aligned.study_week.to_numpy()
    y = aligned.y.to_numpy(dtype=int)
    pair_counts = _pair_type_counts(patient, study_week, y)
    n_rows = len(y)
    n_events = int(y.sum())
    n_event_free = n_rows - n_events
    if not n_events or not n_event_free:
        raise ValueError("A1 requires at least one event and one event-free test row")

    # Match main_table_v10's single-draw shuffle method: seed its RNG with the
    # fitted-permutation seed, shuffle outcomes within each patient, then rebuild
    # the rate so far with table4_v10's existing history function.
    seed = int(pm.SEED)
    rng_perm = np.random.default_rng(seed)
    target = pm.generate_permuted_targets(
        context["bundle"]["y"], context["bundle"]["pid"], 1, rng_perm
    )[0]
    shuffled_rate_all = table4.rebuild_updated_rate(
        pm, config, context["bundle"], target, context["auxiliary"]
    )
    shuffled_rate = np.asarray(shuffled_rate_all, dtype=float)[context_order]
    # The shuffled rate so far is scored against the shuffled outcomes it was built from (as in main_table_v10).
    shuffled_y = np.asarray(target, dtype=int)[context_order]

    forecasts = (
        ("run-in rate", aligned.runin_rate.to_numpy(dtype=float), None, y),
        ("shuffled rate so far", shuffled_rate, seed, shuffled_y),
        ("fitted gradient-boosting model", aligned.gradient_boosting.to_numpy(dtype=float), None, y),
    )
    rows: list[dict[str, Any]] = []
    total_pairs = pair_counts["total_pairs"]
    for pair_type in (
        "same_patient",
        "same_study_week_different_patients",
        "other",
    ):
        count = pair_counts[pair_type]
        rows.append({
            "result_type": "pair_share",
            "statistic": pair_type,
            "forecast": "",
            "value": count / total_pairs,
            "numerator": count,
            "denominator": total_pairs,
            "eligible_patients": "",
            "eligible_weeks": "",
            "n_scored_rows": n_rows,
            "n_event_rows": n_events,
            "n_event_free_rows": n_event_free,
            "shuffle_seed": "",
        })

    for forecast_name, score, forecast_seed, y_scored in forecasts:
        pooled = float(auc(y_scored, score))
        patient_mean, n_patients = _patient_auc_mean(patient, y_scored, score, auc, np, pd)
        week_mean, n_weeks = _week_auc_mean(study_week, y_scored, score, auc, np, pd)
        for statistic, value, eligible_patients, eligible_weeks in (
            ("pooled_auc", pooled, "", ""),
            ("per_patient_auc", patient_mean, n_patients, ""),
            ("per_week_auc", week_mean, "", n_weeks),
        ):
            rows.append({
                "result_type": "auc",
                "statistic": statistic,
                "forecast": forecast_name,
                "value": value,
                "numerator": "",
                "denominator": "",
                "eligible_patients": eligible_patients,
                "eligible_weeks": eligible_weeks,
                "n_scored_rows": n_rows,
                "n_event_rows": n_events,
                "n_event_free_rows": n_event_free,
                "shuffle_seed": forecast_seed if forecast_seed is not None else "",
            })
    return rows


def run() -> None:
    # A6 needs only the already-exported test rows, so it remains available in
    # environments where the model-fitting dependency (scikit-learn) is absent.
    if not INPUT.exists():
        raise FileNotFoundError(f"forecast export not found: {INPUT}")

    main_table, table4, np, pd, auc = _load_modules()
    frame = pd.read_csv(INPUT)
    _write_selection(frame, pd)

    pm = table4.load_permutation_module()
    try:
        aligned, context, context_order, config = _prepare_aligned_rows(
            frame, pm, main_table, table4, np, pd
        )
    except ModuleNotFoundError as exc:
        if exc.name == "sklearn" or (exc.name and exc.name.startswith("sklearn.")):
            raise RuntimeError(
                "A6_selection.csv was written, but A1 needs scikit-learn to call "
                "cases_model_fixed_permutation.fit_ctn(); rerun with Wesley's Python."
            ) from exc
        raise

    a1 = _a1_rows(aligned, context, context_order, config, pm, table4, np, pd, auc)
    _write_csv(
        A1_OUTPUT,
        a1,
        [
            "result_type", "statistic", "forecast", "value", "numerator", "denominator",
            "eligible_patients", "eligible_weeks", "n_scored_rows", "n_event_rows",
            "n_event_free_rows", "shuffle_seed",
        ],
    )
    print(f"Wrote {A1_OUTPUT}")
    print(f"Wrote {A6_OUTPUT}")


if __name__ == "__main__":
    run()
