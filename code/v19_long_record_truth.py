"""Compute long-record per-patient truths for every ranked metric.

Run from any working directory with the project Python environment:
    python code/v19_long_record_truth.py

The calibration length is included in the CSV because the simulation fits a
different version of model A for 30-, 100-, and 365-day records.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import ranking_sim as sim  # noqa: E402


SOURCE_JSON = ROOT / "open_data" / "results" / "v10" / "ranking_sim.json"
OUTPUT_CSV = ROOT / "results" / "v19" / "A2_long_record_truth.csv"

# These are the per-patient definitions selected by ranking_sim._patient_statistics.
# All three Brier references are included because the ranking compares updated-rate,
# oracle, and population-rate Brier skill in different ranking rows.
MEASURES = (
    ("auc", "auc"),
    ("average_precision", "average_precision"),
    ("brier_skill_updated_rate", "brier_skill_updated_rate"),
    ("brier_skill_oracle", "brier_skill_oracle"),
    ("brier_skill_population_rate", "brier_skill_population_rate"),
    ("ppv", "ppv"),
    ("net_benefit", "net_benefit"),
)


def _truth_groups(payload: dict[str, Any], *, long_days: int | None = None,
                  n_patients: int | None = None) -> list[dict[str, Any]]:
    """Return per-pair/spread/A-calibration/model means for the requested run."""
    run = payload["run"]
    benchmark = payload["long_record_benchmark"]
    fits = run["cohort_fits"]
    calibrations = run["calibrations"]
    specs = sim.cohort_sim.COHORTS

    days_by_cohort: dict[str, set[int]] = {cohort: set() for cohort in sim.COHORTS}
    pairs_by_cohort: dict[str, set[str]] = {cohort: set() for cohort in sim.COHORTS}
    for key, value in benchmark.items():
        cohort, days_text, pair_name = key.split("|", 2)
        days_by_cohort[cohort].add(int(value["record_days_calibration"]))
        pairs_by_cohort[cohort].add(pair_name)

    pairs = [dict(sim.BASE_PAIR)]
    if run.get("extra_pair_included"):
        pairs.append(dict(sim.EXTRA_PAIR))
    for cohort in sim.COHORTS:
        expected_pairs = {pair["name"] for pair in pairs}
        if pairs_by_cohort[cohort] != expected_pairs:
            raise ValueError(f"Benchmark pairs for {cohort} do not match run settings")

    observed_long_days = {
        int(value["long_record_days"])
        for value in benchmark.values()
    }
    observed_patients = {
        int(value["long_record_patients"])
        for value in benchmark.values()
    }
    if len(observed_long_days) != 1 or len(observed_patients) != 1:
        raise ValueError("Long-record settings vary within the stored benchmark")
    days = int(long_days if long_days is not None else next(iter(observed_long_days)))
    patients = int(n_patients if n_patients is not None else next(iter(observed_patients)))

    groups: list[dict[str, Any]] = []
    for ci, cohort in enumerate(sim.COHORTS):
        if not days_by_cohort[cohort]:
            continue
        spec, fit = specs[cohort], fits[cohort]
        rng = sim._rng(90, ci)
        b = sim._recentered_levels(rng, fit, spec, patients)
        x, y = sim._draw_outcomes(rng, b, days)
        noise = rng.standard_normal(x.shape)
        population_rate = float(spec["rate"])
        patient_rate = sim.who_when.normal_expit_expectation(b, sim.X_SD * sim.X_SD)
        threshold = population_rate

        for pair in pairs:
            if pair["name"] not in pairs_by_cohort[cohort]:
                continue
            daily_a = sim._daily_proxy(x, noise, pair["rho_a"])
            daily_b = sim._daily_proxy(x, noise, pair["rho_b"])
            forecast_b = sim._b_forecast(b, daily_b, pair["rho_b"])
            stats_b = sim._patient_statistics(y, forecast_b, threshold,
                                              population_rate, patient_rate)
            b_means = {
                measure: sim._finite_mean(sim._patient_means(stats_b[stat_key]))
                for measure, stat_key in MEASURES
            }

            for calibration_days in sorted(days_by_cohort[cohort]):
                cal_key = f"{cohort}|{calibration_days}|{pair['rho_a']:.1f}"
                calibration = calibrations[cal_key]
                forecast_a = sim._a_forecast(y, daily_a, population_rate, calibration)
                stats_a = sim._patient_statistics(y, forecast_a, threshold,
                                                  population_rate, patient_rate)
                a_means = {
                    measure: sim._finite_mean(sim._patient_means(stats_a[stat_key]))
                    for measure, stat_key in MEASURES
                }

                group = {
                    "pair": pair["name"],
                    "spread": "low" if cohort == "REPLACE-BG" else "high",
                    "cohort": cohort,
                    "record_days_calibration": calibration_days,
                    "a": a_means,
                    "b": b_means,
                }
                groups.append(group)
    return groups


def _stored_ranking_auc(payload: dict[str, Any]) -> dict[tuple[str, str, int], tuple[float, float]]:
    """Read the benchmark AUCs repeated on ranking rows and check consistency."""
    stored: dict[tuple[str, str, int], tuple[float, float]] = {}
    for row in payload["ranking"]:
        key = (row["pair"], row["patient_spread"], int(row["record_days"]))
        values = (float(row["long_record_a_true_patient_auc"]),
                  float(row["long_record_b_true_patient_auc"]))
        if key in stored:
            assert all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)
                       for a, b in zip(stored[key], values)), (
                f"Inconsistent stored ranking AUC values for {key}")
        else:
            stored[key] = values
    return stored


def _rows(payload: dict[str, Any], groups: list[dict[str, Any]], *,
          verify_stored_auc: bool) -> list[dict[str, Any]]:
    stored_auc = _stored_ranking_auc(payload) if verify_stored_auc else {}
    result: list[dict[str, Any]] = []
    for group in groups:
        pair, spread = group["pair"], group["spread"]
        cohort, calibration_days = group["cohort"], group["record_days_calibration"]
        benchmark_key = f"{cohort}|{calibration_days}|{pair}"
        benchmark = payload["long_record_benchmark"][benchmark_key]

        for measure, _ in MEASURES:
            a_value, b_value = group["a"][measure], group["b"][measure]
            a_better = (bool(a_value > b_value)
                        if math.isfinite(a_value) and math.isfinite(b_value) else None)
            if verify_stored_auc and measure == "auc":
                expected_a, expected_b = stored_auc[(pair, spread, calibration_days)]
                assert math.isclose(a_value, expected_a, rel_tol=0.0, abs_tol=1e-12), (
                    f"Computed A AUC {a_value} != stored long_record_a_true_patient_auc "
                    f"{expected_a} for {(pair, spread, calibration_days)}")
                assert math.isclose(b_value, expected_b, rel_tol=0.0, abs_tol=1e-12), (
                    f"Computed B AUC {b_value} != stored long_record_b_true_patient_auc "
                    f"{expected_b} for {(pair, spread, calibration_days)}")
                assert math.isclose(a_value, float(benchmark["a_mean_patient_auc"]),
                                    rel_tol=0.0, abs_tol=1e-12)
                assert math.isclose(b_value, float(benchmark["b_mean_patient_auc"]),
                                    rel_tol=0.0, abs_tol=1e-12)

            for model, value in (("A", a_value), ("B", b_value)):
                result.append({
                    "pair": pair,
                    "spread": spread,
                    "record_days_calibration": calibration_days,
                    "model": model,
                    "measure": measure,
                    "long_record_value": value,
                    "a_better": a_better,
                })
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("pair", "spread", "record_days_calibration", "model", "measure",
               "long_record_value", "a_better")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", type=Path, default=SOURCE_JSON)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--check-tiny", action="store_true",
                        help="run a 5-patient, 30-day in-memory smoke check without writing CSV")
    args = parser.parse_args()

    payload = json.loads(args.source_json.read_text(encoding="utf-8"))
    if args.check_tiny:
        groups = _truth_groups(payload, long_days=30, n_patients=5)
        rows = _rows(payload, groups, verify_stored_auc=False)
        assert groups and rows
        assert all(row["model"] in {"A", "B"} for row in rows)
        print(f"Tiny check passed: {len(groups)} groups, {len(rows)} rows; no file written.")
        return

    groups = _truth_groups(payload)
    rows = _rows(payload, groups, verify_stored_auc=True)
    _write_csv(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
