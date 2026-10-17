"""Estimate finite-horizon AUC skill for the closer Table 6 model pair.

Run from the project root with:
    python code/v19_finite_horizon.py

Each simulated patient contributes many independent records at one horizon.
For each patient, the AUC pools event-control day pairs across different
replicates only; each patient's cross-replicate AUCs are averaged equally.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import ranking_sim as sim  # noqa: E402


INPUT_JSON = ROOT / "open_data" / "results" / "v10" / "ranking_sim.json"
DEFAULT_OUTPUT = ROOT / "results" / "v19" / "A3_finite_horizon.csv"
PAIR_NAME = "A daily rho=0.4 vs B daily rho=0.3"
RHO_A = 0.4
RHO_B = 0.3
DEFAULT_PATIENTS = 2_000
DEFAULT_REPLICATES = 100
TARGET_SE = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", type=int, default=DEFAULT_PATIENTS,
                        help=f"independent patients per spread/length setting (default {DEFAULT_PATIENTS})")
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES,
                        help=f"independent records per patient (default {DEFAULT_REPLICATES})")
    parser.add_argument("--spread", choices=("low", "high"), action="append",
                        help="run only this spread; repeat to select both (default: both)")
    parser.add_argument("--days", type=int, choices=sim.LENGTHS, action="append",
                        help="run only this record length; repeat for multiple (default: all)")
    parser.add_argument("--input-json", type=Path, default=INPUT_JSON,
                        help="ranking_sim.json with Table 6 references and A calibrations")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="CSV destination (default: results/v19/A3_finite_horizon.csv)")
    args = parser.parse_args()
    if args.patients < 2 or args.replicates < 2:
        parser.error("--patients and --replicates must both be at least 2")
    return args


def _references(input_json: Path
                ) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, dict[str, float]]]]:
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    run_info = payload["run"]
    required = {}
    for row in payload["ranking"]:
        if (row.get("patients") == 50 and row.get("pair") == PAIR_NAME
                and row.get("scoring") == "tailored_excess" and row.get("metric") == "auc"):
            required[(row["cohort"], int(row["record_days"]))] = {
                "A": {
                    "long_record_skill": float(row["long_record_a_true_patient_auc"]) - 0.5,
                    "mean_shuffled_excess": float(row["mean_a_score"]),
                },
                "B": {
                    "long_record_skill": float(row["long_record_b_true_patient_auc"]) - 0.5,
                    "mean_shuffled_excess": float(row["mean_b_score"]),
                },
            }
    expected = {(cohort, days) for cohort in sim.COHORTS for days in sim.LENGTHS}
    if not expected.issubset(required):
        missing = sorted(expected.difference(required))
        raise ValueError(f"{input_json} lacks 50-patient tailored AUC references for {PAIR_NAME}: {missing}")
    if not run_info.get("extra_pair_included", False):
        raise ValueError(f"{input_json} does not contain the closer pair {PAIR_NAME}")
    return payload, required


def _cross_replicate_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Pair-weighted AUC after removing all within-replicate event-control pairs."""
    yy = np.asarray(y, dtype=np.int8)
    ss = np.asarray(score, dtype=float)
    if yy.ndim != 2 or ss.shape != yy.shape:
        raise ValueError("y and score must have the same replicates-by-days shape")

    # The pooled AUC contains comparisons within and between replicates. Remove
    # the within-record Mann-Whitney numerators and denominators to retain only
    # event days from one replicate compared with controls from another.
    events = yy.sum(axis=1, dtype=np.int64)
    controls = yy.shape[1] - events
    total_events = int(events.sum())
    total_controls = int(controls.sum())
    all_pairs = total_events * total_controls
    same_record_pairs = int(np.dot(events, controls))
    cross_pairs = all_pairs - same_record_pairs
    if cross_pairs <= 0:
        return float("nan")

    pooled_auc = float(sim._auc_matrix(yy.reshape(1, -1), ss.reshape(1, -1))[0, 0])
    replicate_auc = sim._auc_matrix(yy, ss)[0]
    valid = (events > 0) & (controls > 0) & np.isfinite(replicate_auc)
    within_numerator = float(np.sum(replicate_auc[valid] * events[valid] * controls[valid]))
    pooled_numerator = pooled_auc * all_pairs
    return float((pooled_numerator - within_numerator) / cross_pairs)


def _patient_auc_pair(fit: dict[str, Any], cohort: str, days: int, patients: int,
                      replicates: int, calibration: dict[str, Any], population_rate: float
                      ) -> tuple[np.ndarray, np.ndarray]:
    cohort_index = sim.COHORTS.index(cohort)
    level_rng = sim._rng(930, cohort_index, days, patients, replicates)
    patient_levels = sim._recentered_levels(
        level_rng, fit, sim.cohort_sim.COHORTS[cohort], patients,
    )
    auc_a = np.full(patients, np.nan, dtype=float)
    auc_b = np.full(patients, np.nan, dtype=float)

    for patient_index, level in enumerate(patient_levels):
        # One reproducible stream per patient; all records share the patient's
        # latent level but otherwise draw independent x, outcome, and proxy noise.
        rng = sim._rng(931, cohort_index, days, patient_index)
        replicate_levels = np.full(replicates, float(level), dtype=float)
        x, y = sim._draw_outcomes(rng, replicate_levels, days)
        noise = rng.standard_normal(x.shape)
        daily_a = sim._daily_proxy(x, noise, RHO_A)
        daily_b = sim._daily_proxy(x, noise, RHO_B)
        forecast_a = sim._a_forecast(y, daily_a, population_rate, calibration)
        forecast_b = sim._b_forecast(replicate_levels, daily_b, RHO_B)
        auc_a[patient_index] = _cross_replicate_auc(y, forecast_a)
        auc_b[patient_index] = _cross_replicate_auc(y, forecast_b)

    return auc_a, auc_b


def _mean_and_se(patient_auc: np.ndarray) -> tuple[float, float, int]:
    valid = np.asarray(patient_auc, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return float("nan"), float("nan"), 0
    mean_auc = float(valid.mean())
    se = float(valid.std(ddof=1) / math.sqrt(valid.size)) if valid.size > 1 else float("nan")
    return mean_auc - 0.5, se, int(valid.size)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    payload, references = _references(args.input_json)
    spreads = args.spread or ["low", "high"]
    days_values = args.days or list(sim.LENGTHS)
    cohort_for_spread = {"low": "REPLACE-BG", "high": "CTN-0094"}
    fits = payload["run"]["cohort_fits"]
    calibrations = payload["run"]["calibrations"]
    rows: list[dict[str, Any]] = []

    for spread in spreads:
        cohort = cohort_for_spread[spread]
        spec = sim.cohort_sim.COHORTS[cohort]
        population_rate = float(spec["rate"])
        for days in days_values:
            calibration_key = f"{cohort}|{days}|{RHO_A:.1f}"
            calibration = calibrations.get(calibration_key)
            if calibration is None:
                raise ValueError(f"{args.input_json} is missing A calibration {calibration_key}")
            a_values, b_values = _patient_auc_pair(
                fits[cohort], cohort, days, args.patients, args.replicates,
                calibration, population_rate,
            )
            for model, values in (("A", a_values), ("B", b_values)):
                skill, se, valid_patients = _mean_and_se(values)
                reference = references[(cohort, days)][model]
                rows.append({
                    "spread": spread,
                    "days": days,
                    "model": model,
                    "finite_horizon_skill": skill,
                    "se": se,
                    "long_record_skill": reference["long_record_skill"],
                    "mean_shuffled_excess": reference["mean_shuffled_excess"],
                })
                print(f"{spread:4s} {days:3d}d {model}: finite skill={skill:.5f}, "
                      f"MC SE={se:.5f} (valid patients {valid_patients}/{args.patients})")
                if np.isfinite(se) and se >= TARGET_SE:
                    print(f"WARNING: {spread}/{days}/{model} exceeds target SE {TARGET_SE:.3f}; "
                          "increase --patients and/or --replicates.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["spread", "days", "model", "finite_horizon_skill", "se",
                  "long_record_skill", "mean_shuffled_excess"]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")
    print(f"Monte Carlo target: SE < {TARGET_SE:.3f}; patients={args.patients}, "
          f"replicates/patient={args.replicates}; Table 6 references use 50 patients.")
    return rows


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    run(args)
    print(f"Elapsed: {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
