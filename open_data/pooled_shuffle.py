"""Compare pooled AUC excess with own within-patient shuffled baselines.

Run from the repository root or from any directory:
    python open_data/pooled_shuffle.py [--quick]

The full design uses both rho pairs, both cohorts (spreads), 50 patients,
30/100/365 days, 100 replications, and 50 within-patient shuffles. Replicate
data and shuffles come directly from ranking_sim's keyed seed streams.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results" / "v18"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "code"))

import ranking_sim as sim  # noqa: E402
import ranking_sim_skill as skill  # noqa: E402


PATIENTS = 50
FORECASTS = ("A", "B", "C")
PAIR_ORDER = (
    "A daily rho=0.4 vs B daily rho=0.3",
    "A daily rho=0.6 vs B daily rho=0.2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=None, help="replications per cell (default 100)")
    parser.add_argument("--shuffles", type=int, default=None, help="within-patient shuffles (default 50)")
    parser.add_argument("--workers", type=int, default=4, help="worker processes (default 4)")
    parser.add_argument("--quick", action="store_true", help="2 replications, 5 shuffles, one cell")
    args = parser.parse_args()
    if args.quick:
        args.reps, args.shuffles = 2, 5
    else:
        args.reps = 100 if args.reps is None else args.reps
        args.shuffles = 50 if args.shuffles is None else args.shuffles
    if min(args.reps, args.shuffles, args.workers) < 1:
        parser.error("replications, shuffles, and workers must be at least 1")
    return args


def cells_for_run(quick: bool) -> list[tuple[str, int, int]]:
    if quick:
        return [(sim.COHORTS[0], sim.LENGTHS[0], PATIENTS)]
    return [(cohort, days, PATIENTS) for cohort in sim.COHORTS for days in sim.LENGTHS]


def _patient_auc_mean(y: np.ndarray, forecast: np.ndarray) -> float:
    return sim._finite_mean(sim._patient_means(sim._auc_matrix(y, forecast)))


def _measure_forecast(y: np.ndarray, forecast: np.ndarray, y_perm: np.ndarray,
                      rebuild, population_rate: float) -> dict[str, float]:
    """Measure actual AUCs and their means over the same shuffled outcomes."""
    pooled_auc = float(sim.metrics_auc(y.ravel(), forecast.ravel()))
    per_patient_auc = _patient_auc_mean(y, forecast)
    pooled_null: list[float] = []
    patient_null: list[np.ndarray] = []
    for start in range(0, len(y_perm), 8):
        yb = y_perm[start:start + 8]
        fb = rebuild(yb)
        patient_null.append(sim._auc_matrix(yb, fb))
        for j in range(len(yb)):
            pooled_null.append(float(sim.metrics_auc(yb[j].ravel(), fb[j].ravel())))
    patient_null_matrix = np.concatenate(patient_null, axis=0)
    pooled_shuffled_baseline = float(np.mean(pooled_null))
    per_patient_shuffled_baseline = sim._finite_mean(sim._patient_means(patient_null_matrix))
    return {
        "pooled_auc": pooled_auc,
        "pooled_shuffled_baseline": pooled_shuffled_baseline,
        "pooled_excess": pooled_auc - pooled_shuffled_baseline,
        "per_patient_excess": per_patient_auc - per_patient_shuffled_baseline,
    }


def task_worker(task: dict) -> list[dict]:
    y, y_perm = task["y"], task["y_perm"]
    b, x, noise = task["b"], task["x"], task["noise"]
    pop = task["population_rate"]
    daily_c = sim._daily_proxy(x, noise, 0.0)
    cal_c = task["calibration_rho0"]
    forecast_c = sim._a_forecast(y, daily_c, pop, cal_c)
    result_c = _measure_forecast(
        y, forecast_c, y_perm,
        lambda yb: sim._a_forecast(yb, daily_c[None, :, :], pop, cal_c), pop)

    records: list[dict] = []
    cohort, days, patients = task["cell"]
    for pair in task["pairs"]:
        rho_a, rho_b = pair["rho_a"], pair["rho_b"]
        daily_a = sim._daily_proxy(x, noise, rho_a)
        daily_b = sim._daily_proxy(x, noise, rho_b)
        cal_a = task["calibrations"][str(rho_a)]
        forecast_a = sim._a_forecast(y, daily_a, pop, cal_a)
        forecast_b = sim._b_forecast(b, daily_b, rho_b)
        values = {
            "A": _measure_forecast(
                y, forecast_a, y_perm,
                lambda yb, d=daily_a, c=cal_a: sim._a_forecast(yb, d[None, :, :], pop, c), pop),
            "B": _measure_forecast(
                y, forecast_b, y_perm,
                lambda yb, f=forecast_b: np.broadcast_to(f[None, :, :], yb.shape), pop),
            "C": result_c,
        }
        for forecast_name in FORECASTS:
            records.append({
                "cohort": cohort,
                "spread": "low" if cohort == "REPLACE-BG" else "high",
                "days": days,
                "patients": patients,
                "rep": task["rep"],
                "pair": pair["name"],
                "forecast": forecast_name,
                **values[forecast_name],
            })
    return records


def _tasks_with_c(cells, reps, shuffles, specs, fits, calibrations, pairs):
    for task in sim._main_tasks(cells, reps, shuffles, specs, fits, calibrations, pairs):
        cohort, days, _ = task["cell"]
        task["calibration_rho0"] = calibrations[f"{cohort}|{days}|0.0"]
        yield task


def reproduction_check(task: dict, rows: list[dict]) -> None:
    """Check pooled AUC and tailored shuffled excess against the original worker."""
    original = sim._main_worker(task)["pairs"]
    for pair in task["pairs"]:
        pair_name = pair["name"]
        old = original[pair_name]
        for forecast, pooled_key, actual_key, chance_key in (
            ("A", "pooled_a", "tailored_a_actual", "tailored_a_chance"),
            ("B", "pooled_b", "tailored_b_actual", "tailored_b_chance_shuffle"),
        ):
            row = next(r for r in rows if r["pair"] == pair_name and r["forecast"] == forecast)
            expected_per_patient = old[actual_key]["auc"] - old[chance_key]["auc"]
            if not np.isclose(row["pooled_auc"], old[pooled_key]["auc"], atol=1e-12):
                raise AssertionError((pair_name, forecast, "pooled AUC mismatch"))
            if not np.isclose(row["per_patient_excess"], expected_per_patient, atol=1e-12):
                raise AssertionError((pair_name, forecast, "per-patient shuffled excess mismatch"))


def _mean_mcse(values) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else float("nan")
    return mean, se


def summarize(rep_rows: list[dict], truth: pd.DataFrame) -> pd.DataFrame:
    reps = pd.DataFrame(rep_rows)
    truth_auc = truth[truth["measure"] == "auc"].set_index(
        ["cohort", "record_days", "pair", "forecast"])["true_skill"].to_dict()
    groups = ["cohort", "spread", "days", "patients", "pair", "forecast"]
    metrics = ("pooled_auc", "pooled_shuffled_baseline", "pooled_excess", "per_patient_excess")
    rows = []
    for key, sub in reps.groupby(groups, sort=False):
        row = dict(zip(groups, key))
        row["n_reps"] = int(sub["rep"].nunique())
        for metric in metrics:
            row[f"{metric}_mean"], row[f"{metric}_mcse"] = _mean_mcse(sub[metric])
        row["true_skill"] = (0.0 if row["forecast"] == "C" else
                             float(truth_auc[(row["cohort"], row["days"], row["pair"], row["forecast"])]))
        rows.append(row)
    result = pd.DataFrame(rows)

    share_rows = []
    for key, sub in reps.groupby(["cohort", "spread", "days", "patients", "pair", "rep"], sort=False):
        vals = sub.set_index("forecast")
        pooled_a, pooled_b = vals.loc["A", "pooled_excess"], vals.loc["B", "pooled_excess"]
        patient_a = vals.loc["A", "per_patient_excess"]
        patient_b = vals.loc["B", "per_patient_excess"]
        share_rows.append({
            **dict(zip(["cohort", "spread", "days", "patients", "pair", "rep"], key)),
            "pooled_a_first": (float(pooled_a > pooled_b)
                               if np.isfinite(pooled_a) and np.isfinite(pooled_b) else np.nan),
            "per_patient_a_first": (float(patient_a > patient_b)
                                    if np.isfinite(patient_a) and np.isfinite(patient_b) else np.nan),
        })
    share_df = pd.DataFrame(share_rows).groupby(
        ["cohort", "spread", "days", "patients", "pair"], sort=False
    )[["pooled_a_first", "per_patient_a_first"]].mean().reset_index()
    result = result.merge(share_df, on=["cohort", "spread", "days", "patients", "pair"], how="left")
    pair_rank = {name: idx for idx, name in enumerate(PAIR_ORDER)}
    result["_pair_order"] = result["pair"].map(pair_rank)
    result["_spread_order"] = result["spread"].map({"low": 0, "high": 1})
    result["_forecast_order"] = result["forecast"].map({"A": 0, "B": 1, "C": 2})
    return result.sort_values(["_pair_order", "_spread_order", "days", "_forecast_order"]).drop(
        columns=["_pair_order", "_spread_order", "_forecast_order"])


def _fmt(value) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.3f}"


def _fmt_mean_se(row, name: str) -> str:
    mean, se = row[f"{name}_mean"], row[f"{name}_mcse"]
    return f"{_fmt(mean)} ({_fmt(se)})"


def markdown(summary: pd.DataFrame, *, profile: str, reps: int, shuffles: int,
             workers: int, check_passed: bool, elapsed: float, full_estimate: float | None) -> str:
    lines = [
        "# Pooled AUC against within-patient shuffled baselines",
        "",
        f"Run: **{profile}**; {reps} replications per cell; {shuffles} shuffles per replication; {workers} workers.",
        "",
        "Each shuffle independently reorders each patient's outcomes. A and C rebuild their running-rate inputs from the shuffled outcomes; B keeps its forecast fixed. The pooled baseline takes one AUC over all patients and days for each shuffle, then averages those AUCs. Per-patient excess uses the equal-patient AUC and its own mean shuffled baseline, following ranking_sim.",
        "",
        "Forecast C uses ranking_sim's rho=0 calibration, which is fitted without a daily covariate (`use_daily=False`). Its score is the calibrated running-rate feature alone; a rho=0 proxy is constructed but its fitted coefficient is zero. C's true skill is set to 0 as requested.",
        "",
        f"Reproduction check against ranking_sim on one replication: **{'passed' if check_passed else 'not run'}**. Elapsed time: **{elapsed:.1f} s**.",
        "",
        "Cells show replication mean (Monte Carlo SE, sample SD / sqrt(n)); true skill is the AUC skill benchmark from `ranking_sim_skill.truth()` (AUC - 0.5) for A/B, and 0 for C. A>B shares compare A with B within the same replication.",
        "",
    ]
    for pair in PAIR_ORDER:
        sub = summary[summary["pair"] == pair]
        label = "close (rho 0.4 vs 0.3)" if "0.4" in pair else "wide (rho 0.6 vs 0.2)"
        lines += [
            f"## {label}",
            "",
            "| Spread | Days | Forecast | Pooled AUC | Pooled shuffled baseline | Pooled excess | Per-patient excess | True skill | A>B pooled | A>B per-patient |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for _, row in sub.iterrows():
            lines.append(
                f"| {row['spread']} | {int(row['days'])} | {row['forecast']} | "
                f"{_fmt_mean_se(row, 'pooled_auc')} | {_fmt_mean_se(row, 'pooled_shuffled_baseline')} | "
                f"{_fmt_mean_se(row, 'pooled_excess')} | {_fmt_mean_se(row, 'per_patient_excess')} | "
                f"{_fmt(row['true_skill'])} | {_fmt(row['pooled_a_first'])} | {_fmt(row['per_patient_a_first'])} |"
            )
        lines.append("")
    if full_estimate is not None:
        lines += [f"Estimated full run time with {workers} workers, based on this run's replication throughput: **{full_estimate:.1f} minutes**.", ""]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cells = cells_for_run(args.quick)
    pairs = [dict(sim.EXTRA_PAIR), dict(sim.BASE_PAIR)]  # close first, then wide
    start = time.perf_counter()
    context_start = time.perf_counter()
    specs, fits, calibrations = sim._context(cells, True)
    context_seconds = time.perf_counter() - context_start

    tasks = iter(_tasks_with_c(cells, args.reps, args.shuffles, specs, fits, calibrations, pairs))
    first_task = next(tasks)
    first_task_start = time.perf_counter()
    first_rows = task_worker(first_task)
    first_task_seconds = time.perf_counter() - first_task_start
    reproduction_check(first_task, first_rows)
    replication_start = time.perf_counter()
    remaining = sim._parent_generated_map(task_worker, tasks, args.workers)
    replication_seconds = time.perf_counter() - replication_start
    rep_rows = first_rows + [row for task_rows in remaining for row in task_rows]

    truth_start = time.perf_counter()
    truth = skill.truth(specs, fits, calibrations, pairs, cells)
    truth_seconds = time.perf_counter() - truth_start
    summary = summarize(rep_rows, truth)

    # Extrapolate simulation work and the fixed calibration/long-truth setup to
    # the requested full design using the in-process first task, which avoids
    # charging one-time Windows process startup to the per-task throughput.
    full_cells = cells_for_run(False)
    def auc_sort_weight(weight_cells, reps, shuffles):
        return sum(patients * days * (math.log2(max(2, days))
                                      + math.log2(max(2, patients * days)))
                   * reps * shuffles * 2.5
                   for _, days, patients in weight_cells)

    pilot_weight = auc_sort_weight([first_task["cell"]], 1, args.shuffles)
    full_weight = auc_sort_weight(full_cells, 100, 50)
    full_replication_seconds = (first_task_seconds * full_weight
                                / max(pilot_weight, 1.0) / args.workers)
    unique_pilot = len({(c, d) for c, d, _ in cells})
    unique_full = len({(c, d) for c, d, _ in full_cells})
    full_context_seconds = context_seconds * unique_full / max(unique_pilot, 1)
    pilot_cohorts = len({c for c, _, _ in cells})
    full_cohorts = len(sim.COHORTS)
    full_truth_seconds = truth_seconds * full_cohorts / max(pilot_cohorts, 1)
    full_estimate_seconds = full_replication_seconds + full_context_seconds + full_truth_seconds
    elapsed = time.perf_counter() - start

    summary.insert(0, "profile", "quick" if args.quick else "full")
    summary.insert(1, "shuffles", args.shuffles)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "pooled_shuffle.csv", index=False)
    (OUT / "pooled_shuffle.md").write_text(
        markdown(summary, profile="quick" if args.quick else "full", reps=args.reps,
                 shuffles=args.shuffles, workers=args.workers, check_passed=True,
                 elapsed=elapsed, full_estimate=full_estimate_seconds / 60.0),
        encoding="utf-8")
    print(f"Reproduction check: passed (pooled AUC and per-patient shuffled excess for A/B).")
    print(f"Calibration/context: {context_seconds:.1f}s; first task: {first_task_seconds:.2f}s; "
          f"remaining replications: {replication_seconds:.1f}s; truth: {truth_seconds:.1f}s; "
          f"elapsed: {elapsed:.1f}s; estimated full run: {full_estimate_seconds / 60.0:.1f} min "
          f"with {args.workers} workers.")


if __name__ == "__main__":
    main()
