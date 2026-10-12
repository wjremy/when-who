"""Simulation: do tailored patient scores rank forecasts by true daily ability?

Run from any working directory with the project Python environment:
    python code/ranking_sim.py [--quick] [--workers N] [--extra-pair]

The quick profile uses the first cell, 20 ranking replications, 50 permutation
draws, and 20 power replications with 50 shuffles. All random arrays are
produced by the parent process before work is sent to workers, so outputs do
not depend on --workers. `--tailored-only` replays ranking seed streams and
merges tailored rows into an existing result JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
WHENEVAL_SRC = CODE / "wheneval" / "src"
for import_path in (str(CODE), str(WHENEVAL_SRC)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

# This script uses only the package's metrics and nine submodules. Register its
# source directory directly so an unrelated stale top-level report export does
# not prevent those submodules from being imported.
wheneval_package = types.ModuleType("wheneval")
wheneval_package.__path__ = [str(WHENEVAL_SRC / "wheneval")]
wheneval_package.__package__ = "wheneval"
sys.modules.setdefault("wheneval", wheneval_package)

import cohort_curves as cohort_sim  # noqa: E402
import who_when_sim as who_when  # noqa: E402
from wheneval.metrics import auc as metrics_auc  # noqa: E402
from wheneval.nine import average_precision  # noqa: E402


BASE_SEED = 20261011
X_SD = 2.0
PRIOR_N = 4.0
LONG_DAYS = 5_000
LONG_PATIENTS = 200
CALIBRATION_ROWS = 200_000
COHORTS = ("REPLACE-BG", "CTN-0094")
LENGTHS = (30, 100, 365)
PATIENT_COUNTS = (50, 150)
BASE_PAIR = {"name": "A daily rho=0.6 vs B daily rho=0.2", "rho_a": 0.6, "rho_b": 0.2}
EXTRA_PAIR = {"name": "A daily rho=0.4 vs B daily rho=0.3", "rho_a": 0.4, "rho_b": 0.3}
METRIC_KEYS = ("auc", "average_precision", "brier_skill", "ppv", "net_benefit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="one cell, 20 reps, 50 shuffles")
    parser.add_argument("--reps", type=int, help="override ranking replications per cell (default 500)")
    parser.add_argument("--permutations", type=int, help="A chance draws per ranking replication (default 50)")
    parser.add_argument("--power-reps", type=int, help="override power replications per cell (default 200)")
    parser.add_argument("--power-permutations", type=int, help="override power shuffles (default 200)")
    parser.add_argument("--workers", type=int, default=1, help="worker processes; random arrays are always drawn in parent")
    parser.add_argument("--b-chance", choices=("shuffle", "levelonly"), default="shuffle",
                        help="B's tailored chance level: within-patient shuffles (default) or its level-only forecast")
    parser.add_argument("--tailored-only", action="store_true",
                        help="recompute tailored rankings only and merge into an existing result JSON")
    parser.add_argument("--input-json", type=Path,
                        help="source JSON for --tailored-only (default: open_data/results/v10/ranking_sim.json)")
    parser.add_argument("--output-suffix", default="",
                        help="append a suffix to output filenames, e.g. _quick")
    pair = parser.add_mutually_exclusive_group()
    pair.add_argument("--extra-pair", dest="extra_pair", action="store_true",
                      help="include the rho 0.4 vs 0.3 pair")
    pair.add_argument("--no-extra-pair", dest="extra_pair", action="store_false",
                      help="omit the optional pair")
    parser.set_defaults(extra_pair=None)
    args = parser.parse_args()
    args.reps = args.reps if args.reps is not None else (20 if args.quick else 500)
    args.permutations = args.permutations if args.permutations is not None else 50
    args.power_reps = args.power_reps if args.power_reps is not None else (20 if args.quick else 200)
    args.power_permutations = args.power_permutations if args.power_permutations is not None else (50 if args.quick else 200)
    if min(args.reps, args.permutations, args.power_reps, args.power_permutations, args.workers) < 1:
        parser.error("repetitions, permutations, and workers must be at least 1")
    if args.quick:
        args.cells = [(COHORTS[0], LENGTHS[0], PATIENT_COUNTS[0])]
    else:
        args.cells = [(cohort, length, patients)
                      for cohort in COHORTS for length in LENGTHS for patients in PATIENT_COUNTS]
    return args


def _rng(*parts: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence((BASE_SEED, *parts)))


def _expit(values: np.ndarray | float) -> np.ndarray:
    return who_when.expit(values)


def _recentered_levels(rng: np.random.Generator, fit: dict[str, Any], spec: dict[str, Any],
                       n_patients: int) -> np.ndarray:
    base = cohort_sim.draw_effects(rng, fit, n_patients)
    z_ref = rng.standard_normal(512)
    shift = cohort_sim.recenter(base, X_SD, float(spec["rate"]), z_ref)
    return np.asarray(base + shift, dtype=float)


def _running_rate(y: np.ndarray, population_rate: float) -> np.ndarray:
    """P(Y_t=1) estimate from strictly earlier outcomes and four prior observations."""
    yy = np.asarray(y, dtype=float)
    n = yy.shape[-1]
    before = np.cumsum(yy, axis=-1) - yy
    times = np.arange(n, dtype=float)
    return (PRIOR_N * population_rate + before) / (PRIOR_N + times)


def _logit_rate(rate: np.ndarray) -> np.ndarray:
    clipped = np.clip(rate, 1e-10, 1.0 - 1e-10)
    return np.log(clipped) - np.log1p(-clipped)


def _daily_proxy(x: np.ndarray, noise: np.ndarray, rho: float) -> np.ndarray:
    return rho * x + math.sqrt(max(0.0, 1.0 - rho * rho)) * X_SD * noise


def _draw_outcomes(rng: np.random.Generator, b: np.ndarray, days: int) -> tuple[np.ndarray, np.ndarray]:
    x = X_SD * rng.standard_normal((len(b), days))
    y = (rng.random(x.shape) < _expit(b[:, None] + x)).astype(np.int8)
    return x, y


def _fit_a_calibration(cohort: str, cohort_index: int, spec: dict[str, Any], fit: dict[str, Any],
                       days: int, rho: float, *, use_daily: bool = True) -> dict[str, Any]:
    """Fit A on a separate >=200k-row cohort, using who_when_sim's logistic IRLS."""
    n_patients = max(1, math.ceil(CALIBRATION_ROWS / days))
    rng = _rng(801, cohort_index, days, int(round(rho * 1000)), int(use_daily))
    b = _recentered_levels(rng, fit, spec, n_patients)
    x, y = _draw_outcomes(rng, b, days)
    noise = rng.standard_normal(x.shape)
    daily = _daily_proxy(x, noise, rho)
    level_logit = _logit_rate(_running_rate(y, float(spec["rate"])))
    covariates = (np.column_stack([level_logit.ravel(), daily.ravel()]) if use_daily
                  else level_logit.ravel()[:, None])
    beta = who_when.metrics_irls(covariates, y.ravel().astype(float))
    if beta is None or not np.isfinite(beta).all():
        raise RuntimeError(f"A calibration IRLS failed for {cohort}, n={days}, rho={rho}")
    return {
        "intercept": float(beta[0]),
        "level_coefficient": float(beta[1]),
        "daily_coefficient": float(beta[2]) if use_daily else 0.0,
        "use_daily": bool(use_daily),
        "calibration_rows": int(y.size),
        "calibration_patients": int(n_patients),
        "calibration_seed_stream": [BASE_SEED, 801, cohort_index, days,
                                    int(round(rho * 1000)), int(use_daily)],
    }


def _a_forecast(y: np.ndarray, daily: np.ndarray, population_rate: float,
                calibration: dict[str, Any]) -> np.ndarray:
    level = _logit_rate(_running_rate(y, population_rate))
    return _expit(float(calibration["intercept"])
                  + float(calibration["level_coefficient"]) * level
                  + float(calibration["daily_coefficient"]) * daily)


def _b_forecast(b: np.ndarray, daily: np.ndarray, rho: float) -> np.ndarray:
    """Exact conditional risk E[expit(b+x)|b,d], as in who_when_sim."""
    residual_variance = (1.0 - rho * rho) * X_SD * X_SD
    return who_when.normal_expit_expectation(b[:, None] + rho * daily, residual_variance)


def _auc_matrix(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Vectorized per-patient AUC for leading dimensions of draws x patients x days."""
    yy = np.asarray(y, dtype=np.int8)
    ss = np.asarray(score, dtype=float)
    if yy.ndim == 2:
        yy = yy[None, ...]
    if ss.ndim == 2:
        ss = ss[None, ...]
    if yy.shape != ss.shape:
        yy = np.broadcast_to(yy, ss.shape)
    order = np.argsort(ss, axis=-1, kind="stable")
    sorted_s = np.take_along_axis(ss, order, axis=-1)
    sorted_y = np.take_along_axis(yy, order, axis=-1).astype(float)
    n = ss.shape[-1]
    positions = np.broadcast_to(np.arange(n, dtype=float), ss.shape)
    starts = np.ones(ss.shape, dtype=bool)
    starts[..., 1:] = sorted_s[..., 1:] != sorted_s[..., :-1]
    ends = np.ones(ss.shape, dtype=bool)
    ends[..., :-1] = sorted_s[..., :-1] != sorted_s[..., 1:]
    left = np.maximum.accumulate(np.where(starts, positions, 0.0), axis=-1)
    rev_right = np.minimum.accumulate(np.where(ends, positions, float(n - 1))[..., ::-1], axis=-1)[..., ::-1]
    ranks = (left + rev_right + 2.0) / 2.0
    n1 = np.sum(yy, axis=-1).astype(float)
    n0 = n - n1
    rank_sum = np.sum(ranks * sorted_y, axis=-1)
    denom = n1 * n0
    out = np.full(n1.shape, np.nan, dtype=float)
    np.divide(rank_sum - n1 * (n1 + 1.0) / 2.0, denom, out=out, where=denom > 0)
    return out


def _patient_statistics(y: np.ndarray, forecast: np.ndarray, threshold: float,
                        population_rate: float, patient_rate: np.ndarray,
                        updated_reference: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Selected nine_metrics/naive_metrics patient definitions, vectorized over draws."""
    yy = np.asarray(y, dtype=np.int8)
    pp = np.asarray(forecast, dtype=float)
    if yy.ndim == 2:
        yy = yy[None, ...]
    if pp.ndim == 2:
        pp = pp[None, ...]
    if yy.shape != pp.shape:
        if pp.shape[-1] == 1 and yy.shape[-1] > 1:
            pp = np.broadcast_to(pp, yy.shape)
        else:
            yy = np.broadcast_to(yy, pp.shape)
    d, k, n = yy.shape
    aucs = _auc_matrix(yy, pp)

    # Non-interpolated AP with all tied scores entering at the same threshold.
    order = np.argsort(-pp, axis=-1, kind="stable")
    sorted_s = np.take_along_axis(pp, order, axis=-1)
    sorted_y = np.take_along_axis(yy, order, axis=-1).astype(float)
    cum_pos = np.cumsum(sorted_y, axis=-1)
    ends = np.ones(sorted_s.shape, dtype=bool)
    ends[..., :-1] = sorted_s[..., :-1] != sorted_s[..., 1:]
    last_group_cum = np.maximum.accumulate(np.where(ends, cum_pos, 0.0), axis=-1)
    prior_group_cum = np.concatenate([np.zeros((*last_group_cum.shape[:-1], 1)),
                                      last_group_cum[..., :-1]], axis=-1)
    block_pos = np.where(ends, cum_pos - prior_group_cum, 0.0)
    total_pos = np.sum(yy, axis=-1).astype(float)
    precision = cum_pos / np.arange(1, n + 1, dtype=float)
    ap = np.full((d, k), np.nan, dtype=float)
    np.divide(np.sum(block_pos * precision, axis=-1), total_pos, out=ap, where=total_pos > 0)

    own_rate = np.mean(yy, axis=-1)
    alarm = pp >= threshold
    alarm_n = np.sum(alarm, axis=-1)
    tp = np.sum(alarm & (yy == 1), axis=-1)
    fp = np.sum(alarm & (yy == 0), axis=-1)
    ppv = np.full((d, k), np.nan, dtype=float)
    np.divide(tp, alarm_n, out=ppv, where=alarm_n > 0)
    ppv_excess = ppv - own_rate
    odds = threshold / (1.0 - threshold)
    raw_nb = tp / n - fp / n * odds
    own_policy_nb = np.where(own_rate >= threshold, own_rate - (1.0 - own_rate) * odds, 0.0)
    nb_excess = raw_nb - own_policy_nb

    def skill(reference: np.ndarray) -> np.ndarray:
        rr = np.asarray(reference, dtype=float)
        if rr.ndim == 1:
            rr = rr[None, :, None]
        elif rr.ndim == 2:
            rr = rr[None, :, :] if rr.shape == yy.shape[1:] else rr[:, :, None]
        model_loss = np.mean((yy - pp) ** 2, axis=-1)
        ref_loss = np.mean((yy - rr) ** 2, axis=-1)
        value = np.full((d, k), np.nan, dtype=float)
        np.divide(model_loss, ref_loss, out=value, where=ref_loss > 0)
        return 1.0 - value

    updated = (_running_rate(yy, population_rate) if updated_reference is None
               else np.asarray(updated_reference, dtype=float))
    return {
        "auc": aucs,
        "average_precision": ap,
        "brier_skill_updated_rate": skill(updated),
        "brier_skill_oracle": skill(np.broadcast_to(patient_rate[None, :, None], yy.shape)),
        "brier_skill_population_rate": skill(np.full((1, k, 1), population_rate)),
        "ppv": ppv_excess,
        "ppv_raw": ppv,
        "net_benefit": nb_excess,
        "net_benefit_raw": raw_nb,
    }


def _patient_means(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    counts = finite.sum(axis=0)
    sums = np.where(finite, arr, 0.0).sum(axis=0)
    out = np.full(arr.shape[1], np.nan, dtype=float)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def _finite_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def _score_vector(y: np.ndarray, forecast: np.ndarray, population_rate: float,
                  patient_rate: np.ndarray) -> dict[str, float]:
    stats = _patient_statistics(y, forecast, population_rate, population_rate, patient_rate)
    return {name: _finite_mean(_patient_means(stats[key]))
            for name, key in (("auc", "auc"), ("average_precision", "average_precision"),
                              ("brier_skill_updated_rate", "brier_skill_updated_rate"),
                              ("brier_skill_oracle", "brier_skill_oracle"),
                              ("ppv", "ppv"), ("net_benefit", "net_benefit"))}


def _pooled_score(y: np.ndarray, forecast: np.ndarray, population_rate: float,
                  patient_rate: np.ndarray) -> dict[str, float]:
    yy = np.asarray(y, dtype=np.int8)
    pp = np.asarray(forecast, dtype=float)
    threshold = population_rate
    updated = _running_rate(yy, population_rate)
    odds = threshold / (1.0 - threshold)
    alarm = pp >= threshold
    tp = int(np.sum(alarm & (yy == 1)))
    fp = int(np.sum(alarm & (yy == 0)))
    out = {
        "auc": float(metrics_auc(yy.ravel(), pp.ravel())),
        "average_precision": float(average_precision(yy.ravel(), pp.ravel())),
        "ppv": float(tp / int(alarm.sum())) if alarm.any() else float("nan"),
        "net_benefit": float(tp / yy.size - fp / yy.size * odds),
    }
    for key, reference in (
        ("brier_skill_updated_rate", updated),
        ("brier_skill_oracle", np.broadcast_to(patient_rate[:, None], yy.shape)),
    ):
        den = float(np.mean((yy - reference) ** 2))
        out[key] = float(1.0 - np.mean((yy - pp) ** 2) / den) if den > 0 else float("nan")
    return out


def _naive_score(y: np.ndarray, forecast: np.ndarray, population_rate: float,
                 patient_rate: np.ndarray) -> dict[str, float]:
    """Raw per-patient values matching the selected naive_metrics definitions."""
    stats = _patient_statistics(y, forecast, population_rate, population_rate, patient_rate)
    output: dict[str, float] = {}
    for name, key in (("auc", "auc"), ("average_precision", "average_precision"),
                      ("brier_skill_population_rate", "brier_skill_population_rate"),
                      ("ppv", "ppv_raw"), ("net_benefit", "net_benefit_raw")):
        output[name] = _finite_mean(_patient_means(stats[key]))
    # naive_metrics.summary.mean reports the equal-patient ratio of mean losses,
    # rather than the mean of its patient-specific Brier-skill ratios.
    model_loss = np.mean((y - forecast) ** 2, axis=-1)
    reference_loss = np.mean((y - population_rate) ** 2, axis=-1)
    denominator = float(np.mean(reference_loss))
    output["brier_skill_population_rate"] = (
        float(1.0 - np.mean(model_loss) / denominator) if denominator > 0 else float("nan"))
    return output


def _permutation_chance(y_perm: np.ndarray, daily: np.ndarray | None, population_rate: float,
                        patient_rate: np.ndarray, calibration: dict[str, Any] | None,
                        threshold: float, *, fixed_forecast: np.ndarray | None = None,
                        chunk_size: int = 8) -> dict[str, np.ndarray]:
    """Chance from within-patient shuffles; A is rebuilt, while B stays fixed."""
    keys = ("auc", "average_precision", "brier_skill_updated_rate", "brier_skill_oracle",
            "ppv", "net_benefit")
    pieces: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    for start in range(0, len(y_perm), chunk_size):
        y = y_perm[start:start + chunk_size]
        if fixed_forecast is None:
            if daily is None or calibration is None:
                raise ValueError("A's shuffle chance requires its daily proxy and calibration")
            forecast = _a_forecast(y, daily[None, :, :], population_rate, calibration)
        else:
            forecast = np.broadcast_to(fixed_forecast[None, :, :], y.shape)
        stats = _patient_statistics(y, forecast, threshold, population_rate, patient_rate)
        for key in keys:
            pieces[key].append(stats[key])
    result: dict[str, np.ndarray] = {}
    for key in keys:
        matrix = np.concatenate(pieces[key], axis=0)
        result[key] = _patient_means(matrix)
    return result


def _make_shuffles(rng: np.random.Generator, y: np.ndarray, count: int) -> np.ndarray:
    keys = rng.random((count, *y.shape))
    order = np.argsort(keys, axis=-1, kind="stable")
    return np.take_along_axis(np.broadcast_to(y, (count, *y.shape)), order, axis=-1).astype(np.int8)


def _main_worker(task: dict[str, Any]) -> dict[str, Any]:
    y = task["y"]
    b = task["b"]
    x = task["x"]
    noise = task["noise"]
    population_rate = task["population_rate"]
    patient_rate = task["patient_rate"]
    threshold = population_rate
    y_perm = task["y_perm"]
    pair_scores: dict[str, dict[str, dict[str, float]]] = {}

    for pair in task["pairs"]:
        name = pair["name"]
        rho_a, rho_b = pair["rho_a"], pair["rho_b"]
        daily_a = _daily_proxy(x, noise, rho_a)
        daily_b = _daily_proxy(x, noise, rho_b)
        cal_a = task["calibrations"][str(rho_a)]
        forecast_a = _a_forecast(y, daily_a, population_rate, cal_a)
        forecast_b = _b_forecast(b, daily_b, rho_b)
        level_only_b = who_when.normal_expit_expectation(b[:, None], X_SD * X_SD)

        tailored_only = bool(task.get("tailored_only", False))
        if not tailored_only:
            pooled_a = _pooled_score(y, forecast_a, population_rate, patient_rate)
            pooled_b = _pooled_score(y, forecast_b, population_rate, patient_rate)
            naive_a = _naive_score(y, forecast_a, population_rate, patient_rate)
            naive_b = _naive_score(y, forecast_b, population_rate, patient_rate)

        actual_a_updated = _patient_statistics(y, forecast_a, threshold, population_rate, patient_rate,
                                               _running_rate(y, population_rate))
        actual_a_oracle = _patient_statistics(y, forecast_a, threshold, population_rate, patient_rate,
                                              np.broadcast_to(_running_rate(y, population_rate), y.shape))
        actual_b_updated = _patient_statistics(y, forecast_b, threshold, population_rate, patient_rate,
                                               _running_rate(y, population_rate))
        actual_b_oracle = _patient_statistics(y, forecast_b, threshold, population_rate, patient_rate,
                                              np.broadcast_to(_running_rate(y, population_rate), y.shape))
        base_b_updated = _patient_statistics(y, level_only_b, threshold, population_rate, patient_rate,
                                             _running_rate(y, population_rate))
        base_b_oracle = _patient_statistics(y, level_only_b, threshold, population_rate, patient_rate,
                                            np.broadcast_to(_running_rate(y, population_rate), y.shape))
        chance_a = _permutation_chance(y_perm, daily_a, population_rate, patient_rate, cal_a,
                                       threshold, chunk_size=task["chunk_size"])
        chance_b_shuffle = _permutation_chance(
            y_perm, None, population_rate, patient_rate, None, threshold,
            fixed_forecast=forecast_b, chunk_size=task["chunk_size"])
        tailored_b_actual = {
            "auc": _finite_mean(_patient_means(actual_b_updated["auc"])),
            "average_precision": _finite_mean(_patient_means(actual_b_updated["average_precision"])),
            "brier_skill_updated_rate": _finite_mean(_patient_means(actual_b_updated["brier_skill_updated_rate"])),
            "brier_skill_oracle": _finite_mean(_patient_means(actual_b_oracle["brier_skill_oracle"])),
            "ppv": _finite_mean(_patient_means(actual_b_updated["ppv"])),
            "net_benefit": _finite_mean(_patient_means(actual_b_updated["net_benefit"])),
        }
        # B against its level-only forecast: its observed AUC against 0.5 (the first run set B's AUC to 0.5 by mistake).
        tailored_b_actual_legacy = dict(tailored_b_actual)

        pair_scores[name] = ({
            "pooled_a": pooled_a,
            "pooled_b": pooled_b,
            "naive_a": naive_a,
            "naive_b": naive_b,
        } if not tailored_only else {}) | {
            "tailored_a_actual": {
                "auc": _finite_mean(_patient_means(actual_a_updated["auc"])),
                "average_precision": _finite_mean(_patient_means(actual_a_updated["average_precision"])),
                "brier_skill_updated_rate": _finite_mean(_patient_means(actual_a_updated["brier_skill_updated_rate"])),
                "brier_skill_oracle": _finite_mean(_patient_means(actual_a_oracle["brier_skill_oracle"])),
                "ppv": _finite_mean(_patient_means(actual_a_updated["ppv"])),
                "net_benefit": _finite_mean(_patient_means(actual_a_updated["net_benefit"])),
            },
            "tailored_a_chance": {key: _finite_mean(value) for key, value in chance_a.items()},
            "tailored_b_actual": tailored_b_actual,
            "tailored_b_actual_legacy": tailored_b_actual_legacy,
            "tailored_b_chance": {
                "auc": 0.5,
                "average_precision": _finite_mean(_patient_means(base_b_updated["average_precision"])),
                "brier_skill_updated_rate": _finite_mean(_patient_means(base_b_updated["brier_skill_updated_rate"])),
                "brier_skill_oracle": _finite_mean(_patient_means(base_b_oracle["brier_skill_oracle"])),
                "ppv": _finite_mean(_patient_means(base_b_updated["ppv"])),
                "net_benefit": _finite_mean(_patient_means(base_b_updated["net_benefit"])),
            },
            "tailored_b_chance_shuffle": {key: _finite_mean(value)
                                          for key, value in chance_b_shuffle.items()},
        }

    return {"cell": task["cell"], "rep": task["rep"], "pairs": pair_scores}


def _power_pvalue(y: np.ndarray, forecast_actual: np.ndarray, y_perm: np.ndarray,
                  *, a_calibration: dict[str, Any] | None, daily: np.ndarray | None,
                  population_rate: float, tie_score: np.ndarray | None = None,
                  chunk_size: int = 8) -> tuple[float, float]:
    score = tie_score if tie_score is not None else forecast_actual
    observed_patients = _auc_matrix(y, score)[0]
    observed = _finite_mean(observed_patients)
    null_chunks: list[np.ndarray] = []
    for start in range(0, len(y_perm), chunk_size):
        ypart = y_perm[start:start + chunk_size]
        if tie_score is not None:
            pred = np.broadcast_to(tie_score, ypart.shape)
        elif daily is not None and a_calibration is not None:
            pred = _a_forecast(ypart, daily[None, :, :], population_rate, a_calibration)
        else:
            pred = np.broadcast_to(forecast_actual, ypart.shape)
        null_chunks.append(_auc_matrix(ypart, pred))
    null = np.concatenate(null_chunks, axis=0)
    null_mean = np.asarray([_finite_mean(row) for row in null], dtype=float)
    valid = null_mean[np.isfinite(null_mean)]
    p_value = ((1 + int(np.sum(valid >= observed))) / (len(valid) + 1)
               if len(valid) and np.isfinite(observed) else float("nan"))
    return p_value, observed


def _power_worker(task: dict[str, Any]) -> dict[str, Any]:
    kind = task["test"]
    y = task["y"]
    b, x, noise = task["b"], task["x"], task["noise"]
    pop, patient_rate = task["population_rate"], task["patient_rate"]
    if kind == "B_daily_rho0_size":
        # B's exact rho=0 probability is constant within patient. A fixed iid
        # tie-breaker gives the requested size diagnostic a non-degenerate AUC.
        pval, obs = _power_pvalue(y, np.empty_like(y, dtype=float), task["y_perm"],
                                  a_calibration=None, daily=None, population_rate=pop,
                                  tie_score=task["tie_score"], chunk_size=task["chunk_size"])
    else:
        if kind == "A_main_power":
            rho, cal = task["rho_a"], task["calibration"]
            daily = _daily_proxy(x, noise, rho)
        else:
            cal = task["calibration"]
            daily = np.zeros_like(x)
        forecast = _a_forecast(y, daily, pop, cal)
        pval, obs = _power_pvalue(y, forecast, task["y_perm"], a_calibration=cal,
                                  daily=daily, population_rate=pop,
                                  chunk_size=task["chunk_size"])
    return {"cell": task["cell"], "test": kind, "rep": task["rep"],
            "reject": bool(np.isfinite(pval) and pval <= 0.05),
            "p_value": pval, "observed_auc": obs}


def _parent_generated_map(worker: Any, tasks: Iterable[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    """Bound in-flight tasks; each task's random arrays have already been made here."""
    if workers == 1:
        return [worker(task) for task in tasks]
    results: list[dict[str, Any]] = []
    iterator = iter(tasks)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending: set[Any] = set()
        for _ in range(workers * 2):
            try:
                pending.add(executor.submit(worker, next(iterator)))
            except StopIteration:
                break
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                results.append(future.result())
                try:
                    pending.add(executor.submit(worker, next(iterator)))
                except StopIteration:
                    pass
    return results


def _cell_index(cell: tuple[str, int, int]) -> int:
    return (COHORTS.index(cell[0]) * 6 + LENGTHS.index(cell[1]) * 2 + PATIENT_COUNTS.index(cell[2]))


def _context(cells: list[tuple[str, int, int]], include_extra: bool
             ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    specs = {name: cohort_sim.COHORTS[name] for name in COHORTS}
    fits = {name: cohort_sim.fit_cohort(specs[name]) for name in COHORTS}
    cohort_indices = {name: idx for idx, name in enumerate(COHORTS)}
    calibrations: dict[str, Any] = {}
    rho_values = (0.6, 0.4) if include_extra else (0.6,)
    used = {(cohort, days) for cohort, days, _ in cells}
    for cohort, days in sorted(used, key=lambda item: (COHORTS.index(item[0]), item[1])):
        for rho in rho_values:
            calibrations[f"{cohort}|{days}|{rho:.1f}"] = _fit_a_calibration(
                cohort, cohort_indices[cohort], specs[cohort], fits[cohort], days, rho)
        calibrations[f"{cohort}|{days}|0.0"] = _fit_a_calibration(
            cohort, cohort_indices[cohort], specs[cohort], fits[cohort], days, 0.0, use_daily=False)
    return specs, fits, calibrations


def _main_tasks(cells: list[tuple[str, int, int]], reps: int, n_perm: int,
                specs: dict[str, Any], fits: dict[str, Any], calibrations: dict[str, Any],
                pairs: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for cell in cells:
        cohort, days, n_patients = cell
        ci = _cell_index(cell)
        spec, fit = specs[cohort], fits[cohort]
        pop = float(spec["rate"])
        cal_pair = {str(pair["rho_a"]): calibrations[f"{cohort}|{days}|{pair['rho_a']:.1f}"] for pair in pairs}
        for rep in range(reps):
            rng = _rng(10, ci, rep)
            b = _recentered_levels(rng, fit, spec, n_patients)
            x, y = _draw_outcomes(rng, b, days)
            noise = rng.standard_normal(x.shape)
            y_perm = _make_shuffles(rng, y, n_perm)
            yield {
                "cell": cell, "rep": rep, "b": b, "x": x, "y": y, "noise": noise,
                "patient_rate": who_when.normal_expit_expectation(b, X_SD * X_SD),
                "population_rate": pop, "y_perm": y_perm, "pairs": pairs,
                "calibrations": cal_pair, "chunk_size": 8,
            }


def _power_tasks(cells: list[tuple[str, int, int]], reps: int, n_perm: int,
                 specs: dict[str, Any], fits: dict[str, Any], calibrations: dict[str, Any]
                 ) -> Iterator[dict[str, Any]]:
    for cell in cells:
        cohort, days, n_patients = cell
        ci = _cell_index(cell)
        spec, fit = specs[cohort], fits[cohort]
        pop = float(spec["rate"])
        for test_no, test_name in enumerate(("A_main_power", "A_daily_rho0_size", "B_daily_rho0_size")):
            for rep in range(reps):
                rng = _rng(20 + test_no, ci, rep)
                b = _recentered_levels(rng, fit, spec, n_patients)
                x, y = _draw_outcomes(rng, b, days)
                noise = rng.standard_normal(x.shape)
                y_perm = _make_shuffles(rng, y, n_perm)
                calibration = (calibrations[f"{cohort}|{days}|0.6"] if test_name == "A_main_power"
                               else calibrations[f"{cohort}|{days}|0.0"])
                yield {
                    "cell": cell, "test": test_name, "rep": rep, "b": b, "x": x, "y": y,
                    "noise": noise, "y_perm": y_perm,
                    "patient_rate": who_when.normal_expit_expectation(b, X_SD * X_SD),
                    "population_rate": pop, "calibration": calibration, "rho_a": 0.6,
                    "tie_score": rng.random(x.shape) if test_name == "B_daily_rho0_size" else None,
                    "chunk_size": 8,
                }


def _long_benchmark(specs: dict[str, Any], fits: dict[str, Any], calibrations: dict[str, Any],
                    pairs: list[dict[str, Any]], cells: list[tuple[str, int, int]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    selected_cohorts = list(dict.fromkeys(cell[0] for cell in cells))
    for ci, cohort in enumerate(COHORTS):
        if cohort not in selected_cohorts:
            continue
        spec, fit = specs[cohort], fits[cohort]
        rng = _rng(90, ci)
        b = _recentered_levels(rng, fit, spec, LONG_PATIENTS)
        x, y = _draw_outcomes(rng, b, LONG_DAYS)
        noise = rng.standard_normal(x.shape)
        pop = float(spec["rate"])
        patient_rate = who_when.normal_expit_expectation(b, X_SD * X_SD)
        for pair in pairs:
            rho_a, rho_b = pair["rho_a"], pair["rho_b"]
            d_a, d_b = _daily_proxy(x, noise, rho_a), _daily_proxy(x, noise, rho_b)
            forecast_b = _b_forecast(b, d_b, rho_b)
            score_b = _auc_matrix(y, forecast_b)[0]
            selected_days = sorted({cell[1] for cell in cells if cell[0] == cohort})
            for days in selected_days:
                cal = calibrations[f"{cohort}|{days}|{rho_a:.1f}"]
                forecast_a = _a_forecast(y, d_a, pop, cal)
                score_a = _auc_matrix(y, forecast_a)[0]
                results[f"{cohort}|{days}|{pair['name']}"] = {
                    "record_days_calibration": days,
                    "long_record_days": LONG_DAYS,
                    "long_record_patients": LONG_PATIENTS,
                    "a_mean_patient_auc": _finite_mean(score_a),
                    "b_mean_patient_auc": _finite_mean(score_b),
                    "a_patient_auc_values": [float(v) if np.isfinite(v) else None for v in score_a],
                    "b_patient_auc_values": [float(v) if np.isfinite(v) else None for v in score_b],
                    "independent_seed_stream": [BASE_SEED, 90, ci],
                }
    return results


def _ranking_rows(main_results: list[dict[str, Any]], cells: list[tuple[str, int, int]],
                  reps: int, pairs: list[dict[str, Any]], long_benchmark: dict[str, Any],
                  b_chance: str = "shuffle", tailored_only: bool = False
                  ) -> list[dict[str, Any]]:
    score_definitions = {
        "pooled": {
            "auc": ("pooled_a", "pooled_b", "auc"),
            "average_precision": ("pooled_a", "pooled_b", "average_precision"),
            "brier_skill_updated_rate": ("pooled_a", "pooled_b", "brier_skill_updated_rate"),
            "brier_skill_oracle": ("pooled_a", "pooled_b", "brier_skill_oracle"),
            "ppv": ("pooled_a", "pooled_b", "ppv"),
            "net_benefit": ("pooled_a", "pooled_b", "net_benefit"),
        },
        "naive": {
            "auc": ("naive_a", "naive_b", "auc"),
            "average_precision": ("naive_a", "naive_b", "average_precision"),
            "brier_skill_population_rate": ("naive_a", "naive_b", "brier_skill_population_rate"),
            "ppv": ("naive_a", "naive_b", "ppv"),
            "net_benefit": ("naive_a", "naive_b", "net_benefit"),
        },
        "tailored_excess": {
            "auc": ("tailored_a_actual", "tailored_b_actual", "auc"),
            "average_precision": ("tailored_a_actual", "tailored_b_actual", "average_precision"),
            "brier_skill_updated_rate": ("tailored_a_actual", "tailored_b_actual", "brier_skill_updated_rate"),
            "brier_skill_oracle": ("tailored_a_actual", "tailored_b_actual", "brier_skill_oracle"),
            "ppv": ("tailored_a_actual", "tailored_b_actual", "ppv"),
            "net_benefit": ("tailored_a_actual", "tailored_b_actual", "net_benefit"),
        },
        "tailored (B against its level-only forecast)": {
            "auc": ("tailored_a_actual", "tailored_b_actual_legacy", "auc"),
            "average_precision": ("tailored_a_actual", "tailored_b_actual_legacy", "average_precision"),
            "brier_skill_updated_rate": ("tailored_a_actual", "tailored_b_actual_legacy", "brier_skill_updated_rate"),
            "brier_skill_oracle": ("tailored_a_actual", "tailored_b_actual_legacy", "brier_skill_oracle"),
            "ppv": ("tailored_a_actual", "tailored_b_actual_legacy", "ppv"),
            "net_benefit": ("tailored_a_actual", "tailored_b_actual_legacy", "net_benefit"),
        },
    }
    if tailored_only:
        score_definitions = {key: value for key, value in score_definitions.items()
                             if key.startswith("tailored")}
    grouped: dict[tuple[Any, ...], list[tuple[float, float]]] = {}
    for result in main_results:
        cell = tuple(result["cell"])
        for pair in pairs:
            scores = result["pairs"][pair["name"]]
            for scoring, definitions in score_definitions.items():
                for metric, (a_key, b_key, score_key) in definitions.items():
                    if scoring.startswith("tailored"):
                        a = scores[a_key][score_key] - scores["tailored_a_chance"][score_key]
                        chance_key = ("tailored_b_chance"
                                      if scoring == "tailored (B against its level-only forecast)"
                                      or b_chance == "levelonly"
                                      else "tailored_b_chance_shuffle")
                        b = scores[b_key][score_key] - scores[chance_key][score_key]
                    else:
                        a, b = scores[a_key][score_key], scores[b_key][score_key]
                    grouped.setdefault((*cell, pair["name"], scoring, metric), []).append((a, b))

    rows: list[dict[str, Any]] = []
    for cell in cells:
        cohort, days, n_patients = cell
        for pair in pairs:
            truth = long_benchmark[f"{cohort}|{days}|{pair['name']}"]
            for key, values in grouped.items():
                if tuple(key[:3]) != tuple(cell) or key[3] != pair["name"]:
                    continue
                finite = [(a, b) for a, b in values if np.isfinite(a) and np.isfinite(b)]
                wins = sum(a > b for a, b in finite)
                rows.append({
                    "cohort": cohort, "patient_spread": "low" if cohort == "REPLACE-BG" else "high",
                    "record_days": days, "patients": n_patients, "pair": pair["name"],
                    "scoring": key[4], "metric": key[5], "replications": len(values),
                    "valid_replications": len(finite),
                    "share_a_ranks_above_b": (wins / len(finite)) if finite else None,
                    "mean_a_score": float(np.mean([v[0] for v in finite])) if finite else None,
                    "mean_b_score": float(np.mean([v[1] for v in finite])) if finite else None,
                    "long_record_a_true_patient_auc": truth["a_mean_patient_auc"],
                    "long_record_b_true_patient_auc": truth["b_mean_patient_auc"],
                    "long_record_true_auc_a_minus_b": truth["a_mean_patient_auc"] - truth["b_mean_patient_auc"],
                })
    return rows


def _power_rows(power_results: list[dict[str, Any]], cells: list[tuple[str, int, int]], reps: int
                ) -> list[dict[str, Any]]:
    rows = []
    tests = ("A_main_power", "A_daily_rho0_size", "B_daily_rho0_size")
    for cell in cells:
        for test in tests:
            selected = [r for r in power_results if tuple(r["cell"]) == tuple(cell) and r["test"] == test]
            rows.append({
                "cohort": cell[0], "patient_spread": "low" if cell[0] == "REPLACE-BG" else "high",
                "record_days": cell[1], "patients": cell[2], "test": test,
                "replications": reps,
                "rejections": int(sum(row["reject"] for row in selected)),
                "rejection_rate": (float(np.mean([row["reject"] for row in selected])) if selected else None),
                "mean_observed_auc": (float(np.nanmean([row["observed_auc"] for row in selected])) if selected else None),
                "mean_permutation_p": (float(np.nanmean([row["p_value"] for row in selected])) if selected else None),
            })
    return rows


def _work_weight(cells: list[tuple[str, int, int]], reps: int, shuffles: int,
                 *, pair_count: int = 1, tests: int = 0) -> float:
    total = 0.0
    for _, days, patients in cells:
        sort_cost = days * math.log2(max(2, days))
        total += patients * sort_cost * (reps * shuffles * pair_count + reps * shuffles * tests)
    return total


def _runtime_estimate(main_seconds: float, power_seconds: float,
                      quick_cell: tuple[str, int, int]) -> dict[str, float]:
    full_cells = [(cohort, days, patients) for cohort in COHORTS for days in LENGTHS for patients in PATIENT_COUNTS]
    one_cell = [quick_cell]
    main_q = _work_weight(one_cell, 20, 50, pair_count=1)
    main_full = _work_weight(full_cells, 500, 50, pair_count=1)
    power_q = _work_weight(one_cell, 20, 50, tests=3)
    power_full = _work_weight(full_cells, 200, 200, tests=3)
    estimated_main = main_seconds * main_full / max(main_q, 1.0)
    estimated_power = power_seconds * power_full / max(power_q, 1.0)
    estimated_total = estimated_main + estimated_power
    return {"full_base_main_hours": estimated_main / 3600,
            "full_base_power_hours": estimated_power / 3600,
            "full_base_total_hours": estimated_total / 3600,
            "full_with_extra_pair_hours_approx": (estimated_main * 2.0 + estimated_power) / 3600}


def _tailored_runtime_estimate(tailored_seconds: float, quick_cell: tuple[str, int, int],
                               shuffles: int, pair_count: int) -> float:
    full_cells = [(cohort, days, patients) for cohort in COHORTS
                  for days in LENGTHS for patients in PATIENT_COUNTS]
    quick_work = _work_weight([quick_cell], 20, shuffles, pair_count=pair_count)
    full_work = _work_weight(full_cells, 500, shuffles, pair_count=pair_count)
    return tailored_seconds * full_work / max(quick_work, 1.0) / 3600.0


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Tailored ranking simulation",
        "",
        f"Run profile: **{payload['run']['profile']}**; seed base **{BASE_SEED}**; workers **{payload['run']['workers']}**.",
        "",
        "## Design and interpretation",
        "",
        "- Latent outcomes use `logit P(Y_kt=1) = level + b_k + x_kt`, daily SD 2, with REPLACE-BG and CTN-0094 cohort fits from `cohort_curves.py`; each fit is recentered to its cohort population rate.",
        "- Scored days start at day 1 with no separate run-in. A's first score uses the population rate, then uses only earlier outcomes with four population-rate pseudo-observations; its rate feature enters on the logit scale.",
        "- A's two-component logistic recalibration is fitted on an independent 200,000-row simulated cohort for each spread, record length, and A daily correlation. B conditions exactly on b and its daily proxy using the 40-node Gauss-Hermite helper from `who_when_sim.py`.",
        f"- Pooled scores are raw day-level values. Naive scores are equal-patient raw values following the selected `naive_metrics` definitions. Tailored scores use the selected per-patient `nine_metrics` definitions. The primary tailored ranking uses B's `{payload['run'].get('b_chance', 'shuffle')}` chance level; the additional `tailored (B against its level-only forecast)` row preserves the former B baseline.",
        "- For shuffled chance, A is recalibrated once and held fixed while its history feature is rebuilt for each shuffle. B's fixed forecast is held unchanged while the same shuffled outcome arrays are scored. Outcome-based yardsticks, including the running rate for Brier skill and the own-rate references for PPV and net benefit, are recomputed from each shuffled outcome.",
        "- Tailored Brier skill is shown against both the running rate based on earlier outcomes and the oracle patient risk integrated over daily x. Naive Brier skill uses the population rate.",
        "- Net benefit uses the population-rate threshold; tailored per-patient net benefit follows `nine_metrics`' excess over the patient's own-rate policy. PPV follows its own-rate excess definition.",
        "- Long-record true patient ability is estimated from 200 independent patients with 5,000 scored days each, separately for each record-length calibration of A. AUC values average over patients with both outcomes.",
        "- The B daily-rho-zero size diagnostic uses a fixed independent random tie-breaker for within-patient AUC. B's exact rho-zero forecast is constant within patient, so conventional tie-aware AUC is identically 0.5 and would make a size check degenerate.",
        "- Random arrays, including shuffles and tie-breakers, are generated in the parent before tasks are sent to workers. Replicate streams are keyed by cell and replicate, so changing `--workers` does not change results.",
        "",
        "## Ranking results",
        "",
        "Share of valid replications where A scores above B; the long-record AUC columns give the simulated reference ability.",
        "",
        "| Spread | Days | Patients | Pair | Scoring | Metric | A above B | True A AUC | True B AUC |",
        "|---|---:|---:|---|---|---|---:|---:|---:|",
    ]
    for row in payload["ranking"]:
        share = "NA" if row["share_a_ranks_above_b"] is None else f"{row['share_a_ranks_above_b']:.3f}"
        a_auc, b_auc = row["long_record_a_true_patient_auc"], row["long_record_b_true_patient_auc"]
        lines.append(f"| {row['patient_spread']} | {row['record_days']} | {row['patients']} | {row['pair']} | {row['scoring']} | {row['metric']} | {share} | {a_auc:.3f} | {b_auc:.3f} |")
    lines += ["", "## Permutation power and size checks", "",
              "One-sided model-fixed within-patient AUC permutation test, alpha 0.05.", "",
              "| Spread | Days | Patients | Test | Replications | Rejections | Rate | Mean observed AUC |",
              "|---|---:|---:|---|---:|---:|---:|---:|"]
    for row in payload["power"]:
        rate = "NA" if row["rejection_rate"] is None else f"{row['rejection_rate']:.3f}"
        auc = "NA" if row["mean_observed_auc"] is None else f"{row['mean_observed_auc']:.3f}"
        lines.append(f"| {row['patient_spread']} | {row['record_days']} | {row['patients']} | {row['test']} | {row['replications']} | {row['rejections']} | {rate} | {auc} |")
    lines += ["", "## Runtime", ""]
    if payload["run"].get("elapsed_seconds") is not None:
        label = "Quick-profile" if payload["run"]["profile"] == "quick" else "Full-profile"
        lines.append(f"{label} elapsed time: **{payload['run']['elapsed_seconds']:.1f} seconds**.")
    estimate = payload["run"].get("full_runtime_estimate_hours")
    if estimate:
        lines.append(f"Extrapolated full run without the optional pair: **{estimate['full_base_total_hours']:.1f} hours** (main ranking {estimate['full_base_main_hours']:.1f} h; power and size checks {estimate['full_base_power_hours']:.1f} h). This scales measured quick work by patients × days × sort cost and requested replication/permutation counts; it is an estimate, not a benchmark of all 12 cells.")
    tailored_estimate = payload["run"].get("tailored_only_full_runtime_estimate_hours")
    if tailored_estimate is not None:
        lines.append(f"Estimated tailored-only full rerun: **{tailored_estimate:.2f} hours** (scaled from the measured tailored-only run).")
    tailored_elapsed = payload["run"].get("tailored_only_last_rerun_seconds")
    if tailored_elapsed is not None:
        lines.append(f"Last tailored-only replay: **{tailored_elapsed:.1f} seconds** with {payload['run'].get('tailored_only_last_workers', 'unknown')} workers.")
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    cells = args.cells
    pairs = [dict(BASE_PAIR)]
    if args.extra_pair:
        pairs.append(dict(EXTRA_PAIR))
    specs, fits, calibrations = _context(cells, bool(args.extra_pair))

    main_start = time.perf_counter()
    main_results = _parent_generated_map(
        _main_worker,
        _main_tasks(cells, args.reps, args.permutations, specs, fits, calibrations, pairs),
        args.workers,
    )
    main_results.sort(key=lambda row: (_cell_index(tuple(row["cell"])), row["rep"]))
    main_seconds = time.perf_counter() - main_start

    power_start = time.perf_counter()
    power_results = _parent_generated_map(
        _power_worker,
        _power_tasks(cells, args.power_reps, args.power_permutations, specs, fits, calibrations),
        args.workers,
    )
    test_order = {"A_main_power": 0, "A_daily_rho0_size": 1, "B_daily_rho0_size": 2}
    power_results.sort(key=lambda row: (_cell_index(tuple(row["cell"])), test_order[row["test"]], row["rep"]))
    power_seconds = time.perf_counter() - power_start

    long_benchmark = _long_benchmark(specs, fits, calibrations, pairs, cells)
    ranking = _ranking_rows(main_results, cells, args.reps, pairs, long_benchmark,
                            b_chance=args.b_chance)
    power = _power_rows(power_results, cells, args.power_reps)
    elapsed = time.perf_counter() - started
    profile = "quick" if args.quick else "full"
    runtime_estimate = (_runtime_estimate(main_seconds, power_seconds, cells[0]) if args.quick else None)
    run_info = {
        "profile": profile, "seed": BASE_SEED, "workers": args.workers,
        "cells": [{"cohort": c, "record_days": n, "patients": k} for c, n, k in cells],
        "ranking_replications_per_cell": args.reps,
        "chance_shuffles_per_replication": args.permutations,
        "power_replications_per_cell_and_test": args.power_reps,
        "power_shuffles": args.power_permutations,
        "extra_pair_included": bool(args.extra_pair), "elapsed_seconds": elapsed,
        "b_chance": args.b_chance,
        "main_seconds": main_seconds, "power_seconds": power_seconds,
        "full_runtime_estimate_hours": (runtime_estimate if runtime_estimate is not None
                                        else getattr(args, "pilot_runtime_estimate", None)),
        "default_extra_pair_rule": "For a full run, include the extra pair by default only if an internal 1-cell quick pilot extrapolates the base 12-cell run to under 3 hours; use --extra-pair to force it.",
        "calibrations": calibrations,
        "cohort_fits": fits,
    }
    return _json_safe({"run": run_info, "ranking": ranking, "power": power,
                       "long_record_benchmark": long_benchmark})


def _tailored_only_payload(args: argparse.Namespace, source_path: Path) -> dict[str, Any]:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    run_info = payload["run"]
    cells = [(cell["cohort"], int(cell["record_days"]), int(cell["patients"]))
             for cell in run_info["cells"]]
    pairs = [dict(BASE_PAIR)]
    if run_info.get("extra_pair_included", False):
        pairs.append(dict(EXTRA_PAIR))
    specs = {name: cohort_sim.COHORTS[name] for name in COHORTS}
    fits = run_info["cohort_fits"]
    calibrations = run_info["calibrations"]
    reps = int(run_info["ranking_replications_per_cell"])
    shuffles = int(run_info["chance_shuffles_per_replication"])

    started = time.perf_counter()
    tasks = _main_tasks(cells, reps, shuffles, specs, fits, calibrations, pairs)
    tailored_tasks = ({**task, "tailored_only": True} for task in tasks)
    results = _parent_generated_map(_main_worker, tailored_tasks, args.workers)
    results.sort(key=lambda row: (_cell_index(tuple(row["cell"])), row["rep"]))
    new_rows = _ranking_rows(
        results, cells, reps, pairs, payload["long_record_benchmark"],
        b_chance=args.b_chance, tailored_only=True,
    )
    elapsed = time.perf_counter() - started

    replaced_scoring = {"tailored_excess", "tailored (B against its level-only forecast)"}
    ranking = [row for row in payload["ranking"] if row["scoring"] not in replaced_scoring]
    ranking.extend(new_rows)
    scoring_order = {"pooled": 0, "naive": 1, "tailored_excess": 2,
                     "tailored (B against its level-only forecast)": 3}
    metric_order = {"auc": 0, "average_precision": 1, "brier_skill_updated_rate": 2,
                    "brier_skill_oracle": 3, "brier_skill_population_rate": 4,
                    "ppv": 5, "net_benefit": 6}
    pair_order = {pair["name"]: index for index, pair in enumerate(pairs)}
    ranking.sort(key=lambda row: (
        _cell_index((row["cohort"], row["record_days"], row["patients"])),
        pair_order.get(row["pair"], 99), scoring_order.get(row["scoring"], 99),
        metric_order.get(row["metric"], 99),
    ))
    payload["ranking"] = ranking
    run_info["b_chance"] = args.b_chance
    run_info["tailored_only_last_rerun_seconds"] = elapsed
    run_info["tailored_only_last_workers"] = args.workers
    if run_info.get("profile") == "quick":
        run_info["tailored_only_full_runtime_estimate_hours"] = _tailored_runtime_estimate(
            elapsed, cells[0], shuffles, len(pairs))
    return payload


def main() -> None:
    args = parse_args()
    out_dir = ROOT / "open_data" / "results" / "v10"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.tailored_only:
        source_path = args.input_json or (out_dir / "ranking_sim.json")
        payload = _tailored_only_payload(args, source_path)
        base_path = source_path
        stem = source_path.stem
    else:
        base_path = out_dir / "ranking_sim.json"
        stem = "ranking_sim"
    # Full mode's default optional-pair decision is based on an actual one-cell pilot.
    # A user-supplied --extra-pair/--no-extra-pair always overrides this rule.
    if not args.tailored_only and not args.quick and args.extra_pair is None:
        pilot = argparse.Namespace(**vars(args))
        pilot.quick = True
        pilot.cells = [(COHORTS[0], LENGTHS[0], PATIENT_COUNTS[0])]
        pilot.reps, pilot.permutations = 20, 50
        pilot.power_reps, pilot.power_permutations = 20, 50
        pilot.extra_pair = False
        pilot_result = run(pilot)
        estimate = pilot_result["run"]["full_runtime_estimate_hours"]
        args.extra_pair = bool(estimate and estimate["full_base_total_hours"] < 3.0)
        args.pilot_runtime_estimate = estimate
    if not args.tailored_only:
        if args.extra_pair is None:
            args.extra_pair = False
        payload = run(args)
    if args.output_suffix and any(sep in args.output_suffix for sep in ("/", "\\")):
        raise ValueError("--output-suffix must not contain path separators")
    output_stem = stem + args.output_suffix
    json_path = base_path.with_name(output_stem + ".json")
    md_path = base_path.with_name(output_stem + ".md")
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    print(f"Elapsed: {payload['run']['elapsed_seconds']:.1f}s")
    if payload["run"].get("full_runtime_estimate_hours"):
        print(f"Estimated full base run: {payload['run']['full_runtime_estimate_hours']['full_base_total_hours']:.2f}h")
    if payload["run"].get("tailored_only_last_rerun_seconds") is not None:
        print(f"Tailored-only elapsed: {payload['run']['tailored_only_last_rerun_seconds']:.1f}s")
    if payload["run"].get("tailored_only_full_runtime_estimate_hours") is not None:
        print(f"Estimated tailored-only full rerun: {payload['run']['tailored_only_full_runtime_estimate_hours']:.2f}h")


if __name__ == "__main__":
    main()
