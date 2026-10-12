"""Reproduce a constant-rate, moving-window within-person AUC check.

The cited paper's GitHub repository is public, but its R source files could not
be fetched in this environment. The design assumptions used here are listed
in open_data/results/v10/lj_baseline_check.md. This script is read-only with
respect to downloaded code and writes no output files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "wheneval" / "src"))
from wheneval.metrics import auc  # noqa: E402


N_PERSONS = 150
N_DAYS = 200
MU = 0.10
SIGMA = 0.25
WINDOW = 30
N_REPLICATIONS = 200
SEED = 20261011


def beta_parameters(mean: float, sd: float) -> tuple[float, float]:
    """Beta parameters from the requested mean and SD of p."""
    variance = sd * sd
    concentration = mean * (1.0 - mean) / variance - 1.0
    alpha = mean * concentration
    beta = (1.0 - mean) * concentration
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("requested mean and SD are not feasible for a Beta distribution")
    return alpha, beta


def logistic_normal_center(mean_probability: float, sd_logit: float) -> float:
    """Calibrate the normal location so E[logistic(X)] equals mean_probability."""
    nodes, weights = np.polynomial.hermite.hermgauss(80)
    z = np.sqrt(2.0) * nodes
    weights = weights / np.sqrt(np.pi)

    def expected_probability(center: float) -> float:
        x = center + sd_logit * z
        p = np.empty_like(x)
        positive = x >= 0
        p[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
        exp_x = np.exp(x[~positive])
        p[~positive] = exp_x / (1.0 + exp_x)
        return float(np.dot(weights, p))

    lo, hi = -40.0, 40.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if expected_probability(mid) < mean_probability:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def make_forecasts(y: np.ndarray, window: int, *, expanding: bool = False,
                   include_current: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """One-step walk-forward forecasts after a fixed 30-day run-in.

    Rows are people and columns are days. Test indices are window..T-1.
    The leaking version uses the last `window` outcomes ending on the scored
    day; the past-only version uses the `window` outcomes immediately before it.
    """
    n, t = y.shape
    test = np.arange(window, t)
    score = np.empty((n, len(test)), dtype=float)
    for j, day in enumerate(test):
        if expanding:
            history = y[:, : day + int(include_current)]
        elif include_current:
            history = y[:, day - window + 1 : day + 1]
        else:
            history = y[:, day - window : day]
        score[:, j] = history.mean(axis=1)
    return y[:, test], score


def person_auc_mean(y_test: np.ndarray, score: np.ndarray, *,
                    drop_ties: bool = False,
                    exclude_constant_forecasts: bool = False,
                    auto_direction: bool = False) -> float:
    """Mean within-person AUC, excluding people without both test classes."""
    values: list[float] = []
    for outcomes, forecasts in zip(y_test, score):
        if outcomes.min() == outcomes.max():
            continue
        if exclude_constant_forecasts and np.ptp(forecasts) == 0:
            continue
        if not drop_ties:
            value = auc(outcomes, forecasts)
            if auto_direction and np.median(forecasts[outcomes == 0]) > np.median(
                forecasts[outcomes == 1]
            ):
                # Diagnostic for ROC implementations that choose orientation
                # from the scored labels (for example, an automatic direction).
                value = auc(outcomes, -forecasts)
        else:
            positive = forecasts[outcomes == 1]
            negative = forecasts[outcomes == 0]
            comparisons = positive[:, None] - negative[None, :]
            untied = comparisons != 0.0
            if not untied.any():
                continue
            value = float(np.count_nonzero(comparisons[untied] > 0.0) / untied.sum())
        if np.isfinite(value):
            values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def simulation() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    alpha, beta = beta_parameters(MU, SIGMA)
    logit_center = logistic_normal_center(MU, SIGMA)

    scenario_names = [
        "Beta(p), moving past 30, half ties",
        "Beta(p), expanding past mean, half ties",
        "Beta(p), moving past 30, ties dropped",
        "Beta(p), moving past 30, constant-score persons excluded",
        "Beta(p), moving 30 including current day (leak)",
        "Logit-normal (SD=.25), moving past 30, half ties",
        "Beta(p), moving past 30, test-median direction (diagnostic)",
    ]
    draws: dict[str, list[float]] = {name: [] for name in scenario_names}

    for _ in range(N_REPLICATIONS):
        p_beta = rng.beta(alpha, beta, size=N_PERSONS)
        z = rng.normal(size=N_PERSONS)
        p_logit = 1.0 / (1.0 + np.exp(-(logit_center + SIGMA * z)))
        # Shared uniforms make the scale comparison less noisy.
        uniforms = rng.random((N_PERSONS, N_DAYS))
        datasets = {
            "Beta(p)": (uniforms < p_beta[:, None]).astype(np.int8),
            "Logit-normal": (uniforms < p_logit[:, None]).astype(np.int8),
        }
        y_beta = datasets["Beta(p)"]

        y_test, score = make_forecasts(y_beta, WINDOW)
        draws[scenario_names[0]].append(person_auc_mean(y_test, score))

        y_test, score = make_forecasts(y_beta, WINDOW, expanding=True)
        draws[scenario_names[1]].append(person_auc_mean(y_test, score))

        y_test, score = make_forecasts(y_beta, WINDOW)
        draws[scenario_names[2]].append(person_auc_mean(y_test, score, drop_ties=True))

        y_test, score = make_forecasts(y_beta, WINDOW)
        draws[scenario_names[3]].append(
            person_auc_mean(y_test, score, exclude_constant_forecasts=True)
        )

        y_test, score = make_forecasts(y_beta, WINDOW, include_current=True)
        draws[scenario_names[4]].append(person_auc_mean(y_test, score))

        y_logit = datasets["Logit-normal"]
        y_test, score = make_forecasts(y_logit, WINDOW)
        draws[scenario_names[5]].append(person_auc_mean(y_test, score))

        y_test, score = make_forecasts(y_beta, WINDOW)
        draws[scenario_names[6]].append(
            person_auc_mean(y_test, score, auto_direction=True)
        )

    rows = []
    for variant, replication_means in draws.items():
        values = np.asarray(replication_means, dtype=float)
        finite = values[np.isfinite(values)]
        rows.append({
            "Variant": variant,
            "Mean within-person AUC": float(np.mean(finite)),
            "MC SD": float(np.std(finite, ddof=1)),
            "Finite replications": int(len(finite)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    result = simulation()
    print("| Variant | Mean within-person AUC | MC SD | Finite replications |")
    print("|---|---:|---:|---:|")
    for row in result.itertuples(index=False):
        print(
            f"| {row.Variant} | {row._1:.4f} | {row._2:.4f} | {row._3} |"
        )


if __name__ == "__main__":
    main()
