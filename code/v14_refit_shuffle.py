"""Shuffled baseline of a model fitted on the scored record itself (v14, B3b).

One record has T independent days with event risk 0.25, so no forecast has
skill. The pipeline is fitted on the scored record by rotating folds within
the patient: the days are split into five folds (day index modulo 5); for
each fold a logistic regression is fitted on the other four folds and
predicts the held-out days. Inputs: a noise covariate x_t (standard normal,
not computed from outcomes) and the patient's event rate among the training
days (for a training day: among the other training days). The two slopes
carry a ridge penalty of 1; the intercept is not penalised.

The per-patient AUC of the held-out predictions is compared with its
shuffled baseline in two ways:

    frozen   the coefficients fitted on the observed record are kept; only
             the outcome-derived input (the training rate) is rebuilt from
             the shuffled outcomes
    refit    the whole pipeline is run again on every shuffled record

Excess = observed AUC minus the mean AUC over B shuffles; one-sided p value
(1 + #{shuffled >= observed}) / (B + 1).

Output: results/v14/B3b_refit.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from v14_common import auc_rows, mean_se  # noqa: E402

SEED = 20261012
RECORDS = 2000
SHUFFLES = 199
RISK = 0.25
FOLDS = 5
RIDGE = np.array([0.0, 1.0, 1.0])
LENGTHS = (60, 120)


def fit_logistic(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ridge logistic regression for a batch: design (B, n, 3), y (B, n) -> (B, 3)."""
    prevalence = np.clip(y.mean(axis=1), 0.02, 0.98)
    beta = np.zeros((design.shape[0], 3))
    beta[:, 0] = np.log(prevalence / (1.0 - prevalence))
    for _ in range(50):
        prob = expit(np.clip(np.einsum("bnj,bj->bn", design, beta), -30.0, 30.0))
        weight = np.maximum(prob * (1.0 - prob), 1e-6)
        gradient = np.einsum("bnj,bn->bj", design, y - prob) - RIDGE * beta
        hessian = np.einsum("bnj,bn,bnk->bjk", design, weight, design) + np.diag(RIDGE + 1e-7)
        step = np.linalg.solve(hessian, gradient[:, :, None])[:, :, 0]
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def pipeline(y: np.ndarray, x: np.ndarray, fold: np.ndarray, frozen=None):
    """Held-out predictions for a batch of records y (B, T) that share the input x (T,)."""
    y = y.astype(float)
    B = y.shape[0]
    prediction = np.empty_like(y)
    coefficients = []
    for k in range(FOLDS):
        test, train = fold == k, fold != k
        y_train = y[:, train]
        n_train = y_train.shape[1]
        rate_others = (y_train.sum(axis=1, keepdims=True) - y_train) / (n_train - 1)
        if frozen is None:
            design = np.stack([np.ones((B, n_train)), np.tile(x[train], (B, 1)), rate_others], axis=2)
            beta = fit_logistic(design, y_train)
        else:
            beta = np.broadcast_to(frozen[k], (B, 3))
        rate_train = y_train.mean(axis=1, keepdims=True)
        linear = beta[:, [0]] + beta[:, [1]] * x[test][None, :] + beta[:, [2]] * rate_train
        prediction[:, test] = expit(linear)
        coefficients.append(beta)
    return prediction, coefficients


def run(T: int, records: int, seed_index: int) -> list[dict]:
    rng = np.random.default_rng([SEED, seed_index])
    fold = np.arange(T) % FOLDS
    observed_auc, results = [], {"frozen": [], "refit": []}
    for _ in range(records):
        y = (rng.random(T) < RISK).astype(np.int8)
        x = rng.standard_normal(T)
        if y.sum() in (0, T):
            continue
        prediction, coefficients = pipeline(y[None, :], x, fold)
        observed = auc_rows(y[None, :], prediction)[0]
        ys = rng.permuted(np.tile(y, (SHUFFLES, 1)), axis=1)
        null = {
            "frozen": auc_rows(ys, pipeline(ys, x, fold, frozen=coefficients)[0]),
            "refit": auc_rows(ys, pipeline(ys, x, fold)[0]),
        }
        observed_auc.append(observed)
        for key, values in null.items():
            p = (1 + np.count_nonzero(values >= observed - 1e-12)) / (SHUFFLES + 1)
            results[key].append((observed - values.mean(), values.mean(), p <= 0.05))
    rows = []
    for key, values in results.items():
        values = np.asarray(values, dtype=float)
        excess, excess_se = mean_se(values[:, 0])
        rows.append({
            "T": T, "shuffle_scope": key, "records": len(values), "shuffles": SHUFFLES,
            "seed": f"{SEED}/{seed_index}",
            "auc_observed_mean": float(np.mean(observed_auc)),
            "auc_shuffled_mean": float(values[:, 1].mean()),
            "excess_mean": excess, "excess_se": excess_se,
            "rejection_rate": float(values[:, 2].mean()),
        })
    return rows


def main() -> None:
    records = int(sys.argv[1]) if len(sys.argv) > 1 else RECORDS
    rows = []
    for index, T in enumerate(LENGTHS):
        rows += run(T, records, index)
        print(pd.DataFrame(rows).round(4).to_string(index=False), flush=True)
    out_dir = ROOT / "results" / "v14"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "B3b_refit.csv", index=False)


if __name__ == "__main__":
    main()
