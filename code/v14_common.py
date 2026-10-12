"""Helpers shared by the v14 simulation scripts (B3)."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auc_rows(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    """AUC of every row (ties count one half); nan for rows with one outcome class."""
    y = np.asarray(y, dtype=float)
    ranks = rankdata(score, axis=1)
    n1 = y.sum(axis=1)
    n0 = y.shape[1] - n1
    with np.errstate(invalid="ignore", divide="ignore"):
        value = ((ranks * y).sum(axis=1) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    value[(n1 == 0) | (n0 == 0)] = np.nan
    return value


def auc_flat(y: np.ndarray, score: np.ndarray) -> float:
    return float(auc_rows(np.ravel(y)[None, :], np.ravel(score)[None, :])[0])


def mean_se(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))
