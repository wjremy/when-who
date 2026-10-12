"""Shuffled baseline when all patients share a time course (v14, B3c).

Every record has T = 100 independent days whose event risk falls linearly
from 0.50 on the first day to 0.10 on the last, the same course for every
patient. Four forecasts; the first three know nothing about the individual
patient's days beyond that course:

    study_day     minus the day index
    rate_so_far   share of event days among the earlier days (0.30 on day 1),
                  rebuilt from the shuffled outcomes in every shuffle
    constant      0.30 on every day
    daily_signal  a daily input x_t (standard normal, fixed in the shuffles)

For each record: per-patient AUC, its excess over the mean AUC of B shuffles
and the one-sided p value (1 + #{shuffled >= observed}) / (B + 1). Shuffles:

    full          all scored days of the record
    block_L       within consecutive blocks of L days (events per block kept)

Two further courses: a constant risk of 0.30 (size of each test), and the
falling course with a real daily signal, logit risk = logit(course) + x_t
(what each shuffle keeps of a forecast that does know the patient's days).

Output: results/v14/B3c_shared_course.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from scipy.special import expit, logit  # noqa: E402
from v14_common import auc_rows, mean_se  # noqa: E402

SEED = 20261012
RECORDS = 4000
SHUFFLES = 199
T = 100
BLOCKS = (None, 20, 10)
PRIOR = 0.30


def rate_so_far(y: np.ndarray) -> np.ndarray:
    y = y.astype(float)
    f = np.empty_like(y)
    f[:, 0] = PRIOR
    f[:, 1:] = np.cumsum(y, axis=1)[:, :-1] / np.arange(1, y.shape[1])
    return f


FORECASTS = {
    "study_day": lambda y, x: np.tile(-np.arange(y.shape[1], dtype=float), (y.shape[0], 1)),
    "rate_so_far": lambda y, x: rate_so_far(y),
    "constant": lambda y, x: np.full(y.shape, PRIOR),
    "daily_signal": lambda y, x: np.tile(x, (y.shape[0], 1)),
}


def shuffles(rng, y: np.ndarray, block) -> np.ndarray:
    tiled = np.tile(y, (SHUFFLES, 1))
    if block is None:
        return rng.permuted(tiled, axis=1)
    return rng.permuted(tiled.reshape(SHUFFLES, T // block, block), axis=2).reshape(SHUFFLES, T)


def run(course: str, risk: np.ndarray, gamma: float, records: int, seed_index: int) -> list[dict]:
    rng = np.random.default_rng([SEED, seed_index])
    store = {(f, b): [] for f in FORECASTS for b in BLOCKS}
    raw = {f: [] for f in FORECASTS}
    for _ in range(records):
        x = rng.standard_normal(T)
        y = (rng.random(T) < expit(logit(risk) + gamma * x)).astype(np.int8)
        if y.sum() in (0, T):
            continue
        for block in BLOCKS:
            ys = shuffles(rng, y, block)
            for name, build in FORECASTS.items():
                observed = auc_rows(y[None, :], build(y[None, :], x))[0]
                null = auc_rows(ys, build(ys, x))
                p = (1 + np.count_nonzero(null >= observed - 1e-12)) / (SHUFFLES + 1)
                store[(name, block)].append((observed - null.mean(), p <= 0.05))
                if block is None:
                    raw[name].append(observed)
    rows = []
    for (name, block), values in store.items():
        values = np.asarray(values, dtype=float)
        excess, excess_se = mean_se(values[:, 0])
        rows.append({
            "course": course, "daily_signal_coefficient": gamma, "forecast": name,
            "shuffle": "full" if block is None else f"block_{block}",
            "records": len(values), "shuffles": SHUFFLES, "seed": f"{SEED}/{seed_index}",
            "auc_mean": float(np.mean(raw[name])),
            "excess_mean": excess, "excess_se": excess_se,
            "rejection_rate": float(values[:, 1].mean()),
        })
    return rows


def main() -> None:
    records = int(sys.argv[1]) if len(sys.argv) > 1 else RECORDS
    falling = np.linspace(0.50, 0.10, T)
    rows = run("falling_0.50_to_0.10", falling, 0.0, records, 0)
    rows += run("constant_0.30", np.full(T, 0.30), 0.0, records, 1)
    rows += run("falling_with_daily_signal", falling, 1.0, records, 2)
    frame = pd.DataFrame(rows)
    out_dir = ROOT / "results" / "v14"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "B3c_shared_course.csv", index=False)
    print(frame.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
