"""Who and when in the pooled, per-patient and per-day AUC (v14, B3a).

K patients by T days. Patient levels p_k are Beta with mean 0.20 and SD 0.05
or 0.15. Outcomes are independent Bernoulli with

    logit P(Y_kt = 1) = logit(p_k) + beta * x_kt + d_t,

x_kt independent standard normal (a daily signal that carries no patient
level) and d_t normal with SD sigma_day, the same for all patients on day t.

Forecasts scored in every cohort:
    who     logit(p_k)                      knows only the patient's level
    when    x_kt                            knows only the daily signal
    day     d_t                             knows only the shared day effect
    both    logit(p_k) + beta * x_kt + d_t  the true risk

AUCs: pooled over all patient-days; per patient (mean over patients with
both outcomes); per day (mean over days with both outcomes).

Output: results/v14/B3a_who_when.csv
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from v14_common import auc_flat, auc_rows, mean_se  # noqa: E402

SEED = 20261012
REPLICATIONS = 400
LEVEL_MEAN = 0.20
K_VALUES = (10, 100)
T_VALUES = (30, 365)
LEVEL_SDS = (0.05, 0.15)
BETAS = (0.0, 0.5, 1.5)
DAY_SDS = (0.0, 1.0)


def beta_parameters(mean: float, sd: float) -> tuple[float, float]:
    nu = mean * (1.0 - mean) / sd ** 2 - 1.0
    return mean * nu, (1.0 - mean) * nu


def one_cohort(rng, K, T, level_sd, beta, day_sd) -> dict[str, float]:
    a, b = beta_parameters(LEVEL_MEAN, level_sd)
    level = np.clip(rng.beta(a, b, size=K), 1e-6, 1 - 1e-6)
    x = rng.standard_normal((K, T))
    d = rng.standard_normal(T) * day_sd
    who = np.repeat(logit(level)[:, None], T, axis=1)
    day = np.repeat(d[None, :], K, axis=0)
    risk = who + beta * x + day
    y = (rng.random((K, T)) < expit(risk)).astype(np.int8)
    out = {"event_rate": float(y.mean())}
    for name, f in (("who", who), ("when", x), ("day", day), ("both", risk)):
        out[f"{name}_pooled"] = auc_flat(y, f)
        out[f"{name}_patient"] = float(np.nanmean(auc_rows(y, f)))
        out[f"{name}_day"] = float(np.nanmean(auc_rows(y.T, f.T)))
    return out


def main() -> None:
    replications = int(sys.argv[1]) if len(sys.argv) > 1 else REPLICATIONS
    rows = []
    cells = list(itertools.product(K_VALUES, T_VALUES, LEVEL_SDS, BETAS, DAY_SDS))
    for index, (K, T, level_sd, beta, day_sd) in enumerate(cells):
        rng = np.random.default_rng([SEED, index])
        draws = pd.DataFrame([one_cohort(rng, K, T, level_sd, beta, day_sd) for _ in range(replications)])
        row = {"K": K, "T": T, "level_sd": level_sd, "beta": beta, "day_sd": day_sd,
               "replications": replications, "seed": f"{SEED}/{index}"}
        for column in draws.columns:
            row[column], row[column + "_se"] = mean_se(draws[column])
        rows.append(row)
        print(f"cell {index + 1}/{len(cells)} K={K} T={T} sd={level_sd} beta={beta} day={day_sd}", flush=True)
    out_dir = ROOT / "results" / "v14"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "B3a_who_when.csv", index=False)


if __name__ == "__main__":
    main()
