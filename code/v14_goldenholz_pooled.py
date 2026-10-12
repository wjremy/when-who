"""Pooled half of the Goldenholz et al. (2024) reconstruction (v14, B1).

Same simulated cohorts as code/v13_goldenholz.py (same seeds, same order of
random draws, so the per-patient AUC of results/v13/A5_goldenholz.csv is
reproduced exactly). For every cohort this script adds

- the pooled AUC of the no-skill 84-day moving average over all scored days
  of all 25 patients;
- A0 from Equation 1, A0 = 1/2 + E|p - p'| / (4 mu (1 - mu)), computed from
  the reconstructed patients' daily seizure rates, each patient weighted by
  the number of scored days (the weights of a pooled analysis);
- the pooled AUC of the forecast that states each patient's true rate.

Two further scenarios are not part of v13. One puts the reported seizure
counts in the scored days only instead of the 84 history days plus the scored
days. The other uses the widest count vector that still has the reported
minimum, median and maximum (twelve patients with 1, one with 13, twelve
with 56); the median-concentrated vector of v13 is the narrowest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "wheneval" / "src"))
from wheneval import auc  # noqa: E402
from v13_goldenholz import (  # noqa: E402
    LOOKBACK_DAYS, REPLICATIONS, SEED, count_profiles, length_profiles,
)


def a0_equation1(rates: np.ndarray, weights: np.ndarray) -> float:
    """Equation 1 for patients with rates `rates` and pooled weights `weights`."""
    w = weights / weights.sum()
    mu = float(w @ rates)
    if mu <= 0.0 or mu >= 1.0:
        return float("nan")
    mean_abs_diff = float(w @ np.abs(rates[:, None] - rates[None, :]) @ w)
    return 0.5 + mean_abs_diff / (4.0 * mu * (1.0 - mu))


def simulate(seed: int, lengths: np.ndarray, counts: np.ndarray, factor: float,
             counts_cover: str = "history_and_scored",
             replications: int = REPLICATIONS) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    out = {k: np.full(replications, np.nan) for k in
           ("per_patient", "pooled", "a0", "pooled_true_rate", "rate_mean", "rate_sd")}
    weights = lengths.astype(float)

    for b in range(replications):
        shuffled_counts = rng.permutation(counts)
        rates = np.empty(len(lengths))
        aucs, ys, fs, ps = [], [], [], []
        for k, (length, total) in enumerate(zip(lengths, shuffled_counts)):
            full_days = LOOKBACK_DAYS + int(length)
            exposure = full_days if counts_cover == "history_and_scored" else int(length)
            p = min(1.0, factor * float(total) / exposure)
            rates[k] = p
            y = rng.binomial(1, p, size=full_days).astype(np.int8)
            cum = np.concatenate(([0], np.cumsum(y, dtype=np.int64)))
            starts = np.arange(int(length), dtype=np.int64)
            f = (cum[starts + LOOKBACK_DAYS] - cum[starts]) / LOOKBACK_DAYS
            scored = y[LOOKBACK_DAYS:]
            value = auc(scored, f)
            if np.isfinite(value):
                aucs.append(value)
            ys.append(scored)
            fs.append(f)
            ps.append(np.full(int(length), p))
        y_all, f_all, p_all = np.concatenate(ys), np.concatenate(fs), np.concatenate(ps)
        out["per_patient"][b] = np.mean(aucs) if aucs else np.nan
        out["pooled"][b] = auc(y_all, f_all)
        out["pooled_true_rate"][b] = auc(y_all, p_all)
        out["a0"][b] = a0_equation1(rates, weights)
        w = weights / weights.sum()
        out["rate_mean"][b] = w @ rates
        out["rate_sd"][b] = np.sqrt(w @ (rates - w @ rates) ** 2)

    row: dict[str, float] = {}
    for key, values in out.items():
        v = values[np.isfinite(values)]
        row[f"{key}_mean"] = float(v.mean())
        row[f"{key}_q025"] = float(np.quantile(v, 0.025))
        row[f"{key}_q975"] = float(np.quantile(v, 0.975))
    return row


def main() -> None:
    lengths = {k: v[0] for k, v in length_profiles().items()}
    counts = {k: v[0] for k, v in count_profiles().items()}
    mid = lengths["primary_midpoint_lengths"]
    spread = counts["reported_range_median_spread_profile"]
    conc = counts["sensitivity_median_concentrated_profile"]
    # Widest count vector with the reported minimum 1, median 13 and maximum 56.
    widest = np.array([1] * 12 + [13] + [56] * 12, dtype=int)
    # Scenario order and seeds as in v13_goldenholz.main().
    scenarios = [
        ("primary_midpoint_lengths_spread_counts", SEED + 0, mid, spread, 1.0, "history_and_scored"),
        ("sensitivity_short_lengths", SEED + 1, lengths["sensitivity_short_lengths"], spread, 1.0, "history_and_scored"),
        ("sensitivity_long_lengths", SEED + 2, lengths["sensitivity_long_lengths"], spread, 1.0, "history_and_scored"),
        ("sensitivity_median_concentrated_counts", SEED + 3, mid, conc, 1.0, "history_and_scored"),
        ("sensitivity_half_as_many_seizure_days", SEED + 4, mid, spread, 0.5, "history_and_scored"),
        ("new_counts_in_scored_days_only", SEED + 5, mid, spread, 1.0, "scored_only"),
        ("new_widest_counts", SEED + 6, mid, widest, 1.0, "history_and_scored"),
    ]
    rows = []
    for name, seed, lens, cnts, factor, cover in scenarios:
        row = {"scenario": name, "seed": seed, "replications": REPLICATIONS,
               "count_to_seizure_day_factor": factor, "counts_cover": cover}
        row.update(simulate(seed, lens, cnts, factor, cover))
        rows.append(row)
        print(name, {k: round(v, 3) for k, v in row.items() if isinstance(v, float)}, flush=True)

    out_dir = ROOT / "results" / "v14"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "B1_goldenholz_pooled.csv", index=False)

    v13 = pd.read_csv(ROOT / "results" / "v13" / "A5_goldenholz.csv").set_index("scenario")
    for row in rows[:5]:
        diff = abs(row["per_patient_mean"] - v13.loc[row["scenario"], "auc_mean"])
        assert diff < 1e-12, (row["scenario"], diff)
    print("per-patient AUC identical to results/v13/A5_goldenholz.csv in the five v13 scenarios")


if __name__ == "__main__":
    main()
