"""Per-cohort arrays for the primary Goldenholz reconstruction in v14."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

# Reuse v14's primary profiles, seed, constants, and Equation 1 implementation.
import v14_goldenholz_pooled as v14  # noqa: E402
from wheneval import auc  # noqa: E402


PRIMARY_SCENARIO = "primary_midpoint_lengths_spread_counts"


def simulate_primary_cohorts() -> pd.DataFrame:
    """Return the v14 primary metrics for each cohort, preserving its RNG order."""
    lengths = v14.length_profiles()["primary_midpoint_lengths"][0]
    counts = v14.count_profiles()["reported_range_median_spread_profile"][0]
    rng = np.random.default_rng(v14.SEED + 0)
    weights = lengths.astype(float)

    per_patient_auc = np.full(v14.REPLICATIONS, np.nan, dtype=float)
    pooled_auc = np.full(v14.REPLICATIONS, np.nan, dtype=float)
    a0 = np.full(v14.REPLICATIONS, np.nan, dtype=float)

    for b in range(v14.REPLICATIONS):
        # This is the same permutation and patient order as v14.simulate().
        shuffled_counts = rng.permutation(counts)
        rates = np.empty(len(lengths))
        aucs, ys, fs = [], [], []
        for k, (length, total) in enumerate(zip(lengths, shuffled_counts)):
            full_days = v14.LOOKBACK_DAYS + int(length)
            p = min(1.0, 1.0 * float(total) / full_days)
            rates[k] = p
            y = rng.binomial(1, p, size=full_days).astype(np.int8)
            cumulative = np.concatenate(([0], np.cumsum(y, dtype=np.int64)))
            starts = np.arange(int(length), dtype=np.int64)
            forecasts = (
                cumulative[starts + v14.LOOKBACK_DAYS] - cumulative[starts]
            ) / v14.LOOKBACK_DAYS
            scored = y[v14.LOOKBACK_DAYS:]
            value = auc(scored, forecasts)
            if np.isfinite(value):
                aucs.append(value)
            ys.append(scored)
            fs.append(forecasts)

        y_all, f_all = np.concatenate(ys), np.concatenate(fs)
        per_patient_auc[b] = np.mean(aucs) if aucs else np.nan
        pooled_auc[b] = auc(y_all, f_all)
        a0[b] = v14.a0_equation1(rates, weights)

    return pd.DataFrame({
        "per_patient_auc": per_patient_auc,
        "pooled_auc": pooled_auc,
        "a0": a0,
    })


def main() -> None:
    cohorts = simulate_primary_cohorts()
    output = ROOT / "results" / "v19" / "F6_cohorts.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    cohorts.to_csv(output, index=False)

    primary = pd.read_csv(ROOT / "results" / "v14" / "B1_goldenholz_pooled.csv")
    primary = primary.loc[primary["scenario"] == PRIMARY_SCENARIO]
    assert len(primary) == 1, f"Expected one v14 primary row, found {len(primary)}"
    primary_row = primary.iloc[0]
    summary = {
        "per_patient_mean": float(cohorts["per_patient_auc"].mean()),
        "per_patient_q025": float(cohorts["per_patient_auc"].quantile(0.025)),
        "per_patient_q975": float(cohorts["per_patient_auc"].quantile(0.975)),
        "pooled_mean": float(cohorts["pooled_auc"].mean()),
        "a0_mean": float(cohorts["a0"].mean()),
    }
    for field, actual in summary.items():
        expected = float(primary_row[field])
        assert abs(actual - expected) <= 1e-9, (
            f"{field}: cohort result {actual:.15g} differs from v14 {expected:.15g}"
        )
    print("Primary cohort summaries:", summary, flush=True)


if __name__ == "__main__":
    main()
