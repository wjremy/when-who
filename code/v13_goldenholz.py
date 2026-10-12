"""No-skill per-patient AUC simulation for Goldenholz et al. (2024).

The prospective paper reports summary ranges, rather than the individual
forecastable lengths and patient-level seizure-day counts. See
results/v13/A5_sources.md for the source locations and explicit assumptions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "wheneval" / "src"))
from wheneval import auc  # noqa: E402


SEED = 20261012
REPLICATIONS = 5000
PATIENTS = 25
LOOKBACK_DAYS = 84


def length_profiles() -> dict[str, tuple[np.ndarray, str]]:
    """Forecastable-day vectors honoring reported min=15, median=max=57."""
    midpoint = np.array([15] + [36] * 11 + [57] * 13, dtype=int)
    shortest = np.array([15] * 12 + [57] * 13, dtype=int)
    longest = np.array([15] + [56] * 11 + [57] * 13, dtype=int)
    return {
        "primary_midpoint_lengths": (
            midpoint,
            "1x15;11x36;13x57 scored days (assumed midpoint for unreported lengths)",
        ),
        "sensitivity_short_lengths": (
            shortest,
            "12x15;13x57 scored days (short feasible profile)",
        ),
        "sensitivity_long_lengths": (
            longest,
            "1x15;11x56;13x57 scored days (long feasible profile)",
        ),
    }


def count_profiles() -> dict[str, tuple[np.ndarray, str]]:
    """Synthetic seizure-count vectors honoring the reported range/median."""
    spread = np.array(list(range(1, 25)) + [56], dtype=int)
    concentrated = np.array([1] + [13] * 23 + [56], dtype=int)
    return {
        "reported_range_median_spread_profile": (
            spread,
            "synthetic counts 1,2,...,24,56; preserves reported min=1, median=13, max=56",
        ),
        "sensitivity_median_concentrated_profile": (
            concentrated,
            "synthetic counts 1, thirteen repeated 23 times, 56; preserves reported summaries",
        ),
    }


def simulate_scenario(
    seed: int,
    lengths: np.ndarray,
    counts: np.ndarray,
    count_to_day_factor: float,
    scenario: str,
    length_description: str,
    count_description: str,
    replications: int = REPLICATIONS,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    cohort_auc = np.full(replications, np.nan, dtype=float)
    eligible_n = np.zeros(replications, dtype=int)

    for b in range(replications):
        # Patient-specific frequency/length pairing is unreported; randomly
        # pair the synthetic seizure-count profile and forecastable lengths.
        shuffled_counts = rng.permutation(counts)
        aucs: list[float] = []

        for length, total_seizures in zip(lengths, shuffled_counts):
            full_days = LOOKBACK_DAYS + int(length)
            p = min(1.0, count_to_day_factor * float(total_seizures) / full_days)
            outcomes = rng.binomial(1, p, size=full_days).astype(np.int8)

            # At each scored day, forecast the fraction of seizure-days in
            # the immediately preceding 84 days. The first 84 days are warm-up.
            cumulative = np.concatenate(([0], np.cumsum(outcomes, dtype=np.int64)))
            starts = np.arange(int(length), dtype=np.int64)
            ends = starts + LOOKBACK_DAYS
            forecasts = (cumulative[ends] - cumulative[starts]) / LOOKBACK_DAYS
            scored_outcomes = outcomes[LOOKBACK_DAYS:]

            value = auc(scored_outcomes, forecasts)
            if np.isfinite(value):
                aucs.append(value)

        if aucs:
            cohort_auc[b] = float(np.mean(aucs))
            eligible_n[b] = len(aucs)

    valid = cohort_auc[np.isfinite(cohort_auc)]
    if len(valid) == 0:
        raise RuntimeError(f"No cohort had an eligible per-patient AUC: {scenario}")

    return {
        "scenario": scenario,
        "n_patients": PATIENTS,
        "replications": replications,
        "seed": seed,
        "lookback_days": LOOKBACK_DAYS,
        "scored_days_min": int(lengths.min()),
        "scored_days_median": int(np.median(lengths)),
        "scored_days_max": int(lengths.max()),
        "scored_days_profile": length_description,
        "seizure_count_profile": count_description,
        "count_to_seizure_day_factor": count_to_day_factor,
        "cohorts_with_defined_auc": int(len(valid)),
        "auc_mean": float(np.mean(valid)),
        "auc_q025": float(np.quantile(valid, 0.025)),
        "auc_q975": float(np.quantile(valid, 0.975)),
        "eligible_patients_mean": float(np.mean(eligible_n)),
        "eligible_patients_q025": float(np.quantile(eligible_n, 0.025)),
        "eligible_patients_q975": float(np.quantile(eligible_n, 0.975)),
    }


def main() -> None:
    out_dir = ROOT / "results" / "v13"
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = length_profiles()
    count_options = count_profiles()
    rows: list[dict[str, object]] = []
    scenario_index = 0

    # Primary assumption: midpoint lengths, a synthetic spread count profile,
    # and one seizure-day per reported seizure. Factors and count profiles not
    # reported by the paper are varied in the sensitivity rows below.
    lengths, length_desc = profiles["primary_midpoint_lengths"]
    counts, count_desc = count_options["reported_range_median_spread_profile"]
    rows.append(
        simulate_scenario(
            SEED + scenario_index, lengths, counts, 1.0,
            "primary_midpoint_lengths_spread_counts", length_desc, count_desc,
        )
    )
    scenario_index += 1

    # Exact patient-level follow-up lengths are not reported in the main text.
    for key in ("sensitivity_short_lengths", "sensitivity_long_lengths"):
        lengths, length_desc = profiles[key]
        rows.append(
            simulate_scenario(
                SEED + scenario_index, lengths, counts, 1.0,
                key, length_desc, count_desc,
            )
        )
        scenario_index += 1

    # Patient-level seizure frequency distribution is not reported; this
    # alternative profile still exactly matches the reported range and median.
    lengths, length_desc = profiles["primary_midpoint_lengths"]
    counts, concentrated_desc = count_options["sensitivity_median_concentrated_profile"]
    rows.append(
        simulate_scenario(
            SEED + scenario_index, lengths, counts, 1.0,
            "sensitivity_median_concentrated_counts", length_desc, concentrated_desc,
        )
    )
    scenario_index += 1

    # The paper reports seizure counts, while the moving average uses seizure-
    # days. Halve counts as a transparent sensitivity for two seizures/day on
    # average; this is illustrative, not estimated from the article.
    counts, spread_desc = count_options["reported_range_median_spread_profile"]
    rows.append(
        simulate_scenario(
            SEED + scenario_index, lengths, counts, 0.5,
            "sensitivity_half_as_many_seizure_days", length_desc,
            spread_desc + "; assumed one seizure-day per two reported seizures",
        )
    )

    pd.DataFrame(rows).to_csv(out_dir / "A5_goldenholz.csv", index=False)


if __name__ == "__main__":
    main()
