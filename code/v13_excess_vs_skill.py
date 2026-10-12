"""A4: compare tailored shuffled-baseline AUC excess with true AUC skill.

Usage:
    python code/v13_excess_vs_skill.py

The A and B rows come from open_data/results/v10/ranking_sim.json. No-skill
rows are simulated with ranking_sim's cohort fitting and outcome-generation
helpers, then scored against 50 rebuilt within-patient shuffles per record.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.dont_write_bytecode = True
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import ranking_sim as sim  # noqa: E402

INPUT = ROOT / "open_data" / "results" / "v10" / "ranking_sim.json"
OUTPUT = ROOT / "results" / "v13" / "A4_excess_vs_skill.csv"
NO_SKILL_REPS = 200
NO_SKILL_SHUFFLES = 50
NO_SKILL_SEED_STREAM = 1304
COHORTS = ("REPLACE-BG", "CTN-0094")
RECORD_DAYS = (30, 100, 365)
PATIENTS = 50


def rate_so_far(y: np.ndarray) -> np.ndarray:
    """Exclusive expanding rate; the first day is left unused as a forecast day."""
    yy = np.asarray(y, dtype=float)
    prior = np.cumsum(yy, axis=-1) - yy
    elapsed = np.arange(yy.shape[-1], dtype=float)
    return np.divide(prior, elapsed, out=np.zeros_like(prior), where=elapsed > 0)


def no_skill_excess_rows() -> list[dict[str, object]]:
    cells = [(cohort, days, PATIENTS) for cohort in COHORTS for days in RECORD_DAYS]
    # _context performs the project cohort fits; no A calibration is needed for this forecast.
    specs, fits, _ = sim._context([], include_extra=False)
    cohort_index = {name: i for i, name in enumerate(sim.COHORTS)}
    result = []
    for cohort, days, n_patients in cells:
        ci = cohort_index[cohort]
        spec, fit = specs[cohort], fits[cohort]
        rep_excesses = np.empty(NO_SKILL_REPS, dtype=float)
        for rep in range(NO_SKILL_REPS):
            rng = sim._rng(NO_SKILL_SEED_STREAM, ci, days, rep)
            b = sim._recentered_levels(rng, fit, spec, n_patients)
            _, y = sim._draw_outcomes(rng, b, days)
            f = rate_so_far(y)
            observed_auc = sim._auc_matrix(y[:, 1:], f[:, 1:])[0]

            # Shuffle all days within each patient and rebuild the rate-so-far forecast.
            # The same number of events per full record is retained in every shuffle.
            y_perm = sim._make_shuffles(rng, y, NO_SKILL_SHUFFLES)
            f_perm = rate_so_far(y_perm)
            shuffled_auc = sim._auc_matrix(y_perm[:, :, 1:], f_perm[:, :, 1:])
            shuffled_valid = np.isfinite(shuffled_auc)
            patient_baseline = np.divide(
                np.nansum(shuffled_auc, axis=0), shuffled_valid.sum(axis=0),
                out=np.full(n_patients, np.nan, dtype=float),
                where=shuffled_valid.sum(axis=0) > 0,
            )
            valid = np.isfinite(observed_auc) & np.isfinite(patient_baseline)
            rep_excesses[rep] = float(np.mean(observed_auc[valid] - patient_baseline[valid]))

        spread = "low" if cohort == "REPLACE-BG" else "high"
        result.append({
            "cohort": cohort,
            "spread": spread,
            "record_days": days,
            "patients": n_patients,
            "pair": "none",
            "forecast": "no-skill",
            "mean_excess_auc": float(np.mean(rep_excesses)),
            "true_skill_auc_minus_half": 0.0,
            "source": (
                "rate so far; ranking_sim._context/_recentered_levels/_draw_outcomes; "
                f"{NO_SKILL_REPS} replications x {NO_SKILL_SHUFFLES} rebuilt shuffles; "
                f"SeedSequence base={sim.BASE_SEED}, stream={NO_SKILL_SEED_STREAM}"
            ),
        })
        print(
            f"A4 no-skill {cohort} T={days}: mean excess={rep_excesses.mean():.6f}",
            flush=True,
        )
    return result


def ranking_rows() -> list[dict[str, object]]:
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    selected = [
        row for row in payload["ranking"]
        if row.get("scoring") == "tailored_excess"
        and row.get("metric") == "auc"
        and row.get("record_days") in RECORD_DAYS
        and row.get("patients") == PATIENTS
        and row.get("cohort") in COHORTS
    ]
    rows = []
    for row in selected:
        for forecast, excess_key, truth_key in (
            ("A", "mean_a_score", "long_record_a_true_patient_auc"),
            ("B", "mean_b_score", "long_record_b_true_patient_auc"),
        ):
            rows.append({
                "cohort": row["cohort"],
                "spread": row["patient_spread"],
                "record_days": row["record_days"],
                "patients": row["patients"],
                "pair": row["pair"],
                "forecast": forecast,
                "mean_excess_auc": row[excess_key],
                "true_skill_auc_minus_half": row[truth_key] - 0.5,
                "source": "open_data/results/v10/ranking_sim.json; scoring=tailored_excess; metric=auc",
            })
    if not selected:
        raise RuntimeError(f"No tailored_excess AUC rows found in {INPUT}")
    return rows


def main() -> None:
    rows = ranking_rows()
    rows.extend(no_skill_excess_rows())
    columns = [
        "cohort", "spread", "record_days", "patients", "pair", "forecast",
        "mean_excess_auc", "true_skill_auc_minus_half", "source",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False, float_format="%.10g")
    print(f"Wrote {len(frame)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
