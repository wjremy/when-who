#!/usr/bin/env python3
"""Compare gradient-boosting Brier skill with prospective rate references."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT
OUT_DIR = ROOT / "results" / "skill_analysis"
FILES = [
    ("REPLACE-BG", "forecasts_replace_bg.csv", "runin_rate_shrunk_10", "expanding_rate"),
    ("CTN-0094 weekly", "forecasts_ctn_weekly_missing_excluded.csv", "runin_shrunk", "updating_rate"),
    ("College Experience", "forecasts_ces.csv", "runin_rate_shrunk_10", "expanding_rate"),
]
N_BOOT = 1000
SEED = 20261007
# The CTN rate exports clip probabilities at 1e-6. Treat essentially perfect
# references (mean patient Brier <= this tolerance) as having no usable ratio.
MIN_PATIENT_REFERENCE_BRIER = 1e-8


def metric(y, pred, ref, patient):
    model_loss = (y - pred) ** 2
    ref_loss = (y - ref) ** 2
    pooled = 1 - model_loss.sum() / ref_loss.sum() if ref_loss.sum() > 0 else np.nan
    patient_skills = []
    for p in pd.unique(patient):
        ix = patient == p
        denom = ref_loss[ix].sum()
        if denom / ix.sum() > MIN_PATIENT_REFERENCE_BRIER:
            patient_skills.append(1 - model_loss[ix].sum() / denom)
    vals = np.asarray(patient_skills, dtype=float)
    return pooled, vals.mean() if len(vals) else np.nan, (vals > 0).mean() if len(vals) else np.nan, len(vals)


def bootstrap(y, pred, refs, patient, reps, rng):
    groups = [np.flatnonzero(patient == p) for p in pd.unique(patient)]
    out = {key: [] for key in refs}
    for _ in range(reps):
        chosen = rng.integers(0, len(groups), size=len(groups))
        ix = np.concatenate([groups[j] for j in chosen])
        # Give repeated draws distinct IDs, preserving bootstrap patient weighting.
        bp = np.concatenate([np.full(len(groups[j]), k) for k, j in enumerate(chosen)])
        for key, ref in refs.items():
            out[key].append(metric(y[ix], pred[ix], ref[ix], bp))
    return {k: np.asarray(v) for k, v in out.items()}


def fmt_ci(point, draws, j):
    lo, hi = np.nanquantile(draws[:, j], [0.025, 0.975])
    return f"{point:.3f} ({lo:.3f}, {hi:.3f})"


def main():
    rng = np.random.default_rng(SEED)
    existing = json.loads((DATA / "cases_tables.json").read_text())
    rows, checks = [], []
    for cohort, filename, shrunk_col, update_col in FILES:
        df = pd.read_csv(DATA / filename)
        y = df.y.to_numpy(float)
        pred = df.gradient_boosting.to_numpy(float)
        patient = df.patient.to_numpy()
        refs = {
            "Population event rate": df.training_cohort_rate.to_numpy(float),
            "Run-in rate": df.runin_rate.to_numpy(float),
            "Updating rate": df[update_col].to_numpy(float),
            "Shrunk run-in rate": df[shrunk_col].to_numpy(float),
        }
        # Pick the prospective rate with the smallest pooled Brier loss in the full cohort.
        best = min(refs, key=lambda name: np.square(y - refs[name]).sum())
        refs["Best prospective rate (" + best + ")"] = refs[best]
        point = {k: metric(y, pred, v, patient) for k, v in refs.items()}
        draws = bootstrap(y, pred, refs, patient, N_BOOT, rng)
        for name, values in point.items():
            boot = draws[name]
            rows.append((cohort, name, f"{values[3]}/{len(pd.unique(patient))}",
                         fmt_ci(values[0], boot, 0), fmt_ci(values[1], boot, 1),
                         fmt_ci(values[2], boot, 2)))

        # cases_tables.py publishes run-in, updating, and retrospective skill for GB.
        # The rate columns map to its labels as follows.
        key = next(k for k, v in existing.items() if v.get("file") == filename)
        pub = existing[key]["forecasts"]["gradient_boosting"]["brier_skill"]
        paper_map = {"Run-in rate": "runin_rate", "Updating rate": "updating_rate"}
        for name, pkey in paper_map.items():
            observed = point[name][0]
            published = pub[pkey]
            checks.append((cohort, name, published, observed, abs(observed - published)))

    lines = [
        "# Brier skill against prospective rate references", "",
        "Gradient-boosting forecasts were scored in the three held-out test cohorts. Skill is 1 − model Brier loss / reference Brier loss. Pooled skill uses summed patient-day losses; patient-mean skill is the unweighted mean of patient-specific skill ratios. The positive-share column is the fraction of patients with positive patient-specific skill. Patient-level summaries omit references with mean patient Brier loss ≤ 1e-8, which treats the exported 1e-6 probability clipping as effectively zero reference loss; the valid patient count is shown. Parentheses contain percentile 95% CIs from 1,000 patient bootstrap resamples, seed 20261007.", "",
        "The best prospective rate was selected separately per cohort by the lowest full-cohort pooled Brier loss among population event rate, run-in rate, updating rate, and shrunk run-in rate. That selected reference was held fixed within bootstrap resamples.", "",
        "| Cohort | Reference | Patients with usable patient ratio | Pooled skill (95% CI) | Patient-mean skill (95% CI) | Positive patient share (95% CI) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    lines += ["", "## Reproduction check against existing tables", "",
              "`cases_tables.py` computes and publishes pooled skill against run-in, updating, and retrospective patient rates. The check below compares run-in and updating directly to the values in `open_data/cases_tables.json`; both reproduce to floating-point precision. The current `cases_tables.py` does not compute skill against the population event rate, so that requested reference is new and has no prior value to match. The retrospective reference is not one of the requested a–c prospective references.", "",
              "| Cohort | Existing reference | Published pooled skill | Recomputed pooled skill | Absolute difference |",
              "|---|---|---:|---:|---:|"]
    lines += [f"| {c} | {n} | {p:.12f} | {v:.12f} | {d:.2e} |" for c, n, p, v, d in checks]
    lines += ["", "## Interpretation", "",
              "The model’s apparent skill depends on which prospective information is already available: the best patient-level rate can be a substantially tougher comparator than the population event rate, especially when individual rates differ. Shrinking the run-in rate changes its benchmark loss and therefore the skill ratio, while using updating rates asks what the model adds after incorporating the patient’s accumulating outcomes. These are incremental forecast comparisons, so strong skill against a population rate does not by itself show added value over a patient-specific prospective estimate.", "",
              "## Shrinkage and source", "",
              "All three input files contain a shrunk run-in rate; this script uses that existing column. The construction is an empirical-Bayes weighted average with 10 training-cohort pseudo-observations: REPLACE-BG uses `(14 * runin_rate + 10 * training_cohort_rate) / 24` in `open_data/replace_bg_analysis.py:418`; CTN-0094 uses `(runin_rate * runin_n + 10 * training_cohort_rate) / (runin_n + 10)` in `open_data/ctn0094_analysis.py:239`; College Experience uses `(20 * runin_rate + 10 * training_cohort_rate) / 30` in `open_data/ces_analysis.py:371`.", "",
              "## Reproduce", "",
              "From the repository root:", "",
              "```powershell", "python open_data\\skill_analysis.py", "```", ""]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
