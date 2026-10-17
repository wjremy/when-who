"""Per-patient AUC and per-patient shuffled baseline of the fitted gradient-boosting model, for
Figure 9 (Section 5.3; the model of Table 5) in CTN-0094, REPLACE-BG and the College Experience Study.

The cohort tables keep only cohort means. This script repeats the model-fixed permutation run of
open_data/cases_model_fixed_permutation.py (same fits, same seed, the same order of cohorts, so the
same 200 shuffles per patient, each with rebuilt history inputs) and keeps each patient's values:
the observed per-patient AUC and the mean of that patient's 200 shuffled AUCs. It checks both against
open_data/results/v10/table4.json (per-patient observed AUCs; cohort mean of the shuffled baseline).

Output: results/v19/F9_patients.csv (cohort, patient, auc, shuffled_baseline, valid_shuffles)
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OPEN = ROOT / "open_data"
OUT = ROOT / "results" / "v19" / "F9_patients.csv"
TABLE4_KEY = {"replace_bg": "replace_bg", "ctn0094": "ctn_weekly_missing_excluded", "ces": "ces"}
LABEL = {"replace_bg": "REPLACE-BG", "ctn0094": "CTN-0094", "ces": "College Experience Study"}


def load_module():
    path = OPEN / "cases_model_fixed_permutation.py"
    spec = importlib.util.spec_from_file_location("_v19_mfp", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    pm = load_module()
    table4 = json.loads((OPEN / "results" / "v10" / "table4.json").read_text(encoding="utf-8"))
    prepared = {"replace_bg": pm.fit_replace_bg(), "ctn0094": pm.fit_ctn(), "ces": pm.fit_ces()}
    rng_perm = np.random.default_rng(pm.SEED)  # as in pm.run(): one generator across the cohorts in this order
    rows = []
    for key in ("replace_bg", "ctn0094", "ces"):
        bundle = prepared[key]
        perms = pm.generate_permuted_targets(bundle["y"], bundle["pid"], pm.N_PERM, rng_perm)
        auc = pm.feature_auc_matrix(bundle["y"], perms, bundle["pid"], bundle["predictions"],
                                    bundle["score_permutation"])["gradient_boosting"]
        null = auc["permutation_auc"]
        baseline = np.nanmean(null, axis=0)
        ref = table4["cohorts"][TABLE4_KEY[key]]
        observed_ref = ref["model"]["metrics"]["auc"]["per_patient"]
        for pid, value, base, valid in zip(auc["observed_ids"], auc["observed_auc"], baseline,
                                           np.isfinite(null).sum(axis=0)):
            assert abs(observed_ref[str(pid)] - value) < 1e-9, (key, pid)
            rows.append({"cohort": LABEL[key], "patient": str(pid), "auc": float(value),
                         "shuffled_baseline": float(base), "valid_shuffles": int(valid)})
        draw_mean = float(np.mean(np.nanmean(null, axis=1)))
        stored = ref["permutation_chance"]["model"]["auc"]["mean_over_draws"]
        print(f"{LABEL[key]}: {len(auc['observed_ids'])} patients, mean AUC {np.mean(auc['observed_auc']):.4f}, "
              f"mean shuffled baseline {np.mean(baseline):.4f} (draw mean {draw_mean:.6f}, table4 {stored:.6f})", flush=True)
        assert abs(draw_mean - stored) < 1e-9, (key, draw_mean, stored)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
