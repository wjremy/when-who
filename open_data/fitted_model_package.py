"""Results 5.3: the CTN-0094 gradient-boosting model of Table 4, evaluated with withinpatient 2.0.

The fitted model is held fixed; in each within-patient shuffle of a test patient's scored weeks its
outcome-history features (expanding rate, recent rate, previous result) are rebuilt from the shuffled
outcomes, as in Table 4 (`cases_model_fixed_permutation.fit_ctn`). The four run-in weeks are not rows of
the scored data: they enter only through the patient's history features and stay fixed.

Outputs: open_data/results/v10/fitted_model_package.md and .json.
Run: PYTHONPATH=code/withinpatient/src python open_data/fitted_model_package.py [--n-perm 200]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "code" / "withinpatient" / "src"))
from withinpatient import evaluate, to_markdown  # noqa: E402

OUT = ROOT / "results" / "v10"
SEED = 20261010
METRIC_LABELS = {
    "auc": "AUC", "average_precision": "Average precision", "sensitivity": "Sensitivity",
    "specificity": "Specificity", "ppv": "PPV (excess over own rate)", "npv": "NPV (excess over own non-event rate)",
    "net_benefit": "Net benefit (vs own-rate policy)", "brier_skill": "Brier skill (vs own rate)",
    "calibration_slope": "Calibration slope",
}


def load_permutation_module():
    path = ROOT / "cases_model_fixed_permutation.py"
    spec = importlib.util.spec_from_file_location("_fmp_cases_model_fixed_permutation", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def fmt(value, digits=3):
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=200)
    args = parser.parse_args()

    pm = load_permutation_module()
    fit = pm.fit_ctn()
    test = fit["test"]
    threshold = float(test.training_cohort_rate.iloc[0]) if "training_cohort_rate" in test else None
    export = pd.read_csv(ROOT / "forecasts_ctn_weekly_missing_excluded.csv")
    if threshold is None:
        threshold = float(export.training_cohort_rate.iloc[0])

    data = pd.DataFrame({
        "patient": test.who.to_numpy(), "week": test.study_week.to_numpy(),
        "y": fit["y"], "forecast": np.asarray(fit["predictions"]["gradient_boosting"], dtype=float),
        "row": test.index.to_numpy(),
    })
    # The fixed model's forecasts must be the Table 4 export (same model, same test weeks).
    if not np.allclose(np.sort(data.forecast.to_numpy()), np.sort(export.gradient_boosting.to_numpy()), atol=1e-10):
        raise AssertionError("fixed-model forecasts differ from forecasts_ctn_weekly_missing_excluded.csv")

    def rebuild_gb(y, scored, rows):
        # One patient's weeks in time order; rebuild the history features from y and score the fixed model.
        pid = rows.patient.iloc[0]
        order = test.loc[test.who.eq(pid)].index
        position = pd.Series(np.arange(len(order)), index=order)
        target = np.empty(len(order), dtype=int)
        target[position.loc[rows.row].to_numpy()] = y
        forecast = fit["gb_for_patient"](pid, target)
        return forecast[position.loc[rows.row].to_numpy()]

    started = time.time()
    result = evaluate(data, patient="patient", time="week", outcome="y", forecast="forecast",
                      threshold=threshold, rebuild=rebuild_gb, n_perm=args.n_perm, seed=SEED)
    seconds = time.time() - started

    table4 = json.loads((OUT / "table4.json").read_text(encoding="utf-8"))
    rows = []
    for name, metric in result["metrics"].items():
        s, c = metric["summary"], metric["comparison"]
        chances = [e["chance_level"] for e in metric["per_patient"].values()]
        rows.append({
            "metric": name, "pooled": c.get("pooled"),
            "usual_mean": c.get("usual_per_patient_mean"), "usual_median": c.get("usual_per_patient_median"),
            "mean_observed": s["mean_observed"], "mean_chance": s["mean_chance_level"],
            "median_chance": float(np.median(chances)) if chances else float("nan"),
            "median_excess": s["median_excess"], "iqr_excess": s["iqr_excess"],
            "share_above_p05": s["share_p_below_0_05"], "share_excess_positive": s["share_excess_positive"],
            "n_eligible": s["n_eligible"],
            "conditional_slope": c.get("conditional_within_slope"),
            "conditional_slope_chance": c.get("conditional_within_slope_chance"),
            "conditional_slope_excess": c.get("conditional_within_slope_excess"),
            "conditional_slope_p": c.get("conditional_within_slope_p"),
        })

    lines = ["# Results 5.3: the CTN-0094 gradient-boosting model, evaluated with withinpatient 2.0", "",
             f"Test patients {result['settings']['patients']}, scored weeks {result['settings']['scored_rows']} "
             f"(weeks 4 onward; the four run-in weeks enter only through the fixed history features). "
             f"Threshold = training-cohort event rate {threshold:.3f}. {args.n_perm} within-patient shuffles per patient, "
             f"seed {SEED}; fitted model fixed, history features rebuilt in every shuffle; yardstick = patient's own scored rate, "
             f"rebuilt per shuffle; chance level = mean over shuffles. Run time {seconds/60:.1f} min.", "",
             "| Measure | Pooled | Usual per patient, mean (median) | Fixed: mean observed | Mean chance | Median excess [IQR] | Patients above chance (p < 0.05) | n |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["metric"] == "calibration_slope":
            lines.append(f"| Calibration slope, conditional within-patient (all patients) | {fmt(r['pooled'])} | "
                         f"{fmt(r['usual_mean'])} | {fmt(r['conditional_slope'])} | {fmt(r['conditional_slope_chance'])} | "
                         f"{fmt(r['conditional_slope_excess'])} (p = {fmt(r['conditional_slope_p'])}) | | {r['n_eligible']} |")
            continue
        lines.append(f"| {METRIC_LABELS[r['metric']]} | {fmt(r['pooled'])} | {fmt(r['usual_mean'])} ({fmt(r['usual_median'])}) | "
                     f"{fmt(r['mean_observed'])} | {fmt(r['mean_chance'])} | {fmt(r['median_excess'])} "
                     f"[{fmt(r['iqr_excess'][0])}, {fmt(r['iqr_excess'][1])}] | {r['share_above_p05']:.0%} | {r['n_eligible']} |")
    lines += ["", "Pooled and usual per-patient values are the package's comparison columns (usual references: 0.5 for AUC, "
              "population base rate for AP, PPV and NPV, population rate for Brier skill, treat-all/none for net benefit).", ""]

    auc = next(r for r in rows if r["metric"] == "auc")
    lines += ["## Check against Table 4", "",
              f"AUC chance level here: mean {fmt(auc['mean_chance'])}, median over patients {fmt(auc['median_chance'])}. "
              "Table 4 CTN-0094 AUC chance entry: see `table4.json` (key printed below).", ""]
    t4 = table4["cohorts"]["ctn_weekly_missing_excluded"]["permutation_chance"]["model"]["auc"]
    lines.append(f"Table 4 CTN-0094 gradient-boosting AUC chance (mean over 200 model-fixed draws of the mean per-patient AUC): "
                 f"{t4['mean_over_draws']:.4f} [2.5-97.5%: {t4['percentiles_2_5_97_5'][0]:.3f}, {t4['percentiles_2_5_97_5'][1]:.3f}]. "
                 f"Package: {auc['mean_chance']:.4f}. Same estimand (mean over patients of each patient's mean over shuffles), different shuffle draws.")
    lines += ["", "## The call", "", "```python",
              "result = evaluate(data, patient=\"patient\", time=\"week\", outcome=\"y\", forecast=\"forecast\",",
              f"                  threshold={threshold:.5f}, rebuild=rebuild_gb, n_perm={args.n_perm}, seed={SEED})",
              "print(to_markdown(result))", "```", "",
              "`rebuild_gb(y, scored, rows)` receives one patient's time-ordered outcomes and rows, rebuilds the history "
              "features with `ctn_rebuild_histories` and returns the fixed model's predictions; the package checks that it "
              "reproduces the forecast column on the observed outcomes.", "",
              "## Package output (to_markdown)", "", to_markdown(result), ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fitted_model_package.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "fitted_model_package.json").write_text(json.dumps(clean({
        "settings": result["settings"], "threshold": threshold, "seconds": seconds, "rows": rows,
        "per_patient": {k: v["per_patient"] for k, v in result["metrics"].items()},
    }), indent=1), encoding="utf-8")
    print(f"done in {seconds/60:.1f} min")


if __name__ == "__main__":
    main()
