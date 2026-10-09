#!/usr/bin/env python3
"""Build paper tables from exported forecasts using the withinpatient package."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "code" / "withinpatient" / "src"))
from withinpatient import (auc, brier_scores, brier_skill, calibration_slopes,
                           pooled_auc, threshold_metrics, within_patient_auc)

RATE_COLUMNS = {"training_cohort_rate", "runin_rate", "runin_rate_shrunk",
    "runin_rate_shrunk_10", "updating_rate", "expanding_rate", "recent_rate",
    "recent_window_rate", "recent_28_night_rate", "last4_rate", "last12_rate",
    "persistence", "persistence_total_div_12", "retrospective_rate",
    "retrospective_rate_reference"}
MODEL_COLUMNS = {"logistic_regression", "gradient_boosting",
    "within_centred_gradient_boosting", "within_patient_centred_gradient_boosting"}
N_BOOT = 500
BOOT_SEED = 20261006


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)): return bool(value)
    return value


def _groups(patient):
    pid = np.asarray(patient)
    return [(key, np.flatnonzero(pid == key)) for key in pd.unique(pid)]


def _metric_values(pid, y, pred, threshold):
    po = pooled_auc(pid, y, pred)
    wi = within_patient_auc(pid, y, pred)
    th = threshold_metrics(pid, y, pred, threshold)
    return {"pooled_auc": po["value"], "within_patient_auc": wi["value"],
        "pooled_sensitivity": th["pooled"]["sensitivity"],
        "pooled_specificity": th["pooled"]["specificity"],
        "within_patient_sensitivity": th["within_patient"]["sensitivity"],
        "within_patient_specificity": th["within_patient"]["specificity"],
        "pooled_youden": th["pooled"]["youden"],
        "within_patient_youden": th["within_patient"]["youden"]}


def bootstrap_intervals(patient, y, pred, threshold, n_boot=N_BOOT, seed=BOOT_SEED):
    groups = [ix for _, ix in _groups(patient)]
    rng = np.random.default_rng(seed)
    draws = {key: [] for key in _metric_values(patient, y, pred, threshold)}
    pid = np.asarray(patient); yy = np.asarray(y, int); pp = np.asarray(pred, float)
    for _ in range(n_boot):
        chosen = rng.integers(0, len(groups), size=len(groups))
        ix = np.concatenate([groups[j] for j in chosen])
        # Relabel each bootstrap draw so duplicate sampled patients remain separate patients.
        bp = np.concatenate([np.full(len(groups[j]), k, dtype=int) for k, j in enumerate(chosen)])
        values = _metric_values(bp, yy[ix], pp[ix], threshold)
        for key, val in values.items(): draws[key].append(val)
    result = {key: [float(np.nanquantile(vals, .025)), float(np.nanquantile(vals, .975))]
              if np.isfinite(vals).any() else [None, None] for key, vals in draws.items()}
    return result


def bootstrap_model_intervals(patient, y, pred, references, n_boot=N_BOOT, seed=BOOT_SEED):
    """Paired patient bootstrap intervals for model Brier skills and calibration slopes."""
    groups = [ix for _, ix in _groups(patient)]
    rng = np.random.default_rng(seed)
    pid = np.asarray(patient); yy = np.asarray(y, int); pp = np.asarray(pred, float)
    refs = {name: np.asarray(value, float) for name, value in references.items()}
    skill_draws = {name: [] for name in refs}
    slope_names = ("pooled_slope", "between_slope", "within_slope")
    slope_draws = {name: [] for name in slope_names}
    slope_skipped = {name: 0 for name in slope_names}
    for _ in range(n_boot):
        chosen = rng.integers(0, len(groups), size=len(groups))
        ix = np.concatenate([groups[j] for j in chosen])
        bp = np.concatenate([np.full(len(groups[j]), k, dtype=int) for k, j in enumerate(chosen)])
        for name, ref in refs.items():
            value = brier_skill(yy[ix], pp[ix], ref[ix])
            if np.isfinite(value): skill_draws[name].append(float(value))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            slopes = calibration_slopes(bp, yy[ix], pp[ix])
        for name in slope_names:
            value = slopes[name]
            if np.isfinite(value): slope_draws[name].append(float(value))
            else: slope_skipped[name] += 1
    interval = lambda values: ([float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
                               if values else [None, None])
    return {"brier_skill_vs": {name: interval(values) for name, values in skill_draws.items()},
            "calibration_slopes": {name: {"ci95": interval(slope_draws[name]),
                                           "skipped": slope_skipped[name]}
                                   for name in slope_names}}


def analyse_file(path):
    df = pd.read_csv(path)
    required = {"patient", "t", "y", "threshold"}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"{path.name}: missing required columns {sorted(missing)}")
    forecasts = [c for c in df.columns if c not in required]
    if not forecasts: raise ValueError(f"{path.name}: no forecast columns")
    if df.empty: raise ValueError(f"{path.name}: no forecast rows")
    pid, y = df.patient.to_numpy(), df.y.astype(int).to_numpy()
    threshold = float(df.threshold.iloc[0])
    if not np.allclose(df.threshold.to_numpy(float), threshold):
        raise ValueError(f"{path.name}: threshold is not constant")
    patients = len(pd.unique(pid))
    entries = {}
    for j, name in enumerate(forecasts):
        p = df[name].to_numpy(float)
        po = pooled_auc(pid, y, p)
        wi = within_patient_auc(pid, y, p)
        if np.nanmin(p) < 0 or np.nanmax(p) > 1:
            # A raw score (for example the last questionnaire total), not a probability:
            # ranking metrics are defined, threshold, Brier and calibration metrics are not.
            entries[name] = {"pooled_auc": po["value"], "within_patient_auc": wi["value"],
                             "within_auc_eligible_patients": wi["eligible"],
                             "within_auc_excluded_patients": wi["excluded"], "not_a_probability": True}
            continue
        th = threshold_metrics(pid, y, p, threshold)
        bs = brier_scores(pid, y, p)
        entry = {"pooled_auc": po["value"],
            "within_patient_auc": wi["value"], "within_auc_eligible_patients": wi["eligible"],
            "within_auc_excluded_patients": wi["excluded"], "threshold": th,
            "brier_pooled": bs["pooled"]["brier"], "brier_patient_mean": bs["patient_mean"]["brier"]}
        if name in MODEL_COLUMNS or name not in RATE_COLUMNS:
            refs = {"runin_rate": "runin_rate" if "runin_rate" in df else None,
                "updating_rate": "updating_rate" if "updating_rate" in df else
                                 ("expanding_rate" if "expanding_rate" in df else None),
                "retrospective_rate": "retrospective_rate" if "retrospective_rate" in df else
                                      ("retrospective_rate_reference" if "retrospective_rate_reference" in df else None)}
            entry["brier_skill"] = {label: brier_skill(y, p, df[col].to_numpy(float))
                                    for label, col in refs.items() if col is not None}
            cal = calibration_slopes(pid, y, p)
            entry["calibration_slopes"] = {key: cal[key] for key in
                ("pooled_slope", "between_slope", "within_slope")}
        else:
            entry["brier_skill"] = {}
            entry["calibration_slopes"] = {}
        if name in ("gradient_boosting", "runin_rate"):
            entry["patient_bootstrap_95"] = bootstrap_intervals(pid, y, p, threshold,
                seed=BOOT_SEED + j)
        if name == "gradient_boosting":
            refs_for_boot = {label: df[col].to_numpy(float) for label, col in refs.items()
                             if col is not None}
            entry["patient_bootstrap_95"].update(bootstrap_model_intervals(
                pid, y, p, refs_for_boot, seed=BOOT_SEED + j))
        entries[name] = entry
    rates = pd.DataFrame({"patient": pid, "rate": y}).groupby("patient", sort=False).rate.mean()
    return {"file": path.name, "patients": patients, "scored_observations": len(df),
        "event_rate": float(np.mean(y)), "patient_event_rate_mean": float(rates.mean()),
        "patient_event_rate_sd": float(rates.std(ddof=1)) if len(rates) > 1 else None,
        "threshold": threshold, "forecasts": entries}


def _fmt(x, digits=3):
    return "NA" if x is None or not np.isfinite(x) else f"{float(x):.{digits}f}"


def _ci(entry, metric):
    interval = entry.get("patient_bootstrap_95", {}).get(metric)
    return f"[{_fmt(interval[0])}, {_fmt(interval[1])}]" if interval else "NA"


def _interval_text(interval):
    return f"[{_fmt(interval[0])}, {_fmt(interval[1])}]" if interval else "NA"


def render_tables(results):
    lines = ["# Case analysis forecast metrics", ""]
    for cohort, data in results.items():
        lines += [f"## {cohort}", "", f"Patients: {data['patients']}; observations: {data['scored_observations']}; event rate: {_fmt(data['event_rate'])}.", "",
            "| Forecast | Pooled AUC | Within AUC (eligible/excluded) | Sensitivity | Specificity | PPV | Youden | Within sens. | Within spec. | Within Youden | Alarms/30 mean | Alarms/30 median | Brier pooled | Brier patient mean | Brier skill | Calibration slopes pooled/between/within |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
        for name, e in data["forecasts"].items():
            if e.get("not_a_probability"): continue  # raw score: ranking metrics only, see JSON
            th=e["threshold"]; po=th["pooled"]; wi=th["within_patient"]
            skill="; ".join(f"{k}: {_fmt(v)}" for k,v in e["brier_skill"].items()) or "—"
            cal=e["calibration_slopes"]
            slopes=" / ".join(_fmt(cal.get(k)) for k in ("pooled_slope","between_slope","within_slope"))
            boot=e.get("patient_bootstrap_95", {})
            if name == "gradient_boosting":
                skill_ci=boot.get("brier_skill_vs", {})
                skill += "; 95% CI: " + "; ".join(f"{k} {_interval_text(v)}" for k,v in skill_ci.items())
                cal_ci=boot.get("calibration_slopes", {})
                slopes += " (95% CI: " + " / ".join(
                    f"{_interval_text(cal_ci.get(k,{}).get('ci95'))}; skipped {cal_ci.get(k,{}).get('skipped',0)}"
                    for k in ("pooled_slope","between_slope","within_slope")) + ")"
            lines.append(f"| {name} | {_fmt(e['pooled_auc'])} | {_fmt(e['within_patient_auc'])} ({e['within_auc_eligible_patients']}/{e['within_auc_excluded_patients']}) | {_fmt(po['sensitivity'])} | {_fmt(po['specificity'])} | {_fmt(po['ppv'])} | {_fmt(po['youden'])} | {_fmt(wi['sensitivity'])} | {_fmt(wi['specificity'])} | {_fmt(wi['youden'])} | {_fmt(th['alarms_per_patient_30_days']['mean'])} | {_fmt(th['alarms_per_patient_30_days']['median'])} | {_fmt(e['brier_pooled'])} | {_fmt(e['brier_patient_mean'])} | {skill} | {slopes} |")
        lines.append("")
    return "\n".join(lines)


def table7(results):
    chosen = [("replace_bg", "forecasts_replace_bg.csv"),
              ("ctn_weekly_missing_excluded", "forecasts_ctn_weekly_missing_excluded.csv"),
              ("ces", "forecasts_ces.csv")]
    rows=[]
    for label, filename in chosen:
        item=next((v for v in results.values() if v["file"] == filename), None)
        if item is None: continue
        f=item["forecasts"]; model=f["gradient_boosting"]; runin=f["runin_rate"]
        mth=model["threshold"]; rth=runin["threshold"]
        skill=model["brier_skill"]
        cal=model["calibration_slopes"]
        retro_name = "retrospective_rate" if "retrospective_rate" in f else "retrospective_rate_reference"
        rows.append([label, mth["eligible"]["both"], item["event_rate"], item["patient_event_rate_sd"],
            f[retro_name]["pooled_auc"], f["runin_rate"]["pooled_auc"],
            f"{_fmt(model['pooled_auc'])} {_ci(model, 'pooled_auc')}",
            f"{_fmt(model['within_patient_auc'])} {_ci(model, 'within_patient_auc')}",
            mth["pooled"]["sensitivity"], mth["pooled"]["specificity"],
            mth["within_patient"]["sensitivity"], mth["within_patient"]["specificity"],
            mth["pooled"]["youden"], mth["within_patient"]["youden"],
            rth["pooled"]["youden"], rth["within_patient"]["youden"],
            skill.get("runin_rate"), skill.get("updating_rate"), skill.get("retrospective_rate"),
            cal.get("pooled_slope"), cal.get("between_slope"), cal.get("within_slope"),
            "{}".format("; ".join(_interval_text(model.get("patient_bootstrap_95",{}).get("brier_skill_vs",{}).get(k)) for k in ("runin_rate","updating_rate","retrospective_rate"))),
            "{}".format(" / ".join(_interval_text(model.get("patient_bootstrap_95",{}).get("calibration_slopes",{}).get(k,{}).get("ci95")) for k in ("pooled_slope","between_slope","within_slope"))),
            "; ".join(str(model.get("patient_bootstrap_95",{}).get("calibration_slopes",{}).get(k,{}).get("skipped",0)) for k in ("pooled_slope","between_slope","within_slope"))])
    headers=["Cohort","Test patients (both outcomes)","Event rate","SD patient rates","Retrospective rate pooled AUC","Run-in pooled AUC","Model pooled AUC (95% CI)","Model within-patient AUC (95% CI)","Model pooled sensitivity","Model pooled specificity","Model within sensitivity","Model within specificity","Model pooled Youden","Model within Youden","Run-in pooled Youden","Run-in within Youden","Brier skill vs run-in","Brier skill vs updating","Brier skill vs retrospective","Pooled slope","Between slope","Within slope","Model Brier skill 95% CIs (run-in / updating / retrospective)","Model slope 95% CIs (pooled / between / within)","Skipped slope bootstrap samples (pooled / between / within)"]
    return "# Table 7. Test forecast performance\n\n| "+" | ".join(headers)+" |\n|"+"|".join(["---"]*len(headers))+"|\n"+"\n".join("| "+" | ".join(_fmt(x) if isinstance(x,(int,float,np.floating)) else str(x) for x in row)+" |" for row in rows)+"\n"


def selftest():
    import tempfile
    # Each patient has both outcomes, while the rate-only score is constant per person.
    toy=pd.DataFrame({"patient":[0,0,1,1,2,2,3,3],"t":[0,1]*4,
        "y":[0,1,1,0,0,1,1,0],"runin_rate":[.2,.2,.8,.8,.3,.3,.7,.7],
        "updating_rate":[.2,.4,.8,.6,.3,.5,.7,.5],"retrospective_rate":[.5]*8,
        "gradient_boosting":[.1,.8,.7,.2,.2,.7,.8,.1],"threshold":[.5]*8})
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/"forecasts_selftest.csv"; toy.to_csv(path,index=False)
        result=analyse_file(path)["forecasts"]["runin_rate"]
        assert result["threshold"]["within_patient"]["youden"] == 0.0
        assert result["within_patient_auc"] == 0.5
        boot = bootstrap_model_intervals(toy.patient.to_numpy(), toy.y.to_numpy(),
            toy.gradient_boosting.to_numpy(), {"runin_rate": toy.runin_rate.to_numpy(),
            "updating_rate": toy.updating_rate.to_numpy(),
            "retrospective_rate": toy.retrospective_rate.to_numpy()}, n_boot=12, seed=9)
        assert set(boot["brier_skill_vs"]) == {"runin_rate", "updating_rate", "retrospective_rate"}
        assert set(boot["calibration_slopes"]) == {"pooled_slope", "between_slope", "within_slope"}
        assert all(0 <= v["skipped"] <= 12 for v in boot["calibration_slopes"].values())
    print("SELFTEST PASS: rate-only within-patient Youden=0 and AUC=0.5")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--selftest",action="store_true"); args=ap.parse_args()
    if args.selftest: selftest(); return
    paths=sorted(p for p in ROOT.glob("forecasts_*.csv") if not p.stem.startswith("forecasts_val_"))
    results={p.stem.removeprefix("forecasts_"):analyse_file(p) for p in paths}
    (ROOT/"cases_tables.json").write_text(json.dumps(clean(results),indent=2,allow_nan=False),encoding="utf-8")
    (ROOT/"cases_tables.md").write_text(render_tables(results),encoding="utf-8")
    (ROOT/"cases_table7.md").write_text(table7(results),encoding="utf-8")
    print(f"Wrote tables for {len(results)} forecast files")


if __name__ == "__main__": main()
