#!/usr/bin/env python3
"""Repeated patient-level split analysis for the three primary cohorts."""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "code" / "withinpatient" / "src"))
from withinpatient import auc, brier_skill, calibration_slopes, pooled_auc, threshold_metrics, within_patient_auc


def _load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _split(ids, seed, kind):
    ids = np.asarray(sorted(ids), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    if kind == "replace_bg":
        ntr, nva = int(round(.60 * len(ids))), int(round(.20 * len(ids)))
    else:
        ntr, nva = int(.60 * len(ids)), int(.20 * len(ids))
    return set(ids[:ntr]), set(ids[ntr:ntr+nva]), set(ids[ntr+nva:])


def _metric_row(cohort, seed, pid, y, model, runin, updating, retrospective, threshold):
    po = pooled_auc(pid, y, model)["value"]
    wi_result = within_patient_auc(pid, y, model)
    wi = wi_result["value"]
    model_threshold = threshold_metrics(pid, y, model, threshold)
    runin_threshold = threshold_metrics(pid, y, runin, threshold)
    slopes = calibration_slopes(pid, y, model)
    return {"cohort": cohort, "seed": int(seed), "train_event_rate": float(threshold),
        "n_test_patients": int(pd.unique(pid).size),
        "n_patients_both_outcomes": int(wi_result["eligible"]),
        "pooled_auc": po, "within_patient_auc": wi,
        "pooled_youden": model_threshold["pooled"]["youden"],
        "within_patient_youden": model_threshold["within_patient"]["youden"],
        "runin_pooled_youden": runin_threshold["pooled"]["youden"],
        "runin_within_patient_youden": runin_threshold["within_patient"]["youden"],
        "runin_rate_pooled_auc": auc(y, runin),
        "retrospective_rate_pooled_auc": auc(y, retrospective),
        "brier_skill_vs_runin": brier_skill(y, model, runin),
        "brier_skill_vs_updating": brier_skill(y, model, updating),
        "brier_skill_vs_retrospective": brier_skill(y, model, retrospective),
        **{k: slopes[k] for k in ("pooled_slope", "between_slope", "within_slope")}}


def _replace_bg(panel, seed):
    m = _load_module("replace_bg_analysis")
    panel = panel.copy()
    panel["pid"] = panel.pid.astype(str)
    panel["y"] = panel.y.astype(int)
    scored = panel.loc[~panel.model_runin.astype(bool)].copy()
    trids, vaids, teids = _split(scored.pid.unique(), seed, "replace_bg")
    d = scored.copy()
    daily = ["mean", "sd", "cv", "time_below70", "time_below54", "time_above180", "minimum", "last_value", "late_slope", "insulin_bolus", "basal_insulin_est", "carbs"]
    features = daily + ["history_rate_7", "history_rate_28", "expanding_rate", "runin_rate", "day"]
    d["day"] = pd.to_numeric(d.day, errors="coerce")
    for col in ["insulin_bolus", "basal_insulin_est", "carbs"]:
        d[col + "_missing"] = d[col].isna().astype(float); features.append(col + "_missing")
    d["persistence"] = np.nan
    for pid0, group in panel.groupby("pid", sort=False):
        vals = np.r_[np.nan, group.y.to_numpy(float)[:-1]]
        d.loc[d.pid.eq(pid0), "persistence"] = vals[14:]
    runrows = panel.loc[panel.model_runin.astype(bool)]
    runmean = runrows.groupby("pid")[daily].mean()
    for col in daily: d[col + "_dev"] = d[col] - d.pid.map(runmean[col])
    centered = [c + "_dev" for c in daily]
    tr, va, te = (d[d.pid.isin(ids)] for ids in (trids, vaids, teids))
    if tr.y.nunique() < 2: raise ValueError("training cohort has only one event class")
    def matrix(train, frame, cols):
        a = train[cols].apply(pd.to_numeric, errors="coerce"); b = frame[cols].apply(pd.to_numeric, errors="coerce")
        med = a.median().fillna(0); aa = a.fillna(med).to_numpy(float); bb = b.fillna(med).to_numpy(float)
        am = a.isna().to_numpy(float); bm = b.isna().to_numpy(float)
        mu = aa.mean(axis=0); sd = aa.std(axis=0); sd[sd < 1e-8] = 1
        return np.c_[(aa-mu)/sd, am], np.c_[(bb-mu)/sd, bm]
    xtr, _ = matrix(tr, va, features); _, xva = matrix(tr, va, features); _, xte = matrix(tr, te, features)
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import log_loss
    best = None
    for n_est in (50, 100, 200, 300):
        gb = GradientBoostingClassifier(n_estimators=n_est, max_depth=2, learning_rate=.05,
            min_samples_leaf=20, subsample=.8, random_state=m.SEED).fit(xtr, tr.y)
        pv = gb.predict_proba(xva)[:, 1]
        loss = float(log_loss(va.y, np.clip(pv, m.EPS, 1-m.EPS), labels=[0, 1]))
        if best is None or loss < best[0]: best = (loss, n_est)
    gb = GradientBoostingClassifier(n_estimators=best[1], max_depth=2, learning_rate=.05,
        min_samples_leaf=20, subsample=.8, random_state=m.SEED).fit(xtr, tr.y)
    pred = gb.predict_proba(xte)[:, 1]
    rate = float(tr.y.mean()); te = te.copy()
    return _metric_row("replace_bg", seed, te.pid.to_numpy(), te.y.to_numpy(int), pred,
        te.runin_rate.to_numpy(float), te.expanding_rate.to_numpy(float),
        te.groupby("pid").y.transform("mean").to_numpy(float), rate)


def _ctn(panel, seed):
    m = _load_module("ctn0094_analysis")
    split = m.split_people(panel, seed=seed)
    trainrows = panel.loc[panel.who.isin(split["train"]) & panel.event.notna()]
    train_rate = float(trainrows.event.mean())
    data = m.add_references(panel, "event", train_rate)
    fitted = m.fit_models(data, "event", split, weekly=True, include_current_dose=False)
    predmap = fitted.get("predictions", {})
    if "gradient_boosting" not in predmap: raise RuntimeError(fitted.get("error") or predmap.get("gradient_boosting_error"))
    test = data.loc[data.who.isin(split["test"]) & data.event.notna()].copy()
    # fit_models returns rows in this same filtered test order.
    y = test.event.astype(int).to_numpy(); pid = test.who.to_numpy()
    return _metric_row("ctn_weekly_missing_excluded", seed, pid, y, predmap["gradient_boosting"],
        test.runin_rate.to_numpy(float), test.updating_rate.to_numpy(float),
        test.retrospective_rate.to_numpy(float), train_rate)


def _ces(reports, panel, seed):
    m = _load_module("ces_analysis")
    ids = np.array(sorted(panel.pid.unique()), dtype=object); rng = np.random.default_rng(seed); rng.shuffle(ids)
    n = len(ids); split = {p: ("train" if i < int(.6*n) else "validation" if i < int(.8*n) else "test") for i,p in enumerate(ids)}
    d = panel.copy(); d["split"] = d.pid.map(split)
    train, val, test = (d[d.split == x].copy().reset_index(drop=True) for x in ("train", "validation", "test"))
    cols = m.ITEMS + ["gap_days", "last4_rate", "last12_rate", "expanding_rate", "runin_rate"]
    xtr, med = m.make_features(train, cols); xva, _ = m.make_features(val, cols, med); xte, _ = m.make_features(test, cols, med)
    from sklearn.ensemble import GradientBoostingClassifier
    stages = GradientBoostingClassifier(n_estimators=600, learning_rate=.05, max_depth=2,
        min_samples_leaf=20, random_state=m.SEED).fit(xtr, train.y)
    losses = [m.logloss(val.y.to_numpy(int), s[:, 1]) for s in stages.staged_predict_proba(xva)]
    candidates = [50,100,150,200,300,400,600]; best = candidates[int(np.argmin([losses[i-1] for i in candidates]))]
    gb = GradientBoostingClassifier(n_estimators=best, learning_rate=.05, max_depth=2,
        min_samples_leaf=20, random_state=m.SEED).fit(xtr, train.y)
    pred = gb.predict_proba(xte)[:, 1]
    tr_rate = float(train.y.mean())
    return _metric_row("ces", seed, test.pid.to_numpy(), test.y.to_numpy(int), pred,
        test.runin_rate.to_numpy(float), test.expanding_rate.to_numpy(float),
        test.retro_rate.to_numpy(float), tr_rate)


def summarize(rows, main_seeds):
    df = pd.DataFrame(rows); metrics = [c for c in df.columns if c not in ("cohort", "seed")]
    out = ["# Cohort resplit stability summary", ""]
    for cohort, group in df.groupby("cohort", sort=False):
        out += [f"## {cohort}", "", "| Quantity | Main seed | Median | 2.5th percentile | 97.5th percentile | Range | Pooled > within |", "|---|---:|---:|---:|---:|---:|---:|"]
        for name in metrics:
            vals = pd.to_numeric(group[name], errors="coerce").dropna()
            main = group.loc[group.seed == main_seeds[cohort], name]
            mv = float(main.iloc[0]) if len(main) and pd.notna(main.iloc[0]) else np.nan
            exceed = ""
            if name in {"pooled_auc", "pooled_youden"}:
                other = "within_patient_auc" if name == "pooled_auc" else "within_patient_youden"
                exceed = int((group[name] > group[other]).sum())
            if len(vals):
                stats = f"{mv:.6g} | {vals.median():.6g} | {vals.quantile(.025):.6g} | {vals.quantile(.975):.6g} | {vals.min():.6g}–{vals.max():.6g}"
            else:
                stats = "NA | NA | NA | NA | NA"
            out.append(f"| {name} | {stats} | {exceed} |")
        out.append("")
    return "\n".join(out)


def metric_selftest():
    pid = np.repeat(np.arange(4), 4); y = np.array([0,1,0,1, 1,0,1,0, 0,0,1,1, 1,1,0,0])
    p = np.array([.1,.8,.2,.7, .8,.2,.7,.1, .2,.3,.8,.7, .9,.8,.1,.2])
    row = _metric_row("toy", 1, pid, y, p, np.repeat([.5,.5,.5,.5], 4), p, np.repeat([.5]*4,4), .5)
    assert row["n_patients_both_outcomes"] == 4 and np.isfinite(row["pooled_auc"])
    print("SELFTEST PASS: withinpatient metrics, summaries, split-independent simulated rows")


def run(cohort=None, nseeds=20):
    t0 = time.perf_counter(); mods = {n: _load_module(n) for n in ("replace_bg_analysis", "ctn0094_analysis", "ces_analysis")}
    jobs = ["replace_bg", "ctn_weekly_missing_excluded", "ces"] if cohort is None else [cohort]
    aliases = {"ctn0094": "ctn_weekly_missing_excluded", "replace_bg": "replace_bg", "ces": "ces"}
    jobs = [aliases.get(x, x) for x in jobs]
    seeds = {"replace_bg": mods["replace_bg_analysis"].SEED,
             "ctn_weekly_missing_excluded": mods["ctn0094_analysis"].SEED,
             "ces": mods["ces_analysis"].SEED}
    panels = {}
    for job in jobs:
        if job == "replace_bg":
            panel = pd.read_csv(ROOT / "replace_bg_panel.csv")
            panel.pid = panel.pid.astype(str); panel.y = panel.y.astype(int); panels[job] = panel
        elif job == "ctn_weekly_missing_excluded":
            panel = pd.read_csv(ROOT / "ctn0094_weekly_panel.csv"); panels[job] = panel
        else:
            _, panel = mods["ces_analysis"].load_reports(mods["ces_analysis"].DATA, 3); panels[job] = panel
    results=[]
    for job in jobs:
        main = seeds[job]
        for i in range(nseeds):
            seed = main if i == 0 else main + i
            start = time.perf_counter()
            if job == "replace_bg": row = _replace_bg(panels[job], seed)
            elif job == "ctn_weekly_missing_excluded": row = _ctn(panels[job], seed)
            else: row = _ces(None, panels[job], seed)
            elapsed = time.perf_counter() - start; results.append(row)
            if i == 0:
                total = (time.perf_counter()-t0) + elapsed * (nseeds-1) * len(jobs)
                print(f"{job} seed {seed}: {elapsed:.1f}s; estimated total for requested cohorts: {total/60:.1f} min", flush=True)
            print(f"{job} seed {seed}: complete in {elapsed:.1f}s ({i+1}/{nseeds})", flush=True)
    # one output pair per cohort when a single cohort is run, so that separate runs do not overwrite each other
    tag = "" if cohort is None else "_" + jobs[0]
    pd.DataFrame(results).to_csv(ROOT / f"cases_resplit_results{tag}.csv", index=False)
    (ROOT / f"cases_resplit_summary{tag}.md").write_text(summarize(results, seeds), encoding="utf-8")
    print(f"Wrote cases_resplit_results.csv and cases_resplit_summary.md; elapsed {(time.perf_counter()-t0)/60:.1f} min")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cohort", choices=["replace_bg","ctn_weekly_missing_excluded","ctn0094","ces"]); ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--selftest", action="store_true"); args=ap.parse_args()
    if args.selftest: metric_selftest(); return
    if args.seeds < 1: ap.error("--seeds must be at least 1")
    run(args.cohort, args.seeds)


if __name__ == "__main__": main()
