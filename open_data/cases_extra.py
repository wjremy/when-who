
#!/usr/bin/env python3
"""Supplementary patient-variation and validation-round analyses.

Run this in the full model environment. It intentionally does not run models
unless invoked without --selftest.
"""
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SEED_REPLACE = 2026
SEED_CTN = 20261006
BOOT = 2000
ROUND_SEEDS = [2026, 2037, 2048, 2059, 2070]
EPS = 1e-6
PATIENT_COLUMNS = ["patient_index", "n_days", "n_events", "auc", "brier_model", "brier_reference"]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPLACE = load_module("replace_bg_analysis_extra", "replace_bg_analysis.py")
CTN = load_module("ctn0094_analysis_extra", "ctn0094_analysis.py")
CES = load_module("ces_analysis_extra", "ces_analysis.py")


def auc(y, score):
    """Use the unchanged tie-aware implementation from the source scripts."""
    # Both scripts implement the same Mann-Whitney AUC; REPLACE's function is
    # used here so all additions follow its tie and undefined-case behavior.
    return REPLACE.auc(y, score)


def brier(y, p):
    return float(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2))


def logloss(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def hanley_mcneil_variance(a, n_events, n_nonevents):
    """Hanley-McNeil large-sample variance for an empirical ROC AUC."""
    n1, n0 = int(n_events), int(n_nonevents)
    if a is None or n1 < 1 or n0 < 1 or not np.isfinite(a):
        return None
    q1 = a / (2.0 - a) if a != 2 else 1.0
    q2 = 2.0 * a * a / (1.0 + a)
    v = (a * (1 - a) + (n1 - 1) * (q1 - a * a) +
         (n0 - 1) * (q2 - a * a)) / (n1 * n0)
    return float(max(0.0, v))


def group_indices(pid):
    return [(key, np.flatnonzero(np.asarray(pid) == key)) for key in pd.unique(pid)]


def percentile_ci(values):
    vals = np.asarray(values, float)
    vals = vals[np.isfinite(vals)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) else [None, None]


def patient_bootstrap_share(differences, seed, nboot=BOOT):
    """Percentile patient bootstrap for the share with lower model Brier."""
    d = np.asarray(differences, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {"n_patients": 0, "share_model_lower": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    draws = np.empty(nboot, float)
    for i in range(nboot):
        ix = rng.integers(0, len(d), size=len(d))
        draws[i] = np.mean(d[ix] < 0)
    return {"n_patients": int(len(d)), "share_model_lower": float(np.mean(d < 0)),
            "ci95": percentile_ci(draws)}


def patient_analysis(y, pred, reference, pid, seed):
    y = np.asarray(y, int)
    pred = np.asarray(pred, float)
    reference = np.asarray(reference, float)
    pid = np.asarray(pid)
    rows = []
    aucs, sampling_vars = [], []
    brier_diffs = []
    for i, (key, ix) in enumerate(group_indices(pid), 1):
        yy = y[ix]
        aa = auc(yy, pred[ix])
        bm, br = brier(yy, pred[ix]), brier(yy, reference[ix])
        rows.append({"patient_index": i, "n_days": int(len(ix)), "n_events": int(yy.sum()),
                     "auc": aa, "brier_model": bm, "brier_reference": br})
        if aa is not None:
            aucs.append(float(aa))
            sampling_vars.append(hanley_mcneil_variance(aa, int(yy.sum()), int(len(yy) - yy.sum())))
        brier_diffs.append(bm - br)
    va = np.asarray(aucs, float)
    vs = np.asarray(sampling_vars, float)
    observed_var = float(np.var(va, ddof=1)) if len(va) > 1 else (0.0 if len(va) else None)
    mean_sampling = float(np.mean(vs)) if len(vs) else None
    attributable = (mean_sampling / observed_var if observed_var and observed_var > 0 else
                    (1.0 if observed_var == 0 and mean_sampling and mean_sampling > 0 else 0.0))
    summary = {
        "n_eligible": int(len(va)),
        "mean": float(np.mean(va)) if len(va) else None,
        "sd": float(np.std(va, ddof=1)) if len(va) > 1 else None,
        "min": float(np.min(va)) if len(va) else None,
        "deciles": {str(q): float(np.percentile(va, q)) for q in range(10, 100, 10)} if len(va) else {},
        "share_gt_0_5": float(np.mean(va > .5)) if len(va) else None,
        "share_gt_0_6": float(np.mean(va > .6)) if len(va) else None,
        "share_gt_0_7": float(np.mean(va > .7)) if len(va) else None,
        "observed_variance": observed_var,
        "mean_hanley_mcneil_sampling_variance": mean_sampling,
        "estimated_true_auc_sd": math.sqrt(max(0.0, observed_var - mean_sampling)) if observed_var is not None and mean_sampling is not None else None,
        "observed_variance_sampling_share": float(attributable) if attributable is not None else None,
        "observed_variance_sampling_share_capped_1": min(1.0, float(attributable)) if attributable is not None else None,
        "brier_comparison": patient_bootstrap_share(brier_diffs, seed),
    }
    return summary, pd.DataFrame(rows, columns=PATIENT_COLUMNS)


def split_replace(scored):
    ids = np.array(sorted(scored.pid.unique()), dtype=object)
    rng = np.random.default_rng(SEED_REPLACE)
    rng.shuffle(ids)
    ntr, nva = int(round(.60 * len(ids))), int(round(.20 * len(ids)))
    return ids[:ntr], ids[ntr:ntr+nva], ids[ntr+nva:]


def replace_data_and_models():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    panel = pd.read_csv(ROOT / "replace_bg_panel.csv")
    panel["pid"] = panel.pid.astype(str)
    panel["y"] = panel.y.astype(int)
    scored = panel.loc[~panel.model_runin.astype(bool)].copy()
    train_ids, val_ids, test_ids = split_replace(scored)
    daily = ["mean", "sd", "cv", "time_below70", "time_below54", "time_above180", "minimum",
             "last_value", "late_slope", "insulin_bolus", "basal_insulin_est", "carbs"]
    features = daily + ["history_rate_7", "history_rate_28", "expanding_rate", "runin_rate", "day"]
    d = scored.copy()
    runrows = panel.loc[panel.model_runin.astype(bool)]
    d["day"] = pd.to_numeric(d.day, errors="coerce")
    for col in ["insulin_bolus", "basal_insulin_est", "carbs"]:
        d[col + "_missing"] = d[col].isna().astype(float)
        features.append(col + "_missing")
    d["persistence"] = np.nan
    for pid, gg in panel.groupby("pid", sort=False):
        yy = gg.y.to_numpy(float)
        d.loc[d.pid.eq(pid), "persistence"] = np.r_[np.nan, yy[:-1]][14:]
    d["recent_rate"] = d.history_rate_28
    runmean = runrows.groupby("pid")[daily].mean()
    for col in daily:
        d[col + "_dev"] = d[col] - d.pid.map(runmean[col])
    centered = [c + "_dev" for c in daily]
    tr, va, te = (d[d.pid.isin(ids)].copy() for ids in (train_ids, val_ids, test_ids))

    # This is copied unchanged from replace_bg_analysis.py's train-only matrix function.
    def matrix(train, frame, cols):
        a = train[cols].apply(pd.to_numeric, errors="coerce"); b = frame[cols].apply(pd.to_numeric, errors="coerce")
        med = a.median().fillna(0); aa = a.fillna(med).to_numpy(float); bb = b.fillna(med).to_numpy(float)
        am = a.isna().to_numpy(float); bm = b.isna().to_numpy(float)
        mu = aa.mean(axis=0); sd = aa.std(axis=0); sd[sd < 1e-8] = 1
        return np.c_[(aa-mu)/sd, am], np.c_[(bb-mu)/sd, bm]

    Xtr, Xte = matrix(tr, te, features)
    _, Xva = matrix(tr, va, features)
    lr = LogisticRegression(C=1., max_iter=1000, solver="lbfgs", random_state=SEED_REPLACE)
    lr.fit(Xtr, tr.y)
    plr = lr.predict_proba(Xte)[:, 1]
    from sklearn.metrics import log_loss as sk_log_loss
    best = None
    for n_est in (50, 100, 200, 300):
        gb = GradientBoostingClassifier(n_estimators=n_est, max_depth=2, learning_rate=.05,
            min_samples_leaf=20, subsample=.8, random_state=SEED_REPLACE)
        gb.fit(Xtr, tr.y)
        pv = gb.predict_proba(Xva)[:, 1]
        loss = float(sk_log_loss(va.y, np.clip(pv, EPS, 1-EPS), labels=[0,1]))
        if best is None or loss < best[0]:
            best = (loss, n_est)
    gb = GradientBoostingClassifier(n_estimators=best[1], max_depth=2, learning_rate=.05,
        min_samples_leaf=20, subsample=.8, random_state=SEED_REPLACE).fit(Xtr, tr.y)
    pgb = gb.predict_proba(Xte)[:, 1]
    return {"name": "replace_bg", "scored": d, "train": tr, "validation": va, "test": te,
            "X_train": Xtr, "X_validation": Xva, "features": features,
            "predictions": {"logistic_regression": plr, "gradient_boosting": pgb},
            "references": {"updating_rate": te.expanding_rate.to_numpy(float)},
            "gb_selected_rounds": best[1], "curve_config": "replace"}


def ctn_data_and_models(event_col):
    raw = pd.read_csv(ROOT / "ctn0094_weekly_panel.csv")
    if event_col == "event_missing_positive":
        raw = CTN.rebuild_missing_positive_history(raw)
    split = CTN.split_people(raw)
    train_y = raw.loc[raw.who.isin(split["train"]) & raw[event_col].notna(), event_col].astype(int)
    train_rate = float(train_y.mean()) if len(train_y) else .5
    data = CTN.add_references(raw, event_col, train_rate)
    fitted = CTN.fit_models(data, event_col, split, True)
    if fitted.get("error"):
        raise RuntimeError(f"CTN source-script model fit failed: {fitted['error']}")
    test = CTN.test_rows(data, split)
    test = test.loc[test[event_col].notna()].copy()
    model_index = fitted.get("test_index", [])
    models = {}
    for name in ["logistic_regression", "gradient_boosting"]:
        p = fitted["predictions"].get(name)
        if p is None:
            raise RuntimeError(f"CTN source-script did not return {name}")
        models[name] = pd.Series(p, index=model_index).reindex(test.index).to_numpy(float)

    tr = data.loc[data.who.isin(split["train"]) & data[event_col].notna()].copy()
    va = data.loc[data.who.isin(split["validation"]) & data[event_col].notna()].copy()
    xtr = CTN.model_features(tr, True, False)
    xva = CTN.model_features(va, True, False)
    xtr = pd.get_dummies(xtr, columns=[c for c in ["project", "arm"] if c in xtr], dtype=float)
    xva = pd.get_dummies(xva, columns=[c for c in ["project", "arm"] if c in xva], dtype=float)
    xva = xva.reindex(columns=xtr.columns, fill_value=0)
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median")
    Xtr, Xva = imp.fit_transform(xtr), imp.transform(xva)
    return {"name": event_col, "event_col": event_col, "scored": data, "train": tr, "validation": va, "test": test,
            "X_train": Xtr, "X_validation": Xva,
            "predictions": models,
            "references": {"updating_rate": test.updating_rate.to_numpy(float)},
            "gb_selected_rounds": fitted["predictions"].get("gradient_boosting_validation", {}).get("selected_iterations"),
            "curve_config": "ctn"}


def ces_data_and_models():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    _, panel = CES.load_reports(CES.DATA, 3)
    split = CES.split_people(panel)
    panel = panel.copy(); panel["split"] = panel.pid.map(split)
    train = panel.loc[panel.split == "train"].copy()
    val = panel.loc[panel.split == "validation"].copy()
    test = panel.loc[panel.split == "test"].copy().reset_index(drop=True)
    cols = CES.ITEMS + ["gap_days", "last4_rate", "last12_rate", "expanding_rate", "runin_rate"]
    xtr, med = CES.make_features(train, cols)
    xv, _ = CES.make_features(val, cols, med)
    xt, _ = CES.make_features(test, cols, med)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=CES.SEED)).fit(xtr, train.y)
    plr = lr.predict_proba(xt)[:, 1]
    candidates = [50, 100, 150, 200, 300, 400, 600]
    staged = GradientBoostingClassifier(n_estimators=max(candidates), learning_rate=.05,
        max_depth=2, min_samples_leaf=20, random_state=CES.SEED).fit(xtr, train.y)
    losses = [CES.logloss(val.y.to_numpy(int), p[:, 1]) for p in staged.staged_predict_proba(xv)]
    best = candidates[int(np.argmin([losses[i-1] for i in candidates]))]
    gb = GradientBoostingClassifier(n_estimators=best, learning_rate=.05, max_depth=2,
        min_samples_leaf=20, random_state=CES.SEED).fit(xtr, train.y)
    return {"name": "ces", "scored": panel, "train": train, "validation": val, "test": test,
        "predictions": {"logistic_regression": plr, "gradient_boosting": gb.predict_proba(xt)[:, 1]},
        "references": {"updating_rate": test.expanding_rate.to_numpy(float)}}


def curve_model_config(which, seed):
    from sklearn.ensemble import GradientBoostingClassifier
    if which == "replace":
        return GradientBoostingClassifier(n_estimators=500, max_depth=2, learning_rate=.05,
            min_samples_leaf=20, subsample=.8, random_state=seed)
    # CTN-0094's source forecast is HistGradientBoostingClassifier (learning_rate=.08,
    # max_leaf_nodes=15, l2_regularization=1). Classic GBC is required for staged
    # probabilities; retain the compatible leaf and learning-rate settings and
    # subsample=.8 to expose the requested seed-dependent stochasticity.
    return GradientBoostingClassifier(n_estimators=500, learning_rate=.08,
        max_leaf_nodes=15, subsample=.8, random_state=seed)


def score_rounds(y, p, pid):
    groups = [ix for _, ix in group_indices(pid)]
    within = [auc(np.asarray(y)[ix], np.asarray(p)[ix]) for ix in groups]
    within = [float(x) for x in within if x is not None]
    per_brier = [brier(np.asarray(y)[ix], np.asarray(p)[ix]) for ix in groups]
    return {"pooled_auc": auc(y, p), "mean_within_patient_auc": float(np.mean(within)) if within else None,
            "pooled_log_loss": logloss(y, p), "pooled_brier": brier(y, p),
            "patient_mean_brier": float(np.mean(per_brier)) if per_brier else None}


def validation_curves(prepared):
    from sklearn.metrics import log_loss as sk_log_loss
    curves = []
    metrics = ["pooled_auc", "mean_within_patient_auc", "pooled_log_loss", "pooled_brier", "patient_mean_brier"]
    y_col = "y" if prepared["curve_config"] == "replace" else prepared["event_col"]
    y = prepared["validation"][y_col].astype(int).to_numpy()
    pid = (prepared["validation"]["pid"].to_numpy() if prepared["curve_config"] == "replace"
           else prepared["validation"]["who"].to_numpy())
    requested_rounds = set(range(10, 501, 10))
    for seed in ROUND_SEEDS:
        model = curve_model_config(prepared["curve_config"], seed)
        model.fit(prepared["X_train"], prepared["train"]["y" if prepared["curve_config"] == "replace" else prepared["name"]].astype(int).to_numpy())
        for round_no, p2 in enumerate(model.staged_predict_proba(prepared["X_validation"]), 1):
            if round_no not in requested_rounds:
                continue
            p = p2[:, 1]
            row = {"dataset": prepared.get("label", prepared["name"]), "seed": seed, "round": round_no}
            row.update(score_rounds(y, p, pid))
            curves.append(row)
    df = pd.DataFrame(curves)
    summaries = {}
    for seed, group in df.groupby("seed", sort=True):
        g = group.sort_values("round")
        by_round = g.set_index("round")
        best = {}
        for metric in metrics:
            vals = by_round[metric]
            idx = vals.idxmax() if metric in ("pooled_auc", "mean_within_patient_auc") else vals.idxmin()
            best[metric] = {"round": int(idx), "value": float(vals.loc[idx])}
        final = by_round.loc[500]
        def plateau(metric):
            vals = by_round[metric]
            for r in vals.index:
                if float(vals.loc[r:].max() - vals.loc[r]) < .002:
                    return int(r)
            return 500
        summaries[str(int(seed))] = {"best": best,
            "round_after_which_pooled_auc_remaining_gain_lt_0_002": plateau("pooled_auc"),
            "round_after_which_within_auc_remaining_gain_lt_0_002": plateau("mean_within_patient_auc"),
            "auc_at_requested_rounds": {str(r): {m: float(by_round.loc[r, m]) for m in ("pooled_auc", "mean_within_patient_auc")}
                                        for r in (10, 50, 100, 200, 300, 500)}}
    aggregate = {}
    for metric in metrics:
        rounds = [v["best"][metric]["round"] for v in summaries.values()]
        aggregate[metric] = {"mean_best_round": float(np.mean(rounds)),
                             "range_best_round": [int(min(rounds)), int(max(rounds))],
                             "rounds_by_seed": {s: v["best"][metric]["round"] for s, v in summaries.items()}}
    aggregate["plateau_rounds"] = {
        "pooled_auc_mean": float(np.mean([v["round_after_which_pooled_auc_remaining_gain_lt_0_002"] for v in summaries.values()])),
        "pooled_auc_range": [min(v["round_after_which_pooled_auc_remaining_gain_lt_0_002"] for v in summaries.values()), max(v["round_after_which_pooled_auc_remaining_gain_lt_0_002"] for v in summaries.values())],
        "within_auc_mean": float(np.mean([v["round_after_which_within_auc_remaining_gain_lt_0_002"] for v in summaries.values()])),
        "within_auc_range": [min(v["round_after_which_within_auc_remaining_gain_lt_0_002"] for v in summaries.values()), max(v["round_after_which_within_auc_remaining_gain_lt_0_002"] for v in summaries.values())]}
    aggregate["at_rounds_mean_range"] = {}
    for round_no in (10, 50, 100, 200, 300, 500):
        aggregate["at_rounds_mean_range"][str(round_no)] = {}
        for metric in ("pooled_auc", "mean_within_patient_auc"):
            vals = [v["auc_at_requested_rounds"][str(round_no)][metric] for v in summaries.values()]
            aggregate["at_rounds_mean_range"][str(round_no)][metric] = {
                "mean": float(np.mean(vals)), "range": [float(min(vals)), float(max(vals))]}
    return {"seed_results": summaries, "aggregate": aggregate}, df


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    return obj


def selftest():
    # Tiny simulated person panel; logistic probabilities are generated with a
    # direct numpy gradient fit so this mode needs no sklearn installation.
    rng = np.random.default_rng(991)
    records = []
    for person in range(12):
        x = rng.normal(size=24)
        logits = -.25 + .9 * x + (person % 3 - 1) * .35
        probability = 1 / (1 + np.exp(-logits))
        y = rng.binomial(1, probability)
        beta = np.zeros(2)
        xx = np.column_stack([np.ones(len(x)), x])
        for _ in range(500):
            p = 1 / (1 + np.exp(-np.clip(xx @ beta, -30, 30)))
            beta -= .08 * (xx.T @ (p - y) / len(y) + np.array([0.0, .01 * beta[1]]))
        pred = 1 / (1 + np.exp(-np.clip(xx @ beta, -30, 30)))
        ref = np.r_[.4, (2 + np.cumsum(y[:-1])) / (5 + np.arange(len(y)-1))]
        for yy, pp, rr in zip(y, pred, ref):
            records.append((person, int(yy), float(pp), float(rr)))
    arr = np.asarray(records, dtype=float)
    summ, table = patient_analysis(arr[:, 1].astype(int), arr[:, 2], arr[:, 3], arr[:, 0].astype(int), 71)
    assert len(table) == 12 and summ["n_eligible"] > 0
    assert summ["estimated_true_auc_sd"] >= 0
    assert 0 <= summ["brier_comparison"]["share_model_lower"] <= 1
    assert hanley_mcneil_variance(.5, 10, 10) >= 0
    assert hanley_mcneil_variance(1.0, 3, 3) == 0
    assert auc([0, 1], [0, 1]) == 1.0
    print("SELFTEST PASS: simulated numpy logistic forecasts, patient metrics/bootstrap, and Hanley-McNeil variance")


def one_dataset(name, prepare, errors, include_rounds=True):
    output = {"dataset": name}
    try:
        prepared = prepare()
        output["source_model"] = {"gradient_boosting_selected_rounds": prepared.get("gb_selected_rounds")}
    except Exception as exc:
        errors[f"{name}:prepare"] = f"{type(exc).__name__}: {exc}"
        print(f"FAILED {name} preparation: {type(exc).__name__}: {exc}")
        return output
    # A: the source-script logistic and gradient boosting forecasts on test patients.
    try:
        test = prepared["test"]
        pid_col = "pid" if "pid" in test else "who"
        y_col = "y" if "y" in test else prepared["event_col"]
        y = test[y_col].astype(int).to_numpy()
        pid = test[pid_col].to_numpy()
        output["test_patient_variation"] = {}
        tables = []
        for j, model_name in enumerate(("gradient_boosting", "logistic_regression")):
            ref = prepared["references"]["updating_rate"]
            summary, table = patient_analysis(y, prepared["predictions"][model_name], ref, pid,
                                              (SEED_REPLACE if name == "replace_bg" else SEED_CTN) + 500 + j)
            output["test_patient_variation"][model_name] = summary
            table.insert(0, "model", model_name)
            tables.append(table)
        output["patient_table"] = pd.concat(tables, ignore_index=True)
    except Exception as exc:
        errors[f"{name}:patient_variation"] = f"{type(exc).__name__}: {exc}"
        print(f"FAILED {name} patient variation: {type(exc).__name__}: {exc}")
    # B: five seeded 500-round staged validation curves.
    if include_rounds:
        try:
            prepared["label"] = name
            summary, curve = validation_curves(prepared)
            output["validation_rounds"] = summary
            output["round_table"] = curve
        except Exception as exc:
            errors[f"{name}:validation_rounds"] = f"{type(exc).__name__}: {exc}"
            print(f"FAILED {name} validation rounds: {type(exc).__name__}: {exc}")
    return output


def fmt(x, digits=4):
    return "NA" if x is None else f"{x:.{digits}f}" if isinstance(x, (int, float)) else str(x)


def make_markdown(results):
    lines = ["# Extra patient variation and boosting-round analyses", "",
             "Patient AUC includes patients with both outcome classes. Sampling variance uses the Hanley-McNeil approximation. The Brier comparison bootstrap resamples patients. For the plateau summary, the reported round is the first sampled round with less than 0.002 maximum remaining AUC gain over later sampled rounds.", ""]
    lines += ["## Test-patient variation", "", "| Dataset | Model | Eligible patients | Mean AUC | SD | Min | AUC > .5 | AUC > .6 | AUC > .7 | True AUC SD estimate | Sampling share | Model Brier wins (95% patient bootstrap) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for d in results["datasets"]:
        for model, s in d.get("test_patient_variation", {}).items():
            boot = s["brier_comparison"]
            ci = boot["ci95"]
            lines.append(f"| {d['dataset']} | {model} | {s['n_eligible']} | {fmt(s['mean'])} | {fmt(s['sd'])} | {fmt(s['min'])} | {fmt(s['share_gt_0_5'])} | {fmt(s['share_gt_0_6'])} | {fmt(s['share_gt_0_7'])} | {fmt(s['estimated_true_auc_sd'])} | {fmt(s['observed_variance_sampling_share'])} | {fmt(boot['share_model_lower'])} [{fmt(ci[0])}, {fmt(ci[1])}] (n={boot['n_patients']}) |")
    lines += ["", "Deciles, observed and mean sampling variances, and per-seed details are in the JSON output.", "", "## Validation over 500 boosting rounds", "", "The CTN-0094 curve uses staged `GradientBoostingClassifier` because the source model is `HistGradientBoostingClassifier` and does not provide the required staged prediction API. It carries over learning rate .08 and max leaf nodes 15; classic GBC has no `l2_regularization` option. REPLACE-BG uses its source GBC settings.", "", "| Dataset | Metric | Mean best round | Range of best rounds across seeds |", "|---|---|---:|---:|"]
    for d in results["datasets"]:
        agg = d.get("validation_rounds", {}).get("aggregate", {})
        for metric, value in agg.items():
            if isinstance(value, dict) and "mean_best_round" in value:
                lines.append(f"| {d['dataset']} | {metric} | {fmt(value['mean_best_round'], 1)} | {value['range_best_round']} |")
    for d in results["datasets"]:
        ag = d.get("validation_rounds", {}).get("aggregate", {})
        if ag:
            lines += ["", f"### {d['dataset']}: AUC by round", "", "| Round | Pooled AUC mean (range) | Mean within-patient AUC mean (range) |", "|---:|---:|---:|"]
            for r, vals in ag.get("at_rounds_mean_range", {}).items():
                p, w = vals["pooled_auc"], vals["mean_within_patient_auc"]
                lines.append(f"| {r} | {fmt(p['mean'])} ({fmt(p['range'][0])}–{fmt(p['range'][1])}) | {fmt(w['mean'])} ({fmt(w['range'][0])}–{fmt(w['range'][1])}) |")
            plateau = ag["plateau_rounds"]
            pl = {"pooled_auc_mean": plateau["pooled_auc_mean"], "pooled_auc_range": plateau["pooled_auc_range"]}
            wi_mean = plateau.get("within_auc_mean")
            wi_range = plateau.get("within_auc_range")
            if wi_mean is None or wi_range is None:
                lines.append(f"\nAUC plateau rounds: pooled mean {fmt(pl['pooled_auc_mean'],1)} (range {pl['pooled_auc_range']}); within-patient mean unavailable in saved results.")
            else:
                lines.append(f"\nAUC plateau rounds: pooled mean {fmt(pl['pooled_auc_mean'],1)} (range {pl['pooled_auc_range']}); within-patient mean {fmt(wi_mean,1)} (range {wi_range}).")
    if results.get("failures"):
        lines += ["", "## Failures", ""]
        lines.extend(f"- `{key}`: {value}" for key, value in results["failures"].items())
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    failures = {}
    datasets = [
        one_dataset("replace_bg", replace_data_and_models, failures),
        one_dataset("weekly_missing_excluded", lambda: ctn_data_and_models("event"), failures),
        one_dataset("weekly_missing_positive", lambda: ctn_data_and_models("event_missing_positive"), failures),
        one_dataset("ces", ces_data_and_models, failures, include_rounds=False),
    ]
    results = {"seed": {"replace_bg": SEED_REPLACE, "ctn0094": SEED_CTN},
               "round_seeds": ROUND_SEEDS, "datasets": [], "failures": failures}
    for item in datasets:
        patient_table = item.pop("patient_table", None)
        round_table = item.pop("round_table", None)
        results["datasets"].append(item)
        if patient_table is not None:
            path = ROOT / f"cases_extra_per_patient_{item['dataset']}.csv"
            try:
                patient_table.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
                print(f"Wrote {path.name}")
            except Exception as exc:
                failures[f"{item['dataset']}:patient_csv"] = f"{type(exc).__name__}: {exc}"
                print(f"FAILED writing patient CSV {path.name}: {type(exc).__name__}: {exc}")
        if round_table is not None:
            path = ROOT / f"cases_extra_rounds_{item['dataset']}.csv"
            try:
                round_table.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
                print(f"Wrote {path.name}")
            except Exception as exc:
                failures[f"{item['dataset']}:round_csv"] = f"{type(exc).__name__}: {exc}"
                print(f"FAILED writing rounds CSV {path.name}: {type(exc).__name__}: {exc}")
    # Failures added during file writes are kept in final deliverables as well.
    results["failures"] = failures
    try:
        (ROOT / "cases_extra_results.json").write_text(json.dumps(clean(results), indent=2, allow_nan=False), encoding="utf-8")
        print("Wrote cases_extra_results.json")
    except Exception as exc:
        failures["json_output"] = f"{type(exc).__name__}: {exc}"
        print(f"FAILED writing JSON: {type(exc).__name__}: {exc}")
    try:
        (ROOT / "cases_extra_results.md").write_text(make_markdown(clean(results)), encoding="utf-8")
        print("Wrote cases_extra_results.md")
    except Exception as exc:
        failures["markdown_output"] = f"{type(exc).__name__}: {exc}"
        print(f"FAILED writing Markdown: {type(exc).__name__}: {exc}")
    if failures:
        print("\n")
        for name, msg in failures.items():
            print(f"- {name}: {msg}")
    else:
        print("All analysis blocks completed.")


if __name__ == "__main__":
    main()
