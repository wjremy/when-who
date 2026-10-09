#!/usr/bin/env python3
"""Patient-level real-data analysis for REPLACE-BG; starts from replace_bg_panel.csv."""
import argparse
import json
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
B_RANK = 1000
B_CAL = 500
N_PERM = 200
ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "replace_bg_panel.csv"
JSON_OUT = ROOT / "replace_bg_results.json"
MD_OUT = ROOT / "replace_bg_results.md"
EPS = 1e-6


def auc(y, score):
    y = np.asarray(y, int); s = np.asarray(score, float)
    n1 = int(y.sum()); n0 = len(y) - n1
    if not n1 or not n0: return None
    order = np.argsort(s, kind="mergesort"); v = s[order]
    starts = np.r_[0, np.flatnonzero(v[1:] != v[:-1]) + 1]
    ends = np.r_[starts[1:], len(v)]
    r = np.empty(len(v), float)
    for lo, hi in zip(starts, ends): r[lo:hi] = (lo + 1 + hi) / 2
    ranks = np.empty(len(v)); ranks[order] = r
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ap_tied(y, score):
    y = np.asarray(y, int); s = np.asarray(score, float); n = int(y.sum())
    if not n: return None
    order = np.argsort(-s, kind="mergesort"); sy = y[order]; ss = s[order]
    starts = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1]) + 1]
    ends = np.r_[starts[1:], len(y)]; ctp = 0; val = 0.
    for lo, hi in zip(starts, ends):
        m = hi - lo; ctp += int(sy[lo:hi].sum()); val += (ctp / hi) * int(sy[lo:hi].sum()) / n
    return float(val)


def ap_tie_expectation(y, score):
    y = np.asarray(y, int); s = np.asarray(score, float); T = len(y); n = int(y.sum())
    if not n: return None
    if T < 2: return 1. if n else None
    order = np.argsort(-s, kind="mergesort"); ss = s[order]
    starts = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1]) + 1]
    ends = np.r_[starts[1:], T]; prior = 0; a = b = 0.
    for lo, hi in zip(starts, ends):
        m = hi - lo; cum = hi
        a += m / cum
        b += m * (prior + m - 1) / cum
        prior = cum
    return float(a / T + (n - 1) * b / (T * (T - 1)))


def logloss(y, p):
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def brier(y, p): return float(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2))


def expit(x):
    x = np.asarray(x, float); z = np.empty_like(x); m = x >= 0
    z[m] = 1 / (1 + np.exp(-x[m])); e = np.exp(x[~m]); z[~m] = e / (1 + e)
    return z


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def irls(X, y, weights=None, max_iter=100, tol=1e-8):
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)]); y = np.asarray(y, float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    beta = np.zeros(X.shape[1]); beta[0] = logit(np.average(y, weights=w))
    for _ in range(max_iter):
        p = expit(np.clip(X @ beta, -30, 30)); v = np.maximum(p * (1 - p), 1e-10) * w
        h = X.T @ (v[:, None] * X); grad = X.T @ (w * (y - p))
        step = np.linalg.solve(h + np.eye(h.shape[0]) * 1e-10, grad)
        beta += step
        if np.max(np.abs(step)) < tol: return beta
    if np.max(np.abs(grad)) > 1e-4: raise RuntimeError("calibration IRLS did not converge")
    return beta


def ci(x):
    a = np.asarray(x, float); a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]


def patient_groups(pid):
    return [(k, np.flatnonzero(np.asarray(pid) == k)) for k in pd.unique(pid)]


def boot_patient_indices(pid, nboot, seed):
    groups = patient_groups(pid); rng = np.random.default_rng(seed)
    for _ in range(nboot):
        chosen = rng.integers(0, len(groups), size=len(groups))
        yield [groups[i][1] for i in chosen]


def paired_boot_ci(pid, fn, nboot, seed):
    out = []
    for parts in boot_patient_indices(pid, nboot, seed):
        ix = np.concatenate(parts); newpid = np.concatenate([np.full(len(x), j) for j, x in enumerate(parts)])
        try: out.append(fn(ix, newpid))
        except (ValueError, np.linalg.LinAlgError, RuntimeError, ZeroDivisionError): out.append(np.nan)
    return ci(out)


def within_auc(y, p, pid):
    vals = [auc(np.asarray(y)[ix], np.asarray(p)[ix]) for _, ix in patient_groups(pid)]
    vals = [x for x in vals if x is not None]
    return (float(np.mean(vals)) if vals else None, len(vals))


def safe_float(v):
    if v is None: return None
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    return v


def selftest():
    assert abs(auc([0, 1], [0, 1]) - 1) < 1e-12
    assert abs(auc([0, 1], [0, 0]) - .5) < 1e-12
    assert abs(ap_tied([1, 0, 1], [1, 0, 0]) - (5/6)) < 1e-12
    assert abs(ap_tie_expectation([1, 0, 1], [1, 0, 0]) - (7/9)) < 1e-12
    fit = irls(np.array([[-2.], [-1.], [1.], [2.], [0.], [0.]]), np.array([0, 0, 1, 1, 0, 1]))
    assert fit.shape == (2,) and fit[1] > 0
    assert abs(float(expit(0)) - .5) < 1e-12
    print("selftest: PASS (AUC ties, tied AP expectation, IRLS, expit)")


def cohort_summary(panel, scored):
    rates = scored.groupby("pid").y.mean()
    counts = scored.groupby("pid").size()
    mu = float(scored.y.mean()); var_p = float(np.average((rates - mu) ** 2, weights=counts.reindex(rates.index)))
    icc = var_p / (mu * (1 - mu)) if 0 < mu < 1 else None
    nplus = int(scored.y.sum()); nminus = len(scored) - nplus
    withinpairs = sum(int((g.y == 1).sum() * (g.y == 0).sum()) for _, g in scored.groupby("pid"))
    allpairs = nplus * nminus
    gaps = np.concatenate([np.diff(g.day.to_numpy()) for _, g in panel.groupby("pid") if len(g) > 1])
    return {"patients": int(scored.pid.nunique()), "scored_nights": len(scored), "events": nplus,
        "event_rate": mu, "patient_event_rate_mean": float(rates.mean()),
        "patient_event_rate_sd": float(rates.std(ddof=1)), "icc_rate": icc,
        "median_gap_days_between_eligible_nights": float(np.median(gaps)) if len(gaps) else None,
        "within_patient_pair_share": withinpairs / allpairs if allpairs else None,
        "night_below70_rate": float(scored.y_night70.mean()),
        "any_15min_below54_next24h_rate_observed_days": float(scored.y_24h54.mean()),
        "next24h_adequate_coverage_share": float(scored.y_24h54.notna().mean()),
        "median_night_coverage_readings": float(scored.night_coverage.median()),
        "median_predictor_coverage_readings": float(scored.predictor_coverage.median()),
        "pump_bolus_day_available_share": float(scored.insulin_bolus_available.mean()),
        "basal_estimate_day_available_share": float(scored.basal_insulin_available.mean()),
        "carbohydrate_entry_day_available_share": float(scored.carbs_available.mean()),
        "treatment_arm_available": False}


def benchmark(scored):
    g = scored.groupby("pid").agg(n=("y", "size"), rate=("y", "mean"))
    mu = float(scored.y.mean()); n = g.n.to_numpy(float); rates = g.rate.to_numpy(float)
    var_obs = float(np.average((rates - mu) ** 2, weights=n))
    sampling = float(np.sum(n * rates * (1 - rates) / np.maximum(n, 1)) / n.sum())
    var_corr = max(0., var_obs - sampling); sd_obs = math.sqrt(max(0., var_obs)); sd_corr = math.sqrt(var_corr)
    normal = lambda s: .5 * (1 + math.erf(s / (math.sqrt(2) * mu * (1 - mu)))) if 0 < mu < 1 else None
    return {"day_weighted_mean_rate": mu, "observed_between_patient_sd": sd_obs,
        "sampling_variance_estimate": sampling, "sampling_corrected_sd": sd_corr,
        "normal_approx_auc_observed_sd": normal(sd_obs), "normal_approx_auc_corrected_sd": normal(sd_corr),
        "exact_auc_retrospective_rate": auc(scored.y, scored.retro_rate),
        "exact_auc_runin_rate": auc(scored.y, scored.runin_rate),
        "exact_auc_expanding_rate": auc(scored.y, scored.expanding_rate)}


def metric_summary(y, pred, pid):
    per = [(brier(y[ix], pred[ix]), logloss(y[ix], pred[ix])) for _, ix in patient_groups(pid)]
    return {"brier_pooled": brier(y, pred), "brier_patient_mean": float(np.mean([x[0] for x in per])),
        "log_loss_pooled": logloss(y, pred), "log_loss_patient_mean": float(np.mean([x[1] for x in per]))}


def cal_summary(y, pred, pid, nboot=B_CAL, seed=SEED + 500):
    def fit(yy, pp, ppids, eq=False):
        lp = logit(pp); means = pd.Series(lp).groupby(ppids, sort=False).transform("mean").to_numpy(); dev = lp - means
        wt = None
        if eq:
            ns = pd.Series(ppids).value_counts().to_dict(); wt = np.array([1 / ns[k] for k in ppids]); wt *= len(wt) / wt.sum()
        pooled = irls(lp[:, None], yy, wt); joint = irls(np.column_stack([means, dev]), yy, wt)
        return [pooled[0], pooled[1], joint[0], joint[1], joint[2]]
    labels = ["pooled_intercept", "pooled_slope", "joint_intercept", "between_slope", "within_slope"]
    out = {}
    for eq, name in [(False, "day_weighted"), (True, "equal_patient")]:
        point = fit(y, pred, pid, eq); draws = []
        for parts in boot_patient_indices(pid, nboot, seed + int(eq)):
            ix = np.concatenate(parts); bid = np.concatenate([np.full(len(z), j) for j, z in enumerate(parts)])
            try: draws.append(fit(np.asarray(y)[ix], np.asarray(pred)[ix], bid, eq))
            except (np.linalg.LinAlgError, RuntimeError, ValueError): draws.append([np.nan] * 5)
        out[name] = {k: safe_float(v) for k, v in zip(labels, point)}
        out[name]["ci95"] = {k: ci(np.asarray(draws)[:, j]) for j, k in enumerate(labels)}
    lp = logit(pred); means = pd.Series(lp).groupby(pid, sort=False).transform("mean").to_numpy(); dev = lp - means
    var = float(np.var(lp))
    out["forecast_logit_variance_between_share"] = float(np.var(means) / var) if var else None
    out["forecast_logit_variance_within_share"] = float(np.mean(dev ** 2) / var) if var else None
    return out


def markdown_report(result):
    c = result.get("1_cohort", {}); rank = result.get("4_ranking_test", {}); score = result.get("6_scores_test", {})
    cal = result.get("7_calibration_test", {}); hist = result.get("8_history_bias_test", {})
    lines = ["# REPLACE-BG daily hypoglycaemia forecasting analysis", "",
        "Primary event: at least 15 minutes below 54 mg/dL in the 00:00–06:00 night, defined by three consecutive 5-minute CGM readings below threshold. Nights and prior predictor days each require at least 70% of nominal CGM slots. The first 14 eligible nights per patient form the forecasting run-in; patients require at least 60 eligible post-run-in nights.", "",
        "## Cohort", "", "| Quantity | Value |", "|---|---:|"]
    for k, label in [("patients", "Patients"), ("scored_nights", "Scored nights"), ("events", "Primary events"),
        ("event_rate", "Primary event rate"), ("patient_event_rate_mean", "Mean patient event rate"),
        ("patient_event_rate_sd", "SD patient event rate"), ("icc_rate", "Event-rate ICC"),
        ("median_gap_days_between_eligible_nights", "Median eligible-night gap (days)"),
        ("within_patient_pair_share", "Within-patient pair share"), ("night_below70_rate", "Night below 70 rate"),
        ("basal_estimate_day_available_share", "Prior-day basal dose estimate available"),
        ("any_15min_below54_next24h_rate_observed_days", "Next-24h below 54 rate (covered days)")]:
        if k in c: lines.append(f"| {label} | {c[k]} |")
    lines += ["", "## Ranking and benchmark", "", "| Forecast | Pooled AUC (95% CI) | Within-patient AUC (95% CI) | Eligible patients |", "|---|---:|---:|---:|"]
    for name, v in rank.get("forecasts", {}).items():
        lines.append(f"| {name} | {v.get('pooled_auc')} ({v.get('pooled_auc_ci95')}) | {v.get('within_auc')} ({v.get('within_auc_ci95')}) | {v.get('within_auc_patients')} |")
    if "benchmark" in rank: lines.append(f"| benchmark | retrospective {rank['benchmark'].get('exact_auc_retrospective_rate')}; run-in {rank['benchmark'].get('exact_auc_runin_rate')}; expanding {rank['benchmark'].get('exact_auc_expanding_rate')} | — | — |")
    lines += ["", "## Scores and skill", "", "| Forecast | Brier pooled | Brier patient mean | Log-loss pooled | Log-loss patient mean | Skill vs run-in | Skill vs expanding | Skill vs retrospective |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, v in score.get("forecasts", {}).items():
        sv=v.get("brier_skill_vs", {})
        lines.append(f"| {name} | {v.get('brier_pooled')} | {v.get('brier_patient_mean')} | {v.get('log_loss_pooled')} | {v.get('log_loss_patient_mean')} | {sv.get('runin_rate',{}).get('summed_loss_ratio_skill')} | {sv.get('expanding_rate',{}).get('summed_loss_ratio_skill')} | {sv.get('retrospective_rate_reference',{}).get('summed_loss_ratio_skill')} |")
    lines += ["", "## Calibration and history bias", "", "| Analysis | Forecast | Pooled / observed | Between / reference | Within / difference |", "|---|---|---:|---:|---:|"]
    for name, v in cal.items():
        d = v.get("day_weighted", {})
        if d: lines.append(f"| Calibration slopes: {name} | Day weighted | {d.get('pooled_slope')} | {d.get('between_slope')} | {d.get('within_slope')} |")
    if not cal or cal.get("models_unavailable"):
        lines.append("| Calibration slopes | Logistic regression / gradient boosting | unavailable | unavailable | unavailable |")
    for name, v in hist.items(): lines.append(f"| History bias | {name} | {v.get('observed_within_auc')} | {v.get('permutation_reference')} | {v.get('observed_minus_reference')} |")
    model_error = result.get("errors", {}).get("3")
    if model_error:
        lines.append(f"Fitted-model forecasts were unavailable because the model block failed: {model_error}.")
    lines += ["", "Treatment assignment and a randomization day are not available in the inspected HRandomization table, so these results do not estimate treatment effects or separate protocol phases. Bolus and carbohydrate variables indicate pump-record availability; missing entries are not evidence of zero exposure. Basal insulin is estimated from recorded rate multiplied by actual duration, split at midnight; it may be incomplete when pump records are absent.", ""]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def rebuild_history(g, rng=None):
    """Fix run-in outcomes; shuffle scored targets and rebuild on eligible-night rows.

    rng=None selects the identity permutation, used by the identity regression test.
    """
    yy = g.y.to_numpy(int).copy()
    scored = np.flatnonzero(~g.model_runin.to_numpy(bool))
    assert np.array_equal(scored, np.arange(14, len(g)))
    shuffled = yy[scored].copy()
    if rng is not None:
        rng.shuffle(shuffled)
    yy[scored] = shuffled
    scores = {
        "recent_28_night_rate": np.array([yy[max(0,i-28):i].mean() for i in scored]),
        "expanding_rate": np.array([yy[:i].mean() for i in scored]),
        "persistence": yy[scored-1].astype(float),
    }
    return yy[scored], scores


def calculate_history_reference(panel, eligible, seed):
    rng=np.random.default_rng(seed); eligible=eligible.sort_values(["pid","night_index"])
    full=panel[panel.pid.isin(eligible.pid.unique())].sort_values(["pid","night_index"])
    keys=["recent_28_night_rate","expanding_rate","persistence"]
    observed={k:{} for k in keys}; accum={k:{} for k in keys}
    for pid,g in eligible.groupby("pid",sort=False):
        vals={"recent_28_night_rate":g.history_rate_28.to_numpy(float),
            "expanding_rate":g.expanding_rate.to_numpy(float),"persistence":g.persistence.to_numpy(float)}
        yy=g.y.to_numpy(int)
        for k,s in vals.items():
            a=auc(yy,s)
            if a is not None: observed[k][pid]=a; accum[k][pid]=[]
    for _ in range(N_PERM):
        for pid,g in full.groupby("pid",sort=False):
            target, scores = rebuild_history(g, rng)
            recent=scores["recent_28_night_rate"]
            expand=scores["expanding_rate"]
            pers=scores["persistence"]
            for k,s in [("recent_28_night_rate",recent),("expanding_rate",expand),("persistence",pers)]:
                a=auc(target,s)
                if a is not None and pid in accum[k]: accum[k][pid].append(a)
    result={}
    for k,per in observed.items():
        common=[p for p in per if accum[k].get(p)]
        dif={p:per[p]-float(np.mean(accum[k][p])) for p in common}
        vals=np.array(list(dif.values()),float); draws=[]; r=np.random.default_rng(seed+10)
        for _ in range(B_RANK): draws.append(float(np.mean([vals[i] for i in r.integers(0,len(vals),len(vals))])) if len(vals) else np.nan)
        result[k]={"permutations":N_PERM,"patients":len(common),
            "observed_within_auc":float(np.mean([per[p] for p in common])) if common else None,
            "permutation_reference":float(np.mean([np.mean(accum[k][p]) for p in common])) if common else None,
            "observed_minus_reference":float(np.mean(vals)) if len(vals) else None,"difference_ci95":ci(draws)}
    return result

def run():
    out = {"metadata": {"seed": SEED, "panel": str(PANEL.name), "day_time_source": "DeviceDtTmDaysFromEnroll and DeviceTm; day offset is from enrollment", "primary_event": "15 consecutive minutes CGM <54 mg/dL, 00:00-06:00", "coverage_rule": ">=70% of nominal 5-minute slots for target night and prior predictor day", "model_runin_nights": 14, "minimum_post_runin_nights": 60}}
    errors = {}
    def block(n, fn):
        try: out.update(fn())
        except Exception as e:
            errors[str(n)] = f"{type(e).__name__}: {e}"
            print(f"block {n} failed: {type(e).__name__}: {e}")
    panel = pd.read_csv(PANEL)
    panel["pid"] = panel.pid.astype(str)
    panel["y"] = panel.y.astype(int)
    panel["persistence"] = panel.groupby("pid").y.shift(1)
    scored_all = panel.loc[~panel.model_runin.astype(bool)].copy()

    # 1. Cohort description and data adequacy.
    block(1, lambda: {"1_cohort": cohort_summary(panel, scored_all)})

    # 2. Seeded patient-level 60/20/20 split.
    def split_block():
        ids = np.array(sorted(scored_all.pid.unique()), dtype=object); rng = np.random.default_rng(SEED); rng.shuffle(ids)
        ntr = int(round(.60 * len(ids))); nva = int(round(.20 * len(ids)))
        tr, va, te = ids[:ntr], ids[ntr:ntr+nva], ids[ntr+nva:]
        return {"2_split": {"seed": SEED, "train_patients": len(tr), "validation_patients": len(va), "test_patients": len(te)}}, tr, va, te
    try: split_result, train_ids, val_ids, test_ids = split_block(); out.update(split_result)
    except Exception as e:
        errors["2"] = f"{type(e).__name__}: {e}"; print(f"block 2 failed: {errors['2']}"); train_ids=val_ids=test_ids=np.array([], object)

    # 3. Model forecasts. The modeling script is intended for the editor's full Python install.
    models = {}
    def model_block():
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import log_loss as sk_log_loss
        daily_features = ["mean", "sd", "cv", "time_below70", "time_below54", "time_above180", "minimum", "last_value", "late_slope", "insulin_bolus", "basal_insulin_est", "carbs"]
        base_features = daily_features + ["history_rate_7", "history_rate_28", "expanding_rate", "runin_rate", "day"]
        d = scored_all.copy(); runrows = panel.loc[panel.model_runin.astype(bool)]
        d["day"] = pd.to_numeric(d.day, errors="coerce")
        for col in ["insulin_bolus", "basal_insulin_est", "carbs"]: d[col + "_missing"] = d[col].isna().astype(float); base_features.append(col + "_missing")
        d["persistence"] = np.nan
        for pid, gg in panel.groupby("pid", sort=False):
            ix = gg.index.to_numpy(); yy = gg.y.to_numpy(float); vals = np.r_[np.nan, yy[:-1]]
            d.loc[d.pid.eq(pid), "persistence"] = vals[14:]
        d["recent_rate"] = d.history_rate_28
        runmean = runrows.groupby("pid")[daily_features].mean()
        centred_features = daily_features
        for col in centred_features:
            d[col + "_dev"] = d[col] - d.pid.map(runmean[col])
        centered = [c + "_dev" for c in centred_features]
        tr = d[d.pid.isin(train_ids)]; va = d[d.pid.isin(val_ids)]; te = d[d.pid.isin(test_ids)]
        if tr.y.nunique() < 2: raise ValueError("training cohort has only one event class")
        def matrix(train, frame, cols):
            a = train[cols].apply(pd.to_numeric, errors="coerce"); b = frame[cols].apply(pd.to_numeric, errors="coerce")
            med = a.median().fillna(0); aa = a.fillna(med).to_numpy(float); bb = b.fillna(med).to_numpy(float)
            # Include missingness indicators for every input; simple train-only imputation.
            am = a.isna().to_numpy(float); bm = b.isna().to_numpy(float)
            mu = aa.mean(axis=0); sd = aa.std(axis=0); sd[sd < 1e-8] = 1
            return np.c_[(aa-mu)/sd, am], np.c_[(bb-mu)/sd, bm]
        Xtr, Xte = matrix(tr, te, base_features); _, Xva = matrix(tr, va, base_features)
        lr = LogisticRegression(C=1., max_iter=1000, solver="lbfgs", random_state=SEED)
        lr.fit(Xtr, tr.y); plr = lr.predict_proba(Xte)[:, 1]; plrv = lr.predict_proba(Xva)[:, 1]
        # Choose boosting rounds on validation log-loss from the fixed candidate set.
        best = None; curves = []
        for n_est in (50, 100, 200, 300):
            gb = GradientBoostingClassifier(n_estimators=n_est, max_depth=2, learning_rate=.05,
                min_samples_leaf=20, subsample=.8, random_state=SEED)
            gb.fit(Xtr, tr.y); pv = gb.predict_proba(Xva)[:, 1]
            loss = float(sk_log_loss(va.y, np.clip(pv, EPS, 1-EPS), labels=[0,1]))
            curves.append({"n_estimators": n_est, "validation_log_loss": loss})
            if best is None or loss < best[0]: best = (loss, n_est)
        gb = GradientBoostingClassifier(n_estimators=best[1], max_depth=2, learning_rate=.05,
            min_samples_leaf=20, subsample=.8, random_state=SEED).fit(Xtr, tr.y)
        pgb = gb.predict_proba(Xte)[:, 1]
        pgbv = gb.predict_proba(Xva)[:, 1]
        # Within-patient-centred GB on run-in-centred daily CGM and device predictors only.
        Xtrc, Xtec = matrix(tr, te, centered); _, Xvac = matrix(tr, va, centered)
        gbc = GradientBoostingClassifier(n_estimators=best[1], max_depth=2, learning_rate=.05,
            min_samples_leaf=20, subsample=.8, random_state=SEED).fit(Xtrc, tr.y)
        pc = gbc.predict_proba(Xtec)[:, 1]
        pcv = gbc.predict_proba(Xvac)[:, 1]
        train_rate = float(tr.y.mean())
        te = te.copy(); te["persistence"] = te.persistence.fillna(train_rate)
        retro = te.groupby("pid").y.transform("mean").to_numpy()
        rate_shr = (14 * te.runin_rate.to_numpy(float) + 10 * train_rate) / 24
        forecasts = {"training_cohort_rate": np.full(len(te), train_rate), "runin_rate": te.runin_rate.to_numpy(float),
            "runin_rate_shrunk_10": rate_shr, "expanding_rate": te.expanding_rate.to_numpy(float),
            "recent_28_night_rate": te.history_rate_28.to_numpy(float), "persistence": te.persistence.to_numpy(float),
            "retrospective_rate_reference": retro, "logistic_regression": plr, "gradient_boosting": pgb,
            "within_patient_centred_gradient_boosting": pc}
        va = va.copy(); va["persistence"] = va.persistence.fillna(train_rate)
        validation_forecasts = {"training_cohort_rate": np.full(len(va), train_rate),
            "runin_rate": va.runin_rate.to_numpy(float),
            "runin_rate_shrunk_10": (14 * va.runin_rate.to_numpy(float) + 10 * train_rate) / 24,
            "expanding_rate": va.expanding_rate.to_numpy(float),
            "recent_28_night_rate": va.history_rate_28.to_numpy(float),
            "persistence": va.persistence.to_numpy(float),
            "retrospective_rate_reference": va.groupby("pid").y.transform("mean").to_numpy(float),
            "logistic_regression": plrv, "gradient_boosting": pgbv,
            "within_patient_centred_gradient_boosting": pcv}
        return {"3_models_test": {"selected_boosting_estimators": best[1], "validation_log_loss_grid": curves,
            "train_event_rate": train_rate, "features": base_features,
            "model_fit_patients": {"train": int(tr.pid.nunique()), "validation": int(va.pid.nunique()), "test": int(te.pid.nunique())}}}, forecasts, te, va, validation_forecasts
    try:
        model_meta, models, test, validation, validation_models = model_block(); out.update(model_meta)
    except Exception as e:
        errors["3"] = f"{type(e).__name__}: {e}"; print(f"block 3 failed: {errors['3']}")
        test = scored_all[scored_all.pid.isin(test_ids)].copy()
        validation = scored_all[scored_all.pid.isin(val_ids)].copy()
        train_rows = scored_all[scored_all.pid.isin(train_ids)]
        train_rate = float(train_rows.y.mean()) if len(train_rows) else float(scored_all.y.mean())
        test["retro_rate"] = test.groupby("pid").y.transform("mean")
        test["persistence"] = test.persistence.fillna(train_rate)
        models = {"training_cohort_rate": np.full(len(test), train_rate),
            "runin_rate": test.runin_rate.to_numpy(float),
            "runin_rate_shrunk_10": (14 * test.runin_rate.to_numpy(float) + 10 * train_rate) / 24,
            "expanding_rate": test.expanding_rate.to_numpy(float),
            "recent_28_night_rate": test.history_rate_28.to_numpy(float),
            "persistence": test.persistence.to_numpy(float),
            "retrospective_rate_reference": test.retro_rate.to_numpy(float)}
        validation["persistence"] = validation.persistence.fillna(train_rate)
        validation_models = {"training_cohort_rate": np.full(len(validation), train_rate),
            "runin_rate": validation.runin_rate.to_numpy(float),
            "runin_rate_shrunk_10": (14 * validation.runin_rate.to_numpy(float) + 10 * train_rate) / 24,
            "expanding_rate": validation.expanding_rate.to_numpy(float),
            "recent_28_night_rate": validation.history_rate_28.to_numpy(float),
            "persistence": validation.persistence.to_numpy(float),
            "retrospective_rate_reference": validation.groupby("pid").y.transform("mean").to_numpy(float)}
        out["3_models_test"] = {"models_unavailable": True, "reason": errors["3"],
            "baseline_forecasts_available": list(models), "train_event_rate": train_rate}

    # Export held-out observations with anonymous sequential patient labels.
    forecast_frame = pd.DataFrame({"_pid": test.pid.to_numpy(), "_time": test.day.to_numpy(), "y": test.y.to_numpy(int)})
    for name, values in models.items(): forecast_frame[name] = np.asarray(values, float)
    forecast_frame = forecast_frame.sort_values(["_pid", "_time"], kind="mergesort").reset_index(drop=True)
    forecast_ids = pd.unique(forecast_frame._pid)
    forecast_frame.insert(0, "patient", forecast_frame._pid.map({value: i for i, value in enumerate(forecast_ids)}).astype(int))
    forecast_frame.insert(1, "t", forecast_frame.groupby("_pid", sort=False).cumcount())
    forecast_frame.drop(columns=["_pid", "_time"], inplace=True)
    forecast_frame["threshold"] = float((out.get("3_models_test") or out.get("3_models_test_unavailable") or {}).get("train_event_rate", float("nan")))
    forecast_frame.to_csv(ROOT / "forecasts_replace_bg.csv", index=False)
    validation_frame = pd.DataFrame({"_pid": validation.pid.to_numpy(), "_time": validation.day.to_numpy(), "y": validation.y.to_numpy(int)})
    for name, values in validation_models.items(): validation_frame[name] = np.asarray(values, float)
    validation_frame = validation_frame.sort_values(["_pid", "_time"], kind="mergesort").reset_index(drop=True)
    validation_ids = pd.unique(validation_frame._pid)
    validation_frame.insert(0, "patient", validation_frame._pid.map({value: i for i, value in enumerate(validation_ids)}).astype(int))
    validation_frame.insert(1, "t", validation_frame.groupby("_pid", sort=False).cumcount())
    validation_frame.drop(columns=["_pid", "_time"], inplace=True)
    validation_frame["threshold"] = float((out.get("3_models_test") or out.get("3_models_test_unavailable") or {}).get("train_event_rate", float("nan")))
    validation_frame.to_csv(ROOT / "forecasts_val_replace_bg.csv", index=False)

    # 4. Pooled and within-patient ranking, person bootstrap, and base-rate benchmark.
    def rank_block():
        y = test.y.to_numpy(int); pid = test.pid.to_numpy(); result = {}; boot_seed = SEED + 40
        groups = [ix for _, ix in patient_groups(pid)]
        for name, p in models.items():
            wp, wn = within_auc(y, p, pid)
            pooled = auc(y, p)
            person_auc = [auc(y[ix], p[ix]) for ix in groups]
            rng = np.random.default_rng(boot_seed)
            draws_p=[]; draws_w=[]; draws_d=[]
            for _ in range(B_RANK):
                chosen=rng.integers(0,len(groups),size=len(groups))
                ix=np.concatenate([groups[j] for j in chosen])
                ap=auc(y[ix],p[ix]); eligible=[person_auc[j] for j in chosen if person_auc[j] is not None]
                aw=float(np.mean(eligible)) if eligible else None
                if ap is not None and aw is not None: draws_p.append(ap); draws_w.append(aw); draws_d.append(ap-aw)
            result[name]={"pooled_auc":pooled,"pooled_auc_ci95":ci(draws_p),"within_auc":wp,
                "within_auc_ci95":ci(draws_w),"within_auc_patients":wn,"pooled_minus_within":pooled-wp if pooled is not None and wp is not None else None,
                "pooled_minus_within_ci95":ci(draws_d)}
        bench_df=test.copy(); bench_df["retro_rate"] = bench_df.groupby("pid").y.transform("mean")
        bench=benchmark(bench_df)
        # Also give model-free estimates on every included patient.
        allsc=scored_all.copy(); allsc["retro_rate"]=allsc.groupby("pid").y.transform("mean")
        bench_all=benchmark(allsc)
        tr_rate=float(scored_all.y.mean())
        all_preds={"training_cohort_rate":np.full(len(allsc),tr_rate),"runin_rate":allsc.runin_rate.to_numpy(float),
            "runin_rate_shrunk_10":(14*allsc.runin_rate.to_numpy(float)+10*tr_rate)/24,
            "expanding_rate":allsc.expanding_rate.to_numpy(float),"recent_28_night_rate":allsc.history_rate_28.to_numpy(float),
            "persistence":allsc.persistence.fillna(tr_rate).to_numpy(float),"retrospective_rate_reference":allsc.retro_rate.to_numpy(float)}
        all_rank={}
        for name,p in all_preds.items():
            aw,nw=within_auc(allsc.y.to_numpy(int),p,allsc.pid.to_numpy())
            all_rank[name]={"pooled_auc":auc(allsc.y,p),"within_auc":aw,"within_auc_patients":nw}
        return {"4_ranking_test":{"forecasts":result,"benchmark":bench,"benchmark_all_included":bench_all},
                "4_ranking_all_included_model_free":{"forecasts":all_rank}}
    block(4, rank_block)

    # 5. Threshold sensitivity/specificity at the training-cohort event rate.
    def threshold_block():
        thr=float(out["3_models_test"]["train_event_rate"]); y=test.y.to_numpy(int); pid=test.pid.to_numpy(); q={}
        for name,p in models.items():
            yh=np.asarray(p)>=thr; sens=[]; spec=[]
            for _,ix in patient_groups(pid):
                yy=y[ix]; hh=yh[ix]
                if yy.sum(): sens.append(float(hh[yy==1].mean()))
                if (yy==0).sum(): spec.append(float((~hh[yy==0]).mean()))
            q[name]={"pooled_sensitivity":float(yh[y==1].mean()) if y.sum() else None,
                "mean_patient_sensitivity":float(np.mean(sens)) if sens else None,"sensitivity_eligible_patients":len(sens),
                "pooled_specificity":float((~yh[y==0]).mean()) if (y==0).sum() else None,
                "mean_patient_specificity":float(np.mean(spec)) if spec else None,"specificity_eligible_patients":len(spec)}
        allres={}
        # Model-free quantities for all included patients, using cohort event rate threshold.
        ya=scored_all.y.to_numpy(int); pa=scored_all.runin_rate.to_numpy(float); pida=scored_all.pid.to_numpy()
        yh=pa>=thr; sens=[]; spec=[]
        for _,ix in patient_groups(pida):
            yy=ya[ix]; hh=yh[ix]
            if yy.sum(): sens.append(float(hh[yy==1].mean()))
            if (yy==0).sum(): spec.append(float((~hh[yy==0]).mean()))
        allres["runin_rate"]={"pooled_sensitivity":float(yh[ya==1].mean()),"mean_patient_sensitivity":float(np.mean(sens)),
            "sensitivity_eligible_patients":len(sens),"pooled_specificity":float((~yh[ya==0]).mean()),
            "mean_patient_specificity":float(np.mean(spec)),"specificity_eligible_patients":len(spec)}
        return {"5_threshold_test":{"threshold":thr,"forecasts":q,"all_included_runin_rate":allres}}
    block(5, threshold_block)

    # 6. Scores and paired Brier skill against every reference forecaster.
    def score_block():
        y=test.y.to_numpy(int); pid=test.pid.to_numpy(); results={}; refs=[x for x in ["training_cohort_rate","runin_rate","runin_rate_shrunk_10","expanding_rate","recent_28_night_rate","persistence","retrospective_rate_reference"] if x in models]
        pg=patient_groups(pid); rng=np.random.default_rng(SEED+606)
        boot_draws=rng.integers(0,len(pg),size=(B_RANK,len(pg)))
        for name,p in models.items():
            results[name]=metric_summary(y,p,pid)
            results[name]["brier_skill_vs"]={}
            for refn in refs:
                ref=models[refn]
                point=1-float(np.sum((p-y)**2)/np.sum((ref-y)**2)) if np.sum((ref-y)**2)>0 else None
                patdiff=float(np.mean([brier(y[ix],ref[ix])-brier(y[ix],p[ix]) for _,ix in patient_groups(pid)]))
                numerator=[]; denominator=[]; improvement=[]
                for _,ix in pg:
                    numerator.append(float(np.sum((p[ix]-y[ix])**2)))
                    denominator.append(float(np.sum((ref[ix]-y[ix])**2)))
                    improvement.append(float(brier(y[ix],ref[ix])-brier(y[ix],p[ix])))
                num=np.asarray(numerator); den=np.asarray(denominator); imp=np.asarray(improvement)
                bnum=num[boot_draws].sum(axis=1); bden=den[boot_draws].sum(axis=1)
                vals=np.divide(bnum,bden,out=np.full(B_RANK,np.nan),where=bden>0); vals=1-vals
                pdiff=imp[boot_draws].mean(axis=1)
                results[name]["brier_skill_vs"][refn]={"summed_loss_ratio_skill":point,"summed_loss_ratio_skill_ci95":ci(vals),
                    "patient_mean_brier_improvement":patdiff,"patient_mean_brier_improvement_ci95":ci(pdiff)}
        # Retrospective-rate expected Brier optimism (day weighted), approximate plug-in value.
        rg=test.groupby("pid").agg(n=("y","size"),rate=("y","mean")); optimism=float(np.sum(rg.rate*(1-rg.rate))/len(test))
        return {"6_scores_test":{"forecasts":results,"retrospective_rate_day_weighted_expected_brier_optimism":optimism}}
    block(6, score_block)

    # 7. Pooled, between, and within calibration slopes/intercepts.
    def calibration_block():
        y=test.y.to_numpy(float); pid=test.pid.to_numpy(); outcal={}
        for name in ("logistic_regression","gradient_boosting"):
            if name in models: outcal[name]=cal_summary(y,np.asarray(models[name]),pid)
        if not outcal: outcal["unavailable"]={"reason":"scikit-learn unavailable; model forecasts were not fitted"}
        return {"7_calibration_test":outcal}
    block(7, calibration_block)

    # 8. Within-patient history bias against order-permutation references.
    def history_block():
        return {"8_history_bias_test":calculate_history_reference(panel,test,SEED+800),
                "8_history_bias_all_included":calculate_history_reference(panel,scored_all,SEED+900)}
    block(8, history_block)

    # 9. Within-patient average precision, with exact tie-matched expectation.
    def ap_block():
        outap={}
        for name, p in [("gradient_boosting",models.get("gradient_boosting")),("runin_rate",models.get("runin_rate"))]:
            if p is None: continue
            y=test.y.to_numpy(int); pid=test.pid.to_numpy(); vals=[]; exps=[]; prev=[]
            for _,ix in patient_groups(pid):
                a=ap_tied(y[ix],np.asarray(p)[ix]); e=ap_tie_expectation(y[ix],np.asarray(p)[ix])
                if a is not None and e is not None: vals.append(a); exps.append(e); prev.append(float(y[ix].mean()))
            outap[name]={"eligible_patients":len(vals),"mean_average_precision":float(np.mean(vals)) if vals else None,
                "mean_prevalence":float(np.mean(prev)) if prev else None,"mean_tie_matched_expectation":float(np.mean(exps)) if exps else None,
                "mean_ap_minus_expectation":float(np.mean(np.asarray(vals)-np.asarray(exps))) if vals else None}
        return {"9_average_precision_test":outap}
    block(9, ap_block)

    # 10. Compact printout and persistent machine/human-readable outputs.
    block(10, lambda: {"10_output":{"json":JSON_OUT.name,"markdown":MD_OUT.name}})
    out["errors"] = errors
    # Convert numpy values and NaN to strict JSON-safe values.
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k,v in x.items()}
        if isinstance(x, (list,tuple)): return [clean(v) for v in x]
        if isinstance(x, np.ndarray): return clean(x.tolist())
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,float)): return safe_float(x)
        return x
    out=clean(out)
    JSON_OUT.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
    markdown_report(out)
    print("REPLACE-BG analysis summary")
    for k,v in out.get("1_cohort",{}).items(): print(f"{k}: {v}")
    print("split:", {k:v for k,v in out.get("2_split",{}).items() if k.endswith("patients")})
    for name, v in out.get("4_ranking_test", {}).get("forecasts", {}).items():
        print(f"AUC {name}: pooled={v.get('pooled_auc')} within={v.get('within_auc')}")
    for name, v in out.get("6_scores_test", {}).get("forecasts", {}).items():
        print(f"Brier {name}: pooled={v.get('brier_pooled')} patient_mean={v.get('brier_patient_mean')}")
    if errors: print("block_errors:", errors)
    print(f"Wrote {JSON_OUT.name} and {MD_OUT.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true", help="run metric and IRLS smoke tests")
    args = parser.parse_args()
    if args.selftest: selftest()
    else: run()
