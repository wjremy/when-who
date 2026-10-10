#!/usr/bin/env python3
"""Model-fixed, within-patient outcome permutation reference for model AUCs.

The cohort splits, loaders, seeds, source feature functions, and estimator
settings follow cases_extra.py and its three source analyses.  The source fit
functions return forecasts but not fitted estimators, so this script performs
the same single fit while retaining each estimator for the permutations.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import cases_extra as CASES  # noqa: E402

REPLACE = CASES.REPLACE
CTN = CASES.CTN
CES = CASES.CES
SEED = 20261009
N_PERM = 200
N_BOOT = 2000
EPS = 1e-6
OUTPUT_DIR = ROOT / "results" / "model_fixed_permutation"

COHORTS = {
    "replace_bg": {
        "label": "REPLACE-BG",
        "existing_dataset": "replace_bg",
        "table4": {"median_auc": 0.78, "share_gt_0_5": 0.98, "mean_auc": 0.76},
    },
    "ctn0094": {
        "label": "CTN-0094 weekly",
        "existing_dataset": "weekly_missing_excluded",
        "table4": {"median_auc": 0.69, "share_gt_0_5": 0.66, "mean_auc": 0.64},  # median 0.68 before the review-2 CTN fix (10 Oct)
    },
    "ces": {
        "label": "College Experience",
        "existing_dataset": "ces",
        "table4": {"median_auc": 0.82, "share_gt_0_5": 0.95, "mean_auc": 0.80},
    },
}


def group_indices(pid):
    """Return stable patient groups as (identifier, row indices)."""
    pid = np.asarray(pid)
    return [(key, np.flatnonzero(pid == key)) for key in pd.unique(pid)]


def auc(y, score):
    """Use the tie-aware AUC implementation used by cases_extra.py."""
    value = CASES.auc(np.asarray(y, dtype=int), np.asarray(score, dtype=float))
    return float(value) if value is not None and np.isfinite(value) else None


def per_patient_auc(y, score, pid):
    values = []
    ids = []
    for key, ix in group_indices(pid):
        value = auc(np.asarray(y)[ix], np.asarray(score)[ix])
        if value is not None:
            ids.append(key)
            values.append(value)
    return ids, np.asarray(values, dtype=float)


def generate_permuted_targets(y, pid, n_perm, rng):
    """Shuffle each patient's scored targets independently, preserving counts."""
    y = np.asarray(y, dtype=int)
    out = np.empty((n_perm, len(y)), dtype=int)
    for _, ix in group_indices(pid):
        for b in range(n_perm):
            out[b, ix] = rng.permutation(y[ix])
    return out


def feature_auc_matrix(y_observed, y_perm, pid, predictions_by_model, score_permutation):
    """Rebuild histories, rescore fixed models, and collect patient AUCs."""
    models = {}
    for name, prediction in predictions_by_model.items():
        ids, observed = per_patient_auc(y_observed, prediction, pid)
        if not len(ids):
            raise AssertionError(f"No eligible patients for model {name}")
        models[name] = {
            "observed_ids": ids,
            "observed_auc": observed,
            "permutation_auc": np.empty((len(y_perm), len(ids)), dtype=float),
        }
    for b, yb in enumerate(y_perm):
        rescored = score_permutation(yb)
        for name, prediction in rescored.items():
            perm_ids, values = per_patient_auc(yb, prediction, pid)
            if perm_ids != models[name]["observed_ids"]:
                raise AssertionError("Outcome permutation changed AUC eligibility")
            models[name]["permutation_auc"][b] = values
    return models


def percentile_interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return [None, None]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def summarize_reference(observed_auc, permutation_auc, n_boot, rng):
    """Summarize permutation references and paired patient-bootstrap deltas.

    Bootstrap contrasts resample whole patients and preserve each
    observed/reference pairing. The median reference is recomputed within each
    resample as the mean of its permutation-level medians; the mean reference
    uses the already-computed per-patient mean permutation AUCs.
    """
    observed_auc = np.asarray(observed_auc, dtype=float)
    permutation_auc = np.asarray(permutation_auc, dtype=float)
    if permutation_auc.ndim != 2 or permutation_auc.shape[1] != len(observed_auc):
        raise ValueError("Permutation AUC matrix does not match eligible patients")
    per_patient_reference = np.mean(permutation_auc, axis=0)
    permutation_medians = np.median(permutation_auc, axis=1)
    permutation_means = np.mean(permutation_auc, axis=1)
    permutation_shares = np.mean(permutation_auc > 0.5, axis=1)

    observed_median = float(np.median(observed_auc))
    observed_mean = float(np.mean(observed_auc))
    observed_share = float(np.mean(observed_auc > 0.5))
    ref_patient_median = float(np.median(per_patient_reference))
    ref_patient_mean = float(np.mean(per_patient_reference))

    boot_median = np.empty(n_boot, dtype=float)
    boot_median_patient_reference = np.empty(n_boot, dtype=float)
    boot_mean = np.empty(n_boot, dtype=float)
    n_patient = len(observed_auc)
    for b in range(n_boot):
        ix = rng.integers(0, n_patient, size=n_patient)
        sampled_permutation_medians = np.median(permutation_auc[:, ix], axis=1)
        boot_median[b] = np.median(observed_auc[ix]) - np.mean(sampled_permutation_medians)
        boot_median_patient_reference[b] = (
            np.median(observed_auc[ix]) - np.median(per_patient_reference[ix])
        )
        boot_mean[b] = np.mean(observed_auc[ix]) - np.mean(per_patient_reference[ix])

    return {
        "eligible_patients": int(n_patient),
        "observed": {
            "median_auc": observed_median,
            "mean_per_patient_auc": observed_mean,
            "share_above_0_5": observed_share,
        },
        "permutation_reference": {
            "permutations": int(len(permutation_auc)),
            "median_auc_mean_over_permutations": float(np.mean(permutation_medians)),
            "median_auc_permutation_percentiles_2_5_97_5": percentile_interval(permutation_medians),
            "mean_per_patient_auc_mean_over_permutations": float(np.mean(permutation_means)),
            "mean_per_patient_auc_permutation_percentiles_2_5_97_5": percentile_interval(permutation_means),
            "share_above_0_5_mean_over_permutations": float(np.mean(permutation_shares)),
            "share_above_0_5_permutation_percentiles_2_5_97_5": percentile_interval(permutation_shares),
            "median_of_per_patient_permutation_mean_auc": ref_patient_median,
            "mean_of_per_patient_permutation_mean_auc": ref_patient_mean,
            "per_patient_mean_auc_summary": {
                "mean": ref_patient_mean,
                "median": ref_patient_median,
                "percentiles_2_5_97_5": percentile_interval(per_patient_reference),
            },
        },
        "observed_minus_reference": {
            "median_auc": float(observed_median - np.mean(permutation_medians)),
            "median_auc_bootstrap_ci95": percentile_interval(boot_median),
            "median_auc_patient_permutation_mean_reference": float(observed_median - ref_patient_median),
            "median_auc_patient_permutation_mean_reference_bootstrap_ci95": percentile_interval(
                boot_median_patient_reference
            ),
            "mean_per_patient_auc": float(observed_mean - ref_patient_mean),
            "mean_per_patient_auc_bootstrap_ci95": percentile_interval(boot_mean),
        },
        "patient_bootstrap_replicates": int(n_boot),
    }


def replace_matrix(train, frame, cols, fitted=None):
    """cases_extra's train-only impute, scale, and missingness-indicator matrix."""
    a = train[cols].apply(pd.to_numeric, errors="coerce") if fitted is None else None
    b = frame[cols].apply(pd.to_numeric, errors="coerce")
    if fitted is None:
        med = a.median().fillna(0)
        aa = a.fillna(med).to_numpy(float)
        mu = aa.mean(axis=0)
        sd = aa.std(axis=0)
        sd[sd < 1e-8] = 1
        fitted = {"median": med, "mean": mu, "sd": sd}
    bb = b.fillna(fitted["median"]).to_numpy(float)
    bm = b.isna().to_numpy(float)
    return np.c_[(bb - fitted["mean"]) / fitted["sd"], bm], fitted


def replace_build_histories(panel, target, test):
    """Rebuild scored REPLACE-BG histories from fixed run-in plus shuffled targets."""
    changed = test.copy()
    changed["y"] = np.asarray(target, dtype=int)
    for pid, rows in changed.groupby("pid", sort=False):
        full = panel.loc[panel.pid.eq(pid)].sort_values("night_index", kind="mergesort")
        yy = full.y.to_numpy(int).copy()
        scored_ix = full.index[~full.model_runin.astype(bool)]
        # Every scored night for a test patient is in the test split.
        local = changed.loc[changed.pid.eq(pid)].sort_values("night_index", kind="mergesort")
        if len(local) != len(scored_ix):
            raise AssertionError("REPLACE-BG test patients have unexpected partial histories")
        yy[14:] = local.y.to_numpy(int)
        target_rows = local.index
        positions = full.index.get_indexer(target_rows)
        for w in (7, 28):
            changed.loc[target_rows, f"history_rate_{w}"] = [
                float(yy[max(0, j - w):j].mean()) for j in positions
            ]
        changed.loc[target_rows, "expanding_rate"] = [
            float(yy[:j].mean()) for j in positions
        ]
    return changed


def fit_replace_bg():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss as sk_log_loss

    panel = pd.read_csv(ROOT / "replace_bg_panel.csv")
    panel["pid"] = panel.pid.astype(str)
    panel["y"] = panel.y.astype(int)
    scored = panel.loc[~panel.model_runin.astype(bool)].copy()
    train_ids, val_ids, test_ids = CASES.split_replace(scored)
    daily = ["mean", "sd", "cv", "time_below70", "time_below54", "time_above180", "minimum",
             "last_value", "late_slope", "insulin_bolus", "basal_insulin_est", "carbs"]
    features = daily + ["history_rate_7", "history_rate_28", "expanding_rate", "runin_rate", "day"]
    d = scored.copy()
    d["day"] = pd.to_numeric(d.day, errors="coerce")
    for col in ["insulin_bolus", "basal_insulin_est", "carbs"]:
        d[col + "_missing"] = d[col].isna().astype(float)
        features.append(col + "_missing")
    tr, va, te = (d[d.pid.isin(ids)].copy() for ids in (train_ids, val_ids, test_ids))
    xtr, transform = replace_matrix(tr, tr, features)
    xva, _ = replace_matrix(tr, va, features, transform)
    xte, _ = replace_matrix(tr, te, features, transform)

    lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=CASES.SEED_REPLACE)
    lr.fit(xtr, tr.y)
    predictions = {"logistic_regression": lr.predict_proba(xte)[:, 1]}

    best = None
    for n_est in (50, 100, 200, 300):
        model = GradientBoostingClassifier(n_estimators=n_est, max_depth=2, learning_rate=0.05,
            min_samples_leaf=20, subsample=0.8, random_state=CASES.SEED_REPLACE)
        model.fit(xtr, tr.y)
        pv = model.predict_proba(xva)[:, 1]
        loss = float(sk_log_loss(va.y, np.clip(pv, EPS, 1 - EPS), labels=[0, 1]))
        if best is None or loss < best[0]:
            best = (loss, n_est)
    gb = GradientBoostingClassifier(n_estimators=best[1], max_depth=2, learning_rate=0.05,
        min_samples_leaf=20, subsample=0.8, random_state=CASES.SEED_REPLACE)
    gb.fit(xtr, tr.y)
    predictions["gradient_boosting"] = gb.predict_proba(xte)[:, 1]

    def score_permutation(yb):
        permuted = replace_build_histories(panel, yb, te)
        xt, _ = replace_matrix(tr, permuted, features, transform)
        return {"logistic_regression": lr.predict_proba(xt)[:, 1],
                "gradient_boosting": gb.predict_proba(xt)[:, 1]}

    return {"test": te, "y": te.y.to_numpy(int), "pid": te.pid.to_numpy(),
            "predictions": predictions, "score_permutation": score_permutation,
            "selected_gb_iterations": int(best[1])}


def ctn_rebuild_histories(data, test, target, event_col):
    """Rebuild CTN history columns while preserving calendar, screening and covariates."""
    changed = data.copy()
    target = np.asarray(target, dtype=int)
    changed.loc[test.index, event_col] = target
    test_ids = set(test.who.unique())
    for pid in test_ids:
        rows = changed.loc[changed.who.eq(pid)].sort_values("study_week", kind="mergesort")
        first = rows.iloc[0]
        initial_events = json.loads(first.history_initial_events)
        week_event = {}
        for week, event in zip(range(4), initial_events):
            if event is not None:
                week_event[week] = int(event)
        scored = rows.loc[rows.study_week.ge(4) & rows.observed.eq(1)]
        for row in scored.itertuples():
            week = int(row.study_week)
            week_event[week] = int(changed.at[row.Index, event_col])

        for row in rows.itertuples():
            week = int(row.study_week)
            origin = week - 1
            prior = [w for w in week_event if w <= origin]
            prior_weeks = len(prior)
            # Match prep_ctn0094_panels.py: positive weeks divided by observed weeks.
            prior_events = sum(week_event[w] for w in prior)
            changed.at[row.Index, "expanding_rate"] = (
                prior_events / prior_weeks if prior_weeks else np.nan
            )
            recent_weeks = [w for w in prior if w >= max(0, origin - 3)]
            changed.at[row.Index, "recent_rate"] = (
                float(np.mean([week_event[w] for w in recent_weeks])) if recent_weeks else np.nan
            )
            changed.at[row.Index, "previous_result"] = (
                week_event[origin] if origin in week_event else np.nan
            )
    return changed


def ces_rebuild_histories(test, target):
    """Rebuild CES report-history rates from fixed run-in/origin plus shuffled targets."""
    changed = test.copy()
    changed["y"] = np.asarray(target, dtype=int)
    for _, row_ix in group_indices(changed.pid.to_numpy()):
        group = changed.iloc[row_ix].sort_values("date", kind="mergesort")
        runin = np.asarray(group.runin_event_history.iloc[0], dtype=int)
        first_origin = int(group.current_event.iloc[0])
        perm_targets = group.y.to_numpy(int)
        history = list(runin) + [first_origin]
        for j, idx in enumerate(group.index.to_numpy()):
            h = np.asarray(history, dtype=float)
            changed.at[idx, "expanding_rate"] = float(h.mean())
            changed.at[idx, "last4_rate"] = float(h[-4:].mean())
            changed.at[idx, "last12_rate"] = float(h[-12:].mean())
            history.append(int(perm_targets[j]))
    return changed


def fit_ctn():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    event_col = "event"
    raw = pd.read_csv(ROOT / "ctn0094_weekly_panel.csv")
    split = CTN.split_people(raw)
    train_y = raw.loc[raw.who.isin(split["train"]) & raw[event_col].notna(), event_col].astype(int)
    train_rate = float(train_y.mean()) if len(train_y) else 0.5
    data = CTN.add_references(raw, event_col, train_rate)
    train = data.loc[data.who.isin(split["train"]) & data[event_col].notna()].copy()
    val = data.loc[data.who.isin(split["validation"]) & data[event_col].notna()].copy()
    test = data.loc[data.who.isin(split["test"]) & data[event_col].notna()].copy()

    def feature_frame(frame):
        x = CTN.model_features(frame, True, False)
        x = pd.get_dummies(x, columns=[c for c in ["project", "arm"] if c in x], dtype=float)
        return x

    xtr = feature_frame(train)
    xva = feature_frame(val).reindex(columns=xtr.columns, fill_value=0)
    xte = feature_frame(test).reindex(columns=xtr.columns, fill_value=0)
    imputer = SimpleImputer(strategy="median")
    atr = imputer.fit_transform(xtr)
    ava = imputer.transform(xva)
    ate = imputer.transform(xte)

    lr_scaler = StandardScaler()
    lr_tr = lr_scaler.fit_transform(atr)
    lr = LogisticRegression(max_iter=1500, C=1.0, random_state=CTN.SEED)
    lr.fit(lr_tr, train[event_col].astype(int))
    predictions = {"logistic_regression": lr.predict_proba(lr_scaler.transform(ate))[:, 1]}

    best_iter, best_loss = 50, np.inf
    yyv = val[event_col].astype(int).to_numpy()
    for n_iter in (50, 100, 150):
        model = HistGradientBoostingClassifier(max_iter=n_iter, learning_rate=0.08,
            max_leaf_nodes=15, l2_regularization=1.0, random_state=CTN.SEED)
        model.fit(atr, train[event_col].astype(int))
        pv = np.clip(model.predict_proba(ava)[:, 1], *CTN.CLIP)
        loss = float(np.mean(-(yyv * np.log(pv) + (1 - yyv) * np.log1p(-pv)))) if len(yyv) else np.inf
        if loss < best_loss:
            best_loss, best_iter = loss, n_iter
    gb = HistGradientBoostingClassifier(max_iter=best_iter, learning_rate=0.08,
        max_leaf_nodes=15, l2_regularization=1.0, random_state=CTN.SEED)
    gb.fit(atr, train[event_col].astype(int))
    predictions["gradient_boosting"] = gb.predict_proba(ate)[:, 1]

    def score_permutation(yb):
        permuted = ctn_rebuild_histories(data, test, yb, event_col)
        x = feature_frame(permuted.loc[test.index]).reindex(columns=xtr.columns, fill_value=0)
        xt = imputer.transform(x)
        return {"logistic_regression": lr.predict_proba(lr_scaler.transform(xt))[:, 1],
                "gradient_boosting": gb.predict_proba(xt)[:, 1]}

    def gb_for_patient(pid, target):
        """Fixed gradient-boosting forecasts for one test patient's rows (test-index order) after
        rebuilding that patient's history features from `target` (used by fitted_model_package.py)."""
        rows = test.loc[test.who.eq(pid)]
        permuted = ctn_rebuild_histories(data.loc[data.who.eq(pid)], rows, target, event_col)
        x = feature_frame(permuted.loc[rows.index]).reindex(columns=xtr.columns, fill_value=0)
        return gb.predict_proba(imputer.transform(x))[:, 1]

    return {"test": test, "y": test[event_col].astype(int).to_numpy(), "pid": test.who.to_numpy(),
            "predictions": predictions, "score_permutation": score_permutation,
            "gb_for_patient": gb_for_patient, "selected_gb_iterations": int(best_iter)}


def fit_ces():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    _, panel = CES.load_reports(CES.DATA, 3)
    split = CES.split_people(panel)
    panel = panel.copy()
    panel["split"] = panel.pid.map(split)
    train = panel.loc[panel.split.eq("train")].copy()
    val = panel.loc[panel.split.eq("validation")].copy()
    test = panel.loc[panel.split.eq("test")].copy().reset_index(drop=True)
    cols = CES.ITEMS + ["gap_days", "last4_rate", "last12_rate", "expanding_rate", "runin_rate"]
    xtr, med = CES.make_features(train, cols)
    xva, _ = CES.make_features(val, cols, med)
    xte, _ = CES.make_features(test, cols, med)

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=CES.SEED))
    lr.fit(xtr, train.y)
    predictions = {"logistic_regression": lr.predict_proba(xte)[:, 1]}

    candidates = [50, 100, 150, 200, 300, 400, 600]
    staged = GradientBoostingClassifier(n_estimators=max(candidates), learning_rate=0.05,
        max_depth=2, min_samples_leaf=20, random_state=CES.SEED).fit(xtr, train.y)
    losses = [CES.logloss(val.y.to_numpy(int), p[:, 1]) for p in staged.staged_predict_proba(xva)]
    best = candidates[int(np.argmin([losses[i - 1] for i in candidates]))]
    gb = GradientBoostingClassifier(n_estimators=best, learning_rate=0.05, max_depth=2,
        min_samples_leaf=20, random_state=CES.SEED)
    gb.fit(xtr, train.y)
    predictions["gradient_boosting"] = gb.predict_proba(xte)[:, 1]

    def score_permutation(yb):
        # Keep the first origin (report 21) fixed; later history is run-in plus
        # permuted targets of earlier scored forecasts.
        changed = ces_rebuild_histories(test, yb)
        xt, _ = CES.make_features(changed, cols, med)
        return {"logistic_regression": lr.predict_proba(xt)[:, 1],
                "gradient_boosting": gb.predict_proba(xt)[:, 1]}

    return {"test": test, "y": test.y.to_numpy(int), "pid": test.pid.to_numpy(),
            "predictions": predictions, "score_permutation": score_permutation,
            "selected_gb_iterations": int(best)}


def verify_against_existing(bundle, cohort_key, existing):
    cfg = COHORTS[cohort_key]
    old_item = next(
        (item for item in existing["datasets"] if item.get("dataset") == cfg["existing_dataset"]),
        None,
    )
    if old_item is None:
        raise AssertionError(f"No saved cases_extra result for {cfg['existing_dataset']}")
    saved = old_item["test_patient_variation"]
    checks = {}
    for model_name, prediction in bundle["predictions"].items():
        ids, values = per_patient_auc(bundle["y"], prediction, bundle["pid"])
        if not len(values):
            raise AssertionError(f"No eligible patients for {cohort_key}/{model_name}")
        old = saved[model_name]
        actual = {"mean": float(np.mean(values)), "median": float(np.median(values)),
                  "share_gt_0_5": float(np.mean(values > 0.5)), "eligible": len(values)}
        expected = {"mean": old["mean"], "median": old["deciles"]["50"],
                    "share_gt_0_5": old["share_gt_0_5"], "eligible": old["n_eligible"]}
        for metric in ("mean", "median", "share_gt_0_5"):
            if not np.isclose(actual[metric], expected[metric], atol=5e-6, rtol=0):
                raise AssertionError(
                    f"{cohort_key}/{model_name} {metric}={actual[metric]:.9f}; "
                    f"saved cases_extra value={expected[metric]:.9f}"
                )
        if actual["eligible"] != expected["eligible"]:
            raise AssertionError(f"Eligible-patient count differs for {cohort_key}/{model_name}")
        checks[model_name] = {"recomputed": actual, "cases_extra_results": expected}

    gb = checks["gradient_boosting"]["recomputed"]
    table4 = cfg["table4"]
    rounded = {"median_auc": float(f"{gb['median']:.2f}"),
               "share_gt_0_5": float(f"{gb['share_gt_0_5']:.2f}"),
               "mean_auc": float(f"{gb['mean']:.2f}")}
    if rounded != table4:
        raise AssertionError(
            f"{cfg['label']} GB Table 4 check failed: got {rounded}, expected {table4}"
        )
    checks["table4_gradient_boosting_rounded"] = {
        "recomputed": rounded,
        "expected": table4,
        "passed": True,
    }
    return checks


def cohort_history_features():
    return {
        "replace_bg": {
            "outcome_history_features": ["history_rate_7", "history_rate_28", "expanding_rate", "runin_rate"],
            "held_fixed_runin_features": ["runin_rate"],
            "detail": "Rates use prior eligible nights only; the 14-night run-in remains fixed.",
        },
        "ctn0094": {
            "outcome_history_features": ["previous_result", "runin_rate", "expanding_rate", "recent_rate"],
            "held_fixed_runin_features": ["runin_rate"],
            "detail": "Run-in rate is fixed. Expanding rate is positive weeks divided by observed weeks, as in the panel builder (identity check: rebuilding from the unpermuted outcomes reproduces every forecast); recent rate averages observed outcomes in the prior four calendar weeks. Previous-observed status and expanding_n describe availability, not outcome values.",
        },
        "ces": {
            "outcome_history_features": ["last4_rate", "last12_rate", "expanding_rate", "runin_rate"],
            "held_fixed_runin_features": ["runin_rate"],
            "detail": "Run-in comprises the first 20 report outcomes; the first forecast origin (report 21) is also held fixed, matching the existing report-history timing. Subsequent rates are rebuilt from permuted scored targets.",
        },
    }


def build_markdown(result):
    lines = [
        "# Model-fixed permutation reference for per-patient AUC",
        "",
        "Each fitted model was held fixed while each test patient's scored outcomes were shuffled within patient. Run-in outcomes stayed fixed; history features were rebuilt before rescoring. No model was refit during a permutation.",
        "",
        f"Seed: `{result['metadata']['seed']}`; permutations: `{result['metadata']['permutations']}`; patient-bootstrap resamples: `{result['metadata']['bootstrap_replicates']}`.",
        "",
        "## Results",
        "",
        "The reference columns give the mean and 2.5th–97.5th percentiles across permutations. The observed-minus-reference bootstrap resamples patients; it recomputes the permutation mean of the median within each patient sample and uses the patient-specific permutation means for the mean-AUC contrast.",
        "",
        "| Cohort | Model | Eligible patients | Observed median AUC | Observed mean AUC | Observed above 0.5 | Reference median AUC mean (2.5–97.5%) | Reference share above 0.5 mean (2.5–97.5%) | Median difference (95% patient-bootstrap CI) | Mean difference (95% patient-bootstrap CI) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cohort_key, cohort in result["cohorts"].items():
        for model_name, summary in cohort["models"].items():
            obs = summary["observed"]
            ref = summary["permutation_reference"]
            delta = summary["observed_minus_reference"]
            med_ci = ref["median_auc_permutation_percentiles_2_5_97_5"]
            share_ci = ref["share_above_0_5_permutation_percentiles_2_5_97_5"]
            lines.append(
                f"| {cohort['label']} | {model_name.replace('_', ' ').title()} | {summary['eligible_patients']} | "
                f"{obs['median_auc']:.3f} | {obs['mean_per_patient_auc']:.3f} | {obs['share_above_0_5']:.1%} | "
                f"{ref['median_auc_mean_over_permutations']:.3f} ({med_ci[0]:.3f}–{med_ci[1]:.3f}) | "
                f"{ref['share_above_0_5_mean_over_permutations']:.1%} ({share_ci[0]:.1%}–{share_ci[1]:.1%}) | "
                f"{delta['median_auc']:.3f} ({delta['median_auc_bootstrap_ci95'][0]:.3f}–{delta['median_auc_bootstrap_ci95'][1]:.3f}) | "
                f"{delta['mean_per_patient_auc']:.3f} ({delta['mean_per_patient_auc_bootstrap_ci95'][0]:.3f}–{delta['mean_per_patient_auc_bootstrap_ci95'][1]:.3f}) |"
            )
    lines += ["", "## Reproduction checks", "",
              "The observed mean, median, share above 0.5, and eligible-patient count were compared with `open_data/cases_extra_results.json` for both models. For GB, Table 4's rounded median, share above 0.5, and mean per-patient AUC were also checked.",
              "", "| Cohort | GB median AUC | GB share above 0.5 | GB mean AUC | Cases-extra checks |", "|---|---:|---:|---:|---|"]
    for cohort_key, cohort in result["cohorts"].items():
        chk = cohort["reproduction_checks"]["gradient_boosting"]["recomputed"]
        lines.append(f"| {cohort['label']} | {chk['median']:.3f} | {chk['share_gt_0_5']:.1%} | {chk['mean']:.3f} | passed; Table 4 values reproduced after rounding |")
    lines += ["", "## Outcome-history features", ""]
    for cohort_key, info in result["metadata"]["history_features"].items():
        lines.append(f"- **{COHORTS[cohort_key]['label']}:** `{', '.join(info['outcome_history_features'])}` depend on past outcomes. `{', '.join(info['held_fixed_runin_features'])}` is held fixed. {info['detail']}")
    lines += ["", "The builders in the existing cohort scripts calculate histories during panel construction and do not accept a replacement outcome sequence. This analysis uses small helpers in the new script to rebuild only those history columns; no existing source file was changed.",
              "", "## Interpretation and calculation", "",
              "For every permutation, the median and share above 0.5 were recomputed across patients with both target classes. Their reported reference means are averages of these permutation-level statistics; percentile limits are the 2.5th and 97.5th percentiles of the permutation-level statistics. The median bootstrap interval resamples patients and, within each sample, compares the observed median with the mean permutation-level median. The mean-AUC bootstrap uses the already-computed patient-specific permutation mean AUCs. The JSON also reports the paired median difference and interval using the median of those patient-specific permutation means.",
              "", "For exact reproduction, run from the repository root:", "", "```powershell", "python open_data/cases_model_fixed_permutation.py", "```", "",
              "`--quick` uses 5 permutations and 100 bootstrap resamples for a smoke run. The full run writes `open_data/results/model_fixed_permutation/model_fixed_permutation.json` and this report.", ""]
    return "\n".join(lines)


def run(quick=False):
    n_perm, n_boot = (5, 100) if quick else (N_PERM, N_BOOT)
    existing_path = ROOT / "cases_extra_results.json"
    existing = json.loads(existing_path.read_text(encoding="utf-8"))
    prepared = {
        "replace_bg": fit_replace_bg(),
        "ctn0094": fit_ctn(),
        "ces": fit_ces(),
    }
    output = {
        "metadata": {
            "seed": SEED,
            "permutations": n_perm,
            "bootstrap_replicates": n_boot,
            "model_fit_policy": "Fit once per cohort/model with existing split, seeds, preprocessing, feature sets and settings; keep estimators fixed for every permutation.",
            "existing_observed_results": str(existing_path.relative_to(ROOT.parent)),
            "history_features": cohort_history_features(),
            "permutation": "Within each test patient, shuffle that patient's observed scored target outcomes; hold the run-in fixed; rebuild outcome-history inputs; keep all non-history predictors fixed.",
        },
        "cohorts": {},
    }
    rng_perm = np.random.default_rng(SEED)
    rng_boot = np.random.default_rng(SEED + 1)
    for cohort_key in ("replace_bg", "ctn0094", "ces"):
        bundle = prepared[cohort_key]
        checks = verify_against_existing(bundle, cohort_key, existing)
        target_permutations = generate_permuted_targets(
            bundle["y"], bundle["pid"], n_perm, rng_perm
        )
        auc_by_model = feature_auc_matrix(
            bundle["y"], target_permutations, bundle["pid"], bundle["predictions"],
            bundle["score_permutation"],
        )
        cohort_result = {
            "label": COHORTS[cohort_key]["label"],
            "test_patients": int(pd.unique(bundle["pid"]).size),
            "test_scored_rows": int(len(bundle["y"])),
            "selected_gb_iterations": bundle["selected_gb_iterations"],
            "reproduction_checks": checks,
            "models": {},
        }
        for model_name, auc_data in auc_by_model.items():
            model_rng = np.random.default_rng(rng_boot.integers(0, 2**32 - 1))
            summary = summarize_reference(
                auc_data["observed_auc"], auc_data["permutation_auc"], n_boot, model_rng
            )
            cohort_result["models"][model_name] = summary
        output["cohorts"][cohort_key] = cohort_result

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "model_fixed_permutation.json"
    md_path = OUTPUT_DIR / "model_fixed_permutation.md"
    json_path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    md_path.write_text(build_markdown(output), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    for cohort_key, cohort in output["cohorts"].items():
        for model, summary in cohort["models"].items():
            obs = summary["observed"]
            ref = summary["permutation_reference"]
            print(f"{cohort_key} {model}: observed median={obs['median_auc']:.3f}, "
                  f"share>{0.5}={obs['share_above_0_5']:.3f}, "
                  f"reference median={ref['median_auc_mean_over_permutations']:.3f}")


def helper_selftest():
    """Small NumPy-only feature and summary checks; does not import sklearn."""
    runin = [0, 1] * 7
    panel = pd.DataFrame({
        "pid": ["p"] * 16, "night_index": list(range(16)),
        "model_runin": [True] * 14 + [False, False], "y": runin + [0, 1],
        "history_rate_7": [np.nan] * 14 + [4 / 7, 3 / 7],
        "history_rate_28": [np.nan] * 14 + [0.5, 7 / 15],
        "expanding_rate": [np.nan] * 14 + [0.5, 7 / 15],
    })
    test = panel.loc[~panel.model_runin].copy()
    rebuilt = replace_build_histories(panel, np.asarray([0, 1]), test)
    if not np.allclose(rebuilt.history_rate_7.to_numpy(), [4 / 7, 3 / 7]):
        raise AssertionError("REPLACE-BG history helper failed the synthetic check")
    if not np.allclose(rebuilt.expanding_rate.to_numpy(), [0.5, 7 / 15]):
        raise AssertionError("REPLACE-BG expanding-rate helper failed the synthetic check")

    ctn = pd.DataFrame({
        "who": [1, 1, 1], "study_week": [4, 5, 6], "observed": [1, 1, 1],
        "event": [0, 1, 0], "screen_count_target": [1, 2, 1],
        "history_initial_events": ['[1,0,1,0]'] * 3,
        "history_initial_screens": ['[1,1,1,1]'] * 3,
        "expanding_rate": [0.5, 0.4, 0.5], "recent_rate": [0.5, 1 / 3, 2 / 3],
        "previous_result": [0, 0, 1],
    })
    ctn_rebuilt = ctn_rebuild_histories(ctn, ctn, np.asarray([1, 0, 0]), "event")
    if not np.allclose(ctn_rebuilt.expanding_rate.to_numpy(), [0.5, 0.6, 0.5]):
        raise AssertionError("CTN history helper failed the synthetic check")
    if ctn_rebuilt.previous_result.to_list() != [0.0, 1.0, 0.0]:
        raise AssertionError("CTN previous-result helper failed the synthetic check")

    ces_runin = [0, 1] * 10
    ces = pd.DataFrame({
        "pid": ["c"] * 3, "date": pd.date_range("2020-01-01", periods=3),
        "runin_event_history": [ces_runin] * 3, "current_event": [1, 0, 1],
        "y": [0, 1, 0], "expanding_rate": [0.0] * 3,
        "last4_rate": [0.0] * 3, "last12_rate": [0.0] * 3,
    })
    ces_rebuilt = ces_rebuild_histories(ces, np.asarray([1, 0, 1]))
    if not np.allclose(ces_rebuilt.expanding_rate.to_numpy(), [11 / 21, 12 / 22, 12 / 23]):
        raise AssertionError("CES expanding-rate helper failed the synthetic check")
    if not np.allclose(ces_rebuilt.last4_rate.to_numpy(), [3 / 4, 3 / 4, 3 / 4]):
        raise AssertionError("CES recent-rate helper failed the synthetic check")

    toy_y = np.asarray([0, 1, 0, 1])
    toy_pid = np.asarray(["a", "a", "b", "b"])
    toy_perms = generate_permuted_targets(toy_y, toy_pid, 4, np.random.default_rng(9))
    for _, ix in group_indices(toy_pid):
        if not np.all(toy_perms[:, ix].sum(axis=1) == toy_y[ix].sum()):
            raise AssertionError("Patient-wise target permutation failed the synthetic check")
    toy_predictions = {"toy": np.asarray([0.1, 0.9, 0.9, 0.1])}
    scorer_calls = []

    def toy_scorer(yb):
        scorer_calls.append(1)
        return {"toy": np.asarray(yb, dtype=float)}

    toy_auc = feature_auc_matrix(toy_y, toy_perms, toy_pid, toy_predictions, toy_scorer)["toy"]["permutation_auc"]
    if len(scorer_calls) != len(toy_perms) or not np.allclose(toy_auc, 1.0):
        raise AssertionError("Fixed-model permutation rescoring failed the synthetic check")

    obs = np.asarray([0.7, 0.8])
    perm = np.asarray([[0.4, 0.6], [0.6, 0.4]])
    summary = summarize_reference(obs, perm, 20, np.random.default_rng(3))
    if not np.isclose(summary["observed"]["median_auc"], 0.75):
        raise AssertionError("Reference summary helper failed the synthetic check")
    if not np.isclose(summary["permutation_reference"]["share_above_0_5_mean_over_permutations"], 0.5):
        raise AssertionError("Permutation-share helper failed the synthetic check")
    print("Pure helper self-check passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run 5 permutations and 100 patient-bootstrap draws")
    parser.add_argument("--selftest", action="store_true", help="Check NumPy/pandas helpers without fitting models")
    args = parser.parse_args()
    if args.selftest:
        helper_selftest()
    else:
        run(quick=args.quick)


if __name__ == "__main__":
    main()
