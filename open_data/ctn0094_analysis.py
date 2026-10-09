#!/usr/bin/env python3
"""Aggregate-only CTN-0094 real-data analysis. Panels are prepared separately."""
from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from history_kernel import history_plan, rebuild_history

SEED = 20261006
N_BOOT = 1000
N_CAL_BOOT = 500
N_PERM = 200
CLIP = (1e-6, 1 - 1e-6)
BASELINES = ['training_cohort_rate', 'runin_rate', 'runin_shrunk', 'updating_rate',
             'recent_window_rate', 'persistence', 'retrospective_rate']


def safe(x):
    if isinstance(x, dict):
        return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return safe(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    if pd.isna(x):
        return None
    return x


def auc(y, score):
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    ok = np.isfinite(score)
    y, score = y[ok], score[ok]
    n1, n0 = int(y.sum()), int((1-y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    _, code = np.unique(score, return_inverse=True)
    ng = int(code.max())+1
    pos = np.bincount(code, weights=y, minlength=ng)
    neg = np.bincount(code, weights=1-y, minlength=ng)
    favorable = float(np.sum(pos*(np.cumsum(neg)-.5*neg)))
    return float(favorable/(n1*n0))


def ap_tie_expected(y, score):
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    order = np.argsort(-score, kind='mergesort')
    s, yy = score[order], y[order]
    N, events = len(y), int(y.sum())
    if events == 0 or N == 0:
        return np.nan
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    ends = np.r_[starts[1:], N]
    cum = 0
    ap = 0.0
    first_sum, second_sum = 0.0, 0.0
    for lo, hi in zip(starts, ends):
        m = hi-lo
        positives = int(yy[lo:hi].sum())
        cum += positives
        if positives:
            ap += positives * (cum / hi)
        first_sum += m/hi
        second_sum += m*(lo+m-1)/hi
    expectation = first_sum/N
    if events > 1 and N > 1:
        expectation += (events-1)/(N*(N-1))*second_sum
    return float(ap/events), float(expectation)


def per_person(y, score, pid, metric):
    vals = []
    for ix in grouped_indices(pid):
        yy, ss = np.asarray(y)[ix], np.asarray(score)[ix]
        if metric == 'auc':
            v = auc(yy, ss)
        elif metric == 'ap':
            ap = ap_tie_expected(yy, ss)
            v = ap[0] if isinstance(ap, tuple) else np.nan
        else:
            v = np.nan
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan, len(vals)


def grouped_indices(pid):
    pid = np.asarray(pid)
    _, inverse = np.unique(pid, return_inverse=True)
    order = np.argsort(inverse, kind='mergesort')
    counts = np.bincount(inverse)
    cuts = np.r_[0, np.cumsum(counts)]
    return [order[cuts[j]:cuts[j+1]] for j in range(len(counts))]


def bootstrap_ci(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return [None, None]
    return [float(np.quantile(x, .025)), float(np.quantile(x, .975))]


def patient_bootstrap_indices(pid, n_boot, seed):
    pid = np.asarray(pid)
    groups = [np.flatnonzero(pid == p) for p in pd.unique(pid)]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        draw = rng.integers(0, len(groups), len(groups))
        yield np.concatenate([groups[j] for j in draw])


def metric_boot_ci(y, pred, pid, metric, n_boot=N_BOOT, seed=SEED+17):
    y = np.asarray(y, int); pred = np.asarray(pred, float); pid = np.asarray(pid)
    unique, inverse = np.unique(pid, return_inverse=True)
    groups = grouped_indices(pid)
    rng = np.random.default_rng(seed)
    vals = []
    if metric == 'within_auc':
        person_auc = np.array([auc(y[ix], pred[ix]) for ix in groups], float)
        eligible = np.isfinite(person_auc)
        for _ in range(n_boot):
            draw = rng.integers(len(groups), size=len(groups))
            v = person_auc[draw]
            vals.append(float(np.nanmean(v)) if np.any(np.isfinite(v)) else np.nan)
    elif metric == 'pooled_auc':
        ok = np.isfinite(pred)
        yy, pp, inv = y[ok], pred[ok], inverse[ok]
        order = np.argsort(pp, kind='mergesort')
        pp, yy, inv = pp[order], yy[order], inv[order]
        starts = np.r_[0, np.flatnonzero(pp[1:] != pp[:-1])+1]
        ends = np.r_[starts[1:], len(pp)]
        for _ in range(n_boot):
            mult = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
            row_w = mult[inv]
            pos = np.add.reduceat(yy*row_w, starts)
            neg = np.add.reduceat((1-yy)*row_w, starts)
            below = np.cumsum(neg)-neg
            denom = pos.sum()*(neg.sum())
            vals.append(float(np.sum(pos*(below+.5*neg))/denom) if denom else np.nan)
    elif metric == 'brier_diff':
        loss = (pred-y)**2
        sums = np.array([loss[ix].sum() for ix in groups])
        counts = np.array([len(ix) for ix in groups])
        for _ in range(n_boot):
            draw = rng.integers(len(groups), size=len(groups))
            vals.append(float(sums[draw].sum()/counts[draw].sum()))
    else:
        raise ValueError(metric)
    return bootstrap_ci(vals)


def expit(x):
    x = np.asarray(x, float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1/(1+np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex/(1+ex)
    return out


def logit(p):
    p = np.clip(np.asarray(p, float), *CLIP)
    return np.log(p)-np.log1p(-p)


def irls(X, y, weights=None, max_iter=80):
    """Unpenalised weighted logistic regression for calibration; intercept is caller supplied."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = np.clip(expit(X @ beta), 1e-8, 1-1e-8)
        v = w*p*(1-p)
        z = X @ beta + (y-p)/(p*(1-p))
        xtwx = X.T @ (v[:, None]*X)
        xtwz = X.T @ (v*z)
        try:
            new = np.linalg.solve(xtwx + 1e-9*np.eye(X.shape[1]), xtwz)
        except np.linalg.LinAlgError:
            return np.full(X.shape[1], np.nan)
        if np.max(np.abs(new-beta)) < 1e-7:
            beta = new
            break
        beta = new
    return beta


def cohort_stats(df, event_col):
    ydf = df.loc[df[event_col].notna()].copy()
    rates = ydf.groupby('who')[event_col].mean()
    mu = float(ydf[event_col].mean()) if len(ydf) else np.nan
    var = float(rates.var(ddof=0)) if len(rates) else np.nan
    icc = var/(mu*(1-mu)) if 0 < mu < 1 else np.nan
    gaps = []
    if 'study_week' in ydf:
        for _, g in ydf.groupby('who'):
            w = np.sort(g.study_week.to_numpy())
            if len(w) > 1:
                gaps.extend(np.diff(w).tolist())
        gap_unit = 'weeks between observed screen weeks'
    else:
        for _, g in ydf.groupby('who'):
            d = np.sort(g.study_day.to_numpy())
            if len(d) > 1:
                gaps.extend(np.diff(d).tolist())
        gap_unit = 'calendar days between scored forecasts'
    return {'patients': int(ydf.who.nunique()), 'scored_steps': int(len(ydf)),
            'event_count': int(ydf[event_col].sum()), 'event_rate': mu,
            'patient_event_rate_mean': float(rates.mean()), 'patient_event_rate_sd': float(rates.std(ddof=1)),
            'icc': icc, 'median_gap': float(np.median(gaps)) if gaps else None,
            'median_gap_unit': gap_unit,
            'median_scored_steps_per_person': float(ydf.groupby('who').size().median()),
            'trial_counts': {str(k): int(v) for k,v in ydf.groupby('project').who.nunique().items()},
            'arm_counts': {str(k): int(v) for k,v in ydf.groupby('arm').who.nunique().items()}}


def add_references(df, event_col, train_rate):
    d = df.copy()
    d.attrs['history_fallback'] = float(train_rate)
    d.attrs['history_event_column'] = event_col
    d['updating_rate'] = d['expanding_rate']
    d['recent_window_rate'] = d['recent_rate']
    d['training_cohort_rate'] = train_rate
    d['runin_shrunk'] = (d.runin_rate*d.get('runin_n', pd.Series(28, index=d.index)) + 10*train_rate) / (
        d.get('runin_n', pd.Series(28, index=d.index)) + 10)
    d['persistence'] = d.persistence.fillna(train_rate)
    d['retrospective_rate'] = d.groupby('who')[event_col].transform('mean')
    for c in BASELINES:
        if c in d:
            d[c] = d[c].fillna(train_rate).clip(*CLIP)
    return d


def rebuild_missing_positive_history(df):
    """For the missing-as-positive sensitivity, rebuild run-in and weekly histories on the full calendar."""
    d=df.copy()
    for _, ix in d.groupby('who',sort=False).groups.items():
        ix=np.asarray(list(ix)); ix=ix[np.argsort(d.loc[ix,'study_week'].to_numpy())]
        first=d.loc[ix[0]]
        rn=int(first.runin_n)
        init_rate=(float(first.runin_rate)*rn+(4-rn))/4
        prior_targets=[]
        for rowix in ix:
            target=int(d.at[rowix,'study_week'])
            origin=target-1
            d.at[rowix,'runin_rate']=init_rate
            d.at[rowix,'runin_n']=4
            # Four run-in weeks are represented by their sensitivity-adjusted average;
            # all subsequent prior weeks use their imputed or observed binary outcome.
            past=[int(d.at[j,'event_missing_positive']) for j in prior_targets]
            d.at[rowix,'expanding_rate']=(init_rate*4+sum(past))/(4+len(past))
            d.at[rowix,'expanding_n']=4+len(past)
            if past:
                trailing=past[-4:]
                if len(trailing)<4:
                    trailing=[init_rate]*(4-len(trailing))+trailing
                d.at[rowix,'recent_rate']=float(np.mean(trailing))
                d.at[rowix,'recent_n']=4
            else:
                d.at[rowix,'recent_rate']=init_rate
                d.at[rowix,'recent_n']=4
            if target>4:
                d.at[rowix,'previous_result']=past[-1] if past else init_rate
                d.at[rowix,'previous_observed']=int(d.at[prior_targets[-1],'observed']) if prior_targets else 0
                d.at[rowix,'persistence']=past[-1] if past else init_rate
            elif not int(d.at[rowix,'previous_observed']):
                d.at[rowix,'previous_result']=1
                d.at[rowix,'persistence']=1
            prior_targets.append(rowix)
    return d


def split_people(df, seed=SEED):
    ids = np.array(sorted(df.who.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_train = int(.6*len(ids))
    n_val = int(.2*len(ids))
    return {'train': set(ids[:n_train]), 'validation': set(ids[n_train:n_train+n_val]),
            'test': set(ids[n_train+n_val:])}


def test_rows(df, split):
    return df.loc[df.who.isin(split['test'])].copy()


def model_features(df, weekly, centred=False, include_current_dose=False):
    if weekly:
        cols = ['previous_result', 'previous_observed', 'runin_rate', 'expanding_rate',
                'recent_rate', 'expanding_n', 'self_report_opioid_days_last_week',
                'dose_last_week', 'study_week', 'project', 'arm']
        if include_current_dose:
            cols.insert(cols.index('study_week'), 'dose_current')
    else:
        cols = ['previous_result', 'runin_rate', 'expanding_rate', 'recent_rate',
                'self_report_opioid_days_last_28', 'study_day', 'project', 'arm']
    x = df[cols].copy()
    if centred:
        # Centre changing patient-specific signals on the available run-in event rate.
        for c in ['previous_result', 'expanding_rate', 'recent_rate']:
            x[c] = pd.to_numeric(x[c], errors='coerce') - pd.to_numeric(df.runin_rate, errors='coerce')
        x['runin_rate'] = 0.0
        if weekly and 'self_report_opioid_days_last_week' in x:
            x['self_report_opioid_days_last_week'] = (x.self_report_opioid_days_last_week - df.runin_self_report_mean)
            x['dose_last_week'] = x.dose_last_week - df.runin_dose_mean
            if include_current_dose:
                x['dose_current'] = x.dose_current - df.runin_dose_mean
            x['previous_observed'] = x.previous_observed.fillna(0)
        # Static trial and arm are omitted in the centred model.
        x = x.drop(columns=['project', 'arm'])
    return x


def fit_models(df, event_col, split, weekly, include_current_dose=False):
    out = {}
    validation_out = {}
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        return {'error': f'{type(exc).__name__}: {exc}', 'predictions': out}
    train = df.loc[df.who.isin(split['train']) & df[event_col].notna()]
    val = df.loc[df.who.isin(split['validation']) & df[event_col].notna()]
    test = df.loc[df.who.isin(split['test']) & df[event_col].notna()]
    if train[event_col].nunique() < 2:
        return {'error': 'Training data contain only one outcome class.', 'predictions': out}
    train_rate = float(train[event_col].mean())
    for name, centered in [('logistic_regression', False), ('gradient_boosting', False),
                           ('within_centred_gradient_boosting', True)]:
        try:
            x_train = model_features(train, weekly, centered, include_current_dose)
            x_val = model_features(val, weekly, centered, include_current_dose)
            x_test = model_features(test, weekly, centered, include_current_dose)
            # One-hot categories are learned on training rows only; validation/test unseen values are ignored.
            x_train = pd.get_dummies(x_train, columns=[c for c in ['project', 'arm'] if c in x_train], dtype=float)
            x_val = pd.get_dummies(x_val, columns=[c for c in ['project', 'arm'] if c in x_val], dtype=float)
            x_test = pd.get_dummies(x_test, columns=[c for c in ['project', 'arm'] if c in x_test], dtype=float)
            x_val = x_val.reindex(columns=x_train.columns, fill_value=0)
            x_test = x_test.reindex(columns=x_train.columns, fill_value=0)
            med = SimpleImputer(strategy='median')
            a = med.fit_transform(x_train)
            b = med.transform(x_val)
            c = med.transform(x_test)
            if name == 'logistic_regression':
                scaler = StandardScaler()
                a, b, c = scaler.fit_transform(a), scaler.transform(b), scaler.transform(c)
                mdl = LogisticRegression(max_iter=1500, C=1.0, random_state=SEED)
                mdl.fit(a, train[event_col].astype(int))
                out[name] = mdl.predict_proba(c)[:, 1]
                validation_out[name] = mdl.predict_proba(b)[:, 1]
            else:
                best_iter, best_loss = 50, np.inf
                yyv = val[event_col].astype(int).to_numpy()
                for n_iter in [50, 100, 150]:
                    mdl = HistGradientBoostingClassifier(max_iter=n_iter, learning_rate=.08,
                        max_leaf_nodes=15, l2_regularization=1.0, random_state=SEED)
                    mdl.fit(a, train[event_col].astype(int))
                    pv = np.clip(mdl.predict_proba(b)[:, 1], *CLIP)
                    loss = float(np.mean(-(yyv*np.log(pv)+(1-yyv)*np.log1p(-pv)))) if len(yyv) else np.inf
                    if loss < best_loss:
                        best_loss, best_iter = loss, n_iter
                final = HistGradientBoostingClassifier(max_iter=best_iter, learning_rate=.08,
                    max_leaf_nodes=15, l2_regularization=1.0, random_state=SEED)
                final.fit(a, train[event_col].astype(int))
                out[name] = final.predict_proba(c)[:, 1]
                validation_out[name] = np.clip(final.predict_proba(b)[:, 1], *CLIP)
                if name == 'gradient_boosting':
                    out['gradient_boosting_validation'] = {'selected_iterations': best_iter,
                                                            'validation_log_loss': best_loss}
        except Exception as exc:
            out[name+'_error'] = f'{type(exc).__name__}: {exc}'
    return {'error': None, 'predictions': out, 'train_rate': train_rate,
            'test_index': test.index.to_list(), 'validation_predictions': validation_out,
            'validation_index': val.index.to_list()}


def threshold_metrics(y, score, pid, threshold):
    y = np.asarray(y, int)
    pred = np.asarray(score, float) >= threshold
    sens_ok, spec_ok = [], []
    for ix in grouped_indices(pid):
        if np.any(y[ix] == 1):
            sens_ok.append(float(pred[ix][y[ix] == 1].mean()))
        if np.any(y[ix] == 0):
            spec_ok.append(float((~pred[ix][y[ix] == 0]).mean()))
    return {'pooled_sensitivity': float(np.sum(pred & (y == 1))/max(1, np.sum(y == 1))),
            'pooled_specificity': float(np.sum((~pred) & (y == 0))/max(1, np.sum(y == 0))),
            'mean_person_sensitivity': float(np.mean(sens_ok)) if sens_ok else None,
            'mean_person_specificity': float(np.mean(spec_ok)) if spec_ok else None,
            'n_sensitivity_people': len(sens_ok), 'n_specificity_people': len(spec_ok)}


def scores_and_skills(y, preds, pid, references):
    y = np.asarray(y, float)
    result = {}
    ll_y = np.clip(y, *CLIP)
    for name, p in preds.items():
        p = np.clip(np.asarray(p, float), *CLIP)
        brier = (p-y)**2
        ll = -(y*np.log(p)+(1-y)*np.log1p(-p))
        per_b, per_l = [], []
        groups = grouped_indices(pid)
        for ix in groups:
            per_b.append(float(brier[ix].mean()))
            per_l.append(float(ll[ix].mean()))
        entry = {'brier_pooled': float(brier.mean()), 'brier_person_mean': float(np.mean(per_b)),
                 'log_loss_pooled': float(ll.mean()), 'log_loss_person_mean': float(np.mean(per_l)),
                 'n_people': len(per_b), 'skills': {}}
        for ref_name, ref0 in references.items():
            ref = np.clip(np.asarray(ref0, float), *CLIP)
            rb = (ref-y)**2
            ref_ll = -(y*np.log(ref)+(1-y)*np.log1p(-ref))
            person_diff, person_ll_diff = [], []
            for ix in groups:
                person_diff.append(float(rb[ix].mean()-brier[ix].mean()))
                person_ll_diff.append(float(ref_ll[ix].mean()-ll[ix].mean()))
            entry['skills'][ref_name] = {
                'brier_skill_summed_losses': float(1-brier.sum()/rb.sum()) if rb.sum() else None,
                'brier_skill_person_mean_difference': float(np.mean(person_diff)),
                'paired_person_boot_ci_skill': None,
                'paired_person_boot_ci_difference': None,
                'log_loss_skill_summed_losses': float(1-ll.sum()/ref_ll.sum()) if ref_ll.sum() else None,
                'log_loss_skill_person_mean_difference': float(np.mean(person_ll_diff)),
                'paired_person_boot_ci_log_loss_skill': None,
                'paired_person_boot_ci_log_loss_difference': None,
            }
        result[name] = entry
    return result


def ranking_results(y, preds, pid, nboot=N_BOOT, seed=SEED+31):
    result = {}
    for j, (name, score) in enumerate(preds.items()):
        p_auc = auc(y, score)
        w_auc, n_w = per_person(y, score, pid, 'auc')
        bpool = metric_boot_ci(y, score, pid, 'pooled_auc', nboot, seed+j)
        bwin = metric_boot_ci(y, score, pid, 'within_auc', nboot, seed+100+j)
        gap_ci = pooled_within_gap_ci(y, score, pid, nboot, seed+200+j)
        result[name] = {'pooled_auc': p_auc, 'pooled_auc_ci': bpool,
                        'within_person_auc': w_auc, 'within_person_auc_ci': bwin,
                        'pooled_minus_within_auc': p_auc-w_auc if np.isfinite(p_auc) and np.isfinite(w_auc) else None,
                        'pooled_minus_within_auc_ci': gap_ci,
                        'eligible_within_people': n_w}
    return result


def pooled_within_gap_ci(y, score, pid, n_boot=N_BOOT, seed=SEED+222):
    y=np.asarray(y,int); score=np.asarray(score,float); pid=np.asarray(pid)
    groups=grouped_indices(pid)
    person_auc=np.array([auc(y[ix],score[ix]) for ix in groups],float)
    unique,inverse=np.unique(pid,return_inverse=True)
    ok=np.isfinite(score)
    yy,ss,inv=y[ok],score[ok],inverse[ok]
    order=np.argsort(ss,kind='mergesort'); ss,yy,inv=ss[order],yy[order],inv[order]
    starts=np.r_[0,np.flatnonzero(ss[1:]!=ss[:-1])+1]
    rng=np.random.default_rng(seed); gaps=[]
    for _ in range(n_boot):
        draw=rng.integers(len(unique),size=len(unique))
        mult=np.bincount(draw,minlength=len(unique))
        row_w=mult[inv]
        pos=np.add.reduceat(yy*row_w,starts)
        neg=np.add.reduceat((1-yy)*row_w,starts)
        den=pos.sum()*neg.sum()
        pool=float(np.sum(pos*(np.cumsum(neg)-neg+.5*neg))/den) if den else np.nan
        sampled=person_auc[draw]
        within=float(np.nanmean(sampled)) if np.isfinite(sampled).any() else np.nan
        gaps.append(pool-within)
    return bootstrap_ci(gaps)


def add_skill_bootstrap(y, preds, pid, refs, results, nboot=N_BOOT, seed=SEED+51):
    # Paired patient bootstrap; report both summed-loss skill and equal-patient loss difference.
    groups = grouped_indices(pid)
    rng = np.random.default_rng(seed)
    for model_i, (name, p) in enumerate(preds.items()):
        for ref_i, (ref_name, ref) in enumerate(refs.items()):
            skill_vals, diff_vals, ll_skill_vals, ll_diff_vals = [], [], [], []
            yy0, pp0, rr0 = np.asarray(y), np.asarray(p), np.asarray(ref)
            model_sums = np.array([np.sum((pp0[ix]-yy0[ix])**2) for ix in groups])
            ref_sums = np.array([np.sum((rr0[ix]-yy0[ix])**2) for ix in groups])
            model_means = np.array([np.mean((pp0[ix]-yy0[ix])**2) for ix in groups])
            ref_means = np.array([np.mean((rr0[ix]-yy0[ix])**2) for ix in groups])
            clip_p, clip_r = np.clip(pp0,*CLIP), np.clip(rr0,*CLIP)
            model_ll = -(yy0*np.log(clip_p)+(1-yy0)*np.log1p(-clip_p))
            ref_ll = -(yy0*np.log(clip_r)+(1-yy0)*np.log1p(-clip_r))
            model_ll_sums = np.array([model_ll[ix].sum() for ix in groups])
            ref_ll_sums = np.array([ref_ll[ix].sum() for ix in groups])
            model_ll_means = np.array([model_ll[ix].mean() for ix in groups])
            ref_ll_means = np.array([ref_ll[ix].mean() for ix in groups])
            for _ in range(nboot):
                draw = rng.integers(0, len(groups), len(groups))
                msum, rsum = model_sums[draw].sum(), ref_sums[draw].sum()
                skill_vals.append(1-msum/rsum if rsum else np.nan)
                diff_vals.append(float(np.mean(ref_means[draw]-model_means[draw])))
                ml, rl = model_ll_sums[draw].sum(), ref_ll_sums[draw].sum()
                ll_skill_vals.append(1-ml/rl if rl else np.nan)
                ll_diff_vals.append(float(np.mean(ref_ll_means[draw]-model_ll_means[draw])))
            results[name]['skills'][ref_name]['paired_person_boot_ci_skill'] = bootstrap_ci(skill_vals)
            results[name]['skills'][ref_name]['paired_person_boot_ci_difference'] = bootstrap_ci(diff_vals)
            results[name]['skills'][ref_name]['paired_person_boot_ci_log_loss_skill'] = bootstrap_ci(ll_skill_vals)
            results[name]['skills'][ref_name]['paired_person_boot_ci_log_loss_difference'] = bootstrap_ci(ll_diff_vals)


def retrospective_optimism(y, pid):
    q, n = [], []
    for ix in grouped_indices(pid):
        if len(ix) > 1:
            qi = float(np.mean(np.asarray(y)[ix]))
            q.append(qi); n.append(len(ix))
    q, n = np.asarray(q), np.asarray(n, float)
    if not len(n):
        return {'day_weighted': None, 'equal_person_weighted': None}
    term = q*(1-q)/(n-1)
    return {'day_weighted': float(np.sum(term*n)/np.sum(n)),
            'equal_person_weighted': float(np.mean(term)),
            'patients_used': int(len(n))}


def calibration_results(y, preds, pid, n_boot=N_CAL_BOOT, seed=SEED+81):
    y = np.asarray(y, float)
    pid = np.asarray(pid)
    result = {}
    unique = pd.unique(pid)
    groups = grouped_indices(pid)
    person_weight = np.zeros(len(pid), float)
    for ix in groups:
        person_weight[ix] = 1/len(ix)
    rng = np.random.default_rng(seed)
    for m, (name, prob) in enumerate(preds.items()):
        z = logit(prob)
        zbar = np.zeros(len(z))
        for ix in groups:
            zbar[ix] = np.mean(z[ix])
        dev = z-zbar
        Xp = np.column_stack([np.ones(len(y)), z])
        Xbw = np.column_stack([np.ones(len(y)), zbar, dev])
        params = {}
        for weighted, wname, weight in [(False, 'day_weighted', None), (True, 'equal_person_weighted', person_weight)]:
            bp = irls(Xp, y, weight)
            bbw = irls(Xbw, y, weight)
            params[wname] = {'pooled_intercept': bp[0], 'pooled_slope': bp[1],
                             'between_intercept': bbw[0], 'between_slope': bbw[1],
                             'within_intercept': bbw[0], 'within_slope': bbw[2]}
        share = float(np.var(zbar)/(np.var(z)+1e-15))
        boot = {weighting: {k: [] for k in ['pooled_intercept','pooled_slope','between_intercept','between_slope','within_intercept','within_slope']}
                for weighting in ['day_weighted','equal_person_weighted']}
        for _ in range(n_boot):
            draw = rng.integers(len(groups), size=len(groups))
            mult = np.bincount(draw, minlength=len(groups))
            # Build patient multiplicities on the original row order.
            wrow = np.zeros(len(y))
            for j, ix in enumerate(groups):
                wrow[ix] = mult[j]
            if np.sum(wrow) == 0:
                continue
            for weighting, weights in [('day_weighted',wrow),
                                       ('equal_person_weighted',wrow*person_weight)]:
                bp = irls(Xp, y, weights)
                bbw = irls(Xbw, y, weights)
                vals = [bp[0],bp[1],bbw[0],bbw[1],bbw[0],bbw[2]]
                for k, v in zip(boot[weighting], vals):
                    boot[weighting][k].append(v)
        result[name] = {'slopes': params,
                        'bootstrap_ci': {w:{k:bootstrap_ci(v) for k,v in vals.items()} for w,vals in boot.items()},
                        'between_forecast_logit_variance_share': share,
                        'within_forecast_logit_variance_share': 1-share}
    return result


def history_bias(df, y, pid, n_perm=N_PERM, seed=SEED+101):
    """Exchangeability reference with rate histories rebuilt after permuting each scored sequence."""
    y = np.asarray(y, int)
    pid = np.asarray(pid)
    rng = np.random.default_rng(seed)
    hist_names = ['updating_rate', 'recent_window_rate', 'persistence']
    person_indices = grouped_indices(pid)
    observed_by_person = {name: np.array([auc(y[ix],df.iloc[ix][name].to_numpy(float)) for ix in person_indices])
                          for name in hist_names}
    refs_by_person = {name: [] for name in hist_names}
    plans = [history_plan(df.iloc[ix], df.attrs['history_event_column'],
                          df.attrs['history_fallback']) for ix in person_indices]
    for _ in range(n_perm):
        perm_scores = {name: np.full(len(y), np.nan) for name in hist_names}
        permuted_targets = np.full(len(y), np.nan)
        for ix, plan in zip(person_indices, plans):
            perm_y = y[ix].copy()
            rng.shuffle(perm_y)
            permuted_targets[ix] = perm_y
            rebuilt = rebuild_history(plan, perm_y)
            for name in hist_names:
                perm_scores[name][ix] = rebuilt[name]
        for name in hist_names:
            refs_by_person[name].append([auc(permuted_targets[ix],perm_scores[name][ix]) for ix in person_indices])
    out = {}
    for name in hist_names:
        obs = observed_by_person[name]
        refs = np.asarray(refs_by_person[name], float)
        eligible = np.isfinite(obs) & np.isfinite(refs).any(axis=0)
        per_patient_ref = np.nanmean(refs[:,eligible],axis=0)
        delta = obs[eligible]-per_patient_ref
        rng_boot = np.random.default_rng(seed+999)
        boot_delta = [float(np.mean(delta[rng_boot.integers(len(delta),size=len(delta))]))
                      for _ in range(N_BOOT)] if len(delta) else []
        reference_draw_means = np.nanmean(refs[:,eligible], axis=1) if eligible.any() else np.full(refs.shape[0], np.nan)
        out[name] = {'observed_within_auc': float(np.nanmean(obs[eligible])) if eligible.any() else None,
                     'eligible_people': int(eligible.sum()),
                     'permutation_reference_mean': float(np.nanmean(per_patient_ref)),
                     'observed_minus_reference': float(np.mean(delta)) if len(delta) else None,
                     'observed_minus_reference_patient_boot_ci': bootstrap_ci(boot_delta),
                     'permutation_reference_sd': float(np.nanstd(reference_draw_means, ddof=1))}
    return out


def ap_results(y, preds, pid):
    out = {}
    for name, score in preds.items():
        ap, chance, prevalence = [], [], []
        for ix in grouped_indices(pid):
            yy, ss = np.asarray(y)[ix], np.asarray(score)[ix]
            x = ap_tie_expected(yy, ss)
            if isinstance(x, tuple) and np.isfinite(x[0]):
                ap.append(x[0]); chance.append(x[1]); prevalence.append(float(np.mean(yy)))
        out[name] = {'mean_average_precision': float(np.mean(ap)) if ap else None,
                     'mean_prevalence': float(np.mean(prevalence)) if prevalence else None,
                     'mean_tie_matched_expectation': float(np.mean(chance)) if chance else None,
                     'mean_difference_from_tie_expectation': float(np.mean(np.asarray(ap)-np.asarray(chance))) if ap else None,
                     'eligible_people': len(ap)}
    return out


def benchmark(y, pid, preds):
    out = {}
    rates = pd.DataFrame({'pid': pid, 'y': y}).groupby('pid').y.agg(['mean','count'])
    weights=rates['count'].to_numpy(float); weights/=weights.sum()
    q=rates['mean'].to_numpy(float)
    mu = float(np.average(q,weights=weights))
    observed_var = float(np.average((q-mu)**2,weights=weights))
    sampling_var = float(np.average(q*(1-q)/np.maximum(1,rates['count'].to_numpy(float)-1),weights=weights))
    corrected_sd = math.sqrt(max(0, observed_var-sampling_var))
    for ref in ['retrospective_rate','runin_rate','updating_rate']:
        if ref in preds:
            out[ref] = {'exact_pooled_auc': auc(y, preds[ref])}
    out['normal_approximation'] = {
        'mean_event_rate': mu, 'observed_patient_rate_sd': math.sqrt(max(0, observed_var)),
        'sampling_corrected_patient_rate_sd': corrected_sd,
        'auc_observed_sd': .5*(1+math.erf((math.sqrt(max(0, observed_var))/math.sqrt(2)/max(1e-12,mu*(1-mu)))/math.sqrt(2))),
        'auc_sampling_corrected_sd': .5*(1+math.erf((corrected_sd/math.sqrt(2)/max(1e-12,mu*(1-mu)))/math.sqrt(2))) }
    return out


def write_forecasts(path, frame, patient_col, time_col, y, predictions, threshold):
    d = frame.reset_index(drop=True)
    out = pd.DataFrame({'_patient_id': d[patient_col], '_time': d[time_col], 'y': np.asarray(y, int)})
    for name, values in predictions.items():
        out[name] = np.asarray(values, float)
    out = out.sort_values(['_patient_id', '_time'], kind='mergesort').reset_index(drop=True)
    ids = pd.unique(out['_patient_id'])
    anon = {value: i for i, value in enumerate(ids)}
    out.insert(0, 'patient', out['_patient_id'].map(anon).astype(int))
    out.insert(1, 't', out.groupby('_patient_id', sort=False).cumcount())
    out = out.drop(columns=['_patient_id', '_time'])
    out['threshold'] = float(threshold)
    out.to_csv(path, index=False)


def analyse(label, raw, event_col, weekly, outdir, include_current_dose=False, suffix=''):
    result = {'label': label, 'event_column': event_col}
    if weekly and event_col == 'event_missing_positive':
        raw = rebuild_missing_positive_history(raw)
    # 1. Cohort and data description.
    try:
        result['1_cohort'] = cohort_stats(raw, event_col)
        result['1_cohort']['rows_in_panel'] = int(len(raw))
        result['1_cohort']['rows_with_missing_outcome'] = int(raw[event_col].isna().sum())
        result['1_cohort']['mean_screen_days_per_observed_week'] = (
            float(raw.loc[raw.observed.eq(1), 'screen_count_target'].mean()) if weekly else None)
    except Exception as exc:
        result['1_cohort_error'] = f'{type(exc).__name__}: {exc}'
    print(f"{label}: block 1 complete",flush=True)
    # 2. Split patients 60/20/20.
    try:
        split = split_people(raw)
        result['2_split'] = {k: len(v) for k,v in split.items()}
        result['2_split']['seed'] = SEED
    except Exception as exc:
        result['2_split_error'] = f'{type(exc).__name__}: {exc}'
        return result
    print(f"{label}: block 2 complete",flush=True)
    train_y = raw.loc[raw.who.isin(split['train']) & raw[event_col].notna(), event_col].astype(int)
    train_rate = float(train_y.mean()) if len(train_y) else .5
    data = add_references(raw, event_col, train_rate)
    # 3. Models and the named references.
    try:
        fitted = fit_models(data, event_col, split, weekly, include_current_dose)
        result['3_models'] = {'training_event_rate': train_rate,
                              'validation_chosen_iterations': fitted.get('predictions', {}).get('gradient_boosting_validation'),
                              'error': fitted.get('error')}
        if fitted.get('error'):
            result['3_models']['unavailable_reason'] = fitted['error']
    except Exception as exc:
        fitted = {'predictions': {}, 'error': f'{type(exc).__name__}: {exc}'}
        result['3_models_error'] = fitted['error']
    print(f"{label}: block 3 complete",flush=True)
    # All model metrics use the held-out patients. Model-free quantities are also summarized on all eligible people.
    try:
        test = test_rows(data, split)
        test = test.loc[test[event_col].notna()].copy()
        y = test[event_col].astype(int).to_numpy()
        pid = test.who.to_numpy()
        predictions = {n: test[n].to_numpy(float) for n in BASELINES}
        fitted = fitted if isinstance(fitted, dict) else {'predictions': {}}
        pred_map = fitted.get('predictions', {})
        model_index = fitted.get('test_index', [])
        # Since fitting removes missing outcomes, align its predictions by original row indexes.
        for name in ['logistic_regression','gradient_boosting','within_centred_gradient_boosting']:
            if name in pred_map and len(pred_map[name]) == len(model_index):
                ser = pd.Series(pred_map[name], index=model_index)
                predictions[name] = ser.reindex(test.index).to_numpy(float)
        result['3_forecasts_available'] = list(predictions)
        write_forecasts(outdir / f'forecasts_{label}{suffix}.csv', test, 'who',
                        'study_week' if weekly else 'study_day', y, predictions, train_rate)
        validation = data.loc[data.who.isin(split['validation']) & data[event_col].notna()].copy()
        validation_predictions = {n: validation[n].to_numpy(float) for n in BASELINES}
        validation_index = fitted.get('validation_index', [])
        for name in ['logistic_regression','gradient_boosting','within_centred_gradient_boosting']:
            values = fitted.get('validation_predictions', {}).get(name)
            if values is not None and len(values) == len(validation_index):
                validation_predictions[name] = pd.Series(values, index=validation_index).reindex(validation.index).to_numpy(float)
        write_forecasts(outdir / f'forecasts_val_{label}{suffix}.csv', validation, 'who',
                        'study_week' if weekly else 'study_day', validation[event_col].astype(int).to_numpy(),
                        validation_predictions, train_rate)
    except Exception as exc:
        result['3_forecast_alignment_error'] = f'{type(exc).__name__}: {exc}'
        return result
    try:
        result['4_ranking_test'] = ranking_results(y, predictions, pid)
        result['4_benchmark_test'] = benchmark(y, pid, predictions)
        # All-included model-free replication requested for items 4, 5, and 8.
        all_df = data.loc[data[event_col].notna()].copy()
        all_y, all_pid = all_df[event_col].astype(int).to_numpy(), all_df.who.to_numpy()
        all_ref = {n: all_df[n].to_numpy(float) for n in BASELINES}
        result['4_ranking_all_model_free'] = ranking_results(all_y, all_ref, all_pid, nboot=1000, seed=SEED+130)
        result['4_benchmark_all_model_free'] = benchmark(all_y, all_pid, all_ref)
    except Exception as exc:
        result['4_error'] = f'{type(exc).__name__}: {exc}'
    print(f"{label}: block 4 complete",flush=True)
    try:
        threshold = train_rate
        result['5_threshold'] = {'threshold': threshold, 'metrics': {
            n: threshold_metrics(y, p, pid, threshold) for n,p in predictions.items()
            if n in ['logistic_regression','gradient_boosting','within_centred_gradient_boosting','runin_rate']}}
        all_df = data.loc[data[event_col].notna()]
        result['5_threshold_all_model_free'] = {'threshold': train_rate,
            'runin_rate': threshold_metrics(all_df[event_col].astype(int),all_df.runin_rate,all_df.who,
                                            train_rate)}
    except Exception as exc:
        result['5_error'] = f'{type(exc).__name__}: {exc}'
    print(f"{label}: block 5 complete",flush=True)
    try:
        refs = {n: test[n].to_numpy(float) for n in BASELINES}
        result['6_scores_and_skill'] = scores_and_skills(y, predictions, pid, refs)
        result['6_scores_and_skill']['retrospective_rate_expected_brier_optimism'] = retrospective_optimism(y,pid)
        add_skill_bootstrap(y, predictions, pid, refs, result['6_scores_and_skill'])
    except Exception as exc:
        result['6_error'] = f'{type(exc).__name__}: {exc}'
    print(f"{label}: block 6 complete",flush=True)
    try:
        # Calibration requested for the two pooled models, when available.
        cal_preds = {n: predictions[n] for n in ['logistic_regression','gradient_boosting'] if n in predictions}
        if cal_preds:
            result['7_calibration'] = calibration_results(y, cal_preds, pid)
        else:
            result['7_calibration'] = {'unavailable': 'No fitted model forecasts.'}
    except Exception as exc:
        result['7_error'] = f'{type(exc).__name__}: {exc}'
    print(f"{label}: block 7 complete",flush=True)
    try:
        # Test cohort and all included participants; observed-minus-permutation reported with person bootstrap CI.
        result['8_history_bias_test'] = history_bias(test, y, pid)
        all_df = data.loc[data[event_col].notna()].copy()
        result['8_history_bias_all_model_free'] = history_bias(all_df, all_df[event_col].astype(int), all_df.who)
    except Exception as exc:
        result['8_error'] = f'{type(exc).__name__}: {exc}'
    print(f"{label}: block 8 complete",flush=True)
    try:
        ap_preds = {n: predictions[n] for n in ['gradient_boosting','runin_rate'] if n in predictions}
        result['9_average_precision'] = ap_results(y, ap_preds, pid)
    except Exception as exc:
        result['9_error'] = f'{type(exc).__name__}: {exc}'
    print(f"{label}: block 9 complete",flush=True)
    return safe(result)


def selftest():
    assert abs(auc([0,1],[0,1])-1) < 1e-12
    assert abs(auc([0,1],[1,1])-.5) < 1e-12
    ap, exp = ap_tie_expected([1,0,1,0],[.5,.5,.1,.1])
    assert np.isfinite(ap) and np.isfinite(exp)
    assert abs(ap_tie_expected([1,0,1,0],[.5,.5,.5,.5])[1]-.5) < 1e-12
    x = np.linspace(-2,2,500)
    yy = (np.sin(x*2) > 0).astype(int)
    b = irls(np.column_stack([np.ones(len(x)),x]), yy)
    assert len(b)==2 and np.all(np.isfinite(b))
    assert abs(expit(0)-.5)<1e-12
    print('SELFTEST PASS: tie-aware AUC/AP, IRLS calibration, and probability transforms')


def markdown(results):
    blocks=results.get('blocks',results)
    lines = ['# CTN-0094 real-data analysis', '',
             'Primary: weekly next-week opioid-positive UDS; missing target weeks excluded. '
             'Sensitivity: missing weeks counted positive. Weekly run-in: weeks 0–3; eligibility: at least 12 observed screen weeks, '
             'including 2 run-in and 8 later weeks. Daily TLFB is exploratory: first 28 inferred coverage days are run-in, '
             'with at least 56 inferred coverage days required. Weeks are seven-day bins anchored at the first randomization.', '']
    lines += ['## Cohort', '', '| Analysis | Patients | Scored steps | Event rate | Mean patient rate | SD patient rates | ICC |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for key in ['weekly_missing_excluded','weekly_missing_positive','daily_self_reported']:
        r=blocks.get(key,{}).get('1_cohort',{})
        lines.append(f"| {key} | {r.get('patients','NA')} | {r.get('scored_steps','NA')} | {fmt(r.get('event_rate'))} | {fmt(r.get('patient_event_rate_mean'))} | {fmt(r.get('patient_event_rate_sd'))} | {fmt(r.get('icc'))} |")
    lines += ['', '## Ranking and benchmark (held-out patients)', '',
              '| Analysis | Forecast | Pooled AUC | 95% CI | Within-person AUC | 95% CI |',
              '|---|---|---:|---:|---:|---:|']
    for key in ['weekly_missing_excluded','weekly_missing_positive','daily_self_reported']:
        for name,r in blocks.get(key,{}).get('4_ranking_test',{}).items():
            lines.append(f"| {key} | {name} | {fmt(r.get('pooled_auc'))} | {fmtci(r.get('pooled_auc_ci'))} | {fmt(r.get('within_person_auc'))} | {fmtci(r.get('within_person_auc_ci'))} |")
        b=blocks.get(key,{}).get('4_benchmark_test',{}).get('normal_approximation',{})
        lines.append(f"| {key} | normal approximation, observed SD | {fmt(b.get('auc_observed_sd'))} | NA | NA | NA |")
        lines.append(f"| {key} | normal approximation, sampling-corrected SD | {fmt(b.get('auc_sampling_corrected_sd'))} | NA | NA | NA |")
    lines += ['', '## Scores and Brier skill (held-out patients)', '',
              '| Analysis | Forecast | Brier | Log loss | Brier skill: run-in / updating / retrospective | Log-loss skill: run-in / updating / retrospective |',
              '|---|---|---:|---:|---|---|']
    for key in ['weekly_missing_excluded','weekly_missing_positive','daily_self_reported']:
        for name,r in blocks.get(key,{}).get('6_scores_and_skill',{}).items():
            if not isinstance(r,dict) or 'brier_pooled' not in r:
                continue
            sk=r.get('skills',{})
            bs=' / '.join(fmt(sk.get(ref,{}).get('brier_skill_summed_losses')) for ref in ['runin_rate','updating_rate','retrospective_rate'])
            ls=' / '.join(fmt(sk.get(ref,{}).get('log_loss_skill_summed_losses')) for ref in ['runin_rate','updating_rate','retrospective_rate'])
            lines.append(f"| {key} | {name} | {fmt(r.get('brier_pooled'))} | {fmt(r.get('log_loss_pooled'))} | {bs} | {ls} |")
    lines += ['', '## Calibration and history bias', '',
              '| Analysis | Model/reference | Day-weighted pooled/between/within slopes | Equal-person pooled/between/within slopes | Between-logit variance share | History AUC observed / permutation / difference |',
              '|---|---|---|---|---:|---|']
    for key in ['weekly_missing_excluded','weekly_missing_positive','daily_self_reported']:
        for name,r in blocks.get(key,{}).get('7_calibration',{}).items():
            if not isinstance(r,dict):
                continue
            day=r.get('slopes',{}).get('day_weighted',{})
            eq=r.get('slopes',{}).get('equal_person_weighted',{})
            lines.append(f"| {key} | {name} | {fmt(day.get('pooled_slope'))} / {fmt(day.get('between_slope'))} / {fmt(day.get('within_slope'))} | {fmt(eq.get('pooled_slope'))} / {fmt(eq.get('between_slope'))} / {fmt(eq.get('within_slope'))} | {fmt(r.get('between_forecast_logit_variance_share'))} | NA |")
        for name,r in blocks.get(key,{}).get('8_history_bias_test',{}).items():
            lines.append(f"| {key} | {name} | NA | NA | NA | {fmt(r.get('observed_within_auc'))} / {fmt(r.get('permutation_reference_mean'))} / {fmt(r.get('observed_minus_reference'))} |")
    lines += ['', 'Patient bootstrap intervals and tie-matched average precision are fully recorded in the JSON file.',
              'Model fitting uses scikit-learn; when it is unavailable, block 3 records the import error and baseline metrics still run.',
              'UDS collection is proxied by a recorded temperature or any UDS finding; all UDS findings are positive-only. '
              'Daily TLFB non-use days are inferred between the first and last recorded TLFB drug-use days.', '']
    return '\n'.join(lines)


def fmt(x):
    return 'NA' if x is None else f'{x:.3f}' if isinstance(x,(int,float)) else str(x)


def fmtci(x):
    return 'NA' if not x else f'[{fmt(x[0])}, {fmt(x[1])}]'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weekly-panel', default=str(Path(__file__).resolve().parent / 'ctn0094_weekly_panel.csv'))
    ap.add_argument('--daily-panel', default=str(Path(__file__).resolve().parent / 'ctn0094_daily_panel.csv'))
    ap.add_argument('--output-json', default=str(Path(__file__).resolve().parent / 'ctn0094_results.json'))
    ap.add_argument('--output-md', default=str(Path(__file__).resolve().parent / 'ctn0094_results.md'))
    ap.add_argument('--with-current-dose', action='store_true',
                    help='Include dose_current in weekly models (legacy feature set).')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    suffix = '_with_current_dose' if args.with_current_dose else ''
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    if suffix:
        output_json = output_json.with_name(output_json.stem + suffix + output_json.suffix)
        output_md = output_md.with_name(output_md.stem + suffix + output_md.suffix)
    results = {'dataset':'CTN-0094 harmonised CTN-0027/0030/0051', 'seed':SEED,
               'primary_analysis':'weekly_missing_excluded', 'blocks':{}}
    results['analysis_design'] = {
        'weekly': {'time_step':'7-day study weeks, week 0 starts at first randomization study day',
                   'event':'at least one opioid-positive urine drug screen finding in the study week',
                   'sample_evidence':'uds_temp sample record or any UDS finding; week with no sample evidence is missing',
                   'primary_missing_rule':'exclude weeks without a recorded sample',
                   'sensitivity_missing_rule':'count weeks without a recorded sample as positive',
                   'run_in':'first 4 study weeks', 'minimum_sequence':'12 observed screen weeks, including at least 2 run-in and 8 later weeks'},
        'daily': {'time_step':'calendar study day over inclusive min/max TLFB drug-record span',
                  'event':'TLFB Heroin, Opioid, Methadone, or Buprenorphine category on the day',
                  'coverage':'inferred continuous daily coverage between first and last TLFB drug-use record',
                  'run_in':'first 28 inferred coverage days', 'minimum_sequence':'56 inferred coverage days'},
        'split':'patient-level 60/20/20 seeded train/validation/test'}
    try:
        weekly = pd.read_csv(args.weekly_panel)
        results['blocks']['weekly_missing_excluded'] = analyse('ctn_weekly_missing_excluded', weekly, 'event', True, Path(__file__).resolve().parent, args.with_current_dose, suffix)
    except Exception as exc:
        results['blocks']['weekly_missing_excluded'] = {'load_error': f'{type(exc).__name__}: {exc}'}
    try:
        weekly = pd.read_csv(args.weekly_panel)
        results['blocks']['weekly_missing_positive'] = analyse('ctn_weekly_missing_positive', weekly, 'event_missing_positive', True, Path(__file__).resolve().parent, args.with_current_dose, suffix)
    except Exception as exc:
        results['blocks']['weekly_missing_positive'] = {'load_error': f'{type(exc).__name__}: {exc}'}
    try:
        daily = pd.read_csv(args.daily_panel)
        results['blocks']['daily_self_reported'] = analyse('ctn_daily_self_reported', daily, 'event', False, Path(__file__).resolve().parent, args.with_current_dose, suffix)
    except Exception as exc:
        results['blocks']['daily_self_reported'] = {'load_error': f'{type(exc).__name__}: {exc}'}
    results = safe(results)
    try:
        output_json.write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
    except Exception as exc:
        print(f"block 10 JSON error: {type(exc).__name__}: {exc}")
    try:
        output_md.write_text(markdown(results), encoding='utf-8')
    except Exception as exc:
        print(f"block 10 Markdown error: {type(exc).__name__}: {exc}")
    print('CTN-0094 analysis complete')
    for name, block in results['blocks'].items():
        c = block.get('1_cohort', {})
        print(f"{name}: patients={c.get('patients','NA')}, scored={c.get('scored_steps','NA')}, "
              f"event_rate={fmt(c.get('event_rate'))}, patient_rate_mean={fmt(c.get('patient_event_rate_mean'))}, "
              f"patient_rate_sd={fmt(c.get('patient_event_rate_sd'))}; model_error={block.get('3_models',{}).get('error')}")
    print(f"Wrote {output_json} and {output_md}")


if __name__ == '__main__':
    main()
