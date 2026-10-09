"""Metrics for repeated binary outcomes and patient-day forecasts."""
from __future__ import annotations

import warnings
import math
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


def _check_missing(name: str, values: np.ndarray) -> None:
    if pd.isna(values).any():
        raise ValueError(f"{name} contains missing values")


def _outcomes(y: Any) -> np.ndarray:
    yy = np.asarray(y)
    _check_missing("y", yy)
    try:
        valid = np.isin(yy, [0, 1])
    except (TypeError, ValueError) as exc:
        raise ValueError("y must contain only 0 and 1") from exc
    if not valid.all():
        raise ValueError("y must contain only 0 and 1")
    return yy.astype(int)


def _probabilities(name: str, values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    _check_missing(name, arr)
    if not np.isfinite(arr).all() or ((arr < 0) | (arr > 1)).any():
        raise ValueError(f"{name} must contain probabilities in [0, 1]")
    return arr


def _arrays(patient: Any, y: Any, forecast: Any = None) -> tuple[np.ndarray, ...]:
    values = [np.asarray(patient), _outcomes(y)]
    if forecast is not None:
        values.append(np.asarray(forecast, dtype=float))
    if any(len(v) != len(values[0]) for v in values):
        raise ValueError("patient, y, and forecast must have equal length")
    _check_missing("patient", values[0])
    if forecast is not None:
        _check_missing("forecast", values[2])
        if not np.isfinite(values[2]).all():
            raise ValueError("forecast must contain finite values")
    return tuple(values)


def _groups(patient: np.ndarray) -> list[tuple[Any, np.ndarray]]:
    return [(key, np.flatnonzero(patient == key)) for key in pd.unique(patient)]


def auc(y: Any, s: Any) -> float:
    """Return rank-based AUC, counting tied event/non-event scores as one half; NaN if either class is absent."""
    yy = _outcomes(y)
    score = np.asarray(s, dtype=float)
    if len(yy) != len(score):
        raise ValueError("y and s must have equal length")
    _check_missing("s", score)
    if not np.isfinite(score).all():
        raise ValueError("s must contain finite scores")
    n1 = int(np.sum(yy == 1)); n0 = int(np.sum(yy == 0))
    if not n1 or not n0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    starts = np.r_[0, np.flatnonzero(sorted_score[1:] != sorted_score[:-1]) + 1]
    ends = np.r_[starts[1:], len(score)]
    ranks = np.empty(len(score), dtype=float)
    for lo, hi in zip(starts, ends):
        ranks[order[lo:hi]] = (lo + 1 + hi) / 2
    return float((ranks[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _auc_summary(patient: Any, y: Any, forecast: Any, weights: str, pooled: bool) -> dict[str, Any]:
    pid, yy, ss = _arrays(patient, y, forecast)
    if weights not in {"patient", "pairs", "days"}:
        raise ValueError("weights must be 'patient', 'pairs', or 'days'")
    per: dict[Any, float] = {}
    counts: dict[Any, dict[str, int]] = {}
    for key, ix in _groups(pid):
        v = auc(yy[ix], ss[ix])
        counts[key] = {"days": len(ix), "pairs": int(np.sum(yy[ix] == 1) * np.sum(yy[ix] == 0))}
        if np.isfinite(v):
            per[key] = v
    keys = list(per)
    if pooled:
        value = auc(yy, ss)
    elif not keys:
        value = float("nan")
    elif weights == "patient":
        value = float(np.mean([per[k] for k in keys]))
    else:
        ww = np.array([counts[k][weights] for k in keys], dtype=float)
        value = float(np.average([per[k] for k in keys], weights=ww))
    return {"value": value, "per_patient": per, "eligible": len(per), "excluded": len(counts) - len(per), "weights": weights}


def pooled_auc(patient: Any, y: Any, forecast: Any, weights: str = "patient") -> dict[str, Any]:
    """Return pooled day-level AUC plus per-patient AUCs and eligibility counts; weights describes within-patient summary weighting."""
    return _auc_summary(patient, y, forecast, weights, pooled=True)


def within_patient_auc(patient: Any, y: Any, forecast: Any, weights: str = "patient") -> dict[str, Any]:
    """Return the mean eligible patient AUC, its per-patient values, and eligible/excluded counts under patient, pair, or day weighting."""
    return _auc_summary(patient, y, forecast, weights, pooled=False)


def per_patient_auc_summary(patient: Any, y: Any, forecast: Any) -> dict[str, Any]:
    """Summarise the distribution of per-patient AUCs (median, IQR, share strictly above 0.5) over eligible patients."""
    summary = within_patient_auc(patient, y, forecast)
    vals = np.asarray(list(summary["per_patient"].values()), dtype=float)
    if not len(vals):
        return {"patients": 0, "excluded": summary["excluded"], "mean": float("nan"), "median": float("nan"), "iqr": [float("nan"), float("nan")], "share_above_0_5": float("nan")}
    q25, q75 = np.percentile(vals, [25, 75])
    return {"patients": len(vals), "excluded": summary["excluded"], "mean": float(np.mean(vals)), "median": float(np.median(vals)),
            "iqr": [float(q25), float(q75)], "share_above_0_5": float(np.mean(vals > 0.5))}


def per_time_point_auc(df: pd.DataFrame, time_col: str, min_patients: int = 5, patient: str = "patient",
                       y: str = "y", forecast: str = "forecast") -> dict[str, Any]:
    """Return AUC within each time point, using time points with both outcomes and at least min_patients distinct patients.

    ``value`` is the equal-weight mean over used time points; ``comparison_weighted`` weights each time point by its
    number of event/non-event pairs. ``per_time`` maps each used time value to its AUC.
    """
    missing = [c for c in (patient, y, forecast, time_col) if c not in df]
    if missing:
        raise ValueError(f"missing dataframe columns: {', '.join(missing)}")
    pid, yy, ss = _arrays(df[patient].to_numpy(), df[y].to_numpy(), df[forecast].to_numpy())
    tt = df[time_col].to_numpy()
    _check_missing(time_col, tt)
    per: dict[Any, float] = {}; pairs: dict[Any, int] = {}; excluded: list[Any] = []
    for key, ix in _groups(tt):
        n1 = int(np.sum(yy[ix] == 1)); n0 = len(ix) - n1
        if n1 and n0 and len(pd.unique(pid[ix])) >= min_patients:
            per[key] = auc(yy[ix], ss[ix]); pairs[key] = n1 * n0
        else:
            excluded.append(key)
    if per:
        vals = np.array([per[k] for k in per], dtype=float)
        value = float(vals.mean())
        median = float(np.median(vals))
        iqr = [float(v) for v in np.percentile(vals, [25, 75])]
        comparison_weighted = float(np.average(vals, weights=np.array([pairs[k] for k in per], dtype=float)))
    else:
        value = comparison_weighted = float("nan")
        median = float("nan")
        iqr = [float("nan"), float("nan")]
    return {"value": value, "comparison_weighted": comparison_weighted, "time_points_used": len(per),
            "time_points_excluded": len(excluded), "per_time": per, "median": median, "iqr": iqr,
            "pairs": pairs, "min_patients": int(min_patients)}


def rate_benchmark(patient: Any, y: Any, rates: Any = None) -> float:
    """Return pooled AUC for patient-rate scores; rates defaults to the scored-sample rate within each patient."""
    pid, yy = _arrays(patient, y)
    if rates is None:
        rr = np.empty(len(yy), dtype=float)
        for _, ix in _groups(pid):
            rr[ix] = np.mean(yy[ix])
    else:
        rr = np.asarray(rates, dtype=float)
        if len(rr) != len(yy):
            raise ValueError("rates must have the same length as y")
    return auc(yy, rr)


def rate_benchmark_from_summary(mean: float, sd: float) -> float:
    """Return the paper's normal-rate approximation A0; valid only for roughly normal patient rates, with p and p' rates of randomly drawn patient-days."""
    mu = float(mean)
    if not 0 < mu < 1:
        return float("nan")
    return float(0.5 + (2 * float(sd) / np.sqrt(np.pi)) / (4 * mu * (1 - mu)))


def threshold_metrics(patient: Any, y: Any, forecast: Any, threshold: float, time: Any = None,
                      per_patient: bool = False) -> dict[str, Any]:
    """Return pooled, descriptive patient-mean, and paper-defined within-patient classification summaries.

    The ``within_patient`` block averages sensitivity, specificity, and Youden index
    over patients with both outcomes. ``patient_mean`` is descriptive only because
    its sensitivity and specificity may average over different patient sets.
    """
    pid, yy, pp = _arrays(patient, y, forecast)
    pp = _probabilities("forecast", pp)
    truth = yy == 1; pred = pp >= threshold
    def ratio(num: int, den: int) -> float:
        return float(num / den) if den else float("nan")
    sens = ratio(int(np.sum(pred & truth)), int(np.sum(truth)))
    spec = ratio(int(np.sum((~pred) & (~truth))), int(np.sum(~truth)))
    ppv = ratio(int(np.sum(pred & truth)), int(np.sum(pred)))
    npv = ratio(int(np.sum((~pred) & (~truth))), int(np.sum(~pred)))
    per_s, per_c, both_s, both_c = [], [], [], []
    alarms: dict[Any, int] = {}
    alarms_30: list[float] = []
    alarm_rates: dict[Any, float] = {}
    elapsed: dict[Any, float] = {}
    tt = None if time is None else np.asarray(time)
    if tt is not None:
        if len(tt) != len(pid): raise ValueError("time must have the same length as patient")
        _check_missing("time", tt)
        try: tt = tt.astype(float)
        except (TypeError, ValueError) as exc: raise ValueError("time must be numeric elapsed time") from exc
        if not np.isfinite(tt).all(): raise ValueError("time must be finite elapsed time")
    for key, ix in _groups(pid):
        t, h = truth[ix], pred[ix]
        if t.any(): per_s.append(float(np.mean(h[t])))
        if (~t).any(): per_c.append(float(np.mean(~h[~t])))
        if t.any() and (~t).any():
            both_s.append(float(np.mean(h[t]))); both_c.append(float(np.mean(~h[~t])))
        alarms[key] = int(h.sum())
        alarms_30.append(float(h.sum() * 30 / len(ix)))
        if tt is not None:
            duration = float(np.max(tt[ix]) - np.min(tt[ix]))
            elapsed[key] = duration
            alarm_rates[key] = float(h.sum() / duration) if duration > 0 else float("nan")
    ms = float(np.mean(per_s)) if per_s else float("nan")
    mc = float(np.mean(per_c)) if per_c else float("nan")
    return {"threshold": float(threshold), "pooled": {"sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
            "youden": sens + spec - 1 if np.isfinite(sens) and np.isfinite(spec) else float("nan")},
        "patient_mean": {"sensitivity": ms, "specificity": mc},
        # Patients with both outcomes only: the two rates then refer to the same patients, so a forecast that is
        # constant within every patient has a within-patient Youden index of exactly zero.
        "within_patient": {"sensitivity": float(np.mean(both_s)) if both_s else float("nan"),
            "specificity": float(np.mean(both_c)) if both_c else float("nan"),
            "youden": float(np.mean(np.array(both_s) + np.array(both_c) - 1)) if both_s else float("nan")},
        "eligible": {"sensitivity": len(per_s), "specificity": len(per_c), "both": len(both_s)},
        # Alarms per patient per 30 scored rows (per 30 days for daily data), as reported in the paper.
        "alarms_per_patient_30_days": {"mean": float(np.mean(alarms_30)), "median": float(np.median(alarms_30))},
        "alarm_counts": {"mean": float(np.mean(list(alarms.values()))), "median": float(np.median(list(alarms.values()))),
            "total": int(sum(alarms.values()))},
        "alarms_per_patient_per_time_unit": _alarm_rate_summary(alarm_rates, elapsed, per_patient) if tt is not None else None,
        **({"alarm_counts_per_patient": alarms} if per_patient else {})}


def _alarm_rate_summary(alarm_rates: dict, elapsed: dict, per_patient: bool) -> dict[str, Any]:
    vals = np.array(list(alarm_rates.values()), dtype=float)
    ok = vals.size > 0 and np.isfinite(vals).any()
    out = {"mean": float(np.nanmean(vals)) if ok else float("nan"),
           "median": float(np.nanmedian(vals)) if ok else float("nan"),
           "unit": "time units in the supplied time column"}
    if per_patient:
        out.update(per_patient=alarm_rates, elapsed_time_per_patient=elapsed)
    return out


def predictive_value_excess(patient: Any, y: Any, forecast: Any, threshold: float = 0.5) -> dict[str, Any]:
    """Return per-patient PPV minus own event rate (patients with an alarm) and NPV minus own non-event rate (patients with a non-alarm).

    Alarms are forecasts at or above ``threshold``. Own rates use all scored days of the patient. Means are equal-weight
    over the patients contributing to each quantity. ``pooled`` gives day-pooled PPV, NPV, and the cohort event rate.
    """
    pid, yy, pp = _arrays(patient, y, forecast)
    pp = _probabilities("forecast", pp)
    alarm = pp >= threshold; truth = yy == 1
    ppv_excess: dict[Any, float] = {}; npv_excess: dict[Any, float] = {}
    for key, ix in _groups(pid):
        rate = float(np.mean(yy[ix])); a = alarm[ix]; t = truth[ix]
        if a.any(): ppv_excess[key] = float(np.mean(t[a]) - rate)
        if (~a).any(): npv_excess[key] = float(np.mean(~t[~a]) - (1 - rate))
    n_alarm = int(alarm.sum()); n_quiet = len(yy) - n_alarm
    return {"threshold": float(threshold),
            "ppv_excess": {"mean": float(np.mean(list(ppv_excess.values()))) if ppv_excess else float("nan"), "patients": len(ppv_excess), "per_patient": ppv_excess},
            "npv_excess": {"mean": float(np.mean(list(npv_excess.values()))) if npv_excess else float("nan"), "patients": len(npv_excess), "per_patient": npv_excess},
            "pooled": {"ppv": float(np.sum(alarm & truth) / n_alarm) if n_alarm else float("nan"),
                       "npv": float(np.sum(~alarm & ~truth) / n_quiet) if n_quiet else float("nan"),
                       "cohort_rate": float(np.mean(yy)) if len(yy) else float("nan")}}


def _losses(y: Any, forecast: Any) -> tuple[float, float]:
    yy = _outcomes(y).astype(float); pp = np.clip(_probabilities("forecast", forecast), 1e-6, 1 - 1e-6)
    return float(np.mean((yy - pp) ** 2)), float(np.mean(-(yy * np.log(pp) + (1 - yy) * np.log1p(-pp))))


def brier_scores(patient: Any, y: Any, forecast: Any) -> dict[str, Any]:
    """Return day-pooled and equal-patient mean Brier score and log-loss."""
    pid, yy, pp = _arrays(patient, y, forecast)
    pp = _probabilities("forecast", pp)
    pb, pl = _losses(yy, pp)
    scores = [_losses(yy[ix], pp[ix]) for _, ix in _groups(pid)]
    return {"pooled": {"brier": pb, "log_loss": pl}, "patient_mean": {"brier": float(np.mean([x[0] for x in scores])), "log_loss": float(np.mean([x[1] for x in scores]))}}


def population_rate(y: Any) -> np.ndarray:
    """Return the scored cohort's overall event rate repeated for every row."""
    yy = _outcomes(y).astype(float)
    return np.full(len(yy), np.mean(yy) if len(yy) else np.nan)


def retrospective_patient_rate(patient: Any, y: Any) -> np.ndarray:
    """Return each patient's rate over the supplied outcomes; this is fitted to the scored outcomes and is not available in advance."""
    pid, yy = _arrays(patient, y)
    out = np.empty(len(yy), dtype=float)
    for _, ix in _groups(pid): out[ix] = np.mean(yy[ix])
    return out


def runin_patient_rate(patient: Any, y: Any, time: Any, runin_days: float, prior_weight: float = 0) -> np.ndarray:
    """Return each patient's smoothed rate from rows with time before runin_days, repeated across that patient's rows."""
    pid, yy = _arrays(patient, y); tt = np.asarray(time)
    if len(tt) != len(yy): raise ValueError("time must have the same length as y")
    _check_missing("time", tt)
    out = np.empty(len(yy), dtype=float)
    for _, ix in _groups(pid):
        mask = tt[ix] < runin_days
        base = (float(np.sum(yy[ix][mask])) + prior_weight * 0.5) / (int(mask.sum()) + prior_weight) if mask.sum() + prior_weight else float("nan")
        out[ix] = base
    return out


def expanding_patient_rate(patient: Any, y: Any, time: Any, prior_rate: Any = None, prior_weight: Any = 1.0) -> np.ndarray:
    """Return each row's rate from strictly earlier patient rows and an optional prior.

    ``prior_rate`` may be a scalar, a row-aligned array, or ``None``. With no prior and
    no earlier row, the reference is NaN. Rows at the same time never enter one another's reference.
    ``prior_weight`` is a nonnegative pseudo-count and may be scalar or row-aligned.
    """
    pid, yy = _arrays(patient, y); tt = np.asarray(time)
    if len(tt) != len(yy): raise ValueError("time must have the same length as y")
    _check_missing("time", tt)
    raw_weight = np.asarray(prior_weight, dtype=float)
    if raw_weight.ndim == 0:
        if not np.isfinite(raw_weight) or float(raw_weight) < 0: raise ValueError("prior_weight must be finite and nonnegative")
    elif len(raw_weight) != len(yy) or not np.isfinite(raw_weight).all() or (raw_weight < 0).any():
        raise ValueError("prior_weight must be finite, nonnegative, and scalar or have the same length as y")
    if prior_rate is None:
        prior = np.full(len(yy), np.nan, dtype=float)
        weights = np.zeros(len(yy), dtype=float)
    else:
        prior = np.asarray(prior_rate, dtype=float)
        if prior.ndim == 0: prior = np.full(len(yy), float(prior))
        if len(prior) != len(yy): raise ValueError("prior_rate must be scalar or have the same length as y")
        if (np.isfinite(prior) & ((prior < 0) | (prior > 1))).any(): raise ValueError("prior_rate must contain probabilities in [0, 1]")
        if raw_weight.ndim == 0: weights = np.full(len(yy), float(raw_weight))
        else:
            if len(raw_weight) != len(yy): raise ValueError("prior_weight must be scalar or have the same length as y")
            weights = raw_weight.copy()
        if not np.isfinite(weights).all() or (weights < 0).any(): raise ValueError("prior_weight must be finite and nonnegative")
        weights[~np.isfinite(prior)] = 0.0
    out = np.empty(len(yy), dtype=float)
    for key, ix in _groups(pid):
        for j in ix:
            earlier = ix[tt[ix] < tt[j]]
            denom = len(earlier) + weights[j]
            out[j] = (float(yy[earlier].sum()) + (prior[j] if weights[j] else 0.0) * weights[j]) / denom if denom else float("nan")
    return out


def population_a0_moment_corrected(patient: Any, y: Any) -> float:
    """Estimate population A0 after subtracting binomial sampling noise from patient-rate variance.

    Uses day weights, the S4 correction ``mean(q*(1-q)/max(1,n-1))``, truncates corrected
    variance at zero, and maps the corrected SD to A0 with the paper's normal-CDF approximation.
    """
    pid, yy = _arrays(patient, y)
    if not len(yy): return float("nan")
    rates = np.asarray([float(np.mean(yy[ix])) for _, ix in _groups(pid)], dtype=float)
    sizes = np.asarray([len(ix) for _, ix in _groups(pid)], dtype=float)
    weights = sizes / sizes.sum()
    mu = float(np.average(rates, weights=weights))
    if not 0 < mu < 1: return float("nan")
    observed_var = float(np.average((rates - mu) ** 2, weights=weights))
    sampling_var = float(np.average(rates * (1 - rates) / np.maximum(1.0, sizes - 1), weights=weights))
    corrected_sd = math.sqrt(max(0.0, observed_var - sampling_var))
    z = corrected_sd / (math.sqrt(2.0) * mu * (1 - mu))
    return float(0.5 * (1 + math.erf(z / math.sqrt(2.0))))


def pooled_auc_permutation_test(patient: Any, y: Any, forecast: Any, n_perm: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Test pooled AUC under within-patient forecast shuffles with outcomes fixed.

    Each draw independently permutes forecast rows within each patient's scored record.
    The one-sided p-value is ``(count(null >= observed) + 1) / (n_perm + 1)``.
    """
    pid, yy, pp = _arrays(patient, y, forecast)
    pp = _probabilities("forecast", pp)
    if n_perm < 1: raise ValueError("n_perm must be at least 1")
    observed = auc(yy, pp)
    if not np.isfinite(observed):
        empty = [float("nan"), float("nan")]
        return {"observed": float("nan"), "null_mean": float("nan"), "null_percentiles": empty,
                "one_sided_p_value": float("nan"), "permutations": int(n_perm), "per_patient": {}}
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    groups = _groups(pid)
    patient_draws: dict[Any, list[float]] = {
        key: [] for key, ix in groups if np.isfinite(auc(yy[ix], pp[ix]))
    }
    for b in range(n_perm):
        permuted = pp.copy()
        for key, ix in groups:
            permuted[ix] = pp[ix[rng.permutation(len(ix))]]
            if key in patient_draws:
                patient_draws[key].append(auc(yy[ix], permuted[ix]))
        null[b] = auc(yy, permuted)
    per_patient: dict[Any, dict[str, Any]] = {}
    for key, ix in groups:
        observed_patient = auc(yy[ix], pp[ix])
        if np.isfinite(observed_patient):
            draws = np.asarray(patient_draws[key], dtype=float)
            per_patient[key] = {"observed": float(observed_patient), "reference_mean": float(np.mean(draws)),
                                "reference_percentiles": [float(v) for v in np.percentile(draws, [2.5, 97.5])]}
    return {"observed": float(observed), "null_mean": float(np.mean(null)),
            "null_percentiles": [float(v) for v in np.percentile(null, [2.5, 97.5])],
            "one_sided_p_value": float((int(np.sum(null >= observed)) + 1) / (n_perm + 1)),
            "permutations": int(n_perm), "per_patient": per_patient}


def brier_skill_summary(patient: Any, y: Any, forecast: Any, reference: Any) -> dict[str, Any]:
    """Return pooled and equal-patient Brier skill plus patient-specific skill ratios.

    Rows with an undefined reference are excluded. A patient ratio is undefined when its reference
    Brier loss is zero; such patients are listed as excluded from patient-level summaries.
    """
    pid, yy, pp = _arrays(patient, y, forecast)
    rr = np.asarray(reference, dtype=float)
    if len(rr) != len(yy): raise ValueError("reference must have the same length as y")
    valid = np.isfinite(rr)
    if (valid & ((rr < 0) | (rr > 1))).any(): raise ValueError("reference must contain probabilities in [0, 1]")
    per: dict[Any, float] = {}
    for key, ix in _groups(pid):
        use = ix[valid[ix]]
        if not len(use): continue
        den = float(np.sum((yy[use] - rr[use]) ** 2))
        if den > 0:
            per[key] = float(1 - np.sum((yy[use] - pp[use]) ** 2) / den)
    used = np.flatnonzero(valid)
    pooled_den = float(np.sum((yy[used] - rr[used]) ** 2))
    pooled = float(1 - np.sum((yy[used] - pp[used]) ** 2) / pooled_den) if len(used) and pooled_den > 0 else float("nan")
    return {"pooled": pooled, "patient_mean": float(np.mean(list(per.values()))) if per else float("nan"),
            "positive_share": float(np.mean(np.asarray(list(per.values())) > 0)) if per else float("nan"),
            "per_patient": per, "eligible_patients": len(per), "excluded_patients": int(len(_groups(pid)) - len(per)),
            "scored_rows": int(len(used)), "excluded_rows": int(len(yy) - len(used))}


def brier_skill(y: Any, forecast: Any, reference: Any) -> float:
    """Return one minus forecast Brier loss divided by reference Brier loss, using summed losses."""
    yy = _outcomes(y).astype(float); pp = _probabilities("forecast", forecast); rr = _probabilities("reference", reference)
    if not (len(yy) == len(pp) == len(rr)): raise ValueError("inputs must have equal length")
    den = float(np.sum((yy - rr) ** 2))
    return 1 - float(np.sum((yy - pp) ** 2)) / den if den > 0 else float("nan")


def _expit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35, 35)
    return 1 / (1 + np.exp(-x))


def _clipped_logit(pp: np.ndarray) -> np.ndarray:
    clipped = np.clip(pp, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def _irls(X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
    design = np.column_stack([np.ones(len(y)), X])
    beta = np.zeros(design.shape[1]); beta[0] = np.log(np.mean(y) / (1 - np.mean(y))) if 0 < np.mean(y) < 1 else 0
    for _ in range(100):
        p = _expit(design @ beta); w = np.maximum(p * (1 - p), 1e-10)
        h = design.T @ (w[:, None] * design)
        try: step = np.linalg.solve(h + np.eye(len(beta)) * 1e-10, design.T @ (y - p))
        except np.linalg.LinAlgError: return None
        beta += step
        if np.max(np.abs(step)) < 1e-8: return beta
    return None


def calibration_slopes(patient: Any, y: Any, forecast: Any) -> dict[str, Any]:
    """Fit pooled and patient-mean/daily-deviation logistic calibration slopes on clipped forecast logits; warn and return NaN when a fit fails or a covariate is constant.

    ``within_slope`` is the within-deviation coefficient of the joint model fitted on the patient-mean logit and the
    daily deviation from it. It is not a conditional (within-patient) slope; see ``conditional_within_slope``.
    """
    pid, yy, pp = _arrays(patient, y, forecast)
    pp = _probabilities("forecast", pp)
    yy = yy.astype(float); logits = np.log(np.clip(pp, 1e-6, 1 - 1e-6) / (1 - np.clip(pp, 1e-6, 1 - 1e-6)))
    means = np.empty(len(yy), dtype=float)
    for _, ix in _groups(pid): means[ix] = np.mean(logits[ix])
    dev = logits - means
    def fit(x: np.ndarray) -> tuple[float, float]:
        if np.ptp(x) <= 1e-12:
            warnings.warn("calibration covariate has no variation", RuntimeWarning)
            return float("nan"), float("nan")
        b = _irls(x[:, None], yy)
        if b is None:
            warnings.warn("calibration IRLS did not converge", RuntimeWarning)
            return float("nan"), float("nan")
        return float(b[0]), float(b[1])
    intercept, slope = fit(logits)
    joint = _irls(np.column_stack([means, dev]), yy)
    if joint is None:
        warnings.warn("joint calibration IRLS did not converge", RuntimeWarning)
        between = within = float("nan"); joint_intercept = float("nan")
    elif np.ptp(means) <= 1e-12 or np.ptp(dev) <= 1e-12:
        warnings.warn("joint calibration covariate has no variation", RuntimeWarning)
        joint_intercept = between = within = float("nan")
    else:
        joint_intercept, between, within = map(float, joint)
    return {"pooled_intercept": intercept, "pooled_slope": slope, "joint_intercept": joint_intercept, "between_slope": between, "within_slope": within}


def conditional_within_slope(patient: Any, y: Any, forecast: Any) -> dict[str, Any]:
    """Return the within-patient calibration slope from conditional logistic regression of y on the clipped forecast logit.

    Patients enter only if they have both outcomes; their intercepts are conditioned out, so the slope uses
    within-patient variation alone. If no such patient has within-patient logit variation, or the fit fails, the
    slope and SE are NaN and ``reason`` says why (a RuntimeWarning is also issued). Requires statsmodels.
    """
    pid, yy, pp = _arrays(patient, y, forecast)
    pp = _probabilities("forecast", pp)
    logits = _clipped_logit(pp)
    both = [ix for _, ix in _groups(pid) if 0 < int(yy[ix].sum()) < len(ix)]
    out: dict[str, Any] = {"slope": float("nan"), "se": float("nan"), "patients": len(both), "estimable": False, "reason": ""}
    if not both:
        out["reason"] = "no patient has both outcomes"
    elif not any(np.ptp(logits[ix]) > 1e-12 for ix in both):
        out["reason"] = "forecast logit does not vary within any patient with both outcomes"
    else:
        try:
            from statsmodels.discrete.conditional_models import ConditionalLogit
        except ImportError:
            out["reason"] = "statsmodels is not installed"
        else:
            keep = np.concatenate(both)
            codes = np.concatenate([np.full(len(ix), j) for j, ix in enumerate(both)])
            try:
                fit = ConditionalLogit(yy[keep].astype(float), logits[keep][:, None], groups=codes).fit(disp=False)
                beta, se = float(fit.params[0]), float(fit.bse[0])
            except Exception as exc:  # statsmodels raises several exception types for degenerate fits
                out["reason"] = f"conditional logit fit failed: {exc}"
            else:
                if np.isfinite(beta) and np.isfinite(se) and se > 0:
                    out.update(slope=beta, se=se, estimable=True)
                else:
                    out["reason"] = "conditional logit returned a non-finite estimate"
    if not out["estimable"]:
        warnings.warn(f"conditional within-patient calibration slope not estimable: {out['reason']}", RuntimeWarning)
    return out


def patient_calibration_in_the_large(patient: Any, y: Any, forecast: Any, tolerance: float = 0.05) -> dict[str, Any]:
    """Return each patient's observed event rate minus mean forecast, summarised over all patients (no exclusions).

    ``share_within_tolerance`` is the share of patients with absolute difference at most ``tolerance``.
    """
    pid, yy, pp = _arrays(patient, y, forecast)
    pp = _probabilities("forecast", pp)
    diffs: dict[Any, float] = {key: float(np.mean(yy[ix]) - np.mean(pp[ix])) for key, ix in _groups(pid)}
    if not diffs:
        return {"mean": float("nan"), "median": float("nan"), "iqr": [float("nan"), float("nan")],
                "tolerance": float(tolerance), "share_within_tolerance": float("nan"), "patients": 0, "per_patient": diffs}
    d = np.asarray(list(diffs.values()), dtype=float)
    q25, q75 = np.percentile(d, [25, 75])
    return {"mean": float(d.mean()), "median": float(np.median(d)), "iqr": [float(q25), float(q75)],
            "tolerance": float(tolerance), "share_within_tolerance": float(np.mean(np.abs(d) <= tolerance + 1e-12)),
            "patients": len(d), "per_patient": diffs}


def permutation_reference(patient: Any, y: Any, time: Any, history_forecast_fn: Callable[[np.ndarray], Any], n_perm: int = 200, seed: int = 0,
                          runin_days: Optional[float] = None) -> dict[str, Any]:
    """Compare observed within AUC with forecasts recomputed after within-patient day permutations; the reference represents exchangeable days and the difference is a contrast with that hypothesis, not a corrected estimate.

    With ``runin_days``, days before it are a fixed run-in history: they are not permuted, and ``history_forecast_fn``
    receives the run-in outcomes followed by the permuted scored outcomes (time order). AUC uses scored days only.
    """
    pid, yy = _arrays(patient, y); tt = np.asarray(time)
    if n_perm < 1: raise ValueError("n_perm must be at least 1")
    if len(tt) != len(yy): raise ValueError("time must have the same length as y")
    _check_missing("time", tt)
    groups = [(k, ix[np.argsort(tt[ix], kind="mergesort")]) for k, ix in _groups(pid)]
    n_run = [0 if runin_days is None else int(np.sum(tt[ix] < runin_days)) for _, ix in groups]
    def score(data: list[np.ndarray]) -> dict[Any, float]:
        vals: dict[Any, float] = {}
        for (key, ix), k, group_y in zip(groups, n_run, data):
            full = np.concatenate([yy[ix][:k], group_y])
            pred = _probabilities("history forecast", history_forecast_fn(full))
            if len(pred) != len(full): raise ValueError("history_forecast_fn must return one forecast per outcome")
            if not np.isfinite(pred).all(): raise ValueError("history_forecast_fn must return finite forecasts")
            a = auc(group_y, pred[k:])
            if np.isfinite(a): vals[key] = float(a)
        return vals
    observed_data = [yy[ix][k:] for (_, ix), k in zip(groups, n_run)]
    observed_by_patient = score(observed_data); rng = np.random.default_rng(seed); draws = []
    patient_draws: dict[Any, list[float]] = {key: [] for key in observed_by_patient}
    for _ in range(n_perm):
        permuted = []
        for arr in observed_data:
            permuted.append(rng.permutation(arr))
        values = score(permuted)
        draws.append(float(np.mean(list(values.values()))) if values else float("nan"))
        for key in patient_draws:
            patient_draws[key].append(values.get(key, float("nan")))
    finite_draws = np.asarray(draws, dtype=float)
    finite_draws = finite_draws[np.isfinite(finite_draws)]
    observed = float(np.mean(list(observed_by_patient.values()))) if observed_by_patient else float("nan")
    ref = float(np.mean(finite_draws)) if len(finite_draws) else float("nan")
    per_patient: dict[Any, dict[str, Any]] = {}
    for key, values in patient_draws.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        ref_mean = float(np.mean(finite)) if len(finite) else float("nan")
        per_patient[key] = {
            "observed": observed_by_patient[key],
            "reference_mean": ref_mean,
            "reference_percentiles": [float(v) for v in np.percentile(finite, [2.5, 97.5])] if len(finite) else [float("nan"), float("nan")],
            "difference": float(observed_by_patient[key] - ref_mean) if np.isfinite(ref_mean) else float("nan"),
        }
    return {"observed": observed, "reference_mean": ref, "difference": observed - ref,
            "reference_percentiles": [float(v) for v in np.percentile(finite_draws, [2.5, 97.5])] if len(finite_draws) else [float("nan"), float("nan")],
            "permutations": int(n_perm), "per_patient": per_patient}


def patient_bootstrap(fn: Callable[..., float], patient: Any, *arrays: Any, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Resample patients with replacement and return the original estimate, bootstrap values, and percentile 95% interval."""
    pid = np.asarray(patient); groups = _groups(pid)
    if not groups: return {"estimate": float("nan"), "interval": [float("nan"), float("nan")], "samples": np.array([])}
    values = [np.asarray(a) for a in arrays]
    if any(len(a) != len(pid) for a in values): raise ValueError("all arrays must match patient length")
    estimate = float(fn(pid, *values)); rng = np.random.default_rng(seed); draws = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(groups), size=len(groups)); pieces = [groups[i][1] for i in chosen]
        ix = np.concatenate(pieces); boot_pid = np.concatenate([np.full(len(part), j) for j, part in enumerate(pieces)])
        draws.append(float(fn(boot_pid, *(a[ix] for a in values))))
    finite = np.asarray(draws, dtype=float); finite = finite[np.isfinite(finite)]
    interval = np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else [float("nan"), float("nan")]
    return {"estimate": estimate, "interval": [float(interval[0]), float(interval[1])], "samples": np.asarray(draws)}
