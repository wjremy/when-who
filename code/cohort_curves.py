"""Cohort-matched reference curves extending signal_sweep.py."""
import csv
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import signal_sweep as ss

COHORTS = {
    "REPLACE-BG": {"rate": 0.059, "sd": 0.052, "n": 208, "runin": 14, "a0": 0.752},
    "CTN-0094": {"rate": 0.312, "sd": 0.340, "n": 17, "runin": 4, "a0": 0.932},
    "College Experience": {"rate": 0.382, "sd": 0.314, "n": 163, "runin": 20, "a0": 0.855},
}
REPS = 40
PATIENTS = 1000
PRIOR_WEIGHT = 10.0
SEED = 20261006
EPS = ss.EPS
# Fine grid over the range in which the within-patient AUC runs from 0.5 to about 0.9 (set by Claude, 6 Oct):
# the figure needs resolution there, not coverage of pooled AUCs near 1.
OMEGA_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)


def logit_mean_for_tau(rate, tau, z):
    lo, hi = -16.0, 8.0
    for _ in range(55):
        mid = (lo + hi) / 2
        if ss.expit(mid + tau * z).mean() < rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fit_cohort(spec):
    """Fit expected observed-rate SD: Var(qhat)=Var(p)+E[p(1-p)]/n."""
    rate, target, n = spec["rate"], spec["sd"], spec["n"]
    nodes, weights = np.polynomial.hermite.hermgauss(96)
    z = nodes*np.sqrt(2.0)
    weights = weights/math.sqrt(math.pi)

    def moments(tau):
        lo, hi = -16.0, 8.0
        for _ in range(50):
            mid = (lo+hi)/2
            if np.sum(weights*ss.expit(mid+tau*z)) < rate:
                lo = mid
            else:
                hi = mid
        m = (lo+hi)/2
        p = ss.expit(m + tau*z)
        pm = np.sum(weights*p)
        observed_var = np.sum(weights*(p-pm)**2) + np.sum(weights*p*(1-p))/n
        return m, math.sqrt(observed_var)

    low_sd, high_sd = moments(0.0)[1], moments(8.0)[1]
    if low_sd <= target <= high_sd:
        lo, hi = 0.0, 8.0
        for _ in range(55):
            mid = (lo+hi)/2
            if moments(mid)[1] < target:
                lo = mid
            else:
                hi = mid
        tau = (lo+hi)/2
    else:
        tau = 0.0 if abs(low_sd-target) <= abs(high_sd-target) else 8.0
    m, achieved = moments(tau)
    if abs(achieved-target) <= 0.003:
        return {"dist": "logit-normal", "mean": m, "tau": tau, "alpha": None, "beta": None,
                "match_sd": achieved, "sd_error": achieved-target, "target_sd": target}

    # Beta patient rates have the same mean and tune their variance analytically.
    # This is only used if the logit-normal family cannot meet the target.
    pvar = (target**2-rate*(1-rate)/n)/(1-1/n)
    pvar = float(np.clip(pvar, 1e-10, rate*(1-rate)*(1-1e-10)))
    concentration = rate*(1-rate)/pvar - 1
    aa, bb = rate*concentration, (1-rate)*concentration
    beta_sd = math.sqrt(pvar*(1-1/n)+rate*(1-rate)/n)
    return {"dist": "beta", "mean": None, "tau": None, "alpha": aa, "beta": bb,
            "match_sd": beta_sd, "sd_error": beta_sd-target, "target_sd": target}


def draw_effects(rng, fit, n):
    if fit["dist"] == "beta":
        return ss.logit(rng.beta(fit["alpha"], fit["beta"], n))
    return rng.normal(fit["mean"], fit["tau"], n)


def recenter(base_alpha, omega, target, z_ref):
    """Add common intercept shift to preserve the cohort mean event rate."""
    lo, hi = -12.0, 8.0
    for _ in range(48):
        mid = (lo+hi)/2
        rate = ss.expit(base_alpha[:, None] + mid + omega*z_ref[None, :]).mean()
        if rate < target:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def row_auc(y, score):
    """Vectorised patient AUCs with average ranks for ties."""
    n, t = y.shape
    order = np.argsort(score, axis=1, kind="mergesort")
    sorted_s = np.take_along_axis(score, order, axis=1)
    idx = np.broadcast_to(np.arange(t), (n, t))
    starts = np.where(np.c_[np.ones(n, dtype=bool), sorted_s[:, 1:] != sorted_s[:, :-1]], idx, -1)
    left = np.maximum.accumulate(starts, axis=1)
    ends = np.where(np.c_[sorted_s[:, 1:] != sorted_s[:, :-1], np.ones(n, dtype=bool)], idx+1, t)
    right = np.minimum.accumulate(ends[:, ::-1], axis=1)[:, ::-1]
    ranks_sorted = (left+1+right)/2
    ranks = np.empty_like(ranks_sorted, dtype=float)
    np.put_along_axis(ranks, order, ranks_sorted, axis=1)
    pos = y.sum(axis=1)
    neg = t-pos
    denom = pos*neg
    out = np.divide(np.sum(ranks*y, axis=1)-pos*(pos+1)/2, denom,
                    out=np.full(n, np.nan, dtype=float), where=denom > 0)
    return out


def skill(y, forecast, reference, within=False):
    if within:
        pred_loss = ((forecast-y)**2).mean(axis=1)
        ref_loss = ((reference-y)**2).mean(axis=1)
        return 1.0 - pred_loss.mean()/ref_loss.mean()
    return 1.0 - np.mean((forecast-y)**2)/np.mean((reference-y)**2)


def calc_rep(rng, cohort, spec, fit, omega, do_slopes=True):
    n, runin, k = spec["n"], spec["runin"], PATIENTS
    z_all = rng.standard_normal((k, runin+n))
    z_scored = z_all[:, runin:]
    zref = rng.standard_normal(512)
    base = draw_effects(rng, fit, k)
    shift = recenter(base, omega, spec["rate"], zref)
    alpha = base + shift
    eta = alpha[:, None] + omega*z_all
    p_all = ss.expit(eta)
    y_all = (rng.random(p_all.shape) < p_all).astype(np.int8)
    y = y_all[:, runin:]
    true_risk = p_all[:, runin:]
    # Patient event rate marginal over the daily signal distribution.
    z_marg = np.random.default_rng(991).standard_normal(256)
    rate_vec = ss.expit(alpha[:, None] + omega*z_marg[None, :]).mean(axis=1)
    rate_only = np.broadcast_to(rate_vec[:, None], y.shape)
    signal_logit = shift + omega*z_scored
    signal_only = ss.expit(signal_logit)
    patient_mean_logit = eta[:, runin:].mean(axis=1)
    damped = {f"damped_{c:g}": ss.expit(patient_mean_logit[:, None] + c*(eta[:, runin:]-patient_mean_logit[:, None]))
              for c in (0.25, 0.5, 2.0)}
    forecasts = {"true_risk": true_risk, "rate_only": rate_only, "signal_only": signal_only, **damped}
    within_true_auc = np.nanmean(row_auc(y, true_risk))
    within_signal_auc = np.nanmean(row_auc(y, signal_only))
    threshold = spec["rate"]
    pos = y == 1
    neg = ~pos
    popref = np.full(y.shape, spec["rate"])
    hist_rate = (y_all[:, :runin].sum(axis=1)+PRIOR_WEIGHT*spec["rate"])/(runin+PRIOR_WEIGHT)
    histref = np.broadcast_to(hist_rate[:, None], y.shape)
    result = {}
    for name, raw in forecasts.items():
        f = np.clip(raw, EPS, 1-EPS)
        pred = f >= threshold
        sens = np.sum(pred & pos)/max(1, int(pos.sum()))
        specp = np.sum((~pred) & neg)/max(1, int(neg.sum()))
        result[name, "auc", "pooled"] = ss.auc(y, f)
        if name == "true_risk":
            result[name, "auc", "within"] = within_true_auc
        elif name == "signal_only":
            result[name, "auc", "within"] = within_signal_auc
        elif name == "rate_only":
            result[name, "auc", "within"] = 0.5
        else:
            # damped forecasts are strictly increasing transformations of the
            # true logit within each patient, so their within AUC is identical.
            result[name, "auc", "within"] = within_true_auc
        result[name, "youden", "pooled"] = sens+specp-1
        npos, nneg = pos.sum(axis=1), neg.sum(axis=1)
        both = (npos > 0) & (nneg > 0)
        sens_k = np.divide((pred & pos).sum(axis=1), npos, out=np.zeros(k), where=npos > 0)
        spec_k = np.divide(((~pred) & neg).sum(axis=1), nneg, out=np.zeros(k), where=nneg > 0)
        result[name, "youden", "within"] = np.mean((sens_k+spec_k-1)[both]) if both.any() else np.nan
        result[name, "patients_both_outcomes", "within"] = int(both.sum())
        for ref_name, ref in (("skill_vs_population", popref), ("skill_vs_runin", histref)):
            result[name, ref_name, "pooled"] = skill(y, f, ref)
            result[name, ref_name, "within"] = skill(y, f, ref, within=True)
        if do_slopes:
            x = ss.logit(f).ravel()
            yp = y.ravel()
            result[name, "calibration_slope", "pooled"] = ss.logistic_slopes(np.c_[np.ones(x.size), x], yp)[1]
            xm = ss.logit(f)
            pm = xm.mean(axis=1)
            dev = xm-pm[:, None]
            X = np.column_stack((np.ones(y.size), np.repeat(pm, n), dev.ravel()))
            slopes = ss.logistic_slopes(X, yp)
            result[name, "calibration_slope", "between"] = slopes[1]
            result[name, "calibration_slope", "within"] = slopes[2]
    return result


def run_setting(cohort, spec, fit, omega, index):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, index]))
    vals = {}
    for rep in range(REPS):
        one = calc_rep(rng, cohort, spec, fit, omega, do_slopes=False)
        for key, value in one.items():
            vals.setdefault(key, []).append(value)
        if omega == 0 and rep == 0:
            if abs(one["true_risk", "auc", "within"]-0.5) > 1e-12:
                raise AssertionError("omega=0 within AUC check failed")
            if one["rate_only", "youden", "within"] != 0:
                raise AssertionError("omega=0 rate-only within Youden check failed")
    slope_rep = calc_rep(rng, cohort, spec, fit, omega, do_slopes=True)
    for key, value in slope_rep.items():
        if key[1] == "calibration_slope":
            vals[key] = [value]
    return vals


def main():
    started = time.perf_counter()
    outdir = Path(__file__).parent
    fits = {name: fit_cohort(spec) for name, spec in COHORTS.items()}
    # Covers within AUC >= 0.85 and signal-only pooled AUC through 0.98.
    maxomega = OMEGA_GRID[-1]
    all_rows = []
    summaries = [(cohort, spec, fits[cohort]) for cohort, spec in COHORTS.items()]
    result_path = outdir/"cohort_curves_results.csv"
    if "--summary-only" in sys.argv:
        with result_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                all_rows.append((row["cohort"], float(row["omega"]), row["forecast"], row["metric"], row["version"],
                                 float(row["mean"]) if row["mean"] not in ("", "NaN") else np.nan,
                                 float(row["sd"]) if row["sd"] not in ("", "NaN") else np.nan,
                                 int(row["n_reps"])))
    else:
        tasks = [(cohort, spec, fits[cohort], float(omega), index)
                 for index, (cohort, spec, omega) in enumerate(
                     ((cohort, spec, omega) for cohort, spec in COHORTS.items()
                      for omega in OMEGA_GRID))]
        workers = 3
        print(f"starting {len(tasks)} settings with {workers} worker processes", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_setting, *task): task for task in tasks}
            for future in as_completed(futures):
                cohort, spec, fit, omega, _ = futures[future]
                vals = future.result()
                for (forecast, metric, version), reps in vals.items():
                    arr = np.asarray(reps, dtype=float)
                    finite = np.isfinite(arr)
                    mean = np.mean(arr[finite]) if finite.any() else np.nan
                    sd = np.std(arr[finite], ddof=1) if finite.sum() > 1 else np.nan
                    all_rows.append((cohort, omega, forecast, metric, version, mean, sd, len(arr)))
                print(f"completed {cohort}: omega={omega:.3f}", flush=True)
        with result_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(("cohort", "omega", "forecast", "metric", "version", "mean", "sd", "n_reps"))
            w.writerows(all_rows)

    summary = ["# Cohort matched reference curves", "", f"Patients per replication: {PATIENTS}; replications per omega for performance metrics: {REPS}; omega grid has 13 points from 0 to {maxomega:g}.", "Calibration slopes are estimated from one additional independent replication per cohort and omega (their CSV n_reps is 1).", ""]
    for cohort, spec, fit in summaries:
        summary += [f"## {cohort}", "", f"Fit: {fit['dist']}; intercept mean={fit['mean'] if fit['mean'] is not None else 'n/a'}; tau={fit['tau'] if fit['tau'] is not None else 'n/a'}; beta alpha={fit['alpha'] if fit['alpha'] is not None else 'n/a'}; beta beta={fit['beta'] if fit['beta'] is not None else 'n/a'}.",
                    f"Match: event rate={spec['rate']:.3f} (target exact by recentering); observed patient-rate SD={fit['match_sd']:.3f} (target {spec['sd']:.3f}; difference {fit['sd_error']:+.4f}) at n={spec['n']}; run-in={spec['runin']}. Distribution used: {fit['dist']}.", "",
                    "| omega | Within AUC | Pooled AUC | Within Youden | Pooled Youden | Skill population | Skill run-in | Model A0 (rate_only) | Observed A0 |",
                    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        model_a0 = next(r[5] for r in all_rows if r[0]==cohort and r[1]==0 and r[2]=="rate_only" and r[3]=="auc" and r[4]=="pooled")
        both_omega0 = next(r[5] for r in all_rows if r[0]==cohort and r[1]==0 and r[2]=="true_risk" and r[3]=="patients_both_outcomes" and r[4]=="within")
        summary.append(f"At omega=0, the mean number of patients with both outcomes per replication was {both_omega0:.1f} of {PATIENTS}.")
        for omega in OMEGA_GRID:
            def val(forecast, metric, version="pooled"):
                found = [r[5] for r in all_rows if r[0] == cohort and abs(r[1]-omega)<1e-8 and r[2]==forecast and r[3]==metric and r[4]==version]
                return found[0] if found else float("nan")
            summary.append(f"| {omega:.3f} | {val('true_risk','auc','within'):.3f} | {val('true_risk','auc'):.3f} | {val('true_risk','youden','within'):.3f} | {val('true_risk','youden'):.3f} | {val('true_risk','skill_vs_population'):.3f} | {val('true_risk','skill_vs_runin'):.3f} | {model_a0:.3f} | {spec['a0']:.3f} |")
        a0 = model_a0
        summary += ["", f"Model A0 minus observed A0: {a0-spec['a0']:+.3f}.", "", "### Reading the curve backwards", "", "| Pooled AUC G | Reference curve: model also knows patients' rates (within AUC) | Model knows nothing about patients' rates (signal-only within AUC) |", "|---:|---:|---:|"]
        gs = np.arange(math.floor((a0+1e-9)/0.02+1)*0.02, 0.9801, 0.02)
        for g in gs:
            def interp(forecast, metric, version):
                pts = sorted((r[5], next(q[5] for q in all_rows if q[0]==cohort and abs(q[1]-r[1])<1e-8 and q[2]==forecast and q[3]==metric and q[4]==version)) for r in all_rows if r[0]==cohort and r[2]==forecast and r[3]=="auc" and r[4]=="pooled")
                xs, ys = np.asarray(pts).T
                # Collapse numerical ties in pooled AUC before interpolation.
                ux, ix = np.unique(xs, return_index=True)
                if g < ux[0] or g > ux[-1]: return float("nan")
                return float(np.interp(g, ux, ys[ix]))
            tr = interp("true_risk", "auc", "within")
            so = interp("signal_only", "auc", "within")
            summary.append(f"| {g:.2f} | {tr:.3f} | {so:.3f} |")
        summary.append("")
    summary += ["### Calibration slope check", "", "Damped-forecast within slopes should be near 1/c. The forecasts use the scored-period patient mean of the true logit.", "", "| Cohort | omega | damped_0.25 within | damped_0.5 within | damped_2 within |", "|---|---:|---:|---:|---:|"]
    for cohort, spec, fit in summaries:
        for omega in OMEGA_GRID:
            def slope(c):
                r = next(r for r in all_rows if r[0]==cohort and abs(r[1]-omega)<1e-8 and r[2]==f"damped_{c:g}" and r[3]=="calibration_slope" and r[4]=="within")
                return r[5]
            summary.append(f"| {cohort} | {omega:.3f} | {slope(.25):.3f} | {slope(.5):.3f} | {slope(2):.3f} |")
    (outdir/"cohort_curves_summary.md").write_text("\n".join(summary)+"\n", encoding="utf-8")
    elapsed = time.perf_counter()-started
    print(f"wrote {result_path}; rows={len(all_rows)}; maxomega={maxomega:g}; workers={workers if '--summary-only' not in sys.argv else 'n/a'}; runtime={elapsed:.1f}s")


if __name__ == "__main__":
    main()
