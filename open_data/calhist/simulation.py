"""Calibrated history-forecast simulations for the calibration methods audit.

Runs prospectively correct forecast-generating DGPs using the observed patient
record lengths in the three test cohorts.  All output is written beside this
script; imports fit_core (the local, side-effect-contained slope implementation).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from fit_core import slopes


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OPEN = ROOT / "open_data"
OUT = OPEN / "results" / "simulation"
SEED = 20261007
NREP = 100
RHOS = (0.3, 0.6, 0.8)


def f_t(T: int, rho: float) -> float:
    """Finite-record variance multiplier for the stationary AR(1) binary mean."""
    return (T + 2.0 * sum((T - j) * rho**j for j in range(1, T))) / T**2


def beta_k_from_rate_variance(mu: float, rate_var: float, lengths: np.ndarray,
                              rho: float) -> tuple[float, float, float]:
    """Method-of-moments concentration, with serial sampling removed.

    For r~Beta(mu*k,(1-mu)*k), the equal-patient variance of the observed
    patient rates is mu(1-mu)*(1+k*mean(f_T))/(k+1).  The empirical sample
    variance across patients estimates that population variance.
    """
    favg = float(np.mean([f_t(int(t), rho) for t in lengths]))
    a = rate_var / (mu * (1 - mu))
    if a <= favg:
        raise ValueError(f"No finite positive k: normalized variance {a:g} <= fbar {favg:g}")
    k = (1 - a) / (a - favg)
    if k <= 0:
        raise ValueError(f"Nonpositive beta concentration {k:g}")
    return float(k), favg, float(a)


def build_histories(lengths: np.ndarray, mu: float, k: float, rho: float,
                    mode: str, burnin: int, rng: np.random.Generator,
                    K: float = 20.0):
    """Generate outcomes and their true one-step conditional probabilities.

    Initialize Y~Bernoulli(r_i), generate `burnin` unscored run-in outcomes,
    then score forecasts that all condition on already-observed outcomes.
    `markov`: p_t = (1-rho) r_i + rho Y_{t-1}.
    `expanding_last`: p_t = (1-rho) * (K*r_i + sum prior Y)/(K+history length)
                      + rho*Y_{t-1}. Each outcome is sampled from its returned p.
    """
    aa, bb = mu * k, (1 - mu) * k
    r = rng.beta(aa, bb, size=len(lengths))
    yy, pp, gg = [], [], []
    for i, (ri, ti) in enumerate(zip(r, lengths)):
        ti = int(ti)
        yi = np.empty(ti, dtype=np.int8)
        pi = np.empty(ti, dtype=float)
        last = int(rng.binomial(1, ri))
        total, hist_n = last, 1
        for _ in range(int(burnin)):
            if mode == "markov":
                smooth = ri
            elif mode == "expanding_last":
                smooth = (K * ri + total) / (K + hist_n)
            else:
                raise ValueError(mode)
            last = int(rng.binomial(1, (1 - rho) * smooth + rho * last))
            total += last
            hist_n += 1
        for t in range(ti):
            smooth = ri if mode == "markov" else (K * ri + total) / (K + hist_n)
            pi[t] = (1 - rho) * smooth + rho * last
            yi[t] = rng.binomial(1, pi[t])
            last = int(yi[t])
            total += last
            hist_n += 1
        yy.append(yi)
        pp.append(pi)
        gg.append(np.full(ti, i, dtype=np.int32))
    return np.concatenate(yy), np.concatenate(pp), np.concatenate(gg)


def stable_logit_slope(X, y):
    """Small-dimensional unpenalized logistic fit used if fit_core IRLS stalls."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    start = np.zeros(X.shape[1])
    start[0] = logit(np.clip(y.mean(), 1e-8, 1 - 1e-8))
    if X.shape[1] > 1:
        start[1:] = 1.0

    def objective(b):
        eta = X @ b
        return float(np.logaddexp(0., eta).sum() - y @ eta)

    def gradient(b):
        resid = expit(X @ b) - y
        return np.sum(X * resid[:, None], axis=0)

    fit = minimize(objective, start, jac=gradient, method="L-BFGS-B",
                   options={"maxiter": 300, "ftol": 1e-13, "gtol": 1e-7, "maxls": 50})
    if not np.all(np.isfinite(fit.x)) or not np.all(np.isfinite(gradient(fit.x))):
        return math.nan
    if np.linalg.norm(gradient(fit.x), ord=np.inf) > 1e-3:
        return math.nan
    return float(fit.x[-1])


def simulate_one(y, p, groups):
    fallback = []
    try:
        pooled, joint, conditional = slopes(y, p, groups)
        if not np.isfinite(pooled):
            lp = logit(np.clip(p, 1e-6, 1 - 1e-6))
            pooled = stable_logit_slope(np.column_stack([np.ones(len(y)), lp]), y)
            fallback.append("pooled")
        if not np.isfinite(joint):
            lp = logit(np.clip(p, 1e-6, 1 - 1e-6))
            counts = np.bincount(groups)
            means = np.bincount(groups, weights=lp) / counts
            joint = stable_logit_slope(
                np.column_stack([np.ones(len(y)), means[groups], lp - means[groups]]), y)
            fallback.append("joint")
        failure = "" if np.all(np.isfinite([pooled, joint, conditional])) else "nonfinite_after_fallback"
        return pooled, joint, conditional, failure, ";".join(fallback)
    except Exception as exc:  # record failures instead of silently dropping them
        return math.nan, math.nan, math.nan, f"{type(exc).__name__}: {exc}", ";".join(fallback)


def summarize(draws: pd.DataFrame) -> pd.DataFrame:
    out = []
    slope_names = ("pooled_slope", "joint_within_slope", "conditional_slope")
    for key, z in draws.groupby(["cohort", "scenario", "mode", "rho"], sort=False):
        row = dict(zip(("cohort", "scenario", "mode", "rho"), key))
        row.update({"replications": len(z), "patients": int(z.patients.iloc[0]),
                    "observations": int(z.observations.iloc[0]), "target_rate": z.target_rate.iloc[0],
                    "k": z.k.iloc[0], "beta_fbar": z.beta_fbar.iloc[0],
                    "target_rate_variance": z.target_rate_variance.iloc[0],
                    "failure_reps": int(z.failure_reason.ne("").sum())})
        for name in slope_names:
            a = z[name].to_numpy(float)
            a = a[np.isfinite(a)]
            row[name + "_n"] = len(a)
            row[name + "_mean"] = float(np.mean(a)) if len(a) else math.nan
            row[name + "_mcse"] = float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else math.nan
            row[name + "_q025"] = float(np.quantile(a, .025)) if len(a) else math.nan
            row[name + "_q975"] = float(np.quantile(a, .975)) if len(a) else math.nan
            row[name + "_bias_from_1"] = row[name + "_mean"] - 1 if len(a) else math.nan
        for name in ("event_rate", "patient_rate_sd", "all0_fraction", "all1_fraction",
                     "patient_error_mean", "patient_error_median", "patient_error_q25",
                     "patient_error_q75", "patient_error_within_005"):
            row[name + "_mean"] = float(z[name].mean())
            row[name + "_sd_across_reps"] = float(z[name].std(ddof=1))
        out.append(row)
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    specs = [
        ("replace_bg", .05982905982905983),
        ("ctn_weekly_missing_excluded", .2999763983950909),
        ("ces", .39883666090237385),
    ]
    draws = []
    runins = {"replace_bg": 14, "ctn_weekly_missing_excluded": 4, "ces": 20}
    design = {"seed": SEED, "replications_per_scenario": NREP,
              "rhos": list(RHOS), "fitted_style_K": 20,
              "stationary_dgp": "initialize Y~Bernoulli(r_i), generate unscored run-in, then score p_t=(1-rho)r_i+rho*Y_(t-1); all scored forecasts condition on observed Y_(t-1)",
              "expanding_style_dgp": "initialize Y~Bernoulli(r_i), generate unscored run-in, then score p_t=(1-rho)*(K*r_i+sum prior Y)/(K+history length)+rho*Y_(t-1); each Y_t is sampled from this exact probability",
              "runin_days": runins,
              "tuning": "stationary beta k selected analytically to match empirical equal-patient observed-rate variance at each rho; the same k used for expanding-style DGP, whose resulting variance is reported because the stationary analytic correction does not apply; expected marginal event rate is target mu",
              "ces_sensitivity": "also simulate CES at requested rounded event rate .38, rho=.6, using same observed lengths and k retuned to observed rate variance on the .38 scale"}
    for cohort, target_mu in specs:
        df = pd.read_csv(OPEN / f"forecasts_{cohort}.csv")
        patient = df.groupby("patient", sort=False).y.agg(T="size", rate="mean")
        lengths = patient["T"].to_numpy(int)
        rate_var = float(patient.rate.var(ddof=1))
        scenario_list = [("matched", target_mu, rho, True) for rho in RHOS]
        if cohort == "ces":
            scenario_list += [("rounded_rate_038", .38, .6, False)]
        for scenario, mu, rho, all_rhos in scenario_list:
            k, favg, avar = beta_k_from_rate_variance(mu, rate_var, lengths, rho)
            modes = ("markov", "expanding_last") if all_rhos else ("markov",)
            print(f"start {cohort} {scenario} rho={rho} k={k:.5g} modes={modes}", flush=True)
            for mode in modes:
                tic = time.time()
                for rep in range(NREP):
                    y, p, groups = build_histories(lengths, mu, k, rho, mode,
                                                  runins[cohort], rng)
                    ps, js, cs, fail, fallback = simulate_one(y, p, groups)
                    ng = len(lengths)
                    counts = np.bincount(groups, minlength=ng)
                    events = np.bincount(groups, weights=y, minlength=ng)
                    pmean = np.bincount(groups, weights=p, minlength=ng) / counts
                    prate = events / counts
                    errors = prate - pmean
                    draws.append({"cohort": cohort, "scenario": scenario, "mode": mode,
                                  "rep": rep, "rho": rho, "K": 20,
                                  "target_rate": mu, "k": k, "beta_fbar": favg,
                                  "target_rate_variance": rate_var,
                                  "patients": ng, "observations": len(y),
                                  "event_rate": float(y.mean()),
                                  "patient_rate_sd": float(np.std(prate, ddof=1)),
                                  "patient_error_mean": float(np.mean(errors)),
                                  "patient_error_median": float(np.median(errors)),
                                  "patient_error_q25": float(np.quantile(errors, .25)),
                                  "patient_error_q75": float(np.quantile(errors, .75)),
                                  "patient_error_within_005": float(np.mean(np.abs(errors) <= .05)),
                                  "all0_fraction": float(np.mean(events == 0)),
                                  "all1_fraction": float(np.mean(events == counts)),
                                  "pooled_slope": ps, "joint_within_slope": js,
                                  "conditional_slope": cs, "failure_reason": fail,
                                  "fit_fallback": fallback})
                print(f"done {cohort} {scenario} {mode}: {time.time()-tic:.1f}s", flush=True)
                if scenario == "matched" and rho == .6:
                    # One large panel with 100 copies of the observed T distribution
                    # reveals the pseudo-target with much less cohort-size noise.
                    large_lengths = np.tile(lengths, 100)
                    ly, lp, lg = build_histories(large_lengths, mu, k, rho, mode,
                                                 runins[cohort], rng)
                    lps, ljs, lcs, lfail, lfallback = simulate_one(ly, lp, lg)
                    lng = len(large_lengths)
                    lcounts = np.bincount(lg, minlength=lng)
                    levents = np.bincount(lg, weights=ly, minlength=lng)
                    lpmean = np.bincount(lg, weights=lp, minlength=lng) / lcounts
                    lrate = levents / lcounts
                    lerrors = lrate - lpmean
                    draws.append({"cohort": cohort, "scenario": "largeN100", "mode": mode,
                                  "rep": 0, "rho": rho, "K": 20,
                                  "target_rate": mu, "k": k, "beta_fbar": favg,
                                  "target_rate_variance": rate_var,
                                  "patients": lng, "observations": len(ly),
                                  "event_rate": float(ly.mean()),
                                  "patient_rate_sd": float(np.std(lrate, ddof=1)),
                                  "patient_error_mean": float(np.mean(lerrors)),
                                  "patient_error_median": float(np.median(lerrors)),
                                  "patient_error_q25": float(np.quantile(lerrors, .25)),
                                  "patient_error_q75": float(np.quantile(lerrors, .75)),
                                  "patient_error_within_005": float(np.mean(np.abs(lerrors) <= .05)),
                                  "all0_fraction": float(np.mean(levents == 0)),
                                  "all1_fraction": float(np.mean(levents == lcounts)),
                                  "pooled_slope": lps, "joint_within_slope": ljs,
                                  "conditional_slope": lcs, "failure_reason": lfail,
                                  "fit_fallback": lfallback})
                    print(f"largeN100 {cohort} {mode}: patients={lng}, obs={len(ly)}", flush=True)
    draw_df = pd.DataFrame(draws)
    summary = summarize(draw_df)
    draw_df.to_csv(OUT / "simulation_draws.csv", index=False)
    summary.to_csv(OUT / "simulation_summary.csv", index=False)
    draw_df.loc[draw_df.scenario.eq("largeN100")].to_csv(
        OUT / "simulation_large_n.csv", index=False)
    design["cohorts"] = {}
    for cohort, _ in specs:
        d = pd.read_csv(OPEN / f"forecasts_{cohort}.csv")
        g = d.groupby("patient", sort=False).y.agg(T="size", rate="mean")
        design["cohorts"][cohort] = {
            "source": f"open_data/forecasts_{cohort}.csv",
            "patients": int(len(g)), "observations": int(len(d)),
            "target_event_rate": float(d.y.mean()),
            "T_min": int(g["T"].min()), "T_median": float(g["T"].median()), "T_max": int(g["T"].max()),
            "observed_equal_patient_rate_variance": float(g.rate.var(ddof=1)),
            "observed_all0_fraction": float(np.mean(g.rate == 0)),
            "observed_all1_fraction": float(np.mean(g.rate == 1)),
        }
    design["completed_scenarios"] = int(len(summary))
    design["completed_replicates"] = int(len(draw_df))
    design["failed_replicates"] = int(draw_df.failure_reason.ne("").sum())
    (OUT / "simulation_design.json").write_text(
        json.dumps(design, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
