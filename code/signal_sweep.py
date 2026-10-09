"""Simulation sweep of pooled and within-patient metrics across daily signal strength.

The DGP follows sim.py: patient logit intercepts are Normal(m, tau**2), daily
signals are standard Normal, and event probabilities are expit(alpha + omega*z).
"""
import csv
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Small IRLS matrices are faster and more predictable without BLAS oversubscription.
for _thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_var] = "1"

import numpy as np

MU = 0.15
TAUS = (0.0, 0.5, 1.0, 1.5)
OMEGAS = np.linspace(0.0, 1.0, 9)
K = 200
N_SCORED = 180
RUNIN = 30
R_REPS = 50
SEED = 20261006
PRIOR_WEIGHT = 10.0
EPS = 1e-6


def expit(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def m_for_mu(tau, omega):
    # Same fixed 200,000-draw numerical calibration convention as sim.py.
    z = np.random.default_rng(0).standard_normal(200_000)
    scale = np.hypot(tau, omega)
    lo, hi = -12.0, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if expit(mid + scale * z).mean() < MU:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def auc(y, score):
    y = np.asarray(y, dtype=np.int8).ravel()
    s = np.asarray(score, dtype=float).ravel()
    npos = int(y.sum())
    nneg = len(y) - npos
    if not npos or not nneg:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    cuts = np.r_[0, np.flatnonzero(sorted_s[1:] != sorted_s[:-1]) + 1, len(s)]
    average_ranks = (cuts[:-1] + 1 + cuts[1:]) / 2.0
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.repeat(average_ranks, np.diff(cuts))
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def logistic_slopes(X, y, max_iter=60):
    """IRLS/Newton logistic coefficients; NaNs for singular or nonconvergent fits."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all() or np.linalg.matrix_rank(X.T @ X) < X.shape[1]:
        return np.full(X.shape[1], np.nan)
    beta = np.zeros(X.shape[1], dtype=float)
    for _ in range(min(max_iter, 30)):
        p = expit(X @ beta)
        w = np.maximum(p * (1.0 - p), 1e-12)
        grad = X.T @ (y - p)
        hess = (X.T * w) @ X
        # Explicit tiny-system solve avoids repeated BLAS/LAPACK startup overhead.
        if len(grad) == 2:
            a, b = hess[0]
            c, d = hess[1]
            det = a*d - b*c
            if abs(det) < 1e-20:
                return np.full(X.shape[1], np.nan)
            step = np.array(((d*grad[0] - b*grad[1]) / det,
                             (-c*grad[0] + a*grad[1]) / det))
        elif len(grad) == 3:
            a, b, c = hess[0]
            d, e, f = hess[1]
            g, h, i = hess[2]
            det = a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g)
            if abs(det) < 1e-20:
                return np.full(X.shape[1], np.nan)
            step = np.array(((e*i-f*h)*grad[0] + (c*h-b*i)*grad[1] + (b*f-c*e)*grad[2],
                             (f*g-d*i)*grad[0] + (a*i-c*g)*grad[1] + (c*d-a*f)*grad[2],
                             (d*h-e*g)*grad[0] + (b*g-a*h)*grad[1] + (a*e-b*d)*grad[2])) / det
        else:
            return np.full(X.shape[1], np.nan)
        # Cap very large early steps; this is damped Newton/IRLS without a costly
        # full-data likelihood line search at every iteration.
        scale = min(1.0, 5.0 / max(float(np.max(np.abs(step))), 5.0))
        beta += scale * step
        if np.max(np.abs(scale * step)) < 1e-7:
            return beta
    # Accept near-converged finite estimates only.
    if np.isfinite(beta).all() and np.max(np.abs(step)) < 1e-5:
        return beta
    return np.full(X.shape[1], np.nan)


def run_one(rng, tau, omega):
    m = m_for_mu(tau, omega)
    alpha = rng.normal(m, tau, K)
    z = rng.standard_normal((K, RUNIN + N_SCORED))
    probs = expit(alpha[:, None] + omega * z)
    yall = (rng.random(probs.shape) < probs).astype(np.int8)
    y = yall[:, RUNIN:]
    true_risk = probs[:, RUNIN:]
    rate_only = np.broadcast_to(expit(alpha)[:, None], (K, N_SCORED))
    signal_only = np.broadcast_to(m, (K, N_SCORED)) + omega * z[:, RUNIN:]
    signal_only = expit(signal_only)
    forecasts = {"true_risk": true_risk, "rate_only": rate_only, "signal_only": signal_only}
    threshold = y.mean()
    pop_ref = np.full((K, N_SCORED), MU)
    hist_rate = (yall[:, :RUNIN].sum(axis=1) + PRIOR_WEIGHT * MU) / (RUNIN + PRIOR_WEIGHT)
    hist_ref = np.broadcast_to(hist_rate[:, None], (K, N_SCORED))
    result = {}

    for name, f in forecasts.items():
        fc = np.clip(f, EPS, 1-EPS)
        predicted = fc >= threshold
        pos = y == 1
        neg = ~pos
        tp = int((predicted & pos).sum())
        tn = int((~predicted & neg).sum())
        sens_p = tp / int(pos.sum()) if pos.any() else np.nan
        spec_p = tn / int(neg.sum()) if neg.any() else np.nan
        result[(name, "auc", "pooled")] = (auc(y, fc), K)
        result[(name, "sensitivity", "pooled")] = (sens_p, K)
        result[(name, "specificity", "pooled")] = (spec_p, K)
        result[(name, "youden", "pooled")] = (sens_p + spec_p - 1 if np.isfinite(sens_p + spec_p) else np.nan, K)

        npos = pos.sum(axis=1)
        nneg = neg.sum(axis=1)
        per_auc = np.array([auc(y[k], fc[k]) if npos[k] and nneg[k] else np.nan for k in range(K)])
        sens_k = np.divide((predicted & pos).sum(axis=1), npos, out=np.full(K, np.nan), where=npos > 0)
        spec_k = np.divide((~predicted & neg).sum(axis=1), nneg, out=np.full(K, np.nan), where=nneg > 0)
        mean_sens = np.nanmean(sens_k) if np.isfinite(sens_k).any() else np.nan
        mean_spec = np.nanmean(spec_k) if np.isfinite(spec_k).any() else np.nan
        both_classes = (npos > 0) & (nneg > 0)
        mean_sens_both = np.mean(sens_k[both_classes]) if both_classes.any() else np.nan
        mean_spec_both = np.mean(spec_k[both_classes]) if both_classes.any() else np.nan
        result[(name, "auc", "within")] = (np.nanmean(per_auc) if np.isfinite(per_auc).any() else np.nan, int(np.isfinite(per_auc).sum()))
        result[(name, "sensitivity", "within")] = (mean_sens, int((npos > 0).sum()))
        result[(name, "specificity", "within")] = (mean_spec, int((nneg > 0).sum()))
        result[(name, "youden", "within")] = (mean_sens_both + mean_spec_both - 1 if np.isfinite(mean_sens_both + mean_spec_both) else np.nan, int(both_classes.sum()))

        loss = (fc - y) ** 2
        for ref_name, ref in (("skill_vs_population", pop_ref), ("skill_vs_runin", hist_ref)):
            ref_loss = (ref - y) ** 2
            pooled_skill = 1.0 - loss.sum() / ref_loss.sum() if ref_loss.sum() > 0 else np.nan
            per_loss = loss.mean(axis=1)
            per_ref = ref_loss.mean(axis=1)
            within_skill = 1.0 - per_loss.mean() / per_ref.mean() if per_ref.mean() > 0 else np.nan
            result[(name, ref_name, "pooled")] = (pooled_skill, K)
            result[(name, ref_name, "within")] = (within_skill, K)

        x = logit(fc).ravel()
        Xp = np.column_stack((np.ones(x.size), x))
        result[(name, "calibration_slope", "pooled")] = (logistic_slopes(Xp, y.ravel())[1], K)
        xmat = logit(fc)
        means = xmat.mean(axis=1)
        dev = xmat - means[:, None]
        if np.std(dev) <= 1e-12:
            # With no within-patient variation, fit the identifiable between
            # coefficient and leave the within coefficient undefined.
            if np.std(means) > 1e-12:
                Xb = np.column_stack((np.ones(y.size), np.repeat(means, N_SCORED)))
                between_slope = logistic_slopes(Xb, y.ravel())[1]
            else:
                between_slope = np.nan
            within_slope = np.nan
        else:
            Xb = np.column_stack((np.ones(y.size), np.repeat(means, N_SCORED), dev.ravel()))
            slopes = logistic_slopes(Xb, y.ravel())
            between_slope = slopes[1]
            within_slope = slopes[2] if name != "rate_only" else np.nan
        result[(name, "calibration_slope", "between")] = (between_slope, K)
        result[(name, "calibration_slope", "within")] = (within_slope, K)
    return result


def run_setting(index, tau, omega):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, index]))
    local = {}
    for _ in range(R_REPS):
        one = run_one(rng, tau, omega)
        for key, (value, eligible) in one.items():
            entry = local.setdefault((tau, omega, *key), {"values": [], "eligible": []})
            entry["values"].append(value)
            entry["eligible"].append(eligible)
    return index, tau, omega, local


def main():
    started = time.perf_counter()
    values = {}
    settings = [(tau, float(omega)) for tau in TAUS for omega in OMEGAS]
    workers = min(8, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_setting, i, tau, omega) for i, (tau, omega) in enumerate(settings)]
        for future in as_completed(futures):
            idx, tau, omega, local = future.result()
            values.update(local)
            print(f"completed {idx+1}/{len(settings)}: tau={tau:g}, omega={omega:g}", flush=True)

    out_path = Path(__file__).with_name("signal_sweep_results.csv")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("tau", "omega", "forecast", "metric", "version", "mean", "sd_over_replications", "n_reps", "mean_eligible_patients"))
        for key in sorted(values):
            tau, omega, forecast, metric, version = key
            vals = np.asarray(values[key]["values"], dtype=float)
            eligible = np.asarray(values[key]["eligible"], dtype=float)
            writer.writerow((tau, omega, forecast, metric, version,
                             np.nanmean(vals) if np.isfinite(vals).any() else "NaN",
                             np.nanstd(vals, ddof=1) if np.isfinite(vals).sum() > 1 else "NaN",
                             len(vals), np.nanmean(eligible)))
    elapsed = time.perf_counter() - started
    print(f"wrote {out_path}; settings={len(settings)}, replications/setting={R_REPS}, workers={workers}, rows={len(values)}, runtime={elapsed:.1f}s")
    print("checks: omega=0 true_risk within AUC; omega=0 rate_only within Youden; tau=0 AUC agreement are evaluated by the caller")


if __name__ == "__main__":
    main()
