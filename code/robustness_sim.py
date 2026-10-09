import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
SEED = 20261005
REPS = 300
K = 200
MU_REF = 0.15
TAU = 1.0
OMEGA_REF = 0.5
RUNIN = 60


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def logit(x):
    return np.log(x / (1.0 - x))


def gh_mean(alpha, omega, nodes=40):
    """Deterministic logistic-normal mean for each patient."""
    x, w = np.polynomial.hermite.hermgauss(nodes)
    zz = np.sqrt(2.0) * x
    return (sigmoid(alpha[:, None] + omega[:, None] * zz) @ w) / math.sqrt(math.pi)


def solve_m():
    x, w = np.polynomial.hermite.hermgauss(80)
    z = np.sqrt(2.0) * x
    lo, hi = -8.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        val = np.sum(sigmoid(mid + math.sqrt(TAU * TAU + OMEGA_REF * OMEGA_REF) * z) * w) / math.sqrt(math.pi)
        if val < MU_REF:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def auc(y, score):
    y = np.asarray(y, dtype=np.uint8).ravel()
    score = np.asarray(score, dtype=np.float64).ravel()
    n1 = int(y.sum())
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(score, kind='mergesort')
    s = score[order]
    yy = y[order]
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    ends = np.r_[starts[1:], len(s)]
    positive_by_group = np.add.reduceat(yy, starts)
    mean_rank = (starts + 1 + ends) / 2.0
    rank_sums = float(np.dot(positive_by_group, mean_rank))
    return float((rank_sums - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auc_by_patient(y, score, lengths):
    # Row-wise Mann-Whitney AUC, vectorised across patients. Padding is sorted last.
    active = np.arange(y.shape[1])[None, :] < lengths[:, None]
    order = np.argsort(np.where(active, score, np.inf), axis=1, kind='quicksort')
    ys = np.take_along_axis(y, order, axis=1).astype(np.int64)
    act = np.take_along_axis(active, order, axis=1)
    neg = ((1 - ys) * act).astype(np.int64)
    lower_neg = np.cumsum(neg, axis=1) - neg
    pos = (ys * act).astype(np.int64)
    npos = pos.sum(axis=1)
    nday = lengths.astype(np.int64)
    nneg = nday - npos
    denom = npos * nneg
    vals = np.divide((pos * lower_neg).sum(axis=1), denom, out=np.full(len(lengths), np.nan), where=denom > 0)
    # The only non-continuous within-patient scores used here are patient constants.
    const = np.ptp(score, axis=1) == 0
    vals[const & (denom > 0)] = 0.5
    elig = denom > 0
    if not elig.any(): return (np.nan, np.nan, np.nan)
    return (float(np.mean(vals[elig])), float(np.average(vals[elig], weights=nday[elig])), float(np.average(vals[elig], weights=denom[elig])))


def binary_metrics(y, score, lengths, cutoff):
    mask = np.arange(y.shape[1])[None, :] < lengths[:, None]
    pred = score >= cutoff
    pos = (y == 1) & mask
    neg = (y == 0) & mask
    npos = pos.sum(axis=1)
    nneg = neg.sum(axis=1)
    tp = (pred & pos).sum(axis=1)
    tn = ((~pred) & neg).sum(axis=1)
    sens = np.divide(tp, npos, out=np.full(len(lengths), np.nan), where=npos > 0)
    spec = np.divide(tn, nneg, out=np.full(len(lengths), np.nan), where=nneg > 0)
    err = ((score - y) ** 2) * mask
    brier_k = err.sum(axis=1) / lengths
    sens_p = float(tp.sum() / npos.sum()) if npos.sum() else np.nan
    spec_p = float(tn.sum() / nneg.sum()) if nneg.sum() else np.nan
    return {
        'sensitivity_pooled': sens_p,
        'sensitivity_patient_mean': float(np.nanmean(sens)) if np.isfinite(sens).any() else np.nan,
        'specificity_pooled': spec_p,
        'specificity_patient_mean': float(np.nanmean(spec)) if np.isfinite(spec).any() else np.nan,
        'brier_pooled': float(err.sum() / lengths.sum()),
        'brier_patient_mean': float(brier_k.mean()),
    }


def weighted_exact_a0(p, lengths):
    wt = lengths / lengths.sum()
    mu = float(np.dot(wt, p))
    mad = float(np.sum(np.abs(p[:, None] - p[None, :]) * wt[:, None] * wt[None, :]))
    return 0.5 + mad / (4 * mu * (1 - mu)), mu


def ranked_lengths(rng, alpha, target):
    # Gaussian copula calibrated to the requested population Spearman rho.
    rho = 2.0 * math.sin(math.pi * target / 6.0)
    x = (alpha - alpha.mean()) / (alpha.std() + 1e-12)
    e = rng.normal(size=K)
    latent = rho * x + math.sqrt(max(0.0, 1 - rho * rho)) * e
    order = np.argsort(latent)
    lengths = np.rint(np.linspace(60, 365, K)).astype(int)
    shuffled = np.empty(K, dtype=int)
    shuffled[order] = lengths
    return shuffled


def scenario_spec(name, m):
    if name == 'reference': return ('normal', 'equal', 'fixed')
    if name.startswith('R1_'): return ('normal', 'rank', {'pos06':0.6,'zero':0.0,'neg06':-0.6}[name[3:]])
    if name == 'R2_gamma': return ('gamma', 'equal', 'fixed')
    if name == 'R2_two_point': return ('two_point', 'equal', 'fixed')
    if name.startswith('R3_lognormal_cv'): return ('normal', 'equal', {'R3_lognormal_cv05':'lncv0.5','R3_lognormal_cv10':'lncv1.0'}[name])
    if name == 'R3_half_zero': return ('normal', 'equal', 'halfzero')
    raise ValueError(name)


def run_scenario(name, rng, m):
    effect_type, length_type, signal_type = scenario_spec(name, m)
    rec = {k: [] for k in ['auc_pooled_true_risk','auc_within_equal_true_risk','auc_within_days_true_risk','auc_within_pairs_true_risk',
                           'auc_pooled_true_base_rate','auc_within_equal_true_base_rate','auc_within_days_true_base_rate','auc_within_pairs_true_base_rate',
                           'auc_pooled_runin_rate','auc_within_equal_runin_rate','auc_within_days_runin_rate','auc_within_pairs_runin_rate',
                           'sensitivity_pooled_true_risk','sensitivity_patient_mean_true_risk','specificity_pooled_true_risk','specificity_patient_mean_true_risk','brier_pooled_true_risk','brier_patient_mean_true_risk',
                           'sensitivity_pooled_true_base_rate','sensitivity_patient_mean_true_base_rate','specificity_pooled_true_base_rate','specificity_patient_mean_true_base_rate','brier_pooled_true_base_rate','brier_patient_mean_true_base_rate',
                           'sensitivity_pooled_runin_rate','sensitivity_patient_mean_runin_rate','specificity_pooled_runin_rate','specificity_patient_mean_runin_rate','brier_pooled_runin_rate','brier_patient_mean_runin_rate',
                           'a0_exact','auc_observed_scored_rate','a0_normal_approx','day_weighted_mu','day_weighted_sigma','spearman_length_intercept','mean_length']}
    for rep in range(REPS):
        if effect_type == 'normal':
            alpha = rng.normal(m, TAU, K)
        elif effect_type == 'gamma':
            alpha = (m - 1.0) + rng.gamma(shape=1.0, scale=1.0, size=K)
        elif effect_type == 'two_point':
            # 10% lower and 90% higher point masses; mean m, SD 1, skewness 2.67.
            p_low = 0.1
            alpha = m + rng.choice([-(math.sqrt((1-p_low)/p_low)), math.sqrt(p_low/(1-p_low))], size=K, p=[p_low, 1-p_low])
        if signal_type == 'fixed':
            omega = np.full(K, OMEGA_REF)
        elif isinstance(signal_type, str) and signal_type.startswith('lncv'):
            cv = float(signal_type[4:])
            sig2 = math.log(1 + cv * cv)
            omega = rng.lognormal(math.log(OMEGA_REF) - sig2 / 2, math.sqrt(sig2), K)
        elif signal_type == 'halfzero':
            omega = np.r_[np.zeros(K // 2), np.full(K - K // 2, 1.0)]
            rng.shuffle(omega)
        else:
            omega = np.full(K, OMEGA_REF)
        if length_type == 'equal':
            lengths = np.full(K, 180, dtype=int)
        else:
            lengths = ranked_lengths(rng, alpha, float(signal_type))
        zrun = rng.normal(size=(K, RUNIN))
        prun = sigmoid(alpha[:, None] + omega[:, None] * zrun)
        yrun = (rng.random((K, RUNIN)) < prun).astype(np.uint8)
        run_rate = yrun.mean(axis=1)
        z = rng.normal(size=(K, int(lengths.max())))
        risk = sigmoid(alpha[:, None] + omega[:, None] * z)
        y = (rng.random(risk.shape) < risk).astype(np.uint8)
        bases = gh_mean(alpha, omega)
        mask = np.arange(risk.shape[1])[None, :] < lengths[:, None]
        scored_p = bases
        threshold = float(y[mask].mean())
        scores = [risk, np.broadcast_to(bases[:, None], risk.shape), np.broadcast_to(run_rate[:, None], risk.shape)]
        names = ['true_risk','true_base_rate','runin_rate']
        for fcname, sc in zip(names, scores):
            rec[f'auc_pooled_{fcname}'].append(auc(y[mask], sc[mask]))
            aa = auc_by_patient(y, sc, lengths)
            for key, val in zip(['equal','days','pairs'], aa):
                rec[f'auc_within_{key}_{fcname}'].append(val)
            bm = binary_metrics(y, sc, lengths, threshold)
            for key, val in bm.items(): rec[f'{key}_{fcname}'].append(val)
        q = np.array([y[k, :n].mean() for k, n in enumerate(lengths)])
        rec['auc_observed_scored_rate'].append(auc(y[mask], np.broadcast_to(q[:, None], risk.shape)[mask]))
        a0, mu = weighted_exact_a0(bases, lengths)
        rec['a0_exact'].append(a0)
        w = lengths / lengths.sum()
        sig = float(np.sqrt(np.sum(w * (bases - mu) ** 2)))
        rec['day_weighted_mu'].append(mu)
        rec['day_weighted_sigma'].append(sig)
        rec['a0_normal_approx'].append(0.5 * (1 + math.erf((sig / (math.sqrt(2) * mu * (1-mu))) / math.sqrt(2))))
        if length_type == 'rank':
            rec['spearman_length_intercept'].append(float(np.corrcoef(np.argsort(np.argsort(alpha)), np.argsort(np.argsort(lengths)))[0,1]))
        rec['mean_length'].append(float(lengths.mean()))
    out = {}
    for key, vals in rec.items():
        a = np.asarray(vals, dtype=float)
        ok = np.isfinite(a)
        out[key] = {'mean': float(np.mean(a[ok])), 'mcse': float(np.std(a[ok], ddof=1) / math.sqrt(ok.sum())), 'n': int(ok.sum())} if ok.any() else {'mean': None,'mcse':None,'n':0}
    return out


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    m = solve_m()
    names = ['reference','R1_pos06','R1_zero','R1_neg06','R2_gamma','R2_two_point','R3_lognormal_cv05','R3_lognormal_cv10','R3_half_zero']
    rng = np.random.default_rng(SEED)
    result = {'design': {'seed': SEED, 'replications': REPS, 'patients': K, 'reference_mu': MU_REF, 'tau': TAU, 'omega': OMEGA_REF,
                         'runin_days': RUNIN, 'reference_scored_days': 180, 'm_calibrated_for_reference_mu': m,
                         'R1_scored_length_range':[60,365], 'R1_target_spearman': {'R1_pos06':0.6,'R1_zero':0.0,'R1_neg06':-0.6},
                         'R2_gamma':'shifted Gamma(shape=1, scale=1), mean m and SD 1, skewness 2',
                         'R2_two_point':'10% and 90% masses, mean m and SD 1, skewness 2.67',
                         'R3_lognormal':'patient omega log-normal with mean 0.5 and CV 0.5 or 1.0',
                         'R3_half_zero':'half omega=0, half omega=1',
                         'threshold':'realized overall event rate among scored days in that replication',
                         'A0':'exact day-weighted benchmark over true patient base rates; normal approximation uses day-weighted rate mean and SD',
                         'within_auc':'eligible patients require at least one event and non-event; summaries are equal patient, scored-day weighted, or within-patient pair weighted'},
              'scenarios': {}, 'elapsed_seconds': None}
    for i, name in enumerate(names):
        print(f'Running {name} ({i+1}/{len(names)})', flush=True)
        result['scenarios'][name] = run_scenario(name, rng, m)
    result['elapsed_seconds'] = time.time() - start
    (OUT / 'robustness_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"Finished in {result['elapsed_seconds']:.1f}s; wrote {OUT / 'robustness_results.json'}")

if __name__ == '__main__':
    main()





