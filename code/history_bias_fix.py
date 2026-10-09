"""Insight 1b: permutation-null correction where recent history truly carries signal.
DGP: logit p_kt = a_k + b * z_kt, z AR(1) (phi=0.9, unit variance), so attacks cluster."""
import numpy as np, json, sys
sys.path.insert(0, "code")
from history_bias import within_auc_matrix, features
rng = np.random.default_rng(7)
from scipy.special import expit, logit

def sim(K, T, b, mu=0.15, phi=0.9):
    z = np.zeros((K, T)); e = rng.normal(0, 1, K)
    for t in range(T):
        e = phi * e + np.sqrt(1 - phi**2) * rng.normal(0, 1, K); z[:, t] = e
    a = logit(mu) + rng.normal(0, 0.5, K)
    return (rng.random((K, T)) < expit(a[:, None] + b * z)).astype(float)

def null_auc(Y, w, start, n_perm=20):
    vals = []
    for _ in range(n_perm):
        Yp = rng.permuted(Y, axis=1)                  # shuffle days within each patient
        vals.append(within_auc_matrix(Yp, features(Yp, "window", w), start)[0])
    return float(np.mean(vals))

res = []
for b in (0.0, 0.5, 1.0):
    truth = {w: within_auc_matrix(Y := sim(60, 20000, b), features(Y, "window", w), 30)[0] for w in (7, 30)}
    for T in (100, 365):
        Y = sim(1500, T, b)
        for w in (7, 30):
            raw = within_auc_matrix(Y, features(Y, "window", w), 30)[0]
            nul = null_auc(Y, w, 30)
            res.append(dict(signal_b=b, T=T, window=w, truth_long_diary=round(truth[w], 3),
                            raw=round(raw, 3), null=round(nul, 3), corrected=round(raw - (nul - 0.5), 3)))
            print(res[-1])
json.dump(res, open("results/history_bias_fix.json", "w"), indent=1)
