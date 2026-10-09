"""Insight 1: within-patient AUC is biased downward for forecasts built from the patient's own event history."""
import numpy as np, json
from scipy.stats import norm
rng = np.random.default_rng(1)

def within_auc_matrix(Y, S, start):
    """Mean over patients of per-patient AUC (ties = 1/2), days >= start. Y,S: (K,T)."""
    Y = Y[:, start:]; S = S[:, start:]
    out = []
    for y, s in zip(Y, S):
        n1 = y.sum(); n0 = len(y) - n1
        if n1 == 0 or n0 == 0: continue
        order = np.argsort(s, kind="mergesort"); ss = s[order]
        # midranks
        _, inv, cnt = np.unique(ss, return_inverse=True, return_counts=True)
        cum = np.cumsum(cnt); mid = cum - (cnt - 1) / 2
        r = np.empty(len(s)); r[order] = mid[inv]
        out.append((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return np.mean(out), len(out)

def features(Y, kind, w=None):
    K, T = Y.shape
    c = np.concatenate([np.zeros((K, 1)), np.cumsum(Y, 1)[:, :-1]], 1)   # events before day t
    if kind == "expanding":
        return c / np.maximum(np.arange(T), 1)
    cw = np.concatenate([np.zeros((K, w)), c], 1)[:, :T]
    return c - cw                                                      # events in last w days

def approx(T, q, w, start):
    """Gaussian leading-order approximation: exchangeable diary with n=qT events.
    Mean gap event vs non-event day = -w/(T-1); var of window count ~ w q(1-q)(1-w/T)."""
    ws = np.arange(start, T) if w is None else np.full(T - start, w)
    d = -ws / (T - 1) / np.sqrt(ws * q * (1 - q) * (1 - ws / T) + 1e-12)
    return 0.5 + np.mean(norm.cdf(d / np.sqrt(2)) - 0.5)

def main():
    res = []
    K = 3000
    for T in (60, 100, 180, 365, 730):
        for p in (0.05, 0.15, 0.30):
            Y = (rng.random((K, T)) < p).astype(float)
            for kind, w in (("expanding", None), ("last30", 30), ("last7", 7)):
                start = 30
                S = features(Y, "expanding" if w is None else "window", w)
                a, n = within_auc_matrix(Y, S, start)
                res.append(dict(T=T, p=p, feature=kind, within_auc=round(a, 4), approx=round(approx(T, p, w, start), 4), patients=n))
                print(res[-1])
    json.dump(res, open("results/history_bias_grid.json", "w"), indent=1)


if __name__ == "__main__":
    main()
