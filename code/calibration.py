"""Insight 4: pooled calibration hides within-patient miscalibration.
Calibration slopes: y ~ logit(p) (pooled); y ~ patient-mean logit + within-deviation (between / within slopes)."""
import numpy as np, pandas as pd, json
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, logit
rng = np.random.default_rng(11)

def slopes(y, p, pid):
    lp = logit(np.clip(p, 1e-6, 1 - 1e-6))
    s = pd.Series(lp); mk = s.groupby(pid).transform("mean").values
    pooled = LogisticRegression(penalty=None, max_iter=1000).fit(lp[:, None], y).coef_[0, 0]
    bw = LogisticRegression(penalty=None, max_iter=1000).fit(np.column_stack([mk, lp - mk]), y).coef_[0]
    bins = pd.qcut(p, 10, duplicates="drop")
    g = pd.DataFrame(dict(y=y, p=p, b=bins)).groupby("b", observed=True)
    ece = float((g.size() / len(y) * (g.y.mean() - g.p.mean()).abs()).sum())
    return dict(pooled_slope=round(pooled, 2), between_slope=round(bw[0], 2), within_slope=round(bw[1], 2), pooled_ECE=round(ece, 4))

out = {"synthetic": []}
K, T, omega, mu = 400, 300, 0.5, 0.15
for tau in (0.5, 1.5):
    a = logit(mu) + rng.normal(0, tau, K); h = rng.normal(0, 1, (K, T))
    y = (rng.random((K, T)) < expit(a[:, None] + omega * h)).ravel().astype(int)
    pid = np.repeat(np.arange(K), T)
    for c in (0.25, 1.0, 3.0):
        eta = a[:, None] + c * omega * h
        eta = eta - np.log(np.mean(expit(eta)) / mu) * 0     # keep base rates as they are
        r = dict(tau=tau, within_scale=c, **slopes(y, expit(eta).ravel(), pid)); out["synthetic"].append(r); print(r)

json.dump(out, open("results/calibration.json", "w"), indent=1)
