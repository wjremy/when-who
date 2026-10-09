"""Pooled vs within-patient evaluation metrics for clustered (patient-day) predictions."""
import numpy as np
from scipy.stats import norm, rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


def pooled_auc(y, s):
    return roc_auc_score(y, s)


def per_patient(y, s, pid, fn):
    """Apply fn(y_k, s_k) to each patient with >=1 event and >=1 non-event."""
    out = {}
    order = np.argsort(pid, kind="stable")
    y, s, pid = y[order], s[order], pid[order]
    cuts = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1], True])
    for a, b in zip(cuts[:-1], cuts[1:]):
        yk = y[a:b]
        npos = yk.sum()
        if 0 < npos < len(yk):
            out[pid[a]] = (fn(yk, s[a:b]), npos, len(yk) - npos)
    return out


def within_auc(y, s, pid, weight="patient"):
    d = per_patient(y, s, pid, roc_auc_score)
    a = np.array([v[0] for v in d.values()])
    if weight == "patient":
        return a.mean()
    w = np.array([v[1] * v[2] for v in d.values()], float)   # pair-weighted
    return (a * w).sum() / w.sum()


def within_aupr_lift(y, s, pid):
    """Mean over patients of AUPR_k minus patient prevalence (AUPR of a random score)."""
    d = per_patient(y, s, pid, lambda yk, sk: average_precision_score(yk, sk) - yk.mean())
    return np.mean([v[0] for v in d.values()])


def within_aupr(y, s, pid):
    d = per_patient(y, s, pid, average_precision_score)
    return np.mean([v[0] for v in d.values()])


def decompose_auc(y, s, pid):
    """Global AUC = W_intra*A_intra + W_inter*A_inter (pair-count weights)."""
    d = per_patient(y, s, pid, roc_auc_score)
    npos_tot, nneg_tot = y.sum(), (1 - y).sum()
    pairs_intra = sum(v[1] * v[2] for v in d.values())
    a_intra = sum(v[0] * v[1] * v[2] for v in d.values()) / pairs_intra
    w_intra = pairs_intra / (npos_tot * nneg_tot)
    g = roc_auc_score(y, s)
    a_inter = (g - w_intra * a_intra) / (1 - w_intra)
    return dict(global_auc=g, w_intra=w_intra, a_intra=a_intra, a_inter=a_inter)


def threshold_metrics(y, s, pid, thr):
    """Pooled and patient-averaged sensitivity/specificity at a fixed threshold."""
    yhat = (s >= thr).astype(int)
    tp = ((yhat == 1) & (y == 1)).sum(); fn = ((yhat == 0) & (y == 1)).sum()
    tn = ((yhat == 0) & (y == 0)).sum(); fp = ((yhat == 1) & (y == 0)).sum()
    pooled = dict(sens=tp / (tp + fn), spec=tn / (tn + fp))
    d = per_patient(y, yhat.astype(float), pid,
                    lambda yk, hk: (hk[yk == 1].mean(), 1 - hk[yk == 0].mean()))
    se = np.mean([v[0][0] for v in d.values()]); sp = np.mean([v[0][1] for v in d.values()])
    return dict(pooled_sens=pooled["sens"], pooled_spec=pooled["spec"],
                pooled_youden=pooled["sens"] + pooled["spec"] - 1,
                within_sens=se, within_spec=sp, within_youden=se + sp - 1)


def icc_binary(y, pid):
    """ANOVA-type ICC on the proportion scale, corrected for binomial noise."""
    import pandas as pd
    g = pd.DataFrame({"y": y, "p": pid}).groupby("p").y.agg(["mean", "size"])
    mu = y.mean()
    var_obs = g["mean"].var(ddof=1)
    noise = (g["mean"] * (1 - g["mean"]) / g["size"]).mean()
    sb2 = max(var_obs - noise, 0.0)
    return dict(mu=mu, mean_pk=g["mean"].mean(), sd_pk_obs=np.sqrt(var_obs), sd_pk=np.sqrt(sb2),
                icc=sb2 / (mu * (1 - mu)), K=len(g))


def floor_auc_beta(mu, sd):
    """Exact pooled AUC of a base-rate-only scorer when p_k ~ Beta(mean=mu, sd=sd)."""
    from scipy import integrate, stats
    k = mu * (1 - mu) / sd ** 2 - 1
    a, b = mu * k, (1 - mu) * k
    X, Y = stats.beta(a + 1, b), stats.beta(a, b + 1)
    return integrate.quad(lambda t: X.sf(t) * Y.pdf(t), 0, 1)[0]


def floor_auc_approx(mu, sd):
    """Normal approximation: Phi(sd / (sqrt(2) * mu * (1 - mu)))."""
    return norm.cdf(sd / (np.sqrt(2) * mu * (1 - mu)))


def floor_auc_empirical(pk, wk=None):
    """Exact pooled AUC of the scorer s = p_k for an empirical set of patient rates p_k
    (equal numbers of days per patient unless weights wk are given)."""
    pk = np.asarray(pk, float)
    wk = np.ones_like(pk) if wk is None else np.asarray(wk, float)
    pos, neg = wk * pk, wk * (1 - pk)
    r = rankdata(pk)                          # handles ties with 1/2 credit via midranks
    order = np.argsort(pk)
    # P(s+ > s-) + 0.5 P(tie) over patient-level mass
    tot = 0.0
    for i in range(len(pk)):
        greater = neg[pk < pk[i]].sum()
        ties = neg[pk == pk[i]].sum()
        tot += pos[i] * (greater + 0.5 * ties)
    return tot / (pos.sum() * neg.sum())
