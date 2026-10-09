import numpy as np
from scipy import stats, integrate
from scipy.special import expit
from metrics import floor_auc_beta, floor_auc_approx
rng = np.random.default_rng(7)

def gmd_mc(x, m=400000):
    a = rng.choice(x, m); b = rng.choice(x, m); return np.abs(a-b).mean()

# (a) exact floor = 1/2 + E|p-p'|/(4 mu (1-mu)) vs Beta closed form
print("mu   sd   beta_exact  gmd_formula  phi_approx  linear_approx")
for mu, sd in [(0.05,0.03),(0.1,0.05),(0.1,0.08),(0.2,0.05),(0.2,0.1),(0.2,0.15),(0.3,0.2),(0.5,0.2)]:
    k = mu*(1-mu)/sd**2-1; a, b = mu*k, (1-mu)*k
    p = rng.beta(a, b, 2_000_000)
    g = gmd_mc(p)
    lin = 0.5 + sd/(2*np.sqrt(np.pi)*mu*(1-mu))
    print(f"{mu:.2f} {sd:.2f}  {floor_auc_beta(mu,sd):.4f}     {0.5+g/(4*mu*(1-mu)):.4f}      {floor_auc_approx(mu,sd):.4f}      {lin:.4f}")

# (c) random-intercept logistic, calibrated scores: A_P vs A_W (population limit)
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(60); gh_w = gh_w/gh_w.sum()
def auc_cal(m, s, n=4000):
    # AUC of calibrated logit-normal risk: 1/2 + E|R-R'|/(4 mu(1-mu)); via sorted-sample formula
    z = stats.norm.ppf((np.arange(n)+0.5)/n); r = np.sort(expit(m + s*z))
    mu = r.mean(); i = np.arange(1, n+1)
    gmd = 2*np.sum((2*i - n - 1)*r)/(n*n)       # E|R-R'| for the empirical (quantile) distribution
    return 0.5 + gmd/(4*mu*(1-mu)), mu
worst = 1
rows = []
for m in [-4,-3,-2,-1.5,-1,0]:
    for tau in [0.1,0.25,0.5,1,1.5,2]:
        for om in [0.05,0.1,0.25,0.5,1,1.5,2]:
            AP,_ = auc_cal(m, np.hypot(tau, om))
            ak = []; wk = []
            for x, w in zip(gh_x, gh_w):
                a, mk = auc_cal(m + tau*x, om, 2000); ak.append(a); wk.append(w)
            AW = np.dot(wk, ak)
            worst = min(worst, AP-AW); rows.append((m,tau,om,AP,AW))
print("min(A_P - A_W) over grid:", round(worst,4))
for r in rows:
    if r[2] in (0.25,1.0) and r[0] in (-2,-1.5): print("m=%.1f tau=%.2f om=%.2f  A_P=%.3f A_W=%.3f gap=%+.3f"%(r[0],r[1],r[2],r[3],r[4],r[3]-r[4]))
