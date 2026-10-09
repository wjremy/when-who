"""Reproduce static working calibration slopes without writing outside this folder."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import ctypes
import shutil
import subprocess
from pathlib import Path
import numpy as np
from scipy.special import expit, logit
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = ROOT / 'open_data' / 'results' / 'fit_core'
LIBRARY = BUILD / ('conditional_dp.dll' if os.name == 'nt' else 'conditional_dp.so')
def _build_library():
    BUILD.mkdir(parents=True, exist_ok=True)
    compiler = os.environ.get('CC') or shutil.which('gcc') or shutil.which('clang')
    if not compiler:
        raise RuntimeError('no C compiler (gcc or clang) on PATH')
    command = [compiler, '-O3', '-shared']
    if os.name == 'nt':
        # The public C function must be exported from the Windows DLL.
        command.append('-Wl,--export-all-symbols')
    else:
        command.append('-fPIC')
    command += [str(HERE / 'conditional_dp.c'), '-o', str(LIBRARY), '-lm']
    subprocess.run(command, check=True)


def _numba_group_eval():
    """Same algorithm as conditional_dp.c, compiled in memory by numba.
    Used when the C library cannot be built or loaded (for example when Windows
    Application Control blocks unsigned DLLs)."""
    import numba

    @numba.njit(cache=False, error_model='numpy')  # C semantics: x/0 gives inf or nan, no exception
    def group_eval(G, starts, counts, x, ysum, beta, out):
        for g in range(G):
            n = starts[g + 1] - starts[g]
            m = counts[g]
            if m == 0:
                out[3 * g] = 0.0; out[3 * g + 1] = 0.0; out[3 * g + 2] = 0.0
                continue
            d = np.zeros(m + 1); v = np.zeros(m + 1); q = np.zeros(m + 1)
            d[0] = 1.0
            scale = 0.0
            for t in range(n):
                z = x[starts[g] + t]
                w = np.exp(beta * z)
                hi = t + 1 if t + 1 < m else m
                for k in range(hi, 0, -1):
                    q[k] += w * (q[k - 1] + 2 * z * v[k - 1] + z * z * d[k - 1])
                    v[k] += w * (v[k - 1] + z * d[k - 1])
                    d[k] += w * d[k - 1]
                mx = 0.0
                for k in range(hi + 1):
                    if d[k] > mx:
                        mx = d[k]
                for k in range(hi + 1):
                    d[k] /= mx; v[k] /= mx; q[k] /= mx
                scale += np.log(mx)
            mu = v[m] / d[m]
            out[3 * g] = beta * ysum[g] - np.log(d[m]) - scale
            out[3 * g + 1] = ysum[g] - mu
            out[3 * g + 2] = max(q[m] / d[m] - mu * mu, 0.0)

    def call(G, starts, counts, x, ysum, beta, out):
        # The C function receives `out` as a flat pointer; pass a flat view of the (G, 3) array.
        group_eval(G, starts, counts, x, ysum, beta, out.reshape(-1))

    return call


class _Lib:
    pass


lib = _Lib()
try:
    if os.environ.get('CONDITIONAL_DP') == 'numba':
        raise OSError('numba requested')
    if not LIBRARY.exists():
        _build_library()
    _c = ctypes.CDLL(str(LIBRARY))
    ptr = np.ctypeslib.ndpointer
    _c.group_eval.argtypes = [ctypes.c_int, ptr(np.int32, flags='C_CONTIGUOUS'),
        ptr(np.int32, flags='C_CONTIGUOUS'), ptr(np.float64, flags='C_CONTIGUOUS'),
        ptr(np.float64, flags='C_CONTIGUOUS'), ctypes.c_double,
        ptr(np.float64, flags='C_CONTIGUOUS')]
    lib.group_eval = _c.group_eval
except (OSError, RuntimeError, subprocess.CalledProcessError):
    lib.group_eval = _numba_group_eval()

class ConditionalFit:
    def __init__(self, y, x, groups):
        self.G = int(np.max(groups))+1
        arrays, counts, sums, starts = [], [], [], [0]
        order = np.arange(len(groups)) if np.all(np.diff(groups)>=0) else np.argsort(groups,kind='stable')
        sx=x[order]; sy=y[order]
        boundaries=np.r_[0,np.cumsum(np.bincount(groups,minlength=self.G))]
        for g in range(self.G):
            xx = sx[boundaries[g]:boundaries[g+1]].copy()
            yy = sy[boundaries[g]:boundaries[g+1]].copy()
            xx -= xx.mean()
            if yy.sum() > len(yy)/2: yy = 1-yy; xx = -xx
            m = int(yy.sum())
            sums.append(float(xx@yy) if m else 0.)
            if not m: xx = np.zeros(0)
            arrays.append(xx); counts.append(m); starts.append(starts[-1]+len(xx))
        self.starts = np.array(starts, np.int32)
        self.counts = np.array(counts, np.int32)
        self.x = np.concatenate(arrays)
        self.sums = np.array(sums, dtype=float)
        self.values = np.zeros((self.G, 3))
        self.last = None

    def evaluate(self, beta):
        if beta != self.last:
            lib.group_eval(self.G, self.starts, self.counts, self.x,
                           self.sums, float(beta), self.values)
            self.last = beta
        return self.values

    def fit(self, mult=None):
        if mult is None: mult = np.ones(self.G)
        def score(b): return mult@self.evaluate(b)[:,1]
        if not self.counts.sum(): return np.nan
        lo, hi = -1., 2.
        for _ in range(9):
            if score(lo)>0 and score(hi)<0:
                return brentq(score, lo, hi, xtol=1e-9)
            if score(lo)<=0: lo *= 2
            if score(hi)>=0: hi *= 2
        return np.nan

def irls(X, y, w=None):
    if w is None: w = np.ones(len(y))
    # Center/scale non-intercept columns and backtrack each Newton step.
    # Calibration logits can have very different scales, particularly in short panels.
    center=np.average(X[:,1:],axis=0,weights=w)
    scale=np.sqrt(np.average((X[:,1:]-center)**2,axis=0,weights=w))
    if np.any(scale<1e-12): return np.full(X.shape[1],np.nan)
    Z=np.column_stack([np.ones(len(y)),(X[:,1:]-center)/scale])
    b=np.zeros(X.shape[1]); b[0]=logit(np.clip(np.average(y,weights=w),1e-6,1-1e-6))
    def loss(coef):
        eta=Z@coef
        return np.sum(w*(np.logaddexp(0,eta)-y*eta))
    def unscale(coef):
        slope=coef[1:]/scale
        return np.r_[coef[0]-center@slope,slope]
    for _ in range(100):
        p=expit(Z@b); v=np.maximum(p*(1-p),1e-12)*w
        score=Z.T@(w*(y-p))
        if np.max(abs(score))/np.sum(w)<1e-10: return unscale(b)
        try: step=np.linalg.solve(Z.T@(v[:,None]*Z)+np.eye(X.shape[1])*1e-12,score)
        except np.linalg.LinAlgError: return np.full(X.shape[1], np.nan)
        previous=loss(b); fraction=1.
        for _ in range(40):
            if loss(b+fraction*step)<=previous+1e-12: break
            fraction*=.5
        b+=fraction*step
        if np.max(abs(fraction*step))<1e-9: return unscale(b)
    return np.full(X.shape[1], np.nan)

def slopes(y, p, groups, clip=1e-6):
    y=np.asarray(y,float); p=np.asarray(p,float); groups=np.asarray(groups,int)
    lp=logit(np.clip(p,clip,1-clip)); n=np.bincount(groups)
    mean=np.bincount(groups,weights=lp)/n
    dev=lp-mean[groups]
    pooled=irls(np.column_stack([np.ones(len(y)),lp]),y)[1]
    if np.max(abs(dev))<1e-12: return pooled,np.nan,np.nan
    joint=irls(np.column_stack([np.ones(len(y)),mean[groups],dev]),y)[2]
    cond=ConditionalFit(y,lp,groups).fit()
    return pooled,joint,cond

if __name__=='__main__':
    import warnings
    from statsmodels.discrete.conditional_models import ConditionalLogit
    rng=np.random.default_rng(937)
    g=np.repeat(np.arange(8),12); x=rng.normal(size=len(g))
    y=rng.binomial(1,expit(np.repeat(rng.normal(size=8),12)+.7*x)).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        stock=ConditionalLogit(y,x[:,None],groups=g)
    ours=ConditionalFit(y,x,g)
    for b in [-1.,0.,.7,2.]:
        v=ours.evaluate(b)
        assert abs(stock.loglike([b])-v[:,0].sum())<1e-9
        assert abs(stock.score([b])[0]-v[:,1].sum())<1e-9
        assert abs(stock.hessian(np.array([b]))[0,0]+v[:,2].sum())<1e-5
    assert abs(stock.fit(disp=False).params[0]-ours.fit())<1e-4
    import statsmodels.api as sm
    X=np.column_stack([np.ones(len(y)),x])
    assert np.allclose(irls(X,y),sm.GLM(y,X,family=sm.families.Binomial()).fit().params,atol=1e-6)
    means=np.bincount(g,weights=x)/np.bincount(g)
    Z=np.column_stack([np.ones(len(y)),means[g],x-means[g]])
    assert np.allclose(irls(Z,y),sm.GLM(y,Z,family=sm.families.Binomial()).fit().params,atol=1e-6)
    print('Validated conditional likelihood, score, Hessian and slope, and pooled/joint optimizer against statsmodels.')
