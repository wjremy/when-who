"""Numerical helpers for the known-calibration simulation."""
import ctypes
import os
import shutil
import subprocess
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'results' / 'calibration_known_sim_build'
LIBRARY = BUILD / ('conditional_dp.dll' if os.name == 'nt' else 'conditional_dp.so')


def _build_library():
    BUILD.mkdir(parents=True, exist_ok=True)
    compiler = os.environ.get('CC') or shutil.which('gcc') or shutil.which('clang')
    if not compiler:
        raise RuntimeError('A C compiler (gcc or clang) is required for the conditional slope.')
    command = [compiler, '-O3', '-shared']
    if os.name == 'nt':
        command.append('-Wl,--export-all-symbols')
    else:
        command.append('-fPIC')
    command += [str(Path(__file__).with_name('calibration_known_dp.c')), '-o', str(LIBRARY), '-lm']
    subprocess.run(command, check=True)


class _Native:
    pass


_native = _Native()
try:
    if os.environ.get('CONDITIONAL_DP') == 'numba':
        raise OSError('numba requested')
    BUILD.mkdir(parents=True, exist_ok=True)
    if not LIBRARY.exists():
        _build_library()
    _c = ctypes.CDLL(str(LIBRARY))
    ptr = np.ctypeslib.ndpointer
    _c.group_eval.argtypes = [
        ctypes.c_int,
        ptr(np.int32, flags='C_CONTIGUOUS'),
        ptr(np.int32, flags='C_CONTIGUOUS'),
        ptr(np.float64, flags='C_CONTIGUOUS'),
        ptr(np.float64, flags='C_CONTIGUOUS'),
        ctypes.c_double,
        ptr(np.float64, flags='C_CONTIGUOUS'),
    ]
    _native.group_eval = _c.group_eval
except (OSError, RuntimeError, subprocess.CalledProcessError):
    # Same algorithm compiled by numba, for machines that cannot build or load
    # the C library (for example when Windows Application Control blocks unsigned DLLs).
    import sys
    sys.path.insert(0, str(ROOT / 'open_data' / 'calhist'))
    from fit_core import _numba_group_eval
    _native.group_eval = _numba_group_eval()


class StableConditionalLogit:
    """Exact conditional-logit slope using the scaled likelihood recurrence."""

    def __init__(self, y, x, groups):
        G = int(np.max(groups)) + 1
        arrays = []
        counts = []
        sums = []
        starts = [0]
        for g in range(G):
            xx = x[groups == g].copy()
            yy = y[groups == g].copy()
            xx -= xx.mean()
            if yy.sum() > len(yy) / 2:
                yy = 1 - yy
                xx = -xx
            m = int(yy.sum())
            if m == 0:
                xx = np.zeros(0)
            arrays.append(xx)
            counts.append(m)
            sums.append(float(xx @ yy) if len(xx) else 0)
            starts.append(starts[-1] + len(xx))
        self.starts = np.array(starts, np.int32)
        self.counts = np.array(counts, np.int32)
        self.xx = np.concatenate(arrays)
        self.ysum = np.array(sums)
        self.weights = np.ones(G)
        self.last = None
        self.values = np.empty((G, 3))
        self.G = G

    def evaluate(self, beta):
        b = float(np.asarray(beta).ravel()[0])
        if b != self.last:
            _native.group_eval(self.G, self.starts, self.counts, self.xx,
                               self.ysum, b, self.values)
            self.last = b
        return self.weights @ self.values

    def estimate(self, weights=None, start=1):
        self.weights = np.ones(self.G) if weights is None else weights
        if self.counts.sum() == 0:
            return np.nan
        center = float(start) if np.isfinite(start) else 0.0
        lo = center - 0.5
        hi = center + 0.5
        for _ in range(7):
            sl = self.evaluate(lo)[1]
            sh = self.evaluate(hi)[1]
            if sl > 0 and sh < 0:
                break
            if sl <= 0:
                lo -= max(1, center - lo)
            if sh >= 0:
                hi += max(1, hi - center)
        else:
            return np.nan
        try:
            return brentq(lambda b: self.evaluate(b)[1], lo, hi, xtol=1e-9)
        except (ValueError, RuntimeError):
            return np.nan


def irls(X, y, w):
    b = np.zeros(X.shape[1])
    mu = np.average(y, weights=w)
    b[0] = logit(np.clip(mu, 1e-6, 1 - 1e-6))
    for _ in range(100):
        p = expit(np.clip(X @ b, -35, 35))
        v = np.maximum(p * (1 - p), 1e-10) * w
        try:
            step = np.linalg.solve(X.T @ (v[:, None] * X) + np.eye(X.shape[1]) * 1e-10,
                                   X.T @ (w * (y - p)))
        except np.linalg.LinAlgError:
            return np.full(X.shape[1], np.nan)
        b += step
        if np.max(abs(step)) < 1e-8:
            return b
    return np.full(X.shape[1], np.nan)
