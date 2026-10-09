"""Exact expected static conditional likelihood under calibrated Markov forecasts."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys
sys.dont_write_bytecode=True
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.optimize import brentq
from fit_core import ConditionalFit

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'open_data'/'results'/'exact_counterexample'
OUT.mkdir(parents=True,exist_ok=True)
out=[]
for T in [3,4,8,12]:
    ys=[]; ps=[]; weights=[]
    for path in product([0,1],repeat=T+1):
        prev=np.array(path[:-1]);y=np.array(path[1:]);p=.2+.6*prev
        weight=.5*np.prod(np.where(y,p,1-p))
        ys.extend(y);ps.extend(p);weights.append(weight)
    g=np.repeat(np.arange(len(weights)),T)
    model=ConditionalFit(np.array(ys,float),logit(np.array(ps)),g)
    w=np.array(weights);assert abs(w.sum()-1)<1e-12
    root=brentq(lambda b:w@model.evaluate(b)[:,1],-1,2,xtol=1e-12)
    # Predictable centered-excursion moment has expectation zero by direct enumeration.
    residual=np.array(ys)-np.array(ps)
    h=[]
    for pp in np.array(ps).reshape(-1,T):
        h.extend(pp-np.r_[pp[0],np.cumsum(pp)[:-1]/np.arange(1,T)])
    moment=w@((np.array(h)*residual).reshape(-1,T).mean(1))
    assert abs(moment)<1e-12
    out.append(dict(scored_length=T,paths=len(weights),conditional_target=root,
        true_sequential_slope=1.,predictable_excursion_moment=moment))
pd.DataFrame(out).to_csv(OUT/'exact_markov.csv',index=False)
print(pd.DataFrame(out).to_string(index=False))
