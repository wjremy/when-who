"""Is the dynamic-control slope shortfall explained by the logit clipping rule?"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
import pandas as pd
from simulation import build_histories,beta_k_from_rate_variance
from fit_core import slopes
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OPEN=ROOT/'open_data'
OUT=OPEN/'results'/'clipping_sensitivity'
OUT.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(20261008);rows=[]
for c,runin in [('replace_bg',14),('ctn_weekly_missing_excluded',4),('ces',20)]:
    d=pd.read_csv(OPEN/f'forecasts_{c}.csv')
    z=d.groupby('patient').y.agg(T='size',rate='mean'); lens=z['T'].to_numpy()
    k,*_=beta_k_from_rate_variance(d.y.mean(),z.rate.var(ddof=1),lens,.6)
    for mode in ['markov','expanding_last']:
        y,p,g=build_histories(np.tile(lens,10),d.y.mean(),k,.6,mode,runin,rng)
        a=slopes(y,p,g,clip=1e-6); b=slopes(y,p,g,clip=1e-15)
        rows.append(dict(cohort=c,mode=mode,patients=len(lens)*10,rho=.6,
            clipped_fraction=np.mean((p<1e-6)|(p>1-1e-6)),
            pooled_clip=a[0],joint_clip=a[1],conditional_clip=a[2],
            pooled_near_unclipped=b[0],joint_near_unclipped=b[1],conditional_near_unclipped=b[2]))
pd.DataFrame(rows).to_csv(OUT/'clipping_sensitivity.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
