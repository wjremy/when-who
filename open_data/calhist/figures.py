"""Standalone descriptive patient calibration figures; matplotlib, no external edits."""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RESULTS=ROOT/'open_data'/'results'
DATA=RESULTS/'cohort_analysis'
OUT=RESULTS/'figures'
OUT.mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(OUT/'.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
citl=pd.read_csv(DATA/'patient_citl.csv')
bins=pd.read_csv(DATA/'within_patient_bins.csv')
names={'replace_bg':'REPLACE-BG','ctn_weekly_missing_excluded':'CTN-0094','ces':'CES'}
fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
for ax, bx, (cohort,name) in zip(axes[0],axes[1],names.items()):
    cohort_error=citl.loc[citl.cohort==cohort,'error']*100
    xmin=min(-6.,float(cohort_error.min())-3.);xmax=max(6.,float(cohort_error.max())+3.)
    for j,(forecast,color) in enumerate([('gradient_boosting','#0066aa'),('runin_rate','#888888'),
            ('updating_rate' if 'ctn' in cohort else 'expanding_rate','#cc6600')]):
        v=np.sort(citl.loc[(citl.cohort==cohort)&(citl.forecast==forecast),'error'].to_numpy())
        ax.step(np.r_[xmin,100*v,xmax],np.r_[0.,np.arange(1,len(v)+1)/len(v),1.],
            where='post',label=forecast.replace('_',' '),color=color)
    ax.axvspan(-5,5,color='#dddddd',alpha=.5);ax.axvline(0,color='black',lw=.6)
    ax.set(title=name,xlabel='Observed minus mean forecast (percentage points)',ylabel='Fraction of patients')
    ax.set_xlim(xmin,xmax);ax.set_ylim(0,1)
    bb=bins[bins.cohort==cohort]
    x=np.arange(len(bb)); y=100*bb.bin_error.to_numpy()
    bx.errorbar(x,y,yerr=np.vstack([y-100*bb.bin_error_lo,100*bb.bin_error_hi-y]),fmt='o',capsize=3,color='#0066aa')
    bx.axhline(0,color='black',lw=.7)
    bx.set(xticks=x,xticklabels=['<−.10','−.10 to\n−.025','−.025 to\n.025','.025 to\n.10','≥.10'],
        xlabel='Current forecast minus mean of prior forecasts',ylabel='Observed minus forecast (percentage points)')
    bx.grid(axis='y',alpha=.2)
axes[0,0].legend(fontsize=8,loc='lower right')
fig.savefig(OUT/'patient_calibration.pdf');fig.savefig(OUT/'patient_calibration.png',dpi=180)
