"""Probability-scale patient summaries and predictable within-patient diagnostics.

No model refits, history-feature reconstruction, or writes outside HERE.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys
sys.dont_write_bytecode=True
import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy, statsmodels
from scipy.stats import t, chi2
from fit_core import slopes

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DATA=ROOT/'open_data'
OUT=DATA/'results'/'cohort_analysis'
SPECS=[('replace_bg','expanding_rate'),('ctn_weekly_missing_excluded','updating_rate'),('ces','expanding_rate')]

def main(nboot=2000,seed=20261007):
    OUT.mkdir(parents=True,exist_ok=True)
    summaries=[]; individual=[]; bins=[]; moments=[]; slope_rows=[]; targets=[]; metadata=[]
    edges=np.array([-np.inf,-.10,-.025,.025,.10,np.inf])
    for k,(cohort,updated) in enumerate(SPECS):
        path=DATA/f'forecasts_{cohort}.csv'
        df=pd.read_csv(path).sort_values(['patient','t'],kind='stable').reset_index(drop=True)
        assert not df.duplicated(['patient','t']).any()
        assert set(df.y.unique())=={0,1}
        assert df[['gradient_boosting','runin_rate',updated]].notna().all().all()
        g,pid=pd.factorize(df.patient,sort=False); G=len(pid); n=np.bincount(g)
        y=df.y.to_numpy(float); event=np.bincount(g,weights=y); rates=event/n
        targets.append(dict(cohort=cohort,patients=G,observations=len(df),event_rate=y.mean(),
            patient_mean_rate=rates.mean(),patient_rate_sd=rates.std(),length_mean=n.mean(),
            length_min=n.min(),length_q25=np.quantile(n,.25),length_median=np.median(n),
            length_q75=np.quantile(n,.75),length_max=n.max(),all_zero=int(sum(event==0)),all_one=int(sum(event==n))))
        rng=np.random.default_rng(seed+k)
        mult=rng.multinomial(G,np.full(G,1/G),size=nboot)
        for col in ['gradient_boosting','runin_rate',updated]:
            p=df[col].to_numpy(float); assert ((p>=0)&(p<=1)).all()
            error=rates-np.bincount(g,weights=p)/n
            boot_mean=(mult@error)/G
            rows=pd.DataFrame(dict(cohort=cohort,forecast=col,patient=pid,
                n=n,events=event,observed_rate=rates,mean_forecast=np.bincount(g,weights=p)/n,
                error=error))
            individual.append(rows)
            summaries.append(dict(cohort=cohort,forecast=col,patients=G,mean=error.mean(),
                mean_lo=np.quantile(boot_mean,.025),mean_hi=np.quantile(boot_mean,.975),
                median=np.median(error),q25=np.quantile(error,.25),q75=np.quantile(error,.75),
                iqr=np.quantile(error,.75)-np.quantile(error,.25),
                within_005=int(sum(abs(error)<=.05)),share_within_005=np.mean(abs(error)<=.05),
                observed_patient_mean=rates.mean(),forecast_patient_mean=(rates-error).mean()))
        p=df.gradient_boosting.to_numpy(float); residual=y-p
        # For first occasion baseline=p_0; subsequent baseline averages strictly prior forecasts.
        prior=np.empty(len(df)); lagres=np.empty(len(df))
        for j in range(G):
            idx=np.flatnonzero(g==j); pp=p[idx]
            prior[idx]=np.r_[pp[0],np.cumsum(pp)[:-1]/np.arange(1,len(pp))]
            lagres[idx]=np.r_[0.,residual[idx][:-1]]
        delta=p-prior
        assignment=np.searchsorted(edges,delta,side='right')-1
        s,jo,co=slopes(y,p,g)
        slope_rows.append(dict(cohort=cohort,pooled=s,joint_within=jo,conditional=co,
            both_outcomes=int(sum((event>0)&(event<n))),all_patients=G))
        patient_m=[]
        for name,h in [('level',np.ones(len(df))),('forecast_excursion',delta),
                       ('lagged_residual',lagres),('high_minus_low',
                        (delta>.025).astype(float)-(delta<-.025).astype(float))]:
            m=np.bincount(g,weights=h*residual,minlength=G)/n
            se=m.std(ddof=1)/np.sqrt(G); stat=m.mean()/se if se else np.nan
            ci=t.ppf(.975,G-1)*se
            bm=mult@m/G
            moments.append(dict(cohort=cohort,instrument=name,mean=m.mean(),cluster_se=se,
                t_lo=m.mean()-ci,t_hi=m.mean()+ci,t_stat=stat,pvalue=2*t.sf(abs(stat),G-1),
                bootstrap_lo=np.quantile(bm,.025),bootstrap_hi=np.quantile(bm,.975)))
            patient_m.append(m)
        # Exploratory joint test of level, excursion, and lagged residual moments.
        M=np.column_stack(patient_m[:3]); cov=np.cov(M,rowvar=False,ddof=1)/G
        rank=np.linalg.matrix_rank(cov); w=float(M.mean(0)@np.linalg.pinv(cov)@M.mean(0))
        moments.append(dict(cohort=cohort,instrument='joint_first_three',mean=np.nan,
            wald_chi2=w,df=rank,pvalue=chi2.sf(w,rank)))
        for b in range(5):
            mask=assignment==b
            exposure=np.bincount(g,weights=mask,minlength=G)/n
            psum=np.bincount(g,weights=mask*p,minlength=G)/n
            ysum=np.bincount(g,weights=mask*y,minlength=G)/n
            err=ysum-psum; exp=exposure.mean()
            # Ratio is a patient-opportunity-weighted bin diagnostic. CI resamples entire patients.
            denom=mult@exposure
            boot=np.divide(mult@err,denom,out=np.full(nboot,np.nan),where=denom>0)
            se=err.std(ddof=1)/np.sqrt(G); c=t.ppf(.975,G-1)*se
            bins.append(dict(cohort=cohort,bin=b,delta_lo=edges[b],delta_hi=edges[b+1],
                cells=int(sum(mask)),patients=int(sum(exposure>0)),equal_patient_exposure=exp,
                mean_forecast=psum.mean()/exp if exp else np.nan,
                observed_rate=ysum.mean()/exp if exp else np.nan,
                bin_error=err.mean()/exp if exp else np.nan,
                bin_error_lo=np.nanquantile(boot,.025),bin_error_hi=np.nanquantile(boot,.975),
                moment=err.mean(),moment_t_lo=err.mean()-c,moment_t_hi=err.mean()+c))
        df.assign(prior_mean_forecast=prior,forecast_excursion=delta,bin=assignment,
            residual=residual).to_csv(OUT/f'{cohort}_gb_diagnostics.csv',index=False)
        metadata.append(dict(cohort=cohort,input=f'open_data/forecasts_{cohort}.csv',
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(cohort,'CITL done, slopes',s,jo,co,flush=True)
    pd.DataFrame(summaries).to_csv(OUT/'patient_citl_summary.csv',index=False)
    pd.concat(individual).to_csv(OUT/'patient_citl.csv',index=False)
    pd.DataFrame(bins).to_csv(OUT/'within_patient_bins.csv',index=False)
    pd.DataFrame(moments).to_csv(OUT/'sequential_moments.csv',index=False)
    pd.DataFrame(slope_rows).to_csv(OUT/'cohort_slopes.csv',index=False)
    pd.DataFrame(targets).to_csv(OUT/'cohort_targets.csv',index=False)
    (OUT/'cohort_metadata.json').write_text(json.dumps(dict(seed=seed,nboot=nboot,
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        scipy=scipy.__version__,statsmodels=statsmodels.__version__,inputs=metadata),indent=2),
        encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--nboot',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=20261007);a=parser.parse_args()
    main(a.nboot,a.seed)
