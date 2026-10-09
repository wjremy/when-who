#!/usr/bin/env python3
"""Open-data reproduction analysis for the Nepal College Experience Study."""
from __future__ import annotations
import argparse, json, math, traceback
from pathlib import Path
import numpy as np
import pandas as pd


SEED=20261006
NBOOT=1000
NCAL=500
NPERM=200
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'ces_openesm'/'0040_nepal_ts.tsv'
OUT_JSON=ROOT/'ces_results.json'
OUT_MD=ROOT/'ces_results.md'
ITEMS=['nervous','worrying','depressed','anhedonia','total','affect','stressed','social_level','perception_worry','appearance_pleased','feel_smart','feel_good']
CURR=ITEMS

# Rank-based AUC and IRLS calibration helpers are defined in this script.
def rankdata(a,method='average'):
    if method!='average': raise ValueError('only average ranks are supported')
    a=np.asarray(a); order=np.argsort(a,kind='mergesort'); z=a[order]
    starts=np.r_[0,np.flatnonzero(z[1:]!=z[:-1])+1]; ends=np.r_[starts[1:],len(a)]
    rr=np.empty(len(a),float)
    for lo,hi in zip(starts,ends): rr[lo:hi]=(lo+1+hi)/2
    out=np.empty(len(a),float); out[order]=rr; return out
def auc(y,s):
    y=np.asarray(y,int); s=np.asarray(s,float); n1=int(y.sum()); n0=len(y)-n1
    if n1==0 or n0==0: return float('nan')
    r=rankdata(s,method='average')
    return float((r[y==1].sum()-n1*(n1+1)/2)/(n1*n0))
def patient_auc_values(y,s,pid):
    ans={}
    for p in np.unique(pid):
        ix=np.flatnonzero(pid==p); a=auc(y[ix],s[ix])
        if np.isfinite(a): ans[p]=a
    return ans
def within_auc(y,s,pid):
    z=patient_auc_values(y,s,pid); return float(np.mean(list(z.values()))) if z else float('nan')
def interval(x):
    a=np.asarray(x,float); a=a[np.isfinite(a)]
    return [float(np.percentile(a,2.5)),float(np.percentile(a,97.5))] if len(a) else [None,None]
def expit(x):
    x=np.asarray(x,float); return np.where(x>=0,1/(1+np.exp(-np.clip(x,-700,700))),np.exp(np.clip(x,-700,700))/(1+np.exp(np.clip(x,-700,700))))
def logit(p):
    p=np.clip(np.asarray(p,float),1e-8,1-1e-8); return np.log(p/(1-p))
def irls(X,y,weights=None,max_iter=100,tol=1e-9):
    X=np.column_stack([np.ones(len(y)),np.asarray(X,float)]); y=np.asarray(y,float)
    w=np.ones(len(y)) if weights is None else np.asarray(weights,float)
    b=np.zeros(X.shape[1]); b[0]=logit(np.average(y,weights=w))
    for _ in range(max_iter):
        mu=expit(np.clip(X@b,-30,30)); v=np.maximum(mu*(1-mu),1e-10)*w
        h=X.T@(v[:,None]*X); g=X.T@(w*(y-mu)); step=np.linalg.solve(h+np.eye(len(b))*1e-10,g)
        scale=1.
        while scale>1e-6:
            b2=b+scale*step; m=expit(np.clip(X@b2,-30,30))
            ll=np.sum(w*(y*np.log(np.clip(m,1e-12,1))+(1-y)*np.log(np.clip(1-m,1e-12,1))))
            m0=expit(np.clip(X@b,-30,30)); ll0=np.sum(w*(y*np.log(np.clip(m0,1e-12,1))+(1-y)*np.log(np.clip(1-m0,1e-12,1))))
            if ll>=ll0-1e-10: break
            scale*=.5
        b=b2
        if np.max(np.abs(scale*step))<tol: return b
    if np.max(np.abs(g))>1e-4: raise RuntimeError('IRLS did not converge')
    return b
def equal_weights(pid):
    _,inv,c=np.unique(pid,return_inverse=True,return_counts=True); w=1/c[inv]; return w*len(w)/w.sum()
def safe(v):
    if isinstance(v,dict): return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [safe(x) for x in v]
    if isinstance(v,np.ndarray): return safe(v.tolist())
    if isinstance(v,(np.integer,)): return int(v)
    if isinstance(v,(np.floating,float)): return float(v) if np.isfinite(v) else None
    if isinstance(v,(np.bool_,)): return bool(v)
    return v

def load_reports(path,cut=3):
    d=pd.read_csv(path,sep='\t',parse_dates=['date'])
    phq=['nervous','worrying','depressed','anhedonia']
    d['complete']=d[phq].notna().all(axis=1)
    d=d[d.complete].copy(); d['total']=d[phq].sum(axis=1); d['event']=(d.total>=cut).astype(int)
    d=d.sort_values(['id','date'],kind='mergesort').reset_index(drop=True)
    panels=[]
    for pid,g in d.groupby('id',sort=False):
        g=g.reset_index(drop=True); n=len(g)
        if n<60: continue
        ev=g.event.to_numpy(int); tot=g.total.to_numpy(float)
        # Each row forecasts the following completed PHQ-4 report. First 20 rows are run-in only.
        ix=np.arange(20,n-1); dates=g.date.to_numpy()
        q=g.iloc[ix][['date']+ITEMS].copy().reset_index(drop=True)
        q['pid']=pid; q['y']=ev[ix+1]; q['current_event']=ev[ix]
        q['gap_days']=(dates[ix]-dates[ix-1]).astype('timedelta64[D]').astype(float)
        q['runin_rate']=ev[:20].mean()
        q['expanding_rate']=np.cumsum(ev[:n-1])[ix]/(ix+1)
        q['last4_rate']=np.array([ev[i-3:i+1].mean() for i in ix])
        q['last12_rate']=np.array([ev[i-11:i+1].mean() for i in ix])
        q['persistence']=tot[ix]
        q['next_persistence']=tot[ix+1]
        q['runin_event_history']=[ev[:20].tolist()]*len(q)
        for c in CURR: q[c+'_centered']=q[c]-g.iloc[:20][c].mean()
        q['retro_rate']=q.y.mean(); panels.append(q)
    panel=pd.concat(panels,ignore_index=True) if panels else pd.DataFrame()
    # Per-person retrospective scored-outcome rate (reference only).
    panel['retro_rate']=panel.groupby('pid').y.transform('mean')
    return d,panel

def cohort_summary(reports,panel):
    n=panel.groupby('pid').size().to_numpy(); y=panel.y.to_numpy(); p=panel.groupby('pid').y.mean().to_numpy()
    mu=float(np.average(p,weights=n)); var=float(np.average((p-mu)**2,weights=n)); icc=var/(mu*(1-mu)) if 0<mu<1 else np.nan
    gaps=reports.sort_values(['id','date']).groupby('id').date.diff().dt.total_seconds().div(86400).dropna()
    within_pairs=float(sum(int(z*(z-1)) for z in n)/(len(y)*(len(y)-1))) if len(y)>1 else np.nan
    return {'included_people':int(panel.pid.nunique()),'scored_forecasts':len(panel),'event_rate':float(y.mean()),
      'mean_personal_event_rate':float(np.mean(p)),'sd_personal_event_rate':float(np.std(p,ddof=1)),
      'forecast_person_event_rate_mean':float(np.mean(p)),'forecast_person_event_rate_sd':float(np.std(p,ddof=1)),
      'icc_day_weighted':float(icc),'median_gap_days_consecutive_complete_reports':float(gaps.median()),
      'within_person_pair_share':within_pairs,'complete_reports':len(reports),'people_with_complete_reports':int(reports.id.nunique()),
      'people_ge60_reports':int((reports.groupby('id').size()>=60).sum()),'median_reports_per_person':float(reports.groupby('id').size().median())}
def split_people(panel):
    ids=np.array(sorted(panel.pid.unique()),dtype=object); rng=np.random.default_rng(SEED); rng.shuffle(ids); n=len(ids)
    return {p:('train' if i<int(.6*n) else 'validation' if i<int(.8*n) else 'test') for i,p in enumerate(ids)}
def make_features(df,cols,train_medians=None):
    x=df[cols].astype(float).copy()
    if train_medians is None: train_medians=x.median().fillna(0)
    miss=x.isna().astype(float); x=x.fillna(train_medians).fillna(0)
    x.columns=[str(c) for c in x.columns]; miss.columns=[str(c)+'__missing' for c in cols]
    return pd.concat([x,miss],axis=1),train_medians
def logloss(y,p):
    p=np.clip(np.asarray(p,float),1e-12,1-1e-12); y=np.asarray(y,float)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
def brier(y,p): return float(np.mean((np.asarray(p)-np.asarray(y))**2))
def score_metrics(y,p): return {'brier_pooled':brier(y,p),'brier_person_mean':float(np.mean([brier(y[ix],p[ix]) for ix in groups(np.arange(len(y)),None)])) if False else None}
def groups(pid):
    return [(k,np.flatnonzero(pid==k)) for k in np.unique(pid)]
def eval_scores(y,p,pid):
    per=[brier(y[ix],p[ix]) for _,ix in groups(pid)]
    return {'brier_pooled':brier(y,p),'brier_person_mean':float(np.mean(per)),
            'logloss_pooled':logloss(y,p),'logloss_person_mean':float(np.mean([logloss(y[ix],p[ix]) for _,ix in groups(pid)]))}
def auc_report(y,s,pid,nboot=NBOOT,seed=SEED+10):
    vals=patient_auc_values(y,s,pid); keys=np.unique(pid); ixmap={k:np.flatnonzero(pid==k) for k in keys}
    pointp=auc(y,s); pointw=float(np.mean(list(vals.values()))) if vals else np.nan
    rng=np.random.default_rng(seed); bp=[]; bw=[]; bg=[]
    for _ in range(nboot):
        sel=rng.choice(keys,len(keys),replace=True); ix=np.concatenate([ixmap[k] for k in sel]); ap=auc(y[ix],s[ix])
        av=[vals[k] for k in sel if k in vals]; aw=float(np.mean(av)) if av else np.nan
        bp.append(ap); bw.append(aw); bg.append(ap-aw)
    return {'pooled_auc':pointp,'pooled_auc_ci95':interval(bp),'within_person_auc':pointw,'within_person_auc_ci95':interval(bw),
            'within_auc_eligible_people':len(vals),'pooled_minus_within':pointp-pointw,'pooled_minus_within_ci95':interval(bg)}
def binary_threshold(y,p,thr,pid=None):
    y=np.asarray(y,int); h=np.asarray(p)>=thr
    out={'threshold':float(thr),'pooled_sensitivity':float(h[y==1].mean()) if (y==1).any() else None,
         'pooled_specificity':float((~h[y==0]).mean()) if (y==0).any() else None}
    if pid is not None:
        se=[]; sp=[]
        for _,ix in groups(pid):
            if y[ix].sum()>0: se.append(float(h[ix][y[ix]==1].mean()))
            if y[ix].sum()<len(ix): sp.append(float((~h[ix][y[ix]==0]).mean()))
        out.update({'sensitivity_person_mean':float(np.mean(se)) if se else None,'sensitivity_eligible_people':len(se),
                    'specificity_person_mean':float(np.mean(sp)) if sp else None,'specificity_eligible_people':len(sp)})
    return out
def boot_pair_diff(y,pid,p,ref,metric='brier',nboot=NBOOT,seed=SEED+20):
    grp=dict(groups(pid)); rng=np.random.default_rng(seed); vals=[]
    def m(ix):
        if metric=='brier': return brier(y[ix],p[ix])-brier(y[ix],ref[ix])
        if metric=='skill_ratio': return 1-np.sum((p[ix]-y[ix])**2)/np.sum((ref[ix]-y[ix])**2)
        return logloss(y[ix],p[ix])-logloss(y[ix],ref[ix])
    keys=np.array(list(grp),dtype=object)
    for _ in range(nboot):
        sel=rng.choice(keys,len(keys),replace=True)
        if metric=='person_brier':
            vals.append(float(np.mean([brier(y[grp[k]],p[grp[k]])-brier(y[grp[k]],ref[grp[k]]) for k in sel])))
        else:
            ix=np.concatenate([grp[k] for k in sel]); vals.append(m(ix))
    return interval(vals)
def skills(y,p,pid,refs):
    out={}
    for name,r in refs.items():
        point=1-np.sum((p-y)**2)/np.sum((r-y)**2)
        d={k:(np.sum((p[ix]-y[ix])**2),np.sum((r[ix]-y[ix])**2)) for k,ix in groups(pid)}
        rng=np.random.default_rng(SEED+130); ks=np.array(list(d),dtype=object); draws=[]
        for _ in range(NBOOT):
            sel=rng.choice(ks,len(ks),replace=True); a=np.array([d[k] for k in sel]); draws.append(1-a[:,0].sum()/a[:,1].sum())
        pdiff=float(np.mean([brier(y[ix],p[ix])-brier(y[ix],r[ix]) for _,ix in groups(pid)]))
        out[name]={'summed_loss_skill':float(point),'summed_loss_skill_ci95':interval(draws),
          'person_mean_brier_difference_model_minus_reference':pdiff,
          'person_mean_brier_difference_ci95':boot_pair_diff(y,pid,p,r,'person_brier',seed=SEED+131)}
    return out
def benchmark(y,pid,rate):
    n=np.array([len(ix) for _,ix in groups(pid)]); q=np.array([y[ix].mean() for _,ix in groups(pid)]); N=n.sum(); mu=y.mean()
    # Day-weighted SD and correction for finite within-person binomial sampling.
    sd=np.sqrt(np.average((q-mu)**2,weights=n)); sampling=np.average(q*(1-q)/(n-1),weights=n)
    corrected=np.sqrt(max(sd*sd-sampling,0))
    a0=auc(y,rate)
    return {'retrospective_exact_pooled_auc':a0,'runin_exact_pooled_auc':None,'expanding_exact_pooled_auc':None,
      'normal_approximation':{'mean':float(mu),'observed_day_weighted_sd':float(sd),'sampling_component_variance':float(sampling),
       'sampling_corrected_sd':float(corrected),'auc_observed_sd':float(.5*(1+math.erf(sd/(math.sqrt(2)*mu*(1-mu))/math.sqrt(2)))) if 0<mu<1 else None,
       'auc_sampling_corrected_sd':float(.5*(1+math.erf(corrected/(math.sqrt(2)*mu*(1-mu))/math.sqrt(2)))) if 0<mu<1 else None},
      'within_person_pair_share':float(sum(x*(x-1) for x in n)/(N*(N-1)))}
def ap_tie_expectation(y,s):
    y=np.asarray(y,int); s=np.asarray(s,float); T=len(y); n=int(y.sum())
    if T==0 or n==0: return np.nan
    _,inv,c=np.unique(s,return_inverse=True,return_counts=True); order=np.argsort(-np.unique(s)); sizes=c[order]; M=0.; a=0.; b=0.
    for m in sizes:
        Mnew=M+m; a+=m/Mnew; b+=m*(M+m-1)/Mnew; M=Mnew
    return float(a/T + ((n-1)/(T*(T-1))*b if T>1 else 0))
def average_precision_grouped(y,s):
    # sklearn-compatible distinct-threshold AP, ties enter as one group.
    y=np.asarray(y,int); s=np.asarray(s,float); pos=int(y.sum())
    if not pos: return np.nan
    order=np.argsort(-s,kind='mergesort'); yy=y[order]; ss=s[order]
    ends=np.r_[np.flatnonzero(np.diff(ss)!=0),len(ss)-1]; tp=np.cumsum(yy)[ends]; fp=1+ends-tp
    recall=tp/pos; precision=tp/(tp+fp)
    return float(np.sum(np.diff(np.r_[0,recall])*precision))
def average_precision_summary(y,s,pid):
    aps=[]; prev=[]; diff=[]
    for _,ix in groups(pid):
        if y[ix].sum()==0: continue
        ap=average_precision_grouped(y[ix],s[ix]); ex=ap_tie_expectation(y[ix],s[ix]); aps.append(ap); prev.append(y[ix].mean()); diff.append(ap-ex)
    return {'people_with_events':len(aps),'mean_average_precision':float(np.mean(aps)) if aps else None,
            'mean_prevalence':float(np.mean(prev)) if prev else None,'mean_tie_matched_permutation_expectation':float(np.mean(np.array(aps)-np.array(diff))) if aps else None,
            'mean_difference_from_tie_matched_expectation':float(np.mean(diff)) if diff else None}
def expected_optimism(y,pid):
    rows=[]
    for _,ix in groups(pid):
        q=y[ix].mean(); n=len(ix); rows.append((q,n))
    return float(sum(q*(1-q)*n/(n-1) for q,n in rows)/sum(n for _,n in rows))
def permute_scored_reports(ev, totals, rng):
    """Fix report 21 (forecast origin); permute only report 22 onward."""
    ev=np.asarray(ev,int); totals=np.asarray(totals,float)
    assert len(ev)==len(totals) and len(ev)>=2
    order=np.r_[0, rng.permutation(np.arange(1,len(ev)))]
    return ev[order], totals[order]

def permutation_bias(panel,metric_names,nperm=NPERM):
    y=panel.y.to_numpy(int); pid=panel.pid.to_numpy(); observed={k:patient_auc_values(y,panel[k].to_numpy(float),pid) for k in metric_names}
    refs={k:[] for k in metric_names}; rng=np.random.default_rng(SEED+50)
    by={p:panel.loc[panel.pid==p].sort_values('date') for p in np.unique(pid)}
    for _ in range(nperm):
        for p,g in by.items():
            # Keep run-in and report 21 fixed; permute scored reports 22 onward,
            # then rebuild next-report labels and all history scores in that order.
            ev=np.r_[g.current_event.to_numpy(int),int(g.y.iloc[-1])]
            totals=np.r_[g.persistence.to_numpy(float),float(g.next_persistence.iloc[-1])]
            runin=np.asarray(g.runin_event_history.iloc[0],int)
            evp,tp=permute_scored_reports(ev,totals,rng)
            prior=np.r_[runin,evp]
            yy=evp[1:]; curr=evp[:-1]; currtot=tp[:-1]
            scores={'persistence':currtot}
            ex=[]; recent=[]
            for j in range(len(curr)):
                history=np.r_[runin,evp[:j+1]]
                ex.append(history.mean()); recent.append(history[-12:].mean())
            scores['expanding_rate']=np.asarray(ex); scores['last12_rate']=np.asarray(recent)
            for k in metric_names:
                aa=auc(yy,scores[k])
                if np.isfinite(aa): refs[k].append((p,aa))
    out={}
    for k in metric_names:
        common=[p for p in observed[k] if any(pp==p for pp,_ in refs[k])]
        # permutation expected within AUC averaged across permutation x eligible patients
        vals=np.array([a for _,a in refs[k]])
        obs=np.mean([observed[k][p] for p in common]) if common else np.nan
        ref=float(vals.mean()) if len(vals) else np.nan
        diffs=[]
        for p in common:
            z=[a for pp,a in refs[k] if pp==p]
            if z: diffs.append((observed[k][p],float(np.mean(z))))
        d=np.array([a-b for a,b in diffs]); rng2=np.random.default_rng(SEED+51)
        boots=[np.mean(rng2.choice(d,len(d),replace=True)) for _ in range(NBOOT)] if len(d) else []
        out[k]={'observed_within_auc':float(obs),'permutation_reference_within_auc':ref,'observed_minus_reference':float(d.mean()) if len(d) else None,
                'difference_ci95':interval(boots),'eligible_people':len(common),'permutations':nperm}
    return out

def calibration_model(y,p,pid,nboot=NCAL,seed=SEED+60):
    lp=logit(p); means=pd.Series(lp).groupby(pid,sort=False).transform('mean').to_numpy(); dev=lp-means
    result={}; rng=np.random.default_rng(seed); gr=dict(groups(pid)); keys=np.array(list(gr),dtype=object)
    for equal,name in [(False,'day_weighted'),(True,'equal_person_weighted')]:
        w=equal_weights(pid) if equal else None
        pooled=irls(lp[:,None],y,w); joint=irls(np.column_stack([means,dev]),y,w)
        point={'intercept':pooled[0],'slope':pooled[1],'between_intercept':joint[0],'between_slope':joint[1],'within_slope':joint[2]}
        bs=[]
        for _ in range(nboot):
            sel=rng.choice(keys,len(keys),replace=True); ix=np.concatenate([gr[k] for k in sel]); bp=np.concatenate([np.full(len(gr[k]),j) for j,k in enumerate(sel)])
            try:
                l=lp[ix]; m=pd.Series(l).groupby(bp).transform('mean').to_numpy(); dv=l-m; ww=equal_weights(bp) if equal else None
                po=irls(l[:,None],y[ix],ww); jo=irls(np.column_stack([m,dv]),y[ix],ww); bs.append([po[0],po[1],jo[0],jo[1],jo[2]])
            except (ValueError,np.linalg.LinAlgError,RuntimeError): bs.append([np.nan]*5)
        labs=list(point); result[name]={**point,'ci95':{lab:interval(np.asarray(bs)[:,i]) for i,lab in enumerate(labs)}}
    result['variance_between_share']=float(np.var(means)/np.var(lp)) if np.var(lp)>0 else None
    result['variance_within_share']=float(np.mean(dev**2)/np.var(lp)) if np.var(lp)>0 else None
    return result

def render_md(res):
    def table(title,headers,rows):
        s=['## '+title,'','| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
        s += ['| '+' | '.join(str(x) for x in r)+' |' for r in rows]; return s+['']
    c=res.get('1_cohort',{}); rank=res.get('4_ranking',{}); scores=res.get('6_scores',{}); cal=res.get('7_calibration',{}); hist=res.get('8_history_bias',{})
    lines=['# Nepal openESM analysis results','']
    lines+=table('Cohort',['Quantity','Value'],[[k,v] for k,v in c.items() if not isinstance(v,(dict,list))])
    rr=[[k,v.get('pooled_auc'),v.get('within_person_auc'),v.get('within_auc_eligible_people'),v.get('pooled_minus_within')] for k,v in rank.get('test',{}).get('forecasts',{}).items()]
    bm=rank.get('test',{}).get('benchmark',{})
    rr += [[k,v,None,None,None] for k,v in [('retrospective exact AUC',bm.get('retrospective_exact_pooled_auc')),('run-in exact AUC',bm.get('runin_exact_pooled_auc')),('expanding exact AUC',bm.get('expanding_exact_pooled_auc')),('normal approx observed SD',bm.get('normal_approximation',{}).get('auc_observed_sd')),('normal approx corrected SD',bm.get('normal_approximation',{}).get('auc_sampling_corrected_sd'))]]
    lines+=table('Ranking with benchmark',['Forecast / reference','Pooled AUC','Within AUC','Eligible people','Pooled minus within'],rr)
    sr=[]
    for k,v in scores.get('test',{}).get('forecasts',{}).items():
        skill=v.get('brier_skill_vs_references',{})
        summary='; '.join(f"{ref}: skill {z.get('summed_loss_skill'):.3f}, person-mean Brier diff {z.get('person_mean_brier_difference_model_minus_reference'):.3f}" for ref,z in skill.items()) if skill else ''
        sr.append([k,v.get('brier_pooled'),v.get('brier_person_mean'),v.get('logloss_pooled'),v.get('logloss_person_mean'),summary])
    lines+=table('Scores and skill',['Forecast','Brier pooled','Brier person mean','Log loss pooled','Log loss person mean','Skill vs references (summed-loss skill; person-mean Brier difference)'],sr)
    cr=[]
    for m,z in cal.get('models',{}).items():
        for weighting in ('day_weighted','equal_person_weighted'):
            a=z.get(weighting,{})
            if a: cr.append([m+' '+weighting,'pooled slope',a.get('slope'),a.get('ci95',{}).get('slope')]); cr.append([m+' '+weighting,'between / within slope',f"{a.get('between_slope')} / {a.get('within_slope')}",f"{a.get('ci95',{}).get('between_slope')} / {a.get('ci95',{}).get('within_slope')}"])
        cr.append([m,'between / within logit variance share',f"{z.get('variance_between_share')} / {z.get('variance_within_share')}",''])
    for cohort,zs in hist.items():
        for m,z in zs.items():
            cr.append([cohort+' '+m,'observed AUC',z.get('observed_within_auc'),''])
            cr.append([cohort+' '+m,'permutation reference AUC',z.get('permutation_reference_within_auc'),''])
            cr.append([cohort+' '+m,'observed - permutation AUC',z.get('observed_minus_reference'),z.get('difference_ci95')])
    lines+=table('Calibration and history bias',['Analysis','Quantity','Estimate','95% interval'],cr)
    return '\n'.join(lines)

def run_analysis(args):
    results={}; reports,panel=load_reports(args.data,args.cutoff); split=split_people(panel); panel['split']=panel.pid.map(split)
    test=panel[panel.split=='test'].copy().reset_index(drop=True); train=panel[panel.split=='train'].copy(); val=panel[panel.split=='validation'].copy()
    y=test.y.to_numpy(int); pid=test.pid.to_numpy(); tr_rate=float(train.y.mean()); threshold=tr_rate
    print(f"Included people={panel.pid.nunique()} scored={len(panel)}; test people={test.pid.nunique()} forecasts={len(test)}")
    def block(n,fn):
        try: results[n]=fn(); print(f'[{n}] complete')
        except Exception as e: results[n]={'error':f'{type(e).__name__}: {e}'}; print(f'[{n}] FAILED: {type(e).__name__}: {e}')
    block('1_cohort',lambda:cohort_summary(reports,panel))
    block('2_split',lambda:{'seed':SEED,'train_people':int(train.pid.nunique()),'validation_people':int(val.pid.nunique()),'test_people':int(test.pid.nunique()),'training_event_rate':tr_rate,'threshold':threshold})
    # Models are separate block so model-free summaries still run without sklearn.
    def fit_models():
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.ensemble import GradientBoostingClassifier
        cols=ITEMS+['gap_days','last4_rate','last12_rate','expanding_rate','runin_rate']
        xtr,med=make_features(train,cols); xv,_=make_features(val,cols,med); xt,_=make_features(test,cols,med)
        lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,random_state=SEED)).fit(xtr,train.y)
        plr=lr.predict_proba(xt)[:,1]
        plrv=lr.predict_proba(xv)[:,1]
        cand=[50,100,150,200,300,400,600]
        gb0=GradientBoostingClassifier(n_estimators=max(cand),learning_rate=.05,max_depth=2,min_samples_leaf=20,random_state=SEED).fit(xtr,train.y)
        stages=list(gb0.staged_predict_proba(xv)); losses=[logloss(val.y.to_numpy(int),s[:,1]) for s in stages]
        best=cand[int(np.argmin([losses[i-1] for i in cand]))]
        gb=GradientBoostingClassifier(n_estimators=best,learning_rate=.05,max_depth=2,min_samples_leaf=20,random_state=SEED).fit(xtr,train.y)
        pg=gb.predict_proba(xt)[:,1]
        pgv=gb.predict_proba(xv)[:,1]
        cw=[c+'_centered' for c in CURR]
        ctr,_=make_features(train,cw); cte,_=make_features(test,cw,ctr.median())
        gbw=GradientBoostingClassifier(n_estimators=best,learning_rate=.05,max_depth=2,min_samples_leaf=20,random_state=SEED).fit(ctr,train.y)
        pw=gbw.predict_proba(cte)[:,1]
        cva,_=make_features(val,cw,ctr.median())
        pwv=gbw.predict_proba(cva)[:,1]
        return {'logistic_regression':plr,'gradient_boosting':pg,'within_centred_gradient_boosting':pw,
                'validation_predictions':{'logistic_regression':plrv,'gradient_boosting':pgv,'within_centred_gradient_boosting':pwv},
                'gbm_iterations':best,'gbm_validation_logloss':losses[best-1]}
    try:
        model=fit_models(); results['3_models']={'gbm_iterations':model['gbm_iterations'],'gbm_validation_logloss':model['gbm_validation_logloss']}; print('[3_models] complete')
    except Exception as e:
        model={'error':f'{type(e).__name__}: {e}'}; results['3_models']=model; print(f"[3_models] FAILED: {type(e).__name__}: {e}")
    model_ok='error' not in model
    # All included and test model-free rankings, thresholds, scores, history references.
    def baselines(df):
        yy=df.y.to_numpy(int); pp=df.pid.to_numpy()
        rate_tr=float(train.y.mean())
        return {'training_cohort_rate':np.full(len(df),rate_tr),'runin_rate':df.runin_rate.to_numpy(float),
          'runin_rate_shrunk_10':(20*df.runin_rate.to_numpy(float)+10*rate_tr)/30,
          'expanding_rate':df.expanding_rate.to_numpy(float),'last12_rate':df.last12_rate.to_numpy(float),
          'persistence':df.persistence.to_numpy(float),'retrospective_rate':df.retro_rate.to_numpy(float)}
    allbase=baselines(panel); testbase=baselines(test)
    export_preds=dict(testbase)
    if model_ok:
        export_preds.update({k:model[k] for k in ('logistic_regression','gradient_boosting','within_centred_gradient_boosting')})
    export_ids=pd.unique(test.pid)
    export_frame=pd.DataFrame({'patient':test.pid.map({value:i for i,value in enumerate(export_ids)}).astype(int).to_numpy(),
        't':test.groupby('pid',sort=False).cumcount().to_numpy(),'y':test.y.to_numpy(int)})
    for name,values in export_preds.items(): export_frame[name]=np.asarray(values,float)
    export_frame['threshold']=threshold
    (ROOT/'forecasts_ces.csv').parent.mkdir(parents=True,exist_ok=True)
    export_frame.to_csv(ROOT/'forecasts_ces.csv',index=False)
    validation_base=baselines(val)
    validation_preds=dict(validation_base)
    if model_ok:
        validation_preds.update(model.get('validation_predictions',{}))
    validation_frame=pd.DataFrame({'patient':val.pid.map({value:i for i,value in enumerate(pd.unique(val.pid))}).astype(int).to_numpy(),
        't':val.groupby('pid',sort=False).cumcount().to_numpy(),'y':val.y.to_numpy(int)})
    for name,values in validation_preds.items(): validation_frame[name]=np.asarray(values,float)
    validation_frame['threshold']=threshold
    validation_frame.to_csv(ROOT/'forecasts_val_ces.csv',index=False)
    all_metrics={'y':panel.y.to_numpy(int),'pid':panel.pid.to_numpy()}
    def rankings(df, base, include_models=False):
        yy=df.y.to_numpy(int); pp=df.pid.to_numpy(); fore={}
        for k,s in base.items(): fore[k]=auc_report(yy,np.asarray(s,float),pp)
        if include_models and model_ok:
            for k in ('logistic_regression','gradient_boosting','within_centred_gradient_boosting'):
                fore[k]=auc_report(yy,model[k],pp)
        return fore
    def rank_block():
        out={'test':{'forecasts':rankings(test,testbase,True)},'all_included_model_free':{'forecasts':rankings(panel,allbase,False)}}
        for cohort,df,bs in [('test',test,testbase),('all_included_model_free',panel,allbase)]:
            yy=df.y.to_numpy(int); pp=df.pid.to_numpy()
            out[cohort]['benchmark']=benchmark(yy,pp,bs['retrospective_rate'])
            out[cohort]['benchmark']['runin_exact_pooled_auc']=auc(yy,bs['runin_rate'])
            out[cohort]['benchmark']['expanding_exact_pooled_auc']=auc(yy,bs['expanding_rate'])
        return out
    block('4_ranking',rank_block)
    def threshold_block():
        out={}
        for name,df,bs in [('test',test,testbase),('all_included_model_free',panel,allbase)]:
            yy=df.y.to_numpy(int); pp=df.pid.to_numpy(); vals={k:binary_threshold(yy,v,threshold,pp) for k,v in bs.items() if k in ['runin_rate','expanding_rate','last12_rate','persistence']}
            if name=='test' and model_ok:
                vals.update({k:binary_threshold(yy,model[k],threshold,pp) for k in ['logistic_regression','gradient_boosting','within_centred_gradient_boosting']})
            out[name]=vals
        return out
    block('5_thresholds',threshold_block)
    def score_block():
        out={}
        for name,df,bs in [('test',test,testbase),('all_included_model_free',panel,allbase)]:
            yy=df.y.to_numpy(int); pp=df.pid.to_numpy(); forecasts={}
            vals=dict(bs)
            if name=='test' and model_ok: vals.update({k:model[k] for k in model if k in ['logistic_regression','gradient_boosting','within_centred_gradient_boosting']})
            refs={'training_cohort_rate':bs['training_cohort_rate'],'runin_rate':bs['runin_rate'],'runin_rate_shrunk_10':bs['runin_rate_shrunk_10'],'expanding_rate':bs['expanding_rate'],'last12_rate':bs['last12_rate'],'persistence_total_div_12':np.clip(bs['persistence']/12,1e-6,1-1e-6),'retrospective_rate':bs['retrospective_rate']}
            for k,p in vals.items():
                if k=='persistence': k='persistence_total_div_12'; p=np.clip(p/12,1e-6,1-1e-6)
                forecasts[k]=eval_scores(yy,p,pp)
                if k in ['logistic_regression','gradient_boosting','within_centred_gradient_boosting']:
                    forecasts[k]['brier_skill_vs_references']=skills(yy,p,pp,refs)
            out[name]={'forecasts':forecasts,'day_weighted_retrospective_optimism':expected_optimism(yy,pp)}
        return out
    block('6_scores',score_block)
    def calibration_block():
        if not model_ok: raise RuntimeError('scikit-learn model forecasts unavailable')
        return {'models':{k:calibration_model(y,model[k],pid) for k in ['logistic_regression','gradient_boosting']}}
    block('7_calibration',calibration_block)
    block('8_history_bias',lambda:{'test':permutation_bias(test,['last12_rate','expanding_rate','persistence']),'all_included':permutation_bias(panel,['last12_rate','expanding_rate','persistence'])})
    def ap_block():
        out={}
        for cohort,df in [('test',test),('all_included',panel)]:
            yy=df.y.to_numpy(int); pp=df.pid.to_numpy(); out[cohort]={}
            choices={'runin_rate':df.runin_rate.to_numpy(float)}
            if cohort=='test' and model_ok: choices['gradient_boosting']=model['gradient_boosting']
            for k,s in choices.items(): out[cohort][k]=average_precision_summary(yy,s,pp)
        return out
    block('9_average_precision',ap_block)
    results['run_metadata']={'data_file':'ces_openesm/0040_nepal_ts.tsv','event_cutoff':args.cutoff,'sensitivity_cutoff':6,'seed':SEED,'bootstraps_auc':NBOOT,'bootstraps_calibration':NCAL,'permutations':NPERM,'persistence_probability_for_brier_logloss':'current PHQ-4 total divided by 12, clipped to [1e-6, 1-1e-6]; ranking/threshold score uses raw total'}
    # Sensitivity endpoint prevalence and cohort sizes are always computed without model fitting.
    try:
        rep6,_=load_reports(args.data,6); results['sensitivity_event_cutoff_6']={'complete_reports':len(rep6),'event_rate_all_complete_reports':float(rep6.event.mean()),'people_with_ge60_reports':int((rep6.groupby('id').size()>=60).sum())}
    except Exception as e: results['sensitivity_event_cutoff_6']={'error':f'{type(e).__name__}: {e}'}
    OUT_JSON.write_text(json.dumps(safe(results),indent=2,allow_nan=False),encoding='utf-8')
    OUT_MD.write_text(render_md(safe(results)),encoding='utf-8')
    print('Wrote ces_results.json and ces_results.md')
    return results

def selftest():
    y=np.array([0,1,0,1,1,0,0,1]); s=np.array([.1,.8,.2,.7,.6,.4,.3,.9]); pid=np.repeat([0,1],4)
    assert abs(auc(y,s)-1)<1e-12 and abs(within_auc(y,s,pid)-1)<1e-12
    # Exact tied-score AP permutation expectation, and IRLS recovery.
    assert abs(ap_tie_expectation(np.array([1,0,1,0]),np.array([2,2,1,1]))-7/12)<1e-12
    assert abs(average_precision_grouped(np.array([1,0,1,0]),np.array([2,2,1,1]))-.5)<1e-12
    rng=np.random.default_rng(8); x=rng.normal(size=3000); pp=expit(-1+.8*x); yy=rng.binomial(1,pp); b=irls(x[:,None],yy)
    assert abs(b[1]-.8)<.15
    assert auc_report(y,s,pid,nboot=20)['within_auc_eligible_people']==2
    print('SELFTEST PASS: AUC/ties, person summaries, grouped AP/tie expectation, IRLS')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default=str(DATA)); ap.add_argument('--cutoff',type=int,default=3); ap.add_argument('--selftest',action='store_true'); a=ap.parse_args()
    if a.selftest: selftest(); return
    run_analysis(a)
if __name__=='__main__': main()

