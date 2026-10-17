"""Redraw manuscript visuals from saved public/simulated results; no model reruns."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch,Rectangle

ROOT=Path(r'C:\Users\wesre\Documents\Claude\pooled-metrics-paper')
HERE=Path(__file__).resolve().parent
OUT=HERE;OUT.mkdir(exist_ok=True)
C={'pooled':'#0072B2','patient':'#AD7700','corrected':'#A64E8A','ink':'#252525','muted':'#646B70','grid':'#E2E6E8','green':'#007B60','orange':'#C44900'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':10.5,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#ADB4B8','axes.labelcolor':C['ink'],'text.color':C['ink']})
used={}
def read(rel):
    p=ROOT/rel;used[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p)
def js(rel):
    p=ROOT/rel;used[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text(encoding='utf-8'))
def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.12)
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight',pad_inches=.12,dpi=180)
    plt.close(fig)
def tidy(ax):
    ax.set_axisbelow(True);ax.grid(axis='y',color=C['grid'],lw=.6)
def title(ax,s):ax.set_title(s,loc='left',fontweight='bold',pad=12)

def f1():
    d=read('results/v19/F1_numbers.csv')
    g=d[d.record_type=='grid_day'].copy();g['patient']=pd.to_numeric(g.patient);g['day']=pd.to_numeric(g.day)
    s=d[(d.panel=='b')&(d.record_type=='summary')].set_index('quantity').value.astype(float)
    fig,(a,b)=plt.subplots(2,1,figsize=(7.2,6.5),gridspec_kw={'height_ratios':[1.5,1]},layout='constrained')
    title(a,'a  Compare days within a patient; pooling also compares patients')
    for val in [0,1]:
        q=g[g.outcome==val];a.scatter(q.day,q.patient,s=32,facecolors=C['ink'] if val else 'white',edgecolors=C['ink'] if val else '#BAC2C6',lw=.7)
    a.add_patch(Rectangle((.5,2.58),21,.84,fill=False,color=C['pooled'],lw=1.5))
    a.add_patch(Rectangle((9.58,.5),.84,8,fill=False,color=C['orange'],lw=1.5))
    a.set(xlim=(.4,21.6),ylim=(8.7,.3),xticks=[1,5,10,15,21],yticks=range(1,9),ylabel='Patient',xlabel='Day')
    a.text(.01,-.26,'Blue row: within-patient days     Orange column: one between-patient example',transform=a.transAxes,fontsize=9)
    a.text(.01,-.37,f'In this example, {s["same_patient_pair_share"]:.1%} of pooled AUC pairs are within patients.',transform=a.transAxes,fontsize=9,fontweight='bold')
    title(b,'b  Constant forecasts can still give a high pooled AUC')
    for y,name,events,rate,col in [(1,'Ben',{1,3,4,7,9},.5,C['pooled']),(0,'Anna',{4},.1,C['orange'])]:
        x=np.arange(1,11);b.plot([1,10],[y,y],color=col,lw=1)
        for event in [0,1]:
            xx=[v for v in x if (v in events)==bool(event)]
            b.scatter(xx,np.full(len(xx),y),s=55,facecolors=col if event else 'white',edgecolors=col,lw=1.1,zorder=3)
        b.text(10.4,y,f'Every forecast: {rate:.0%}',va='center',fontsize=10)
    b.set(xlim=(.4,15),ylim=(-.6,1.7),yticks=[0,1],yticklabels=['Anna','Ben'],xticks=range(1,11),xlabel='Day (filled circle = event)')
    b.text(.01,-.34,'Pooled AUC = 0.74         AUC within each patient = 0.50',transform=b.transAxes,fontweight='bold')
    b.spines[['left','bottom']].set_visible(False);b.tick_params(length=0)
    save(fig,'F1')

def f2():
    d=read('results/v19/F2_numbers.csv');q=d[d.panel=='c'];curve=read('results/v13/A2b_flip_curve.csv').sort_values('days_scored')
    fig,axs=plt.subplots(2,2,figsize=(7.2,6.2),layout='constrained',gridspec_kw={'hspace':.16,'wspace':.1})
    a,b,c,e=axs.flat
    title(a,'a  Which outcomes enter the forecast?');a.axis('off')
    for yy,label,txt,col in [(.78,'Prospective rule','Earlier days only',C['pooled']),(.45,'Thought experiment','Other days, including future days',C['orange']),(.12,'Whole-record rate','All days, including the scored day',C['muted'])]:
        a.text(0,yy,label,fontweight='bold',color=col,transform=a.transAxes)
        a.text(0,yy-.12,txt,fontsize=9,transform=a.transAxes)
    title(b,'b  Other-nine-days thought experiment')
    b.scatter([2/9,3/9],[1,0],s=65,color=[C['orange'],C['pooled']]);b.set(yticks=[0,1],yticklabels=['Event-free days','Event days'],xlim=(.1,.5),ylim=(-.6,1.5),xlabel='Forecast probability')
    b.text(.97,.95,'AUC = 0',ha='right',transform=b.transAxes,fontweight='bold',va='top');b.text(2/9,1.18,'2/9',ha='center');b.text(3/9,.18,'3/9',ha='center');b.grid(axis='x',color=C['grid'])
    title(c,'c  Prospective rate so far')
    c.plot(q.day,q.rate_so_far,color=C['muted'],lw=1,zorder=1)
    for event in [0,1]:
        z=q[q.outcome==event];c.scatter(z.day,z.rate_so_far,s=45,facecolors=C['pooled'] if event else 'white',edgecolors=C['pooled'],lw=1.2,zorder=3,label='Event day' if event else 'Event-free day')
    c.set(xlim=(.5,10.5),ylim=(-.04,.63),xticks=[1,2,5,9,10],xlabel='Day (day 1 unscored)',ylabel='Forecast probability');tidy(c)
    c.text(.98,.95,'AUC = 0',ha='right',va='top',transform=c.transAxes,fontweight='bold');c.legend(loc='upper center',bbox_to_anchor=(.5,-.26),frameon=False,ncol=1)
    title(e,'d  Across independent simulated records')
    e.plot(curve.days_scored,curve.auc_rate_so_far_mean,'o-',color=C['pooled'],label='Rate so far',ms=4)
    e.axhline(.5,color=C['muted'],ls='--');e.set(xscale='log',ylim=(.25,.53),xlabel='Days scored (log scale)',ylabel='Mean per-patient AUC')
    e.set_xticks([10,100,1000,5000],['10','100','1,000','5,000']);tidy(e);e.text(.95,.91,'Usual baseline 0.5',ha='right',transform=e.transAxes,fontsize=9)
    save(fig,'F2')
    fig,a=plt.subplots(figsize=(7.2,3.3),layout='constrained')
    a.errorbar(curve.days_scored,curve.auc_flipped_mean,yerr=curve.auc_flipped_mcse,color=C['orange'],marker='s',label='Flipped rate so far',capsize=2)
    a.plot(curve.days_scored,curve.auc_rate_so_far_mean,'o-',color=C['pooled'],label='Rate so far')
    a.axhline(.5,color=C['muted'],ls='--');a.set(xscale='log',ylim=(.25,.75),xlabel='Days scored (log scale)',ylabel='Mean per-patient AUC')
    a.set_xticks([10,100,1000,5000],['10','100','1,000','5,000']);tidy(a);a.legend(frameon=False)
    save(fig,'S2')

def f3():
    n=read('results/v19/F4_numbers.csv');v=n[n.row_type=='shuffle_auc'].auc.astype(float);summary=n[n.row_type=='summary'].iloc[0]
    assert round(summary.auc_observed,3)==.330 and round(v.mean(),3)==.421
    fig=plt.figure(figsize=(7.2,6.4));gs=fig.add_gridspec(2,1,height_ratios=[1.35,1],hspace=.38);fig.subplots_adjust(left=.12,right=.97,top=.94,bottom=.11)
    a=fig.add_subplot(gs[0]);a.axis('off');title(a,'a  Repeat the original forecasting procedure on shuffled outcomes')
    boxes=[(.02,.62,'1  Shuffle outcomes','Within each patient; preserve the\nchosen patient or block event counts.'),(.53,.62,'2  Rebuild the rule','Recalculate outcome-derived inputs;\nrepeat fitting and reference construction.'),(.53,.08,'3  Score each rebuilt record','Use the same patient-level measure.\nRepeat the shuffle and rebuild.'),(.02,.08,'4  Average, then subtract','Baseline = mean defined shuffled score.\nExcess = observed score − baseline.')]
    for x,y,h,t in boxes:
        a.add_patch(FancyBboxPatch((x,y),.44,.29,boxstyle='round,pad=.015',facecolor='#F3F6F8',edgecolor='#BAC4CA',transform=a.transAxes))
        a.text(x+.018,y+.225,h,fontweight='bold',fontsize=9.8,transform=a.transAxes)
        a.text(x+.018,y+.06,t,fontsize=8.8,linespacing=1.5,transform=a.transAxes)
    for p0,p1 in [((.47,.77),(.52,.77)),((.75,.61),(.75,.40)),((.52,.23),(.47,.23))]:a.annotate('',xy=p1,xytext=p0,xycoords='axes fraction',arrowprops={'arrowstyle':'->','lw':1.4,'color':C['ink']})
    a.text(.5,-.10,'Keep run-in outcomes and independently learned quantities fixed.',ha='center',transform=a.transAxes,fontsize=9,color=C['green'])
    b=fig.add_subplot(gs[1]);title(b,'b  Worked example: the baseline is computed, not assumed')
    b.hist(v,bins=np.linspace(.25,.6,27),color=C['corrected'],alpha=.65,edgecolor='white')
    for x,col,ls,label in [(summary.auc_observed,C['pooled'],'-','Observed 0.330'),(v.mean(),C['corrected'],'-','Shuffled mean 0.421'),(.5,C['muted'],'--','Usual baseline 0.5')]:b.axvline(x,color=col,ls=ls,lw=1.7,label=label)
    b.set(xlim=(.25,.62),xlabel='AUC across 200 rebuilt shuffled records',ylabel='Shuffles');tidy(b)
    b.legend(frameon=False,loc='upper right',fontsize=8.6)
    b.text(.02,.93,'Excess = 0.330 − 0.421 = −0.091',transform=b.transAxes,fontweight='bold',fontsize=9.5,bbox={'facecolor':'white','edgecolor':'none','alpha':.95})
    save(fig,'F3')

def f4():
    d=read('results/v19/F5_numbers.csv');d=d[(d.dataset=='CTN-0094')&d.measure.notna()]
    measures=['auc','average_precision','ppv','npv','youden_j','net_benefit','brier_skill','calibration_slope']
    labels=['AUC','Average precision','PPV','NPV','Sensitivity +\nspecificity − 1','Net benefit','Brier skill','Calibration slope']
    methods=['pooled','per_patient','per_patient_own_rate','shuffled_baseline']
    heads=['Pooled / usual\nreference','Per patient /\nusual reference','Per patient /\nown-rate reference','Per patient /\nshuffled baseline']
    fig,axs=plt.subplots(8,4,figsize=(7.2,8.3));fig.subplots_adjust(left=.23,right=.98,top=.9,bottom=.1,hspace=.7,wspace=.22)
    for row,(metric,label) in enumerate(zip(measures,labels)):
        ds=d[d.measure==metric];vals=pd.to_numeric(ds.excess,errors='coerce');lim=max(.05,float(vals.abs().max())*1.2)
        # Use one raw-score scale across methods within a metric, never across metrics.
        for col,method in enumerate(methods):
            ax=axs[row,col];ax.set(xlim=(-lim,lim),ylim=(-.5,.5),yticks=[],xticks=[-lim/1.2,0,lim/1.2]);ax.set_xticklabels([f'{-lim/1.2:.2g}','0',f'{lim/1.2:.2g}'],fontsize=8)
            if col==3:ax.set_facecolor('#F7EDF4')
            ax.axvline(0,color=C['ink'],lw=1);ax.spines[['left','top','right']].set_visible(False)
            sub=ds[ds.scoring==method]
            if sub.excess.notna().sum()==0:
                ax.text(.5,.5,'n/a',ha='center',va='center',transform=ax.transAxes,color=C['muted']);ax.set_xticks([])
            else:
                for forecast,y,mark in [('shuffled rate so far',.18,'o'),('run-in rate',-.18,'^')]:
                    q=sub[sub.forecast==forecast]
                    if len(q) and pd.notna(q.iloc[0].excess):ax.scatter(q.iloc[0].excess,y,marker=mark,s=35,color=C['corrected'] if col==3 else C['pooled'] if col==0 else C['patient'],zorder=3)
                if metric=='calibration_slope' and col in [1,3]:ax.text(.97,.04,'run-in n/a',ha='right',transform=ax.transAxes,fontsize=7.5,color=C['muted'])
            if row==0:ax.set_title(heads[col],fontweight='bold',fontsize=9,pad=15)
            if col==0:ax.set_ylabel(label,rotation=0,ha='right',va='center',labelpad=8,fontsize=9)
    fig.text(.58,.045,'Score minus reference (scales differ between rows)',ha='center',fontsize=10)
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color=C['ink'],label='Shuffled rate so far'),Line2D([],[],marker='^',ls='',color=C['ink'],label='Run-in rate')],loc='lower center',bbox_to_anchor=(.57,.001),ncol=2,frameon=False)
    save(fig,'F4')

def f5():
    d=read('results/v19/F9_patients.csv');names=['CTN-0094','REPLACE-BG','College Experience Study']
    fig,axs=plt.subplots(1,3,figsize=(7.2,3.4));fig.subplots_adjust(left=.08,right=.99,top=.78,bottom=.21,wspace=.2)
    for i,(a,name) in enumerate(zip(axs,names)):
        q=d[d.cohort==name];up=q.auc>q.shuffled_baseline
        a.plot([0,1],[0,1],color=C['ink'],lw=1);a.axhline(.5,color=C['muted'],ls='--',lw=.9)
        for yes in [True,False]:
            z=q[up==yes];a.scatter(z.shuffled_baseline,z.auc,s=17,facecolors=C['pooled'] if yes else 'white',edgecolors=C['pooled'],lw=.7,alpha=.85)
        x,y=q.shuffled_baseline.mean(),q.auc.mean();a.scatter(x,y,marker='X',color=C['orange'],s=65,zorder=5)
        short='College Experience' if i==2 else name
        a.set_title(f'{chr(97+i)}  {short}\nMean baseline {x:.2f}\nMean score {y:.2f}',loc='left',fontweight='bold',fontsize=9,pad=10)
        a.set(xlim=(-.02,1.02),ylim=(-.02,1.02),xticks=[0,.5,1],yticks=[0,.5,1]);a.set_aspect('equal');a.grid(color=C['grid'],lw=.5)
    axs[0].set_ylabel('Observed per-patient AUC')
    fig.text(.52,.07,"Patient's shuffled baseline (AUC)",ha='center')
    fig.text(.52,.97,'Above diagonal: positive excess     Below diagonal: negative excess',ha='center',fontsize=9)
    save(fig,'F5')

def f6():
    d=read('results/v19/F6_cohorts.csv');b=read('results/v14/B1_goldenholz_pooled.csv');other=b[b.scenario!='primary_midpoint_lengths_spread_counts'];primary=b[b.scenario=='primary_midpoint_lengths_spread_counts'].iloc[0]
    for pooled in [False,True]:
        fig,a=plt.subplots(figsize=(7.2,3.4),layout='constrained')
        if not pooled:
            a.axvspan(other.per_patient_mean.min(),other.per_patient_mean.max(),color='#B8BFC3',alpha=.4,label='Alternative reconstruction means')
            a.hist(d.per_patient_auc,bins=np.linspace(.25,.55,31),color=C['patient'],alpha=.8,edgecolor='white')
            a.axvline(.43,color=C['orange'],lw=1.8,label='Published per-patient AUC 0.43');a.set(xlim=(.23,.55),xlabel='Mean per-patient AUC of the no-skill reconstruction')
        else:
            a.hist(d.pooled_auc,bins=np.linspace(.48,.9,43),color=C['pooled'],alpha=.8,edgecolor='white');a.axvline(primary.a0_mean,color=C['corrected'],lw=1.8,label=f'Patient-constant maximum A₀ {primary.a0_mean:.2f}')
            a.axvline(.82,color=C['orange'],lw=1.8,label='Published deep-learning AUC 0.82');a.set(xlim=(.48,.9),xlabel='Pooled AUC of the no-skill moving-average reconstruction')
        a.axvline(.5,color=C['muted'],ls='--',label='Usual baseline 0.5');a.set_ylabel('Cohorts');tidy(a);a.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.2),ncol=2,fontsize=9)
        a.set_ylim(0,a.get_ylim()[1]*1.08)
        save(fig,'S3' if pooled else 'F6')

def ranking_data():
    data=js('open_data/results/v10/ranking_sim.json')
    rows=[r for r in data['ranking'] if r['pair']=='A daily rho=0.4 vs B daily rho=0.3' and r['patients']==50 and r['metric'] in ['auc','average_precision'] and r['scoring'] in ['pooled','naive','tailored_excess']]
    assert len(rows)==36
    pd.DataFrame(rows).to_csv(OUT/'selection_numbers.csv',index=False)
    return pd.DataFrame(rows)
def f7():
    d=ranking_data();fig,axs=plt.subplots(2,2,figsize=(7.2,6.1),sharex=True,sharey=True);fig.subplots_adjust(left=.12,right=.98,top=.9,bottom=.18,hspace=.3,wspace=.15)
    for row,(metric,label) in enumerate([('auc','AUC'),('average_precision','Average precision')]):
        for col,spread in enumerate(['low','high']):
            a=axs[row,col];q=d[(d.metric==metric)&(d.patient_spread==spread)]
            for method,color,mark,ls,offset in [('pooled',C['pooled'],'s',':',-.04),('naive',C['patient'],'^','--',0),('tailored_excess',C['corrected'],'o','-',.04)]:
                z=q[q.scoring==method].sort_values('record_days');a.plot(np.arange(3)+offset,z.share_a_ranks_above_b,marker=mark,ls=ls,color=color,lw=1.4,ms=5)
            a.axhline(.5,color=C['muted'],ls='--',lw=.7);a.set(ylim=(-.04,1.04),xticks=[0,1,2],xticklabels=['30','100','365'],yticks=[0,.5,1],yticklabels=['0%','50%','100%']);tidy(a)
            title(a,f'{chr(97+row*2+col)}  {label}\n{spread.capitalize()} patient-level spread')
            if col==0:a.set_ylabel('Replications ranking A above B')
            if row==1:a.set_xlabel('Record length (days)')
    fig.legend(handles=[Line2D([],[],marker=m,ls=ls,color=color,label=label) for m,ls,color,label in [('s',':',C['pooled'],'Pooled'),('^','--',C['patient'],'Per patient'),('o','-',C['corrected'],'Corrected excess')]],loc='lower center',bbox_to_anchor=(.53,.035),ncol=3,frameon=False)
    save(fig,'F7')

def f8():
    h=read('results/v19/A3_finite_horizon.csv');d=ranking_data();d=d[d.metric=='auc']
    ghost=read('results/v19/A3_finite_horizon_measures.csv')
    ghost=ghost[(ghost.pair=='A daily rho=0.4 vs B daily rho=0.3')&(ghost.measure=='auc')&(ghost.outcomes=='ghost')]
    assert len(ghost)==12
    records=[]
    for r in h.itertuples():
        sub=d[(d.patient_spread==r.spread)&(d.record_days==r.days)]
        corr=sub[sub.scoring=='tailored_excess'].iloc[0]
        key='mean_a_score' if r.model=='A' else 'mean_b_score'
        truth='long_record_a_true_patient_auc' if r.model=='A' else 'long_record_b_true_patient_auc'
        assert abs(float(corr[key])-r.mean_shuffled_excess)<1e-10
        assert abs(float(corr[truth])-.5-r.long_record_skill)<1e-10
        g=ghost[(ghost.spread==r.spread)&(ghost.days==r.days)&(ghost.model==r.model)].iloc[0]
        rec={'spread':r.spread,'days':r.days,'model':r.model,'Long-record benchmark':r.long_record_skill,'Fresh-outcome benchmark':float(g.value)-.5,'Corrected excess':r.mean_shuffled_excess}
        for method,label in [('pooled','Pooled minus 0.5'),('naive','Per patient minus 0.5')]:rec[label]=float(sub[sub.scoring==method].iloc[0][key])-.5
        records.append(rec)
    out=pd.DataFrame(records);out.to_csv(OUT/'estimated_skill_numbers.csv',index=False)
    fig,axs=plt.subplots(2,2,figsize=(7.2,6),sharex=True,sharey=True);fig.subplots_adjust(left=.11,right=.98,top=.91,bottom=.23,hspace=.3,wspace=.15)
    styles=[('Pooled minus 0.5',C['pooled'],'s',':'),('Per patient minus 0.5',C['patient'],'^','--'),('Corrected excess',C['corrected'],'o','-'),('Long-record benchmark',C['ink'],'x','--'),('Fresh-outcome benchmark',C['green'],'D',':')]
    for i,model in enumerate(['A','B']):
        for j,spread in enumerate(['low','high']):
            a=axs[i,j];q=out[(out.model==model)&(out.spread==spread)].sort_values('days')
            for label,col,marker,ls in styles:a.plot(range(3),q[label],color=col,marker=marker,ls=ls,ms=4,lw=1.1,alpha=.95)
            title(a,f'{chr(97+i*2+j)}  Model {model}: {spread} patient-level spread')
            a.set(ylim=(0,.4),xticks=range(3),xticklabels=['30','100','365']);tidy(a)
            if j==0:a.set_ylabel('AUC excess / benchmark skill')
            if i==1:a.set_xlabel('Record length (days)')
    fig.legend(handles=[Line2D([],[],color=col,marker=mark,ls=ls,label=label) for label,col,mark,ls in styles],loc='lower center',bbox_to_anchor=(.53,.015),ncol=2,frameon=False,columnspacing=1.1)
    save(fig,'F8')

if __name__=='__main__':
    for fun in [f1,f2,f3,f4,f5,f6,f7,f8]:fun()
    (OUT/'input_sha256.json').write_text(json.dumps(used,indent=2),encoding='utf-8')
    print('Rendered F1–F8 and S2 from saved result files; input hashes recorded.')
