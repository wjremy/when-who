from pathlib import Path
import importlib.util,json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
R=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("base",R/"figures/v20/draw_figures.py")
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.OUT=Path(__file__).resolve().parent
d=base.read('results/v19/F9_patients.csv')
names=['CTN-0094','REPLACE-BG','College Experience Study']
assert [len(d[d.cohort==name]) for name in names]==[154,42,37]
fig=plt.figure(figsize=(7.2,6.4))
gs=fig.add_gridspec(2,3,height_ratios=[1,.98],hspace=.65,wspace=.23)
fig.subplots_adjust(left=.13,right=.98,top=.80,bottom=.14)
a=fig.add_subplot(gs[0,:])
a.set_title('a  Mean score and matched baseline',loc='left',fontweight='bold',pad=56)
a.axvline(.5,ls='--',lw=1,color=base.C['muted'])
summary=[]
for i,name in enumerate(names):
    q=d[d.cohort==name];x=float(q.shuffled_baseline.mean());y=float(q.auc.mean());row=2-i
    summary.append({'cohort':name,'n':len(q),'score':y,'baseline':x,'excess':y-x})
    a.plot([x,y],[row,row],color=base.C['corrected'],lw=3,zorder=1)
    a.scatter(x,row,color=base.C['patient'],marker='o',s=38,zorder=3)
    a.scatter(y,row,color=base.C['pooled'],marker='o',s=40,zorder=3)
    a.text(x-.014,row-.16,f'{x:.2f}',ha='right',va='top',fontsize=9)
    a.text(y+.014,row-.16,f'{y:.2f}',ha='left',va='top',fontsize=9)
    a.text((x+y)/2,row+.17,f'Excess +{y-x:.2f}',ha='center',fontsize=9,fontweight='bold',color=base.C['corrected'])
a.set(xlim=(0,1),ylim=(-.47,2.5),yticks=[2,1,0],yticklabels=['CTN-0094','REPLACE-BG','College Experience'],xticks=[0,.25,.5,.75,1],xlabel='Mean per-patient AUC')
a.spines['left'].set_visible(False);a.tick_params(axis='y',length=0,pad=8)
a.grid(axis='x',color=base.C['grid'],lw=.5)
a.legend(handles=[Line2D([],[],marker='o',ls='',color=base.C['patient'],label='Rebuilt shuffled\nbaseline'),Line2D([],[],marker='o',ls='',color=base.C['pooled'],label='Observed score'),Line2D([],[],ls='--',color=base.C['muted'],label='Conventional\n0.5 reference')],loc='lower left',bbox_to_anchor=(0,1.02),ncol=3,fontsize=10,frameon=False)
for i,name in enumerate(names):
    ax=fig.add_subplot(gs[1,i]);q=d[d.cohort==name];up=q.auc>q.shuffled_baseline
    ax.add_patch(Polygon([[0,0],[0,.5],[.5,.5]],facecolor='#FAEAD4',edgecolor='none',zorder=0))
    ax.plot([0,1],[0,1],color=base.C['ink'],lw=1)
    ax.axhline(.5,ls='--',lw=.8,color=base.C['muted'])
    for yes in [True,False]:
        z=q[up==yes];ax.scatter(z.shuffled_baseline,z.auc,s=16,facecolors=base.C['pooled'] if yes else 'white',edgecolors=base.C['pooled'],lw=.7,alpha=.85,zorder=3)
    label='College Experience' if i==2 else name
    ax.set_title(f'{chr(98+i)}  {label}\n{len(q)} patients',loc='left',fontweight='bold',fontsize=9,pad=9)
    ax.set(xlim=(-.02,1.02),ylim=(-.02,1.02),xticks=[0,.5,1],yticks=[0,.5,1]);ax.set_aspect('equal');ax.grid(color=base.C['grid'],lw=.5)
    if i==0:ax.set_ylabel('Observed patient AUC')
fig.text(.55,.08,"Patient’s rebuilt shuffled baseline (AUC)",ha='center',fontsize=9)
fig.text(.55,.005,'Filled: positive excess   Open: negative excess\nShaded: below 0.5 but above own baseline',ha='center',fontsize=10,linespacing=1.35)
base.save(fig,'F5')
(base.OUT/'mean_auc_numbers.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(base.OUT/'source_hashes.json').write_text(json.dumps(base.used,indent=2),encoding='utf-8')
print(json.dumps(summary))
