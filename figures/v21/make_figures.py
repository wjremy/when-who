from pathlib import Path
import importlib.util,ast,json,hashlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle,Polygon

P=Path(__file__).resolve().parent
R=P.parents[1] if P.parent.name=='figures' else Path(r'C:\Users\wesre\Documents\Claude\pooled-metrics-paper')
spec=importlib.util.spec_from_file_location('visual_base',R/'figures/v20/draw_figures.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.OUT=P if P.parent.name=='figures' else P/'figures';base.OUT.mkdir(exist_ok=True)
source=(R/'figures/v20/draw_figures.py').read_text(encoding='utf-8')
functions={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}

f2=functions['f2']
start=f2.index("    title(a,'a  Which outcomes")
end=f2.index("    title(b,",start)
f2=f2[:start]+'''    title(a,'a  Which outcomes enter the forecast?');a.axis('off')
    a.set(xlim=(-3.8,7.8),ylim=(.05,4.35))
    a.add_patch(Rectangle((3.83,.65),.94,2.95,facecolor='#EDF4F1',edgecolor=C['green'],lw=1,ls='--'))
    a.text(2.3,3.9,'Earlier',ha='center',fontsize=8.5)
    a.text(4.3,3.9,'Day t',ha='center',fontsize=8.5,fontweight='bold',color=C['green'])
    a.text(6.3,3.9,'Later',ha='center',fontsize=8.5)
    for yy,label,selected,col in [(3,'Past only',{1,2,3},C['pooled']),(2,'Other days*',{1,2,3,5,6,7},C['orange']),(1,'Whole record',set(range(1,8)),C['muted'])]:
        a.text(-3.6,yy,label,va='center',fontsize=8.4,fontweight='bold',color=col)
        for xx in range(1,8):
            a.add_patch(Rectangle((xx-.01,yy-.21),.62,.42,facecolor=col if xx in selected else 'white',edgecolor=col if xx in selected else '#C5CDD1',lw=.9))
    a.add_patch(Rectangle((-.05,.28),.4,.27,facecolor=C['ink']))
    a.text(.6,.4,'Outcome used',va='center',fontsize=8.2)
    a.text(-3.6,.08,'*Thought experiment: includes future outcomes',fontsize=7.9)
''' +f2[end:]
f2=f2.replace('b  Other-nine-days thought experiment','b  Other-nine-days rule').replace('c  Prospective rate so far','c  Rate so far (past only)').replace('d  Across independent simulated records','d  Across independent records')
exec(f2,base.__dict__);base.f2()

f4=functions['f4'].replace("figsize=(7.2,8.3)","figsize=(7.2,7.3)")
f4=f4.replace("bottom=.1,hspace=.7","bottom=.14,hspace=.7")
f4=f4.replace("'n/a',ha='center',va='center',transform=ax.transAxes,color=C['muted']","'Not\\napplicable',ha='center',va='center',transform=ax.transAxes,color=C['muted'],fontsize=7.6")
f4=f4.replace("'run-in n/a'","'run-in undefined'")
f4=f4.replace("color=C['muted']);ax.set_xticks([])","color=C['muted']);ax.set_xticks([])")
f4=f4.replace("ax.set_xticks([])","ax.set_xticks([]);[line.set_visible(False) for line in ax.lines]")
f4=f4.replace("fontsize=7.5,color=C['muted'])","fontsize=7,color=C['muted'],bbox={'facecolor':'white','edgecolor':'none','pad':.3})")
f4=f4.replace("fig.text(.58,.045,'Score minus reference (scales differ between rows)',ha='center',fontsize=10)","fig.text(.58,.09,'Forecast score − reference (zero = score equals reference)',ha='center',fontsize=9)")
f4=f4.replace("label='Shuffled rate so far'","label='Shuffled rate-so-far forecast'").replace("label='Run-in rate'","label='Constant run-in forecast'")
f4=f4.replace("bbox_to_anchor=(.57,.001),ncol=2,frameon=False","bbox_to_anchor=(.57,.004),ncol=1,frameon=False,title='Forecast whose score is plotted',fontsize=8.5,title_fontsize=8.5")
exec(f4,base.__dict__);base.f4()

d=base.read('results/v19/F9_patients.csv')
names=['CTN-0094','REPLACE-BG','College Experience Study']
assert [len(d[d.cohort==name]) for name in names]==[154,42,37]
fig=plt.figure(figsize=(7.2,6.4))
gs=fig.add_gridspec(2,3,height_ratios=[1,.98],hspace=.65,wspace=.23)
fig.subplots_adjust(left=.13,right=.98,top=.83,bottom=.14)
a=fig.add_subplot(gs[0,:])
a.set_title('a  Mean score and matched baseline',loc='left',fontweight='bold',pad=36)
a.axvline(.5,ls='--',lw=1,color=base.C['muted'])
summary=[]
for i,name in enumerate(names):
    q=d[d.cohort==name];x=float(q.shuffled_baseline.mean());y=float(q.auc.mean());row=2-i
    summary.append({'cohort':name,'n':len(q),'score':y,'baseline':x,'excess':y-x})
    a.plot([x,y],[row,row],color=base.C['corrected'],lw=3,zorder=1)
    a.scatter(x,row,color=base.C['patient'],marker='s',s=38,zorder=3)
    a.scatter(y,row,color=base.C['pooled'],marker='o',s=40,zorder=3)
    a.text(x-.014,row-.16,f'{x:.2f}',ha='right',va='top',fontsize=9)
    a.text(y+.014,row-.16,f'{y:.2f}',ha='left',va='top',fontsize=9)
    a.text((x+y)/2,row+.17,f'Excess +{y-x:.2f}',ha='center',fontsize=9,fontweight='bold',color=base.C['corrected'])
a.set(xlim=(0,1),ylim=(-.47,2.5),yticks=[2,1,0],yticklabels=['CTN-0094','REPLACE-BG','College Experience'],xticks=[0,.25,.5,.75,1],xlabel='Mean per-patient AUC')
a.spines['left'].set_visible(False);a.tick_params(axis='y',length=0,pad=8)
a.grid(axis='x',color=base.C['grid'],lw=.5)
a.legend(handles=[Line2D([],[],marker='s',ls='',color=base.C['patient'],label='Rebuilt shuffled baseline'),Line2D([],[],marker='o',ls='',color=base.C['pooled'],label='Observed score'),Line2D([],[],ls='--',color=base.C['muted'],label='Conventional 0.5 reference')],loc='lower left',bbox_to_anchor=(0,1.02),ncol=3,fontsize=8,frameon=False)
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
fig.text(.55,.025,'Filled: positive excess   Open: negative excess   Shaded: below 0.5 but above own baseline',ha='center',fontsize=8)
base.save(fig,'F5')
(base.OUT/'mean_auc_numbers.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(base.OUT/'source_hashes.json').write_text(json.dumps(base.used,indent=2),encoding='utf-8')
print(json.dumps(summary))
