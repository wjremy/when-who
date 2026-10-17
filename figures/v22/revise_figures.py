from pathlib import Path
import ast,importlib.util,sys,shutil,json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

R=Path(r'C:\Users\wesre\Documents\Claude\pooled-metrics-paper')
P=Path(__file__).resolve().parent
OUT=P if P.parent.name=='figures' else P/'revised';OUT.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('figbase',R/'figures/v20/draw_figures.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.OUT=OUT
source=(R/'figures/v20/draw_figures.py').read_text(encoding='utf-8')
functions={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}

# Retain the original matrix and axis wording; encode the forecasts by color.
f4=functions['f4']
old="if len(q) and pd.notna(q.iloc[0].excess):ax.scatter(q.iloc[0].excess,y,marker=mark,s=35,color=C['corrected'] if col==3 else C['pooled'] if col==0 else C['patient'],zorder=3)"
new="""if len(q) and pd.notna(q.iloc[0].excess):
                        value=float(q.iloc[0].excess)
                        color=C['pooled'] if forecast=='shuffled rate so far' else C['orange']
                        ax.plot([0,value],[y,y],color=color,lw=2.2,zorder=2,solid_capstyle='round')
                        ax.scatter(value,y,marker='o',s=30,color=color,zorder=3)"""
assert old in f4;f4=f4.replace(old,new)
f4=f4.replace("label='Shuffled rate so far'","label='Shuffled rate so far forecast'").replace("label='Run-in rate'","label='Run-in rate forecast'")
f4=f4.replace("marker='^',ls='',color=C['ink']","marker='o',ls='',color=C['orange']").replace("marker='o',ls='',color=C['ink']","marker='o',ls='',color=C['pooled']")
f4=f4.replace("ax.set_xticks([])","ax.set_xticks([]);[line.set_visible(False) for line in ax.lines]")
# Preserve familiar n/a notation; the caption and body spell out its two causes.
exec(f4,base.__dict__);base.f4()

# Keep Figure 5's excess segments; same marker, distinct colors for endpoints.
v21=(R/'figures/v21/make_figures.py').read_text(encoding='utf-8')
start=v21.index("d=base.read('results/v19/F9_patients.csv')")
f5=v21[start:]
f5=f5.replace("marker='s',s=38","marker='o',s=38").replace("marker='s',ls=''","marker='o',ls=''")
exec(f5,dict(base=base,plt=plt,Line2D=Line2D,Polygon=__import__('matplotlib.patches',fromlist=['Polygon']).Polygon,json=json))

# Approval preview only: square day cells, matching Figure 2a's visual vocabulary.
f1=functions['f1']
cell=__import__('matplotlib.path',fromlist=['Path']).Path([[-.5,-.34],[.5,-.34],[.5,.34],[-.5,.34],[-.5,-.34]])
base.cell=cell
f1=f1.replace("s=32,facecolors=","marker=cell,s=65,facecolors=")
f1=f1.replace("s=55,facecolors=","marker=cell,s=85,facecolors=")
f1=f1.replace("Day (filled circle = event)","Day (filled cell = event)")
exec(f1,base.__dict__);base.f1()
dest=Path(r'C:\Users\wesre\Documents\Codex\2026-10-15\we-x20\outputs')
shutil.copy2(OUT/'F1.png',dest/'Figure-1-day-cells-preview.png')
shutil.copy2(OUT/'F1.pdf',dest/'Figure-1-day-cells-preview.pdf')
for n in ['F4','F5']:shutil.copy2(OUT/(n+'.png'),dest/(n+'-revised.png'))
print('Revised F4/F5 and approval-only F1 preview generated.')
