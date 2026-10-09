"""Shared reconstruction for CTN identity tests and permutation references."""
import json
import numpy as np
NAMES=['updating_rate','recent_window_rate','persistence']

def history_plan(g, event_col, fallback):
    weekly='study_week' in g
    first=g.iloc[0]
    initial=np.array(json.loads(first.history_initial_events),dtype=float)
    positive=weekly and event_col=='event_missing_positive'
    if weekly:
        positions=g.study_week.to_numpy(int)
        if positive:
            # Match the observed sensitivity: averaged run-in padding, not individual weeks.
            initial=np.full(4,float(np.mean(np.where(np.isfinite(initial),initial,1))))
        window=4
        denom=g.expanding_n.to_numpy(float)
        lag0=float(first.previous_result) if np.isfinite(first.previous_result) else np.nan
    else:
        positions=g.study_day.to_numpy(int)-int(first.inferred_first_day)
        window=28
        denom=positions.astype(float)
        lag0=initial[-1]
    assert np.all(np.diff(positions)>0)
    assert positions[0]>=len(initial)
    base=np.full(positions[-1]+1,np.nan)
    base[:len(initial)]=initial
    base[positions]=0  # Every scored row will be replaced, retaining missing calendar slots.
    counts=np.r_[0,np.cumsum(np.isfinite(base))]
    start=np.maximum(0,positions-window)
    return dict(base=base, positions=positions, start=start,
                recent_count=counts[positions]-counts[start],denom=denom,
                lag0=lag0,positive=positive, fallback=float(fallback))

def rebuild_history(plan, targets):
    """Build scores against these exact target labels; supplying observed labels is identity."""
    sequence=plan['base'].copy(); t=plan['positions']
    sequence[t]=np.asarray(targets,int)
    sums=np.r_[0,np.cumsum(np.nan_to_num(sequence,nan=0.))]
    count=plan['recent_count']
    recent=np.divide(sums[t]-sums[plan['start']],count,
                     out=np.full(len(t),np.nan),where=count>0)
    persistence=sequence[t-1].copy()
    if plan['positive']:
        # Sensitivity's first persistence lag is the actual week-3 result (or 1).
        persistence[0]=plan['lag0']
    scores={'updating_rate':sums[t]/plan['denom'],
            'recent_window_rate':recent,'persistence':persistence}
    return {name:np.clip(np.where(np.isfinite(v),v,plan['fallback']),1e-6,1-1e-6)
            for name,v in scores.items()}
