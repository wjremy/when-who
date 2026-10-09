"""Numerical audit checks for saved outputs; all writes stay in this folder."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RESULTS=ROOT/'open_data'/'results'
COHORT=RESULTS/'cohort_analysis'
SIMULATION=RESULTS/'simulation'
EXACT=RESULTS/'exact_counterexample'
sl=pd.read_csv(COHORT/'cohort_slopes.csv')
old=pd.read_csv(HERE/'frozen_public_slopes.csv')
for r in sl.itertuples():
    prior=old[(old.cohort==r.cohort)&(old.forecast=='gradient_boosting')].iloc[0]
    assert abs(r.pooled-prior.pooled_slope)<1e-6
    assert abs(r.joint_within-prior.joint_within_slope)<1e-6
    assert abs(r.conditional-prior.conditional_slope)<1e-6
print('All three GB pooled/joint/conditional slopes reproduce the frozen public reference to 1e-6.')
individual=pd.read_csv(COHORT/'patient_citl.csv')
summary=pd.read_csv(COHORT/'patient_citl_summary.csv')
for r in summary.itertuples():
    z=individual[(individual.cohort==r.cohort)&(individual.forecast==r.forecast)]
    assert len(z)==r.patients
    assert np.allclose(z.error,z.observed_rate-z.mean_forecast)
    assert abs(z.error.mean()-r.mean)<1e-12
    assert abs(np.median(z.error)-r.median)<1e-12
    assert int(sum(abs(z.error)<=.05))==r.within_005
    assert np.all(z.events>=0) and np.all(z.events<=z.n)
print('All patient-error summaries include all patients and reproduce raw probability arithmetic.')
bins=pd.read_csv(COHORT/'within_patient_bins.csv')
for cohort,z in bins.groupby('cohort'):
    r=summary[(summary.cohort==cohort)&(summary.forecast=='gradient_boosting')].iloc[0]
    assert abs(z.equal_patient_exposure.sum()-1)<1e-12
    assert abs(z.moment.sum()-r['mean'])<1e-12
    assert np.allclose(z.bin_error,z.observed_rate-z.mean_forecast)
    assert abs(np.sum(z.equal_patient_exposure*z.observed_rate)-r.observed_patient_mean)<1e-12
print('Bin exposure totals, probability ratios and patient-mean residual contributions reconcile.')
ex=pd.read_csv(EXACT/'exact_markov.csv')
assert np.allclose(ex.conditional_target,[.0374456,.236328,.565058,.699408],atol=1e-6)
assert np.max(abs(ex.predictable_excursion_moment))<1e-12
print('Exact reviewer Markov targets and zero predictable-instrument moments verified.')
if (SIMULATION/'simulation_summary.csv').exists():
    sim=pd.read_csv(SIMULATION/'simulation_summary.csv')
    draws=pd.read_csv(SIMULATION/'simulation_draws.csv')
    assert len(draws)==sim.replications.sum()
    for r in sim.itertuples():
        z=draws[(draws.cohort==r.cohort)&(draws.scenario==r.scenario)&
            (draws['mode']==r.mode)&(draws.rho==r.rho)]
        assert len(z)==r.replications
        for key in ['pooled_slope','joint_within_slope','conditional_slope']:
            v=z[key].dropna()
            assert len(v)==getattr(r,key+'_n')
            assert abs(v.mean()-getattr(r,key+'_mean'))<1e-12
    print('Simulation summaries reconcile with every saved replicate; failures are explicit.')
print('PASS')
