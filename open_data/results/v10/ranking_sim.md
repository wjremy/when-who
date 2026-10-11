# Tailored ranking simulation

Run profile: **full**; seed base **20261011**; workers **4**.

## Design and interpretation

- Latent outcomes use `logit P(Y_kt=1) = level + b_k + x_kt`, daily SD 2, with REPLACE-BG and CTN-0094 cohort fits from `cohort_curves.py`; each fit is recentered to its cohort population rate.
- Scored days start at day 1 with no separate run-in. A's first score uses the population rate, then uses only earlier outcomes with four population-rate pseudo-observations; its rate feature enters on the logit scale.
- A's two-component logistic recalibration is fitted on an independent 200,000-row simulated cohort for each spread, record length, and A daily correlation. B conditions exactly on b and its daily proxy using the 40-node Gauss-Hermite helper from `who_when_sim.py`.
- Pooled scores are raw day-level values. Naive scores are equal-patient raw values following the selected `naive_metrics` definitions. Tailored scores use the selected per-patient `nine_metrics` definitions. The primary tailored ranking uses B's `shuffle` chance level; the additional `tailored (B against its level-only forecast)` row preserves the former B baseline.
- For shuffled chance, A is recalibrated once and held fixed while its history feature is rebuilt for each shuffle. B's fixed forecast is held unchanged while the same shuffled outcome arrays are scored. Outcome-based yardsticks, including the running rate for Brier skill and the own-rate references for PPV and net benefit, are recomputed from each shuffled outcome.
- Tailored Brier skill is shown against both the running rate based on earlier outcomes and the oracle patient risk integrated over daily x. Naive Brier skill uses the population rate.
- Net benefit uses the population-rate threshold; tailored per-patient net benefit follows `nine_metrics`' excess over the patient's own-rate policy. PPV follows its own-rate excess definition.
- Long-record true patient ability is estimated from 200 independent patients with 5,000 scored days each, separately for each record-length calibration of A. AUC values average over patients with both outcomes.
- The B daily-rho-zero size diagnostic uses a fixed independent random tie-breaker for within-patient AUC. B's exact rho-zero forecast is constant within patient, so conventional tie-aware AUC is identically 0.5 and would make a size check degenerate.
- Random arrays, including shuffles and tie-breakers, are generated in the parent before tasks are sent to workers. Replicate streams are keyed by cell and replicate, so changing `--workers` does not change results.

## Ranking results

Share of valid replications where A scores above B; the long-record AUC columns give the simulated reference ability.

| Spread | Days | Patients | Pair | Scoring | Metric | A above B | True A AUC | True B AUC |
|---|---:|---:|---|---|---|---:|---:|---:|
| low | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | auc | 0.982 | 0.750 | 0.584 |
| low | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | average_precision | 0.950 | 0.750 | 0.584 |
| low | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | brier_skill_updated_rate | 0.968 | 0.750 | 0.584 |
| low | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | brier_skill_oracle | 0.968 | 0.750 | 0.584 |
| low | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | ppv | 0.996 | 0.750 | 0.584 |
| low | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | net_benefit | 0.904 | 0.750 | 0.584 |
| low | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | auc | 1.000 | 0.750 | 0.584 |
| low | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | average_precision | 1.000 | 0.750 | 0.584 |
| low | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | brier_skill_population_rate | 0.968 | 0.750 | 0.584 |
| low | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | ppv | 0.900 | 0.750 | 0.584 |
| low | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | net_benefit | 0.904 | 0.750 | 0.584 |
| low | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | auc | 1.000 | 0.750 | 0.584 |
| low | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | average_precision | 1.000 | 0.750 | 0.584 |
| low | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | brier_skill_updated_rate | 1.000 | 0.750 | 0.584 |
| low | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | brier_skill_oracle | 1.000 | 0.750 | 0.584 |
| low | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | ppv | 0.978 | 0.750 | 0.584 |
| low | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | net_benefit | 1.000 | 0.750 | 0.584 |
| low | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | auc | 1.000 | 0.750 | 0.584 |
| low | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | average_precision | 0.812 | 0.750 | 0.584 |
| low | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | brier_skill_updated_rate | 0.998 | 0.750 | 0.584 |
| low | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | brier_skill_oracle | 1.000 | 0.750 | 0.584 |
| low | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | ppv | 0.980 | 0.750 | 0.584 |
| low | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | net_benefit | 1.000 | 0.750 | 0.584 |
| low | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | auc | 0.110 | 0.668 | 0.627 |
| low | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | average_precision | 0.216 | 0.668 | 0.627 |
| low | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | brier_skill_updated_rate | 0.136 | 0.668 | 0.627 |
| low | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | brier_skill_oracle | 0.136 | 0.668 | 0.627 |
| low | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | ppv | 0.580 | 0.668 | 0.627 |
| low | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | net_benefit | 0.212 | 0.668 | 0.627 |
| low | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | auc | 0.004 | 0.668 | 0.627 |
| low | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | average_precision | 0.064 | 0.668 | 0.627 |
| low | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | brier_skill_population_rate | 0.136 | 0.668 | 0.627 |
| low | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | ppv | 0.312 | 0.668 | 0.627 |
| low | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | net_benefit | 0.212 | 0.668 | 0.627 |
| low | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | auc | 1.000 | 0.668 | 0.627 |
| low | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | average_precision | 0.708 | 0.668 | 0.627 |
| low | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | brier_skill_updated_rate | 1.000 | 0.668 | 0.627 |
| low | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | brier_skill_oracle | 0.998 | 0.668 | 0.627 |
| low | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | ppv | 0.746 | 0.668 | 0.627 |
| low | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | net_benefit | 0.988 | 0.668 | 0.627 |
| low | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | auc | 1.000 | 0.668 | 0.627 |
| low | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | average_precision | 0.000 | 0.668 | 0.627 |
| low | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | brier_skill_updated_rate | 0.996 | 0.668 | 0.627 |
| low | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | brier_skill_oracle | 1.000 | 0.668 | 0.627 |
| low | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | ppv | 0.740 | 0.668 | 0.627 |
| low | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | net_benefit | 0.994 | 0.668 | 0.627 |
| low | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | auc | 1.000 | 0.750 | 0.584 |
| low | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | average_precision | 0.998 | 0.750 | 0.584 |
| low | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | brier_skill_updated_rate | 0.998 | 0.750 | 0.584 |
| low | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | brier_skill_oracle | 0.998 | 0.750 | 0.584 |
| low | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | ppv | 1.000 | 0.750 | 0.584 |
| low | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | net_benefit | 0.992 | 0.750 | 0.584 |
| low | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | auc | 1.000 | 0.750 | 0.584 |
| low | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | average_precision | 1.000 | 0.750 | 0.584 |
| low | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | brier_skill_population_rate | 0.998 | 0.750 | 0.584 |
| low | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | ppv | 0.966 | 0.750 | 0.584 |
| low | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | net_benefit | 0.992 | 0.750 | 0.584 |
| low | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | auc | 1.000 | 0.750 | 0.584 |
| low | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | average_precision | 1.000 | 0.750 | 0.584 |
| low | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | brier_skill_updated_rate | 1.000 | 0.750 | 0.584 |
| low | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | brier_skill_oracle | 1.000 | 0.750 | 0.584 |
| low | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | ppv | 1.000 | 0.750 | 0.584 |
| low | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | net_benefit | 1.000 | 0.750 | 0.584 |
| low | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | auc | 1.000 | 0.750 | 0.584 |
| low | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | average_precision | 0.966 | 0.750 | 0.584 |
| low | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | brier_skill_updated_rate | 1.000 | 0.750 | 0.584 |
| low | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | brier_skill_oracle | 1.000 | 0.750 | 0.584 |
| low | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | ppv | 1.000 | 0.750 | 0.584 |
| low | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | net_benefit | 1.000 | 0.750 | 0.584 |
| low | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | auc | 0.006 | 0.668 | 0.627 |
| low | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | average_precision | 0.086 | 0.668 | 0.627 |
| low | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | brier_skill_updated_rate | 0.014 | 0.668 | 0.627 |
| low | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | brier_skill_oracle | 0.014 | 0.668 | 0.627 |
| low | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | ppv | 0.628 | 0.668 | 0.627 |
| low | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | net_benefit | 0.050 | 0.668 | 0.627 |
| low | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | auc | 0.000 | 0.668 | 0.627 |
| low | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | average_precision | 0.000 | 0.668 | 0.627 |
| low | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | brier_skill_population_rate | 0.014 | 0.668 | 0.627 |
| low | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | ppv | 0.190 | 0.668 | 0.627 |
| low | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | net_benefit | 0.050 | 0.668 | 0.627 |
| low | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | auc | 1.000 | 0.668 | 0.627 |
| low | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | average_precision | 0.842 | 0.668 | 0.627 |
| low | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | brier_skill_updated_rate | 1.000 | 0.668 | 0.627 |
| low | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | brier_skill_oracle | 1.000 | 0.668 | 0.627 |
| low | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | ppv | 0.796 | 0.668 | 0.627 |
| low | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | net_benefit | 1.000 | 0.668 | 0.627 |
| low | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | auc | 1.000 | 0.668 | 0.627 |
| low | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | average_precision | 0.000 | 0.668 | 0.627 |
| low | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | brier_skill_updated_rate | 1.000 | 0.668 | 0.627 |
| low | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | brier_skill_oracle | 1.000 | 0.668 | 0.627 |
| low | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | ppv | 0.802 | 0.668 | 0.627 |
| low | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | net_benefit | 1.000 | 0.668 | 0.627 |
| low | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | auc | 1.000 | 0.749 | 0.584 |
| low | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | average_precision | 1.000 | 0.749 | 0.584 |
| low | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | brier_skill_updated_rate | 1.000 | 0.749 | 0.584 |
| low | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | brier_skill_oracle | 1.000 | 0.749 | 0.584 |
| low | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | ppv | 0.998 | 0.749 | 0.584 |
| low | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | net_benefit | 1.000 | 0.749 | 0.584 |
| low | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | auc | 1.000 | 0.749 | 0.584 |
| low | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | average_precision | 1.000 | 0.749 | 0.584 |
| low | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | brier_skill_population_rate | 1.000 | 0.749 | 0.584 |
| low | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | ppv | 0.928 | 0.749 | 0.584 |
| low | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | net_benefit | 1.000 | 0.749 | 0.584 |
| low | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | auc | 1.000 | 0.749 | 0.584 |
| low | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | average_precision | 1.000 | 0.749 | 0.584 |
| low | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | brier_skill_updated_rate | 1.000 | 0.749 | 0.584 |
| low | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | brier_skill_oracle | 1.000 | 0.749 | 0.584 |
| low | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | ppv | 0.982 | 0.749 | 0.584 |
| low | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | net_benefit | 1.000 | 0.749 | 0.584 |
| low | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | auc | 1.000 | 0.749 | 0.584 |
| low | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | average_precision | 1.000 | 0.749 | 0.584 |
| low | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | brier_skill_updated_rate | 1.000 | 0.749 | 0.584 |
| low | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | brier_skill_oracle | 1.000 | 0.749 | 0.584 |
| low | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | ppv | 0.982 | 0.749 | 0.584 |
| low | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | net_benefit | 1.000 | 0.749 | 0.584 |
| low | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | auc | 0.074 | 0.667 | 0.627 |
| low | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | average_precision | 0.240 | 0.667 | 0.627 |
| low | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | brier_skill_updated_rate | 0.116 | 0.667 | 0.627 |
| low | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | brier_skill_oracle | 0.116 | 0.667 | 0.627 |
| low | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | ppv | 0.384 | 0.667 | 0.627 |
| low | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | net_benefit | 0.212 | 0.667 | 0.627 |
| low | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | auc | 0.024 | 0.667 | 0.627 |
| low | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | average_precision | 0.372 | 0.667 | 0.627 |
| low | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | brier_skill_population_rate | 0.116 | 0.667 | 0.627 |
| low | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | ppv | 0.280 | 0.667 | 0.627 |
| low | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | net_benefit | 0.212 | 0.667 | 0.627 |
| low | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | auc | 1.000 | 0.667 | 0.627 |
| low | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | average_precision | 0.830 | 0.667 | 0.627 |
| low | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | brier_skill_updated_rate | 1.000 | 0.667 | 0.627 |
| low | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | brier_skill_oracle | 1.000 | 0.667 | 0.627 |
| low | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | ppv | 0.618 | 0.667 | 0.627 |
| low | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | net_benefit | 1.000 | 0.667 | 0.627 |
| low | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | auc | 1.000 | 0.667 | 0.627 |
| low | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | average_precision | 0.000 | 0.667 | 0.627 |
| low | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | brier_skill_updated_rate | 1.000 | 0.667 | 0.627 |
| low | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | brier_skill_oracle | 1.000 | 0.667 | 0.627 |
| low | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | ppv | 0.640 | 0.667 | 0.627 |
| low | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | net_benefit | 1.000 | 0.667 | 0.627 |
| low | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | auc | 1.000 | 0.749 | 0.584 |
| low | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | average_precision | 1.000 | 0.749 | 0.584 |
| low | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | brier_skill_updated_rate | 1.000 | 0.749 | 0.584 |
| low | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | brier_skill_oracle | 1.000 | 0.749 | 0.584 |
| low | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | ppv | 1.000 | 0.749 | 0.584 |
| low | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | net_benefit | 1.000 | 0.749 | 0.584 |
| low | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | auc | 1.000 | 0.749 | 0.584 |
| low | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | average_precision | 1.000 | 0.749 | 0.584 |
| low | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | brier_skill_population_rate | 1.000 | 0.749 | 0.584 |
| low | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | ppv | 0.980 | 0.749 | 0.584 |
| low | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | net_benefit | 1.000 | 0.749 | 0.584 |
| low | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | auc | 1.000 | 0.749 | 0.584 |
| low | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | average_precision | 1.000 | 0.749 | 0.584 |
| low | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | brier_skill_updated_rate | 1.000 | 0.749 | 0.584 |
| low | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | brier_skill_oracle | 1.000 | 0.749 | 0.584 |
| low | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | ppv | 1.000 | 0.749 | 0.584 |
| low | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | net_benefit | 1.000 | 0.749 | 0.584 |
| low | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | auc | 1.000 | 0.749 | 0.584 |
| low | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | average_precision | 1.000 | 0.749 | 0.584 |
| low | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | brier_skill_updated_rate | 1.000 | 0.749 | 0.584 |
| low | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | brier_skill_oracle | 1.000 | 0.749 | 0.584 |
| low | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | ppv | 1.000 | 0.749 | 0.584 |
| low | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | net_benefit | 1.000 | 0.749 | 0.584 |
| low | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | auc | 0.000 | 0.667 | 0.627 |
| low | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | average_precision | 0.076 | 0.667 | 0.627 |
| low | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | brier_skill_updated_rate | 0.022 | 0.667 | 0.627 |
| low | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | brier_skill_oracle | 0.022 | 0.667 | 0.627 |
| low | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | ppv | 0.326 | 0.667 | 0.627 |
| low | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | net_benefit | 0.068 | 0.667 | 0.627 |
| low | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | auc | 0.000 | 0.667 | 0.627 |
| low | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | average_precision | 0.188 | 0.667 | 0.627 |
| low | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | brier_skill_population_rate | 0.022 | 0.667 | 0.627 |
| low | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | ppv | 0.150 | 0.667 | 0.627 |
| low | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | net_benefit | 0.068 | 0.667 | 0.627 |
| low | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | auc | 1.000 | 0.667 | 0.627 |
| low | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | average_precision | 0.938 | 0.667 | 0.627 |
| low | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | brier_skill_updated_rate | 1.000 | 0.667 | 0.627 |
| low | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | brier_skill_oracle | 1.000 | 0.667 | 0.627 |
| low | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | ppv | 0.622 | 0.667 | 0.627 |
| low | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | net_benefit | 1.000 | 0.667 | 0.627 |
| low | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | auc | 1.000 | 0.667 | 0.627 |
| low | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | average_precision | 0.000 | 0.667 | 0.627 |
| low | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | brier_skill_updated_rate | 1.000 | 0.667 | 0.627 |
| low | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | brier_skill_oracle | 1.000 | 0.667 | 0.627 |
| low | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | ppv | 0.626 | 0.667 | 0.627 |
| low | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | net_benefit | 1.000 | 0.667 | 0.627 |
| low | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | auc | 1.000 | 0.748 | 0.584 |
| low | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | average_precision | 1.000 | 0.748 | 0.584 |
| low | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | brier_skill_updated_rate | 1.000 | 0.748 | 0.584 |
| low | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | brier_skill_oracle | 1.000 | 0.748 | 0.584 |
| low | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | ppv | 1.000 | 0.748 | 0.584 |
| low | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | net_benefit | 1.000 | 0.748 | 0.584 |
| low | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | auc | 1.000 | 0.748 | 0.584 |
| low | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | average_precision | 1.000 | 0.748 | 0.584 |
| low | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | brier_skill_population_rate | 1.000 | 0.748 | 0.584 |
| low | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | ppv | 0.966 | 0.748 | 0.584 |
| low | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | net_benefit | 1.000 | 0.748 | 0.584 |
| low | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | auc | 1.000 | 0.748 | 0.584 |
| low | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | average_precision | 1.000 | 0.748 | 0.584 |
| low | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | brier_skill_updated_rate | 1.000 | 0.748 | 0.584 |
| low | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | brier_skill_oracle | 1.000 | 0.748 | 0.584 |
| low | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | ppv | 0.980 | 0.748 | 0.584 |
| low | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | net_benefit | 1.000 | 0.748 | 0.584 |
| low | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | auc | 1.000 | 0.748 | 0.584 |
| low | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | average_precision | 1.000 | 0.748 | 0.584 |
| low | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | brier_skill_updated_rate | 1.000 | 0.748 | 0.584 |
| low | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | brier_skill_oracle | 1.000 | 0.748 | 0.584 |
| low | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | ppv | 0.980 | 0.748 | 0.584 |
| low | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | net_benefit | 1.000 | 0.748 | 0.584 |
| low | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | auc | 0.782 | 0.665 | 0.627 |
| low | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | average_precision | 0.912 | 0.665 | 0.627 |
| low | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | brier_skill_updated_rate | 0.866 | 0.665 | 0.627 |
| low | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | brier_skill_oracle | 0.866 | 0.665 | 0.627 |
| low | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | ppv | 0.412 | 0.665 | 0.627 |
| low | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | net_benefit | 0.686 | 0.665 | 0.627 |
| low | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | auc | 0.938 | 0.665 | 0.627 |
| low | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | average_precision | 0.980 | 0.665 | 0.627 |
| low | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | brier_skill_population_rate | 0.866 | 0.665 | 0.627 |
| low | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | ppv | 0.326 | 0.665 | 0.627 |
| low | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | net_benefit | 0.686 | 0.665 | 0.627 |
| low | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | auc | 1.000 | 0.665 | 0.627 |
| low | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | average_precision | 0.996 | 0.665 | 0.627 |
| low | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | brier_skill_updated_rate | 1.000 | 0.665 | 0.627 |
| low | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | brier_skill_oracle | 1.000 | 0.665 | 0.627 |
| low | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | ppv | 0.602 | 0.665 | 0.627 |
| low | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | net_benefit | 1.000 | 0.665 | 0.627 |
| low | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | auc | 1.000 | 0.665 | 0.627 |
| low | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | average_precision | 0.202 | 0.665 | 0.627 |
| low | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | brier_skill_updated_rate | 1.000 | 0.665 | 0.627 |
| low | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | brier_skill_oracle | 1.000 | 0.665 | 0.627 |
| low | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | ppv | 0.604 | 0.665 | 0.627 |
| low | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | net_benefit | 1.000 | 0.665 | 0.627 |
| low | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | auc | 1.000 | 0.748 | 0.584 |
| low | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | average_precision | 1.000 | 0.748 | 0.584 |
| low | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | brier_skill_updated_rate | 1.000 | 0.748 | 0.584 |
| low | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | brier_skill_oracle | 1.000 | 0.748 | 0.584 |
| low | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | ppv | 1.000 | 0.748 | 0.584 |
| low | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | net_benefit | 1.000 | 0.748 | 0.584 |
| low | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | auc | 1.000 | 0.748 | 0.584 |
| low | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | average_precision | 1.000 | 0.748 | 0.584 |
| low | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | brier_skill_population_rate | 1.000 | 0.748 | 0.584 |
| low | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | ppv | 0.998 | 0.748 | 0.584 |
| low | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | net_benefit | 1.000 | 0.748 | 0.584 |
| low | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | auc | 1.000 | 0.748 | 0.584 |
| low | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | average_precision | 1.000 | 0.748 | 0.584 |
| low | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | brier_skill_updated_rate | 1.000 | 0.748 | 0.584 |
| low | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | brier_skill_oracle | 1.000 | 0.748 | 0.584 |
| low | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | ppv | 1.000 | 0.748 | 0.584 |
| low | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | net_benefit | 1.000 | 0.748 | 0.584 |
| low | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | auc | 1.000 | 0.748 | 0.584 |
| low | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | average_precision | 1.000 | 0.748 | 0.584 |
| low | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | brier_skill_updated_rate | 1.000 | 0.748 | 0.584 |
| low | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | brier_skill_oracle | 1.000 | 0.748 | 0.584 |
| low | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | ppv | 1.000 | 0.748 | 0.584 |
| low | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | net_benefit | 1.000 | 0.748 | 0.584 |
| low | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | auc | 0.886 | 0.665 | 0.627 |
| low | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | average_precision | 0.966 | 0.665 | 0.627 |
| low | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | brier_skill_updated_rate | 0.960 | 0.665 | 0.627 |
| low | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | brier_skill_oracle | 0.960 | 0.665 | 0.627 |
| low | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | ppv | 0.378 | 0.665 | 0.627 |
| low | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | net_benefit | 0.792 | 0.665 | 0.627 |
| low | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | auc | 0.986 | 0.665 | 0.627 |
| low | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | average_precision | 1.000 | 0.665 | 0.627 |
| low | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | brier_skill_population_rate | 0.960 | 0.665 | 0.627 |
| low | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | ppv | 0.222 | 0.665 | 0.627 |
| low | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | net_benefit | 0.792 | 0.665 | 0.627 |
| low | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | auc | 1.000 | 0.665 | 0.627 |
| low | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | average_precision | 1.000 | 0.665 | 0.627 |
| low | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | brier_skill_updated_rate | 1.000 | 0.665 | 0.627 |
| low | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | brier_skill_oracle | 1.000 | 0.665 | 0.627 |
| low | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | ppv | 0.580 | 0.665 | 0.627 |
| low | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | net_benefit | 1.000 | 0.665 | 0.627 |
| low | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | auc | 1.000 | 0.665 | 0.627 |
| low | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | average_precision | 0.126 | 0.665 | 0.627 |
| low | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | brier_skill_updated_rate | 1.000 | 0.665 | 0.627 |
| low | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | brier_skill_oracle | 1.000 | 0.665 | 0.627 |
| low | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | ppv | 0.570 | 0.665 | 0.627 |
| low | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | net_benefit | 1.000 | 0.665 | 0.627 |
| high | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | auc | 0.168 | 0.726 | 0.580 |
| high | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | average_precision | 0.390 | 0.726 | 0.580 |
| high | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | brier_skill_updated_rate | 0.222 | 0.726 | 0.580 |
| high | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | brier_skill_oracle | 0.222 | 0.726 | 0.580 |
| high | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | ppv | 0.530 | 0.726 | 0.580 |
| high | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | net_benefit | 0.270 | 0.726 | 0.580 |
| high | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | auc | 1.000 | 0.726 | 0.580 |
| high | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | average_precision | 0.984 | 0.726 | 0.580 |
| high | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | brier_skill_population_rate | 0.222 | 0.726 | 0.580 |
| high | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | ppv | 0.000 | 0.726 | 0.580 |
| high | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | net_benefit | 0.270 | 0.726 | 0.580 |
| high | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | auc | 1.000 | 0.726 | 0.580 |
| high | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | average_precision | 1.000 | 0.726 | 0.580 |
| high | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | brier_skill_updated_rate | 1.000 | 0.726 | 0.580 |
| high | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | brier_skill_oracle | 1.000 | 0.726 | 0.580 |
| high | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | ppv | 0.966 | 0.726 | 0.580 |
| high | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | net_benefit | 1.000 | 0.726 | 0.580 |
| high | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | auc | 1.000 | 0.726 | 0.580 |
| high | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | average_precision | 0.652 | 0.726 | 0.580 |
| high | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | brier_skill_updated_rate | 1.000 | 0.726 | 0.580 |
| high | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | brier_skill_oracle | 1.000 | 0.726 | 0.580 |
| high | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | ppv | 0.964 | 0.726 | 0.580 |
| high | 30 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | net_benefit | 1.000 | 0.726 | 0.580 |
| high | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | auc | 0.000 | 0.647 | 0.619 |
| high | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | average_precision | 0.000 | 0.647 | 0.619 |
| high | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | brier_skill_updated_rate | 0.000 | 0.647 | 0.619 |
| high | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | brier_skill_oracle | 0.000 | 0.647 | 0.619 |
| high | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | ppv | 0.126 | 0.647 | 0.619 |
| high | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | net_benefit | 0.000 | 0.647 | 0.619 |
| high | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | auc | 0.000 | 0.647 | 0.619 |
| high | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | average_precision | 0.000 | 0.647 | 0.619 |
| high | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | brier_skill_population_rate | 0.000 | 0.647 | 0.619 |
| high | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | ppv | 0.000 | 0.647 | 0.619 |
| high | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | net_benefit | 0.000 | 0.647 | 0.619 |
| high | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | auc | 0.486 | 0.647 | 0.619 |
| high | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | average_precision | 0.178 | 0.647 | 0.619 |
| high | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | brier_skill_updated_rate | 1.000 | 0.647 | 0.619 |
| high | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | brier_skill_oracle | 1.000 | 0.647 | 0.619 |
| high | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | ppv | 0.536 | 0.647 | 0.619 |
| high | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | net_benefit | 0.934 | 0.647 | 0.619 |
| high | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | auc | 0.480 | 0.647 | 0.619 |
| high | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | average_precision | 0.000 | 0.647 | 0.619 |
| high | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | brier_skill_updated_rate | 1.000 | 0.647 | 0.619 |
| high | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | brier_skill_oracle | 1.000 | 0.647 | 0.619 |
| high | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | ppv | 0.528 | 0.647 | 0.619 |
| high | 30 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | net_benefit | 0.988 | 0.647 | 0.619 |
| high | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | auc | 0.034 | 0.726 | 0.580 |
| high | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | average_precision | 0.238 | 0.726 | 0.580 |
| high | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | brier_skill_updated_rate | 0.106 | 0.726 | 0.580 |
| high | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | brier_skill_oracle | 0.106 | 0.726 | 0.580 |
| high | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | ppv | 0.576 | 0.726 | 0.580 |
| high | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | net_benefit | 0.232 | 0.726 | 0.580 |
| high | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | auc | 1.000 | 0.726 | 0.580 |
| high | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | average_precision | 0.998 | 0.726 | 0.580 |
| high | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | brier_skill_population_rate | 0.106 | 0.726 | 0.580 |
| high | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | ppv | 0.000 | 0.726 | 0.580 |
| high | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | net_benefit | 0.232 | 0.726 | 0.580 |
| high | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | auc | 1.000 | 0.726 | 0.580 |
| high | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | average_precision | 1.000 | 0.726 | 0.580 |
| high | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | brier_skill_updated_rate | 1.000 | 0.726 | 0.580 |
| high | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | brier_skill_oracle | 1.000 | 0.726 | 0.580 |
| high | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | ppv | 1.000 | 0.726 | 0.580 |
| high | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | net_benefit | 1.000 | 0.726 | 0.580 |
| high | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | auc | 1.000 | 0.726 | 0.580 |
| high | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | average_precision | 0.742 | 0.726 | 0.580 |
| high | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | brier_skill_updated_rate | 1.000 | 0.726 | 0.580 |
| high | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | brier_skill_oracle | 1.000 | 0.726 | 0.580 |
| high | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | ppv | 1.000 | 0.726 | 0.580 |
| high | 30 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | net_benefit | 1.000 | 0.726 | 0.580 |
| high | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | auc | 0.000 | 0.647 | 0.619 |
| high | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | average_precision | 0.000 | 0.647 | 0.619 |
| high | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | brier_skill_updated_rate | 0.000 | 0.647 | 0.619 |
| high | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | brier_skill_oracle | 0.000 | 0.647 | 0.619 |
| high | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | ppv | 0.030 | 0.647 | 0.619 |
| high | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | net_benefit | 0.000 | 0.647 | 0.619 |
| high | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | auc | 0.000 | 0.647 | 0.619 |
| high | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | average_precision | 0.000 | 0.647 | 0.619 |
| high | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | brier_skill_population_rate | 0.000 | 0.647 | 0.619 |
| high | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | ppv | 0.000 | 0.647 | 0.619 |
| high | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | net_benefit | 0.000 | 0.647 | 0.619 |
| high | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | auc | 0.482 | 0.647 | 0.619 |
| high | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | average_precision | 0.048 | 0.647 | 0.619 |
| high | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | brier_skill_updated_rate | 1.000 | 0.647 | 0.619 |
| high | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | brier_skill_oracle | 1.000 | 0.647 | 0.619 |
| high | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | ppv | 0.496 | 0.647 | 0.619 |
| high | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | net_benefit | 0.992 | 0.647 | 0.619 |
| high | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | auc | 0.470 | 0.647 | 0.619 |
| high | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | average_precision | 0.000 | 0.647 | 0.619 |
| high | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | brier_skill_updated_rate | 1.000 | 0.647 | 0.619 |
| high | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | brier_skill_oracle | 1.000 | 0.647 | 0.619 |
| high | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | ppv | 0.490 | 0.647 | 0.619 |
| high | 30 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | net_benefit | 0.998 | 0.647 | 0.619 |
| high | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | auc | 1.000 | 0.728 | 0.580 |
| high | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | average_precision | 1.000 | 0.728 | 0.580 |
| high | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | brier_skill_updated_rate | 1.000 | 0.728 | 0.580 |
| high | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | brier_skill_oracle | 1.000 | 0.728 | 0.580 |
| high | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | ppv | 0.926 | 0.728 | 0.580 |
| high | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | net_benefit | 0.988 | 0.728 | 0.580 |
| high | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | auc | 1.000 | 0.728 | 0.580 |
| high | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | average_precision | 1.000 | 0.728 | 0.580 |
| high | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | brier_skill_population_rate | 1.000 | 0.728 | 0.580 |
| high | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | ppv | 0.000 | 0.728 | 0.580 |
| high | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | net_benefit | 0.988 | 0.728 | 0.580 |
| high | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | auc | 1.000 | 0.728 | 0.580 |
| high | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | average_precision | 1.000 | 0.728 | 0.580 |
| high | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | brier_skill_updated_rate | 1.000 | 0.728 | 0.580 |
| high | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | brier_skill_oracle | 1.000 | 0.728 | 0.580 |
| high | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | ppv | 0.992 | 0.728 | 0.580 |
| high | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | net_benefit | 1.000 | 0.728 | 0.580 |
| high | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | auc | 1.000 | 0.728 | 0.580 |
| high | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | average_precision | 1.000 | 0.728 | 0.580 |
| high | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | brier_skill_updated_rate | 1.000 | 0.728 | 0.580 |
| high | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | brier_skill_oracle | 1.000 | 0.728 | 0.580 |
| high | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | ppv | 0.990 | 0.728 | 0.580 |
| high | 100 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | net_benefit | 1.000 | 0.728 | 0.580 |
| high | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | auc | 0.000 | 0.649 | 0.619 |
| high | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | average_precision | 0.002 | 0.649 | 0.619 |
| high | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | brier_skill_updated_rate | 0.000 | 0.649 | 0.619 |
| high | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | brier_skill_oracle | 0.000 | 0.649 | 0.619 |
| high | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | ppv | 0.324 | 0.649 | 0.619 |
| high | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | net_benefit | 0.030 | 0.649 | 0.619 |
| high | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | auc | 0.006 | 0.649 | 0.619 |
| high | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | average_precision | 0.028 | 0.649 | 0.619 |
| high | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | brier_skill_population_rate | 0.000 | 0.649 | 0.619 |
| high | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | ppv | 0.000 | 0.649 | 0.619 |
| high | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | net_benefit | 0.030 | 0.649 | 0.619 |
| high | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | auc | 0.886 | 0.649 | 0.619 |
| high | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | average_precision | 0.548 | 0.649 | 0.619 |
| high | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | brier_skill_updated_rate | 1.000 | 0.649 | 0.619 |
| high | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | brier_skill_oracle | 1.000 | 0.649 | 0.619 |
| high | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | ppv | 0.506 | 0.649 | 0.619 |
| high | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | net_benefit | 0.998 | 0.649 | 0.619 |
| high | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | auc | 0.884 | 0.649 | 0.619 |
| high | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | average_precision | 0.002 | 0.649 | 0.619 |
| high | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | brier_skill_updated_rate | 1.000 | 0.649 | 0.619 |
| high | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | brier_skill_oracle | 1.000 | 0.649 | 0.619 |
| high | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | ppv | 0.494 | 0.649 | 0.619 |
| high | 100 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | net_benefit | 1.000 | 0.649 | 0.619 |
| high | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | auc | 1.000 | 0.728 | 0.580 |
| high | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | average_precision | 1.000 | 0.728 | 0.580 |
| high | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | brier_skill_updated_rate | 1.000 | 0.728 | 0.580 |
| high | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | brier_skill_oracle | 1.000 | 0.728 | 0.580 |
| high | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | ppv | 0.992 | 0.728 | 0.580 |
| high | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | net_benefit | 1.000 | 0.728 | 0.580 |
| high | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | auc | 1.000 | 0.728 | 0.580 |
| high | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | average_precision | 1.000 | 0.728 | 0.580 |
| high | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | brier_skill_population_rate | 1.000 | 0.728 | 0.580 |
| high | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | ppv | 0.000 | 0.728 | 0.580 |
| high | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | net_benefit | 1.000 | 0.728 | 0.580 |
| high | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | auc | 1.000 | 0.728 | 0.580 |
| high | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | average_precision | 1.000 | 0.728 | 0.580 |
| high | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | brier_skill_updated_rate | 1.000 | 0.728 | 0.580 |
| high | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | brier_skill_oracle | 1.000 | 0.728 | 0.580 |
| high | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | ppv | 1.000 | 0.728 | 0.580 |
| high | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | net_benefit | 1.000 | 0.728 | 0.580 |
| high | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | auc | 1.000 | 0.728 | 0.580 |
| high | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | average_precision | 1.000 | 0.728 | 0.580 |
| high | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | brier_skill_updated_rate | 1.000 | 0.728 | 0.580 |
| high | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | brier_skill_oracle | 1.000 | 0.728 | 0.580 |
| high | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | ppv | 1.000 | 0.728 | 0.580 |
| high | 100 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | net_benefit | 1.000 | 0.728 | 0.580 |
| high | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | auc | 0.000 | 0.649 | 0.619 |
| high | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | average_precision | 0.000 | 0.649 | 0.619 |
| high | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | brier_skill_updated_rate | 0.000 | 0.649 | 0.619 |
| high | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | brier_skill_oracle | 0.000 | 0.649 | 0.619 |
| high | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | ppv | 0.254 | 0.649 | 0.619 |
| high | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | net_benefit | 0.000 | 0.649 | 0.619 |
| high | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | auc | 0.000 | 0.649 | 0.619 |
| high | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | average_precision | 0.002 | 0.649 | 0.619 |
| high | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | brier_skill_population_rate | 0.000 | 0.649 | 0.619 |
| high | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | ppv | 0.000 | 0.649 | 0.619 |
| high | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | net_benefit | 0.000 | 0.649 | 0.619 |
| high | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | auc | 0.986 | 0.649 | 0.619 |
| high | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | average_precision | 0.494 | 0.649 | 0.619 |
| high | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | brier_skill_updated_rate | 1.000 | 0.649 | 0.619 |
| high | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | brier_skill_oracle | 1.000 | 0.649 | 0.619 |
| high | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | ppv | 0.488 | 0.649 | 0.619 |
| high | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | net_benefit | 1.000 | 0.649 | 0.619 |
| high | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | auc | 0.992 | 0.649 | 0.619 |
| high | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | average_precision | 0.000 | 0.649 | 0.619 |
| high | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | brier_skill_updated_rate | 1.000 | 0.649 | 0.619 |
| high | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | brier_skill_oracle | 1.000 | 0.649 | 0.619 |
| high | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | ppv | 0.480 | 0.649 | 0.619 |
| high | 100 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | net_benefit | 1.000 | 0.649 | 0.619 |
| high | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | auc | 1.000 | 0.728 | 0.580 |
| high | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | average_precision | 1.000 | 0.728 | 0.580 |
| high | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | brier_skill_updated_rate | 1.000 | 0.728 | 0.580 |
| high | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | brier_skill_oracle | 1.000 | 0.728 | 0.580 |
| high | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | ppv | 0.992 | 0.728 | 0.580 |
| high | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | pooled | net_benefit | 1.000 | 0.728 | 0.580 |
| high | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | auc | 1.000 | 0.728 | 0.580 |
| high | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | average_precision | 1.000 | 0.728 | 0.580 |
| high | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | brier_skill_population_rate | 1.000 | 0.728 | 0.580 |
| high | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | ppv | 0.008 | 0.728 | 0.580 |
| high | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | naive | net_benefit | 1.000 | 0.728 | 0.580 |
| high | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | auc | 1.000 | 0.728 | 0.580 |
| high | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | average_precision | 1.000 | 0.728 | 0.580 |
| high | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | brier_skill_updated_rate | 1.000 | 0.728 | 0.580 |
| high | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | brier_skill_oracle | 1.000 | 0.728 | 0.580 |
| high | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | ppv | 0.998 | 0.728 | 0.580 |
| high | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | net_benefit | 1.000 | 0.728 | 0.580 |
| high | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | auc | 1.000 | 0.728 | 0.580 |
| high | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | average_precision | 1.000 | 0.728 | 0.580 |
| high | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | brier_skill_updated_rate | 1.000 | 0.728 | 0.580 |
| high | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | brier_skill_oracle | 1.000 | 0.728 | 0.580 |
| high | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | ppv | 0.998 | 0.728 | 0.580 |
| high | 365 | 50 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | net_benefit | 1.000 | 0.728 | 0.580 |
| high | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | auc | 0.952 | 0.650 | 0.619 |
| high | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | average_precision | 0.980 | 0.650 | 0.619 |
| high | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | brier_skill_updated_rate | 0.986 | 0.650 | 0.619 |
| high | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | brier_skill_oracle | 0.986 | 0.650 | 0.619 |
| high | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | ppv | 0.798 | 0.650 | 0.619 |
| high | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | pooled | net_benefit | 0.802 | 0.650 | 0.619 |
| high | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | auc | 0.840 | 0.650 | 0.619 |
| high | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | average_precision | 0.938 | 0.650 | 0.619 |
| high | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | brier_skill_population_rate | 0.986 | 0.650 | 0.619 |
| high | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | ppv | 0.000 | 0.650 | 0.619 |
| high | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | naive | net_benefit | 0.802 | 0.650 | 0.619 |
| high | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | auc | 1.000 | 0.650 | 0.619 |
| high | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | average_precision | 0.986 | 0.650 | 0.619 |
| high | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | brier_skill_updated_rate | 1.000 | 0.650 | 0.619 |
| high | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | brier_skill_oracle | 1.000 | 0.650 | 0.619 |
| high | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | ppv | 0.556 | 0.650 | 0.619 |
| high | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | net_benefit | 1.000 | 0.650 | 0.619 |
| high | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | auc | 1.000 | 0.650 | 0.619 |
| high | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | average_precision | 0.722 | 0.650 | 0.619 |
| high | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | brier_skill_updated_rate | 1.000 | 0.650 | 0.619 |
| high | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | brier_skill_oracle | 1.000 | 0.650 | 0.619 |
| high | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | ppv | 0.560 | 0.650 | 0.619 |
| high | 365 | 50 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | net_benefit | 1.000 | 0.650 | 0.619 |
| high | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | auc | 1.000 | 0.728 | 0.580 |
| high | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | average_precision | 1.000 | 0.728 | 0.580 |
| high | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | brier_skill_updated_rate | 1.000 | 0.728 | 0.580 |
| high | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | brier_skill_oracle | 1.000 | 0.728 | 0.580 |
| high | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | ppv | 1.000 | 0.728 | 0.580 |
| high | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | pooled | net_benefit | 1.000 | 0.728 | 0.580 |
| high | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | auc | 1.000 | 0.728 | 0.580 |
| high | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | average_precision | 1.000 | 0.728 | 0.580 |
| high | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | brier_skill_population_rate | 1.000 | 0.728 | 0.580 |
| high | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | ppv | 0.000 | 0.728 | 0.580 |
| high | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | naive | net_benefit | 1.000 | 0.728 | 0.580 |
| high | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | auc | 1.000 | 0.728 | 0.580 |
| high | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | average_precision | 1.000 | 0.728 | 0.580 |
| high | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | brier_skill_updated_rate | 1.000 | 0.728 | 0.580 |
| high | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | brier_skill_oracle | 1.000 | 0.728 | 0.580 |
| high | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | ppv | 1.000 | 0.728 | 0.580 |
| high | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored_excess | net_benefit | 1.000 | 0.728 | 0.580 |
| high | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | auc | 1.000 | 0.728 | 0.580 |
| high | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | average_precision | 1.000 | 0.728 | 0.580 |
| high | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | brier_skill_updated_rate | 1.000 | 0.728 | 0.580 |
| high | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | brier_skill_oracle | 1.000 | 0.728 | 0.580 |
| high | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | ppv | 1.000 | 0.728 | 0.580 |
| high | 365 | 150 | A daily rho=0.6 vs B daily rho=0.2 | tailored (B against its level-only forecast) | net_benefit | 1.000 | 0.728 | 0.580 |
| high | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | auc | 0.994 | 0.650 | 0.619 |
| high | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | average_precision | 1.000 | 0.650 | 0.619 |
| high | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | brier_skill_updated_rate | 1.000 | 0.650 | 0.619 |
| high | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | brier_skill_oracle | 1.000 | 0.650 | 0.619 |
| high | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | ppv | 0.904 | 0.650 | 0.619 |
| high | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | pooled | net_benefit | 0.912 | 0.650 | 0.619 |
| high | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | auc | 0.918 | 0.650 | 0.619 |
| high | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | average_precision | 0.980 | 0.650 | 0.619 |
| high | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | brier_skill_population_rate | 1.000 | 0.650 | 0.619 |
| high | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | ppv | 0.000 | 0.650 | 0.619 |
| high | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | naive | net_benefit | 0.912 | 0.650 | 0.619 |
| high | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | auc | 1.000 | 0.650 | 0.619 |
| high | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | average_precision | 1.000 | 0.650 | 0.619 |
| high | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | brier_skill_updated_rate | 1.000 | 0.650 | 0.619 |
| high | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | brier_skill_oracle | 1.000 | 0.650 | 0.619 |
| high | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | ppv | 0.614 | 0.650 | 0.619 |
| high | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored_excess | net_benefit | 1.000 | 0.650 | 0.619 |
| high | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | auc | 1.000 | 0.650 | 0.619 |
| high | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | average_precision | 0.772 | 0.650 | 0.619 |
| high | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | brier_skill_updated_rate | 1.000 | 0.650 | 0.619 |
| high | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | brier_skill_oracle | 1.000 | 0.650 | 0.619 |
| high | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | ppv | 0.618 | 0.650 | 0.619 |
| high | 365 | 150 | A daily rho=0.4 vs B daily rho=0.3 | tailored (B against its level-only forecast) | net_benefit | 1.000 | 0.650 | 0.619 |

## Permutation power and size checks

One-sided model-fixed within-patient AUC permutation test, alpha 0.05.

| Spread | Days | Patients | Test | Replications | Rejections | Rate | Mean observed AUC |
|---|---:|---:|---|---:|---:|---:|---:|
| low | 30 | 50 | A_main_power | 200 | 200 | 1.000 | 0.705 |
| low | 30 | 50 | A_daily_rho0_size | 200 | 9 | 0.045 | 0.162 |
| low | 30 | 50 | B_daily_rho0_size | 200 | 7 | 0.035 | 0.497 |
| low | 30 | 150 | A_main_power | 200 | 200 | 1.000 | 0.706 |
| low | 30 | 150 | A_daily_rho0_size | 200 | 12 | 0.060 | 0.162 |
| low | 30 | 150 | B_daily_rho0_size | 200 | 8 | 0.040 | 0.499 |
| low | 100 | 50 | A_main_power | 200 | 200 | 1.000 | 0.716 |
| low | 100 | 50 | A_daily_rho0_size | 200 | 14 | 0.070 | 0.298 |
| low | 100 | 50 | B_daily_rho0_size | 200 | 8 | 0.040 | 0.501 |
| low | 100 | 150 | A_main_power | 200 | 200 | 1.000 | 0.716 |
| low | 100 | 150 | A_daily_rho0_size | 200 | 10 | 0.050 | 0.300 |
| low | 100 | 150 | B_daily_rho0_size | 200 | 12 | 0.060 | 0.500 |
| low | 365 | 50 | A_main_power | 200 | 200 | 1.000 | 0.731 |
| low | 365 | 50 | A_daily_rho0_size | 200 | 9 | 0.045 | 0.398 |
| low | 365 | 50 | B_daily_rho0_size | 200 | 11 | 0.055 | 0.499 |
| low | 365 | 150 | A_main_power | 200 | 200 | 1.000 | 0.730 |
| low | 365 | 150 | A_daily_rho0_size | 200 | 7 | 0.035 | 0.398 |
| low | 365 | 150 | B_daily_rho0_size | 200 | 9 | 0.045 | 0.499 |
| high | 30 | 50 | A_main_power | 200 | 200 | 1.000 | 0.630 |
| high | 30 | 50 | A_daily_rho0_size | 200 | 17 | 0.085 | 0.325 |
| high | 30 | 50 | B_daily_rho0_size | 200 | 14 | 0.070 | 0.500 |
| high | 30 | 150 | A_main_power | 200 | 200 | 1.000 | 0.628 |
| high | 30 | 150 | A_daily_rho0_size | 200 | 6 | 0.030 | 0.320 |
| high | 30 | 150 | B_daily_rho0_size | 200 | 14 | 0.070 | 0.502 |
| high | 100 | 50 | A_main_power | 200 | 200 | 1.000 | 0.681 |
| high | 100 | 50 | A_daily_rho0_size | 200 | 17 | 0.085 | 0.388 |
| high | 100 | 50 | B_daily_rho0_size | 200 | 7 | 0.035 | 0.501 |
| high | 100 | 150 | A_main_power | 200 | 200 | 1.000 | 0.680 |
| high | 100 | 150 | A_daily_rho0_size | 200 | 12 | 0.060 | 0.388 |
| high | 100 | 150 | B_daily_rho0_size | 200 | 9 | 0.045 | 0.500 |
| high | 365 | 50 | A_main_power | 200 | 200 | 1.000 | 0.708 |
| high | 365 | 50 | A_daily_rho0_size | 200 | 11 | 0.055 | 0.430 |
| high | 365 | 50 | B_daily_rho0_size | 200 | 5 | 0.025 | 0.501 |
| high | 365 | 150 | A_main_power | 200 | 200 | 1.000 | 0.707 |
| high | 365 | 150 | A_daily_rho0_size | 200 | 14 | 0.070 | 0.432 |
| high | 365 | 150 | B_daily_rho0_size | 200 | 8 | 0.040 | 0.500 |

## Runtime

Full-profile elapsed time: **5505.9 seconds**.
