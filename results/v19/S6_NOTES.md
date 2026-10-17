# S6: Monte Carlo errors and undefined counts of the ranking simulation (Programming, 15 Oct 2026)

`code/v19_ranking_mc.py` reruns the ranking part of code/ranking_sim.py (same cells, seeds, tasks and worker; 500 replications per cell, 50 shuffles; 6 workers, 67 min) and keeps the per-replication scores. All 408 stored ranking rows of open_data/results/v10/ranking_sim.json (pooled, per-patient and shuffled-baseline scoring) reproduce exactly: means, shares and valid replications (0 mismatches). Outputs: `results/v19/S6_mc.csv` (per cell, pair, scoring and measure: share A above B with its SE, mean of A and B with their SEs, SDs) and `results/v19/S6_undefined.csv` (per cell, pair and model).

## Monte Carlo errors (for the S6 text)
- Means (Table 6 and the S6 tables): SE at most 0.0016 for every AUC mean (all scorings, all 24 settings, both pairs), at most 0.0022 for any mean on any measure. So the Monte Carlo error of Table 6's means is about one unit of the third decimal or less.
- Shares of replications ranking A first (Figure 7, S6): SE = sqrt(p(1-p)/500), at most 0.022 (at p near 0.5); 0 where the share is 0 or 1. Every replication gave a defined score on every ranked measure (valid replications 500 of 500 everywhere).
- Example, closer pair, 50 patients, shuffled-baseline AUC excess, A vs B (SE): high spread 30 days 0.111 (0.001) vs 0.111 (0.001), share 0.486 (0.022); 100 days 0.128 (0.001) vs 0.115 (0.001), share 0.886 (0.014).

## Patients with an undefined per-patient score, mean per replication (max)
- AUC undefined (no event or no event-free day; the same for both models): low spread (REPLACE-BG-like) 30 days 12.2 of 50 (max 23), 100 days 1.7 (7), 365 days 0.04 (2); high spread (CTN-0094-like) 30 days 6.5 of 50 (15), 100 days 2.8 (9), 365 days 1.0 (5). With 150 patients about three times as many.
- PPV undefined (no alarm at the population-rate threshold): model A rarely (low spread at most 0.07 of 50 per replication; high spread 1.5 to 8.1 of 50). Model B often, because it knows each patient's level and never alarms for patients with a low level: closer pair, 50 patients, low spread 2.3 to 7.6, high spread 17.3 to 20.7; wide pair, low spread 6.6 to 12.6, high spread 21.0 to 23.5. So model B's per-patient PPV averages over a subset of patients (in high spread, 53% to 65% of them at 50 patients).
- Simulation alarm threshold: the population event rate (spec rate), for both models and all scorings.
