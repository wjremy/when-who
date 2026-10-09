# Real-time patient-grid discrimination

## 10-line summary
1. Exported forecast rows were matched exactly to reconstructed held-out panel rows by anonymous patient and outcome sequence in all cohorts.
2. REPLACE-BG follow-up time is the daily `day` offset from enrollment; no calendar date is present in the forecast panel.
3. CTN follow-up time is `study_week`, where week 0 begins at the first randomization study day; no calendar date is present in the weekly panel.
4. CES supports calendar-date grouping and elapsed follow-up week from each participant’s first complete report.
5. Real-time columns retain only clock values with both outcome classes and at least 5, 10, or 20 distinct patients, respectively.
6. Column mean is the equal-weight mean of retained time-specific AUCs; pair-weighted mean weights each column by its event/non-event cell pairs.
7. Pair coverage is the fraction of all pooled event/non-event cell pairs whose two cells share a retained clock value.
8. Confidence intervals are patient-cluster bootstrap percentile intervals (1,000 replicates; seed 20261007).
9. Pooled AUC decomposes exactly into same-patient, same-clock/different-patient, and neither pair strata; ties count 0.5.
10. Whether pooled AUC remains close to same-time column AUC depends on cohort and forecast; see the comparison below.

## Alignment check

| Cohort | Panel rows | Forecast rows | Panel patients | Forecast patients | Patient labels rowwise equal | y sequence rowwise equal |
|---|---:|---:|---:|---:|:---:|:---:|
| REPLACE-BG | 9477 | 9477 | 45 | 45 | PASS | PASS |
| CTN | 4237 | 4237 | 246 | 246 | PASS | PASS |
| CES | 6361 | 6361 | 38 | 38 | PASS | PASS |

Checks compare the full ordered anonymous patient-label vector and the full ordered binary outcome vector, row by row. Forecast scores were attached only after both checks passed.

## Results by cohort and clock

### REPLACE-BG

#### Follow-up

| Forecast | Min patients | Equal-weight mean [95% bootstrap CI] | Pair-weighted mean | Columns | Event/non-event pair coverage |
|---|---:|---:|---:|---:|---:|
| gradient_boosting | 5 | 0.830 [0.798, 0.858] | 0.829 | 233 | 0.37% |
| gradient_boosting | 10 | 0.831 [0.796, 0.857] | 0.829 | 228 | 0.37% |
| gradient_boosting | 20 | 0.823 [0.790, 0.854] | 0.828 | 208 | 0.36% |
| runin_rate | 5 | 0.585 [0.516, 0.658] | 0.597 | 233 | 0.37% |
| runin_rate | 10 | 0.586 [0.515, 0.658] | 0.597 | 228 | 0.37% |
| runin_rate | 20 | 0.578 [0.511, 0.656] | 0.596 | 208 | 0.36% |
| updating_rate | 5 | 0.710 [0.656, 0.761] | 0.712 | 233 | 0.37% |
| updating_rate | 10 | 0.712 [0.655, 0.763] | 0.711 | 228 | 0.37% |
| updating_rate | 20 | 0.708 [0.652, 0.758] | 0.711 | 208 | 0.36% |
| retrospective_rate | 5 | 0.758 [0.709, 0.806] | 0.756 | 233 | 0.37% |
| retrospective_rate | 10 | 0.761 [0.709, 0.807] | 0.756 | 228 | 0.37% |
| retrospective_rate | 20 | 0.758 [0.705, 0.804] | 0.755 | 208 | 0.36% |

Pair-composition identity for this clock: pooled AUC = Σ(pair share × stratum concordance). Shares are calculated over all pooled event/non-event pairs.

| Forecast | Stratum | Pairs | Pair share | Concordance |
|---|---|---:|---:|---:|
| gradient_boosting | Same patient | 107,640 | 2.13% | 0.763062 |
| gradient_boosting | Same clock, different patients | 18,506 | 0.37% | 0.828839 |
| gradient_boosting | Neither | 4,925,824 | 97.50% | 0.829996 |
| gradient_boosting | **Identity check** | 5,051,970 total pairs | Σshare=1 | 0.828565688236 vs pooled 0.828565688236; Δ=+0.00e+00 |
| runin_rate | Same patient | 107,640 | 2.13% | 0.500000 |
| runin_rate | Same clock, different patients | 18,506 | 0.37% | 0.597266 |
| runin_rate | Neither | 4,925,824 | 97.50% | 0.599229 |
| runin_rate | **Identity check** | 5,051,970 total pairs | Σshare=1 | 0.597107366037 vs pooled 0.597107366037; Δ=-1.11e-16 |
| updating_rate | Same patient | 107,640 | 2.13% | 0.409894 |
| updating_rate | Same clock, different patients | 18,506 | 0.37% | 0.711580 |
| updating_rate | Neither | 4,925,824 | 97.50% | 0.713110 |
| updating_rate | **Identity check** | 5,051,970 total pairs | Σshare=1 | 0.706644140801 vs pooled 0.706644140801; Δ=+1.11e-16 |
| retrospective_rate | Same patient | 107,640 | 2.13% | 0.500000 |
| retrospective_rate | Same clock, different patients | 18,506 | 0.37% | 0.755809 |
| retrospective_rate | Neither | 4,925,824 | 97.50% | 0.757296 |
| retrospective_rate | **Identity check** | 5,051,970 total pairs | Σshare=1 | 0.751808700368 vs pooled 0.751808700368; Δ=-1.11e-16 |

### CTN

#### Follow-up

| Forecast | Min patients | Equal-weight mean [95% bootstrap CI] | Pair-weighted mean | Columns | Event/non-event pair coverage |
|---|---:|---:|---:|---:|---:|
| gradient_boosting | 5 | 0.918 [0.893, 0.942] | 0.930 | 33 | 4.24% |
| gradient_boosting | 10 | 0.929 [0.906, 0.943] | 0.930 | 29 | 4.24% |
| gradient_boosting | 20 | 0.926 [0.910, 0.943] | 0.930 | 28 | 4.24% |
| runin_rate | 5 | 0.765 [0.726, 0.811] | 0.810 | 33 | 4.24% |
| runin_rate | 10 | 0.771 [0.731, 0.813] | 0.810 | 29 | 4.24% |
| runin_rate | 20 | 0.773 [0.734, 0.821] | 0.810 | 28 | 4.24% |
| updating_rate | 5 | 0.865 [0.843, 0.901] | 0.893 | 33 | 4.24% |
| updating_rate | 10 | 0.888 [0.857, 0.910] | 0.893 | 29 | 4.24% |
| updating_rate | 20 | 0.889 [0.861, 0.910] | 0.893 | 28 | 4.24% |
| retrospective_rate | 5 | 0.921 [0.900, 0.947] | 0.935 | 33 | 4.24% |
| retrospective_rate | 10 | 0.933 [0.910, 0.949] | 0.935 | 29 | 4.24% |
| retrospective_rate | 20 | 0.935 [0.915, 0.949] | 0.935 | 28 | 4.24% |

Pair-composition identity for this clock: pooled AUC = Σ(pair share × stratum concordance). Shares are calculated over all pooled event/non-event pairs.

| Forecast | Stratum | Pairs | Pair share | Concordance |
|---|---|---:|---:|---:|
| gradient_boosting | Same patient | 7,156 | 0.19% | 0.663150 |
| gradient_boosting | Same clock, different patients | 159,994 | 4.24% | 0.930326 |
| gradient_boosting | Neither | 3,602,636 | 95.57% | 0.929746 |
| gradient_boosting | **Identity check** | 3,769,786 total pairs | Σshare=1 | 0.929264287151 vs pooled 0.929264287151; Δ=+0.00e+00 |
| runin_rate | Same patient | 7,156 | 0.19% | 0.500000 |
| runin_rate | Same clock, different patients | 159,994 | 4.24% | 0.810224 |
| runin_rate | Neither | 3,602,636 | 95.57% | 0.801190 |
| runin_rate | **Identity check** | 3,769,786 total pairs | Σshare=1 | 0.801002099323 vs pooled 0.801002099323; Δ=+0.00e+00 |
| updating_rate | Same patient | 7,156 | 0.19% | 0.475545 |
| updating_rate | Same clock, different patients | 159,994 | 4.24% | 0.893162 |
| updating_rate | Neither | 3,602,636 | 95.57% | 0.889665 |
| updating_rate | **Identity check** | 3,769,786 total pairs | Σshare=1 | 0.889027520395 vs pooled 0.889027520395; Δ=+0.00e+00 |
| retrospective_rate | Same patient | 7,156 | 0.19% | 0.500000 |
| retrospective_rate | Same clock, different patients | 159,994 | 4.24% | 0.934916 |
| retrospective_rate | Neither | 3,602,636 | 95.57% | 0.932964 |
| retrospective_rate | **Identity check** | 3,769,786 total pairs | Σshare=1 | 0.932225065295 vs pooled 0.932225065295; Δ=-1.11e-16 |

### CES

#### Calendar date

| Forecast | Min patients | Equal-weight mean [95% bootstrap CI] | Pair-weighted mean | Columns | Event/non-event pair coverage |
|---|---:|---:|---:|---:|---:|
| gradient_boosting | 5 | 0.942 [0.917, 0.964] | 0.946 | 579 | 0.06% |
| gradient_boosting | 10 | 0.947 [0.914, 0.975] | 0.946 | 49 | 0.01% |
| gradient_boosting | 20 | NA [NA, NA] | NA | 0 | 0.00% |
| runin_rate | 5 | 0.715 [0.587, 0.819] | 0.723 | 579 | 0.06% |
| runin_rate | 10 | 0.761 [0.586, 0.855] | 0.771 | 49 | 0.01% |
| runin_rate | 20 | NA [NA, NA] | NA | 0 | 0.00% |
| updating_rate | 5 | 0.868 [0.804, 0.917] | 0.876 | 579 | 0.06% |
| updating_rate | 10 | 0.897 [0.812, 0.941] | 0.899 | 49 | 0.01% |
| updating_rate | 20 | NA [NA, NA] | NA | 0 | 0.00% |
| retrospective_rate | 5 | 0.881 [0.820, 0.926] | 0.882 | 579 | 0.06% |
| retrospective_rate | 10 | 0.899 [0.799, 0.943] | 0.896 | 49 | 0.01% |
| retrospective_rate | 20 | NA [NA, NA] | NA | 0 | 0.00% |

Pair-composition identity for this clock: pooled AUC = Σ(pair share × stratum concordance). Shares are calculated over all pooled event/non-event pairs.

| Forecast | Stratum | Pairs | Pair share | Concordance |
|---|---|---:|---:|---:|
| gradient_boosting | Same patient | 193,309 | 1.99% | 0.851900 |
| gradient_boosting | Same clock, different patients | 7,250 | 0.07% | 0.944000 |
| gradient_boosting | Neither | 9,500,929 | 97.93% | 0.936938 |
| gradient_boosting | **Identity check** | 9,701,488 total pairs | Σshare=1 | 0.935248489716 vs pooled 0.935248489716; Δ=-1.11e-16 |
| runin_rate | Same patient | 193,309 | 1.99% | 0.500000 |
| runin_rate | Same clock, different patients | 7,250 | 0.07% | 0.716345 |
| runin_rate | Neither | 9,500,929 | 97.93% | 0.699013 |
| runin_rate | **Identity check** | 9,701,488 total pairs | Σshare=1 | 0.695060231997 vs pooled 0.695060231997; Δ=-1.11e-16 |
| updating_rate | Same patient | 193,309 | 1.99% | 0.649106 |
| updating_rate | Same clock, different patients | 7,250 | 0.07% | 0.871310 |
| updating_rate | Neither | 9,500,929 | 97.93% | 0.846381 |
| updating_rate | **Identity check** | 9,701,488 total pairs | Σshare=1 | 0.842469165555 vs pooled 0.842469165555; Δ=+0.00e+00 |
| retrospective_rate | Same patient | 193,309 | 1.99% | 0.500000 |
| retrospective_rate | Same clock, different patients | 7,250 | 0.07% | 0.879724 |
| retrospective_rate | Neither | 9,500,929 | 97.93% | 0.862125 |
| retrospective_rate | **Identity check** | 9,701,488 total pairs | Σshare=1 | 0.854922822149 vs pooled 0.854922822149; Δ=+0.00e+00 |

#### Follow-up week

| Forecast | Min patients | Equal-weight mean [95% bootstrap CI] | Pair-weighted mean | Columns | Event/non-event pair coverage |
|---|---:|---:|---:|---:|---:|
| gradient_boosting | 5 | 0.933 [0.902, 0.949] | 0.934 | 184 | 0.58% |
| gradient_boosting | 10 | 0.933 [0.901, 0.951] | 0.934 | 180 | 0.58% |
| gradient_boosting | 20 | 0.934 [0.899, 0.953] | 0.933 | 119 | 0.50% |
| runin_rate | 5 | 0.688 [0.570, 0.801] | 0.711 | 184 | 0.58% |
| runin_rate | 10 | 0.689 [0.572, 0.806] | 0.712 | 180 | 0.58% |
| runin_rate | 20 | 0.703 [0.584, 0.813] | 0.716 | 119 | 0.50% |
| updating_rate | 5 | 0.835 [0.764, 0.889] | 0.847 | 184 | 0.58% |
| updating_rate | 10 | 0.841 [0.764, 0.892] | 0.847 | 180 | 0.58% |
| updating_rate | 20 | 0.845 [0.777, 0.896] | 0.848 | 119 | 0.50% |
| retrospective_rate | 5 | 0.861 [0.799, 0.909] | 0.862 | 184 | 0.58% |
| retrospective_rate | 10 | 0.866 [0.802, 0.913] | 0.863 | 180 | 0.58% |
| retrospective_rate | 20 | 0.872 [0.810, 0.919] | 0.863 | 119 | 0.50% |

Pair-composition identity for this clock: pooled AUC = Σ(pair share × stratum concordance). Shares are calculated over all pooled event/non-event pairs.

| Forecast | Stratum | Pairs | Pair share | Concordance |
|---|---|---:|---:|---:|
| gradient_boosting | Same patient | 193,309 | 1.99% | 0.851900 |
| gradient_boosting | Same clock, different patients | 56,323 | 0.58% | 0.937912 |
| gradient_boosting | Neither | 9,451,856 | 97.43% | 0.936937 |
| gradient_boosting | **Identity check** | 9,701,488 total pairs | Σshare=1 | 0.935248489716 vs pooled 0.935248489716; Δ=+0.00e+00 |
| runin_rate | Same patient | 193,309 | 1.99% | 0.500000 |
| runin_rate | Same clock, different patients | 56,323 | 0.58% | 0.712897 |
| runin_rate | Neither | 9,451,856 | 97.43% | 0.698943 |
| runin_rate | **Identity check** | 9,701,488 total pairs | Σshare=1 | 0.695060231997 vs pooled 0.695060231997; Δ=+1.11e-16 |
| updating_rate | Same patient | 193,309 | 1.99% | 0.649106 |
| updating_rate | Same clock, different patients | 56,323 | 0.58% | 0.852645 |
| updating_rate | Neither | 9,451,856 | 97.43% | 0.846363 |
| updating_rate | **Identity check** | 9,701,488 total pairs | Σshare=1 | 0.842469165555 vs pooled 0.842469165555; Δ=+0.00e+00 |
| retrospective_rate | Same patient | 193,309 | 1.99% | 0.500000 |
| retrospective_rate | Same clock, different patients | 56,323 | 0.58% | 0.864904 |
| retrospective_rate | Neither | 9,451,856 | 97.43% | 0.862122 |
| retrospective_rate | **Identity check** | 9,701,488 total pairs | Σshare=1 | 0.854922822149 vs pooled 0.854922822149; Δ=+1.11e-16 |

## Comparison with observation-index columns

For the real-time comparison, “close” means an absolute difference no larger than 0.03 between pooled AUC and the equal-weight column mean at minimum column size 5. The out_grid observation-index column mean is recalculated directly from each supplied forecast file using `t` and the same both-class/minimum-five rule.

| Cohort | Forecast | Pooled AUC | Observation-index columns | Real-time clock | Real-time columns | Difference pooled − real-time | Close under real time? |
|---|---|---:|---:|---|---:|---:|:---:|
| REPLACE-BG | gradient_boosting | 0.829 | 0.831 | Follow-up | 0.830 | -0.002 | Yes |
| REPLACE-BG | runin_rate | 0.597 | 0.594 | Follow-up | 0.585 | +0.012 | Yes |
| REPLACE-BG | updating_rate | 0.707 | 0.709 | Follow-up | 0.710 | -0.003 | Yes |
| REPLACE-BG | retrospective_rate | 0.752 | 0.747 | Follow-up | 0.758 | -0.006 | Yes |
| CTN | gradient_boosting | 0.929 | 0.930 | Follow-up | 0.918 | +0.011 | Yes |
| CTN | runin_rate | 0.801 | 0.782 | Follow-up | 0.765 | +0.036 | No |
| CTN | updating_rate | 0.889 | 0.872 | Follow-up | 0.865 | +0.024 | Yes |
| CTN | retrospective_rate | 0.932 | 0.926 | Follow-up | 0.921 | +0.011 | Yes |
| CES | gradient_boosting | 0.935 | 0.945 | Calendar date | 0.942 | -0.007 | Yes |
| CES | gradient_boosting | 0.935 | 0.945 | Follow-up week | 0.933 | +0.002 | Yes |
| CES | runin_rate | 0.695 | 0.647 | Calendar date | 0.715 | -0.020 | Yes |
| CES | runin_rate | 0.695 | 0.647 | Follow-up week | 0.688 | +0.007 | Yes |
| CES | updating_rate | 0.842 | 0.836 | Calendar date | 0.868 | -0.025 | Yes |
| CES | updating_rate | 0.842 | 0.836 | Follow-up week | 0.835 | +0.007 | Yes |
| CES | retrospective_rate | 0.855 | 0.883 | Calendar date | 0.881 | -0.026 | Yes |
| CES | retrospective_rate | 0.855 | 0.883 | Follow-up week | 0.861 | -0.006 | Yes |

Using the stated 0.03 absolute-difference rule, pooled AUC remains close to the gradient-boosting equal-weight column AUC under every real-time clock in all three cohorts. The rate-only results are also close in almost every case, but CTN run-in rate differs by 0.036, so the statement does not hold uniformly for rate-only forecasts.

## Reproduction

Run from the repository root; inputs are read from this script’s directory and this report is written to `open_data/results/realtime_analysis/report.md`.

```bash
python open_data/realtime_analysis.py
```
