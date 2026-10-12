# B2. Baselines for the rate-only forecasts

For each dataset and forecast, cells give the equal-patient score; shuffled-rate scores also give the 2.5th to 97.5th percentiles across draws. `eligible_n` is the minimum-to-maximum patient count across score and independent shuffled-baseline draws (a single count when fixed).

Usual baselines are AUC 0.5; AP at both the scored-set event rate and the training-cohort rate; sensitivity at the training-cohort rate; specificity at its complement; PPV/NPV at the training-cohort event/non-event rates; net benefit of the population-rate treat-all or treat-none policy; Brier skill 0; and slope 1. Within-patient baselines use the same eligible patients: AUC 0.5, strict-random-ranking AP, Youden 0, own event/non-event rates for PPV/NPV, for net benefit the policy of alarming on all days or on none (selected by the patient's own scored rate, which is the better of the two in hindsight, or by the forecast's level, its mean over the scored time points), and the own-rate-only forecast's Brier skill against the training-cohort rate. Sensitivity and specificity have no within-patient baseline; the level-only conditional slope baseline is 0 even where the run-in slope is not estimable. For shuffled-rate scores, each independent shuffle keeps run-in outcomes fixed and rebuilds the expanding rate. The independent shuffled-baseline draws use a separate seed; score seed 20261012, baseline seed 20261013, 200 draws each.

The repository's `withinpatient.nine.average_precision` groups tied forecast values at one threshold; a constant forecast therefore has AP equal to the patient's scored-set event rate n/T. The within-patient AP baseline instead uses the exact expected AP of a uniformly random strict ranking.

## CTN-0094

| Forecast | Measure | Eligible n | Score | Usual baseline | Within-patient baseline | Own rate | Shuffled baseline | Score - within | Score - shuffled |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| runin_rate | AUC | 154 | 0.500 | 0.500 | 0.500 | 0.424 | 0.500 (0.500) | 0.000 | 0.000 |
| runin_rate | Average precision (patients with both outcomes) | 154 | 0.424 | 0.300 (scored); 0.320 (training) | 0.511 | 0.424 | 0.424 (0.424) | -0.087 | -0.000 |
| runin_rate | Average precision (patients with an event) | 168 | 0.472 | 0.300 (scored); 0.320 (training) | 0.552 | 0.472 | 0.472 (0.472) | -0.080 | -0.000 |
| runin_rate | Sensitivity + specificity - 1 | 154 | 0.000 | 0.000 | 0.000 | 0.424 | 0.000 (0.000) | 0.000 | 0.000 |
| runin_rate | Sensitivity | 168 | 0.679 | 0.320 | n/a | 0.472 | 0.679 (0.679) | n/a | 0.000 |
| runin_rate | Specificity | 232 | 0.526 | 0.680 | n/a | 0.282 | 0.526 (0.526) | n/a | 0.000 |
| runin_rate | PPV | 124 | 0.525 | 0.320 | 0.525 | 0.525 | 0.525 (0.525) | 0.000 | 0.000 |
| runin_rate | NPV | 122 | 0.883 | 0.680 | 0.883 | 0.117 | 0.883 (0.883) | 0.000 | 0.000 |
| runin_rate | Net benefit (baseline: all or none by own rate) | 246 | 0.152 | 0.000 | 0.225 | 0.323 | 0.152 (0.152) | -0.073 | -0.000 |
| runin_rate | Net benefit (baseline: all or none by the forecast's level) | 246 | 0.152 | 0.000 | 0.152 | 0.323 | 0.152 (0.152) | 0.000 | -0.000 |
| runin_rate | Brier skill | 246 | 0.143 | 0.000 | 0.552 | 0.323 | 0.143 (0.143) | -0.409 | 0.000 |
| runin_rate | Calibration slope | 0 | n/a | 1.000 | 0.000 | n/a | n/a | n/a | n/a |
| shuffled_rate_so_far | AUC | 154 | 0.290 (0.267 to 0.315) | 0.500 | 0.500 | 0.424 | 0.290 (0.262 to 0.316) | -0.210 | 0.000 |
| shuffled_rate_so_far | Average precision (patients with both outcomes) | 154 | 0.400 (0.391 to 0.409) | 0.300 (scored); 0.320 (training) | 0.511 | 0.424 | 0.400 (0.390 to 0.413) | -0.112 | -0.000 |
| shuffled_rate_so_far | Average precision (patients with an event) | 168 | 0.450 (0.442 to 0.458) | 0.300 (scored); 0.320 (training) | 0.552 | 0.472 | 0.450 (0.441 to 0.462) | -0.102 | -0.000 |
| shuffled_rate_so_far | Sensitivity + specificity - 1 | 154 | -0.044 (-0.062 to -0.023) | 0.000 | 0.000 | 0.424 | -0.044 (-0.061 to -0.024) | -0.044 | -0.000 |
| shuffled_rate_so_far | Sensitivity | 168 | 0.594 (0.572 to 0.614) | 0.320 | n/a | 0.472 | 0.594 (0.572 to 0.617) | n/a | -0.000 |
| shuffled_rate_so_far | Specificity | 232 | 0.595 (0.583 to 0.606) | 0.680 | n/a | 0.282 | 0.595 (0.582 to 0.607) | n/a | 0.000 |
| shuffled_rate_so_far | PPV | 140 to 152 | 0.484 (0.470 to 0.499) | 0.320 | 0.515 | 0.515 | 0.484 (0.470 to 0.498) | -0.031 | -0.000 |
| shuffled_rate_so_far | NPV | 162 to 173 | 0.836 (0.817 to 0.855) | 0.680 | 0.873 | 0.127 | 0.837 (0.817 to 0.856) | -0.037 | -0.000 |
| shuffled_rate_so_far | Net benefit (baseline: all or none by own rate) | 246 | 0.199 (0.196 to 0.202) | 0.000 | 0.225 | 0.323 | 0.199 (0.197 to 0.202) | -0.025 | 0.000 |
| shuffled_rate_so_far | Net benefit (baseline: all or none by the forecast's level) | 246 | 0.199 (0.196 to 0.202) | 0.000 | 0.210 | 0.323 | 0.199 (0.197 to 0.202) | -0.011 | 0.000 |
| shuffled_rate_so_far | Brier skill | 246 | 0.426 (0.424 to 0.428) | 0.000 | 0.552 | 0.323 | 0.426 (0.424 to 0.428) | -0.127 | -0.000 |
| shuffled_rate_so_far | Calibration slope | 154 | -0.068 (-0.096 to -0.041) | 1.000 | 0.000 | 0.424 | -0.068 (-0.099 to -0.038) | -0.068 | -0.000 |

## REPLACE-BG

| Forecast | Measure | Eligible n | Score | Usual baseline | Within-patient baseline | Own rate | Shuffled baseline | Score - within | Score - shuffled |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| runin_rate | AUC | 42 | 0.500 | 0.500 | 0.500 | 0.065 | 0.500 (0.500) | 0.000 | 0.000 |
| runin_rate | Average precision (patients with both outcomes) | 42 | 0.065 | 0.060 (scored); 0.055 (training) | 0.088 | 0.065 | 0.065 (0.065) | -0.023 | 0.000 |
| runin_rate | Average precision (patients with an event) | 42 | 0.065 | 0.060 (scored); 0.055 (training) | 0.088 | 0.065 | 0.065 (0.065) | -0.023 | 0.000 |
| runin_rate | Sensitivity + specificity - 1 | 42 | 0.000 | 0.000 | 0.000 | 0.065 | 0.000 (0.000) | 0.000 | 0.000 |
| runin_rate | Sensitivity | 42 | 0.548 | 0.055 | n/a | 0.065 | 0.548 (0.548) | n/a | 0.000 |
| runin_rate | Specificity | 45 | 0.489 | 0.945 | n/a | 0.061 | 0.489 (0.489) | n/a | 0.000 |
| runin_rate | PPV | 23 | 0.085 | 0.055 | 0.085 | 0.085 | 0.085 (0.085) | 0.000 | 0.000 |
| runin_rate | NPV | 22 | 0.964 | 0.945 | 0.964 | 0.036 | 0.964 (0.964) | 0.000 | 0.000 |
| runin_rate | Net benefit (baseline: all or none by own rate) | 45 | 0.016 | 0.000 | 0.025 | 0.061 | 0.016 (0.016) | -0.009 | -0.000 |
| runin_rate | Net benefit (baseline: all or none by the forecast's level) | 45 | 0.016 | 0.000 | 0.016 | 0.061 | 0.016 (0.016) | 0.000 | -0.000 |
| runin_rate | Brier skill | 45 | -0.067 | 0.000 | 0.049 | 0.061 | -0.067 (-0.067) | -0.117 | -0.000 |
| runin_rate | Calibration slope | 0 | n/a | 1.000 | 0.000 | n/a | n/a | n/a | n/a |
| shuffled_rate_so_far | AUC | 42 | 0.372 (0.340 to 0.406) | 0.500 | 0.500 | 0.065 | 0.374 (0.344 to 0.405) | -0.128 | -0.001 |
| shuffled_rate_so_far | Average precision (patients with both outcomes) | 42 | 0.059 (0.055 to 0.062) | 0.060 (scored); 0.055 (training) | 0.088 | 0.065 | 0.059 (0.056 to 0.062) | -0.029 | -0.000 |
| shuffled_rate_so_far | Average precision (patients with an event) | 42 | 0.059 (0.055 to 0.062) | 0.060 (scored); 0.055 (training) | 0.088 | 0.065 | 0.059 (0.056 to 0.062) | -0.029 | -0.000 |
| shuffled_rate_so_far | Sensitivity + specificity - 1 | 42 | -0.029 (-0.052 to -0.007) | 0.000 | 0.000 | 0.065 | -0.027 (-0.048 to -0.004) | -0.029 | -0.002 |
| shuffled_rate_so_far | Sensitivity | 42 | 0.437 (0.395 to 0.482) | 0.055 | n/a | 0.065 | 0.439 (0.404 to 0.485) | n/a | -0.001 |
| shuffled_rate_so_far | Specificity | 45 | 0.564 (0.528 to 0.598) | 0.945 | n/a | 0.061 | 0.565 (0.536 to 0.594) | n/a | -0.000 |
| shuffled_rate_so_far | PPV | 27 to 35 | 0.072 (0.064 to 0.079) | 0.055 | 0.080 | 0.080 | 0.072 (0.065 to 0.079) | -0.008 | -0.000 |
| shuffled_rate_so_far | NPV | 33 to 40 | 0.924 (0.880 to 0.954) | 0.945 | 0.956 | 0.044 | 0.927 (0.881 to 0.954) | -0.032 | -0.003 |
| shuffled_rate_so_far | Net benefit (baseline: all or none by own rate) | 45 | 0.022 (0.021 to 0.022) | 0.000 | 0.025 | 0.061 | 0.022 (0.021 to 0.022) | -0.004 | -0.000 |
| shuffled_rate_so_far | Net benefit (baseline: all or none by the forecast's level) | 45 | 0.022 (0.021 to 0.022) | 0.000 | 0.024 | 0.061 | 0.022 (0.021 to 0.022) | -0.002 | -0.000 |
| shuffled_rate_so_far | Brier skill | 45 | 0.029 (0.028 to 0.030) | 0.000 | 0.049 | 0.061 | 0.029 (0.028 to 0.030) | -0.020 | 0.000 |
| shuffled_rate_so_far | Calibration slope | 42 | -0.041 (-0.094 to -0.000) | 1.000 | 0.000 | 0.065 | -0.040 (-0.091 to 0.010) | -0.041 | -0.001 |

## College Experience Study

| Forecast | Measure | Eligible n | Score | Usual baseline | Within-patient baseline | Own rate | Shuffled baseline | Score - within | Score - shuffled |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| runin_rate | AUC | 37 | 0.500 | 0.500 | 0.500 | 0.455 | 0.500 (0.500) | 0.000 | 0.000 |
| runin_rate | Average precision (patients with both outcomes) | 37 | 0.455 | 0.399 (scored); 0.394 (training) | 0.471 | 0.455 | 0.455 (0.455) | -0.016 | 0.000 |
| runin_rate | Average precision (patients with an event) | 37 | 0.455 | 0.399 (scored); 0.394 (training) | 0.471 | 0.455 | 0.455 (0.455) | -0.016 | 0.000 |
| runin_rate | Sensitivity + specificity - 1 | 37 | 0.000 | 0.000 | 0.000 | 0.455 | 0.000 (0.000) | 0.000 | 0.000 |
| runin_rate | Sensitivity | 37 | 0.378 | 0.394 | n/a | 0.455 | 0.378 (0.378) | n/a | 0.000 |
| runin_rate | Specificity | 38 | 0.632 | 0.606 | n/a | 0.443 | 0.632 (0.632) | n/a | -0.000 |
| runin_rate | PPV | 14 | 0.620 | 0.394 | 0.620 | 0.620 | 0.620 (0.620) | 0.000 | -0.000 |
| runin_rate | NPV | 24 | 0.660 | 0.606 | 0.660 | 0.340 | 0.660 (0.660) | 0.000 | 0.000 |
| runin_rate | Net benefit (baseline: all or none by own rate) | 38 | 0.138 | 0.000 | 0.263 | 0.443 | 0.138 (0.138) | -0.125 | -0.000 |
| runin_rate | Net benefit (baseline: all or none by the forecast's level) | 38 | 0.138 | 0.000 | 0.138 | 0.443 | 0.138 (0.138) | 0.000 | -0.000 |
| runin_rate | Brier skill | 38 | 0.053 | 0.000 | 0.408 | 0.443 | 0.053 (0.053) | -0.355 | -0.000 |
| runin_rate | Calibration slope | 0 | n/a | 1.000 | 0.000 | n/a | n/a | n/a | n/a |
| shuffled_rate_so_far | AUC | 37 | 0.413 (0.386 to 0.442) | 0.500 | 0.500 | 0.455 | 0.411 (0.386 to 0.446) | -0.087 | 0.001 |
| shuffled_rate_so_far | Average precision (patients with both outcomes) | 37 | 0.431 (0.422 to 0.441) | 0.399 (scored); 0.394 (training) | 0.471 | 0.455 | 0.431 (0.422 to 0.439) | -0.040 | 0.001 |
| shuffled_rate_so_far | Average precision (patients with an event) | 37 | 0.431 (0.422 to 0.441) | 0.399 (scored); 0.394 (training) | 0.471 | 0.455 | 0.431 (0.422 to 0.439) | -0.040 | 0.001 |
| shuffled_rate_so_far | Sensitivity + specificity - 1 | 37 | -0.004 (-0.019 to 0.012) | 0.000 | 0.000 | 0.455 | -0.006 (-0.018 to 0.012) | -0.004 | 0.002 |
| shuffled_rate_so_far | Sensitivity | 37 | 0.491 (0.468 to 0.517) | 0.394 | n/a | 0.455 | 0.491 (0.470 to 0.518) | n/a | -0.000 |
| shuffled_rate_so_far | Specificity | 38 | 0.518 (0.492 to 0.544) | 0.606 | n/a | 0.443 | 0.516 (0.491 to 0.542) | n/a | 0.002 |
| shuffled_rate_so_far | PPV | 23 to 25 | 0.618 (0.590 to 0.637) | 0.394 | 0.625 | 0.625 | 0.616 (0.589 to 0.630) | -0.007 | 0.002 |
| shuffled_rate_so_far | NPV | 29 to 32 | 0.619 (0.585 to 0.650) | 0.606 | 0.647 | 0.353 | 0.617 (0.582 to 0.656) | -0.028 | 0.001 |
| shuffled_rate_so_far | Net benefit (baseline: all or none by own rate) | 38 | 0.243 (0.241 to 0.245) | 0.000 | 0.263 | 0.443 | 0.243 (0.241 to 0.244) | -0.020 | 0.000 |
| shuffled_rate_so_far | Net benefit (baseline: all or none by the forecast's level) | 38 | 0.243 (0.241 to 0.245) | 0.000 | 0.256 | 0.443 | 0.243 (0.241 to 0.244) | -0.013 | 0.000 |
| shuffled_rate_so_far | Brier skill | 38 | 0.351 (0.351 to 0.351) | 0.000 | 0.408 | 0.443 | 0.351 (0.351 to 0.351) | -0.057 | 0.000 |
| shuffled_rate_so_far | Calibration slope | 37 | -0.065 (-0.195 to 0.031) | 1.000 | 0.000 | 0.455 | -0.073 (-0.177 to 0.017) | -0.065 | 0.008 |

