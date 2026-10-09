# v7.1 review analyses B13–B15

All analyses use the exported held-out test forecasts; no model is refit. The fixed seed is 20261009.

## B13. Pooled AUC and patient-rate references

Entries are estimates with percentile 95% intervals from paired whole-patient bootstrap resamples (2,000 by default). A0 is the pooled AUC of each patient's observed scored-period rate. Population A0 applies the S4 binomial-noise moment correction and normal approximation.

| Measure | REPLACE-BG | CTN-0094 | College Experience |
|---|---:|---:|---:|
| GB pooled AUC | 0.829 (0.800, 0.855) | 0.929 (0.914, 0.943) | 0.935 (0.904, 0.956) |
| Retrospective A0 | 0.752 (0.709, 0.785) | 0.932 (0.915, 0.946) | 0.855 (0.785, 0.902) |
| GB pooled AUC − retrospective A0 | 0.077 (0.053, 0.109) | -0.003 (-0.010, 0.005) | 0.080 (0.046, 0.135) |
| Population A0, moment corrected | 0.737 (0.692, 0.774) | 0.866 (0.851, 0.880) | 0.811 (0.764, 0.847) |
| Run-in rate pooled AUC | 0.597 (0.524, 0.659) | 0.801 (0.762, 0.838) | 0.695 (0.572, 0.798) |
| Updated rate pooled AUC | 0.707 (0.655, 0.745) | 0.889 (0.867, 0.908) | 0.842 (0.774, 0.893) |

This shows how pooled GB discrimination compares with the hindsight rate benchmark, the noise-corrected population benchmark, and patient-specific rates available when forecasts are issued.

## B14. Within-patient forecast permutation test

Outcomes stay fixed while each patient's forecasts are permuted across that patient's scored days. Cells show observed pooled AUC; null mean (2.5th, 97.5th percentiles); and one-sided plus-one p-value.

| Model | REPLACE-BG | CTN-0094 | College Experience |
|---|---:|---:|---:|
| GB | 0.829; 0.656 (0.639, 0.673); p<0.001 | 0.929; 0.884 (0.878, 0.890); p<0.001 | 0.935; 0.779 (0.772, 0.787); p<0.001 |
| LR | 0.818; 0.628 (0.608, 0.649); p<0.001 | 0.923; 0.880 (0.874, 0.885); p<0.001 | 0.934; 0.778 (0.770, 0.786); p<0.001 |

This tests whether the fitted forecasts' pooled AUC exceeds the within-patient no-order reference while preserving each patient's outcomes and forecast distribution.

## B15. Calibration-in-the-large noise reference

Shares are proportions of patients. Simulated intervals are the 2.5th and 97.5th percentiles across 1,000 outcome draws from the exported forecasts.

| Share within 0.05 of mean GB forecast | REPLACE-BG | CTN-0094 | College Experience |
|---|---:|---:|---:|
| Observed | 0.933 | 0.463 | 0.842 |
| Expected if perfectly calibrated | 0.997 (0.978, 1.000) | 0.537 (0.484, 0.589) | 0.916 (0.816, 1.000) |

This shows how much patient-level calibration-in-the-large agreement the observed record lengths can produce through binomial noise when forecasts are perfectly calibrated.

## Reproduction checks

The script asserts the exported-data pooled AUC, per-patient AUC, and retrospective A0 against the existing results, and checks that the calibration shares round to the Table 4 percentages (93%, 46%, 84%).

