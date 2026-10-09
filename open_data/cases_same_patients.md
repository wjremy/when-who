# Pooled metrics on the patients who enter the within-patient metrics (gradient-boosting model, test patients)

| Cohort | Patients: all / both outcomes | Event rate: all / restricted | A0: all / restricted | Pooled AUC: all / restricted (95% CI) | Within-patient AUC | Pooled sens; spec, restricted | Pooled Youden: all / restricted (95% CI) | Within-patient Youden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| REPLACE-BG | 45 / 42 | 0.06 / 0.06 | 0.75 / 0.73 | 0.83 / 0.82 (0.79 to 0.84) | 0.76 | 0.70; 0.76 | 0.48 / 0.46 (0.42 to 0.51) | 0.35 |
| CTN-0094 | 246 / 154 | 0.30 / 0.41 | 0.93 / 0.85 | 0.93 / 0.86 (0.84 to 0.88) | 0.64 | 0.82; 0.77 | 0.72 / 0.59 (0.54 to 0.63) | 0.26 |
| College Experience | 38 / 37 | 0.40 / 0.40 | 0.85 / 0.85 | 0.94 / 0.93 (0.90 to 0.96) | 0.80 | 0.86; 0.86 | 0.72 / 0.72 (0.64 to 0.78) | 0.37 |

## Brier improvement of the model, equally weighted over patients

Mean over all test patients of (Brier score of the reference minus Brier score of the model), each patient's score being the mean over that patient's observations; 95% interval from 2,000 resamples of patients. The last column is the share of patients with a positive difference.

| Cohort | Reference | Mean per-patient improvement (95% CI) | Mean per-patient Brier of reference | Patients improved |
| --- | --- | ---: | ---: | ---: |
| REPLACE-BG | run-in rate | 0.013 (0.008 to 0.017) | 0.061 | 84% |
| REPLACE-BG | updated rate | 0.007 (0.004 to 0.010) | 0.055 | 76% |
| REPLACE-BG | rate over the scored period | 0.006 (0.003 to 0.009) | 0.054 | 64% |
| CTN-0094 | run-in rate | 0.092 (0.071 to 0.115) | 0.187 | 57% |
| CTN-0094 | updated rate | 0.028 (0.021 to 0.035) | 0.122 | 54% |
| CTN-0094 | rate over the scored period | 0.003 (-0.003 to 0.009) | 0.098 | 31% |
| College Experience | run-in rate | 0.135 (0.092 to 0.181) | 0.236 | 92% |
| College Experience | updated rate | 0.060 (0.042 to 0.079) | 0.162 | 87% |
| College Experience | rate over the scored period | 0.046 (0.029 to 0.062) | 0.148 | 76% |
