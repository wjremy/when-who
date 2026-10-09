# Predictive values, pooled and within patients (test patients)

Gain = predictive value minus the event rate (PPV) or minus one minus the event rate (NPV). Pooled: against the cohort rate. Within: each patient's value against that patient's own rate over the scored period, averaged over patients with at least one alarm (PPV) or one non-alarm (NPV). Intervals from 2,000 bootstrap resamples of patients.

| Cohort | Alarm | Event rate | Pooled PPV | Pooled PPV gain | Within-patient PPV gain | Pooled NPV | Pooled NPV gain | Within-patient NPV gain | Patients with alarm / non-alarm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| REPLACE-BG | model | 0.06 | 0.17 | 0.11 (0.09 to 0.12) | 0.08 (0.06 to 0.09) | 0.98 | 0.04 (0.03 to 0.05) | 0.03 (0.02 to 0.04) | 43 / 45 |
| REPLACE-BG | run-in rate | 0.06 | 0.08 | 0.02 (0.01 to 0.04) | 0.00 (0.00 to 0.00) | 0.96 | 0.02 (0.01 to 0.04) | -0.00 (-0.00 to 0.00) | 22 / 23 |
| CTN-0094 | model | 0.30 | 0.74 | 0.44 (0.40 to 0.48) | 0.13 (0.10 to 0.17) | 0.93 | 0.23 (0.19 to 0.27) | 0.08 (0.05 to 0.10) | 166 / 206 |
| CTN-0094 | run-in rate | 0.30 | 0.50 | 0.20 (0.16 to 0.25) | 0.00 (0.00 to 0.00) | 0.89 | 0.19 (0.15 to 0.23) | -0.00 (-0.00 to -0.00) | 124 / 122 |
| College Experience | model | 0.40 | 0.81 | 0.41 (0.32 to 0.49) | 0.23 (0.18 to 0.30) | 0.90 | 0.30 (0.21 to 0.39) | 0.18 (0.13 to 0.24) | 37 / 35 |
| College Experience | run-in rate | 0.40 | 0.57 | 0.17 (0.02 to 0.34) | 0.00 (0.00 to 0.00) | 0.67 | 0.07 (0.01 to 0.15) | 0.00 (-0.00 to 0.00) | 14 / 24 |
