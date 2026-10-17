# Block-shuffle model-fixed permutation results

Seed: `20261009`; permutations per shuffle: `200`.

Blocks contain consecutive scored rows after sorting by the cohort time index. The full-shuffle rows use the original analysis generator. Each patient's baseline is its mean shuffled AUC; excess is observed AUC minus that baseline. The p-value share uses the one-sided plus-one convention described in the JSON.

| Cohort | Time unit | Model | Shuffle | Patients | Mean patient AUC | Mean shuffled baseline | Mean excess | Median excess [Q1, Q3] | Patients with p < 0.05 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| REPLACE-BG | scored night (night_index; daily/nightly steps) | Logistic Regression | full | 42 | 0.751 | 0.491 | 0.260 | 0.295 [0.172, 0.341] | 81.0% |
| REPLACE-BG | scored night (night_index; daily/nightly steps) | Logistic Regression | block_7 | 42 | 0.751 | 0.546 | 0.206 | 0.228 [0.135, 0.270] | 66.7% |
| REPLACE-BG | scored night (night_index; daily/nightly steps) | Logistic Regression | block_14 | 42 | 0.751 | 0.525 | 0.226 | 0.241 [0.155, 0.299] | 73.8% |
| REPLACE-BG | scored night (night_index; daily/nightly steps) | Gradient Boosting | full | 42 | 0.759 | 0.485 | 0.275 | 0.292 [0.189, 0.351] | 83.3% |
| REPLACE-BG | scored night (night_index; daily/nightly steps) | Gradient Boosting | block_7 | 42 | 0.759 | 0.550 | 0.209 | 0.222 [0.135, 0.277] | 71.4% |
| REPLACE-BG | scored night (night_index; daily/nightly steps) | Gradient Boosting | block_14 | 42 | 0.759 | 0.524 | 0.236 | 0.249 [0.172, 0.300] | 76.2% |
| CTN-0094 weekly | observed scored study week | Logistic Regression | full | 154 | 0.647 | 0.405 | 0.242 | 0.278 [0.042, 0.434] | 39.0% |
| CTN-0094 weekly | observed scored study week | Logistic Regression | block_4 | 154 | 0.647 | 0.562 | 0.085 | 0.071 [-0.000, 0.160] | 7.8% |
| CTN-0094 weekly | observed scored study week | Logistic Regression | block_8 | 154 | 0.647 | 0.509 | 0.138 | 0.094 [0.008, 0.285] | 22.1% |
| CTN-0094 weekly | observed scored study week | Gradient Boosting | full | 154 | 0.638 | 0.400 | 0.238 | 0.263 [0.050, 0.457] | 35.7% |
| CTN-0094 weekly | observed scored study week | Gradient Boosting | block_4 | 154 | 0.638 | 0.540 | 0.098 | 0.072 [-0.014, 0.176] | 7.8% |
| CTN-0094 weekly | observed scored study week | Gradient Boosting | block_8 | 154 | 0.638 | 0.486 | 0.152 | 0.135 [0.008, 0.290] | 18.8% |
| College Experience | scored complete-report row (date-ordered; nominally daily) | Logistic Regression | full | 37 | 0.806 | 0.489 | 0.316 | 0.337 [0.275, 0.403] | 86.5% |
| College Experience | scored complete-report row (date-ordered; nominally daily) | Logistic Regression | block_7 | 37 | 0.806 | 0.783 | 0.022 | 0.019 [0.003, 0.063] | 24.3% |
| College Experience | scored complete-report row (date-ordered; nominally daily) | Logistic Regression | block_14 | 37 | 0.806 | 0.730 | 0.075 | 0.047 [0.030, 0.111] | 51.4% |
| College Experience | scored complete-report row (date-ordered; nominally daily) | Gradient Boosting | full | 37 | 0.800 | 0.484 | 0.316 | 0.335 [0.280, 0.405] | 86.5% |
| College Experience | scored complete-report row (date-ordered; nominally daily) | Gradient Boosting | block_7 | 37 | 0.800 | 0.783 | 0.017 | 0.020 [0.001, 0.058] | 24.3% |
| College Experience | scored complete-report row (date-ordered; nominally daily) | Gradient Boosting | block_14 | 37 | 0.800 | 0.730 | 0.071 | 0.054 [0.030, 0.113] | 45.9% |

Full-shuffle reproduction check (CTN-0094 gradient boosting): passed.

For every shuffle, outcomes are assigned back to their original scored rows. The existing cohort-specific callback then rebuilds outcome-history inputs from those outcomes while keeping run-in/origin inputs and all non-history inputs fixed; no model is refit.
