# CTN-0094 real-data analysis

Primary: weekly next-week opioid-positive UDS; missing target weeks excluded. Sensitivity: missing weeks counted positive. Weekly run-in: weeks 0–3; eligibility: at least 12 observed screen weeks, including 2 run-in and 8 later weeks. Daily TLFB is exploratory: first 28 inferred coverage days are run-in, with at least 56 inferred coverage days required. Weeks are seven-day bins anchored at the first randomization.

## Cohort

| Analysis | Patients | Scored steps | Event rate | Mean patient rate | SD patient rates | ICC |
|---|---:|---:|---:|---:|---:|---:|
| weekly_missing_excluded | 1224 | 20992 | 0.312 | 0.329 | 0.340 | 0.537 |
| weekly_missing_positive | 1224 | 32287 | 0.553 | 0.553 | 0.249 | 0.250 |
| daily_self_reported | 1896 | 320705 | 0.170 | 0.179 | 0.200 | 0.283 |

## Ranking and benchmark (held-out patients)

| Analysis | Forecast | Pooled AUC | 95% CI | Within-person AUC | 95% CI |
|---|---|---:|---:|---:|---:|
| weekly_missing_excluded | training_cohort_rate | 0.500 | [0.500, 0.500] | 0.500 | [0.500, 0.500] |
| weekly_missing_excluded | runin_rate | 0.801 | [0.762, 0.838] | 0.500 | [0.500, 0.500] |
| weekly_missing_excluded | runin_shrunk | 0.813 | [0.777, 0.848] | 0.500 | [0.500, 0.500] |
| weekly_missing_excluded | updating_rate | 0.889 | [0.865, 0.909] | 0.436 | [0.399, 0.477] |
| weekly_missing_excluded | recent_window_rate | 0.882 | [0.862, 0.899] | 0.574 | [0.539, 0.608] |
| weekly_missing_excluded | persistence | 0.817 | [0.795, 0.836] | 0.618 | [0.588, 0.647] |
| weekly_missing_excluded | retrospective_rate | 0.932 | [0.916, 0.946] | 0.500 | [0.500, 0.500] |
| weekly_missing_excluded | logistic_regression | 0.923 | [0.907, 0.937] | 0.653 | [0.611, 0.693] |
| weekly_missing_excluded | gradient_boosting | 0.929 | [0.915, 0.943] | 0.638 | [0.595, 0.682] |
| weekly_missing_excluded | within_centred_gradient_boosting | 0.918 | [0.902, 0.933] | 0.644 | [0.597, 0.687] |
| weekly_missing_excluded | normal approximation, observed SD | 0.872 | NA | NA | NA |
| weekly_missing_excluded | normal approximation, sampling-corrected SD | 0.866 | NA | NA | NA |
| weekly_missing_positive | training_cohort_rate | 0.500 | [0.500, 0.500] | 0.500 | [0.500, 0.500] |
| weekly_missing_positive | runin_rate | 0.679 | [0.648, 0.708] | 0.500 | [0.500, 0.500] |
| weekly_missing_positive | runin_shrunk | 0.679 | [0.649, 0.710] | 0.500 | [0.500, 0.500] |
| weekly_missing_positive | updating_rate | 0.767 | [0.745, 0.789] | 0.495 | [0.468, 0.521] |
| weekly_missing_positive | recent_window_rate | 0.805 | [0.788, 0.823] | 0.619 | [0.598, 0.641] |
| weekly_missing_positive | persistence | 0.726 | [0.706, 0.745] | 0.615 | [0.596, 0.634] |
| weekly_missing_positive | retrospective_rate | 0.786 | [0.765, 0.807] | 0.500 | [0.500, 0.500] |
| weekly_missing_positive | logistic_regression | 0.844 | [0.828, 0.859] | 0.689 | [0.664, 0.713] |
| weekly_missing_positive | gradient_boosting | 0.874 | [0.860, 0.887] | 0.723 | [0.699, 0.747] |
| weekly_missing_positive | within_centred_gradient_boosting | 0.864 | [0.850, 0.878] | 0.711 | [0.686, 0.737] |
| weekly_missing_positive | normal approximation, observed SD | 0.762 | NA | NA | NA |
| weekly_missing_positive | normal approximation, sampling-corrected SD | 0.749 | NA | NA | NA |
| daily_self_reported | training_cohort_rate | 0.500 | [0.500, 0.500] | 0.500 | [0.500, 0.500] |
| daily_self_reported | runin_rate | 0.550 | [0.522, 0.575] | 0.500 | [0.500, 0.500] |
| daily_self_reported | runin_shrunk | 0.550 | [0.523, 0.573] | 0.500 | [0.500, 0.500] |
| daily_self_reported | updating_rate | 0.737 | [0.707, 0.766] | 0.636 | [0.612, 0.659] |
| daily_self_reported | recent_window_rate | 0.867 | [0.849, 0.881] | 0.745 | [0.727, 0.762] |
| daily_self_reported | persistence | 0.835 | [0.810, 0.855] | 0.775 | [0.759, 0.791] |
| daily_self_reported | retrospective_rate | 0.842 | [0.820, 0.861] | 0.500 | [0.500, 0.500] |
| daily_self_reported | logistic_regression | 0.895 | [0.880, 0.908] | 0.805 | [0.788, 0.822] |
| daily_self_reported | gradient_boosting | 0.927 | [0.915, 0.937] | 0.812 | [0.795, 0.829] |
| daily_self_reported | within_centred_gradient_boosting | 0.925 | [0.913, 0.935] | 0.805 | [0.788, 0.823] |
| daily_self_reported | normal approximation, observed SD | 0.841 | NA | NA | NA |
| daily_self_reported | normal approximation, sampling-corrected SD | 0.839 | NA | NA | NA |

## Scores and Brier skill (held-out patients)

| Analysis | Forecast | Brier | Log loss | Brier skill: run-in / updating / retrospective | Log-loss skill: run-in / updating / retrospective |
|---|---|---:|---:|---|---|
| weekly_missing_excluded | training_cohort_rate | 0.210 | 0.612 | -0.141 / -0.770 / -1.193 | 0.651 / -0.165 / -1.082 |
| weekly_missing_excluded | runin_rate | 0.184 | 1.755 | 0.000 / -0.551 / -0.923 | 0.000 / -2.343 / -4.976 |
| weekly_missing_excluded | runin_shrunk | 0.174 | 0.530 | 0.056 / -0.464 / -0.814 | 0.698 / -0.009 / -0.804 |
| weekly_missing_excluded | updating_rate | 0.119 | 0.525 | 0.355 / 0.000 / -0.239 | 0.701 / 0.000 / -0.787 |
| weekly_missing_excluded | recent_window_rate | 0.119 | 0.983 | 0.354 / -0.002 / -0.242 | 0.440 / -0.872 / -2.346 |
| weekly_missing_excluded | persistence | 0.160 | 1.601 | 0.135 / -0.342 / -0.663 | 0.088 / -2.049 / -4.450 |
| weekly_missing_excluded | retrospective_rate | 0.096 | 0.294 | 0.480 / 0.193 / 0.000 | 0.833 / 0.441 / 0.000 |
| weekly_missing_excluded | logistic_regression | 0.097 | 0.323 | 0.474 / 0.185 / -0.011 | 0.816 / 0.386 / -0.098 |
| weekly_missing_excluded | gradient_boosting | 0.093 | 0.306 | 0.496 / 0.219 / 0.032 | 0.826 / 0.417 / -0.042 |
| weekly_missing_excluded | within_centred_gradient_boosting | 0.102 | 0.333 | 0.447 / 0.142 / -0.064 | 0.810 / 0.366 / -0.133 |
| weekly_missing_positive | training_cohort_rate | 0.247 | 0.688 | 0.045 / -0.189 / -0.335 | 0.620 / 0.061 / -0.271 |
| weekly_missing_positive | runin_rate | 0.259 | 1.808 | 0.000 / -0.245 / -0.398 | 0.000 / -1.469 / -2.342 |
| weekly_missing_positive | runin_shrunk | 0.225 | 0.641 | 0.131 / -0.082 / -0.214 | 0.645 / 0.124 / -0.185 |
| weekly_missing_positive | updating_rate | 0.208 | 0.732 | 0.197 / 0.000 / -0.123 | 0.595 / 0.000 / -0.353 |
| weekly_missing_positive | recent_window_rate | 0.187 | 1.340 | 0.279 / 0.102 / -0.008 | 0.259 / -0.830 / -1.477 |
| weekly_missing_positive | persistence | 0.271 | 3.746 | -0.047 / -0.304 / -0.464 | -1.071 / -4.115 / -5.922 |
| weekly_missing_positive | retrospective_rate | 0.185 | 0.541 | 0.285 / 0.109 / 0.000 | 0.701 / 0.261 / 0.000 |
| weekly_missing_positive | logistic_regression | 0.158 | 0.485 | 0.390 / 0.240 / 0.147 | 0.732 / 0.337 / 0.103 |
| weekly_missing_positive | gradient_boosting | 0.141 | 0.437 | 0.455 / 0.321 / 0.238 | 0.759 / 0.404 / 0.193 |
| weekly_missing_positive | within_centred_gradient_boosting | 0.147 | 0.455 | 0.433 / 0.294 / 0.207 | 0.749 / 0.379 / 0.160 |
| daily_self_reported | training_cohort_rate | 0.139 | 0.452 | 0.815 / 0.272 / -0.383 | 0.947 / 0.273 / -0.380 |
| daily_self_reported | runin_rate | 0.752 | 8.486 | 0.000 / -2.931 / -6.464 | 0.000 / -12.670 / -24.929 |
| daily_self_reported | runin_shrunk | 0.470 | 1.195 | 0.375 / -1.458 / -3.667 | 0.859 / -0.924 / -2.650 |
| daily_self_reported | updating_rate | 0.191 | 0.621 | 0.746 / 0.000 / -0.899 | 0.927 / 0.000 / -0.897 |
| daily_self_reported | recent_window_rate | 0.114 | 0.444 | 0.849 / 0.406 / -0.128 | 0.948 / 0.284 / -0.358 |
| daily_self_reported | persistence | 0.093 | 1.290 | 0.876 / 0.512 / 0.073 | 0.848 / -1.077 / -2.940 |
| daily_self_reported | retrospective_rate | 0.101 | 0.327 | 0.866 / 0.473 / 0.000 | 0.961 / 0.473 / 0.000 |
| daily_self_reported | logistic_regression | 0.070 | 0.251 | 0.907 / 0.635 / 0.307 | 0.970 / 0.596 / 0.234 |
| daily_self_reported | gradient_boosting | 0.064 | 0.223 | 0.915 / 0.666 / 0.366 | 0.974 / 0.641 / 0.320 |
| daily_self_reported | within_centred_gradient_boosting | 0.066 | 0.228 | 0.912 / 0.656 / 0.346 | 0.973 / 0.633 / 0.304 |

## Calibration and history bias

| Analysis | Model/reference | Day-weighted pooled/between/within slopes | Equal-person pooled/between/within slopes | Between-logit variance share | History AUC observed / permutation / difference |
|---|---|---|---|---:|---|
| weekly_missing_excluded | logistic_regression | 1.084 / 1.179 / 0.713 | 1.095 / 1.192 / 0.714 | 0.810 | NA |
| weekly_missing_excluded | gradient_boosting | 1.069 / 1.194 / 0.747 | 1.080 / 1.206 / 0.756 | 0.832 | NA |
| weekly_missing_excluded | updating_rate | NA | NA | NA | 0.436 / 0.296 / 0.141 |
| weekly_missing_excluded | recent_window_rate | NA | NA | NA | 0.574 / 0.426 / 0.148 |
| weekly_missing_excluded | persistence | NA | NA | NA | 0.618 / 0.475 / 0.142 |
| weekly_missing_positive | logistic_regression | 1.036 / 1.187 / 0.893 | 1.028 / 1.195 / 0.860 | 0.542 | NA |
| weekly_missing_positive | gradient_boosting | 1.059 / 1.290 / 0.923 | 1.042 / 1.244 / 0.905 | 0.397 | NA |
| weekly_missing_positive | updating_rate | NA | NA | NA | 0.495 / 0.337 / 0.159 |
| weekly_missing_positive | recent_window_rate | NA | NA | NA | 0.619 / 0.440 / 0.179 |
| weekly_missing_positive | persistence | NA | NA | NA | 0.615 / 0.481 / 0.134 |
| daily_self_reported | logistic_regression | 0.964 / 1.364 / 0.835 | 0.936 / 1.332 / 0.809 | 0.386 | NA |
| daily_self_reported | gradient_boosting | 0.988 / 1.217 / 0.904 | 0.979 / 1.225 / 0.890 | 0.485 | NA |
| daily_self_reported | updating_rate | NA | NA | NA | 0.636 / 0.472 / 0.164 |
| daily_self_reported | recent_window_rate | NA | NA | NA | 0.745 / 0.463 / 0.282 |
| daily_self_reported | persistence | NA | NA | NA | 0.775 / 0.496 / 0.279 |

Patient bootstrap intervals and tie-matched average precision are fully recorded in the JSON file.
Model fitting uses scikit-learn; when it is unavailable, block 3 records the import error and baseline metrics still run.
UDS collection is proxied by a recorded temperature or any UDS finding; all UDS findings are positive-only. Daily TLFB non-use days are inferred between the first and last recorded TLFB drug-use days.
