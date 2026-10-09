# Case analysis forecast metrics

## ces

Patients: 38; observations: 6361; event rate: 0.399.

| Forecast | Pooled AUC | Within AUC (eligible/excluded) | Sensitivity | Specificity | PPV | Youden | Within sens. | Within spec. | Within Youden | Alarms/30 mean | Alarms/30 median | Brier pooled | Brier patient mean | Brier skill | Calibration slopes pooled/between/within |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| training_cohort_rate | 0.500 | 0.500 (37/1) | 1.000 | 0.000 | 0.399 | 0.000 | 1.000 | 0.000 | 0.000 | 30.000 | 30.000 | 0.240 | 0.249 | — | NA / NA / NA |
| runin_rate | 0.695 | 0.500 (37/1) | 0.432 | 0.781 | 0.567 | 0.213 | 0.378 | 0.622 | 0.000 | 11.053 | 0.000 | 0.237 | 0.236 | — | NA / NA / NA |
| runin_rate_shrunk_10 | 0.695 | 0.500 (37/1) | 0.432 | 0.781 | 0.567 | 0.213 | 0.378 | 0.622 | 0.000 | 11.053 | 0.000 | 0.221 | 0.222 | — | NA / NA / NA |
| expanding_rate | 0.842 | 0.578 (37/1) | 0.677 | 0.814 | 0.707 | 0.491 | 0.482 | 0.573 | 0.055 | 13.474 | 7.383 | 0.161 | 0.162 | — | NA / NA / NA |
| last12_rate | 0.905 | 0.704 (37/1) | 0.821 | 0.823 | 0.755 | 0.644 | 0.619 | 0.606 | 0.225 | 14.514 | 15.121 | 0.120 | 0.126 | — | NA / NA / NA |
| retrospective_rate | 0.855 | 0.500 (37/1) | 0.818 | 0.712 | 0.653 | 0.530 | 0.568 | 0.432 | 0.000 | 16.579 | 30.000 | 0.150 | 0.148 | — | NA / NA / NA |
| logistic_regression | 0.934 | 0.806 (37/1) | 0.852 | 0.873 | 0.816 | 0.724 | 0.681 | 0.665 | 0.346 | 13.950 | 14.157 | 0.098 | 0.103 | runin_rate: 0.586; updating_rate: 0.390; retrospective_rate: 0.344 | 1.062 / 1.116 / 0.996 |
| gradient_boosting | 0.935 | 0.800 (37/1) | 0.858 | 0.862 | 0.805 | 0.720 | 0.704 | 0.662 | 0.365 | 14.229 | 14.488 | 0.097 | 0.102 | runin_rate: 0.590; updating_rate: 0.396; retrospective_rate: 0.350; 95% CI: runin_rate [0.465, 0.696]; updating_rate [0.289, 0.506]; retrospective_rate [0.240, 0.467] | 1.072 / 1.138 / 1.003 (95% CI: [0.986, 1.152]; skipped 0 / [1.049, 1.216]; skipped 0 / [0.895, 1.111]; skipped 0) |
| within_centred_gradient_boosting | 0.893 | 0.797 (37/1) | 0.796 | 0.829 | 0.755 | 0.625 | 0.711 | 0.694 | 0.405 | 13.816 | 13.710 | 0.129 | 0.144 | runin_rate: 0.456; updating_rate: 0.199; retrospective_rate: 0.138 | 1.143 / 1.519 / 0.962 |

## ctn_daily_self_reported

Patients: 380; observations: 66114; event rate: 0.167.

| Forecast | Pooled AUC | Within AUC (eligible/excluded) | Sensitivity | Specificity | PPV | Youden | Within sens. | Within spec. | Within Youden | Alarms/30 mean | Alarms/30 median | Brier pooled | Brier patient mean | Brier skill | Calibration slopes pooled/between/within |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| training_cohort_rate | 0.500 | 0.500 (375/5) | 1.000 | 0.000 | 0.167 | 0.000 | 1.000 | 0.000 | 0.000 | 30.000 | 30.000 | 0.139 | 0.144 | — | NA / NA / NA |
| runin_rate | 0.550 | 0.500 (375/5) | 1.000 | 0.003 | 0.168 | 0.003 | 0.997 | 0.003 | 0.000 | 29.921 | 30.000 | 0.752 | 0.743 | — | NA / NA / NA |
| runin_shrunk | 0.550 | 0.500 (375/5) | 1.000 | 0.003 | 0.168 | 0.003 | 0.997 | 0.003 | 0.000 | 29.921 | 30.000 | 0.470 | 0.465 | runin_rate: 0.375; updating_rate: -1.458; retrospective_rate: -3.667 | 0.630 / NA / NA |
| updating_rate | 0.737 | 0.636 (375/5) | 0.967 | 0.149 | 0.186 | 0.116 | 0.934 | 0.094 | 0.028 | 27.045 | 30.000 | 0.191 | 0.219 | — | NA / NA / NA |
| recent_window_rate | 0.867 | 0.745 (375/5) | 0.839 | 0.729 | 0.384 | 0.569 | 0.724 | 0.617 | 0.341 | 12.611 | 10.453 | 0.114 | 0.135 | — | NA / NA / NA |
| persistence | 0.835 | 0.775 (375/5) | 0.727 | 0.943 | 0.718 | 0.670 | 0.625 | 0.924 | 0.550 | 5.311 | 3.560 | 0.093 | 0.099 | — | NA / NA / NA |
| retrospective_rate | 0.842 | 0.500 (375/5) | 0.757 | 0.750 | 0.378 | 0.506 | 0.371 | 0.629 | 0.000 | 10.974 | 0.000 | 0.101 | 0.106 | — | NA / NA / NA |
| logistic_regression | 0.895 | 0.805 (375/5) | 0.753 | 0.918 | 0.648 | 0.671 | 0.653 | 0.880 | 0.533 | 6.240 | 4.356 | 0.070 | 0.076 | runin_rate: 0.907; updating_rate: 0.635; retrospective_rate: 0.307 | 0.964 / 1.364 / 0.835 |
| gradient_boosting | 0.927 | 0.812 (375/5) | 0.804 | 0.901 | 0.620 | 0.705 | 0.680 | 0.873 | 0.553 | 6.588 | 4.227 | 0.064 | 0.069 | runin_rate: 0.915; updating_rate: 0.666; retrospective_rate: 0.366; 95% CI: runin_rate [0.906, 0.925]; updating_rate [0.642, 0.695]; retrospective_rate [0.316, 0.411] | 0.988 / 1.217 / 0.904 (95% CI: [0.952, 1.025]; skipped 0 / [1.179, 1.254]; skipped 0 / [0.864, 0.945]; skipped 0) |
| within_centred_gradient_boosting | 0.925 | 0.805 (375/5) | 0.795 | 0.902 | 0.620 | 0.697 | 0.670 | 0.872 | 0.542 | 6.544 | 4.216 | 0.066 | 0.071 | runin_rate: 0.912; updating_rate: 0.656; retrospective_rate: 0.346 | 0.979 / 1.213 / 0.893 |

## ctn_daily_self_reported_with_current_dose

Patients: 380; observations: 66114; event rate: 0.167.

| Forecast | Pooled AUC | Within AUC (eligible/excluded) | Sensitivity | Specificity | PPV | Youden | Within sens. | Within spec. | Within Youden | Alarms/30 mean | Alarms/30 median | Brier pooled | Brier patient mean | Brier skill | Calibration slopes pooled/between/within |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| training_cohort_rate | 0.500 | 0.500 (375/5) | 1.000 | 0.000 | 0.167 | 0.000 | 1.000 | 0.000 | 0.000 | 30.000 | 30.000 | 0.139 | 0.144 | — | NA / NA / NA |
| runin_rate | 0.550 | 0.500 (375/5) | 1.000 | 0.003 | 0.168 | 0.003 | 0.997 | 0.003 | 0.000 | 29.921 | 30.000 | 0.752 | 0.743 | — | NA / NA / NA |
| runin_shrunk | 0.550 | 0.500 (375/5) | 1.000 | 0.003 | 0.168 | 0.003 | 0.997 | 0.003 | 0.000 | 29.921 | 30.000 | 0.470 | 0.465 | runin_rate: 0.375; updating_rate: -1.458; retrospective_rate: -3.667 | 0.630 / NA / NA |
| updating_rate | 0.737 | 0.636 (375/5) | 0.967 | 0.149 | 0.186 | 0.116 | 0.934 | 0.094 | 0.028 | 27.045 | 30.000 | 0.191 | 0.219 | — | NA / NA / NA |
| recent_window_rate | 0.867 | 0.745 (375/5) | 0.839 | 0.729 | 0.384 | 0.569 | 0.724 | 0.617 | 0.341 | 12.611 | 10.453 | 0.114 | 0.135 | — | NA / NA / NA |
| persistence | 0.835 | 0.775 (375/5) | 0.727 | 0.943 | 0.718 | 0.670 | 0.625 | 0.924 | 0.550 | 5.311 | 3.560 | 0.093 | 0.099 | — | NA / NA / NA |
| retrospective_rate | 0.842 | 0.500 (375/5) | 0.757 | 0.750 | 0.378 | 0.506 | 0.371 | 0.629 | 0.000 | 10.974 | 0.000 | 0.101 | 0.106 | — | NA / NA / NA |
| logistic_regression | 0.895 | 0.805 (375/5) | 0.753 | 0.918 | 0.648 | 0.671 | 0.653 | 0.880 | 0.533 | 6.240 | 4.356 | 0.070 | 0.076 | runin_rate: 0.907; updating_rate: 0.635; retrospective_rate: 0.307 | 0.964 / 1.364 / 0.835 |
| gradient_boosting | 0.927 | 0.812 (375/5) | 0.804 | 0.901 | 0.620 | 0.705 | 0.680 | 0.873 | 0.553 | 6.588 | 4.227 | 0.064 | 0.069 | runin_rate: 0.915; updating_rate: 0.666; retrospective_rate: 0.366; 95% CI: runin_rate [0.906, 0.925]; updating_rate [0.642, 0.695]; retrospective_rate [0.316, 0.411] | 0.988 / 1.217 / 0.904 (95% CI: [0.952, 1.025]; skipped 0 / [1.179, 1.254]; skipped 0 / [0.864, 0.945]; skipped 0) |
| within_centred_gradient_boosting | 0.925 | 0.805 (375/5) | 0.795 | 0.902 | 0.620 | 0.697 | 0.670 | 0.872 | 0.542 | 6.544 | 4.216 | 0.066 | 0.071 | runin_rate: 0.912; updating_rate: 0.656; retrospective_rate: 0.346 | 0.979 / 1.213 / 0.893 |

## ctn_weekly_missing_excluded

Patients: 246; observations: 4237; event rate: 0.300.

| Forecast | Pooled AUC | Within AUC (eligible/excluded) | Sensitivity | Specificity | PPV | Youden | Within sens. | Within spec. | Within Youden | Alarms/30 mean | Alarms/30 median | Brier pooled | Brier patient mean | Brier skill | Calibration slopes pooled/between/within |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| training_cohort_rate | 0.500 | 0.500 (154/92) | 1.000 | 0.000 | 0.300 | 0.000 | 1.000 | 0.000 | 0.000 | 30.000 | 30.000 | 0.210 | 0.219 | — | NA / NA / NA |
| runin_rate | 0.801 | 0.500 (154/92) | 0.814 | 0.653 | 0.501 | 0.467 | 0.649 | 0.351 | 0.000 | 15.122 | 30.000 | 0.184 | 0.187 | — | NA / NA / NA |
| runin_shrunk | 0.813 | 0.500 (154/92) | 0.814 | 0.653 | 0.501 | 0.467 | 0.649 | 0.351 | 0.000 | 15.122 | 30.000 | 0.174 | 0.180 | runin_rate: 0.056; updating_rate: -0.464; retrospective_rate: -0.814 | 3.023 / NA / NA |
| updating_rate | 0.889 | 0.436 (154/92) | 0.806 | 0.809 | 0.644 | 0.615 | 0.537 | 0.466 | 0.004 | 11.936 | 3.542 | 0.119 | 0.122 | — | NA / NA / NA |
| recent_window_rate | 0.882 | 0.574 (154/92) | 0.857 | 0.791 | 0.637 | 0.648 | 0.670 | 0.482 | 0.152 | 12.834 | 9.000 | 0.119 | 0.123 | — | NA / NA / NA |
| persistence | 0.817 | 0.618 (154/92) | 0.836 | 0.695 | 0.540 | 0.530 | 0.702 | 0.501 | 0.203 | 14.945 | 13.125 | 0.160 | 0.165 | — | NA / NA / NA |
| retrospective_rate | 0.932 | 0.500 (154/92) | 0.875 | 0.819 | 0.675 | 0.694 | 0.578 | 0.422 | 0.000 | 12.561 | 0.000 | 0.096 | 0.098 | — | NA / NA / NA |
| logistic_regression | 0.923 | 0.653 (154/92) | 0.821 | 0.880 | 0.746 | 0.702 | 0.591 | 0.585 | 0.176 | 10.668 | 4.000 | 0.097 | 0.099 | runin_rate: 0.474; updating_rate: 0.185; retrospective_rate: -0.011 | 1.084 / 1.179 / 0.713 |
| gradient_boosting | 0.929 | 0.638 (154/92) | 0.848 | 0.872 | 0.739 | 0.720 | 0.670 | 0.588 | 0.258 | 11.099 | 5.970 | 0.093 | 0.095 | runin_rate: 0.496; updating_rate: 0.219; retrospective_rate: 0.032; 95% CI: runin_rate [0.434, 0.556]; updating_rate [0.178, 0.263]; retrospective_rate [-0.019, 0.083] | 1.069 / 1.194 / 0.747 (95% CI: [1.005, 1.141]; skipped 0 / [1.128, 1.280]; skipped 0 / [0.633, 0.857]; skipped 0) |
| within_centred_gradient_boosting | 0.918 | 0.644 (154/92) | 0.821 | 0.865 | 0.723 | 0.687 | 0.654 | 0.604 | 0.257 | 10.957 | 7.212 | 0.102 | 0.104 | runin_rate: 0.447; updating_rate: 0.142; retrospective_rate: -0.064 | 1.107 / 1.360 / 0.663 |

## ctn_weekly_missing_excluded_with_current_dose

Patients: 246; observations: 4237; event rate: 0.300.

| Forecast | Pooled AUC | Within AUC (eligible/excluded) | Sensitivity | Specificity | PPV | Youden | Within sens. | Within spec. | Within Youden | Alarms/30 mean | Alarms/30 median | Brier pooled | Brier patient mean | Brier skill | Calibration slopes pooled/between/within |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| training_cohort_rate | 0.500 | 0.500 (154/92) | 1.000 | 0.000 | 0.300 | 0.000 | 1.000 | 0.000 | 0.000 | 30.000 | 30.000 | 0.210 | 0.219 | — | NA / NA / NA |
| runin_rate | 0.801 | 0.500 (154/92) | 0.814 | 0.653 | 0.501 | 0.467 | 0.649 | 0.351 | 0.000 | 15.122 | 30.000 | 0.184 | 0.187 | — | NA / NA / NA |
| runin_shrunk | 0.813 | 0.500 (154/92) | 0.814 | 0.653 | 0.501 | 0.467 | 0.649 | 0.351 | 0.000 | 15.122 | 30.000 | 0.174 | 0.180 | runin_rate: 0.056; updating_rate: -0.464; retrospective_rate: -0.814 | 3.023 / NA / NA |
| updating_rate | 0.889 | 0.436 (154/92) | 0.806 | 0.809 | 0.644 | 0.615 | 0.537 | 0.466 | 0.004 | 11.936 | 3.542 | 0.119 | 0.122 | — | NA / NA / NA |
| recent_window_rate | 0.882 | 0.574 (154/92) | 0.857 | 0.791 | 0.637 | 0.648 | 0.670 | 0.482 | 0.152 | 12.834 | 9.000 | 0.119 | 0.123 | — | NA / NA / NA |
| persistence | 0.817 | 0.618 (154/92) | 0.836 | 0.695 | 0.540 | 0.530 | 0.702 | 0.501 | 0.203 | 14.945 | 13.125 | 0.160 | 0.165 | — | NA / NA / NA |
| retrospective_rate | 0.932 | 0.500 (154/92) | 0.875 | 0.819 | 0.675 | 0.694 | 0.578 | 0.422 | 0.000 | 12.561 | 0.000 | 0.096 | 0.098 | — | NA / NA / NA |
| logistic_regression | 0.923 | 0.654 (154/92) | 0.827 | 0.881 | 0.749 | 0.708 | 0.604 | 0.587 | 0.190 | 10.696 | 4.000 | 0.097 | 0.099 | runin_rate: 0.475; updating_rate: 0.185; retrospective_rate: -0.010 | 1.083 / 1.175 / 0.718 |
| gradient_boosting | 0.929 | 0.647 (154/92) | 0.851 | 0.867 | 0.734 | 0.719 | 0.664 | 0.576 | 0.241 | 11.232 | 6.158 | 0.093 | 0.095 | runin_rate: 0.495; updating_rate: 0.217; retrospective_rate: 0.030; 95% CI: runin_rate [0.434, 0.554]; updating_rate [0.173, 0.261]; retrospective_rate [-0.023, 0.080] | 1.073 / 1.193 / 0.758 (95% CI: [1.009, 1.148]; skipped 0 / [1.125, 1.277]; skipped 0 / [0.643, 0.867]; skipped 0) |
| within_centred_gradient_boosting | 0.918 | 0.648 (154/92) | 0.828 | 0.863 | 0.721 | 0.690 | 0.659 | 0.603 | 0.262 | 11.034 | 6.623 | 0.103 | 0.105 | runin_rate: 0.443; updating_rate: 0.136; retrospective_rate: -0.071 | 1.104 / 1.349 / 0.674 |

## ctn_weekly_missing_positive

Patients: 246; observations: 6632; event rate: 0.553.

| Forecast | Pooled AUC | Within AUC (eligible/excluded) | Sensitivity | Specificity | PPV | Youden | Within sens. | Within spec. | Within Youden | Alarms/30 mean | Alarms/30 median | Brier pooled | Brier patient mean | Brier skill | Calibration slopes pooled/between/within |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| training_cohort_rate | 0.500 | 0.500 (232/14) | 1.000 | 0.000 | 0.553 | 0.000 | 1.000 | 0.000 | 0.000 | 30.000 | 30.000 | 0.247 | 0.248 | — | NA / NA / NA |
| runin_rate | 0.679 | 0.500 (232/14) | 0.496 | 0.816 | 0.769 | 0.312 | 0.323 | 0.677 | 0.000 | 10.854 | 0.000 | 0.259 | 0.255 | — | NA / NA / NA |
| runin_shrunk | 0.679 | 0.500 (232/14) | 0.496 | 0.816 | 0.769 | 0.312 | 0.323 | 0.677 | 0.000 | 10.854 | 0.000 | 0.225 | 0.224 | runin_rate: 0.131; updating_rate: -0.082; retrospective_rate: -0.214 | 1.726 / NA / NA |
| updating_rate | 0.767 | 0.495 (232/14) | 0.537 | 0.846 | 0.812 | 0.383 | 0.328 | 0.675 | 0.004 | 11.094 | 2.146 | 0.208 | 0.207 | — | NA / NA / NA |
| recent_window_rate | 0.805 | 0.619 (232/14) | 0.682 | 0.832 | 0.834 | 0.514 | 0.538 | 0.692 | 0.231 | 13.439 | 10.323 | 0.187 | 0.189 | — | NA / NA / NA |
| persistence | 0.726 | 0.615 (232/14) | 0.756 | 0.695 | 0.754 | 0.451 | 0.650 | 0.579 | 0.229 | 16.553 | 15.000 | 0.271 | 0.276 | — | NA / NA / NA |
| retrospective_rate | 0.786 | 0.500 (232/14) | 0.613 | 0.823 | 0.810 | 0.435 | 0.384 | 0.616 | 0.000 | 12.561 | 0.000 | 0.185 | 0.184 | — | NA / NA / NA |
| logistic_regression | 0.844 | 0.689 (232/14) | 0.745 | 0.818 | 0.835 | 0.563 | 0.616 | 0.675 | 0.292 | 14.642 | 11.270 | 0.158 | 0.160 | runin_rate: 0.390; updating_rate: 0.240; retrospective_rate: 0.147 | 1.036 / 1.187 / 0.893 |
| gradient_boosting | 0.874 | 0.723 (232/14) | 0.764 | 0.838 | 0.854 | 0.602 | 0.674 | 0.700 | 0.374 | 14.785 | 11.250 | 0.141 | 0.143 | runin_rate: 0.455; updating_rate: 0.321; retrospective_rate: 0.238; 95% CI: runin_rate [0.418, 0.489]; updating_rate [0.291, 0.351]; retrospective_rate [0.203, 0.275] | 1.059 / 1.290 / 0.923 (95% CI: [1.009, 1.123]; skipped 0 / [1.200, 1.399]; skipped 0 / [0.869, 0.983]; skipped 0) |
| within_centred_gradient_boosting | 0.864 | 0.711 (232/14) | 0.758 | 0.831 | 0.847 | 0.589 | 0.674 | 0.699 | 0.372 | 14.817 | 11.111 | 0.147 | 0.148 | runin_rate: 0.433; updating_rate: 0.294; retrospective_rate: 0.207 | 1.101 / 1.366 / 0.957 |

## ctn_weekly_missing_positive_with_current_dose

Patients: 246; observations: 6632; event rate: 0.553.

| Forecast | Pooled AUC | Within AUC (eligible/excluded) | Sensitivity | Specificity | PPV | Youden | Within sens. | Within spec. | Within Youden | Alarms/30 mean | Alarms/30 median | Brier pooled | Brier patient mean | Brier skill | Calibration slopes pooled/between/within |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| training_cohort_rate | 0.500 | 0.500 (232/14) | 1.000 | 0.000 | 0.553 | 0.000 | 1.000 | 0.000 | 0.000 | 30.000 | 30.000 | 0.247 | 0.248 | — | NA / NA / NA |
| runin_rate | 0.679 | 0.500 (232/14) | 0.496 | 0.816 | 0.769 | 0.312 | 0.323 | 0.677 | 0.000 | 10.854 | 0.000 | 0.259 | 0.255 | — | NA / NA / NA |
| runin_shrunk | 0.679 | 0.500 (232/14) | 0.496 | 0.816 | 0.769 | 0.312 | 0.323 | 0.677 | 0.000 | 10.854 | 0.000 | 0.225 | 0.224 | runin_rate: 0.131; updating_rate: -0.082; retrospective_rate: -0.214 | 1.726 / NA / NA |
| updating_rate | 0.767 | 0.495 (232/14) | 0.537 | 0.846 | 0.812 | 0.383 | 0.328 | 0.675 | 0.004 | 11.094 | 2.146 | 0.208 | 0.207 | — | NA / NA / NA |
| recent_window_rate | 0.805 | 0.619 (232/14) | 0.682 | 0.832 | 0.834 | 0.514 | 0.538 | 0.692 | 0.231 | 13.439 | 10.323 | 0.187 | 0.189 | — | NA / NA / NA |
| persistence | 0.726 | 0.615 (232/14) | 0.756 | 0.695 | 0.754 | 0.451 | 0.650 | 0.579 | 0.229 | 16.553 | 15.000 | 0.271 | 0.276 | — | NA / NA / NA |
| retrospective_rate | 0.786 | 0.500 (232/14) | 0.613 | 0.823 | 0.810 | 0.435 | 0.384 | 0.616 | 0.000 | 12.561 | 0.000 | 0.185 | 0.184 | — | NA / NA / NA |
| logistic_regression | 0.847 | 0.691 (232/14) | 0.746 | 0.823 | 0.839 | 0.569 | 0.616 | 0.682 | 0.299 | 14.592 | 11.056 | 0.157 | 0.159 | runin_rate: 0.395; updating_rate: 0.246; retrospective_rate: 0.154 | 1.033 / 1.193 / 0.884 |
| gradient_boosting | 0.875 | 0.726 (232/14) | 0.773 | 0.841 | 0.857 | 0.613 | 0.676 | 0.708 | 0.384 | 14.785 | 11.315 | 0.141 | 0.143 | runin_rate: 0.456; updating_rate: 0.323; retrospective_rate: 0.240; 95% CI: runin_rate [0.420, 0.491]; updating_rate [0.295, 0.351]; retrospective_rate [0.206, 0.274] | 1.072 / 1.311 / 0.934 (95% CI: [1.019, 1.137]; skipped 0 / [1.221, 1.421]; skipped 0 / [0.881, 0.994]; skipped 0) |
| within_centred_gradient_boosting | 0.863 | 0.716 (232/14) | 0.752 | 0.837 | 0.851 | 0.589 | 0.670 | 0.709 | 0.379 | 14.631 | 11.026 | 0.147 | 0.148 | runin_rate: 0.431; updating_rate: 0.291; retrospective_rate: 0.204 | 1.107 / 1.399 / 0.955 |

## replace_bg

Patients: 45; observations: 9477; event rate: 0.060.

| Forecast | Pooled AUC | Within AUC (eligible/excluded) | Sensitivity | Specificity | PPV | Youden | Within sens. | Within spec. | Within Youden | Alarms/30 mean | Alarms/30 median | Brier pooled | Brier patient mean | Brier skill | Calibration slopes pooled/between/within |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| training_cohort_rate | 0.500 | 0.500 (42/3) | 1.000 | 0.000 | 0.060 | 0.000 | 1.000 | 0.000 | 0.000 | 30.000 | 30.000 | 0.056 | 0.057 | — | NA / NA / NA |
| runin_rate | 0.597 | 0.500 (42/3) | 0.681 | 0.513 | 0.082 | 0.194 | 0.524 | 0.476 | 0.000 | 14.667 | 0.000 | 0.061 | 0.061 | — | NA / NA / NA |
| runin_rate_shrunk_10 | 0.597 | 0.500 (42/3) | 0.681 | 0.513 | 0.082 | 0.194 | 0.524 | 0.476 | 0.000 | 14.667 | 0.000 | 0.057 | 0.057 | — | NA / NA / NA |
| expanding_rate | 0.707 | 0.387 (42/3) | 0.734 | 0.588 | 0.102 | 0.321 | 0.454 | 0.528 | -0.017 | 13.173 | 8.036 | 0.055 | 0.055 | — | NA / NA / NA |
| recent_28_night_rate | 0.684 | 0.469 (42/3) | 0.665 | 0.624 | 0.101 | 0.289 | 0.415 | 0.574 | -0.011 | 11.934 | 9.349 | 0.056 | 0.056 | — | NA / NA / NA |
| persistence | 0.537 | 0.506 (42/3) | 0.129 | 0.944 | 0.129 | 0.073 | 0.076 | 0.937 | 0.012 | 1.817 | 1.341 | 0.104 | 0.106 | — | NA / NA / NA |
| retrospective_rate_reference | 0.752 | 0.500 (42/3) | 0.737 | 0.647 | 0.117 | 0.384 | 0.405 | 0.595 | 0.000 | 11.333 | 0.000 | 0.053 | 0.054 | — | NA / NA / NA |
| logistic_regression | 0.818 | 0.750 (42/3) | 0.772 | 0.720 | 0.150 | 0.493 | 0.644 | 0.685 | 0.330 | 9.439 | 7.819 | 0.050 | 0.050 | runin_rate: 0.182; updating_rate: 0.093; retrospective_rate: 0.073 | 1.153 / 1.440 / 1.028 |
| gradient_boosting | 0.829 | 0.759 (42/3) | 0.702 | 0.780 | 0.169 | 0.482 | 0.597 | 0.753 | 0.349 | 7.570 | 6.849 | 0.048 | 0.049 | runin_rate: 0.208; updating_rate: 0.122; retrospective_rate: 0.103; 95% CI: runin_rate [0.153, 0.273]; updating_rate [0.079, 0.166]; retrospective_rate [0.060, 0.148] | 1.059 / 1.524 / 0.952 (95% CI: [0.951, 1.174]; skipped 0 / [1.311, 1.844]; skipped 0 / [0.814, 1.088]; skipped 0) |
| within_patient_centred_gradient_boosting | 0.785 | 0.760 (42/3) | 0.665 | 0.755 | 0.147 | 0.420 | 0.623 | 0.738 | 0.361 | 8.191 | 7.886 | 0.051 | 0.051 | runin_rate: 0.163; updating_rate: 0.072; retrospective_rate: 0.052 | 1.138 / 1.365 / 1.098 |

## replace_bg_v1panel

Patients: 45; observations: 9477; event rate: 0.060.

| Forecast | Pooled AUC | Within AUC (eligible/excluded) | Sensitivity | Specificity | PPV | Youden | Within sens. | Within spec. | Within Youden | Alarms/30 mean | Alarms/30 median | Brier pooled | Brier patient mean | Brier skill | Calibration slopes pooled/between/within |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| training_cohort_rate | 0.500 | 0.500 (42/3) | 1.000 | 0.000 | 0.060 | 0.000 | 1.000 | 0.000 | 0.000 | 30.000 | 30.000 | 0.056 | 0.057 | — | NA / NA / NA |
| runin_rate | 0.597 | 0.500 (42/3) | 0.681 | 0.513 | 0.082 | 0.194 | 0.524 | 0.476 | 0.000 | 14.667 | 0.000 | 0.061 | 0.061 | — | NA / NA / NA |
| runin_rate_shrunk_10 | 0.597 | 0.500 (42/3) | 0.681 | 0.513 | 0.082 | 0.194 | 0.524 | 0.476 | 0.000 | 14.667 | 0.000 | 0.057 | 0.057 | — | NA / NA / NA |
| expanding_rate | 0.707 | 0.387 (42/3) | 0.734 | 0.588 | 0.102 | 0.321 | 0.454 | 0.528 | -0.017 | 13.173 | 8.036 | 0.055 | 0.055 | — | NA / NA / NA |
| recent_28_night_rate | 0.684 | 0.469 (42/3) | 0.665 | 0.624 | 0.101 | 0.289 | 0.415 | 0.574 | -0.011 | 11.934 | 9.349 | 0.056 | 0.056 | — | NA / NA / NA |
| persistence | 0.537 | 0.506 (42/3) | 0.129 | 0.944 | 0.129 | 0.073 | 0.076 | 0.937 | 0.012 | 1.817 | 1.341 | 0.104 | 0.106 | — | NA / NA / NA |
| retrospective_rate_reference | 0.752 | 0.500 (42/3) | 0.737 | 0.647 | 0.117 | 0.384 | 0.405 | 0.595 | 0.000 | 11.333 | 0.000 | 0.053 | 0.054 | — | NA / NA / NA |
| logistic_regression | 0.818 | 0.750 (42/3) | 0.772 | 0.721 | 0.150 | 0.493 | 0.644 | 0.685 | 0.330 | 9.436 | 7.819 | 0.050 | 0.050 | runin_rate: 0.182; updating_rate: 0.093; retrospective_rate: 0.073 | 1.153 / 1.440 / 1.028 |
| gradient_boosting | 0.828 | 0.758 (42/3) | 0.707 | 0.778 | 0.168 | 0.485 | 0.604 | 0.750 | 0.354 | 7.643 | 6.704 | 0.048 | 0.049 | runin_rate: 0.206; updating_rate: 0.120; retrospective_rate: 0.101; 95% CI: runin_rate [0.152, 0.273]; updating_rate [0.078, 0.165]; retrospective_rate [0.058, 0.147] | 1.057 / 1.528 / 0.950 (95% CI: [0.949, 1.172]; skipped 0 / [1.315, 1.846]; skipped 0 / [0.811, 1.088]; skipped 0) |
| within_patient_centred_gradient_boosting | 0.785 | 0.759 (42/3) | 0.663 | 0.755 | 0.147 | 0.419 | 0.622 | 0.739 | 0.361 | 8.176 | 7.543 | 0.051 | 0.051 | runin_rate: 0.163; updating_rate: 0.072; retrospective_rate: 0.052 | 1.137 / 1.372 / 1.095 |
