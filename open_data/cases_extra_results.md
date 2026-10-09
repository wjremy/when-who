# Extra patient variation and boosting-round analyses

Patient AUC includes patients with both outcome classes. Sampling variance uses the Hanley-McNeil approximation. The Brier comparison bootstrap resamples patients. For the plateau summary, the reported round is the first sampled round with less than 0.002 maximum remaining AUC gain over later sampled rounds.

## Test-patient variation

| Dataset | Model | Eligible patients | Mean AUC | SD | Min | AUC > .5 | AUC > .6 | AUC > .7 | True AUC SD estimate | Sampling share | Model Brier wins (95% patient bootstrap) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| replace_bg | gradient_boosting | 42 | 0.7587 | 0.1020 | 0.4617 | 0.9762 | 0.9762 | 0.6905 | 0.0000 | 1.0035 | 0.7556 [0.6222, 0.8667] (n=45) |
| replace_bg | logistic_regression | 42 | 0.7504 | 0.1154 | 0.3243 | 0.9762 | 0.9524 | 0.6667 | 0.0543 | 0.7787 | 0.7778 [0.6444, 0.8889] (n=45) |
| weekly_missing_excluded | gradient_boosting | 154 | 0.6379 | 0.2803 | 0.0000 | 0.6558 | 0.5714 | 0.4740 | 0.2320 | 0.3152 | 0.5447 [0.4797, 0.6098] (n=246) |
| weekly_missing_excluded | logistic_regression | 154 | 0.6527 | 0.2522 | 0.0000 | 0.7078 | 0.6299 | 0.4805 | 0.1893 | 0.4367 | 0.5488 [0.4878, 0.6138] (n=246) |
| weekly_missing_positive | gradient_boosting | 232 | 0.7230 | 0.1867 | 0.0000 | 0.8664 | 0.7974 | 0.6681 | 0.1432 | 0.4121 | 0.8252 [0.7763, 0.8699] (n=246) |
| weekly_missing_positive | logistic_regression | 232 | 0.6887 | 0.1940 | 0.0000 | 0.8621 | 0.7371 | 0.5388 | 0.1536 | 0.3732 | 0.8008 [0.7480, 0.8496] (n=246) |
| ces | gradient_boosting | 37 | 0.8003 | 0.1277 | 0.4619 | 0.9459 | 0.8919 | 0.8108 | 0.1078 | 0.2873 | 0.8684 [0.7632, 0.9737] (n=38) |
| ces | logistic_regression | 37 | 0.8056 | 0.1269 | 0.4381 | 0.9459 | 0.9189 | 0.7838 | 0.1086 | 0.2668 | 0.8421 [0.7105, 0.9474] (n=38) |

Deciles, observed and mean sampling variances, and per-seed details are in the JSON output.

## Validation over 500 boosting rounds

The CTN-0094 curve uses staged `GradientBoostingClassifier` because the source model is `HistGradientBoostingClassifier` and does not provide the required staged prediction API. It carries over learning rate .08 and max leaf nodes 15; classic GBC has no `l2_regularization` option. REPLACE-BG uses its source GBC settings.

| Dataset | Metric | Mean best round | Range of best rounds across seeds |
|---|---|---:|---:|
| replace_bg | pooled_auc | 154.0 | [120, 190] |
| replace_bg | mean_within_patient_auc | 274.0 | [150, 500] |
| replace_bg | pooled_log_loss | 184.0 | [140, 250] |
| replace_bg | pooled_brier | 204.0 | [120, 340] |
| replace_bg | patient_mean_brier | 158.0 | [130, 190] |
| weekly_missing_excluded | pooled_auc | 120.0 | [90, 150] |
| weekly_missing_excluded | mean_within_patient_auc | 86.0 | [30, 110] |
| weekly_missing_excluded | pooled_log_loss | 126.0 | [110, 150] |
| weekly_missing_excluded | pooled_brier | 102.0 | [80, 110] |
| weekly_missing_excluded | patient_mean_brier | 102.0 | [80, 110] |
| weekly_missing_positive | pooled_auc | 214.0 | [140, 270] |
| weekly_missing_positive | mean_within_patient_auc | 462.0 | [410, 500] |
| weekly_missing_positive | pooled_log_loss | 284.0 | [230, 330] |
| weekly_missing_positive | pooled_brier | 254.0 | [210, 310] |
| weekly_missing_positive | patient_mean_brier | 254.0 | [210, 310] |

### replace_bg: AUC by round

| Round | Pooled AUC mean (range) | Mean within-patient AUC mean (range) |
|---:|---:|---:|
| 10 | 0.7510 (0.7426–0.7580) | 0.6600 (0.6540–0.6636) |
| 50 | 0.7785 (0.7765–0.7803) | 0.6953 (0.6921–0.6989) |
| 100 | 0.7859 (0.7852–0.7863) | 0.7010 (0.6994–0.7019) |
| 200 | 0.7867 (0.7851–0.7879) | 0.7030 (0.6996–0.7049) |
| 300 | 0.7854 (0.7841–0.7865) | 0.7044 (0.7034–0.7054) |
| 500 | 0.7833 (0.7815–0.7850) | 0.7041 (0.7025–0.7055) |

AUC plateau rounds: pooled mean 100.0 (range [90, 110]); within-patient mean 154.0 (range [110, 260]).

### weekly_missing_excluded: AUC by round

| Round | Pooled AUC mean (range) | Mean within-patient AUC mean (range) |
|---:|---:|---:|
| 10 | 0.9086 (0.9078–0.9092) | 0.6432 (0.6390–0.6460) |
| 50 | 0.9156 (0.9150–0.9162) | 0.6611 (0.6572–0.6677) |
| 100 | 0.9166 (0.9159–0.9173) | 0.6639 (0.6560–0.6684) |
| 200 | 0.9160 (0.9153–0.9166) | 0.6585 (0.6546–0.6614) |
| 300 | 0.9150 (0.9144–0.9155) | 0.6561 (0.6525–0.6599) |
| 500 | 0.9130 (0.9128–0.9131) | 0.6545 (0.6492–0.6584) |

AUC plateau rounds: pooled mean 44.0 (range [40, 50]); within-patient mean 54.0 (range [30, 90]).

### weekly_missing_positive: AUC by round

| Round | Pooled AUC mean (range) | Mean within-patient AUC mean (range) |
|---:|---:|---:|
| 10 | 0.8475 (0.8464–0.8483) | 0.7099 (0.7061–0.7161) |
| 50 | 0.8553 (0.8548–0.8556) | 0.7188 (0.7161–0.7225) |
| 100 | 0.8593 (0.8585–0.8599) | 0.7305 (0.7290–0.7318) |
| 200 | 0.8602 (0.8590–0.8610) | 0.7348 (0.7335–0.7374) |
| 300 | 0.8600 (0.8591–0.8604) | 0.7379 (0.7365–0.7393) |
| 500 | 0.8580 (0.8573–0.8587) | 0.7405 (0.7388–0.7429) |

AUC plateau rounds: pooled mean 88.0 (range [80, 90]); within-patient mean 330.0 (range [260, 380]).
