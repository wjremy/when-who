# Extra patient variation and boosting-round analyses

Patient AUC includes patients with both outcome classes. Sampling variance uses the Hanley-McNeil approximation. The Brier comparison bootstrap resamples patients. For the plateau summary, the reported round is the first sampled round with less than 0.002 maximum remaining AUC gain over later sampled rounds.

## Test-patient variation

| Dataset | Model | Eligible patients | Mean AUC | SD | Min | AUC > .5 | AUC > .6 | AUC > .7 | True AUC SD estimate | Sampling share | Model Brier wins (95% patient bootstrap) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| replace_bg | gradient_boosting | 42 | 0.7594 | 0.1014 | 0.4707 | 0.9762 | 0.9762 | 0.7143 | 0.0000 | 1.0321 | 0.7333 [0.6000, 0.8450] (n=45) |
| replace_bg | logistic_regression | 42 | 0.7513 | 0.1161 | 0.3153 | 0.9762 | 0.9286 | 0.6667 | 0.0559 | 0.7683 | 0.7556 [0.6222, 0.8667] (n=45) |
| weekly_missing_excluded | gradient_boosting | 154 | 0.6380 | 0.2787 | 0.0000 | 0.6623 | 0.5649 | 0.4805 | 0.2298 | 0.3204 | 0.5325 [0.4715, 0.5976] (n=246) |
| weekly_missing_excluded | logistic_regression | 154 | 0.6470 | 0.2526 | 0.0000 | 0.7013 | 0.6104 | 0.4545 | 0.1890 | 0.4402 | 0.5488 [0.4878, 0.6138] (n=246) |
| weekly_missing_positive | gradient_boosting | 232 | 0.7230 | 0.1867 | 0.0000 | 0.8664 | 0.7974 | 0.6681 | 0.1432 | 0.4121 | 0.8252 [0.7763, 0.8699] (n=246) |
| weekly_missing_positive | logistic_regression | 232 | 0.6887 | 0.1940 | 0.0000 | 0.8621 | 0.7371 | 0.5388 | 0.1536 | 0.3732 | 0.8008 [0.7480, 0.8496] (n=246) |
| ces | gradient_boosting | 37 | 0.8003 | 0.1277 | 0.4619 | 0.9459 | 0.8919 | 0.8108 | 0.1078 | 0.2873 | 0.8684 [0.7632, 0.9737] (n=38) |
| ces | logistic_regression | 37 | 0.8056 | 0.1269 | 0.4381 | 0.9459 | 0.9189 | 0.7838 | 0.1086 | 0.2668 | 0.8421 [0.7105, 0.9474] (n=38) |

Deciles, observed and mean sampling variances, and per-seed details are in the JSON output.

## Validation over 500 boosting rounds

The CTN-0094 curve uses staged `GradientBoostingClassifier` because the source model is `HistGradientBoostingClassifier` and does not provide the required staged prediction API. It carries over learning rate .08 and max leaf nodes 15; classic GBC has no `l2_regularization` option. REPLACE-BG uses its source GBC settings.

| Dataset | Metric | Mean best round | Range of best rounds across seeds |
|---|---|---:|---:|
| replace_bg | pooled_auc | 158.0 | [130, 220] |
| replace_bg | mean_within_patient_auc | 350.0 | [110, 480] |
| replace_bg | pooled_log_loss | 214.0 | [170, 270] |
| replace_bg | pooled_brier | 172.0 | [130, 210] |
| replace_bg | patient_mean_brier | 170.0 | [130, 210] |
| weekly_missing_excluded | pooled_auc | 106.0 | [90, 150] |
| weekly_missing_excluded | mean_within_patient_auc | 52.0 | [30, 90] |
| weekly_missing_excluded | pooled_log_loss | 98.0 | [90, 120] |
| weekly_missing_excluded | pooled_brier | 90.0 | [90, 90] |
| weekly_missing_excluded | patient_mean_brier | 90.0 | [90, 90] |
| weekly_missing_positive | pooled_auc | 214.0 | [140, 270] |
| weekly_missing_positive | mean_within_patient_auc | 480.0 | [410, 500] |
| weekly_missing_positive | pooled_log_loss | 284.0 | [230, 330] |
| weekly_missing_positive | pooled_brier | 248.0 | [210, 280] |
| weekly_missing_positive | patient_mean_brier | 248.0 | [210, 280] |

### replace_bg: AUC by round

| Round | Pooled AUC mean (range) | Mean within-patient AUC mean (range) |
|---:|---:|---:|
| 10 | 0.7441 (0.7387–0.7498) | 0.6592 (0.6537–0.6627) |
| 50 | 0.7789 (0.7766–0.7803) | 0.6964 (0.6893–0.7002) |
| 100 | 0.7857 (0.7852–0.7864) | 0.7032 (0.7014–0.7052) |
| 200 | 0.7869 (0.7857–0.7879) | 0.7066 (0.7058–0.7071) |
| 300 | 0.7855 (0.7840–0.7871) | 0.7071 (0.7055–0.7082) |
| 500 | 0.7835 (0.7809–0.7859) | 0.7072 (0.7047–0.7091) |

AUC plateau rounds: pooled mean 100.0 (range [80, 110]); within-patient mean 164.0 (range [90, 220]).

### weekly_missing_excluded: AUC by round

| Round | Pooled AUC mean (range) | Mean within-patient AUC mean (range) |
|---:|---:|---:|
| 10 | 0.9093 (0.9089–0.9098) | 0.6459 (0.6385–0.6509) |
| 50 | 0.9156 (0.9152–0.9160) | 0.6610 (0.6575–0.6647) |
| 100 | 0.9169 (0.9164–0.9172) | 0.6599 (0.6575–0.6639) |
| 200 | 0.9162 (0.9158–0.9170) | 0.6590 (0.6519–0.6653) |
| 300 | 0.9149 (0.9145–0.9155) | 0.6566 (0.6520–0.6593) |
| 500 | 0.9126 (0.9118–0.9132) | 0.6537 (0.6496–0.6575) |

AUC plateau rounds: pooled mean 48.0 (range [40, 50]); within-patient mean 44.0 (range [30, 70]).

### weekly_missing_positive: AUC by round

| Round | Pooled AUC mean (range) | Mean within-patient AUC mean (range) |
|---:|---:|---:|
| 10 | 0.8475 (0.8464–0.8483) | 0.7099 (0.7061–0.7161) |
| 50 | 0.8553 (0.8548–0.8556) | 0.7188 (0.7161–0.7225) |
| 100 | 0.8593 (0.8585–0.8599) | 0.7305 (0.7290–0.7318) |
| 200 | 0.8602 (0.8590–0.8610) | 0.7348 (0.7335–0.7373) |
| 300 | 0.8599 (0.8591–0.8604) | 0.7379 (0.7364–0.7393) |
| 500 | 0.8579 (0.8571–0.8586) | 0.7406 (0.7391–0.7430) |

AUC plateau rounds: pooled mean 88.0 (range [80, 90]); within-patient mean 328.0 (range [260, 380]).
