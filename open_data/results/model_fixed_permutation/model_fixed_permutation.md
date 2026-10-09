# Model-fixed permutation reference for per-patient AUC

Each fitted model was held fixed while each test patient's scored outcomes were shuffled within patient. Run-in outcomes stayed fixed; history features were rebuilt before rescoring. No model was refit during a permutation.

Seed: `20261009`; permutations: `200`; patient-bootstrap resamples: `2000`.

## Results

The reference columns give the mean and 2.5th–97.5th percentiles across permutations. The observed-minus-reference bootstrap resamples patients; it recomputes the permutation mean of the median within each patient sample and uses the patient-specific permutation means for the mean-AUC contrast.

| Cohort | Model | Eligible patients | Observed median AUC | Observed mean AUC | Observed above 0.5 | Reference median AUC mean (2.5–97.5%) | Reference share above 0.5 mean (2.5–97.5%) | Median difference (95% patient-bootstrap CI) | Mean difference (95% patient-bootstrap CI) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| REPLACE-BG | Logistic Regression | 42 | 0.775 | 0.750 | 97.6% | 0.491 (0.457–0.527) | 46.1% (31.0%–61.9%) | 0.284 (0.233–0.331) | 0.259 (0.224–0.291) |
| REPLACE-BG | Gradient Boosting | 42 | 0.780 | 0.759 | 97.6% | 0.486 (0.449–0.520) | 44.4% (28.6%–59.5%) | 0.293 (0.236–0.330) | 0.274 (0.246–0.304) |
| CTN-0094 weekly | Logistic Regression | 154 | 0.696 | 0.653 | 70.8% | 0.407 (0.372–0.440) | 30.2% (23.4%–37.0%) | 0.289 (0.234–0.339) | 0.245 (0.206–0.284) |
| CTN-0094 weekly | Gradient Boosting | 154 | 0.678 | 0.638 | 65.6% | 0.411 (0.371–0.446) | 31.1% (23.4%–38.3%) | 0.267 (0.181–0.329) | 0.230 (0.187–0.272) |
| College Experience | Logistic Regression | 37 | 0.820 | 0.806 | 94.6% | 0.490 (0.466–0.513) | 43.1% (27.0%–59.5%) | 0.329 (0.303–0.379) | 0.316 (0.274–0.355) |
| College Experience | Gradient Boosting | 37 | 0.817 | 0.800 | 94.6% | 0.489 (0.467–0.512) | 42.3% (27.0%–56.8%) | 0.329 (0.307–0.361) | 0.316 (0.275–0.353) |

## Reproduction checks

The observed mean, median, share above 0.5, and eligible-patient count were compared with `open_data/cases_extra_results.json` for both models. For GB, Table 4's rounded median, share above 0.5, and mean per-patient AUC were also checked.

| Cohort | GB median AUC | GB share above 0.5 | GB mean AUC | Cases-extra checks |
|---|---:|---:|---:|---|
| REPLACE-BG | 0.780 | 97.6% | 0.759 | passed; Table 4 values reproduced after rounding |
| CTN-0094 weekly | 0.678 | 65.6% | 0.638 | passed; Table 4 values reproduced after rounding |
| College Experience | 0.817 | 94.6% | 0.800 | passed; Table 4 values reproduced after rounding |

## Outcome-history features

- **REPLACE-BG:** `history_rate_7, history_rate_28, expanding_rate, runin_rate` depend on past outcomes. `runin_rate` is held fixed. Rates use prior eligible nights only; the 14-night run-in remains fixed.
- **CTN-0094 weekly:** `previous_result, runin_rate, expanding_rate, recent_rate` depend on past outcomes. `runin_rate` is held fixed. Run-in rate is fixed. Expanding rate is positive weeks divided by screens, as in the panel builder (identity check: rebuilding from the unpermuted outcomes reproduces every forecast); recent rate averages observed outcomes in the prior four calendar weeks. Previous-observed status and expanding_n describe availability, not outcome values.
- **College Experience:** `last4_rate, last12_rate, expanding_rate, runin_rate` depend on past outcomes. `runin_rate` is held fixed. Run-in comprises the first 20 report outcomes; the first forecast origin (report 21) is also held fixed, matching the existing report-history timing. Subsequent rates are rebuilt from permuted scored targets.

The builders in the existing cohort scripts calculate histories during panel construction and do not accept a replacement outcome sequence. This analysis uses small helpers in the new script to rebuild only those history columns; no existing source file was changed.

## Interpretation and calculation

For every permutation, the median and share above 0.5 were recomputed across patients with both target classes. Their reported reference means are averages of these permutation-level statistics; percentile limits are the 2.5th and 97.5th percentiles of the permutation-level statistics. The median bootstrap interval resamples patients and, within each sample, compares the observed median with the mean permutation-level median. The mean-AUC bootstrap uses the already-computed patient-specific permutation mean AUCs. The JSON also reports the paired median difference and interval using the median of those patient-specific permutation means.

For exact reproduction, run from the repository root:

```powershell
python open_data/cases_model_fixed_permutation.py
```

`--quick` uses 5 permutations and 100 bootstrap resamples for a smoke run. The full run writes `open_data/results/model_fixed_permutation/model_fixed_permutation.json` and this report.
