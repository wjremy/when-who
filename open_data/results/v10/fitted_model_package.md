# Results 5.3: the CTN-0094 gradient-boosting model, evaluated with wheneval 3.0

Test patients 246, scored weeks 4237 (weeks 4 onward; the four run-in weeks enter only through the fixed history features). Threshold = training-cohort event rate 0.320. 200 within-patient shuffles per patient, seed 20261010; fitted model fixed, history features rebuilt in every shuffle; yardstick = patient's own scored rate, rebuilt per shuffle; chance level = mean over shuffles. Run time 60.2 min.

| Measure | Pooled | Usual per patient, mean (median) | Fixed: mean observed | Mean chance | Median excess [IQR] | Patients above chance (p < 0.05) | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| AUC | 0.929 | 0.638 (0.690) | 0.638 | 0.399 | 0.258 [0.051, 0.455] | 36% | 154 |
| Average precision | 0.852 | 0.682 (0.798) | 0.682 | 0.513 | 0.103 [0.000, 0.300] | 32% | 168 |
| Sensitivity | 0.854 | 0.702 (0.875) | 0.675 | 0.503 | 0.023 [0.000, 0.315] | 15% | 154 |
| Specificity | 0.871 | 0.721 (0.909) | 0.585 | 0.467 | 0.041 [0.000, 0.199] | 23% | 154 |
| PPV (excess over own rate) | 0.739 | 0.607 (0.667) | 0.136 | -0.014 | 0.026 [0.000, 0.271] | 25% | 166 |
| NPV (excess over own non-event rate) | 0.933 | 0.877 (1.000) | 0.077 | -0.022 | 0.000 [0.000, 0.145] | 17% | 203 |
| Net benefit (vs own-rate policy) | 0.214 | 0.236 (0.032) | 0.011 | -0.031 | 0.000 [0.000, 0.065] | 17% | 246 |
| Brier skill (vs own rate) | 0.557 | 0.568 (0.664) | 0.013 | -0.326 | 0.269 [0.035, 0.556] | 40% | 154 |
| Calibration slope, conditional within-patient (all patients) | 1.081 | 0.671 | 0.671 | -0.177 | 0.848 (p = 0.005) | | 137 |

Pooled and usual per-patient values are the package's comparison columns (usual references: 0.5 for AUC, population base rate for AP, PPV and NPV, population rate for Brier skill, treat-all/none for net benefit).

## Check against Table 4

AUC chance level here: mean 0.399, median over patients 0.426. Table 4 CTN-0094 AUC chance entry: see `table4.json` (key printed below).

Table 4 CTN-0094 gradient-boosting AUC chance (mean over 200 model-fixed draws of the mean per-patient AUC): 0.3996 [2.5-97.5%: 0.365, 0.431]. Package: 0.3995. Same estimand (mean over patients of each patient's mean over shuffles), different shuffle draws.

## The call

```python
result = evaluate(data, patient_id="patient", time="week", outcome="y", prediction="forecast",
                  threshold=0.31957, history_rule=rebuild_gb, n_shuffles=200, seed=20261010)
print(to_markdown(result))
```

`rebuild_gb(y, scored, rows)` receives one patient's time-ordered outcomes and rows, rebuilds the history features with `ctn_rebuild_histories` and returns the fixed model's predictions; the package checks that it reproduces the forecast column on the observed outcomes.

## Package output (to_markdown)

| Metric | Score (mean) | Shuffled baseline (mean) | Excess (median [IQR]) | Excess > 0 | p < 0.05 | Patients | Pooled score (for comparison) | Per-patient score (mean; for comparison) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| auc | 0.638 | 0.399 | 0.258 [0.051, 0.455] | 81% | 36% | 154 | 0.929 | 0.638 |
| average_precision | 0.682 | 0.513 | 0.103 [0.000, 0.300] | 71% | 32% | 168 | 0.852 | 0.682 |
| sensitivity | 0.675 | 0.503 | 0.023 [0.000, 0.315] | 66% | 15% | 154 | 0.854 | 0.702 |
| specificity | 0.585 | 0.467 | 0.041 [0.000, 0.199] | 67% | 23% | 154 | 0.871 | 0.721 |
| ppv | 0.136 | -0.014 | 0.026 [0.000, 0.271] | 66% | 25% | 166 | 0.739 | 0.607 |
| npv | 0.077 | -0.022 | 0.000 [0.000, 0.145] | 48% | 17% | 203 | 0.933 | 0.877 |
| net_benefit | 0.011 | -0.031 | 0.000 [0.000, 0.065] | 48% | 17% | 246 | 0.214 | 0.236 |
| brier_skill | 0.013 | -0.326 | 0.269 [0.035, 0.556] | 81% | 40% | 154 | 0.557 | 0.568 |
| Calibration slope (conditional, all patients) | 0.671 | -0.177 | 0.848 (p = 0.005) |  |  | 137 | 1.081 |  |

Threshold 0.320; 200 within-patient shuffles per patient (seed 20261010); history rule: callable; yardstick: patient_rate.
