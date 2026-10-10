# Table 4. Pooled and per-patient performance in the clinical test sets

The model is gradient boosting. Per-patient cells show the median (IQR), followed by the permutation chance mean (2.5th–97.5th percentile), except Brier skill (equal-patient ratio of mean losses) and calibration slope (one conditional within-patient slope). Run-in cells show pooled performance and the observed per-patient summary.

| Metric | REPLACE-BG model pooled | REPLACE-BG model per patient | REPLACE-BG run-in pooled (per-patient summary) | CTN-0094 model pooled | CTN-0094 model per patient | CTN-0094 run-in pooled (per-patient summary) | College Experience model pooled | College Experience model per patient | College Experience run-in pooled (per-patient summary) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AUC | 0.830 | 0.775 (0.684–0.828) [0.485 (0.446–0.518)] | 0.609; 0.500 (0.500–0.500) | 0.929 | 0.690 (0.462–0.880) [0.400 (0.365–0.431)] | 0.801; 0.500 (0.500–0.500) | 0.935 | 0.817 (0.771–0.892) [0.484 (0.456–0.513)] | 0.695; 0.500 (0.500–0.500) |
| Average precision | 0.325† | 0.284 (0.135–0.454) [0.085 (0.074–0.098)] | 0.081; 0.046 (0.025–0.095) | 0.852 | 0.759 (0.419–0.903) [0.469 (0.449–0.489)] | 0.600; 0.392 (0.155–0.667) | 0.910 | 0.813 (0.556–0.927) [0.464 (0.455–0.475)] | 0.600; 0.444 (0.154–0.687) |
| Sensitivity | 0.702 | 0.655 (0.500–0.734) [0.260 (0.221–0.299)] | 0.726; 1.000 (0.000–1.000) | 0.854 | 0.857 (0.350–1.000) [0.502 (0.481–0.530)] | 0.814; 1.000 (0.000–1.000) | 0.858 | 0.808 (0.595–0.929) [0.515 (0.504–0.526)] | 0.432; 0.000 (0.000–1.000) |
| Specificity | 0.784 | 0.782 (0.647–0.899) [0.732 (0.724–0.741)] | 0.497; 0.000 (0.000–1.000) | 0.871 | 0.699 (0.149–0.927) [0.467 (0.452–0.482)] | 0.653; 0.000 (0.000–1.000) | 0.862 | 0.802 (0.452–0.953) [0.482 (0.473–0.494)] | 0.781; 1.000 (0.000–1.000) |
| PPV (pooled [base rate]; per patient: excess over own rate) | 0.172 [0.060] | 0.074 (0.041–0.104) [-0.002 (-0.010–0.008)] | 0.085 [0.060]; 0.000 (0.000–0.000) | 0.739 [0.300] | 0.000 (0.000–0.252) [-0.015 (-0.031–0.004)] | 0.501 [0.300]; 0.000 (0.000–0.000) | 0.805 [0.399] | 0.253 (0.067–0.335) [-0.004 (-0.022–0.019)] | 0.567 [0.399]; 0.000 (0.000–0.000) |
| NPV (pooled [base rate]; per patient: excess over own rate) | 0.976 [0.940] | 0.020 (0.006–0.044) [-0.001 (-0.005–0.004)] | 0.966 [0.940]; 0.000 (0.000–0.000) | 0.933 [0.700] | 0.000 (0.000–0.106) [-0.021 (-0.038–-0.005)] | 0.891 [0.700]; 0.000 (0.000–0.000) | 0.901 [0.601] | 0.143 (0.029–0.287) [-0.008 (-0.036–0.029)] | 0.674 [0.601]; 0.000 (0.000–0.000) |
| Net benefit | 0.030 | 0.010 (0.000–0.016) [-0.011 (-0.013–-0.009)] | 0.016; 0.000 (0.000–0.001) | 0.214 | 0.000 (0.000–0.045) [-0.006 (-0.012–-0.001)] | 0.130; 0.000 (0.000–0.000) | 0.288 | 0.074 (0.000–0.116) [-0.010 (-0.016–-0.004)] | 0.087; 0.000 (-0.041–0.000) |
| Brier skill | 0.154 | 0.126 [-0.115 (-0.132–-0.096)] | -0.070; -0.099 | 0.558 | 0.236 [-0.033 (-0.062–-0.006)] | 0.123; -0.512 | 0.595 | 0.372 [-0.088 (-0.107–-0.068)] | 0.011; -0.461 |
| Calibration slope | 1.063 | 0.966 [-0.026 (-0.121–0.072)] | 0.080; not estimable (constant forecast) | 1.081 | 0.671 [-0.176 (-0.284–-0.088)] | 0.113; not estimable (constant forecast) | 1.072 | 0.981 [-0.018 (-0.081–0.039)] | 0.165; not estimable (constant forecast) |

† The paired patient-bootstrap 95% interval for pooled minus per-patient mean includes zero.

Yardsticks: pooled Brier skill cells use `training_cohort_rate`; per-patient Brier skill uses the updated rate as 1 - equal-patient mean model Brier loss / equal-patient mean updated-rate Brier loss; per-patient net benefit is model NB minus treat-all NB `r̄ - (1-r̄)t/(1-t)` when that patient's mean row-level updated rate `r̄ >= t`, otherwise treat-none NB (0).
Calibration slope is one conditional within-patient slope fitted with patient intercepts.

Eligibility counts are shown as model/run-in. For AUC, average precision, sensitivity, specificity and calibration slope, eligibility follows the nine-metric package definitions; PPV requires at least one alarm and NPV at least one non-alarm. Net benefit includes all patients with usable forecasts; Brier skill excludes patients with a zero yardstick Brier score.

- **AUC eligible n (model/run-in):** REPLACE-BG 42/42; CTN-0094 154/154; College Experience 37/37.
- **Average precision eligible n (model/run-in):** REPLACE-BG 42/42; CTN-0094 154/154; College Experience 37/37.
- **Sensitivity eligible n (model/run-in):** REPLACE-BG 42/42; CTN-0094 154/154; College Experience 37/37.
- **Specificity eligible n (model/run-in):** REPLACE-BG 42/42; CTN-0094 154/154; College Experience 37/37.
- **PPV (pooled [base rate]; per patient: excess over own rate) eligible n (model/run-in):** REPLACE-BG 43/23; CTN-0094 166/124; College Experience 37/14.
- **NPV (pooled [base rate]; per patient: excess over own rate) eligible n (model/run-in):** REPLACE-BG 45/22; CTN-0094 203/122; College Experience 35/24.
- **Net benefit eligible n (model/run-in):** REPLACE-BG 45/45; CTN-0094 246/246; College Experience 38/38.
- **Brier skill eligible n (model/run-in):** REPLACE-BG 42/42; CTN-0094 246/246; College Experience 37/37.
- **Calibration slope eligible n (model/run-in):** REPLACE-BG 42/42; CTN-0094 154/154; College Experience 37/37.

Alarm thresholds equal each training-cohort event rate; values from the forecast files (and `cases_tables.json`): REPLACE-BG 0.054686; CTN-0094 0.319570; College Experience 0.393980.
The updated-rate yardsticks are REPLACE-BG `expanding_rate`, CTN-0094 `updating_rate` (an alias of the rebuilt `expanding_rate`), and College Experience `expanding_rate`.

## Run-in-rate yardstick sensitivity

Brier skill and net benefit are recomputed with run-in rate as the row-aligned yardstick. PPV/NPV patient excesses are re-expressed against each patient's run-in event/non-event rate; pooled PPV/NPV remain raw pooled predictive values and show the test-set population base rate in brackets.

| Metric | REPLACE-BG model (pooled / patient mean) | REPLACE-BG run-in (pooled / patient mean) | CTN-0094 model (pooled / patient mean) | CTN-0094 run-in (pooled / patient mean) | College Experience model (pooled / patient mean) | College Experience run-in (pooled / patient mean) |
|---|---:|---:|---:|---:|---:|---:|
| Brier skill | 0.209 / 0.205 | 0.000 / 0.000 | 0.496 / 0.495 | 0.000 / 0.000 | 0.590 / 0.570 | 0.000 / 0.000 |
| PPV excess vs run-in | 0.172 [0.060] / 0.066 | 0.085 [0.060] / -0.049 | 0.739 [0.300] / 0.056 | 0.501 [0.300] / -0.214 | 0.805 [0.399] / 0.372 | 0.567 [0.399] / 0.009 |
| NPV excess vs run-in | 0.976 [0.940] / 0.040 | 0.966 [0.940] / -0.036 | 0.933 [0.700] / 0.153 | 0.891 [0.700] / -0.076 | 0.901 [0.601] / 0.043 | 0.674 [0.601] / -0.206 |
| Net benefit | 0.030 / 0.014 | 0.016 / 0.000 | 0.214 / 0.084 | 0.130 / 0.000 | 0.288 / 0.191 | 0.087 / 0.000 |

Run-in-yardstick per-patient permutation means (2.5th–97.5th percentile) for the model are in `table4.json`; the primary table's bracketed chance levels use the rebuilt updated-rate yardstick.
