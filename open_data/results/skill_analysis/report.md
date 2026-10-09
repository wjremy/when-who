# Brier skill against prospective rate references

Gradient-boosting forecasts were scored in the three held-out test cohorts. Skill is 1 − model Brier loss / reference Brier loss. Pooled skill uses summed patient-day losses; patient-mean skill is the unweighted mean of patient-specific skill ratios. The positive-share column is the fraction of patients with positive patient-specific skill. Patient-level summaries omit references with mean patient Brier loss ≤ 1e-8, which treats the exported 1e-6 probability clipping as effectively zero reference loss; the valid patient count is shown. Parentheses contain percentile 95% CIs from 1,000 patient bootstrap resamples, seed 20261007.

The best prospective rate was selected separately per cohort by the lowest full-cohort pooled Brier loss among population event rate, run-in rate, updating rate, and shrunk run-in rate. That selected reference was held fixed within bootstrap resamples.

| Cohort | Reference | Patients with usable patient ratio | Pooled skill (95% CI) | Patient-mean skill (95% CI) | Positive patient share (95% CI) |
|---|---|---:|---:|---:|---:|
| REPLACE-BG | Population event rate | 45/45 | 0.148 (0.103, 0.196) | 0.172 (0.113, 0.244) | 0.889 (0.800, 0.978) |
| REPLACE-BG | Run-in rate | 42/45 | 0.208 (0.148, 0.271) | 0.183 (0.130, 0.246) | 0.905 (0.805, 0.977) |
| REPLACE-BG | Updating rate | 42/45 | 0.122 (0.081, 0.164) | 0.103 (0.071, 0.134) | 0.810 (0.690, 0.929) |
| REPLACE-BG | Shrunk run-in rate | 45/45 | 0.158 (0.109, 0.204) | 0.160 (0.106, 0.226) | 0.756 (0.622, 0.867) |
| REPLACE-BG | Best prospective rate (Updating rate) | 42/45 | 0.122 (0.081, 0.164) | 0.103 (0.071, 0.134) | 0.810 (0.690, 0.929) |
| CTN-0094 weekly | Population event rate | 246/246 | 0.559 (0.509, 0.609) | 0.596 (0.550, 0.641) | 0.907 (0.870, 0.943) |
| CTN-0094 weekly | Run-in rate | 173/246 | 0.496 (0.432, 0.549) | 0.376 (0.316, 0.428) | 0.809 (0.745, 0.866) |
| CTN-0094 weekly | Updating rate | 181/246 | 0.219 (0.173, 0.261) | 0.194 (0.115, 0.258) | 0.740 (0.678, 0.806) |
| CTN-0094 weekly | Shrunk run-in rate | 246/246 | 0.466 (0.420, 0.516) | 0.564 (0.516, 0.609) | 0.907 (0.870, 0.943) |
| CTN-0094 weekly | Best prospective rate (Updating rate) | 181/246 | 0.219 (0.173, 0.261) | 0.194 (0.115, 0.258) | 0.740 (0.678, 0.806) |
| College Experience | Population event rate | 38/38 | 0.595 (0.507, 0.670) | 0.585 (0.509, 0.654) | 1.000 (1.000, 1.000) |
| College Experience | Run-in rate | 37/38 | 0.590 (0.476, 0.688) | 0.418 (0.321, 0.520) | 0.946 (0.865, 1.000) |
| College Experience | Updating rate | 37/38 | 0.396 (0.295, 0.508) | 0.297 (0.219, 0.380) | 0.892 (0.784, 0.974) |
| College Experience | Shrunk run-in rate | 38/38 | 0.559 (0.449, 0.656) | 0.501 (0.423, 0.575) | 1.000 (1.000, 1.000) |
| College Experience | Best prospective rate (Updating rate) | 37/38 | 0.396 (0.295, 0.508) | 0.297 (0.219, 0.380) | 0.892 (0.784, 0.974) |

## Reproduction check against existing tables

`cases_tables.py` computes and publishes pooled skill against run-in, updating, and retrospective patient rates. The check below compares run-in and updating directly to the values in `open_data/cases_tables.json`; both reproduce to floating-point precision. The current `cases_tables.py` does not compute skill against the population event rate, so that requested reference is new and has no prior value to match. The retrospective reference is not one of the requested a–c prospective references.

| Cohort | Existing reference | Published pooled skill | Recomputed pooled skill | Absolute difference |
|---|---|---:|---:|---:|
| REPLACE-BG | Run-in rate | 0.207674175949 | 0.207674175949 | 0.00e+00 |
| REPLACE-BG | Updating rate | 0.121746663420 | 0.121746663420 | 0.00e+00 |
| CTN-0094 weekly | Run-in rate | 0.496425684027 | 0.496425684027 | 0.00e+00 |
| CTN-0094 weekly | Updating rate | 0.218855932301 | 0.218855932301 | 0.00e+00 |
| College Experience | Run-in rate | 0.589812487090 | 0.589812487090 | 0.00e+00 |
| College Experience | Updating rate | 0.395796957055 | 0.395796957055 | 0.00e+00 |

## Interpretation

The model’s apparent skill depends on which prospective information is already available: the best patient-level rate can be a substantially tougher comparator than the population event rate, especially when individual rates differ. Shrinking the run-in rate changes its benchmark loss and therefore the skill ratio, while using updating rates asks what the model adds after incorporating the patient’s accumulating outcomes. These are incremental forecast comparisons, so strong skill against a population rate does not by itself show added value over a patient-specific prospective estimate.

## Shrinkage and source

All three input files contain a shrunk run-in rate; this script uses that existing column. The construction is an empirical-Bayes weighted average with 10 training-cohort pseudo-observations: REPLACE-BG uses `(14 * runin_rate + 10 * training_cohort_rate) / 24` in `open_data/replace_bg_analysis.py:418`; CTN-0094 uses `(runin_rate * runin_n + 10 * training_cohort_rate) / (runin_n + 10)` in `open_data/ctn0094_analysis.py:239`; College Experience uses `(20 * runin_rate + 10 * training_cohort_rate) / 30` in `open_data/ces_analysis.py:371`.

## Reproduce

From the repository root:

```powershell
python open_data\skill_analysis.py
```
