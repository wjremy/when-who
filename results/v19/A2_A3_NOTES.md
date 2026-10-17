# ASTRA18 A2 and A3: what "the better model" is in the ranking simulation (Programming, 15 Oct 2026)

Closer pair (Table 6: A daily rho 0.4, B daily rho 0.3). Scripts reuse code/ranking_sim.py's generation, forecasts, threshold and per-patient definitions.

## A2: long-record truth on every ranked measure
`code/v19_long_record_truth.py` -> `results/v19/A2_long_record_truth.csv` (19 s; asserts its AUCs equal the stored `long_record_*_true_patient_auc` to 1e-12).
Records of 5,000 days, 200 patients, model A with the calibration fitted for each record length.
- Wide pair (0.6 vs 0.2): A above B on every measure, both spreads, all calibration lengths.
- Closer pair: A above B on AUC and average precision everywhere. Exceptions: high spread, 30-day calibration, PPV excess (A 0.059, B 0.073); low spread, 30-day calibration, Brier skill (all three references) and net benefit; low spread, 100-day calibration, Brier skill (net benefit tied, 0.0032 vs 0.0031).

## A3: finite-horizon AUC target (cross-replicate)
`code/v19_finite_horizon.py` -> `results/v19/A3_finite_horizon.csv` (2,000 patients x 100 records per setting, 9.6 min; MC SE <= 0.0009).
AUC of event days of one record against event-free days of the patient's other records, minus 0.5.

| spread | days | A finite | B finite | A long record | B long record | A mean shuffled excess | B mean shuffled excess |
|---|---|---|---|---|---|---|---|
| low | 30 | 0.160 | 0.130 | 0.168 | 0.127 | 0.168 | 0.128 |
| low | 100 | 0.151 | 0.130 | 0.167 | 0.127 | 0.163 | 0.130 |
| low | 365 | 0.150 | 0.131 | 0.165 | 0.127 | 0.160 | 0.130 |
| high | 30 | 0.090 | 0.118 | 0.147 | 0.119 | 0.111 | 0.111 |
| high | 100 | 0.109 | 0.117 | 0.149 | 0.119 | 0.128 | 0.115 |
| high | 365 | 0.130 | 0.116 | 0.150 | 0.119 | 0.140 | 0.116 |

Caveat: comparing across records also penalises A for differences in its learned level between records, which within-record ranking never sees. Hence the second target below.

## A3b: skill on the record at hand (ghost outcomes), all measures
`code/v19_finite_horizon_measures.py` -> `results/v19/A3_finite_horizon_measures.csv` (20,000 patients per setting, one record each, 4.4 min; both pairs).
"actual": the record's forecasts scored on its own outcomes. "ghost": the same forecasts scored on fresh outcomes drawn from the same patient level and the same daily components, so outcomes no longer feed into later forecasts. For day-by-day sums the two have the same expectation (Proposition 1). Values are means over patients of the per-patient measure, as in ranking_sim.

Closer pair, ghost values (A vs B); the AUC rows minus 0.5 give the skill:

| spread | days | AUC | Average precision | PPV excess | Net benefit excess | Brier skill (rate so far) |
|---|---|---|---|---|---|---|
| low | 30 | 0.654 vs 0.630 | 0.273 vs 0.256 | 0.034 vs 0.025 | -0.0081 vs -0.0051 | -0.43 vs -0.20 |
| low | 100 | 0.653 vs 0.629 | 0.178 vs 0.163 | 0.029 vs 0.025 | -0.0008 vs 0.0005 | -0.09 vs 0.01 |
| low | 365 | 0.656 vs 0.629 | 0.136 vs 0.121 | 0.028 vs 0.025 | 0.0028 vs 0.0025 | 0.016 vs 0.021 |
| high | 30 | 0.587 vs 0.615 | 0.480 vs 0.503 | 0.035 vs 0.044 | -0.0170 vs -0.0005 | 0.07 vs 0.23 |
| high | 100 | 0.617 vs 0.616 | 0.427 vs 0.428 | 0.048 vs 0.050 | -0.0014 vs 0.0030 | 0.04 vs 0.14 |
| high | 365 | 0.636 vs 0.617 | 0.406 vs 0.394 | 0.061 vs 0.058 | 0.0055 vs 0.0044 | 0.035 vs 0.071 |

Facts for the text (judgement is Conceptual's):
1. AUC, low spread: A is the better ranker on records of every length, on both finite targets. High spread: B is better on records of 30 days (ghost 0.615 vs 0.587; cross-replicate 0.618 vs 0.590), they tie at 100 days (0.616 vs 0.617; cross-replicate B ahead by 0.008), and A is better at 365 days.
2. A's mean shuffled excess exceeds A's ghost skill by 0.024 (high, 30 d), 0.011 (high, 100 d), 0.004 (high, 365 d), 0.014, 0.010 and 0.004 (low; 30, 100, 365 d). B's excess is within 0.004 of B's ghost skill. So the excess is not an unbiased estimate of the learning model's skill on the record at hand: its shuffled baseline removes more record dependence than the record itself carries. The guarantee (zero mean excess without within-patient skill) is unaffected.
3. With these targets, the 30-day high-spread "split evenly" result has B as the better model on the record at hand, and the "at least 89% on records of 100 days or more" result includes a 100-day setting in which A and B are tied.
4. Brier skill and net benefit as raw per-patient values put B ahead on short records in both spreads, because they credit knowing the level, which B has exactly. They are not within-patient skill targets. The text's claim "On Brier skill and net benefit, it ranked model A first in at least 93% of replications in every setting" therefore needs a stated truth: A is better there only by within-patient skill as the AUC defines it (long record, or the finite targets in low spread / long records), not by the raw measure on these records.
5. Brier skill against the patient's true rate ("oracle") per patient is unstable on short records (patients with rates near 0 give near-zero reference loss; means in the thousands): not usable as a target; ignored above.

## A4: model B, widely separated pair (0.6 vs 0.2)
Mean shuffled-baseline AUC excess minus long-record skill (AUC minus 0.5), all 12 settings (2 spreads x 3 lengths x 50/150 patients): `results/v19/A4_modelB_wide.csv` (from open_data/results/v10/ranking_sim.json). Range -0.0063 to +0.0035, largest absolute value 0.0063 (high spread, 30 days, 50 patients).
All four models, largest absolute difference over the 12 settings: closer pair A 0.036, B 0.008; wide pair A 0.032, B 0.006. So every model in every setting came within 0.04 of its long-record AUC skill.
