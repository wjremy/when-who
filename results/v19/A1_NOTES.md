# ASTRA18 A1: mean excess next to median excess (Programming, 15 Oct 2026)

`code/v19_mean_excess.py` -> `results/v19/A1_mean_excess.csv` (stored values only; no rerun needed, seconds).
The shuffled baseline guarantees zero MEAN excess without within-patient skill; the paper reports the median [IQR] over patients. Mean and SE of the mean (SD/sqrt(n)) over patients:

## Table 5 (fitted gradient boosting), five measures in the table
Sources: CTN-0094 `open_data/results/v10/fitted_model_package.json` per_patient; REPLACE-BG and CES `work/2026-10-13_v18_analyses/task1_per_patient.json` (rerun 15 Oct with the fixed p value). Every reported median reproduces.

| Dataset | Measure | Median [IQR] (reported) | Mean (SE) | n |
|---|---|---|---|---|
| CTN-0094 | AUC | 0.26 [0.05, 0.45] | 0.24 (0.02) | 154 |
| CTN-0094 | PPV | 0.03 [0.00, 0.27] | 0.15 (0.02) | 166 |
| CTN-0094 | Net benefit | 0.000 [0.000, 0.065] | 0.042 (0.005) | 246 |
| CTN-0094 | Brier skill | 0.27 [0.04, 0.56] | 0.34 (0.03) | 154 |
| REPLACE-BG | AUC | 0.29 [0.19, 0.35] | 0.28 (0.02) | 42 |
| REPLACE-BG | PPV | 0.07 [0.04, 0.11] | 0.08 (0.01) | 43 |
| REPLACE-BG | Net benefit | 0.018 [0.004, 0.032] | 0.020 (0.002) | 45 |
| REPLACE-BG | Brier skill | 0.22 [0.07, 0.31] | 0.22 (0.03) | 42 |
| College Experience Study | AUC | 0.34 [0.29, 0.40] | 0.32 (0.02) | 37 |
| College Experience Study | PPV | 0.25 [0.07, 0.34] | 0.24 (0.03) | 37 |
| College Experience Study | Net benefit | 0.075 [0.003, 0.158] | 0.096 (0.017) | 38 |
| College Experience Study | Brier skill | 0.41 [0.13, 0.64] | 0.40 (0.05) | 37 |

Calibration slope: Table 5 shows the conditional within-patient slope (one slope per dataset), not a per-patient median; per-patient slope medians are in the CSV for reference only.

Largest gaps: CTN-0094 alarms. PPV mean 0.15 against median 0.03; net benefit mean 0.042 against median 0.000; also NPV 0.10 vs 0.00, sensitivity 0.17 vs 0.02, specificity 0.12 vs 0.04, average precision 0.17 vs 0.10 (S-table measures). In CTN-0094 most patients gain nothing on alarms, and a minority gains much: the distribution is a spike at zero with a long right tail. Elsewhere mean and median agree within about 0.03.
Consequence for the text (Conceptual's judgement): "the model added little to the patient's own rate. The median excess was 0.03 for the PPV and 0.000 for net benefit" is true of the typical patient. The mean excess, which is the quantity the correction guarantees to be zero without skill, is 0.15 (SE 0.02) and 0.042 (SE 0.005): clearly above zero.

## Supplement S10, Table S16 (block shuffles, AUC)
`open_data/results/v16/block_results.json` stores mean, median and IQR (no patient vectors, so no SE). Mean vs median, excess:
CTN-0094 gradient boosting: full 0.238 vs 0.263; 8-week 0.152 vs 0.135; 4-week 0.098 vs 0.072. Logistic regression: full 0.242 vs 0.278; 8-week 0.138 vs 0.094; 4-week 0.085 vs 0.071.
REPLACE-BG gradient boosting: full 0.275 vs 0.292; 14-day 0.236 vs 0.249; 7-day 0.209 vs 0.222. Logistic: 0.260 vs 0.295; 0.226 vs 0.241; 0.206 vs 0.228.
College Experience Study gradient boosting: full 0.316 vs 0.335; 14-day 0.071 vs 0.054; 7-day 0.017 vs 0.020. Logistic: 0.316 vs 0.337; 0.075 vs 0.047; 0.022 vs 0.019.
An SE would need a rerun of open_data/block_shuffle.py storing patient vectors (the script's `--rerun-missing` part; not run).

## Section 5.2 (no-skill models)
The text reports means over shuffles there, not medians, so no mean/median pair is needed. The run-in rate's shuffled baseline equals its score exactly; the shuffled rate so far's equals it in expectation (Supplement S8). The script's Section 5.2 rerun part was therefore not run.
