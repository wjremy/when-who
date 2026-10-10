# V10 yardstick check

Both forecasts are scored per patient. Run-in uses the observed scored-period outcomes. For each shuffled draw, the outcomes are permuted within patient, the running rate is rebuilt, and that draw is scored against its permuted outcomes, matching `main_table_v10.py`. Yardsticks (a), (b), and (c1) use the training rate, the patient's unchanged scored-period rate, and the fixed run-in rate; (c2) is the observed Table 4 past-days-only rate, row by row.

Draws: 200 (seed 20261009; Table 4 cohort order). Values are equal-patient summaries, rounded to four decimals. Shuffled intervals are the 2.5th-97.5th percentiles across draw-level summaries.

Brier skill has two rows: mean of patient skills (only patients with positive yardstick Brier loss) and the ratio of equal-patient mean losses (all patients; its patient median is not defined). PPV rows include patients with at least one alarm. For (c2), PPV's reference rate is averaged over that patient's alarmed rows; treat-all NB is averaged over all rows. All alarms use forecast >= the training-cohort threshold.

## Theory predictions for run-in against (b)

Statuses compare the observed equal-patient summaries with the stated zero/negative prediction. The PPV identity is also checked patient by patient. These are dataset checks, not repeated-sample uncertainty estimates.

| Dataset | Prediction | Estimate | Status |
|---|---|---:|---|
| CTN-0094 | Run-in PPV minus the patient's scored-period rate is exactly 0 for every eligible patient. | mean 0.000000; max |patient| 0.000000 | holds |
| CTN-0094 | Run-in net benefit minus treat-all NB at the patient's scored-period rate has expected value 0. | 0.147863 | fails at zero |
| CTN-0094 | Run-in net benefit minus max(treat-all, treat-none) at the patient's scored-period rate is not 0. | -0.072567 | nonzero as predicted |
| CTN-0094 | Run-in Brier skill against the patient's scored-period rate is biased low (negative). | -0.872563 | negative as predicted |
| REPLACE-BG | Run-in PPV minus the patient's scored-period rate is exactly 0 for every eligible patient. | mean 0.000000; max |patient| 0.000000 | holds |
| REPLACE-BG | Run-in net benefit minus treat-all NB at the patient's scored-period rate has expected value 0. | 0.009790 | fails at zero |
| REPLACE-BG | Run-in net benefit minus max(treat-all, treat-none) at the patient's scored-period rate is not 0. | -0.008928 | nonzero as predicted |
| REPLACE-BG | Run-in Brier skill against the patient's scored-period rate is biased low (negative). | -0.261055 | negative as predicted |
| College Experience | Run-in PPV minus the patient's scored-period rate is exactly 0 for every eligible patient. | mean 0.000000; max |patient| 0.000000 | holds |
| College Experience | Run-in net benefit minus treat-all NB at the patient's scored-period rate has expected value 0. | 0.056614 | fails at zero |
| College Experience | Run-in net benefit minus max(treat-all, treat-none) at the patient's scored-period rate is not 0. | -0.125209 | nonzero as predicted |
| College Experience | Run-in Brier skill against the patient's scored-period rate is biased low (negative). | -0.989200 | negative as predicted |

## CTN-0094

246 patients; 4237 rows; training rate / threshold 0.319570.

| Metric x yardstick | Run-in mean | Run-in median | Shuffled mean [2.5-97.5] | Shuffled median [2.5-97.5] | Eligible n (run-in / shuffled) |
|---|---:|---:|---:|---:|---:|
| Brier skill: mean individual skills x (a) Training-cohort population rate | 0.0983 | 0.3145 | 0.4529 [0.4512, 0.4544] | 0.5121 [0.4966, 0.5247] | 246 / 246 |
| Brier skill: ratio of mean losses x (a) Training-cohort population rate | 0.1434 | n/a | 0.4258 [0.4240, 0.4273] | n/a | 246 / 246 |
| PPV minus yardstick rate x (a) Training-cohort population rate | 0.2056 | 0.2249 | 0.1649 [0.1507, 0.1783] | 0.1724 [0.1410, 0.2042] | 124 / 140-151 |
| Net benefit minus treat-all NB x (a) Training-cohort population rate | 0.1523 | 0.0000 | 0.1995 [0.1972, 0.2018] | 0.0000 [0.0000, 0.0000] | 246 / 246 |
| Brier skill: mean individual skills x (b) Patient rate over scored period (hindsight) | -0.8726 | -0.3148 | -0.2710 [-0.2762, -0.2654] | -0.1941 [-0.2048, -0.1821] | 154 / 154 |
| Brier skill: ratio of mean losses x (b) Patient rate over scored period (hindsight) | -0.9135 | n/a | -0.2828 [-0.2868, -0.2794] | n/a | 246 / 246 |
| PPV minus yardstick rate x (b) Patient rate over scored period (hindsight) | 0.0000 | 0.0000 | -0.0302 [-0.0392, -0.0203] | 0.0000 [0.0000, 0.0000] | 124 / 140-151 |
| Net benefit minus treat-all NB x (b) Patient rate over scored period (hindsight) | 0.1479 | 0.0000 | 0.1951 [0.1928, 0.1973] | 0.1608 [0.1463, 0.1743] | 246 / 246 |
| Net benefit minus max(treat-all, treat-none) x (b) Patient rate over scored period (hindsight) | -0.0726 | 0.0000 | -0.0254 [-0.0277, -0.0231] | 0.0000 [0.0000, 0.0000] | 246 / 246 |
| Brier skill: mean individual skills x (c1) Patient run-in rate | 0.0000 | 0.0000 | 0.1640 [0.1612, 0.1669] | 0.0000 [0.0000, 0.0000] | 246 / 246 |
| Brier skill: ratio of mean losses x (c1) Patient run-in rate | 0.0000 | n/a | 0.3296 [0.3275, 0.3314] | n/a | 246 / 246 |
| PPV minus yardstick rate x (c1) Patient run-in rate | -0.2141 | -0.1994 | -0.1567 [-0.1666, -0.1467] | -0.1437 [-0.1820, -0.1026] | 124 / 140-151 |
| Net benefit minus treat-all NB x (c1) Patient run-in rate | 0.0444 | 0.1022 | 0.0916 [0.0893, 0.0939] | 0.1022 [0.1022, 0.1022] | 246 / 246 |
| Brier skill: mean individual skills x (c2) Past-days-only updated rate (row-wise) | -0.5463 | 0.0000 | -0.0143 [-0.0245, -0.0035] | 0.0000 [0.0000, 0.0000] | 246 / 246 |
| Brier skill: ratio of mean losses x (c2) Past-days-only updated rate (row-wise) | -0.5123 | n/a | 0.0209 [0.0034, 0.0387] | n/a | 246 / 246 |
| PPV minus yardstick rate x (c2) Past-days-only updated rate (row-wise) | -0.0987 | -0.0916 | -0.1289 [-0.1385, -0.1186] | -0.1040 [-0.1214, -0.0909] | 124 / 140-151 |
| Net benefit minus treat-all NB x (c2) Past-days-only updated rate (row-wise) | 0.1074 | 0.0936 | 0.1546 [0.1524, 0.1569] | 0.2155 [0.2078, 0.2187] | 246 / 246 |

## REPLACE-BG

45 patients; 9477 rows; training rate / threshold 0.054686.

| Metric x yardstick | Run-in mean | Run-in median | Shuffled mean [2.5-97.5] | Shuffled median [2.5-97.5] | Eligible n (run-in / shuffled) |
|---|---:|---:|---:|---:|---:|
| Brier skill: mean individual skills x (a) Training-cohort population rate | -0.1054 | -0.0080 | 0.0843 [0.0831, 0.0854] | 0.0054 [0.0018, 0.0084] | 45 / 45 |
| Brier skill: ratio of mean losses x (a) Training-cohort population rate | -0.0673 | n/a | 0.0290 [0.0280, 0.0299] | n/a | 45 / 45 |
| PPV minus yardstick rate x (a) Training-cohort population rate | 0.0300 | 0.0178 | 0.0174 [0.0108, 0.0249] | 0.0136 [-0.0031, 0.0204] | 23 / 27-35 |
| Net benefit minus treat-all NB x (a) Training-cohort population rate | 0.0162 | 0.0000 | 0.0217 [0.0209, 0.0224] | 0.0000 [0.0000, 0.0000] | 45 / 45 |
| Brier skill: mean individual skills x (b) Patient rate over scored period (hindsight) | -0.2611 | -0.0368 | -0.0296 [-0.0310, -0.0283] | -0.0178 [-0.0196, -0.0158] | 42 / 42 |
| Brier skill: ratio of mean losses x (b) Patient rate over scored period (hindsight) | -0.1226 | n/a | -0.0213 [-0.0223, -0.0204] | n/a | 45 / 45 |
| PPV minus yardstick rate x (b) Patient rate over scored period (hindsight) | 0.0000 | 0.0000 | -0.0079 [-0.0116, -0.0035] | -0.0023 [-0.0064, 0.0000] | 23 / 27-35 |
| Net benefit minus treat-all NB x (b) Patient rate over scored period (hindsight) | 0.0098 | 0.0000 | 0.0153 [0.0144, 0.0159] | 0.0072 [0.0018, 0.0094] | 45 / 45 |
| Net benefit minus max(treat-all, treat-none) x (b) Patient rate over scored period (hindsight) | -0.0089 | 0.0000 | -0.0035 [-0.0043, -0.0028] | -0.0003 [-0.0013, 0.0000] | 45 / 45 |
| Brier skill: mean individual skills x (c1) Patient run-in rate | 0.0000 | 0.0000 | 0.0874 [0.0862, 0.0886] | 0.0198 [0.0165, 0.0228] | 42 / 42 |
| Brier skill: ratio of mean losses x (c1) Patient run-in rate | 0.0000 | n/a | 0.0902 [0.0893, 0.0910] | n/a | 45 / 45 |
| PPV minus yardstick rate x (c1) Patient run-in rate | -0.0489 | -0.0456 | -0.0286 [-0.0337, -0.0238] | -0.0127 [-0.0407, 0.0000] | 23 / 27-35 |
| Net benefit minus treat-all NB x (c1) Patient run-in rate | 0.0019 | 0.0578 | 0.0073 [0.0065, 0.0080] | 0.0564 [0.0527, 0.0578] | 45 / 45 |
| Brier skill: mean individual skills x (c2) Past-days-only updated rate (row-wise) | -0.1975 | -0.0224 | -0.0073 [-0.0129, -0.0032] | -0.0070 [-0.0108, -0.0034] | 42 / 42 |
| Brier skill: ratio of mean losses x (c2) Past-days-only updated rate (row-wise) | -0.0993 | n/a | -0.0068 [-0.0108, -0.0030] | n/a | 45 / 45 |
| PPV minus yardstick rate x (c2) Past-days-only updated rate (row-wise) | -0.0097 | -0.0087 | -0.0196 [-0.0245, -0.0134] | -0.0075 [-0.0144, -0.0036] | 23 / 27-35 |
| Net benefit minus treat-all NB x (c2) Past-days-only updated rate (row-wise) | 0.0088 | 0.0083 | 0.0142 [0.0134, 0.0149] | 0.0148 [0.0136, 0.0162] | 45 / 45 |

## College Experience

38 patients; 6361 rows; training rate / threshold 0.393980.

| Metric x yardstick | Run-in mean | Run-in median | Shuffled mean [2.5-97.5] | Shuffled median [2.5-97.5] | Eligible n (run-in / shuffled) |
|---|---:|---:|---:|---:|---:|
| Brier skill: mean individual skills x (a) Training-cohort population rate | 0.0457 | 0.0105 | 0.3395 [0.3394, 0.3397] | 0.2491 [0.2486, 0.2496] | 38 / 38 |
| Brier skill: ratio of mean losses x (a) Training-cohort population rate | 0.0525 | n/a | 0.3509 [0.3508, 0.3511] | n/a | 38 / 38 |
| PPV minus yardstick rate x (a) Training-cohort population rate | 0.2262 | 0.2898 | 0.2242 [0.1969, 0.2427] | 0.2516 [0.2348, 0.2829] | 14 / 23-25 |
| Net benefit minus treat-all NB x (a) Training-cohort population rate | 0.1375 | -0.0000 | 0.2428 [0.2409, 0.2447] | 0.0238 [0.0051, 0.0350] | 38 / 38 |
| Brier skill: mean individual skills x (b) Patient rate over scored period (hindsight) | -0.9892 | -0.2836 | -0.1614 [-0.1622, -0.1608] | -0.0433 [-0.0445, -0.0412] | 37 / 37 |
| Brier skill: ratio of mean losses x (b) Patient rate over scored period (hindsight) | -0.6000 | n/a | -0.0961 [-0.0964, -0.0958] | n/a | 38 / 38 |
| PPV minus yardstick rate x (b) Patient rate over scored period (hindsight) | 0.0000 | 0.0000 | -0.0073 [-0.0214, 0.0100] | -0.0004 [-0.0019, 0.0000] | 14 / 23-25 |
| Net benefit minus treat-all NB x (b) Patient rate over scored period (hindsight) | 0.0566 | 0.0000 | 0.1619 [0.1600, 0.1638] | 0.0000 [0.0000, 0.0000] | 38 / 38 |
| Net benefit minus max(treat-all, treat-none) x (b) Patient rate over scored period (hindsight) | -0.1252 | 0.0000 | -0.0199 [-0.0219, -0.0181] | -0.0000 [-0.0000, 0.0000] | 38 / 38 |
| Brier skill: mean individual skills x (c1) Patient run-in rate | 0.0000 | 0.0000 | 0.2148 [0.2141, 0.2153] | 0.1704 [0.1639, 0.1734] | 37 / 37 |
| Brier skill: ratio of mean losses x (c1) Patient run-in rate | 0.0000 | n/a | 0.3149 [0.3147, 0.3151] | n/a | 38 / 38 |
| PPV minus yardstick rate x (c1) Patient run-in rate | 0.0095 | 0.0434 | 0.1779 [0.1586, 0.1954] | 0.2226 [0.2171, 0.2235] | 14 / 23-25 |
| Net benefit minus treat-all NB x (c1) Patient run-in rate | 0.2774 | 0.3583 | 0.3827 [0.3807, 0.3846] | 0.4836 [0.4744, 0.4851] | 38 / 38 |
| Brier skill: mean individual skills x (c2) Past-days-only updated rate (row-wise) | -0.4987 | -0.2052 | 0.0576 [0.0481, 0.0671] | 0.0191 [0.0083, 0.0316] | 37 / 37 |
| Brier skill: ratio of mean losses x (c2) Past-days-only updated rate (row-wise) | -0.4612 | n/a | 0.0845 [0.0752, 0.0949] | n/a | 38 / 38 |
| PPV minus yardstick rate x (c2) Past-days-only updated rate (row-wise) | -0.0033 | 0.0167 | 0.0266 [0.0096, 0.0399] | 0.0502 [0.0433, 0.0727] | 14 / 23-25 |
| Net benefit minus treat-all NB x (c2) Past-days-only updated rate (row-wise) | 0.1574 | 0.1964 | 0.2627 [0.2608, 0.2646] | 0.2447 [0.2284, 0.2522] | 38 / 38 |
