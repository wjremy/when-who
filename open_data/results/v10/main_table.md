# Main table: forecasts without knowledge of the day

The run-in forecast is each patient's fixed run-in event rate. For the shuffled running-rate forecast, scored outcomes are shuffled separately within each patient, run-in outcomes stay fixed, and the expanding rate is rebuilt on each draw with Table 4's history builder.

For shuffled forecasts, cells show the mean across draws and the 2.5th–97.5th percentiles. Pooled values are row weighted; their Brier skill uses the training-cohort rate, as in Table 4. Pooled AUC uses 0.5; pooled AP, PPV, and NPV use the scored-set event/non-event rate; pooled sensitivity and specificity use the training-cohort rate and its complement; pooled net benefit shows treat-none / treat-all; pooled Brier skill uses 0; pooled calibration slope uses 1.

Naive values are equal-patient means from `naive_metrics`; sensitivity and specificity use separate eligible patient groups, and Youden is their means added minus one, read against 0. Naive chance levels follow the training-cohort rate in `naive_metrics`; an estimable naive calibration slope is read against the calibrated-twin slope of 1, while the constant run-in slope is n/a. Tailored values use `nine_metrics`: run-in is referenced to Table 4's updated-rate column, while each shuffled draw uses its rebuilt updated rate. The tailored bracket is the score of that same no-within-patient-information forecast by construction.

## CTN-0094

246 test patients; 4237 scored rows; training-cohort rate / alarm threshold 0.319570; scored-set event rate 0.299976.

| Metric | Run-in pooled | Run-in naive patient | Run-in tailored patient | Shuffled running pooled | Shuffled running naive patient | Shuffled running tailored patient |
|---|---:|---:|---:|---:|---:|---:|
| AUC | 0.801 [0.500] | 0.500 [0.500] | 0.500 [0.500] | 0.885 (0.884, 0.886) [0.500] | 0.291 (0.263, 0.315) [0.500] | 0.291 (0.263, 0.315) [0.291 (0.263, 0.315)] |
| Average precision | 0.600 [0.300] | 0.472 [0.320] | 0.424 [0.424] | 0.758 (0.752, 0.763) [0.300] | 0.451 (0.442, 0.459) [0.320] | 0.401 (0.391, 0.409) [0.401 (0.391, 0.409)] |
| Sensitivity | 0.814 [0.320] | 0.679 [0.320] | 0.649 [0.649] | 0.837 (0.826, 0.851) [0.320] | 0.595 (0.574, 0.616) [0.320] | 0.559 (0.535, 0.581) [0.559 (0.535, 0.581)] |
| Specificity | 0.653 [0.680] | 0.526 [0.680] | 0.351 [0.351] | 0.779 (0.769, 0.790) [0.680] | 0.594 (0.582, 0.606) [0.680] | 0.398 (0.379, 0.416) [0.398 (0.379, 0.416)] |
| PPV | 0.501 [0.300] | 0.525 [0.320] | 0.000 [0.000] | 0.619 (0.612, 0.629) [0.300] | 0.484 (0.470, 0.498) [0.320] | -0.030 (-0.039, -0.020) [-0.030 (-0.039, -0.020)] |
| NPV | 0.891 [0.700] | 0.883 [0.680] | -0.000 [-0.000] | 0.918 (0.913, 0.923) [0.700] | 0.836 (0.818, 0.855) [0.680] | -0.037 (-0.051, -0.023) [-0.037 (-0.051, -0.023)] |
| Net benefit | 0.130 [0.000 / -0.029] | 0.152 [0.000] | -0.047 [-0.047] | 0.179 (0.177, 0.181) [0.000 / -0.029] | 0.199 (0.197, 0.202) [0.000] | 0.000 (0.000, 0.000) [0.000 (0.000, 0.000)] |
| Brier skill | 0.123 [0.000] | 0.143 [0.000] | -0.512 [-0.512] | 0.418 (0.416, 0.419) [0.000] | 0.426 (0.424, 0.427) [0.000] | 0.000 (0.000, 0.000) [0.000 (0.000, 0.000)] |
| Calibration slope | 0.113 [1.000] | n/a [n/a (no within-patient forecast variation)] | n/a [n/a] | 0.248 (0.245, 0.251) [1.000] | -0.069 (-0.099, -0.038) [1.000] | -0.069 (-0.099, -0.038) [-0.069 (-0.099, -0.038)] |
*Eligible patient n (run-in / shuffled running):* naive: AUC 154/154, AP 168/168, sens 168/168, spec 232/232, PPV 124/140–151, NPV 122/163–172, NB 246/246, Brier 246/246, slope 0/154; tailored: AUC 154/154, AP 154/154, sens 154/154, spec 154/154, PPV 124/140–151, NPV 122/163–172, NB 246/246, Brier 246/246, slope 0/154. Youden groups: run-in sens 168; spec 232; shuffled sens 168; spec 232.


*Naive Youden (separate sensitivity/specificity groups; chance 0):* run-in 0.204 [0.000]; shuffled running 0.190 (0.174, 0.208) [0.000].

*Eligibility and definitions:* pooled scores use all scored rows where the metric is defined. Per-patient eligible counts are in each JSON cell's `eligible_n` (for shuffled draws, a minimum–maximum if it varies). Naive sensitivity uses patients with at least one event; naive specificity uses patients with at least one non-event; naive AP uses patients with an event; naive PPV/NPV require an alarm/no alarm. Tailored AUC/AP/sensitivity/specificity require both outcome classes; tailored PPV/NPV require an alarm/no alarm; tailored Brier skill requires a nonzero yardstick loss; calibration slope requires within-patient forecast variation. PPV/NPV are raw for pooled and naive scoring, and tailored PPV/NPV are excesses over each patient's observed event/non-event rate. Net benefit is raw versus treat-none for pooled and naive scoring, and an excess over the yardstick policy for tailored scoring. Brier skill summaries use the ratio of mean patient losses.

## REPLACE-BG

45 test patients; 9477 scored rows; training-cohort rate / alarm threshold 0.054686; scored-set event rate 0.060146.

| Metric | Run-in pooled | Run-in naive patient | Run-in tailored patient | Shuffled running pooled | Shuffled running naive patient | Shuffled running tailored patient |
|---|---:|---:|---:|---:|---:|---:|
| AUC | 0.609 [0.500] | 0.500 [0.500] | 0.500 [0.500] | 0.708 (0.705, 0.712) [0.500] | 0.374 (0.343, 0.406) [0.500] | 0.374 (0.343, 0.406) [0.374 (0.343, 0.406)] |
| Average precision | 0.081 [0.060] | 0.065 [0.055] | 0.065 [0.065] | 0.123 (0.120, 0.128) [0.060] | 0.059 (0.056, 0.063) [0.055] | 0.059 (0.056, 0.063) [0.059 (0.056, 0.063)] |
| Sensitivity | 0.726 [0.055] | 0.548 [0.055] | 0.548 [0.548] | 0.734 (0.700, 0.767) [0.055] | 0.439 (0.393, 0.485) [0.055] | 0.439 (0.393, 0.485) [0.439 (0.393, 0.485)] |
| Specificity | 0.497 [0.945] | 0.489 [0.945] | 0.452 [0.452] | 0.591 (0.560, 0.627) [0.945] | 0.566 (0.535, 0.603) [0.945] | 0.535 (0.501, 0.574) [0.535 (0.501, 0.574)] |
| PPV | 0.085 [0.060] | 0.085 [0.055] | 0.000 [0.000] | 0.103 (0.099, 0.108) [0.060] | 0.072 (0.065, 0.080) [0.055] | -0.008 (-0.012, -0.004) [-0.008 (-0.012, -0.004)] |
| NPV | 0.966 [0.940] | 0.964 [0.945] | 0.000 [0.000] | 0.972 (0.970, 0.974) [0.940] | 0.929 (0.892, 0.955) [0.945] | -0.028 (-0.062, -0.003) [-0.028 (-0.062, -0.003)] |
| Net benefit | 0.016 [0.000 / 0.006] | 0.016 [0.000] | -0.006 [-0.006] | 0.022 (0.021, 0.023) [0.000 / 0.006] | 0.022 (0.021, 0.022) [0.000] | 0.000 (0.000, 0.000) [0.000 (0.000, 0.000)] |
| Brier skill | -0.070 [0.000] | -0.067 [0.000] | -0.099 [-0.099] | 0.031 (0.030, 0.032) [0.000] | 0.029 (0.028, 0.030) [0.000] | 0.000 (0.000, 0.000) [0.000 (0.000, 0.000)] |
| Calibration slope | 0.080 [1.000] | n/a [n/a (no within-patient forecast variation)] | n/a [n/a] | 0.298 (0.287, 0.312) [1.000] | -0.043 (-0.101, 0.007) [1.000] | -0.043 (-0.101, 0.007) [-0.043 (-0.101, 0.007)] |
*Eligible patient n (run-in / shuffled running):* naive: AUC 42/42, AP 42/42, sens 42/42, spec 45/45, PPV 23/27–35, NPV 22/33–40, NB 45/45, Brier 45/45, slope 0/42; tailored: AUC 42/42, AP 42/42, sens 42/42, spec 42/42, PPV 23/27–35, NPV 22/33–40, NB 45/45, Brier 42/42, slope 0/42. Youden groups: run-in sens 42; spec 45; shuffled sens 42; spec 45.


*Naive Youden (separate sensitivity/specificity groups; chance 0):* run-in 0.037 [0.000]; shuffled running 0.005 (-0.018, 0.025) [0.000].

*Eligibility and definitions:* pooled scores use all scored rows where the metric is defined. Per-patient eligible counts are in each JSON cell's `eligible_n` (for shuffled draws, a minimum–maximum if it varies). Naive sensitivity uses patients with at least one event; naive specificity uses patients with at least one non-event; naive AP uses patients with an event; naive PPV/NPV require an alarm/no alarm. Tailored AUC/AP/sensitivity/specificity require both outcome classes; tailored PPV/NPV require an alarm/no alarm; tailored Brier skill requires a nonzero yardstick loss; calibration slope requires within-patient forecast variation. PPV/NPV are raw for pooled and naive scoring, and tailored PPV/NPV are excesses over each patient's observed event/non-event rate. Net benefit is raw versus treat-none for pooled and naive scoring, and an excess over the yardstick policy for tailored scoring. Brier skill summaries use the ratio of mean patient losses.

## College Experience

38 test patients; 6361 scored rows; training-cohort rate / alarm threshold 0.393980; scored-set event rate 0.398837.

| Metric | Run-in pooled | Run-in naive patient | Run-in tailored patient | Shuffled running pooled | Shuffled running naive patient | Shuffled running tailored patient |
|---|---:|---:|---:|---:|---:|---:|
| AUC | 0.695 [0.500] | 0.500 [0.500] | 0.500 [0.500] | 0.834 (0.833, 0.835) [0.500] | 0.412 (0.386, 0.443) [0.500] | 0.412 (0.386, 0.443) [0.412 (0.386, 0.443)] |
| Average precision | 0.600 [0.399] | 0.455 [0.394] | 0.455 [0.455] | 0.784 (0.782, 0.786) [0.399] | 0.431 (0.423, 0.441) [0.394] | 0.431 (0.423, 0.441) [0.431 (0.423, 0.441)] |
| Sensitivity | 0.432 [0.394] | 0.378 [0.394] | 0.378 [0.378] | 0.726 (0.700, 0.750) [0.394] | 0.492 (0.470, 0.519) [0.394] | 0.492 (0.470, 0.519) [0.492 (0.470, 0.519)] |
| Specificity | 0.781 [0.606] | 0.632 [0.606] | 0.622 [0.622] | 0.761 (0.737, 0.786) [0.606] | 0.516 (0.491, 0.541) [0.606] | 0.503 (0.477, 0.528) [0.503 (0.477, 0.528)] |
| PPV | 0.567 [0.399] | 0.620 [0.394] | 0.000 [0.000] | 0.669 (0.654, 0.685) [0.399] | 0.618 (0.591, 0.637) [0.394] | -0.007 (-0.021, 0.010) [-0.007 (-0.021, 0.010)] |
| NPV | 0.674 [0.601] | 0.660 [0.606] | 0.000 [0.000] | 0.808 (0.799, 0.816) [0.601] | 0.616 (0.588, 0.648) [0.606] | -0.030 (-0.058, 0.001) [-0.030 (-0.058, 0.001)] |
| Net benefit | 0.087 [0.000 / 0.008] | 0.138 [0.000] | -0.107 [-0.107] | 0.196 (0.195, 0.198) [0.000 / 0.008] | 0.243 (0.241, 0.245) [0.000] | 0.000 (0.000, 0.000) [0.000 (0.000, 0.000)] |
| Brier skill | 0.011 [0.000] | 0.053 [0.000] | -0.461 [-0.461] | 0.328 (0.328, 0.328) [0.000] | 0.351 (0.351, 0.351) [0.000] | 0.000 (0.000, 0.000) [0.000 (0.000, 0.000)] |
| Calibration slope | 0.165 [1.000] | n/a [n/a (no within-patient forecast variation)] | n/a [n/a] | 1.021 (0.990, 1.055) [1.000] | -0.069 (-0.196, 0.018) [1.000] | -0.069 (-0.196, 0.018) [-0.069 (-0.196, 0.018)] |
*Eligible patient n (run-in / shuffled running):* naive: AUC 37/37, AP 37/37, sens 37/37, spec 38/38, PPV 14/23–25, NPV 24/29–32, NB 38/38, Brier 38/38, slope 0/37; tailored: AUC 37/37, AP 37/37, sens 37/37, spec 37/37, PPV 14/23–25, NPV 24/29–32, NB 38/38, Brier 37/37, slope 0/37. Youden groups: run-in sens 37; spec 38; shuffled sens 37; spec 38.


*Naive Youden (separate sensitivity/specificity groups; chance 0):* run-in 0.010 [0.000]; shuffled running 0.008 (-0.006, 0.025) [0.000].

*Eligibility and definitions:* pooled scores use all scored rows where the metric is defined. Per-patient eligible counts are in each JSON cell's `eligible_n` (for shuffled draws, a minimum–maximum if it varies). Naive sensitivity uses patients with at least one event; naive specificity uses patients with at least one non-event; naive AP uses patients with an event; naive PPV/NPV require an alarm/no alarm. Tailored AUC/AP/sensitivity/specificity require both outcome classes; tailored PPV/NPV require an alarm/no alarm; tailored Brier skill requires a nonzero yardstick loss; calibration slope requires within-patient forecast variation. PPV/NPV are raw for pooled and naive scoring, and tailored PPV/NPV are excesses over each patient's observed event/non-event rate. Net benefit is raw versus treat-none for pooled and naive scoring, and an excess over the yardstick policy for tailored scoring. Brier skill summaries use the ratio of mean patient losses.
