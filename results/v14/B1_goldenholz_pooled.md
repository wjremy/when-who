# B1. Goldenholz reconstruction, pooled half (12 Oct 2026)

Script: `code/v14_goldenholz_pooled.py` (imports the profiles of `code/v13_goldenholz.py`; same seeds and order of random draws). Numbers: `results/v14/B1_goldenholz_pooled.csv`. Check built into the script: the per-patient AUC of the five v13 reconstructions equals `results/v13/A5_goldenholz.csv` to 1e-12, so these are the same simulated cohorts.

Each row: 5,000 cohorts of 25 patients, independent seizure days at a constant rate per patient, forecast = share of seizure days in the previous 84 days, scored on the days after the first 84. Mean over cohorts, with the 2.5th to 97.5th percentile over cohorts in parentheses.

- Pooled AUC: all scored days of all 25 patients in one ranking.
- A0: Equation 1, A0 = 1/2 + E|p - p'| / (4 mu (1 - mu)), from the reconstructed patients' true daily rates, each patient weighted by the number of scored days. It varies over cohorts only because counts and record lengths are paired at random.
- "True rate": pooled AUC of the forecast that states each patient's true rate (the finite-sample counterpart of A0).

| Reconstruction | Per-patient AUC | Pooled AUC, no-skill moving average | A0 (Equation 1) | Pooled AUC, true rate | Rate: mean, SD over patients |
|---|---|---|---|---|---|
| Primary (lengths 1x15, 11x36, 13x57; counts 1, 2, ..., 24, 56) | 0.35 (0.30 to 0.41) | **0.69 (0.64 to 0.74)** | **0.71 (0.69 to 0.73)** | 0.71 (0.66 to 0.76) | 0.108, 0.083 |
| Short records (12x15, 13x57) | 0.34 (0.28 to 0.41) | 0.69 (0.62 to 0.76) | 0.71 (0.68 to 0.75) | 0.71 (0.65 to 0.77) | 0.109, 0.086 |
| Long records (1x15, 11x56, 13x57) | 0.36 (0.31 to 0.41) | 0.69 (0.64 to 0.73) | 0.71 (0.69 to 0.72) | 0.71 (0.67 to 0.75) | 0.102, 0.078 |
| Counts concentrated at the median (1, 23x13, 56) | 0.36 (0.31 to 0.41) | 0.58 (0.53 to 0.64) | 0.60 (0.58 to 0.61) | 0.60 (0.55 to 0.65) | 0.108, 0.068 |
| Half as many seizure days | 0.33 (0.26 to 0.40) | 0.66 (0.59 to 0.73) | 0.70 (0.68 to 0.72) | 0.70 (0.63 to 0.76) | 0.054, 0.042 |
| NEW: widest counts (12x1, 13, 12x56) | 0.37 (0.32 to 0.43) | 0.82 (0.78 to 0.85) | 0.82 (0.79 to 0.85) | 0.82 (0.79 to 0.86) | 0.211, 0.206 |
| NEW: counts fall in the scored days only | 0.38 (0.34 to 0.43) | 0.76 (0.71 to 0.81) | 0.77 (0.73 to 0.81) | 0.77 (0.72 to 0.82) | 0.298, 0.219 |

Three decimals for the primary row: per patient 0.354 (0.301 to 0.409), pooled 0.689 (0.637 to 0.740), A0 0.712 (0.693 to 0.728).

## What fixes the patients' seizure rates

Nothing in the publication does. It gives, for the 25 patients, only the minimum, median and maximum of the total seizures per patient (1, 13, 56) and of the forecastable days (15, 57, 57). The reconstruction sets each patient's daily rate to (seizure count) / (84 history days + scored days). Three choices of ours therefore fix the rates: (i) the 22 counts between the reported minimum, median and maximum (primary: 1, 2, ..., 24, 56), (ii) that a seizure count is a count of seizure days, (iii) that the counts cover the 84 history days plus the scored days. The pairing of counts with record lengths is random per cohort.

## Sensitivity

- The per-patient value hardly depends on these choices: 0.33 to 0.36 over the five v13 reconstructions, 0.33 to 0.38 with the two new ones.
- The pooled value depends on them almost entirely through the spread of the rates, as Equation 1 says. Over the five v13 reconstructions the pooled no-skill AUC is 0.58 to 0.69 (A0 0.60 to 0.71). Record lengths do not matter (0.69 in all three length profiles); the spread of the counts does. The narrowest count vector that has the reported minimum, median and maximum gives 0.58; the widest gives 0.82 (A0 0.82). So the published summaries allow any pooled no-skill value from about 0.58 to 0.82, and the published pooled AUC of 0.82 is at the top of that range.
- In every reconstruction the no-skill moving average scores far higher pooled than per patient (gap 0.22 to 0.44), and its pooled AUC stays within 0.04 of A0.

## For the text (Conceptual decides)

- Supported: in the same simulated cohorts in which the no-skill moving average scored 0.35 per patient, it scored 0.69 pooled (0.64 to 0.74); a forecast with no skill at all thus shows the published pattern of a high pooled and a below-0.5 per-patient AUC.
- Not supported as "explains 0.82": the pooled no-skill value reaches 0.82 only with the widest spread of seizure counts that the published summaries allow; with the primary reconstruction it is 0.69. The publication does not report the patients' rates, so the pooled half cannot be pinned down more closely than 0.58 to 0.82.
- If the two new reconstructions are mentioned, "Other reconstructions gave 0.33 to 0.36" (v13, 6.7) becomes 0.33 to 0.38.
- The published 0.82 is the group-level AUC of the AI forecast (see `results/v13/A5_sources.md`); the simulation is of the moving average.
