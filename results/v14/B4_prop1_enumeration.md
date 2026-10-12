# B4. Exact check of the extension of Proposition 1

Script: `code/v14_prop1_enumeration.py`. Numbers per rule: `results/v14/B4_prop1_enumeration.csv`.

Exact enumeration in rational arithmetic: every record length T from 2 to 12, every event count 0 < n < T, all C(T, n) arrangements (8,166 arrangements per rule), 21 forecast rules, and for the alarm-based measures every cutoff that changes the alarms. Each rule was first verified on all 2^T outcome vectors to ignore the day's own outcome and never to decrease when another day becomes an event day.

Rules (19 that respond to outcomes, plus a constant and a fixed ranking as equality cases): `rate_so_far_smoothed`, `rate_so_far_raw`, `rate_other_days`, `rate_later_days`, `previous_day`, `window_2_smoothed`, `window_4_smoothed`, `two_of_last_three`, `any_earlier_event`, `exponential_weights`, `or_and`, `random_weighted_threshold_1`, `random_weighted_threshold_2`, `random_weighted_threshold_3`, `random_weighted_sum_1`, `random_weighted_sum_2`, `random_weighted_sum_3`, `random_dnf_1`, `random_dnf_2`; `constant`, `fixed_ranking`.

A case is one combination of T, n, rule and, where it applies, cutoff and tau. Gap = the side that must be smaller minus the side that must be larger; a violation is a gap above zero.

| Statement | Cases | Violations | Equalities | Strict cases | Gap closest to zero among strict cases |
|---|---:|---:|---:|---:|---:|
| E[AUC] <= 1/2 (Proposition 1) | 1,386 | 0 | 168 | 1,218 | -0.00101 |
| E[sensitivity + specificity - 1] <= 0 (Proposition 1) | 99,637 | 0 | 3,278 | 96,359 | -0.00018 |
| (i) E[true alarms] <= (n/T) E[alarms] | 99,637 | 0 | 3,278 | 96,359 | -0.000526 |
| (ii) E[true non-alarms] <= (1 - n/T) E[non-alarms] | 99,637 | 0 | 3,278 | 96,359 | -0.000526 |
| (iii) E[net benefit] <= (E[alarms]/T)(q - tau)/(1 - tau) | 2,052,836 | 0 | 65,461 | 1,987,375 | -4.61e-05 |
| (iii) E[net benefit] <= that of always or never alarming | 2,052,836 | 0 | 29,800 | 2,023,036 | -7.52e-05 |
| (iv) E[Brier] >= (n/T)(1 - n/T) + mean E[(f_t - n/T)^2] | 1,386 | 0 | 168 | 1,218 | -0.000481 |
| (v) E[AP] <= random-ranking value, ties grouped | 1,386 | 0 | 66 | 1,320 | -0.0159 |
| (v) E[AP] <= random-ranking value, ties broken at random | 1,386 | 0 | 168 | 1,218 | -0.000269 |
| (vi) E[PPV | at least one alarm] <= n/T | 98,251 | 0 | 2,011 | 96,240 | -4.78e-05 |
| (vi) E[NPV | at least one day without alarm] <= 1 - n/T | 98,251 | 0 | 1,892 | 96,359 | -4.78e-05 |

No counterexample to any statement.

Statements (i), (ii) and the sensitivity-plus-specificity statement of Proposition 1 are one inequality: E[true non-alarms] - (1 - q) E[non-alarms] = E[true alarms] - q E[alarms], and E[sensitivity + specificity - 1] is that quantity divided by n (1 - q). Their cases, equalities and signs therefore coincide.

Equalities occur where the alarms or the ranking do not depend on the outcomes (all or no days alarmed, the constant forecast, the fixed ranking, forecasts that are constant for the given T and n) and, for the bound by always or never alarming, where tau = q. With ties grouped, a constant forecast has average precision n/T, below the random-ranking value; with ties broken at random it has exactly that value.
