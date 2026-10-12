# wheneval 3.0.0

## Changes in 3.0.0

The package is now named `wheneval` (formerly `withinpatient`). `evaluate()` uses `patient_id`, `prediction`, `history_rule`, `n_shuffles`, and `baseline`; the old names `patient`, `forecast`, `rebuild`, `n_perm`, and `chance` remain accepted through `...`. History rules now include `refit()` for models fitted on each patient's own record, and `prediction = NULL` computes observed predictions from the rule. Result keys now use `score`, `shuffled_baseline`, `mean_score`, `mean_shuffled_baseline`, and `per_patient_*` comparison names.

The default print table reports score, shuffled baseline, and excess first, followed by pooled and per-patient score comparisons.

```r
result <- evaluate(forecasts, patient_id = "patient_id", time = "time",
  outcome = "outcome", prediction = "prediction", runin = 7,
  history_rule = "rate_so_far", yardstick = "rate_so_far",
  n_shuffles = 200, seed = 17)
print(result)
```

# wheneval 2.0.0

Ported the two-step patient-wise shuffle procedure to R. Forecasts based on outcome history and outcome-based yardsticks are rebuilt after each within-patient shuffle. Added the nine and naive metric helpers, exact conditional-logit calibration slope, cross-language permutation-index fixtures, and a tailored-first print method. The earlier report API remains available as `evaluate_v1()`.
