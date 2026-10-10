# withinpatient 2.0.0

* Ported the Python 2.0.0 evaluation API to R. The two-step fix scores each patient against within-patient outcome shuffles with history forecasts and outcome-based yardsticks rebuilt after every shuffle, then reports per-patient excesses and equal-patient summaries alongside pooled and usual per-patient comparisons.
* Added the nine and naive metric helpers, the exact conditional-logit calibration slope, cross-language permutation-index fixtures, and a tailored-first print method. The old report API remains available as `evaluate_v1()` with a deprecation warning.

```r
result <- evaluate(forecasts, patient = "patient", time = "time",
  outcome = "y", forecast = "forecast", runin = 7,
  rebuild = "running_rate", yardstick = "running_rate",
  n_perm = 200, seed = 17)
print(result)
```
