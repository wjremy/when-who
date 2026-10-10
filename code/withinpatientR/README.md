# withinpatient

`withinpatient` evaluates repeated binary forecasts patient by patient and compares each observed metric with chance after shuffling outcomes within that patient. The v2 interface ports Python `withinpatient` 2.0.0, including rebuilt history forecasts, rebuilt yardsticks, custom metrics, and the exact conditional calibration slope.

## Install

Install the package from this source directory in R:

```r
install.packages("code/withinpatientR", repos = NULL, type = "source")
```

For development, install `testthat` and use `devtools::load_all("code/withinpatientR")`.

## The two-step evaluation

1. Compute each patient's observed metric, shuffle only that patient's scored outcomes, rebuild any outcome-history forecast or yardstick, and compute the same metric for every shuffle.
2. Report each patient's observed value, permutation chance level, excess and upper-tail p-value, then summarize patient excesses equally and show pooled and usual per-patient comparisons.

## Quickstart

```r
library(withinpatient)
set.seed(17)
n_patient <- 40
n_time <- 35
patient <- rep(seq_len(n_patient), each = n_time)
time <- rep(seq_len(n_time), times = n_patient)
rate <- rep(rbeta(n_patient, 3, 12), each = n_time)
forecast <- pmin(0.99, pmax(0.01, rate + rnorm(length(patient), 0, 0.08)))
y <- rbinom(length(patient), 1, forecast)
data <- data.frame(patient, time, y, forecast)

result <- evaluate(data, time = "time", runin = 7,
                   threshold = 0.25, n_perm = 200, seed = 17)
print(result)
```

The default threshold is the cohort's scored event rate. Rows before day 7 remain available as history and are excluded from scores. By default, `yardstick = NULL` is each patient's scored event rate recalculated in every shuffle. Use `rebuild = "running_rate"`, a `window_count` specification, a `lagged_outcomes` prediction function, or a custom function to rebuild forecasts after each shuffle.

`evaluate_v1()` retains the previous report API and warns that it is deprecated.

## Custom metrics

```r
result <- evaluate(data, runin = 7, metrics = list(
  mean_forecast = function(y, p) mean(p)
), n_perm = 200, seed = 17)
```

Select built-ins by name, for example `metrics = c("auc", "average_precision")`. `nine_metrics()` and `naive_metrics()` expose the two comparison summaries, and `conditional_within_slope()` fits the exact conditional logistic slope.

For the preprint and software references, run `citation("withinpatient")`.
