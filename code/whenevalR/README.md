# wheneval

`wheneval` evaluates repeated binary forecasts patient by patient and compares each score with a shuffled baseline. Each patient's scored outcomes are shuffled within their record; outcome-history predictions and references are rebuilt after every shuffle. Results include per-patient scores and excesses, equal-patient summaries, and pooled and per-patient score comparisons.

## Install

```r
install.packages("code/whenevalR", repos = NULL, type = "source")
```

For development, install `testthat` and use `devtools::load_all("code/whenevalR")`.

## Quickstart

```r
library(wheneval)
set.seed(17)
n_patient <- 40
n_time <- 35
patient_id <- rep(seq_len(n_patient), each = n_time)
time <- rep(seq_len(n_time), times = n_patient)
rate <- rep(rbeta(n_patient, 3, 12), each = n_time)
prediction <- pmin(0.99, pmax(0.01, rate + rnorm(length(patient_id), 0, 0.08)))
outcome <- rbinom(length(patient_id), 1, prediction)
data <- data.frame(patient_id, time, outcome, prediction)

result <- evaluate(data, time = "time", runin = 7,
                   threshold = 0.25, n_shuffles = 200, seed = 17)
print(result)
```

The default threshold is the cohort's scored event rate. Rows before the run-in cutoff remain available as history and are excluded from scores. By default, `yardstick = NULL` is each patient's scored event rate recalculated in every shuffle. Use `history_rule = "rate_so_far"`, `history_rule = "window_count"` with `window = 7`, a `lagged_outcomes` rule, or a custom function to rebuild predictions after each shuffle.

Use `refit(fit_predict, folds = 5)` for a model fitted on a patient's own record with folds rotating over that record. `fit_predict(train_rows, test_rows)` is called again in every shuffle; test outcomes are missing. Set `prediction = NULL` to compute observed predictions from the history rule. When a prediction column is supplied, the rule must reproduce its predictions on scored rows.

`result$metrics[[name]]$per_patient` contains `score`, `shuffled_baseline`, `excess`, `p_value`, and `n_scored`. The summary uses `mean_score` and `mean_shuffled_baseline`, and comparisons use `per_patient_mean`, `per_patient_median`, `per_patient_iqr`, and `per_patient_n` alongside `pooled`.

## Custom metrics

```r
result <- evaluate(data, runin = 7, metrics = list(
  mean_prediction = function(y, p) mean(p)
), n_shuffles = 200, seed = 17)
```

Select built-ins by name, for example `metrics = c("auc", "average_precision")`. `nine_metrics()` and `naive_metrics()` expose metric helpers, and `conditional_within_slope()` fits the exact conditional logistic slope.

For the preprint and software references, run `citation("wheneval")`.
