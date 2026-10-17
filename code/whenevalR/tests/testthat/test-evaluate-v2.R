read_v2_maps <- function(permutations) {
  draws <- split(permutations, permutations$iteration)
  lapply(draws, function(draw) {
    groups <- split(draw, draw$patient)
    lapply(groups, function(rows) as.integer(rows$source[order(rows$position)]))
  })
}

flatten_v2 <- function(x, prefix = "") {
  out <- numeric()
  if (is.list(x)) {
    for (i in seq_along(x)) {
      nm <- names(x)[[i]]
      path <- if (is.null(nm) || !nzchar(nm)) paste0(prefix, "[", i, "]") else if (nzchar(prefix)) paste(prefix, nm, sep = ".") else nm
      out <- c(out, flatten_v2(x[[i]], path))
    }
  } else if ((is.numeric(x) || is.logical(x)) && length(x)) {
    if (length(x) == 1L) out[prefix] <- as.numeric(x)
    else for (i in seq_along(x)) {
      nm <- names(x)[[i]]
      path <- if (is.null(nm) || !nzchar(nm)) paste0(prefix, "[", i, "]") else paste(prefix, nm, sep = ".")
      out[path] <- as.numeric(x[[i]])
    }
  }
  out
}

test_that("v2 evaluate matches Python results with shared permutation indices", {
  fixture_dir <- testthat::test_path("fixtures")
  data <- utils::read.csv(file.path(fixture_dir, "sim.csv"), stringsAsFactors = FALSE)
  perms <- utils::read.csv(file.path(fixture_dir, "permutations.csv"), stringsAsFactors = FALSE)
  expected_df <- utils::read.csv(file.path(fixture_dir, "evaluate_expected.csv"), stringsAsFactors = FALSE, colClasses = "character")
  expected <- setNames(expected_df$value, expected_df$path)
  maps <- read_v2_maps(perms)
  n_perm <- length(maps)
  runin <- 4
  fixed <- evaluate(data, patient_id = "patient", time = "time", outcome = "y", prediction = "forecast",
    runin = runin, n_shuffles = n_perm, seed = 19, perm_index = maps)
  custom <- evaluate(data, patient_id = "patient", time = "time", outcome = "y", prediction = "forecast",
    runin = runin, n_shuffles = n_perm, seed = 19,
    metrics = list(mean_forecast = function(y, p) mean(p)), perm_index = maps)
  running_data <- data
  running_data$forecast <- unlist(lapply(split(data, data$patient), function(rows) {
    stops <- vapply(rows$time, function(t) sum(rows$time < t), integer(1))
    (0.5 + c(0, cumsum(rows$y))[stops + 1L]) / (1 + stops)
  }), use.names = FALSE)
  running <- evaluate(running_data, patient_id = "patient", time = "time", outcome = "y", prediction = "forecast",
    runin = runin, history_rule = "rate_so_far", yardstick = "rate_so_far",
    n_shuffles = n_perm, seed = 19,
    metrics = c("auc", "average_precision", "calibration_slope"), perm_index = maps)
  window_data <- data
  window_data$forecast <- unlist(lapply(split(data, data$patient), function(rows) {
    stops <- vapply(rows$time, function(t) sum(rows$time < t), integer(1))
    csum <- c(0, cumsum(rows$y)); starts <- pmax(stops - 5L, 0L)
    events <- csum[stops + 1L] - csum[starts + 1L]
    counts <- pmin(stops, 5L)
    ifelse(counts > 0, events / counts, 0)
  }), use.names = FALSE)
  window <- evaluate(window_data, patient_id = "patient", time = "time", outcome = "y", prediction = "forecast",
    runin = runin, history_rule = "window_count", window = 5L,
    n_shuffles = n_perm, seed = 19, metrics = c("auc", "average_precision"), perm_index = maps)
  fit_predict <- function(train_rows, test_rows) {
    rep((sum(train_rows$y) + 1) / (nrow(train_rows) + 2), nrow(test_rows))
  }
  refit_data <- data
  for (key in unique(data$patient)) {
    ix <- which(data$patient == key)
    rows <- data[ix, , drop = FALSE]
    fold_id <- (seq_len(nrow(rows)) - 1L) %% 5L
    pred <- numeric(nrow(rows))
    for (fold in 0:4) {
      test_ix <- which(fold_id == fold); train_ix <- which(fold_id != fold)
      test_rows <- rows[test_ix, , drop = FALSE]; test_rows$y <- NA_real_
      pred[test_ix] <- fit_predict(rows[train_ix, , drop = FALSE], test_rows)
    }
    refit_data$forecast[ix] <- pred
  }
  rule <- refit(fit_predict, folds = 5L)
  refit_column <- evaluate(refit_data, patient_id = "patient", time = "time", outcome = "y", prediction = "forecast",
    runin = runin, history_rule = rule, n_shuffles = n_perm, seed = 19,
    metrics = c("auc", "average_precision"), perm_index = maps)
  refit_none <- evaluate(refit_data[, setdiff(names(refit_data), "forecast"), drop = FALSE],
    patient_id = "patient", time = "time", outcome = "y", prediction = NULL,
    history_rule = rule, runin = runin, n_shuffles = n_perm, seed = 19,
    metrics = c("auc", "average_precision"), perm_index = maps)
  expect_equal(refit_column, refit_none)
  actual <- c(flatten_v2(fixed, "fixed_default_yardstick"),
    flatten_v2(custom, "fixed_user_metric"),
    flatten_v2(running, "running_rate_yardstick"),
    flatten_v2(window, "window_count_default_yardstick"),
    flatten_v2(refit_column, "refit_prediction_column"),
    flatten_v2(refit_none, "refit_prediction_none"))
  expect_setequal(names(actual), names(expected))
  for (path in names(expected)) {
    want <- expected[[path]]; got <- actual[[path]]
    if (want %in% c("NaN", "Inf", "-Inf")) {
      expect_identical(is.nan(got), identical(want, "NaN"), info = path)
    } else {
      tolerance <- if (grepl("calibration_slope", path, fixed = TRUE)) 1e-5 else 1e-8
      expect_true(is.finite(got) && abs(got - as.numeric(want)) <= tolerance, info = path)
    }
  }
})

test_that("v2 seeds are reproducible and p-values are bounded", {
  fixture_dir <- testthat::test_path("fixtures")
  data <- utils::read.csv(file.path(fixture_dir, "sim.csv"), stringsAsFactors = FALSE)
  a <- evaluate(data, patient_id = "patient", outcome = "y", prediction = "forecast",
    runin = 4, n_shuffles = 17, seed = 23, metrics = c("auc", "average_precision"))
  b <- evaluate(data, patient_id = "patient", outcome = "y", prediction = "forecast",
    runin = 4, n_shuffles = 17, seed = 23, metrics = c("auc", "average_precision"))
  expect_equal(a, b)
  p <- unlist(lapply(a$metrics, function(m) vapply(m$per_patient, `[[`, numeric(1), "p_value")))
  expect_true(all(p >= 1 / 18 & p <= 1))
})

test_that("level-only forecast has near-zero permutation excess", {
  set.seed(11)
  d <- expand.grid(patient = paste0("p", 1:16), time = 1:30)
  rates <- setNames(runif(16, 0.08, 0.5), paste0("p", 1:16))
  d$forecast <- unname(rates[d$patient])
  d$y <- rbinom(nrow(d), 1, d$forecast)
  fit <- evaluate(d, patient_id = "patient", outcome = "y", prediction = "forecast",
    metrics = "auc", n_shuffles = 199, seed = 3)
  expect_true(abs(fit$metrics$auc$summary$mean_excess) < 0.12)
})

test_that("evaluate_v1 warns about deprecation", {
  fixture_dir <- testthat::test_path("fixtures")
  data <- utils::read.csv(file.path(fixture_dir, "sim.csv"), stringsAsFactors = FALSE)
  expect_warning(evaluate_v1(data, time = "time", n_perm = 1), "deprecated")
})

test_that("the v2 print method puts tailored results first", {
  fixture_dir <- testthat::test_path("fixtures")
  data <- utils::read.csv(file.path(fixture_dir, "sim.csv"), stringsAsFactors = FALSE)
  result <- evaluate(data, patient_id = "patient", outcome = "y", prediction = "forecast",
    runin = 4, n_shuffles = 3, metrics = "auc")
  printed <- capture.output(print(result))
  expect_true(grepl("Shuffled baseline", printed[[1L]], fixed = TRUE))
  expect_true(grepl("| Metric | Score (mean) | Shuffled baseline (mean) | Excess (mean) | Excess (median [IQR]) |", printed[[1L]], fixed = TRUE))
  expect_true(grepl("auc", printed[[3L]], fixed = TRUE))
  expect_true(grepl("Threshold", printed[[length(printed)]], fixed = TRUE))
})

test_that("legacy names work and invalid arguments are rejected", {
  fixture_dir <- testthat::test_path("fixtures")
  data <- utils::read.csv(file.path(fixture_dir, "sim.csv"), stringsAsFactors = FALSE)
  canonical <- evaluate(data, patient_id = "patient", time = "time", outcome = "y", prediction = "forecast",
    metrics = "auc", n_shuffles = 7, seed = 3)
  legacy <- evaluate(data, patient = "patient", time = "time", outcome = "y", forecast = "forecast",
    metrics = "auc", n_perm = 7, seed = 3, chance = "mean")
  expect_equal(canonical, legacy)
  expect_error(evaluate(data, patient_id = "patient", patient = "patient"), "both")
  expect_error(evaluate(data, unknown_argument = TRUE), "unused argument")
  expect_error(evaluate(data, patient_id = "patient", outcome = "y", prediction = "forecast",
    history_rule = "window_count", metrics = "auc"), "requires a width")
  expect_error(evaluate(data, patient_id = "patient", outcome = "y", prediction = "forecast",
    history_rule = list(name = "window_count", w = 5), window = 7, metrics = "auc"), "conflict")
  expect_error(evaluate(data, patient_id = "patient", outcome = "y", prediction = NULL), "requires a history_rule")
  expect_error(refit(function(train, test) rep(0.5, nrow(test)), folds = 1), "at least 2")
})

test_that("refitting reaches the shuffled baseline on finite independent records", {
  set.seed(20261012)
  n_patients <- 300L; n_days <- 40L; folds <- 5L
  d <- data.frame(patient_id = rep(seq_len(n_patients), each = n_days),
    time = rep(seq_len(n_days), times = n_patients),
    outcome = rbinom(n_patients * n_days, 1, 0.3))
  fit_predict <- function(train_rows, test_rows) {
    rep((sum(train_rows$outcome) + 1) / (nrow(train_rows) + 2), nrow(test_rows))
  }
  d$prediction <- NA_real_
  for (pid in seq_len(n_patients)) {
    ix <- which(d$patient_id == pid); rows <- d[ix, , drop = FALSE]
    fold_id <- (seq_len(nrow(rows)) - 1L) %% folds
    for (fold in 0:(folds - 1L)) {
      test_ix <- which(fold_id == fold); train_ix <- which(fold_id != fold)
      test_rows <- rows[test_ix, , drop = FALSE]; test_rows$outcome <- NA_real_
      d$prediction[ix[test_ix]] <- fit_predict(rows[train_ix, , drop = FALSE], test_rows)
    }
  }
  refitted <- evaluate(d, history_rule = refit(fit_predict, folds = folds),
    metrics = "auc", n_shuffles = 99, seed = 17)
  frozen_rule <- function(y, scored, rows) rows$prediction
  frozen <- evaluate(d, history_rule = frozen_rule, metrics = "auc", n_shuffles = 99, seed = 17)
  refit_summary <- refitted$metrics$auc$summary
  frozen_summary <- frozen$metrics$auc$summary
  expect_lt(refit_summary$mean_score, 0.5)
  expect_lt(abs(refit_summary$mean_excess), 0.02)
  expect_lt(abs(frozen_summary$mean_shuffled_baseline - 0.5), 0.03)
  expect_gt(abs(refit_summary$mean_shuffled_baseline - frozen_summary$mean_shuffled_baseline), 0.03)
})
