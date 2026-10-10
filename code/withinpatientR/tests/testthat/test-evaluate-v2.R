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
  fixed <- evaluate(data, time = "time", runin = runin, n_perm = n_perm, seed = 19, perm_index = maps)
  custom <- evaluate(data, time = "time", runin = runin, n_perm = n_perm, seed = 19,
    metrics = list(mean_forecast = function(y, p) mean(p)), perm_index = maps)
  running_data <- data
  running_data$forecast <- unlist(lapply(split(data, data$patient), function(rows) {
    stops <- vapply(rows$time, function(t) sum(rows$time < t), integer(1))
    (0.5 + c(0, cumsum(rows$y))[stops + 1L]) / (1 + stops)
  }), use.names = FALSE)
  running <- evaluate(running_data, time = "time", runin = runin, rebuild = "running_rate",
    yardstick = "running_rate", n_perm = n_perm, seed = 19,
    metrics = c("auc", "average_precision", "calibration_slope"), perm_index = maps)
  window_data <- data
  window_data$forecast <- unlist(lapply(split(data, data$patient), function(rows) {
    stops <- vapply(rows$time, function(t) sum(rows$time < t), integer(1))
    csum <- c(0, cumsum(rows$y)); starts <- pmax(stops - 5L, 0L)
    events <- csum[stops + 1L] - csum[starts + 1L]
    counts <- pmin(stops, 5L)
    ifelse(counts > 0, events / counts, 0)
  }), use.names = FALSE)
  window <- evaluate(window_data, time = "time", runin = runin,
    rebuild = list(name = "window_count", w = 5L, as_rate = TRUE),
    n_perm = n_perm, seed = 19, metrics = c("auc", "average_precision"), perm_index = maps)
  actual <- c(flatten_v2(fixed, "fixed_default_yardstick"),
    flatten_v2(custom, "fixed_user_metric"),
    flatten_v2(running, "running_rate_yardstick"),
    flatten_v2(window, "window_count_default_yardstick"))
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
  a <- evaluate(data, runin = 4, n_perm = 17, seed = 23, metrics = c("auc", "average_precision"))
  b <- evaluate(data, runin = 4, n_perm = 17, seed = 23, metrics = c("auc", "average_precision"))
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
  fit <- evaluate(d, metrics = "auc", n_perm = 199, seed = 3)
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
  result <- evaluate(data, runin = 4, n_perm = 3, metrics = "auc")
  printed <- capture.output(print(result))
  expect_true(grepl("Mean observed", printed[[1L]], fixed = TRUE))
  expect_true(grepl("auc", printed[[3L]], fixed = TRUE))
  expect_true(grepl("Threshold", printed[[length(printed)]], fixed = TRUE))
})
