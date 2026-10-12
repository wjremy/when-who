flatten_numeric <- function(x, prefix = "") {
  out <- numeric()
  if (is.list(x)) {
    for (i in seq_along(x)) {
      nm <- names(x)[[i]]
      if (is.null(nm) || is.na(nm) || !nzchar(nm)) nm <- paste0("[", i, "]")
      path <- if (startsWith(nm, "[")) paste0(prefix, nm) else if (nzchar(prefix)) paste(prefix, nm, sep = ".") else nm
      out <- c(out, flatten_numeric(x[[i]], path))
    }
  } else if (is.numeric(x) || is.logical(x)) {
    if (length(x) == 1L) {
      out[prefix] <- as.numeric(x)
    } else {
      nms <- names(x)
      for (i in seq_along(x)) {
        nm <- if (!is.null(nms) && nzchar(nms[[i]])) nms[[i]] else paste0("[", i, "]")
        path <- if (startsWith(nm, "[")) paste0(prefix, nm) else if (nzchar(prefix)) paste(prefix, nm, sep = ".") else nm
        out[path] <- as.numeric(x[[i]])
      }
    }
  }
  out
}

read_fixture_maps <- function(permutations) {
  draws <- split(permutations, permutations$iteration)
  lapply(draws, function(draw) {
    groups <- split(draw, draw$patient)
    lapply(groups, function(rows) rows$source[order(rows$position)])
  })
}

test_that("R metrics and evaluate agree with fixed Python fixtures", {
  fixture_dir <- testthat::test_path("fixtures")
  data <- utils::read.csv(file.path(fixture_dir, "sim.csv"), stringsAsFactors = FALSE)
  perms <- utils::read.csv(file.path(fixture_dir, "permutations.csv"), stringsAsFactors = FALSE)
  expected_df <- utils::read.csv(file.path(fixture_dir, "expected.csv"), stringsAsFactors = FALSE, colClasses = "character")
  expected <- setNames(expected_df$value, expected_df$path)
  maps <- read_fixture_maps(perms)
  scored <- data[data$time >= 4, , drop = FALSE]
  runin <- runin_patient_rate(data$patient, data$y, data$time, 4)
  expanding <- expanding_patient_rate(data$patient, data$y, data$time)
  expanding_prior <- expanding_patient_rate(data$patient, data$y, data$time, prior_rate = 0.18, prior_weight = 1.25)

  direct <- list(
    auc = auc(scored$y, scored$forecast),
    pooled_auc = pooled_auc(scored$patient, scored$y, scored$forecast),
    within_patient_auc = within_patient_auc(scored$patient, scored$y, scored$forecast),
    within_patient_auc_pairs = within_patient_auc(scored$patient, scored$y, scored$forecast, weights = "pairs"),
    within_patient_auc_days = within_patient_auc(scored$patient, scored$y, scored$forecast, weights = "days"),
    per_patient_auc_summary = per_patient_auc_summary(scored$patient, scored$y, scored$forecast),
    per_time_point_auc = per_time_point_auc(scored, "time", min_patients = 5),
    rate_benchmark = rate_benchmark(scored$patient, scored$y),
    rate_benchmark_runin = rate_benchmark(scored$patient, scored$y, runin[data$time >= 4]),
    population_a0_moment_corrected = population_a0_moment_corrected(scored$patient, scored$y),
    brier_scores = brier_scores(scored$patient, scored$y, scored$forecast),
    brier_skill_runin = brier_skill(scored$y, scored$forecast, runin[data$time >= 4]),
    brier_skill_expanding = brier_skill(scored$y, scored$forecast, expanding[data$time >= 4]),
    brier_skill_summary_runin = brier_skill_summary(scored$patient, scored$y, scored$forecast, runin[data$time >= 4]),
    brier_skill_summary_expanding = brier_skill_summary(scored$patient, scored$y, scored$forecast, expanding[data$time >= 4]),
    patient_calibration_in_the_large = patient_calibration_in_the_large(scored$patient, scored$y, scored$forecast),
    runin_patient_rate = runin,
    expanding_patient_rate = expanding,
    expanding_patient_rate_prior = expanding_prior,
    pooled_auc_permutation_test = pooled_auc_permutation_test(scored$patient, scored$y, scored$forecast,
      n_perm = length(maps), permutations = maps),
    permutation_reference = permutation_reference(data$patient, data$y, data$time,
      function(outcomes) {
        earlier <- c(0, head(cumsum(outcomes), -1))
        (0.25 + earlier) / seq_along(outcomes)
      }, n_perm = length(maps), runin_days = 4, permutations = maps)
  )

  result <- suppressWarnings(evaluate_v1(data, time = "time", runin_days = 4, n_perm = length(maps),
    history_forecast_fn = function(outcomes) {
      earlier <- c(0, head(cumsum(outcomes), -1))
      (0.25 + earlier) / seq_along(outcomes)
    }, permutations = maps, return_references = TRUE))
  wrapper <- result[c("cohort", "auc", "brier", "skill", "calibration_in_the_large", "excluded_patients", "references")]
  actual <- c(flatten_numeric(list(direct = direct)), flatten_numeric(list(evaluate = wrapper)))
  expect_setequal(names(actual), names(expected))
  for (path in names(expected)) {
    want <- expected[[path]]
    got <- actual[[path]]
    if (want %in% c("NaN", "Inf", "-Inf")) {
      expect_identical(is.nan(got), identical(want, "NaN"), info = path)
      if (want == "Inf") expect_identical(got, Inf, info = path)
      if (want == "-Inf") expect_identical(got, -Inf, info = path)
    } else {
      expect_true(is.finite(got) && abs(got - as.numeric(want)) <= 1e-10, info = path)
    }
  }
})

test_that("the print method gives a compact summary", {
  fixture_dir <- testthat::test_path("fixtures")
  data <- utils::read.csv(file.path(fixture_dir, "sim.csv"), stringsAsFactors = FALSE)
  result <- suppressWarnings(evaluate_v1(data, time = "time", runin_days = 4, n_perm = 3))
  printed <- capture.output(print(result))
  expect_true(any(grepl("Pooled AUC", printed, fixed = TRUE)))
  expect_true(any(grepl("Retrospective A0", printed, fixed = TRUE)))
})
