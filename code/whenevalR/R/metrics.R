#' Validate a binary outcome vector.
#' @keywords internal
.validate_y <- function(y) {
  if (!is.atomic(y) || is.matrix(y) || anyNA(y) || !all(y %in% c(0, 1))) {
    stop("y must contain only 0 and 1 with no missing values", call. = FALSE)
  }
  as.integer(y)
}

#' Validate a finite vector.
#' @keywords internal
.validate_finite <- function(x, name) {
  if (!is.numeric(x) || is.matrix(x) || anyNA(x) || any(!is.finite(x))) {
    stop(sprintf("%s must contain finite values", name), call. = FALSE)
  }
  as.numeric(x)
}

#' Validate binary outcomes, patient IDs, and optional scores.
#' @keywords internal
.validate_arrays <- function(patient, y, forecast = NULL) {
  yy <- .validate_y(y)
  if (!is.atomic(patient) || is.matrix(patient) || anyNA(patient)) {
    stop("patient must contain non-missing identifiers", call. = FALSE)
  }
  if (length(patient) != length(yy)) {
    stop("patient, y, and forecast must have equal length", call. = FALSE)
  }
  out <- list(patient = patient, y = yy)
  if (!is.null(forecast)) {
    pp <- .validate_finite(forecast, "forecast")
    if (length(pp) != length(yy)) stop("patient, y, and forecast must have equal length", call. = FALSE)
    out$forecast <- pp
  }
  out
}

#' Split row positions into patient groups in first-seen order.
#' @keywords internal
.groups <- function(patient) {
  values <- unique(patient)
  ids <- vapply(seq_along(values), function(i) as.character(values[[i]]), character(1))
  groups <- lapply(seq_along(values), function(i) which(patient == values[[i]]))
  names(groups) <- make.unique(ids, sep = "#")
  attr(groups, "labels") <- ids
  groups
}

#' Quantiles matching NumPy's default linear percentile convention.
#' @keywords internal
.percentiles <- function(x, probs = c(0.25, 0.75)) {
  if (!length(x)) return(rep(NaN, length(probs)))
  as.numeric(stats::quantile(x, probs = probs, names = FALSE, type = 7))
}

#' Compute tie-aware binary AUC.
#'
#' @param y Binary outcomes encoded as 0 and 1.
#' @param s Numeric scores; ties receive average ranks.
#' @return A scalar AUC, or `NaN` if either outcome class is absent.
#' @export
auc <- function(y, s) {
  yy <- .validate_y(y)
  score <- .validate_finite(s, "s")
  if (length(yy) != length(score)) stop("y and s must have equal length", call. = FALSE)
  n1 <- sum(yy == 1L)
  n0 <- sum(yy == 0L)
  if (!n1 || !n0) return(NaN)
  ranks <- base::rank(score, ties.method = "average")
  (sum(ranks[yy == 1L]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

#' Compute pooled and patient-specific AUCs.
#'
#' @param patient Patient identifier for each row.
#' @param y Binary outcomes encoded as 0 and 1.
#' @param forecast Numeric forecast score for each row.
#' @param weights Weighting for any patient-level average: `"patient"`, `"pairs"`, or `"days"`.
#' @return A list with pooled `value`, named `per_patient` AUCs, eligibility counts, and `weights`.
#' @export
pooled_auc <- function(patient, y, forecast, weights = "patient") {
  .auc_summary(patient, y, forecast, weights, pooled = TRUE)
}

#' Average AUC over eligible patients.
#'
#' @inheritParams pooled_auc
#' @return A list with the requested average `value`, named `per_patient` AUCs, and eligibility counts.
#' @export
within_patient_auc <- function(patient, y, forecast, weights = "patient") {
  .auc_summary(patient, y, forecast, weights, pooled = FALSE)
}

#' @keywords internal
.auc_summary <- function(patient, y, forecast, weights, pooled) {
  a <- .validate_arrays(patient, y, forecast)
  if (length(weights) != 1L || !weights %in% c("patient", "pairs", "days")) {
    stop("weights must be 'patient', 'pairs', or 'days'", call. = FALSE)
  }
  groups <- .groups(a$patient)
  per <- vapply(groups, function(ix) auc(a$y[ix], a$forecast[ix]), numeric(1))
  names(per) <- names(groups)
  valid <- is.finite(per)
  counts <- vapply(groups, function(ix) sum(a$y[ix] == 1L) * sum(a$y[ix] == 0L), numeric(1))
  day_counts <- lengths(groups)
  kept <- per[valid]
  if (pooled) {
    value <- auc(a$y, a$forecast)
  } else if (!length(kept)) {
    value <- NaN
  } else if (weights == "patient") {
    value <- mean(kept)
  } else {
    ww <- if (weights == "pairs") counts[valid] else day_counts[valid]
    value <- if (sum(ww) > 0) stats::weighted.mean(kept, ww) else NaN
  }
  list(value = unname(value), per_patient = kept, eligible = sum(valid),
       excluded = length(groups) - sum(valid), weights = weights)
}

#' Summarize eligible per-patient AUCs.
#'
#' @param patient Patient identifier for each row.
#' @param y Binary outcomes encoded as 0 and 1.
#' @param forecast Numeric forecast score for each row.
#' @return Patient count, excluded count, mean, median, quartiles, and share strictly above 0.5.
#' @export
per_patient_auc_summary <- function(patient, y, forecast) {
  s <- within_patient_auc(patient, y, forecast)
  vals <- unname(s$per_patient)
  if (!length(vals)) return(list(patients = 0L, excluded = s$excluded, mean = NaN,
    median = NaN, iqr = c(NaN, NaN), share_above_0_5 = NaN))
  list(patients = length(vals), excluded = s$excluded, mean = mean(vals),
       median = stats::median(vals), iqr = .percentiles(vals),
       share_above_0_5 = mean(vals > 0.5))
}

#' Compute AUC within each time point.
#'
#' @param data Data frame containing one row per patient-time observation.
#' @param time_col Name of the time-point column.
#' @param min_patients Minimum distinct patients required at a time point (default 5).
#' @param patient,y,forecast Column names for patient, outcome, and score.
#' @return Equal-weight and comparison-weighted AUC summaries, per-time AUCs, and used/excluded counts.
#' @export
per_time_point_auc <- function(data, time_col, min_patients = 5L,
                               patient = "patient", y = "y", forecast = "forecast") {
  columns <- c(patient, y, forecast, time_col)
  missing <- setdiff(columns, names(data))
  if (length(missing)) stop(sprintf("missing dataframe columns: %s", paste(missing, collapse = ", ")), call. = FALSE)
  if (length(min_patients) != 1L || is.na(min_patients) || min_patients < 1 || min_patients != as.integer(min_patients)) {
    stop("min_patients must be a positive integer", call. = FALSE)
  }
  a <- .validate_arrays(data[[patient]], data[[y]], data[[forecast]])
  tt <- data[[time_col]]
  if (!is.atomic(tt) || anyNA(tt)) stop(sprintf("%s contains missing values", time_col), call. = FALSE)
  times <- unique(tt)
  labels <- as.character(times)
  per <- pairs <- stats::setNames(numeric(), character())
  excluded <- character()
  for (i in seq_along(times)) {
    ix <- which(tt == times[[i]])
    n1 <- sum(a$y[ix] == 1L); n0 <- length(ix) - n1
    if (n1 && n0 && length(unique(a$patient[ix])) >= min_patients) {
      per[labels[[i]]] <- auc(a$y[ix], a$forecast[ix])
      pairs[labels[[i]]] <- n1 * n0
    } else excluded <- c(excluded, labels[[i]])
  }
  vals <- unname(per)
  if (length(vals)) {
    value <- mean(vals)
    comparison_weighted <- stats::weighted.mean(vals, unname(pairs))
    med <- stats::median(vals)
    iqr <- .percentiles(vals)
  } else {
    value <- comparison_weighted <- med <- NaN
    iqr <- c(NaN, NaN)
  }
  list(value = value, comparison_weighted = comparison_weighted,
       time_points_used = length(per), time_points_excluded = length(excluded),
       per_time = per, median = med, iqr = iqr, pairs = pairs,
       min_patients = as.integer(min_patients))
}

#' Compute retrospective pooled AUC of patient event-rate scores.
#'
#' @param patient Patient identifier for each row.
#' @param y Binary outcomes encoded as 0 and 1.
#' @param rates Optional row-aligned patient rates. If omitted, uses each patient's observed rate in `y`.
#' @return Pooled AUC of the patient-rate scores.
#' @export
rate_benchmark <- function(patient, y, rates = NULL) {
  a <- .validate_arrays(patient, y)
  if (is.null(rates)) {
    rr <- numeric(length(a$y))
    for (ix in .groups(a$patient)) rr[ix] <- mean(a$y[ix])
  } else {
    rr <- .validate_finite(rates, "rates")
    if (length(rr) != length(a$y)) stop("rates must have the same length as y", call. = FALSE)
  }
  auc(a$y, rr)
}

#' Estimate population A0 using the S4 moment correction.
#'
#' @param patient Patient identifier for each row.
#' @param y Binary outcomes encoded as 0 and 1.
#' @return Moment-corrected population A0 using day-weighted patient rates, or `NaN` if the overall rate is 0 or 1.
#' @export
population_a0_moment_corrected <- function(patient, y) {
  a <- .validate_arrays(patient, y)
  groups <- .groups(a$patient)
  if (!length(a$y)) return(NaN)
  sizes <- lengths(groups)
  rates <- vapply(groups, function(ix) mean(a$y[ix]), numeric(1))
  w <- sizes / sum(sizes)
  mu <- sum(w * rates)
  if (mu <= 0 || mu >= 1) return(NaN)
  observed_var <- sum(w * (rates - mu)^2)
  sampling_var <- sum(w * rates * (1 - rates) / pmax(1, sizes - 1))
  corrected_sd <- sqrt(max(0, observed_var - sampling_var))
  z <- corrected_sd / (sqrt(2) * mu * (1 - mu))
  stats::pnorm(z)
}

#' Return patient-specific run-in event rates.
#'
#' @param patient Patient identifier for each row.
#' @param y Binary outcomes encoded as 0 and 1.
#' @param time Numeric elapsed-time vector.
#' @param runin_days Rows before this cutoff form the run-in.
#' @param prior_weight Nonnegative pseudo-count centered at 0.5; defaults to 0 (the unsmoothed run-in rate).
#' @return A row-aligned vector repeating each patient's run-in rate; patients without run-in rows receive `NaN` when unsmoothed.
#' @export
runin_patient_rate <- function(patient, y, time, runin_days, prior_weight = 0) {
  a <- .validate_arrays(patient, y)
  tt <- .validate_finite(time, "time")
  if (length(tt) != length(a$y)) stop("time must have the same length as y", call. = FALSE)
  if (length(prior_weight) != 1L || !is.finite(prior_weight) || prior_weight < 0) stop("prior_weight must be finite and nonnegative", call. = FALSE)
  out <- rep(NaN, length(a$y))
  for (ix in .groups(a$patient)) {
    use <- ix[tt[ix] < runin_days]
    den <- length(use) + prior_weight
    rate <- if (den > 0) (sum(a$y[use]) + prior_weight * 0.5) / den else NaN
    out[ix] <- rate
  }
  out
}

#' Compute a strictly expanding patient event rate.
#'
#' @param patient Patient identifier for each row.
#' @param y Binary outcomes encoded as 0 and 1.
#' @param time Numeric elapsed-time vector.
#' @param prior_rate Optional external probability used only as a prior; omitted means no prior.
#' @param prior_weight Nonnegative prior pseudo-count.
#' @return Row-aligned rates using outcomes at strictly earlier times only. Equal-time and future rows are excluded.
#' @export
expanding_patient_rate <- function(patient, y, time, prior_rate = NULL, prior_weight = 1) {
  a <- .validate_arrays(patient, y)
  tt <- .validate_finite(time, "time")
  if (length(tt) != length(a$y)) stop("time must have the same length as y", call. = FALSE)
  if (length(prior_weight) != 1L || !is.finite(prior_weight) || prior_weight < 0) stop("prior_weight must be finite and nonnegative", call. = FALSE)
  if (is.null(prior_rate)) {
    prior <- rep(NaN, length(a$y)); weights <- rep(0, length(a$y))
  } else {
    if (length(prior_rate) == 1L) prior_rate <- rep(prior_rate, length(a$y))
    if (!is.numeric(prior_rate) || length(prior_rate) != length(a$y) || any(is.finite(prior_rate) & (prior_rate < 0 | prior_rate > 1))) {
      stop("prior_rate must be a probability or row-aligned probabilities in [0, 1]", call. = FALSE)
    }
    prior <- as.numeric(prior_rate)
    weights <- rep(prior_weight, length(a$y))
    weights[!is.finite(prior)] <- 0
  }
  out <- rep(NaN, length(a$y))
  for (ix in .groups(a$patient)) {
    for (j in ix) {
      earlier <- ix[tt[ix] < tt[[j]]]
      den <- length(earlier) + weights[[j]]
      if (den > 0) out[[j]] <- (sum(a$y[earlier]) + if (weights[[j]] > 0) prior[[j]] * weights[[j]] else 0) / den
    }
  }
  out
}

#' Compute pooled and equal-patient Brier score and log loss.
#'
#' @param patient Patient identifier for each row.
#' @param y Binary outcomes encoded as 0 and 1.
#' @param forecast Forecast probabilities in `[0, 1]`.
#' @return `pooled` day-weighted and `patient_mean` equal-patient Brier and log-loss summaries.
#' @export
brier_scores <- function(patient, y, forecast) {
  a <- .validate_arrays(patient, y, forecast)
  if (any(a$forecast < 0 | a$forecast > 1)) stop("forecast must contain probabilities in [0, 1]", call. = FALSE)
  losses <- function(yy, pp) {
    p <- pmin(1 - 1e-6, pmax(1e-6, pp))
    c(brier = mean((yy - p)^2), log_loss = mean(-(yy * log(p) + (1 - yy) * log1p(-p))))
  }
  pooled <- losses(a$y, a$forecast)
  by_patient <- vapply(.groups(a$patient), function(ix) losses(a$y[ix], a$forecast[ix]), numeric(2))
  list(pooled = list(brier = unname(pooled[["brier"]]), log_loss = unname(pooled[["log_loss"]])),
       patient_mean = list(brier = mean(by_patient["brier", ]), log_loss = mean(by_patient["log_loss", ])))
}

#' Compute Brier skill for one forecast and reference.
#'
#' @param y Binary outcomes encoded as 0 and 1.
#' @param forecast Forecast probabilities.
#' @param reference Reference probabilities.
#' @return One minus the forecast's summed Brier loss divided by the reference's summed Brier loss.
#' @export
brier_skill <- function(y, forecast, reference) {
  yy <- .validate_y(y)
  pp <- .validate_finite(forecast, "forecast")
  rr <- .validate_finite(reference, "reference")
  if (!(length(yy) == length(pp) && length(yy) == length(rr))) stop("inputs must have equal length", call. = FALSE)
  if (any(pp < 0 | pp > 1)) stop("forecast must contain probabilities in [0, 1]", call. = FALSE)
  if (any(rr < 0 | rr > 1)) stop("reference must contain probabilities in [0, 1]", call. = FALSE)
  den <- sum((yy - rr)^2)
  if (den <= 0) NaN else 1 - sum((yy - pp)^2) / den
}

#' Compute pooled and per-patient Brier skill.
#'
#' @param patient Patient identifier for each row.
#' @param y Binary outcomes encoded as 0 and 1.
#' @param forecast Forecast probabilities.
#' @param reference Row-aligned probabilities; missing values exclude those rows.
#' @return Pooled skill, equal-patient mean, positive share, per-patient skill, and eligibility counts.
#' @export
brier_skill_summary <- function(patient, y, forecast, reference) {
  a <- .validate_arrays(patient, y, forecast)
  rr <- as.numeric(reference)
  if (length(rr) != length(a$y)) stop("reference must have the same length as y", call. = FALSE)
  valid <- is.finite(rr)
  if (any(valid & (rr < 0 | rr > 1))) stop("reference must contain probabilities in [0, 1]", call. = FALSE)
  if (any(a$forecast < 0 | a$forecast > 1)) stop("forecast must contain probabilities in [0, 1]", call. = FALSE)
  per <- numeric()
  groups <- .groups(a$patient)
  for (nm in names(groups)) {
    ix <- groups[[nm]][valid[groups[[nm]]]]
    if (!length(ix)) next
    den <- sum((a$y[ix] - rr[ix])^2)
    if (den > 0) per[nm] <- 1 - sum((a$y[ix] - a$forecast[ix])^2) / den
  }
  used <- which(valid)
  den <- sum((a$y[used] - rr[used])^2)
  pooled <- if (length(used) && den > 0) 1 - sum((a$y[used] - a$forecast[used])^2) / den else NaN
  list(pooled = pooled, patient_mean = if (length(per)) mean(per) else NaN,
       positive_share = if (length(per)) mean(per > 0) else NaN, per_patient = per,
       eligible_patients = length(per), excluded_patients = length(groups) - length(per),
       scored_rows = length(used), excluded_rows = length(a$y) - length(used))
}

#' Summarize calibration-in-the-large by patient.
#'
#' @param patient Patient identifier for each row.
#' @param y Binary outcomes encoded as 0 and 1.
#' @param forecast Forecast probabilities.
#' @param tolerance Absolute difference considered within calibration tolerance (default 0.05).
#' @return Equal-patient mean, median, IQR and share within tolerance for observed minus mean forecast, with per-patient values.
#' @export
patient_calibration_in_the_large <- function(patient, y, forecast, tolerance = 0.05) {
  a <- .validate_arrays(patient, y, forecast)
  if (any(a$forecast < 0 | a$forecast > 1)) stop("forecast must contain probabilities in [0, 1]", call. = FALSE)
  diffs <- vapply(.groups(a$patient), function(ix) mean(a$y[ix]) - mean(a$forecast[ix]), numeric(1))
  if (!length(diffs)) return(list(mean = NaN, median = NaN, iqr = c(NaN, NaN), tolerance = tolerance,
    share_within_tolerance = NaN, patients = 0L, per_patient = diffs))
  list(mean = mean(diffs), median = stats::median(diffs), iqr = .percentiles(diffs),
       tolerance = tolerance, share_within_tolerance = mean(abs(diffs) <= tolerance + 1e-12),
       patients = length(diffs), per_patient = diffs)
}

#' Resolve caller-supplied within-group permutation maps.
#' @keywords internal
.permutation_maps <- function(permutations, n_perm, groups, scored_lengths = lengths(groups)) {
  if (is.null(permutations)) {
    return(lapply(seq_len(n_perm), function(b) lapply(scored_lengths, function(n) if (n) sample.int(n) else integer())))
  }
  if (!is.list(permutations) || length(permutations) != n_perm) stop("permutations must be a list with one draw per permutation", call. = FALSE)
  lapply(permutations, function(draw) {
    if (!is.list(draw) || length(draw) != length(groups)) stop("each permutation draw must provide one local index vector per patient", call. = FALSE)
    if (is.null(names(draw))) names(draw) <- names(groups)
    if (!all(names(groups) %in% names(draw))) stop("each permutation draw must name every patient group", call. = FALSE)
    out <- vector("list", length(groups)); names(out) <- names(groups)
    for (nm in names(groups)) {
      p <- as.integer(draw[[nm]])
      n <- scored_lengths[[nm]]
      if (length(p) != n || !identical(sort(p), seq_len(n))) stop("each supplied index vector must be a permutation of 1:n for that patient", call. = FALSE)
      out[[nm]] <- p
    }
    out
  })
}

#' Generate shuffles without changing the caller's random-number state.
#' @keywords internal
.seeded_permutation_maps <- function(n_perm, groups, scored_lengths = lengths(groups), seed = 0L) {
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_seed) old_seed <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  old_kind <- RNGkind()
  on.exit({
    do.call(RNGkind, as.list(old_kind))
    if (had_seed) assign(".Random.seed", old_seed, envir = .GlobalEnv)
    else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) rm(".Random.seed", envir = .GlobalEnv)
  }, add = TRUE)
  set.seed(seed)
  .permutation_maps(NULL, n_perm, groups, scored_lengths)
}

#' Test the pooled AUC under within-patient forecast permutations. This tests the pooled AUC only and does not estimate the per-patient shuffled baseline: shuffling a history-built forecast against fixed outcomes misses the finite-record effect. For per-patient shuffled baselines use evaluate() (outcomes shuffled, forecast and reference rebuilt).
#'
#' @param patient Patient identifier for each row.
#' @param y Binary outcomes encoded as 0 and 1.
#' @param forecast Forecast probabilities.
#' @param n_perm Number of permutations (at least one).
#' @param seed Seed used only when `permutations` is not supplied.
#' @param permutations Optional list of draws, each a named list of local 1-based row-order vectors by patient. Supplying these makes results reproducible across languages.
#' @return Observed pooled AUC, null mean and 2.5th/97.5th percentiles, one-sided add-one p-value, and per-patient summaries.
#' @export
pooled_auc_permutation_test <- function(patient, y, forecast, n_perm = 2000L, seed = 0L, permutations = NULL) {
  a <- .validate_arrays(patient, y, forecast)
  if (any(a$forecast < 0 | a$forecast > 1)) stop("forecast must contain probabilities in [0, 1]", call. = FALSE)
  if (length(n_perm) != 1L || is.na(n_perm) || n_perm < 1 || n_perm != as.integer(n_perm)) stop("n_perm must be at least 1", call. = FALSE)
  groups <- .groups(a$patient)
  maps <- if (is.null(permutations)) .seeded_permutation_maps(as.integer(n_perm), groups, seed = seed) else
    .permutation_maps(permutations, as.integer(n_perm), groups)
  observed <- auc(a$y, a$forecast)
  if (!is.finite(observed)) return(list(observed = NaN, null_mean = NaN, null_percentiles = c(NaN, NaN),
    one_sided_p_value = NaN, permutations = as.integer(n_perm), per_patient = list()))
  draws <- numeric(n_perm)
  patient_draws <- lapply(groups, function(ix) numeric(n_perm))
  for (b in seq_len(n_perm)) {
    pp <- a$forecast
    for (nm in names(groups)) {
      ix <- groups[[nm]]; pp[ix] <- a$forecast[ix[maps[[b]][[nm]]]]
      patient_draws[[nm]][b] <- auc(a$y[ix], pp[ix])
    }
    draws[[b]] <- auc(a$y, pp)
  }
  per <- list()
  for (nm in names(groups)) {
    ix <- groups[[nm]]; obs <- auc(a$y[ix], a$forecast[ix])
    if (is.finite(obs)) per[[nm]] <- list(observed = obs, reference_mean = mean(patient_draws[[nm]]),
      reference_percentiles = .percentiles(patient_draws[[nm]], c(0.025, 0.975)))
  }
  list(observed = observed, null_mean = mean(draws), null_percentiles = .percentiles(draws, c(0.025, 0.975)),
       one_sided_p_value = (sum(draws >= observed) + 1) / (n_perm + 1),
       permutations = as.integer(n_perm), per_patient = per)
}

#' Compare the observed per-patient AUC with its shuffled baseline from within-patient shuffles of the outcomes, recomputing the forecast from each shuffled history. The difference is the excess over the shuffled baseline, which has expectation zero when the patient's scored days are exchangeable (paper, Appendix A.5).
#'
#' @param patient Patient identifier for each row.
#' @param y Binary outcomes.
#' @param time Numeric elapsed time used to order each patient's record.
#' @param history_forecast_fn Function taking a patient's ordered outcome vector and returning one forecast per outcome.
#' @param n_perm Number of permutations (at least one).
#' @param seed Seed used only when `permutations` is not supplied.
#' @param runin_days Optional cutoff; earlier outcomes remain fixed and scored outcomes are permuted.
#' @param permutations Optional list of draws, each a named list of local 1-based permutations of each patient's scored rows.
#' @return Observed equal-patient AUC, permutation mean, difference, percentiles, and per-patient summaries.
#' @export
permutation_reference <- function(patient, y, time, history_forecast_fn, n_perm = 200L,
                                  seed = 0L, runin_days = NULL, permutations = NULL) {
  a <- .validate_arrays(patient, y)
  tt <- .validate_finite(time, "time")
  if (length(tt) != length(a$y)) stop("time must have the same length as y", call. = FALSE)
  if (length(n_perm) != 1L || is.na(n_perm) || n_perm < 1 || n_perm != as.integer(n_perm)) stop("n_perm must be at least 1", call. = FALSE)
  if (!is.function(history_forecast_fn)) stop("history_forecast_fn must be a function", call. = FALSE)
  orig_groups <- .groups(a$patient)
  groups <- lapply(orig_groups, function(ix) ix[order(tt[ix], seq_along(ix), method = "radix")])
  n_run <- vapply(groups, function(ix) if (is.null(runin_days)) 0L else sum(tt[ix] < runin_days), integer(1))
  names(n_run) <- names(groups)
  n_scored <- vapply(names(groups), function(nm) length(groups[[nm]]) - n_run[[nm]], integer(1))
  maps <- if (is.null(permutations)) .seeded_permutation_maps(as.integer(n_perm), groups, n_scored, seed) else
    .permutation_maps(permutations, as.integer(n_perm), groups, n_scored)
  score <- function(scored, collect_forecasts = FALSE) {
    vals <- numeric(); predictions <- list()
    for (nm in names(groups)) {
      ix <- groups[[nm]]; k <- n_run[[nm]]
      runin_ix <- if (k > 0L) seq_len(k) else integer()
      scored_ix <- if (k < length(ix)) seq.int(k + 1L, length(ix)) else integer()
      full <- c(a$y[ix][runin_ix], scored[[nm]])
      pred <- .validate_finite(history_forecast_fn(full), "history forecast")
      if (length(pred) != length(full)) stop("history_forecast_fn must return one forecast per outcome", call. = FALSE)
      if (any(pred < 0 | pred > 1)) stop("history forecast must contain probabilities in [0, 1]", call. = FALSE)
      val <- auc(scored[[nm]], pred[if (k < length(pred)) seq.int(k + 1L, length(pred)) else integer()])
      if (is.finite(val)) vals[nm] <- val
      if (collect_forecasts) predictions[[nm]] <- pred
    }
    if (collect_forecasts) attr(vals, "predictions") <- predictions
    vals
  }
  observed_data <- lapply(names(groups), function(nm) {
    ix <- groups[[nm]]; k <- n_run[[nm]]
    ix <- ix[if (k < length(ix)) seq.int(k + 1L, length(ix)) else integer()]
    a$y[ix]
  }); names(observed_data) <- names(groups)
  obs_by <- score(observed_data)
  patient_draws <- lapply(obs_by, function(x) numeric(n_perm))
  draws <- rep(NaN, n_perm)
  for (b in seq_len(n_perm)) {
    permuted <- lapply(names(groups), function(nm) observed_data[[nm]][maps[[b]][[nm]]])
    names(permuted) <- names(groups)
    values <- score(permuted)
    draws[[b]] <- if (length(values)) mean(values) else NaN
    for (nm in names(patient_draws)) patient_draws[[nm]][[b]] <- if (is.null(values[[nm]])) NaN else values[[nm]]
  }
  finite <- draws[is.finite(draws)]
  observed <- if (length(obs_by)) mean(obs_by) else NaN
  ref <- if (length(finite)) mean(finite) else NaN
  per <- lapply(names(obs_by), function(nm) {
    values <- patient_draws[[nm]]; values <- values[is.finite(values)]
    ref_mean <- if (length(values)) mean(values) else NaN
    list(observed = unname(obs_by[[nm]]), reference_mean = ref_mean,
         reference_percentiles = if (length(values)) .percentiles(values, c(0.025, 0.975)) else c(NaN, NaN),
         difference = if (is.finite(ref_mean)) unname(obs_by[[nm]] - ref_mean) else NaN)
  }); names(per) <- names(obs_by)
  list(observed = observed, reference_mean = ref, difference = observed - ref,
       reference_percentiles = if (length(finite)) .percentiles(finite, c(0.025, 0.975)) else c(NaN, NaN),
       permutations = as.integer(n_perm), per_patient = per)
}
