#' Evaluate repeated binary forecasts using the legacy report API.
#'
#' This pre-2.0.0 interface is retained for compatibility. Use [evaluate()] for
#' patient-wise permutation comparisons.
#'
#' @name evaluate_v1
#' @aliases evaluate_v1
#'
#' Rows before `runin_days` are used as history and excluded from scored metrics.
#' The expanding-rate Brier reference uses only strictly earlier outcomes and an
#' optional externally supplied prior. The retrospective rate benchmarks use
#' scored outcomes and are descriptive.
#'
#' @param data Long-format data frame.
#' @param patient,y,forecast Column names identifying patient, binary outcome, and forecast probability.
#' @param time Optional numeric elapsed-time column. Used for per-time-point AUC and patient-specific references.
#' @param runin_days Optional cutoff; rows with time below the cutoff are not scored.
#' @param prior_rate Optional external event rate to seed the expanding reference; no prior is used if omitted.
#' @param prior_weight Nonnegative pseudo-count for `prior_rate` (default 1).
#' @param n_perm Number of permutation draws (default 2000).
#' @param seed Seed used when permutation maps are generated internally.
#' @param history_forecast_fn Optional function rebuilding a history-based forecast from one ordered outcome vector.
#' @param permutations Optional common permutation maps for cross-language reproducibility; see [permutation_reference()].
#' @param return_references Include row-aligned reference-rate vectors in the result.
#' @param min_patients Minimum distinct patients required for a time-point AUC (default 5).
#' @return A named list of cohort, discrimination, A0, calibration, Brier, skill, and optional permutation results.
#' @examples
#' set.seed(1)
#' rate <- rep(runif(30, 0.05, 0.6), each = 40)
#' d <- data.frame(patient = rep(1:30, each = 40), time = rep(1:40, 30),
#'                 y = rbinom(1200, 1, rate), forecast = rate)
#' # A forecast that knows only each patient's rate: high pooled AUC, per-patient AUC 0.5.
#' evaluate(d, time = "time", n_perm = 200)
#' @export
evaluate_v1 <- function(data, patient = "patient", y = "y", forecast = "forecast",
                     time = NULL, runin_days = NULL, prior_rate = NULL,
                     prior_weight = 1, n_perm = 2000L, seed = 0L,
                     history_forecast_fn = NULL, permutations = NULL,
                     return_references = FALSE, min_patients = 5L) {
  warning("evaluate() v1 is deprecated; use the v2 evaluate() interface instead", call. = FALSE,
          immediate. = TRUE)
  cols <- c(patient, y, forecast, if (!is.null(time)) time)
  missing <- setdiff(cols, names(data))
  if (length(missing)) stop(sprintf("missing dataframe columns: %s", paste(missing, collapse = ", ")), call. = FALSE)
  if (!is.null(runin_days) && is.null(time)) stop("time is required when runin_days is given", call. = FALSE)
  if (!is.null(prior_rate) && is.null(time)) stop("time is required when prior_rate is given", call. = FALSE)
  if (length(prior_weight) != 1L || !is.finite(prior_weight) || prior_weight < 0) stop("prior_weight must be finite and nonnegative", call. = FALSE)
  if (!is.null(prior_rate) && (length(prior_rate) != 1L || !is.finite(prior_rate) || prior_rate < 0 || prior_rate > 1)) {
    stop("prior_rate must be a probability in [0, 1]", call. = FALSE)
  }
  a <- .validate_arrays(data[[patient]], data[[y]], data[[forecast]])
  if (any(a$forecast < 0 | a$forecast > 1)) stop("forecast must contain probabilities in [0, 1]", call. = FALSE)
  tt <- NULL
  if (!is.null(time)) {
    tt <- .validate_finite(data[[time]], time)
    if (length(tt) != length(a$y)) stop("time must have the same length as y", call. = FALSE)
  }
  score_mask <- if (is.null(runin_days)) rep(TRUE, length(a$y)) else tt >= runin_days
  if (!any(score_mask)) stop("no scored rows remain after the run-in period", call. = FALSE)
  pid_s <- a$patient[score_mask]; yy_s <- a$y[score_mask]; pp_s <- a$forecast[score_mask]
  tt_s <- if (is.null(tt)) NULL else tt[score_mask]
  groups <- .groups(pid_s)

  pooled <- pooled_auc(pid_s, yy_s, pp_s)
  within <- within_patient_auc(pid_s, yy_s, pp_s)
  patient_summary <- per_patient_auc_summary(pid_s, yy_s, pp_s)
  patient_mean_forecast <- numeric(length(yy_s))
  for (ix in groups) patient_mean_forecast[ix] <- mean(pp_s[ix])
  patient_mean_auc <- auc(yy_s, patient_mean_forecast)
  denom <- pooled$value - 0.5
  patient_mean_share <- if (is.finite(denom) && abs(denom) > 1e-15) (patient_mean_auc - 0.5) / denom else NaN

  patient_rates <- vapply(groups, function(ix) mean(yy_s[ix]), numeric(1))
  n_events <- sum(yy_s == 1L); n_nonevents <- length(yy_s) - n_events
  within_pairs <- sum(vapply(groups, function(ix) sum(yy_s[ix] == 1L) * sum(yy_s[ix] == 0L), numeric(1)))
  cohort <- list(patients = length(groups), days = length(yy_s), events = n_events,
    overall_rate = mean(yy_s), patient_rate_mean = mean(patient_rates),
    patient_rate_sd = if (length(patient_rates) > 1L) stats::sd(patient_rates) else NaN,
    within_patient_pair_share = if (n_events && n_nonevents) within_pairs / (n_events * n_nonevents) else NaN,
    rows_excluded_runin = sum(!score_mask), runin_days = runin_days)

  auc_result <- list(pooled = pooled, within_patient = within, per_patient_summary = patient_summary,
    per_time_point = NULL, A0 = rate_benchmark(pid_s, yy_s),
    A0_population = population_a0_moment_corrected(pid_s, yy_s),
    patient_mean = list(pooled_auc = patient_mean_auc,
      share_of_pooled_auc_above_0_5 = patient_mean_share))
  if (!is.null(time)) {
    scored_data <- data[score_mask, , drop = FALSE]
    auc_result$per_time_point <- per_time_point_auc(scored_data, time,
      min_patients = min_patients, patient = patient, y = y, forecast = forecast)
  }
  brier <- brier_scores(pid_s, yy_s, pp_s)

  references <- list()
  if (!is.null(time)) {
    if (!is.null(runin_days)) references$runin_patient_rate <- runin_patient_rate(a$patient, a$y, tt, runin_days)[score_mask]
    references$expanding_patient_rate <- expanding_patient_rate(a$patient, a$y, tt,
      prior_rate = prior_rate, prior_weight = prior_weight)[score_mask]
  }
  skill <- lapply(references, function(ref) brier_skill_summary(pid_s, yy_s, pp_s, ref))
  cil <- patient_calibration_in_the_large(pid_s, yy_s, pp_s)
  result <- list(cohort = cohort, auc = auc_result, brier = brier, skill = skill,
    calibration_in_the_large = cil,
    excluded_patients = list(pooled_auc = pooled$excluded, within_auc = within$excluded))

  if (n_perm < 1L || n_perm != as.integer(n_perm)) stop("n_perm must be at least 1", call. = FALSE)
  result$pooled_auc_permutation <- pooled_auc_permutation_test(pid_s, yy_s, pp_s,
    n_perm = n_perm, seed = seed, permutations = permutations)
  result$permutation_reference <- NULL
  if (!is.null(history_forecast_fn) && !is.null(time) && !is.null(runin_days)) {
    result$permutation_reference <- permutation_reference(a$patient, a$y, tt,
      history_forecast_fn, n_perm = n_perm, seed = seed, runin_days = runin_days,
      permutations = permutations)
  }
  if (return_references) result$references <- references
  class(result) <- c("withinpatient_evaluation", "list")
  result
}

#' Print an evaluation as a compact summary table.
#'
#' @param x An object returned by [evaluate()].
#' @param ... Ignored.
#' @return The input, invisibly.
#' @export
print.withinpatient_evaluation <- function(x, ...) {
  if (!is.null(x$metrics)) {
    fmt <- function(value) {
      if (length(value) != 1L || is.na(value) || !is.finite(value)) return(as.character(value))
      formatC(value, digits = 3, format = "f")
    }
    cat("| Metric | Mean observed | Mean chance | Median excess [IQR] | Excess > 0 | p < 0.05 | Patients | Pooled | Usual per-patient mean |\n")
    cat("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for (name in names(x$metrics)) {
      z <- x$metrics[[name]]
      s <- z$summary; comparison <- z$comparison
      if (identical(name, "calibration_slope") && !is.null(comparison$conditional_within_slope_chance)) {
        row <- c("calibration_slope (conditional, all patients)", fmt(comparison$conditional_within_slope),
          fmt(comparison$conditional_within_slope_chance),
          paste0(fmt(comparison$conditional_within_slope_excess), " (p = ", fmt(comparison$conditional_within_slope_p), ")"),
          "", "", as.character(s$n_eligible), fmt(comparison$pooled), "")
        cat(paste0("| ", paste(row, collapse = " | "), " |\n"))
        next
      }
      row <- c(name, fmt(s$mean_observed), fmt(s$mean_chance_level),
        paste0(fmt(s$median_excess), " [", fmt(s$iqr_excess[[1L]]), ", ", fmt(s$iqr_excess[[2L]]), "]"),
        if (is.finite(s$share_excess_positive)) sprintf("%.0f%%", 100 * s$share_excess_positive) else "nan",
        if (is.finite(s$share_p_below_0_05)) sprintf("%.0f%%", 100 * s$share_p_below_0_05) else "nan",
        as.character(s$n_eligible), fmt(comparison$pooled), fmt(comparison$usual_per_patient_mean))
      cat(paste0("| ", paste(row, collapse = " | "), " |\n"))
    }
    st <- x$settings
    cat(sprintf("\nThreshold %.3f; %s within-patient shuffles per patient (seed %s); rebuild: %s; yardstick: %s.\n",
      st$threshold, st$n_perm, st$seed, if (is.null(st$rebuild)) "None" else st$rebuild, st$yardstick))
    return(invisible(x))
  }
  rows <- list(
    c("Patients", x$cohort$patients), c("Scored observations", x$cohort$days),
    c("Scored events", x$cohort$events), c("Pooled AUC", x$auc$pooled$value),
    c("Per-patient AUC (equal weight)", x$auc$within_patient$value),
    c("Per-patient AUC median", x$auc$per_patient_summary$median),
    c("Per-patient AUC IQR", paste(format(x$auc$per_patient_summary$iqr, digits = 4), collapse = " to ")),
    c("Patients with AUC above 0.5", x$auc$per_patient_summary$share_above_0_5),
    c("Per-time-point AUC", if (is.null(x$auc$per_time_point)) "not computed" else x$auc$per_time_point$value),
    c("Retrospective A0", x$auc$A0), c("Population A0 (moment corrected)", x$auc$A0_population),
    c("Calibration-in-the-large mean", x$calibration_in_the_large$mean),
    c("Patients within calibration tolerance", x$calibration_in_the_large$share_within_tolerance),
    c("Pooled Brier score", x$brier$pooled$brier), c("Patient-mean Brier score", x$brier$patient_mean$brier)
  )
  for (nm in names(x$skill)) rows[[length(rows) + 1L]] <- c(paste("Brier skill vs", nm), x$skill[[nm]]$pooled)
  rows[[length(rows) + 1L]] <- c("Within-patient pooled-AUC permutation p", x$pooled_auc_permutation$one_sided_p_value)
  if (!is.null(x$permutation_reference)) rows[[length(rows) + 1L]] <- c("History permutation AUC difference", x$permutation_reference$difference)
  tab <- do.call(rbind, rows)
  values <- vapply(tab[, 2], function(value) {
    num <- suppressWarnings(as.numeric(value))
    if (is.na(num)) as.character(value) else if (is.finite(num)) formatC(num, digits = 3, format = "f") else "NA"
  }, character(1))
  cat("| Quantity | Value |\n|---|---:|\n")
  for (i in seq_len(nrow(tab))) cat(sprintf("| %s | %s |\n", tab[i, 1], values[[i]]))
  invisible(x)
}
