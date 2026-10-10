# Helpers and the v2 evaluator. The formulas follow code/withinpatient's v2 API.

.wp_summary <- function(values, mean_override = NULL, n_override = NULL) {
  values <- as.numeric(values); values <- values[is.finite(values)]
  if (length(values)) {
    q <- .percentiles(values)
    mean_value <- if (is.null(mean_override)) mean(values) else mean_override
    median_value <- stats::median(values)
  } else {
    q <- c(NaN, NaN)
    mean_value <- if (is.null(mean_override)) NaN else mean_override
    median_value <- NaN
  }
  list(mean = as.numeric(mean_value), median = as.numeric(median_value),
       iqr = q, n = if (is.null(n_override)) length(values) else as.integer(n_override))
}

.wp_groups <- function(patient) {
  vals <- unique(patient)
  groups <- lapply(vals, function(v) which(patient == v))
  names(groups) <- make.unique(as.character(vals), sep = "#")
  attr(groups, "labels") <- as.character(vals)
  groups
}

.wp_ap <- function(y, p) {
  if (!length(y) || !any(y == 1L)) return(NaN)
  ord <- order(-p, seq_along(p), method = "radix")
  yy <- y[ord]; pp <- p[ord]
  starts <- c(1L, which(pp[-1L] != pp[-length(pp)]) + 1L)
  ends <- c(starts[-1L] - 1L, length(y))
  cum <- cumsum(yy); positives <- sum(yy); out <- 0
  for (i in seq_along(starts)) {
    hi <- ends[[i]]; before <- if (starts[[i]] > 1L) cum[starts[[i]] - 1L] else 0
    block <- cum[[hi]] - before
    out <- out + block / positives * cum[[hi]] / hi
  }
  out
}

.wp_net_benefit <- function(y, p, threshold) {
  alarm <- p >= threshold
  (sum(alarm & y == 1L) / length(y) - sum(alarm & y == 0L) / length(y) *
     threshold / (1 - threshold))
}

.wp_loglik_terms <- function(y, x, groups, beta) {
  ll <- score <- info <- 0
  for (ix in groups) {
    yy <- as.integer(y[ix]); xx <- as.numeric(x[ix]); n <- length(yy); m <- sum(yy)
    if (m == 0L || m == n) next
    if (m > n %/% 2L) { yy <- 1L - yy; xx <- -xx; m <- n - m }
    xx <- xx - mean(xx); event_x <- sum(yy * xx)
    logits <- beta * xx
    if (any(!is.finite(logits))) return(c(NaN, NaN, NaN))
    shift <- max(logits); weights <- exp(logits - shift)
    e <- d1 <- d2 <- numeric(m + 1L); e[[1L]] <- 1
    log_scale <- 0
    for (i in seq_len(n)) {
      degree <- min(i, m)
      if (degree > 0L) for (k in seq.int(degree, 1L)) {
        j <- k + 1L; prev <- k
        d2[[j]] <- d2[[j]] + weights[[i]] * (d2[[prev]] + 2 * xx[[i]] * d1[[prev]] + xx[[i]]^2 * e[[prev]])
        d1[[j]] <- d1[[j]] + weights[[i]] * (d1[[prev]] + xx[[i]] * e[[prev]])
        e[[j]] <- e[[j]] + weights[[i]] * e[[prev]]
      }
      scale <- max(e[seq_len(degree + 1L)])
      if (!is.finite(scale) || scale <= 0) return(c(NaN, NaN, NaN))
      e[seq_len(degree + 1L)] <- e[seq_len(degree + 1L)] / scale
      d1[seq_len(degree + 1L)] <- d1[seq_len(degree + 1L)] / scale
      d2[seq_len(degree + 1L)] <- d2[seq_len(degree + 1L)] / scale
      log_scale <- log_scale + log(scale)
    }
    denominator <- e[[m + 1L]]
    if (!is.finite(denominator) || denominator <= 0) return(c(NaN, NaN, NaN))
    mean_sum <- d1[[m + 1L]] / denominator
    variance <- max(d2[[m + 1L]] / denominator - mean_sum^2, 0)
    ll <- ll + beta * event_x - (m * shift + log_scale + log(denominator))
    score <- score + event_x - mean_sum
    info <- info + variance
  }
  c(ll, score, info)
}

.wp_conditional_fit <- function(y, x, groups) {
  beta <- 0; terms <- .wp_loglik_terms(y, x, groups, beta)
  if (any(!is.finite(terms))) return(NULL)
  for (iter in seq_len(100L)) {
    ll <- terms[[1L]]; gradient <- terms[[2L]]; information <- terms[[3L]]
    if (!is.finite(information) || information <= 0) return(NULL)
    step <- gradient / information
    if (!is.finite(step)) return(NULL)
    if (abs(step) <= 1e-9 * (1 + abs(beta))) return(c(beta, 1 / sqrt(information)))
    trial_step <- step; accepted <- FALSE
    for (line in seq_len(60L)) {
      trial_beta <- beta + trial_step
      if (is.finite(trial_beta)) {
        trial <- .wp_loglik_terms(y, x, groups, trial_beta)
        if (is.finite(trial[[1L]]) && trial[[1L]] >= ll - 1e-12) { accepted <- TRUE; break }
      }
      trial_step <- trial_step / 2
    }
    if (!accepted) return(NULL)
    beta <- trial_beta; terms <- trial
  }
  NULL
}

#' Exact conditional within-patient calibration slope.
#' @param patient Patient ID per observation.
#' @param y Binary outcome.
#' @param forecast Forecast probability.
#' @return Slope, standard error, number of eligible patients, and fit status.
#' @export
conditional_within_slope <- function(patient, y, forecast) {
  a <- .validate_arrays(patient, y, forecast)
  if (any(a$forecast < 0 | a$forecast > 1)) stop("forecast must contain probabilities in [0, 1]", call. = FALSE)
  x <- qlogis(pmin(1 - 1e-12, pmax(1e-12, a$forecast)))
  candidates <- Filter(function(ix) 0 < sum(a$y[ix]) && sum(a$y[ix]) < length(ix), .wp_groups(a$patient))
  out <- list(slope = NaN, se = NaN, patients = length(candidates), estimable = FALSE, reason = "")
  if (!length(candidates)) out$reason <- "no patient has both outcomes"
  else if (!any(vapply(candidates, function(ix) diff(range(x[ix])) > 1e-12, logical(1)))) {
    out$reason <- "forecast logit does not vary within any patient with both outcomes"
  } else {
    keep <- unlist(candidates, use.names = FALSE)
    local_groups <- vector("list", length(candidates)); start <- 0L
    for (i in seq_along(candidates)) {
      ni <- length(candidates[[i]]); local_groups[[i]] <- seq.int(start + 1L, start + ni); start <- start + ni
    }
    fit <- .wp_conditional_fit(a$y[keep], x[keep], local_groups)
    if (is.null(fit)) out$reason <- "exact conditional logit fit failed to converge"
    else if (all(is.finite(fit)) && fit[[2L]] > 0) {
      out$slope <- fit[[1L]]; out$se <- fit[[2L]]; out$estimable <- TRUE
    } else out$reason <- "exact conditional logit returned a non-finite estimate"
  }
  if (!out$estimable) warning(paste("conditional within-patient calibration slope not estimable:", out$reason), call. = FALSE)
  out
}

.wp_slope <- function(y, p) {
  suppressWarnings(conditional_within_slope(rep.int(1L, length(y)), y, p)$slope)
}

.wp_select_metrics <- function(metrics) {
  builtins <- c("auc", "average_precision", "sensitivity", "specificity", "ppv", "npv", "net_benefit", "brier_skill", "calibration_slope")
  aliases <- c(ap = "average_precision", stats::setNames(builtins, builtins))
  custom <- list(); names_out <- character()
  if (is.null(metrics)) return(list(custom = custom, names = builtins))
  if (is.function(metrics)) metrics <- setNames(list(metrics), "custom_metric")
  if (is.list(metrics) && !is.null(names(metrics)) && all(nzchar(names(metrics)))) {
    for (nm in names(metrics)) {
      if (is.function(metrics[[nm]])) {
        if (tolower(nm) %in% names(aliases)) stop(sprintf("custom metric name %s conflicts with a built-in", nm), call. = FALSE)
        custom[[nm]] <- metrics[[nm]]; names_out <- c(names_out, nm)
      } else names_out <- c(names_out, as.character(metrics[[nm]]))
    }
  } else {
    for (item in metrics) {
      if (is.function(item)) {
        nm <- deparse(substitute(item)); if (tolower(nm) %in% names(aliases)) nm <- paste0(nm, "_custom")
        custom[[nm]] <- item; names_out <- c(names_out, nm)
      } else names_out <- c(names_out, as.character(item))
    }
  }
  custom_names <- names(custom)
  names_out <- vapply(names_out, function(nm) {
    if (nm %in% custom_names) return(nm)
    key <- tolower(gsub("[ -]", "_", trimws(nm)))
    if (!key %in% names(aliases)) stop(sprintf("unknown built-in metric %s", nm), call. = FALSE)
    unname(aliases[[key]])
  }, character(1))
  # Restore custom labels (the names supplied by the caller are not aliases).
  if (anyDuplicated(names_out)) stop("metric names must be unique", call. = FALSE)
  list(custom = custom, names = names_out)
}

.wp_metric_result <- function(pooled, values, mean_override = NULL, n_override = NULL, ...) {
  list(pooled = as.numeric(pooled), summary = .wp_summary(values, mean_override, n_override),
       per_patient = values, ...)
}

.wp_pooled_slope <- function(y, p) {
  if (length(unique(y)) < 2L || length(unique(p)) < 2L) return(NaN)
  fit <- tryCatch(suppressWarnings(stats::glm(y ~ qlogis(pmin(1 - 1e-12, pmax(1e-12, p))),
    family = stats::binomial())), error = function(e) NULL)
  if (is.null(fit) || length(stats::coef(fit)) < 2L) NaN else unname(stats::coef(fit)[[2L]])
}

#' Compute the paper's nine pooled and within-patient metrics.
#' @param patient Patient ID per observation.
#' @param y Binary outcome.
#' @param forecast Forecast probability.
#' @param threshold Classification threshold, strictly between zero and one.
#' @param yardstick Optional row-aligned reference probabilities; defaults to each patient's event rate.
#' @return Named metric results with pooled, per-patient, and summary values.
#' @export
nine_metrics <- function(patient, y, forecast, threshold, yardstick = NULL) {
  if (length(threshold) != 1L || !is.finite(threshold) || threshold <= 0 || threshold >= 1) stop("threshold must be strictly between 0 and 1", call. = FALSE)
  a <- .validate_arrays(patient, y, forecast); yy <- a$y; pp <- a$forecast
  if (any(pp < 0 | pp > 1)) stop("forecast must contain probabilities in [0, 1]", call. = FALSE)
  groups <- .wp_groups(a$patient); labels <- attr(groups, "labels")
  ref <- numeric(length(yy))
  if (is.null(yardstick)) for (ix in groups) ref[ix] <- mean(yy[ix])
  else {
    ref <- .validate_finite(yardstick, "yardstick")
    if (length(ref) != length(yy)) stop("yardstick must have the same length as y", call. = FALSE)
  }
  if (any(ref < 0 | ref > 1)) stop("yardstick must contain probabilities in [0, 1]", call. = FALSE)
  both <- vapply(groups, function(ix) any(yy[ix] == 1L) && any(yy[ix] == 0L), logical(1))
  auc_per <- ap_per <- numeric()
  for (i in which(both)) { ix <- groups[[i]]; auc_per[labels[[i]]] <- auc(yy[ix], pp[ix]); ap_per[labels[[i]]] <- .wp_ap(yy[ix], pp[ix]) }
  n <- length(yy); alarm <- pp >= threshold; events <- sum(yy); nonevents <- n - events
  tp <- sum(alarm & yy == 1L); fp <- sum(alarm & yy == 0L); tn <- nonevents - fp; quiet <- n - sum(alarm)
  rate <- if (n) mean(yy) else NaN
  sens_per <- spec_per <- ppv_per <- npv_per <- nb_per <- brier_per <- numeric()
  model_loss <- ref_loss <- numeric(length(groups))
  for (i in seq_along(groups)) {
    ix <- groups[[i]]; yi <- yy[ix]; pi <- pp[ix]; ri <- ref[ix]; ai <- pi >= threshold; r <- mean(yi)
    if (any(yi == 1L) && any(yi == 0L)) {
      sens_per[labels[[i]]] <- mean(ai[yi == 1L]); spec_per[labels[[i]]] <- mean(!ai[yi == 0L])
    }
    if (any(ai)) ppv_per[labels[[i]]] <- mean(yi[ai]) - r
    if (any(!ai)) npv_per[labels[[i]]] <- mean(yi[!ai] == 0L) - (1 - r)
    policy <- as.numeric(ri >= threshold)
    nb_per[labels[[i]]] <- .wp_net_benefit(yi, pi, threshold) - .wp_net_benefit(yi, policy, threshold)
    model_loss[[i]] <- mean((yi - pi)^2); ref_loss[[i]] <- mean((yi - ri)^2)
    if (ref_loss[[i]] > 0) brier_per[labels[[i]]] <- 1 - model_loss[[i]] / ref_loss[[i]]
  }
  pooled_nb <- if (n) .wp_net_benefit(yy, pp, threshold) else NaN
  pooled_ref_loss <- if (n) mean((yy - ref)^2) else NaN
  pooled_model_loss <- if (n) mean((yy - pp)^2) else NaN
  equal_ref <- if (length(ref_loss)) mean(ref_loss) else NaN
  equal_skill <- if (is.finite(equal_ref) && equal_ref > 0) 1 - mean(model_loss) / equal_ref else NaN
  conditional <- suppressWarnings(conditional_within_slope(a$patient, yy, pp))
  slope_per <- if (is.finite(conditional$slope)) stats::setNames(rep(conditional$slope, sum(both)), labels[both]) else numeric()
  list(
    auc = .wp_metric_result(auc(yy, pp), auc_per),
    average_precision = .wp_metric_result(.wp_ap(yy, pp), ap_per),
    sensitivity = .wp_metric_result(if (events) tp / events else NaN, sens_per),
    specificity = .wp_metric_result(if (nonevents) tn / nonevents else NaN, spec_per),
    ppv = .wp_metric_result(if (sum(alarm)) tp / sum(alarm) else NaN, ppv_per,
      per_patient_definition = "PPV minus own event rate"),
    npv = .wp_metric_result(if (quiet) tn / quiet else NaN, npv_per,
      per_patient_definition = "NPV minus own non-event rate"),
    net_benefit = .wp_metric_result(pooled_nb, nb_per,
      pooled_treat_all = if (n) rate - (1 - rate) * threshold / (1 - threshold) else NaN,
      per_patient_definition = "model NB minus own-rate policy NB on observed outcomes"),
    brier_skill = .wp_metric_result(if (pooled_ref_loss > 0) 1 - pooled_model_loss / pooled_ref_loss else NaN,
      brier_per, mean_override = equal_skill,
      definition = "1 - mean(patient Brier model) / mean(patient Brier yardstick)"),
    calibration_slope = .wp_metric_result(.wp_pooled_slope(yy, pp), slope_per,
      mean_override = conditional$slope, n_override = conditional$patients,
      conditional_fit = conditional)
  )
}

#' Compute the usual patient-wise metrics against population-level chance values.
#' @inheritParams nine_metrics
#' @param population_rate Population event rate used as a reference.
#' @return Named metric results with per-patient values, summaries, and chance levels.
#' @export
naive_metrics <- function(patient, y, forecast, threshold, population_rate) {
  if (length(threshold) != 1L || !is.finite(threshold) || threshold <= 0 || threshold >= 1) stop("threshold must be strictly between 0 and 1", call. = FALSE)
  if (length(population_rate) != 1L || !is.finite(population_rate) || population_rate < 0 || population_rate > 1) stop("population_rate must be a probability in [0, 1]", call. = FALSE)
  a <- .validate_arrays(patient, y, forecast); yy <- a$y; pp <- a$forecast
  if (any(pp < 0 | pp > 1)) stop("forecast must contain probabilities in [0, 1]", call. = FALSE)
  groups <- .wp_groups(a$patient); labels <- attr(groups, "labels"); alarm <- pp >= threshold
  per <- replicate(8L, numeric(), simplify = FALSE)
  names(per) <- c("auc", "average_precision", "sensitivity", "specificity", "ppv", "npv", "net_benefit", "brier_skill")
  model_loss <- ref_loss <- numeric(length(groups))
  for (i in seq_along(groups)) {
    ix <- groups[[i]]; yi <- yy[ix]; pi <- pp[ix]; ai <- alarm[ix]; key <- labels[[i]]
    if (any(yi == 1L) && any(yi == 0L)) per$auc[key] <- auc(yi, pi)
    if (any(yi == 1L)) per$average_precision[key] <- .wp_ap(yi, pi)
    if (any(yi == 1L)) per$sensitivity[key] <- mean(ai[yi == 1L])
    if (any(yi == 0L)) per$specificity[key] <- mean(!ai[yi == 0L])
    if (any(ai)) per$ppv[key] <- mean(yi[ai])
    if (any(!ai)) per$npv[key] <- mean(yi[!ai] == 0L)
    per$net_benefit[key] <- .wp_net_benefit(yi, pi, threshold)
    model_loss[[i]] <- mean((yi - pi)^2); ref_loss[[i]] <- mean((yi - population_rate)^2)
    if (ref_loss[[i]] > 0) per$brier_skill[key] <- 1 - model_loss[[i]] / ref_loss[[i]]
  }
  conditional <- suppressWarnings(conditional_within_slope(a$patient, yy, pp)); slope <- conditional$slope
  both <- vapply(groups, function(ix) any(yy[ix] == 1L) && any(yy[ix] == 0L), logical(1))
  slope_per <- if (is.finite(slope)) stats::setNames(rep(slope, sum(both)), labels[both]) else numeric()
  rate_skill <- if (mean(ref_loss) > 0) 1 - mean(model_loss) / mean(ref_loss) else NaN
  entry <- function(values, chance, mean_override = NULL) list(per_patient = values,
    summary = .wp_summary(values, mean_override), chance_level = chance)
  result <- list(auc = entry(per$auc, 0.5), average_precision = entry(per$average_precision, population_rate),
    sensitivity = entry(per$sensitivity, population_rate), specificity = entry(per$specificity, 1 - population_rate),
    ppv = entry(per$ppv, population_rate), npv = entry(per$npv, 1 - population_rate),
    net_benefit = entry(per$net_benefit, 0),
    brier_skill = entry(per$brier_skill, 0, rate_skill),
    calibration_slope = if (is.finite(slope)) list(per_patient = slope_per,
      summary = .wp_summary(slope_per, slope, conditional$patients),
      chance_level = "n/a (no within-patient forecast variation)", conditional_fit = conditional) else
      list(per_patient = numeric(), summary = list(mean = "n/a", median = "n/a", iqr = c("n/a", "n/a"), n = 0L),
        chance_level = "n/a (no within-patient forecast variation)", conditional_fit = conditional),
    population_rate = population_rate)
  sens_mean <- mean(result$sensitivity$summary$mean, na.rm = TRUE)
  spec_mean <- mean(result$specificity$summary$mean, na.rm = TRUE)
  result$naive_youden <- if (is.finite(sens_mean) && is.finite(spec_mean)) sens_mean + spec_mean - 1 else NaN
  result
}

.wp_spec <- function(value, label) {
  if (is.null(value)) return(NULL)
  if (is.function(value)) return(list(name = "callable", fn = value))
  if (is.character(value) && length(value) == 1L) spec <- list(name = value)
  else if (is.list(value)) {
    spec <- value
    if (is.null(spec$name)) spec$name <- spec$method %||% spec$type
  } else stop(sprintf("%s must be a built-in name, configuration list, or function", label), call. = FALSE)
  nm <- tolower(gsub("-", "_", trimws(as.character(spec$name %||% ""))))
  if (!nm %in% c("running_rate", "window_count", "lagged_outcomes")) stop(sprintf("%s must name running_rate, window_count, or lagged_outcomes", label), call. = FALSE)
  spec$name <- nm; spec
}

`%||%` <- function(x, y) if (is.null(x)) y else x

.wp_running_rate <- function(y, stops, spec) {
  weight <- as.numeric(spec$prior_weight %||% spec$pseudo_count %||% 1)
  prior <- spec$prior_rate %||% 0.5
  if (length(weight) != 1L || !is.finite(weight) || weight < 0) stop("running_rate prior_weight must be finite and nonnegative", call. = FALSE)
  if (is.null(prior)) { if (weight) stop("running_rate prior_rate is required when prior_weight is positive", call. = FALSE); prior <- 0 }
  prior <- as.numeric(prior)
  if (length(prior) != 1L || !is.finite(prior) || prior < 0 || prior > 1) stop("running_rate prior_rate must be a probability in [0, 1]", call. = FALSE)
  csum <- c(0, cumsum(as.numeric(y)))
  den <- stops + weight; out <- rep(NaN, length(y)); valid <- den > 0
  out[valid] <- (csum[stops[valid] + 1L] + prior * weight) / den[valid]
  out
}

.wp_window_count <- function(y, stops, spec) {
  if (is.null(spec$w) || length(spec$w) != 1L || !is.finite(spec$w) || spec$w < 1 || spec$w != as.integer(spec$w)) stop("window_count requires a positive integer w", call. = FALSE)
  width <- as.integer(spec$w); csum <- c(0, cumsum(as.numeric(y)))
  starts <- pmax(stops - width, 0); events <- csum[stops + 1L] - csum[starts + 1L]
  counts <- pmin(stops, width); as_rate <- spec$as_rate %||% if (identical(spec$mode, "count")) FALSE else TRUE
  if (is.character(as_rate)) as_rate <- tolower(as_rate) != "count"
  if (!isTRUE(as_rate)) return(events)
  ifelse(counts > 0, events / counts, 0)
}

.wp_call_y <- function(fn, y, scored, frame = NULL) {
  # f(y), f(y, scored) or f(y, scored, rows), by the number of arguments (as in Python).
  f <- formals(fn)
  n_args <- if (is.null(f) || "..." %in% names(f)) 3L else length(f)
  if (n_args >= 3L) return(fn(y, scored, frame))
  if (n_args == 2L) fn(y, scored) else fn(y)
}

.wp_as_rows <- function(raw, n, scored, label) {
  if (is.matrix(raw) && ncol(raw) == 2L) raw <- raw[, 2L]
  raw <- as.numeric(raw); count <- sum(scored)
  if (length(raw) == n) return(raw)
  if (length(raw) == count) { out <- rep(NaN, n); out[scored] <- raw; return(out) }
  stop(sprintf("%s must return %d rows or %d scored rows", label, n, count), call. = FALSE)
}

.wp_history <- function(spec, frame, y, scored, stops, outcome, forecast) {
  if (spec$name == "running_rate") return(.wp_running_rate(y, stops, spec))
  if (spec$name == "window_count") return(.wp_window_count(y, stops, spec))
  if (spec$name == "callable") return(.wp_as_rows(.wp_call_y(spec$fn, y, scored, frame), length(y), scored, "rebuild function"))
  if (spec$name == "lagged_outcomes") {
    lags <- spec$lags
    if (length(lags) != 1L || !is.finite(lags) || lags < 1 || lags != as.integer(lags)) stop("lagged_outcomes requires a positive integer lags", call. = FALSE)
    if (!is.function(spec$predict)) stop("lagged_outcomes requires a fixed-model predict function", call. = FALSE)
    design <- frame[, setdiff(names(frame), c(outcome, forecast)), drop = FALSE]
    for (lag in seq_len(as.integer(lags))) {
      pos <- stops - lag; col <- rep(NaN, length(y)); ok <- pos >= 0
      col[ok] <- y[pos[ok] + 1L]; design[[paste0(outcome, "_lag", lag)]] <- col
    }
    return(.wp_as_rows(spec$predict(design), length(y), scored, "lagged_outcomes predict"))
  }
  stop(sprintf("unsupported rebuild method %s", spec$name), call. = FALSE)
}

.wp_reference <- function(value, frame, y, scored, stops, outcome, forecast, static = NULL) {
  if (is.null(value) || (is.character(value) && identical(value, "patient_rate"))) return(rep(mean(y[scored]), length(y)))
  if (is.character(value) || is.list(value)) {
    spec <- .wp_spec(value, "yardstick")
    if (!spec$name %in% c("running_rate", "window_count")) stop("yardstick may rebuild running_rate or window_count references", call. = FALSE)
    return(.wp_history(spec, frame, y, scored, stops, outcome, forecast))
  }
  if (is.function(value)) return(.wp_as_rows(.wp_call_y(value, y, scored, frame), length(y), scored, "yardstick function"))
  if (!is.null(static)) return(static)
  if (length(value) == 1L) return(rep(as.numeric(value), length(y)))
  if (length(value) == length(y)) return(as.numeric(value))
  stop("yardstick must be a column, scalar, row-aligned vector, or function", call. = FALSE)
}

.wp_null_reference <- function(value, draws, scored, stops, frame, outcome, forecast, static = NULL) {
  ns <- sum(scored); nperm <- nrow(draws)
  if (is.null(value) || (is.character(value) && identical(value, "patient_rate"))) {
    rates <- rowMeans(draws[, scored, drop = FALSE]); return(matrix(rates, nrow = nperm, ncol = ns))
  }
  if (is.character(value) || is.list(value)) {
    spec <- .wp_spec(value, "yardstick")
    if (spec$name == "running_rate") return(t(vapply(seq_len(nperm), function(i) .wp_running_rate(draws[i, ], stops, spec)[scored], numeric(ns))))
    if (spec$name == "window_count") return(t(vapply(seq_len(nperm), function(i) .wp_window_count(draws[i, ], stops, spec)[scored], numeric(ns))))
    stop("yardstick may rebuild running_rate or window_count references", call. = FALSE)
  }
  if (is.function(value)) {
    return(t(vapply(seq_len(nperm), function(i) .wp_reference(value, frame, draws[i, ], scored, stops, outcome, forecast)[scored], numeric(ns))))
  }
  if (is.null(static)) static <- as.numeric(value)
  matrix(rep(static[scored], each = nperm), nrow = nperm)
}

.wp_builtin <- function(y, p, ref, threshold, need_slope = FALSE) {
  draws <- nrow(y); out <- lapply(c("auc", "average_precision", "sensitivity", "specificity", "ppv", "npv", "net_benefit", "brier_skill"), function(.) rep(NaN, draws))
  names(out) <- c("auc", "average_precision", "sensitivity", "specificity", "ppv", "npv", "net_benefit", "brier_skill")
  if (need_slope) out$calibration_slope <- rep(NaN, draws)
  for (i in seq_len(draws)) {
    yi <- as.integer(y[i, ]); pi <- p[i, ]; ri <- ref[i, ]; n <- length(yi); ev <- sum(yi); nev <- n - ev
    alarm <- pi >= threshold; tp <- sum(alarm & yi == 1L); fp <- sum(alarm & yi == 0L); tn <- nev - fp; quiet <- n - sum(alarm); rate <- ev / n
    if (ev > 0 && nev > 0) {
      out$auc[[i]] <- auc(yi, pi); out$sensitivity[[i]] <- tp / ev; out$specificity[[i]] <- tn / nev
    }
    out$average_precision[[i]] <- .wp_ap(yi, pi)
    if (sum(alarm) > 0) out$ppv[[i]] <- tp / sum(alarm) - rate
    if (quiet > 0) out$npv[[i]] <- tn / quiet - (1 - rate)
    policy <- ri >= threshold
    out$net_benefit[[i]] <- .wp_net_benefit(yi, pi, threshold) - .wp_net_benefit(yi, as.numeric(policy), threshold)
    ref_loss <- mean((yi - ri)^2)
    if (ref_loss > 0) out$brier_skill[[i]] <- 1 - mean((yi - pi)^2) / ref_loss
    if (need_slope) out$calibration_slope[[i]] <- .wp_slope(yi, pi)
  }
  out
}

.wp_custom_batch <- function(fn, y, p) vapply(seq_len(nrow(y)), function(i) {
  value <- fn(as.integer(y[i, ]), as.numeric(p[i, ]))
  if (length(value) != 1L || !is.numeric(value)) stop("custom metric must return one numeric value", call. = FALSE)
  as.numeric(value)
}, numeric(1))

#' Evaluate patient-level observed metrics against rebuilt permutation chance.
#'
#' @param data Long-format data frame.
#' @param patient,time,outcome,forecast Column names (time may be NULL).
#' @param runin Optional time cutoff; earlier outcomes remain history but are not scored.
#' @param threshold Decision threshold; defaults to the cohort scored event rate.
#' @param rebuild NULL, a built-in rebuild specification, or a function.
#' @param metrics NULL for all nine metrics, metric names, or a named list of functions `f(y, p)`.
#' @param n_perm Number of within-patient shuffles (default 200).
#' @param seed Seed for reproducible within-patient shuffles.
#' @param yardstick NULL for each shuffled patient's event rate, a column/vector, function, or history specification.
#' @param perm_index Internal permutation-index hook for deterministic cross-language fixtures.
#' @return An object with patient results, summaries, comparisons, and settings.
#' @export
evaluate <- function(data, patient = "patient", time = "time", outcome = "y", forecast = "forecast",
                     runin = NULL, threshold = NULL, rebuild = NULL, metrics = NULL,
                     n_perm = 200L, seed = 0L, yardstick = NULL, chance = "mean",
                     perm_index = NULL) {
  if (!is.data.frame(data)) stop("data must be a data frame", call. = FALSE)
  if (length(n_perm) != 1L || is.na(n_perm) || n_perm < 1 || n_perm != as.integer(n_perm)) stop("n_perm must be a positive integer", call. = FALSE)
  if (length(seed) != 1L || is.na(seed) || seed != as.integer(seed)) stop("seed must be an integer", call. = FALSE)
  if (length(chance) != 1L || !chance %in% c("mean", "median")) stop("chance must be 'mean' or 'median'", call. = FALSE)
  if (!is.null(runin) && is.null(time)) stop("time is required when runin is given", call. = FALSE)
  cols <- c(patient, outcome, forecast, if (!is.null(time)) time)
  missing <- setdiff(cols, names(data)); if (length(missing)) stop(sprintf("missing dataframe columns: %s", paste(missing, collapse = ", ")), call. = FALSE)
  a <- .validate_arrays(data[[patient]], data[[outcome]])
  p_all <- suppressWarnings(as.numeric(data[[forecast]]))
  if (length(p_all) != length(a$y)) stop("forecast and outcome must have equal length", call. = FALSE)
  tt <- if (is.null(time)) NULL else data[[time]]
  if (!is.null(tt) && (!is.atomic(tt) || anyNA(tt))) stop(sprintf("%s contains missing values", time), call. = FALSE)
  score_all <- if (is.null(runin)) rep(TRUE, length(a$y)) else tryCatch(tt >= runin, error = function(e) stop("runin must be comparable with time", call. = FALSE))
  if (!any(score_all)) stop("no scored rows remain after runin", call. = FALSE)
  if (any(!is.finite(p_all[score_all]))) stop("forecast must be finite on scored rows", call. = FALSE)
  if (is.null(threshold)) threshold <- mean(a$y[score_all])
  threshold <- as.numeric(threshold)
  if (length(threshold) != 1L || !is.finite(threshold) || threshold <= 0 || threshold >= 1) stop("threshold must be strictly between 0 and 1", call. = FALSE)
  selected <- .wp_select_metrics(metrics); custom <- selected$custom; metric_names <- selected$names
  rebuild_spec <- .wp_spec(rebuild, "rebuild")
  if (!is.null(rebuild_spec) && is.null(time)) stop("time is required to rebuild a history-based forecast", call. = FALSE)
  rate_mode <- if (is.null(rebuild_spec) || rebuild_spec$name != "window_count") TRUE else rebuild_spec$as_rate %||% !identical(rebuild_spec$mode, "count")
  if (is.character(rate_mode)) rate_mode <- tolower(rate_mode) != "count"
  count_mode <- !is.null(rebuild_spec) && rebuild_spec$name == "window_count" && !isTRUE(rate_mode)
  prob_metrics <- c("sensitivity", "specificity", "ppv", "npv", "net_benefit", "brier_skill", "calibration_slope")
  if (count_mode && any(metric_names %in% prob_metrics)) stop("window_count count mode supports AUC, average_precision, and user metrics; use as_rate=TRUE for probability metrics", call. = FALSE)
  if (!count_mode && any(p_all[score_all] < 0 | p_all[score_all] > 1)) stop("forecast must contain probabilities in [0, 1]", call. = FALSE)
  static_all <- NULL; yard_value <- yardstick
  if (is.character(yardstick) && length(yardstick) == 1L && !yardstick %in% c("patient_rate", "running_rate", "window_count")) {
    if (!yardstick %in% names(data)) stop(sprintf("yardstick column %s is missing", yardstick), call. = FALSE)
    static_all <- as.numeric(data[[yardstick]]); yard_value <- static_all
  } else if (!is.null(yardstick) && !is.function(yardstick) && !is.character(yardstick) && !is.list(yardstick)) {
    static_all <- as.numeric(yardstick); if (length(static_all) == 1L) static_all <- rep(static_all, nrow(data))
    if (length(static_all) != nrow(data)) stop("yardstick vector must have one value per data row", call. = FALSE)
  }
  groups0 <- .wp_groups(a$patient); labels0 <- attr(groups0, "labels")
  groups <- vector("list", length(groups0)); names(groups) <- names(groups0)
  for (i in seq_along(groups0)) {
    ix <- groups0[[i]]
    if (!is.null(tt)) ix <- ix[order(tt[ix], seq_along(ix), method = "radix")]
    groups[[i]] <- ix
  }
  refs_all <- rep(NaN, length(a$y)); stops_by <- vector("list", length(groups))
  for (i in seq_along(groups)) {
    ix <- groups[[i]]; local_score <- score_all[ix]; n <- length(ix)
    stops <- if (is.null(tt)) seq.int(0L, n - 1L) else vapply(tt[ix], function(value) sum(tt[ix] < value), integer(1))
    stops_by[[i]] <- stops; frame <- data[ix, , drop = FALSE]; local_y <- a$y[ix]
    static_local <- if (is.null(static_all)) NULL else static_all[ix]
    reference <- .wp_reference(yard_value, frame, local_y, local_score, stops, outcome, forecast, static_local)
    if (length(reference) != n) stop(sprintf("yardstick for patient %s must align with the patient's rows", labels0[[i]]), call. = FALSE)
    if (any(!is.finite(reference[local_score])) || any(reference[local_score] < 0 | reference[local_score] > 1)) stop("yardstick must be finite probabilities in [0, 1]", call. = FALSE)
    refs_all[ix[local_score]] <- reference[local_score]
    if (!is.null(rebuild_spec)) {
      rebuilt <- .wp_history(rebuild_spec, frame, local_y, local_score, stops, outcome, forecast)[local_score]
      if (any(!is.finite(rebuilt)) || any(abs(rebuilt - p_all[ix][local_score]) > 1e-7)) stop(sprintf("rebuilt forecast for patient %s does not match forecast on scored rows", labels0[[i]]), call. = FALSE)
    }
  }
  ys <- a$y[score_all]; ps <- p_all[score_all]; pids <- a$patient[score_all]; ref_s <- refs_all[score_all]
  if (!count_mode) {
    observed_nine <- nine_metrics(pids, ys, ps, threshold, yardstick = ref_s)
    observed_naive <- naive_metrics(pids, ys, ps, threshold, population_rate = mean(ys))
  } else { observed_nine <- list(auc = list(pooled = auc(ys, ps)), average_precision = list(pooled = .wp_ap(ys, ps))); observed_naive <- list() }
  result_metrics <- stats::setNames(vector("list", length(metric_names)), metric_names)
  slope_null <- "calibration_slope" %in% metric_names && !count_mode
  null_y <- null_p <- null_pid <- list()
  if (is.null(perm_index)) set.seed(as.integer(seed))
  for (g in seq_along(groups)) {
    ix <- groups[[g]]; local_score <- score_all[ix]
    if (!any(local_score)) next
    frame <- data[ix, , drop = FALSE]; local_y <- a$y[ix]; obs_y <- local_y[local_score]; obs_p <- p_all[ix][local_score]
    ns <- length(obs_y); stops <- stops_by[[g]]; key <- labels0[[g]]
    perm_draws <- matrix(0L, nrow = as.integer(n_perm), ncol = ns)
    if (is.null(perm_index)) {
      for (b in seq_len(as.integer(n_perm))) perm_draws[b, ] <- sample.int(ns, ns)
    } else {
      for (b in seq_len(as.integer(n_perm))) {
        draw <- perm_index[[b]]
        vals <- if (is.list(draw)) draw[[key]] else if (!is.null(dim(draw))) draw[, key] else draw
        if (is.null(vals) || length(vals) != ns || !identical(sort(as.integer(vals)), seq_len(ns))) stop(sprintf("perm_index draw %d is missing a valid index vector for patient %s", b, key), call. = FALSE)
        perm_draws[b, ] <- as.integer(vals)
      }
    }
    y_draws <- t(vapply(seq_len(as.integer(n_perm)), function(b) obs_y[perm_draws[b, ]], integer(ns)))
    full_draws <- matrix(rep(local_y, each = as.integer(n_perm)), nrow = as.integer(n_perm)); full_draws[, local_score] <- y_draws
    if (is.null(rebuild_spec)) p_draws <- matrix(rep(obs_p, each = as.integer(n_perm)), nrow = as.integer(n_perm))
    else if (rebuild_spec$name %in% c("running_rate", "window_count")) {
      builder <- if (rebuild_spec$name == "running_rate") .wp_running_rate else .wp_window_count
      p_draws <- t(vapply(seq_len(as.integer(n_perm)), function(b) builder(full_draws[b, ], stops, rebuild_spec)[local_score], numeric(ns)))
    } else p_draws <- t(vapply(seq_len(as.integer(n_perm)), function(b) .wp_history(rebuild_spec, frame, full_draws[b, ], local_score, stops, outcome, forecast)[local_score], numeric(ns)))
    static_local <- if (is.null(static_all)) NULL else static_all[ix]
    if (is.null(yard_value) || (is.character(yard_value) && identical(yard_value, "patient_rate"))) {
      ref_draws <- matrix(rowMeans(y_draws), nrow = as.integer(n_perm), ncol = ns)
    } else if (is.character(yard_value) || is.list(yard_value)) {
      ref_spec <- .wp_spec(yard_value, "yardstick")
      if (ref_spec$name == "running_rate") ref_draws <- t(vapply(seq_len(as.integer(n_perm)), function(b) .wp_running_rate(full_draws[b, ], stops, ref_spec)[local_score], numeric(ns)))
      else if (ref_spec$name == "window_count") ref_draws <- t(vapply(seq_len(as.integer(n_perm)), function(b) .wp_window_count(full_draws[b, ], stops, ref_spec)[local_score], numeric(ns)))
      else stop("yardstick may rebuild running_rate or window_count references", call. = FALSE)
    } else if (is.function(yard_value)) ref_draws <- t(vapply(seq_len(as.integer(n_perm)), function(b) .wp_reference(yard_value, frame, full_draws[b, ], local_score, stops, outcome, forecast)[local_score], numeric(ns)))
    else ref_draws <- matrix(rep(static_local[local_score], each = as.integer(n_perm)), nrow = as.integer(n_perm))
    if (!count_mode && any(!is.finite(p_draws) | p_draws < 0 | p_draws > 1)) stop("rebuilt forecast must contain probabilities in [0, 1]", call. = FALSE)
    if (any(!is.finite(ref_draws) | ref_draws < 0 | ref_draws > 1)) stop("rebuilt yardstick must contain probabilities in [0, 1]", call. = FALSE)
    if (slope_null) {
      null_y[[length(null_y) + 1L]] <- y_draws
      null_p[[length(null_p) + 1L]] <- p_draws
      null_pid[[length(null_pid) + 1L]] <- rep.int(g, ns)
    }
    built <- .wp_builtin(y_draws, p_draws, ref_draws, threshold, "calibration_slope" %in% metric_names)
    user_null <- lapply(custom[intersect(names(custom), metric_names)], .wp_custom_batch, y = y_draws, p = p_draws)
    observed <- list()
    for (name in metric_names) {
      if (name %in% names(custom)) observed[[name]] <- as.numeric(custom[[name]](obs_y, obs_p))
      else if (name == "auc") observed[[name]] <- auc(obs_y, obs_p)
      else if (name == "average_precision") observed[[name]] <- .wp_ap(obs_y, obs_p)
      else if (name == "calibration_slope") observed[[name]] <- .wp_slope(obs_y, obs_p)
      else observed[[name]] <- built[[name]][[1L]] # replaced below with direct observed value
    }
    direct <- .wp_builtin(matrix(obs_y, nrow = 1L), matrix(obs_p, nrow = 1L), matrix(ref_s[match(ix[local_score], which(score_all))], nrow = 1L), threshold, "calibration_slope" %in% metric_names)
    for (name in setdiff(metric_names, names(custom))) if (name != "auc" && name != "average_precision") observed[[name]] <- direct[[name]][[1L]]
    for (name in metric_names) {
      value <- observed[[name]]; null <- if (name %in% names(custom)) user_null[[name]] else built[[name]]
      finite <- is.finite(null)
      if (!is.finite(value) || !any(finite)) next
      patient_chance <- if (chance == "mean") mean(null[finite]) else median(null[finite])
      pvalue <- (sum(null[finite] >= value) + 1) / (as.integer(n_perm) + 1)
      result_metrics[[name]]$per_patient[[key]] <- list(observed = value, chance_level = patient_chance,
        excess = value - patient_chance, p_value = pvalue, n_scored = ns)
    }
  }
  for (name in metric_names) {
    entries <- result_metrics[[name]]$per_patient; vals <- lapply(entries, `[[`, "excess")
    if (length(vals)) {
      excess <- vapply(vals, as.numeric, numeric(1)); obs <- vapply(entries, function(z) z$observed, numeric(1)); chances <- vapply(entries, function(z) z$chance_level, numeric(1)); pv <- vapply(entries, function(z) z$p_value, numeric(1))
      summary <- list(median_excess = median(excess), iqr_excess = .percentiles(excess),
        share_p_below_0_05 = mean(pv < 0.05), share_excess_positive = mean(excess > 1e-9),
        mean_observed = mean(obs), mean_chance_level = mean(chances), mean_excess = mean(excess), n_eligible = length(excess))
    } else summary <- list(median_excess = NaN, iqr_excess = c(NaN, NaN), share_p_below_0_05 = NaN,
      share_excess_positive = NaN, mean_observed = NaN, mean_chance_level = NaN, mean_excess = NaN, n_eligible = 0L)
    if (name %in% names(observed_nine) && name %in% names(observed_naive)) {
      usual <- observed_naive[[name]]$summary
      comparison <- list(pooled = observed_nine[[name]]$pooled,
        usual_per_patient_mean = usual$mean, usual_per_patient_median = usual$median,
        usual_per_patient_iqr = usual$iqr, usual_per_patient_n = usual$n)
      if (name == "brier_skill") {
        # The usual pooled reference is the cohort's scored event rate, not the patient yardstick.
        cohort_loss <- mean((ys - mean(ys))^2)
        comparison$pooled <- if (cohort_loss > 0) 1 - mean((ys - ps)^2) / cohort_loss else NaN
      }
      if (name == "calibration_slope") {
        observed_slope <- observed_nine[[name]]$summary$mean
        comparison$conditional_within_slope <- observed_slope
        if (slope_null && length(null_y)) {
          null_slopes <- rep(NaN, as.integer(n_perm))
          for (b in seq_len(as.integer(n_perm))) {
            ys_b <- unlist(lapply(null_y, function(z) z[b, ]), use.names = FALSE)
            ps_b <- unlist(lapply(null_p, function(z) z[b, ]), use.names = FALSE)
            pid_b <- unlist(null_pid, use.names = FALSE)
            null_slopes[[b]] <- suppressWarnings(conditional_within_slope(pid_b, ys_b, ps_b)$slope)
          }
          ok <- is.finite(null_slopes)
          if (any(ok) && is.finite(observed_slope)) {
            level <- if (chance == "mean") mean(null_slopes[ok]) else median(null_slopes[ok])
            comparison$conditional_within_slope_chance <- level
            comparison$conditional_within_slope_excess <- observed_slope - level
            comparison$conditional_within_slope_p <- (sum(null_slopes[ok] >= observed_slope) + 1) / (as.integer(n_perm) + 1)
          }
        }
      }
    } else {
      vals <- if (length(entries)) vapply(entries, function(z) z$observed, numeric(1)) else numeric()
      pooled <- if (name %in% names(custom)) as.numeric(custom[[name]](ys, ps)) else if (name == "auc") auc(ys, ps) else .wp_ap(ys, ps)
      comparison <- list(pooled = pooled,
        usual_per_patient_mean = if (length(vals)) mean(vals) else NaN,
        usual_per_patient_median = if (length(vals)) median(vals) else NaN,
        usual_per_patient_iqr = if (length(vals)) .percentiles(vals) else c(NaN, NaN),
        usual_per_patient_n = length(vals))
    }
    result_metrics[[name]]$summary <- summary; result_metrics[[name]]$comparison <- comparison
  }
  yard_label <- if (is.null(yardstick)) "patient_rate" else if (is.character(yardstick)) yardstick else if (is.list(yardstick)) as.character(yardstick$name %||% yardstick$method %||% "custom") else if (is.function(yardstick)) "callable" else "row_aligned"
  out <- list(metrics = result_metrics, settings = list(n_perm = as.integer(n_perm), seed = as.integer(seed),
    threshold = threshold, runin = runin, rebuild = if (is.null(rebuild_spec)) NULL else rebuild_spec$name, chance = chance,
    yardstick = yard_label, scored_rows = sum(score_all), patients = length(groups0)))
  class(out) <- c("withinpatient_evaluation", "list")
  out
}
