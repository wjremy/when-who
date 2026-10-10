\name{evaluate}
\alias{evaluate}
\title{Evaluate repeated binary forecasts against within-patient chance}
\description{
Computes patient-level observed metrics and compares them with values from
within-patient outcome shuffles. History-based forecasts and yardsticks are
rebuilt for each shuffle. Reports patient excesses and equal-patient summaries
alongside pooled and usual per-patient comparisons.
}
\usage{
evaluate(data, patient = "patient", time = "time", outcome = "y",
  forecast = "forecast", runin = NULL, threshold = NULL, rebuild = NULL,
  metrics = NULL, n_perm = 200L, seed = 0L, yardstick = NULL,
  chance = "mean", perm_index = NULL)
}
\arguments{
\item{data}{Long-format data frame.}
\item{patient,time,outcome,forecast}{Column names for patient, optional time, binary outcome, and forecast.}
\item{runin}{Optional time cutoff. Earlier outcomes remain in history but are not scored.}
\item{threshold}{Decision threshold. Defaults to the cohort's scored event rate.}
\item{rebuild}{NULL for a fixed forecast, \code{"running_rate"}, a window-count or lagged-outcome specification, or a function. Rebuilt observed forecasts must match scored rows.}
\item{metrics}{NULL for all nine built-ins, metric names, or a named list of functions \code{f(y, p)}.}
\item{n_perm}{Positive number of within-patient shuffles; default 200.}
\item{seed}{Seed for reproducible R permutation streams.}
\item{yardstick}{NULL for each patient's scored event rate recalculated per shuffle, a column/vector, function, or running-rate/window-count specification.}
\item{chance}{Use the mean (default) or median of each patient's shuffle distribution as chance level.}
\item{perm_index}{Internal hook: a list of permutation draws, each containing a named 1-based row-index vector per patient.}
}
\value{An object with \code{metrics}, \code{settings}, per-patient results, summaries, and comparisons.}
\examples{
\dontrun{
result <- evaluate(forecasts, time = "time", runin = 7,
  rebuild = "running_rate", yardstick = "running_rate", n_perm = 200)
print(result)
}
}
