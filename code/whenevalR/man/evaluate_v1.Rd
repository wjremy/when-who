\name{evaluate_v1}
\alias{evaluate_v1}
\title{Evaluate repeated forecasts using the legacy report API}
\description{Runs the pre-2.0.0 reporting API. This function is deprecated; use \code{evaluate()} instead.}
\usage{
evaluate_v1(data, patient = "patient", y = "y", forecast = "forecast",
  time = NULL, runin_days = NULL, prior_rate = NULL, prior_weight = 1,
  n_perm = 2000L, seed = 0L, history_forecast_fn = NULL,
  permutations = NULL, return_references = FALSE, min_patients = 5L)
}
\arguments{
\item{data}{Long-format data frame.}
\item{patient,y,forecast}{Column names identifying patient, binary outcome, and forecast.}
\item{time}{Optional numeric elapsed-time column.}
\item{runin_days}{Optional cutoff; earlier rows are not scored.}
\item{prior_rate}{Optional external event rate.}
\item{prior_weight}{Nonnegative prior pseudo-count.}
\item{n_perm}{Number of permutation draws.}
\item{seed}{Random seed.}
\item{history_forecast_fn}{Optional history forecast function.}
\item{permutations}{Optional explicit permutation maps.}
\item{return_references}{Whether to return reference-rate vectors.}
\item{min_patients}{Minimum patients for time-point AUC.}
}
\value{A legacy evaluation list.}
\seealso{\code{evaluate()}}
