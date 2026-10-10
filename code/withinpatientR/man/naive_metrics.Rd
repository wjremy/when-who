\name{naive_metrics}
\alias{naive_metrics}
\title{Compute usual patient-wise metrics}
\usage{naive_metrics(patient, y, forecast, threshold, population_rate)}
\arguments{
\item{patient}{Patient identifier per observation.}
\item{y}{Binary outcome vector.}
\item{forecast}{Forecast probabilities.}
\item{threshold}{Decision threshold in (0, 1).}
\item{population_rate}{Population event-rate reference.}
}
\value{Named per-patient summaries with usual chance levels.}
\description{Returns customary patient-wise metrics against population-level chance values.}
