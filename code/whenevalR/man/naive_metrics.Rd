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
\value{Named per-patient summaries with usual population-level reference values.}
\description{Returns customary per-patient metrics against population-level reference values.}
