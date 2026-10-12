\name{nine_metrics}
\alias{nine_metrics}
\title{Compute pooled and within-patient paper metrics}
\usage{nine_metrics(patient, y, forecast, threshold, yardstick = NULL)}
\arguments{
\item{patient}{Patient identifier per observation.}
\item{y}{Binary outcome vector.}
\item{forecast}{Forecast probabilities.}
\item{threshold}{Decision threshold in (0, 1).}
\item{yardstick}{Optional row-aligned reference probabilities; defaults to patient event rates.}
}
\value{Named results for the nine pooled and per-patient metrics.}
\description{Returns pooled metrics, per-patient values, and equal-patient summaries. Calibration uses the exact conditional within-patient slope.}
