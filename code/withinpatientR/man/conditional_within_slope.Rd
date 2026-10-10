\name{conditional_within_slope}
\alias{conditional_within_slope}
\title{Fit an exact conditional within-patient calibration slope}
\usage{conditional_within_slope(patient, y, forecast)}
\arguments{
\item{patient}{Patient identifier per observation.}
\item{y}{Binary outcome vector.}
\item{forecast}{Forecast probabilities.}
}
\value{A list with slope, standard error, contributing patient count, estimability, and reason.}
\description{Fits a one-covariate conditional logistic model using an exact elementary-symmetric-polynomial likelihood and damped Newton updates.}
