# withinpatient

`withinpatient` computes reporting quantities for repeated binary forecasts in long format: one row per patient and time point, with a patient identifier, elapsed time, binary outcome, and forecast probability. The quantities follow the When-Who reporting checklist for repeated forecasts.

## Install

```console
python -m pip install withinpatient
```

Or install this source tree with `python -m pip install ./code/withinpatient`. Python 3.9 or later is required. NumPy and pandas provide the core metrics; statsmodels is used for the conditional within-patient calibration slope.

## Minimal simulated example

```python
import numpy as np
import pandas as pd
from withinpatient import evaluate, to_markdown

rng = np.random.default_rng(17)
n_patients, n_times = 40, 35
patient = np.repeat(np.arange(n_patients), n_times)
time = np.tile(np.arange(n_times), n_patients)
patient_rate = np.repeat(rng.beta(3, 12, n_patients), n_times)
forecast = np.clip(patient_rate + rng.normal(0, 0.08, len(patient)), 0.01, 0.99)
y = rng.binomial(1, forecast)
data = pd.DataFrame({"patient": patient, "time": time, "y": y, "forecast": forecast})

def history_forecast(outcomes):
    """Simple rate forecast using outcomes strictly before each row."""
    outcomes = np.asarray(outcomes, dtype=float)
    earlier_events = np.r_[0.0, np.cumsum(outcomes[:-1])]
    return (0.25 + earlier_events) / (1.0 + np.arange(len(outcomes)))

result = evaluate(
    data, time="time", threshold=0.25, runin_days=7,
    history_forecast_fn=history_forecast,
    n_perm=199, n_boot=100, seed=17,
)
print(to_markdown(result))
```

Rows before day 7 supply history and are excluded from every scored quantity. `evaluate` returns the values as a nested dictionary; `to_markdown` renders its main summaries. For smaller or larger cohorts, adjust `n_perm` and `n_boot` to the desired computation and precision.

## `evaluate` inputs and reference rates

```python
evaluate(
    df, patient="patient", y="y", forecast="forecast", time="time",
    threshold=None, runin_days=None, prior_rate=None, prior_weight=1.0,
    n_perm=2000, n_boot=0, seed=0,
)
```

`time` is a numeric elapsed-time column. If `runin_days` is set, the run-in consists of rows with `time < runin_days`; all reported scores use rows at or after the cutoff. Run-in outcomes can initialize the patient's run-in rate and the expanding rate. The expanding rate at each time uses strictly earlier rows, so same-time, current, and future outcomes never enter that row's reference.

`prior_rate` is optional and must be supplied from outside the evaluation records, typically from a training cohort. It seeds only the expanding reference with `prior_weight` pseudo-observations. With no `prior_rate`, the expanding reference uses earlier outcomes only and is undefined until an earlier row exists; rows with an undefined reference are excluded from that skill calculation. When a run-in is supplied, its earlier outcomes provide the patient's own initial rate. `population_rate` and `retrospective_patient_rate` are included as descriptive hindsight comparisons; they use scored outcomes and are not prospective forecasts.

The `skill` result maps each reference name to pooled Brier skill, equal-patient mean skill, patient-specific skill ratios, positive share, and eligibility counts. A patient ratio is omitted if its reference is unavailable or has zero Brier loss. Reference arrays are included only with `return_references=True`; these arrays align with the scored rows.

## Quantities returned

- **Intended use:** checklist item 1 remains an author-supplied description of the decision; patient, time, outcome, and forecast columns cannot determine it.
- **Discrimination:** pooled AUC; per-patient AUC and its equal-patient mean, median, IQR, and share above 0.5; per-time-point AUC with equal-weight, comparison-weighted, median, IQR, and used/excluded counts. Per-patient AUC requires both outcomes. Per-time-point AUC also requires both outcomes and at least five patients by default.
- **Rate benchmarks:** retrospective `A0` is the pooled AUC from each patient's scored event rate. `A0_population` subtracts the S4 binomial sampling-noise moment from the day-weighted variance of patient rates and maps the corrected SD with the normal-CDF approximation. This population value is an approximate benchmark.
- **Pooled-AUC references:** `pooled_auc_permutation` shuffles each patient's forecasts among their scored rows while outcomes stay fixed. It reports the observed pooled AUC, null mean, 2.5th and 97.5th percentiles, and one-sided `(count + 1) / (B + 1)` p-value. `auc.patient_mean` gives the pooled AUC of each patient's mean forecast and its share of the model pooled AUC above 0.5.
- **Thresholds:** pooled and equal-patient sensitivity/specificity/Youden, predictive-value excess over each patient's own rate, alarm counts for every patient, and alarms per patient per elapsed time unit. Alarm rates divide each count by `max(time) - min(time)` over that patient's scored rows; the rate is undefined for zero elapsed duration.
- **Scores and calibration:** pooled and patient-mean Brier/log-loss; Brier skill against each named reference, pooled and by patient; calibration-in-the-large per patient; pooled, between-patient and joint within-deviation calibration slopes; and a conditional within-patient slope (requires statsmodels).
- **History permutation:** if `history_forecast_fn`, `time`, and `runin_days` are given, `permutation_reference` fixes run-in outcomes, permutes each patient's scored outcomes, rebuilds forecasts from each permuted history, and reports cohort and per-patient observed values beside their reference means and percentiles.
- **Intervals:** `n_boot > 0` adds `bootstrap.intervals`, percentile intervals from resampling whole patient records. Each sampled copy of a patient is treated as a separate patient. Intervals are off by default.

## Functions

Every public function is also exported from `withinpatient`.

| Function | Purpose |
|---|---|
| `evaluate` | Compute the checklist quantities from a dataframe and optionally add permutation tests and patient-bootstrap intervals. |
| `to_markdown` | Render the main `evaluate` summaries as a Markdown table. |
| `auc` | Compute tie-aware binary AUC; return NaN if either outcome class is absent. |
| `pooled_auc` | Compute pooled AUC and per-patient AUCs with eligibility counts; optional summaries can weight patients, pairs, or days. |
| `within_patient_auc` | Average patient-specific AUCs, with patient, pair, or day weighting. |
| `per_patient_auc_summary` | Summarize eligible patient AUCs by mean, median, IQR, and share above 0.5. |
| `per_time_point_auc` | Compute AUC within each time value and summarize equal-weight, comparison-weighted, median, and IQR results. |
| `rate_benchmark` | Compute the pooled AUC of patient-rate scores; by default uses retrospective scored-period rates. |
| `rate_benchmark_from_summary` | Apply the first-order normal-rate approximation from a supplied mean and SD. |
| `population_a0_moment_corrected` | Estimate population A0 after the S4 binomial-noise moment correction. |
| `pooled_auc_permutation_test` | Permute forecasts within patients with outcomes fixed and test the pooled AUC against that null. |
| `threshold_metrics` | Compute pooled/patient-level classification measures, alarm counts, and elapsed-time alarm rates when time is supplied. |
| `predictive_value_excess` | Compute each patient's PPV above own event rate and NPV above own non-event rate, plus pooled values. |
| `brier_scores` | Compute pooled and equal-patient Brier score and log-loss. |
| `brier_skill` | Compute Brier skill for one forecast/reference pair using summed losses. |
| `brier_skill_summary` | Compute pooled skill and patient-specific skill ratios, omitting rows with missing references. |
| `population_rate` | Repeat the supplied outcomes' overall rate; descriptive and retrospective when called on evaluation outcomes. |
| `retrospective_patient_rate` | Repeat each patient's rate over the supplied outcomes; a hindsight reference. |
| `runin_patient_rate` | Estimate each patient's rate on rows before a cutoff and repeat it across their rows. |
| `expanding_patient_rate` | Compute each row's rate from strictly earlier rows, with an optional scalar or row-specific prior. |
| `calibration_slopes` | Fit pooled and joint between-patient/within-deviation logistic calibration slopes. |
| `conditional_within_slope` | Fit the conditional logistic within-patient calibration slope with statsmodels. |
| `patient_calibration_in_the_large` | Return observed-minus-mean-forecast differences per patient and cohort summaries. |
| `permutation_reference` | Rebuild history-based forecasts after within-patient outcome permutations, preserving run-in history. |
| `patient_bootstrap` | Apply a caller-supplied statistic to whole-patient bootstrap resamples. |

## CSV command

```console
python -m withinpatient cohort.csv --patient patient_id --y event --forecast risk --time day --threshold 0.2 --runin 14 --prior-rate 0.18 --permutations 2000 --bootstrap 500
```

`--prior-rate` is optional; omit it to use only earlier patient outcomes for the expanding reference. The command prints a Markdown summary to standard output.
