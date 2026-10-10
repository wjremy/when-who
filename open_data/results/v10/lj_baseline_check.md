# Langener & Jacobson moving-rate baseline check

## Source check

Table 7 reports a moving-window baseline within-person AUC of 0.58. The published article says each person receives a probability drawn from a Beta distribution, that this probability is held constant across that person's days, and that the daily outcomes are Bernoulli draws. Thus, σ = 0.25 is on the **probability scale**, not the logit scale. The article also says that people without both outcome classes in the test split are omitted from within-person AUCs. [Article](https://doi.org/10.1177/25152459261418960)

The [GitHub repository](https://github.com/AnnaLangener/Justintime_Paper) is public and its landing page lists [`Run_Simulation.R`](https://github.com/AnnaLangener/Justintime_Paper/blob/main/Run_Simulation.R) and [`Simulation_Functions_notshiny.R`](https://github.com/AnnaLangener/Justintime_Paper/blob/main/Simulation_Functions_notshiny.R). The landing page and article loaded, but the R file contents could not be fetched here. Consequently, the code's exact split, default window, AUC implementation, and tie convention could not be verified. The article describes a 30-day example (train on days 1–30, predict day 31), which is the window used below.

## Assumptions used

- Draw each of 150 person probabilities independently from Beta(α,β), with mean 0.1 and SD 0.25; then draw 200 independent Bernoulli outcomes per person conditional on that probability. This matches the outcome generator described in the article. The contemporaneous feature variables used by its random forest are omitted because they are not used by the rate baseline.
- Use days 1–30 as initial history, then score days 31–200 one at a time. For each scored day, the primary forecast is the mean of that person's preceding 30 outcomes; each scored outcome joins the history for later forecasts. The published text illustrates a 30-day window but does not establish the code's default.
- For the expanding-mean variant, average all prior outcomes starting on day 1. For the leak variant, use the most recent 30 outcomes ending on the scored day, including the target outcome.
- Compute AUC separately for each person with both classes in the 170 scored outcomes, then average eligible people equally. Use `withinpatient.metrics.auc` for half-concordant ties. In the ties-dropped variant, omit tied event/non-event pairs; in the constant-score exclusion variant, omit people whose forecast is constant throughout the test period.
- The logit-scale sensitivity uses a logistic-normal person probability with logit SD 0.25; its center is calibrated so the mean probability remains 0.1. The diagnostic automatic-direction row selects ROC direction from test-set medians, as in [pROC's documented `direction = "auto"` behavior](https://search.r-project.org/CRAN/refmans/pROC/html/roc.html); the source code does not establish that the authors used this convention.
- The MC SD is the sample SD across the 200 replication-level mean within-person AUCs. NumPy's generator is seeded with 20261011; its random stream does not reproduce R's.

## Results

200 replications; seed 20261011. The Table 7 value is shown for comparison, not as a simulation variant.

| Variant | Mean within-person AUC | MC SD |
|---|---:|---:|
| Table 7 reported moving-window baseline | 0.580 | — |
| Beta probability, moving past 30 days, ties half | 0.4491 | 0.0126 |
| Beta probability, expanding past mean, ties half | 0.3586 | 0.0215 |
| Beta probability, moving past 30 days, ties dropped | 0.3645 | 0.0343 |
| Beta probability, moving past 30 days, constant-score persons excluded | 0.4490 | 0.0126 |
| Beta probability, moving 30 days including current day (leak) | 0.7144 | 0.0197 |
| Logit-normal, logit SD 0.25, moving past 30 days, ties half | 0.4569 | 0.0063 |
| Beta probability, moving past 30 days, pROC-style automatic direction diagnostic | 0.4964 | 0.0150 |

The fixed-direction, past-only running rate averages 0.4491, below chance, in the direction predicted by the finite-record effect; dropping ties makes it lower, and excluding constant-score people barely changes it. Including the current day reverses the sign (0.7144), while the logit-scale and constant-score variants remain below 0.5; none of the requested variants reproduces 0.58. The source code remains necessary to determine whether its precise split or ROC direction accounts for the gap; the automatic-direction diagnostic shown here does not.

## Run

```powershell
$env:PYTHONPATH = 'code/withinpatient/src'
& "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" code/lj_baseline_check.py
```

## Addendum (Claude, 10 Oct): what their code does

Read from the repository's raw files (`Simulation_Functions_notshiny.R`, `Run_Simulation.R`; read only, nothing executed):
- The sliding-window baseline is a random-intercept logistic model (`glmer(y ~ 1 + (1|subject))`) fitted on the training window; when the function is called for one subject at a time, as `Run_Simulation.R` does, it is the training-window mean of y. Split: `SlidingWindow_CV(n_samples, windowsize, 1)`, training = the previous `windowsize` time points, test = the next one (horizon 1), so past only; `windowsize = 14` in `Run_Simulation.R`.
- Within-person AUC: `pROC::auc(pROC::roc(true, pred, direction = "<"))` per person (fixed direction; pROC counts ties as half), persons without both classes dropped; mean over persons.
- The baseline's `auc_individual` in the Baseline block comes from `run_simulation(features_sample, "row-wise", 1, testsize = 0.3)`, i.e. a row-wise random split, not the sliding window. With a row-wise split the baseline is constant within each person in the test rows, so its within-person AUC is exactly 0.5 under pROC's tie handling.
- The data generator (`create_data`) is in a file not included in the repository (`Simulation_continous.R`), with `time_type = "DCP"`; whether outcomes have time structure in the reported scenario cannot be checked.

With their window (14 days, past only, window mean, half ties, fixed direction), our reproduction gives a mean within-person AUC of 0.476 (MC SD 0.001; 100 replications), and 0.450 with 30 days. Neither reading of their pipeline yields 0.58 for a forecast without time structure: the sliding-window mean is below 0.5 (finite-record effect), and the row-wise baseline is exactly 0.5. A value of 0.58 therefore needs time structure in the generated outcomes or a different pipeline than the one in the repository.
