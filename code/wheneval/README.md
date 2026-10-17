# wheneval

`wheneval` scores repeated forecasts within each patient and compares each score with a shuffled baseline. For every patient, scored outcomes are shuffled within that patient's record, forecasts and outcome-based references are rebuilt, and the same metric is recomputed. The report gives each patient's score and excess over the shuffled baseline, equal-patient summaries, and pooled and per-patient comparisons.

## Install

```console
python -m pip install ./code/wheneval
```

Python 3.9 or later is required. NumPy and pandas provide the metrics; the exact conditional calibration-slope calculation uses the package's numba kernel when available and a Python fallback otherwise.

## Quickstart: a fitted model

Give the data, the fitted model and the columns it uses as inputs. wheneval computes the model's forecasts, shuffles each patient's scored outcomes, rebuilds the forecasts and returns each patient's score, shuffled baseline, excess and permutation p value.

```python
import pandas as pd
import wheneval as we

data = pd.read_csv("diary.csv")  # patient_id, time, outcome and the model's input columns
result = we.evaluate(data, patient_id="patient_id", time="time", outcome="outcome",
                     model=fitted_model, features=["glucose", "steps", "rate_7", "last"],
                     history={"rate_7": we.window_rate(7), "last": we.lag(1)})
print(we.to_markdown(result))
```

`model` is any fitted model with `predict_proba` (scikit-learn and similar), a model with `predict` returning probabilities, or a function `f(X)`.

**Which inputs need `history`?** Every input computed from the patient's earlier outcomes: the last outcome, an event rate or count over recent occasions, time since the last event. On every shuffle these must be recomputed from the shuffled outcomes; otherwise the shuffled baseline misses the finite-record bias. Name each such column in `history` with its recipe:

| Recipe | Value at each row |
|---|---|
| `we.lag(k)` | the outcome `k` occasions earlier |
| `we.rate_so_far()` | event rate over all earlier occasions, run-in included |
| `we.window_rate(w)` | event rate over the last `w` earlier occasions |
| `we.window_count(w)` | number of events over the last `w` earlier occasions |
| `we.time_since_event()` | time since the last earlier event, in units of the time column |

Recipes count occasions (rows) in time order and use only rows strictly before the current one. wheneval first checks each declared column against its recipe on the real outcomes and stops with the first mismatch. Rows without history (the first occasions) keep their value from the data, since it does not depend on the outcomes. All other input columns are kept fixed: sensor readings, questionnaire items, time of day. `history` is optional; if a column you did not declare equals a standard recipe, wheneval warns and prints the line to add.

For inputs no recipe covers, pass your own feature function as `features`: `features=make_features` receives one patient's rows in time order, with the (shuffled) outcome column filled in, and returns the model's inputs. It is called again on every shuffle.

A model fitted on each patient's own record (rather than on other patients) needs refitting on every shuffle: use `history_rule=we.refit(fit_predict)` below.

## Quickstart: saved forecasts

The default metrics are AUC, average precision, sensitivity, specificity, PPV, NPV, net benefit, Brier skill, and calibration slope. The threshold defaults to the cohort's scored event rate, `n_shuffles=200`, and `seed=0`. `runin` is a time cutoff: earlier outcomes remain available to history rules but are not scored. The default `yardstick=None` is each patient's scored-period event rate, recalculated in every shuffle. `baseline="mean"` (default) or `baseline="median"` summarizes each patient's shuffled score distribution.

```python
import pandas as pd
import wheneval as we

data = pd.read_csv("forecasts.csv")  # patient_id, time, outcome, prediction columns
result = we.evaluate(data=data, patient_id="patient_id", time="time",
                     outcome="outcome", prediction="prediction",
                     runin=7, threshold=0.25, n_shuffles=200, seed=17)
print(we.to_markdown(result))
```

`result["metrics"][name]["per_patient"]` maps each eligible patient to `score`, `shuffled_baseline`, `excess`, `p_value`, and `n_scored`. Summaries use `mean_score` and `mean_shuffled_baseline`; comparisons include the pooled score and the usual per-patient score.

## History rules

`history_rule=None` keeps a prediction fixed. The built-in `"rate_so_far"` (alias `"running_rate"`) and `"window_count"` rules are rebuilt after every shuffle. For `window_count`, pass `window=7` or use a mapping with `w=7`. A `lagged_outcomes` mapping applies a fixed model to rebuilt lag inputs.

Use `we.refit(fit_predict, folds=5)` for a model fitted on each patient's own record with folds rotating over the record. It calls `fit_predict(train_rows, test_rows)` again for every shuffle and sets held-out outcome values to missing. Pass `prediction=None` to have `evaluate()` create observed predictions from the history rule. Otherwise, predictions on scored rows must match the supplied prediction column.

```python
result = we.evaluate(data=data, history_rule="window_count", window=7,
                     n_shuffles=200)
```

## User metrics

Supply a callable `f(y, p) -> float`. Higher values should mean better performance because the permutation p-value is the upper-tail probability.

```python
from wheneval import auc

result = we.evaluate(data, metrics={"auc_copy": auc}, n_shuffles=200, seed=17)
```

Built-in names can be selected with `metrics=["auc", "average_precision"]`; custom metrics can be mixed into that list or supplied in a `{name: function}` mapping.

## CLI

```console
python -m wheneval forecasts.csv --patient patient_id --time time --outcome outcome --forecast prediction --runin 7 --threshold 0.25 --permutations 200 --seed 17
```

Use `--rebuild running_rate` or `--rebuild window_count --window 7` to rebuild a history forecast. The CLI prints patient score and shuffled-baseline results first.

## Changes in 3.1.0

`evaluate()` takes a fitted `model` with its `features` (a column list or a feature function) and an optional `history` mapping of outcome-derived columns to recipes (`lag`, `rate_so_far`, `window_rate`, `window_count`, `time_since_event`). Declared history columns are checked against the data and rebuilt on every shuffle; undeclared columns that equal a standard recipe raise a warning. The summary table leads with the mean excess, with median [IQR] beside it, and permutation p values use only shuffles on which the score is defined. The R port does not yet have the `model`/`features` interface; there, pass a rebuild function.

## Changes in 3.0.0

The package is now named `wheneval` (formerly `withinpatient`). `evaluate()` uses `patient_id`, `prediction`, `history_rule`, `n_shuffles`, and `baseline`; the old keyword names `patient`, `forecast`, `rebuild`, `n_perm`, and `chance` remain accepted. History rules now include `refit()` for models fitted on a patient's own record, and `prediction=None` computes observed predictions from the rule. Result keys now use `score`, `shuffled_baseline`, and `mean_score` / `mean_shuffled_baseline`, with comparison keys named `per_patient_*`.

Earlier releases added within-patient shuffling, history and yardstick rebuilding, custom metrics, and tailored-first reports. Lower-level helpers such as `expanding_patient_rate` remain available, and `evaluate_v1()` retains the original reporting API.
