# withinpatient

`withinpatient` applies the two-step fix in Section 3 to repeated binary forecasts.

1. Score every measure separately within each patient; sensitivity and specificity use the same patients with both outcomes.
2. For each patient, shuffle scored outcomes while keeping run-in outcomes fixed.
3. Rebuild any forecast that uses outcome history after every shuffle; fitted models stay fixed.
4. Rebuild outcome-based yardsticks too, then recompute the same measure to estimate chance and its permutation p-value.
5. Report patient-level excesses and equal-weight summaries, with pooled and usual per-patient values beside them.

## Install

```console
python -m pip install ./code/withinpatient
```

Python 3.9 or later is required. NumPy and pandas provide the metrics; the exact conditional calibration-slope calculation uses the package's numba kernel when available and a Python fallback otherwise.

## Quickstart

The default metrics are AUC, average precision, sensitivity, specificity, PPV, NPV, net benefit, Brier skill, and calibration slope. The default threshold is the cohort's scored event rate, `n_perm=200`, and `seed=0`. `runin` is a time cutoff; earlier outcomes remain available to history builders but are not scored. The default `yardstick=None` is the patient's scored-period event rate, recalculated in each shuffle. `chance="mean"` (default) takes each patient's chance level as the mean of their shuffle distribution; `chance="median"` uses the median. Calibration is summarised by one conditional within-patient slope (patient intercepts conditioned out) against its own chance level from the same shuffles; per-patient slopes are reported but are noisy on short records and serve as description only.

```python
import pandas as pd
from withinpatient import evaluate, to_markdown

data = pd.read_csv("forecasts.csv")  # patient, time, y, forecast columns
result = evaluate(data, patient="patient", time="time", outcome="y",
                  forecast="forecast", runin=7, threshold=0.25,
                  n_perm=200, seed=17)
print(to_markdown(result))
```

The returned `result["metrics"][name]["per_patient"]` maps each eligible patient to `observed`, `chance_level`, `excess`, and `p_value`. The summary gives median excess, its IQR, the share with `p < 0.05`, the share with positive excess, and the eligible count. Markdown and the CLI list these tailored columns first.

## User metric

Supply any callable `f(y, p) -> float`. Higher values should mean better performance because the permutation p-value is the upper-tail probability.

```python
import numpy as np
from withinpatient import auc, evaluate

def auc_copy(y, p):
    return auc(y, p)

result = evaluate(data, metrics={"auc_copy": auc_copy}, n_perm=200, seed=17)
```

Built-in names can be selected with `metrics=["auc", "average_precision"]`; custom metrics can be mixed into that list or supplied in a `{name: function}` mapping.

## Forecasts that use history

For `rebuild="running_rate"`, the default builder uses strictly earlier outcomes and a 0.5 prior rate with one pseudo-observation. To set them explicitly, pass a mapping. The observed forecast column must match the configured builder on scored rows. Use `yardstick="running_rate"` to rebuild an updated-rate reference in every draw.

```python
import numpy as np

data = data.sort_values(["patient", "time"]).copy()
data["forecast"] = np.nan
for _, rows in data.groupby("patient", sort=False):
    y = rows["y"].to_numpy(float)
    t = rows["time"].to_numpy()
    stops = np.searchsorted(t, t, side="left")  # same-time outcomes are not past outcomes
    previous = np.r_[0.0, np.cumsum(y)][stops]
    data.loc[rows.index, "forecast"] = (0.5 + previous) / (1 + stops)

result = evaluate(data, runin=7, rebuild="running_rate", yardstick="running_rate",
                  threshold=0.25, n_perm=200, seed=17)
```

Other builders are `{"name": "window_count", "w": 7, "as_rate": True}` and `{"name": "lagged_outcomes", "lags": L, "predict": fixed_model_predict}`. For `lagged_outcomes`, the predict function receives the patient's ordered data frame with `outcome_lag1` through `outcome_lagL`; it must call an already fitted model and return probabilities. `window_count` can return counts with `as_rate=False`; count mode supports ranking measures (AUC/AP) and user metrics. A custom `rebuild(y_history)` function can return a forecast for every row, or scored rows only. When it accepts a second argument, it also receives the Boolean scored-row mask.

## CLI

```console
python -m withinpatient forecasts.csv --patient patient --time time --outcome y --forecast forecast --runin 7 --threshold 0.25 --permutations 200 --seed 17
```

Add `--rebuild running_rate` to rebuild a running rate, or `--rebuild window_count --window 7` for a rolling rate. The CLI prints the tailored results first.

## Compatibility and version history

The v2 `evaluate` API is a breaking change. The previous reporting API remains as `evaluate_v1(...)` and raises a `DeprecationWarning`; lower-level metric helpers such as `expanding_patient_rate` remain available. The `code/withinpatientR` package was not changed in this version.

- **2.0.0** — Add within-patient outcome permutations, history and yardstick rebuilding, custom metrics, and tailored-first reports. Retain the old report function as `evaluate_v1`.
- **1.0.0** — Original pooled and within-patient reporting quantities, permutation helpers, and patient bootstrap intervals.
