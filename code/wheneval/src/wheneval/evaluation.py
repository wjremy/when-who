"""Paper Section 3: patient-wise scoring against rebuilt shuffled baselines."""
from __future__ import annotations

import inspect
from numbers import Integral
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd

from .metrics import _check_missing, _outcomes, _probabilities, auc, conditional_within_slope
from .nine import average_precision, naive_metrics, nine_metrics
from .history import HistoryFeature, _candidates, _same

METRICS = ("auc", "average_precision", "sensitivity", "specificity", "ppv", "npv",
           "net_benefit", "brier_skill", "calibration_slope")
PROBABILITY_METRICS = {"sensitivity", "specificity", "ppv", "npv", "net_benefit",
                       "brier_skill", "calibration_slope"}
ALIASES = {"ap": "average_precision", **{name: name for name in METRICS}}


# Excesses within this distance of zero count as zero when a sign is taken (floating-point noise).
SIGN_TOL = 1e-9
_UNSET = object()


def refit(fit_predict: Any, folds: int = 5) -> dict[str, Any]:
    """Build a rule for refitting a model on one patient's own record.

    This is for cross-validation rotating over the patient's record. Since each
    shuffle changes the outcomes used to fit the model, ``fit_predict`` is
    called again in every shuffle. Rows are assigned to folds by their
    time-sorted position modulo ``folds``. For each fold with scored rows,
    ``fit_predict(train_rows, test_rows)`` receives the other rows for training
    and that fold for testing; test outcome values are set to missing.
    """
    if not callable(fit_predict):
        raise TypeError("fit_predict must be callable")
    if isinstance(folds, bool) or not isinstance(folds, Integral) or int(folds) < 2:
        raise ValueError("folds must be an integer of at least 2")
    return {"name": "refit", "fit_predict": fit_predict, "folds": int(folds)}


def _predict(model: Any, X: Any) -> np.ndarray:
    """Event probabilities from a fitted model: predict_proba (positive class), a callable, or predict."""
    if hasattr(model, "predict_proba"):
        out = np.asarray(model.predict_proba(X), dtype=float)
        return out[:, -1] if out.ndim == 2 else out
    if callable(model):
        return np.asarray(model(X), dtype=float)
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X), dtype=float)
    raise TypeError("model needs predict_proba or predict, or must be a callable f(X)")


def _model_rule(data: pd.DataFrame, patient_id: Any, time: Optional[str], outcome: str,
                model: Any, features: Any, history: Any) -> tuple[Any, Optional[np.ndarray]]:
    """Turn model + features (+ history) into a rebuild rule.

    Returns (rule, fixed_predictions): a callable rule when some input depends on the
    outcomes, otherwise fixed predictions for every row.
    """
    if features is None:
        raise ValueError("model requires features: a list of columns or a function f(rows) -> inputs")
    hist = {} if history is None else dict(history)
    for col, recipe in hist.items():
        if not isinstance(recipe, HistoryFeature):
            raise TypeError(f"history[{col!r}] must be a recipe such as we.lag(1) or we.window_rate(7)")
    if callable(features):
        if hist:
            raise ValueError("history applies to a list of feature columns; a features function "
                             "must recompute its own history inputs from the outcome column")
        if time is None:
            raise ValueError("time is required with a features function")

        def rule(y, scored, frame):
            rows = frame.copy()
            rows[outcome] = np.asarray(y)
            return _predict(model, features(rows))
        return rule, None
    cols = [features] if isinstance(features, str) else list(features)
    missing = [c for c in cols if c not in data]
    if missing:
        raise ValueError(f"missing feature columns: {', '.join(map(str, missing))}")
    if outcome in cols:
        raise ValueError("the outcome itself cannot be a feature")
    stray = [c for c in hist if c not in cols]
    if stray:
        raise ValueError(f"history columns are not among the features: {', '.join(map(str, stray))}")
    if time is None:
        raise ValueError("time is required to check and rebuild history features")

    # One pass over patients in time order: check declared recipes, look for undeclared ones.
    undeclared = [c for c in cols if c not in hist and pd.api.types.is_numeric_dtype(data[c])]
    cands = _candidates()
    found: dict[Any, list[np.ndarray]] = {c: [] for c in undeclared}
    built: dict[int, list[np.ndarray]] = {i: [] for i in range(len(cands))}
    pid = data[patient_id].to_numpy()
    times_all = data[time].to_numpy()
    y_all = data[outcome].to_numpy()
    for key in pd.unique(pid):
        ix = np.flatnonzero(pid == key)
        ix = ix[np.argsort(times_all[ix], kind="stable")]
        y = y_all[ix].astype(float)
        t = times_all[ix]
        stops = np.searchsorted(t, t, side="left").astype(int)
        for col, recipe in hist.items():
            got = recipe.compute(y, stops, t)
            have = data[col].to_numpy(dtype=float)[ix]
            bad = np.isfinite(got) & ~(np.isfinite(have) & np.isclose(got, have, rtol=0, atol=1e-7))
            if bad.any():
                r = int(np.flatnonzero(bad)[0])
                raise ValueError(f"history column {col!r} is not {recipe!r} of {outcome!r}: patient {key!r}, "
                                 f"time {t[r]!r} has {have[r]!r}, the recipe gives {got[r]!r}")
        if undeclared:
            for c in undeclared:
                found[c].append(data[c].to_numpy(dtype=float)[ix])
            for i, recipe in enumerate(cands):
                built[i].append(recipe.compute(y, stops, t))
    for c in undeclared:
        have = np.concatenate(found[c]) if found[c] else np.array([])
        for i, recipe in enumerate(cands):
            if len(have) and _same(np.concatenate(built[i]), have):
                warnings.warn(f"feature column {c!r} equals the {recipe.label} ({recipe!r}): it looks computed "
                              f"from the outcome. If so, pass history={{{c!r}: {recipe!r}}}; otherwise "
                              f"the shuffled baseline keeps it fixed and misses the finite-record bias.",
                              UserWarning, stacklevel=3)
                break

    if not hist:
        return None, _predict(model, data[cols])

    def rule(y, scored, frame):
        rows = frame.copy()
        t = rows[time].to_numpy()
        stops = np.searchsorted(t, t, side="left").astype(int)
        for col, recipe in hist.items():
            new = recipe.compute(np.asarray(y, dtype=float), stops, t)
            old = rows[col].to_numpy(dtype=float)
            # Rows without history keep their value: it does not depend on the outcomes.
            rows[col] = np.where(np.isfinite(new), new, old)
        return _predict(model, rows[cols])
    return rule, None

def _select_metrics(metrics: Any) -> tuple[dict[str, Any], list[str]]:
    if metrics is None:
        return {}, list(METRICS)
    selected = [metrics] if isinstance(metrics, (str, bytes)) or callable(metrics) else list(
        metrics.items() if isinstance(metrics, dict) else metrics
    )
    custom: dict[str, Any] = {}
    names: list[str] = []
    for item in selected:
        if isinstance(item, tuple) and len(item) == 2:
            name, fn = item
            if not callable(fn):
                raise TypeError(f"custom metric {name!r} must be callable")
            label = str(name)
            if label.lower() in ALIASES:
                raise ValueError(f"custom metric name {label!r} conflicts with a built-in")
            custom[label] = fn
            names.append(label)
        elif callable(item):
            label = getattr(item, "__name__", "custom_metric")
            if label in ALIASES:
                label += "_custom"
            custom[label] = item
            names.append(label)
        else:
            key = str(item).strip().lower().replace(" ", "_").replace("-", "_")
            if key not in ALIASES:
                raise ValueError(f"unknown built-in metric {item!r}; choose from {METRICS}")
            name = ALIASES[key]
            if name not in names:
                names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("metric names must be unique")
    return custom, names


def _spec(value: Any, label: str) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if callable(value):
        return {"name": "callable", "function": value}
    if isinstance(value, str):
        spec = {"name": value}
    elif isinstance(value, dict):
        spec = dict(value)
        spec["name"] = spec.get("name", spec.get("method", spec.get("type")))
    else:
        raise TypeError(f"{label} must be a built-in name, configuration mapping, or callable")
    name = str(spec.get("name", "")).strip().lower().replace("-", "_")
    if name == "rate_so_far":
        name = "running_rate"
    if name not in {"running_rate", "window_count", "lagged_outcomes", "refit"}:
        raise ValueError(f"{label} must name rate_so_far, running_rate, window_count, lagged_outcomes, or refit")
    spec["name"] = name
    if name == "refit":
        if not callable(spec.get("fit_predict")):
            raise TypeError("refit requires a fit_predict callable")
        folds = spec.get("folds", 5)
        if isinstance(folds, bool) or not isinstance(folds, Integral) or int(folds) < 2:
            raise ValueError("folds must be an integer of at least 2")
        spec["folds"] = int(folds)
    return spec


def _stops(times: Optional[np.ndarray], n: int) -> np.ndarray:
    return np.arange(n, dtype=int) if times is None else np.searchsorted(times, times, side="left").astype(int)


def _running_rate_batch(y: np.ndarray, stops: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    weight = float(spec.get("prior_weight", spec.get("pseudo_count", 1.0)))
    prior = spec.get("prior_rate", 0.5)
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("running_rate prior_weight must be finite and nonnegative")
    if prior is None:
        if weight:
            raise ValueError("running_rate prior_rate is required when prior_weight is positive")
        prior = 0.0
    prior = float(prior)
    if not np.isfinite(prior) or not 0 <= prior <= 1:
        raise ValueError("running_rate prior_rate must be a probability in [0, 1]")
    cum = np.concatenate([np.zeros((len(values), 1)), np.cumsum(values, axis=1)], axis=1)
    denominator = stops.astype(float) + weight
    numerator = cum[:, stops] + prior * weight
    out = np.full(values.shape, np.nan, dtype=float)
    np.divide(numerator, denominator[None, :], out=out, where=denominator[None, :] > 0)
    return out


def _window_count_batch(y: np.ndarray, stops: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    try:
        width = int(spec["w"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("window_count requires a positive integer w") from exc
    if width < 1 or float(spec["w"]) != width:
        raise ValueError("window_count requires a positive integer w")
    cum = np.concatenate([np.zeros((len(values), 1)), np.cumsum(values, axis=1)], axis=1)
    starts = np.maximum(stops - width, 0)
    events = cum[:, stops] - cum[:, starts]
    counts = np.minimum(stops, width).astype(float)
    as_rate = spec.get("as_rate", spec.get("mode", "rate") != "count")
    if isinstance(as_rate, str):
        as_rate = as_rate.lower() != "count"
    if not as_rate:
        return events
    out = np.zeros(events.shape, dtype=float)
    np.divide(events, counts[None, :], out=out, where=counts[None, :] > 0)
    return out


def _call_y(function: Any, y: np.ndarray, scored: np.ndarray,
            frame: Optional[pd.DataFrame] = None) -> Any:
    """Call f(y), f(y, scored) or f(y, scored, frame), by the number of positional parameters."""
    try:
        signature = inspect.signature(function)
        params = list(signature.parameters.values())
        if any(p.kind == p.VAR_POSITIONAL for p in params):
            n_args = 3
        else:
            n_args = sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params)
    except (TypeError, ValueError):
        n_args = 1
    if n_args >= 3:
        return function(y.copy(), scored.copy(), None if frame is None else frame.copy())
    return function(y.copy(), scored.copy()) if n_args == 2 else function(y.copy())


def _as_rows(raw: Any, n: int, scored: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(raw, dtype=float)
    if values.ndim == 2 and values.shape[1] == 2:
        values = values[:, 1]
    if values.ndim != 1:
        raise ValueError(f"{label} must return a one-dimensional vector")
    count = int(scored.sum())
    if len(values) == n:
        return values
    if len(values) == count:
        out = np.full(n, np.nan, dtype=float)
        out[scored] = values
        return out
    raise ValueError(f"{label} must return {n} rows or {count} scored rows")


def _history_forecast(spec: dict[str, Any], frame: pd.DataFrame, y: np.ndarray,
                      scored: np.ndarray, stops: np.ndarray, outcome: str,
                      forecast: str) -> np.ndarray:
    name = spec["name"]
    if name == "running_rate":
        return _running_rate_batch(y, stops, spec)[0]
    if name == "window_count":
        return _window_count_batch(y, stops, spec)[0]
    if name == "callable":
        return _as_rows(_call_y(spec["function"], y, scored, frame), len(y), scored, "rebuild function")
    if name == "refit":
        folds = int(spec["folds"])
        n_rows = len(y)
        if folds > n_rows:
            raise ValueError(f"refit folds must be at most the patient's {n_rows} rows")
        current = frame.copy(deep=False)
        current[outcome] = np.asarray(y)
        positions = np.arange(n_rows, dtype=int)
        fold_id = positions % folds
        out = np.full(n_rows, np.nan, dtype=float)
        for fold in range(folds):
            test_ix = np.flatnonzero(fold_id == fold)
            if not scored[test_ix].any():
                continue
            train_ix = np.flatnonzero(fold_id != fold)
            train_rows = current.iloc[train_ix].copy()
            test_rows = current.iloc[test_ix].copy()
            test_rows[outcome] = np.nan
            predictions = np.asarray(spec["fit_predict"](train_rows, test_rows), dtype=float)
            if predictions.ndim != 1 or len(predictions) != len(test_ix):
                raise ValueError("refit fit_predict must return one prediction per test row")
            out[test_ix] = predictions
        return out
    if name != "lagged_outcomes":
        raise ValueError(f"unsupported rebuild method {name!r}")
    try:
        lags = int(spec["lags"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("lagged_outcomes requires a positive integer lags") from exc
    if lags < 1 or float(spec["lags"]) != lags:
        raise ValueError("lagged_outcomes requires a positive integer lags")
    predict = spec.get("predict")
    if not callable(predict):
        raise TypeError("lagged_outcomes requires a fixed-model predict callable")
    design = frame.drop(columns=[outcome, forecast], errors="ignore").copy()
    for lag in range(1, lags + 1):
        pos = stops - lag
        column = np.full(len(y), np.nan, dtype=float)
        valid = pos >= 0
        column[valid] = y[pos[valid]]
        design[f"{outcome}_lag{lag}"] = column
    return _as_rows(predict(design), len(y), scored, "lagged_outcomes predict")


def _reference_rows(value: Any, frame: pd.DataFrame, y: np.ndarray, scored: np.ndarray,
                    stops: np.ndarray, outcome: str, forecast: str,
                    static: Optional[np.ndarray] = None) -> np.ndarray:
    if value is None or (isinstance(value, str) and value == "patient_rate"):
        return np.full(len(y), float(np.mean(y[scored])), dtype=float)
    spec = _spec(value, "yardstick") if isinstance(value, (str, dict)) else None
    if spec is not None:
        if spec["name"] not in {"running_rate", "window_count"}:
            raise ValueError("yardstick may rebuild running_rate or window_count references")
        return _history_forecast(spec, frame, y, scored, stops, outcome, forecast)
    if callable(value):
        return _as_rows(_call_y(value, y, scored, frame), len(y), scored, "yardstick function")
    if static is not None:
        return static.copy()
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(len(y), float(arr), dtype=float)
    if arr.ndim == 1 and len(arr) == len(y):
        return arr.copy()
    raise ValueError("yardstick must be a column, scalar, row-aligned vector, or callable")


def _auc_batch(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    draws, n = score.shape
    pos_count = y.sum(axis=1).astype(float)
    neg_count = n - pos_count
    order = np.argsort(score, axis=1, kind="stable")
    s = np.take_along_axis(score, order, axis=1)
    yy = np.take_along_axis(y, order, axis=1)
    j = np.broadcast_to(np.arange(n), (draws, n))
    starts = np.ones((draws, n), dtype=bool)
    starts[:, 1:] = s[:, 1:] != s[:, :-1]
    first = np.maximum.accumulate(np.where(starts, j, 0), axis=1)
    ends = np.ones((draws, n), dtype=bool)
    ends[:, :-1] = s[:, :-1] != s[:, 1:]
    last = np.minimum.accumulate(np.where(ends, j, n - 1)[:, ::-1], axis=1)[:, ::-1]
    ranks = (first + last + 2) / 2.0
    rank_sum = np.sum(ranks * yy, axis=1)
    result = np.full(draws, np.nan, dtype=float)
    valid = (pos_count > 0) & (neg_count > 0)
    result[valid] = (rank_sum[valid] - pos_count[valid] * (pos_count[valid] + 1) / 2) / (pos_count[valid] * neg_count[valid])
    return result


def _ap_batch(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    draws, n = score.shape
    pos_count = y.sum(axis=1).astype(float)
    order = np.argsort(-score, axis=1, kind="stable")
    s = np.take_along_axis(score, order, axis=1)
    yy = np.take_along_axis(y, order, axis=1)
    cum = np.cumsum(yy, axis=1)
    j = np.broadcast_to(np.arange(n), (draws, n))
    starts = np.ones((draws, n), dtype=bool)
    starts[:, 1:] = s[:, 1:] != s[:, :-1]
    first = np.maximum.accumulate(np.where(starts, j, 0), axis=1)
    before = np.take_along_axis(cum, np.maximum(first - 1, 0), axis=1)
    before = np.where(first == 0, 0.0, before)
    block_pos = cum - before
    ends = np.ones((draws, n), dtype=bool)
    ends[:, :-1] = s[:, :-1] != s[:, 1:]
    precision = cum / (j + 1)
    result = np.full(draws, np.nan, dtype=float)
    valid = pos_count > 0
    result[valid] = np.sum(np.where(ends[valid], block_pos[valid] * precision[valid], 0), axis=1) / pos_count[valid]
    return result


def _slope(y: np.ndarray, p: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(conditional_within_slope(np.zeros(len(y), dtype=int), y, p)["slope"])


def _builtins_batch(y: np.ndarray, p: np.ndarray, ref: np.ndarray,
                    threshold: float, slope: bool) -> dict[str, np.ndarray]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    ref = np.asarray(ref, dtype=float)
    draws, n = y.shape
    alarm = p >= threshold
    events = y.sum(axis=1).astype(float)
    nonevents = n - events
    rate = events / n
    tp = np.sum(alarm & (y == 1), axis=1).astype(float)
    fp = np.sum(alarm & (y == 0), axis=1).astype(float)
    tn = nonevents - fp
    nalarm = alarm.sum(axis=1).astype(float)
    nquiet = n - nalarm
    odds = threshold / (1 - threshold)
    model_nb = tp / n - fp / n * odds
    policy = ref >= threshold
    policy_nb = (np.sum(policy & (y == 1), axis=1) / n
                 - np.sum(policy & (y == 0), axis=1) / n * odds)
    model_loss = np.mean((y - p) ** 2, axis=1)
    ref_loss = np.mean((y - ref) ** 2, axis=1)
    brier = np.full(draws, np.nan)
    np.divide(model_loss, ref_loss, out=brier, where=ref_loss > 0)
    ppv = np.full(draws, np.nan)
    npv = np.full(draws, np.nan)
    sens = np.full(draws, np.nan)
    spec = np.full(draws, np.nan)
    np.divide(tp, nalarm, out=ppv, where=nalarm > 0)
    np.divide(tn, nquiet, out=npv, where=nquiet > 0)
    np.divide(tp, events, out=sens, where=events > 0)
    np.divide(tn, nonevents, out=spec, where=nonevents > 0)
    both = (events > 0) & (nonevents > 0)
    values = {
        "auc": _auc_batch(y, p),
        "average_precision": _ap_batch(y, p),
        "sensitivity": np.where(both, sens, np.nan),
        "specificity": np.where(both, spec, np.nan),
        "ppv": ppv - rate,
        "npv": npv - (1 - rate),
        "net_benefit": model_nb - policy_nb,
        "brier_skill": 1 - brier,
    }
    if slope:
        values["calibration_slope"] = np.asarray([_slope(y[i], p[i]) for i in range(draws)])
    return values


def _custom_batch(function: Any, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    values = np.full(len(y), np.nan)
    for i in range(len(y)):
        result = function(y[i].copy(), p[i].copy())
        try:
            values[i] = float(result)
        except (TypeError, ValueError) as exc:
            raise TypeError("custom metric must return one numeric value") from exc
    return values


def _null_references(value: Any, frame: pd.DataFrame, draws: np.ndarray,
                     scored: np.ndarray, stops: np.ndarray, outcome: str,
                     forecast: str, static: Optional[np.ndarray]) -> np.ndarray:
    n_scored = int(scored.sum())
    if value is None or (isinstance(value, str) and value == "patient_rate"):
        rates = draws[:, scored].mean(axis=1)
        return np.repeat(rates[:, None], n_scored, axis=1)
    spec = _spec(value, "yardstick") if isinstance(value, (str, dict)) else None
    if spec is not None:
        if spec["name"] == "running_rate":
            return _running_rate_batch(draws, stops, spec)[:, scored]
        if spec["name"] == "window_count":
            return _window_count_batch(draws, stops, spec)[:, scored]
        raise ValueError("yardstick may rebuild running_rate or window_count references")
    if callable(value):
        result = np.empty((len(draws), n_scored), dtype=float)
        for i, y in enumerate(draws):
            result[i] = _reference_rows(value, frame, y, scored, stops,
                                        outcome, forecast)[scored]
        return result
    if static is None:
        static = np.asarray(value, dtype=float)
    return np.repeat(static[scored][None, :], len(draws), axis=0)


def evaluate(data: pd.DataFrame, patient_id: Any = _UNSET, time: Optional[str] = "time",
             outcome: str = "outcome", prediction: Any = _UNSET,
             history_rule: Any = _UNSET, window: Optional[int] = None,
             runin: Optional[float] = None, threshold: Optional[float] = None,
             metrics: Any = None, n_shuffles: Any = _UNSET, seed: int = 0,
             yardstick: Any = None, baseline: Any = _UNSET, model: Any = None,
             features: Any = None, history: Any = None, **legacy: Any) -> dict[str, Any]:
    """Score repeated forecasts against a within-patient shuffled baseline.

    ``runin`` is a time cutoff: earlier outcomes remain in history but are not
    scored. ``history_rule`` may be ``None`` (keep predictions fixed),
    ``"rate_so_far"``/``"running_rate"``, ``"window_count"`` with ``window``,
    a ``lagged_outcomes`` mapping, a callable ``f(y)``, ``f(y, scored)`` or
    ``f(y, scored, rows)``, or a :func:`refit` rule. A refit rule cross-validates
    within each patient's record and refits after every shuffle. A supplied
    prediction column must match its rule on scored rows; set ``prediction=None``
    to compute observed predictions from the rule.

    The default ``yardstick=None`` is each patient's scored-period event rate,
    recalculated in every shuffle. ``metrics`` defaults to the nine paper
    metrics and also accepts metric names or ``name -> f(y, p)`` mappings.
    ``n_shuffles`` defaults to 200. Patient-specific random streams derive from
    ``seed``. ``baseline`` is the mean (default) or median of each patient's
    shuffled score distribution. The old keyword names ``patient``,
    ``forecast``, ``rebuild``, ``n_perm`` and ``chance`` remain accepted.

    The simplest call gives a fitted ``model`` and its ``features``: a list of
    columns, or a function ``f(rows) -> inputs`` applied to one patient's rows in
    time order with the outcome column filled in. Columns computed from the
    outcome are named in ``history`` with a recipe, e.g.
    ``history={"rate_7": we.window_rate(7), "last": we.lag(1)}``; they are checked
    against the data and recomputed on every shuffle, and every other column is
    kept fixed. Undeclared columns that match a standard recipe raise a warning.
    Observed predictions are computed from the model.
    """
    alias_map = {"patient": "patient_id", "forecast": "prediction",
                 "rebuild": "history_rule", "n_perm": "n_shuffles", "chance": "baseline"}
    values = {"patient_id": patient_id, "prediction": prediction,
              "history_rule": history_rule, "n_shuffles": n_shuffles,
              "baseline": baseline}
    for old, new in alias_map.items():
        if old in legacy:
            if values[new] is not _UNSET:
                raise TypeError(f"evaluate() got both '{new}' and legacy keyword '{old}'")
            values[new] = legacy.pop(old)
    if legacy:
        name = next(iter(legacy))
        raise TypeError(f"evaluate() got an unexpected keyword argument '{name}'")
    patient_id = "patient_id" if values["patient_id"] is _UNSET else values["patient_id"]
    prediction = "prediction" if values["prediction"] is _UNSET else values["prediction"]
    history_rule = None if values["history_rule"] is _UNSET else values["history_rule"]
    n_shuffles = 200 if values["n_shuffles"] is _UNSET else values["n_shuffles"]
    baseline = "mean" if values["baseline"] is _UNSET else values["baseline"]
    if model is not None:
        if history_rule is not None:
            raise ValueError("give either model (with features and history) or history_rule, not both")
        if values["prediction"] is not _UNSET:
            raise ValueError("with model, leave prediction out: predictions are computed from the model")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        for col in [patient_id, outcome] + ([time] if time is not None else []):
            if col not in data:
                raise ValueError(f"missing dataframe columns: {col}")
        rule, fixed = _model_rule(data, patient_id, time, outcome, model, features, history)
        if fixed is not None:
            prediction = "__wheneval_prediction"
            data = data.assign(**{prediction: fixed})
        else:
            history_rule, prediction = rule, None
    elif features is not None or history is not None:
        raise ValueError("features and history need a model")

    if baseline not in {"mean", "median"}:
        raise ValueError("baseline must be 'mean' or 'median'")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if int(n_shuffles) != n_shuffles or n_shuffles < 1:
        raise ValueError("n_shuffles must be a positive integer")
    if int(seed) != seed:
        raise ValueError("seed must be an integer")
    if runin is not None and time is None:
        raise ValueError("time is required when runin is given")
    history_spec = _spec(history_rule, "history_rule")
    if prediction is None and history_spec is None:
        raise ValueError("prediction=None requires a history_rule")
    if history_spec is not None and history_spec["name"] == "window_count":
        if window is not None:
            if "w" in history_spec:
                try:
                    same_width = float(history_spec["w"]) == float(window)
                except (TypeError, ValueError):
                    same_width = False
                if not same_width:
                    raise ValueError("window and history_rule window_count width w conflict")
            history_spec["w"] = window
        if "w" not in history_spec:
            raise ValueError("window_count requires a width through window or history_rule['w']")
    columns = [patient_id, outcome] + ([prediction] if prediction is not None else []) + ([time] if time is not None else [])
    missing = [col for col in columns if col not in data]
    if missing:
        raise ValueError(f"missing dataframe columns: {', '.join(missing)}")
    pid = data[patient_id].to_numpy()
    _check_missing(patient_id, pid)
    y_all = _outcomes(data[outcome].to_numpy())
    p_all = np.full(len(y_all), np.nan, dtype=float) if prediction is None else np.asarray(data[prediction], dtype=float)
    if p_all.ndim != 1 or len(p_all) != len(y_all):
        raise ValueError("prediction and outcome must have equal length")
    time_all = None if time is None else data[time].to_numpy()
    if time_all is not None:
        _check_missing(time, time_all)
    score_all = np.ones(len(y_all), dtype=bool)
    if runin is not None:
        try:
            score_all = np.asarray(time_all >= runin, dtype=bool)
        except TypeError as exc:
            raise ValueError("runin must be comparable with the time column") from exc
    if not score_all.any():
        raise ValueError("no scored rows remain after runin")
    if threshold is None:
        # Paper convention: the cohort's scored event rate.
        threshold = float(np.mean(y_all[score_all]))
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between 0 and 1")

    custom, names = _select_metrics(metrics)
    if history_spec is not None and time is None:
        raise ValueError("time is required to rebuild a history-based forecast")
    window_count_mode = (history_spec.get("as_rate", history_spec.get("mode", "rate") != "count")
                         if history_spec and history_spec["name"] == "window_count" else True)
    count_forecast = bool(history_spec and history_spec["name"] == "window_count" and
                          (window_count_mode is False or str(window_count_mode).lower() == "count"))
    if count_forecast and any(name in PROBABILITY_METRICS for name in names):
        raise ValueError("window_count count mode supports AUC, average_precision, and user metrics; use as_rate=True for probability metrics")
    if prediction is not None and not count_forecast:
        _probabilities("prediction", p_all[score_all])

    yardstick_value = yardstick
    static_all: Optional[np.ndarray] = None
    if isinstance(yardstick, str) and yardstick not in {"patient_rate", "rate_so_far", "running_rate", "window_count"}:
        if yardstick not in data:
            raise ValueError(f"yardstick column {yardstick!r} is missing")
        static_all = data[yardstick].to_numpy(dtype=float)
        yardstick_value = static_all
    elif yardstick is not None and not callable(yardstick) and not isinstance(yardstick, (str, dict)):
        static_all = np.asarray(yardstick, dtype=float)
        if static_all.ndim == 0:
            static_all = np.full(len(data), float(static_all))
        if static_all.ndim != 1 or len(static_all) != len(data):
            raise ValueError("yardstick vector must have one value per data row")

    # Patient records are sorted stably by time, while all reported rows retain their input labels.
    groups: list[tuple[Any, np.ndarray, np.ndarray, np.ndarray]] = []
    for key in pd.unique(pid):
        ix = np.flatnonzero(pid == key)
        if time_all is not None:
            ix = ix[np.argsort(time_all[ix], kind="stable")]
        local_score = score_all[ix]
        stops = _stops(None if time_all is None else time_all[ix], len(ix))
        groups.append((key, ix, local_score, stops))

    refs_all = np.full(len(y_all), np.nan, dtype=float)
    for key, ix, local_score, stops in groups:
        frame = data.iloc[ix].copy()
        local_y = y_all[ix].astype(int)
        static_local = None if static_all is None else static_all[ix]
        reference = _reference_rows(yardstick_value, frame, local_y, local_score,
                                    stops, outcome, prediction or "", static=static_local)
        if len(reference) != len(ix):
            raise ValueError(f"yardstick for patient {key!r} must align with the patient's rows")
        if not np.isfinite(reference[local_score]).all() or ((reference[local_score] < 0) | (reference[local_score] > 1)).any():
            raise ValueError("yardstick must be finite probabilities in [0, 1]")
        refs_all[ix[local_score]] = reference[local_score]
        if history_spec is not None:
            check_all = _history_forecast(history_spec, frame, local_y, local_score,
                                          stops, outcome, prediction or "")
            check = check_all[local_score]
            if prediction is None:
                p_all[ix] = check_all
            elif not np.isfinite(check).all() or not np.allclose(check, p_all[ix][local_score], rtol=0, atol=1e-7):
                raise ValueError(f"history_rule predictions for patient {key!r} do not match prediction on scored rows")

    if not np.isfinite(p_all[score_all]).all():
        raise ValueError("prediction must be finite on scored rows")
    if not count_forecast:
        _probabilities("prediction", p_all[score_all])

    p_s, y_s, pid_s = p_all[score_all], y_all[score_all], pid[score_all]
    if not count_forecast:
        observed_nine = nine_metrics(pid_s, y_s, p_s, threshold, yardstick=refs_all[score_all])
        observed_naive = naive_metrics(pid_s, y_s, p_s, threshold, population_rate=float(np.mean(y_s)))
    else:
        observed_nine, observed_naive = {}, {}
        for name in ("auc", "average_precision"):
            pooled_value = auc(y_s, p_s) if name == "auc" else average_precision(y_s, p_s)
            observed_nine[name] = {"pooled": pooled_value}

    output: dict[str, dict[str, Any]] = {}
    slope_null = "calibration_slope" in names and not count_forecast
    null_pid: list[np.ndarray] = []
    null_y: list[np.ndarray] = []
    null_p: list[np.ndarray] = []
    for group_no, (key, ix, local_score, stops) in enumerate(groups):
        if not local_score.any():
            continue
        frame = data.iloc[ix].copy()
        local_y = y_all[ix].astype(int)
        obs_y = local_y[local_score]
        obs_p = p_all[ix][local_score]
        n_scored = len(obs_y)
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), group_no]))
        y_draws = np.asarray([obs_y[rng.permutation(n_scored)] for _ in range(n_shuffles)], dtype=int)
        full_draws = np.repeat(local_y[None, :], n_shuffles, axis=0)
        full_draws[:, local_score] = y_draws
        if history_spec is None:
            p_draws = np.repeat(obs_p[None, :], n_shuffles, axis=0)
        elif history_spec["name"] in {"running_rate", "window_count"}:
            all_p = (_running_rate_batch(full_draws, stops, history_spec)
                     if history_spec["name"] == "running_rate"
                     else _window_count_batch(full_draws, stops, history_spec))
            p_draws = all_p[:, local_score]
        else:
            p_draws = np.empty((n_shuffles, n_scored), dtype=float)
            for b in range(n_shuffles):
                p_draws[b] = _history_forecast(history_spec, frame, full_draws[b],
                                               local_score, stops, outcome, prediction or "")[local_score]
        static_local = None if static_all is None else static_all[ix]
        if yardstick_value is None or (isinstance(yardstick_value, str) and yardstick_value == "patient_rate"):
            rate = y_draws.mean(axis=1)
            ref_draws = np.repeat(rate[:, None], n_scored, axis=1)
        elif isinstance(yardstick_value, (str, dict)):
            ref_spec = _spec(yardstick_value, "yardstick")
            if ref_spec["name"] == "running_rate":
                ref_draws = _running_rate_batch(full_draws, stops, ref_spec)[:, local_score]
            elif ref_spec["name"] == "window_count":
                ref_draws = _window_count_batch(full_draws, stops, ref_spec)[:, local_score]
            else:
                raise ValueError("yardstick may rebuild running_rate or window_count references")
        elif callable(yardstick_value):
            ref_draws = np.asarray([
                _reference_rows(yardstick_value, frame, full_draws[b], local_score,
                                stops, outcome, prediction or "")[local_score]
                for b in range(n_shuffles)
            ], dtype=float)
        else:
            local_ref = np.asarray(static_local, dtype=float)
            ref_draws = np.repeat(local_ref[local_score][None, :], n_shuffles, axis=0)
        if not count_forecast:
            _probabilities("rebuilt prediction", p_draws.ravel())
        _probabilities("rebuilt yardstick", ref_draws.ravel())
        if slope_null:
            null_pid.append(np.repeat(np.asarray([group_no]), n_scored))
            null_y.append(y_draws)
            null_p.append(p_draws)
        built = _builtins_batch(y_draws, p_draws, ref_draws, threshold,
                                "calibration_slope" in names)
        user_null = {name: _custom_batch(fn, y_draws, p_draws)
                     for name, fn in custom.items() if name in names}

        observed_values: dict[str, float] = {}
        for name in names:
            if name in custom:
                observed_values[name] = float(custom[name](obs_y.copy(), obs_p.copy()))
            elif name == "calibration_slope":
                observed_values[name] = _slope(obs_y, obs_p)
            elif name == "average_precision":
                # AP needs at least one positive, but remains defined for all-positive records.
                observed_values[name] = float(average_precision(obs_y, obs_p))
            elif count_forecast:
                observed_values[name] = float(auc(obs_y, obs_p) if name == "auc"
                                              else average_precision(obs_y, obs_p))
            else:
                observed_values[name] = float(observed_nine[name]["per_patient"].get(key, np.nan))
        for name in names:
            value = observed_values[name]
            null = built[name] if name in built else user_null[name]
            finite = np.isfinite(null)
            if not np.isfinite(value) or not finite.any():
                continue
            shuffled_baseline = float(np.mean(null[finite]) if baseline == "mean" else np.median(null[finite]))
            # Denominator: the shuffles on which the measure is defined (B_D), the same shuffles
            # the baseline averages (Proposition 2).
            p_value = float((int(np.sum(null[finite] >= value)) + 1) / (int(finite.sum()) + 1))
            target = output.setdefault(name, {"per_patient": {}, "comparison": {}})
            target["per_patient"][key] = {
                "score": float(value), "shuffled_baseline": shuffled_baseline,
                "excess": float(value - shuffled_baseline), "p_value": p_value,
                "n_scored": n_scored,
            }

    for name in names:
        target = output.setdefault(name, {"per_patient": {}, "comparison": {}})
        per = list(target["per_patient"].values())
        excesses = np.asarray([entry["excess"] for entry in per], dtype=float)
        if len(excesses):
            q25, q75 = np.percentile(excesses, [25, 75])
            target["summary"] = {
                "median_excess": float(np.median(excesses)),
                "iqr_excess": [float(q25), float(q75)],
                "share_p_below_0_05": float(np.mean([entry["p_value"] < 0.05 for entry in per])),
                "share_excess_positive": float(np.mean(excesses > SIGN_TOL)),
                "mean_score": float(np.mean([entry["score"] for entry in per])),
                "mean_shuffled_baseline": float(np.mean([entry["shuffled_baseline"] for entry in per])),
                "mean_excess": float(np.mean(excesses)),
                "n_eligible": len(excesses),
            }
        else:
            target["summary"] = {"median_excess": float("nan"), "iqr_excess": [float("nan"), float("nan")],
                                  "share_p_below_0_05": float("nan"), "share_excess_positive": float("nan"),
                                  "mean_score": float("nan"), "mean_shuffled_baseline": float("nan"), "mean_excess": float("nan"),
                                  "n_eligible": 0}
        if name in observed_nine and name in observed_naive:
            naive = observed_naive[name]["summary"]
            target["comparison"] = {
                "pooled": observed_nine[name].get("pooled", float("nan")),
                "per_patient_mean": naive.get("mean", float("nan")),
                "per_patient_median": naive.get("median", float("nan")),
                "per_patient_iqr": naive.get("iqr", [float("nan"), float("nan")]),
                "per_patient_n": naive.get("n", 0),
            }
            if name == "brier_skill":
                # The usual pooled reference is the cohort's scored event rate, not the patient yardstick.
                cohort_loss = float(np.mean((y_s - np.mean(y_s)) ** 2))
                target["comparison"]["pooled"] = (float(1 - np.mean((y_s - p_s) ** 2) / cohort_loss)
                                                  if cohort_loss > 0 else float("nan"))
            if name == "calibration_slope":
                # Section 3 definition: one conditional slope with patient intercepts.
                observed_slope = observed_nine[name]["summary"].get("mean", float("nan"))
                target["comparison"]["conditional_within_slope"] = observed_slope
                if slope_null and null_y:
                    # The same shuffles, pooled across patients into one conditional fit per shuffle.
                    all_pid = np.concatenate(null_pid)
                    nulls = []
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        for b in range(int(n_shuffles)):
                            fit = conditional_within_slope(all_pid, np.concatenate([d[b] for d in null_y]),
                                                           np.concatenate([d[b] for d in null_p]))
                            nulls.append(fit["slope"])
                    nulls = np.asarray(nulls, dtype=float)
                    ok = np.isfinite(nulls)
                    if ok.any() and np.isfinite(observed_slope):
                        level = float(np.mean(nulls[ok]) if baseline == "mean" else np.median(nulls[ok]))
                        target["comparison"]["conditional_within_slope_shuffled_baseline"] = level
                        target["comparison"]["conditional_within_slope_excess"] = float(observed_slope - level)
                        target["comparison"]["conditional_within_slope_p"] = float(
                            (int(np.sum(nulls[ok] >= observed_slope)) + 1) / (int(ok.sum()) + 1))
        else:
            vals = [entry["score"] for entry in per]
            if name in custom:
                pooled = float(custom[name](y_s.copy(), p_s.copy()))
            elif name == "auc":
                pooled = float(auc(y_s, p_s))
            else:
                pooled = float(average_precision(y_s, p_s))
            target["comparison"] = {
                "pooled": pooled,
                "per_patient_mean": float(np.mean(vals)) if vals else float("nan"),
                "per_patient_median": float(np.median(vals)) if vals else float("nan"),
                "per_patient_iqr": [float(x) for x in np.percentile(vals, [25, 75])] if vals else [float("nan"), float("nan")],
                "per_patient_n": len(vals),
            }

    if yardstick is None:
        yardstick_label = "patient_rate"
    elif isinstance(yardstick, str):
        yardstick_label = yardstick
    elif isinstance(yardstick, dict):
        yardstick_label = str(yardstick.get("name", yardstick.get("method", "custom")))
    elif callable(yardstick):
        yardstick_label = "callable"
    else:
        yardstick_label = "row_aligned"
    output = {name: output[name] for name in names if name in output}
    return {"metrics": output,
            "settings": {"n_shuffles": int(n_shuffles), "seed": int(seed), "threshold": threshold,
                         "runin": runin, "history_rule": history_spec["name"] if history_spec else None, "baseline": baseline,
                         "yardstick": yardstick_label,
                         "scored_rows": int(score_all.sum()), "patients": len(groups)}}


# Keep the public signature readable while internal sentinels distinguish an
# omitted new keyword from one explicitly supplied alongside its legacy alias.
evaluate.__signature__ = inspect.Signature([
    inspect.Parameter("data", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    inspect.Parameter("patient_id", inspect.Parameter.POSITIONAL_OR_KEYWORD, default="patient_id"),
    inspect.Parameter("time", inspect.Parameter.POSITIONAL_OR_KEYWORD, default="time"),
    inspect.Parameter("outcome", inspect.Parameter.POSITIONAL_OR_KEYWORD, default="outcome"),
    inspect.Parameter("prediction", inspect.Parameter.POSITIONAL_OR_KEYWORD, default="prediction"),
    inspect.Parameter("history_rule", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None),
    inspect.Parameter("window", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None),
    inspect.Parameter("runin", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None),
    inspect.Parameter("threshold", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None),
    inspect.Parameter("metrics", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None),
    inspect.Parameter("n_shuffles", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=200),
    inspect.Parameter("seed", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=0),
    inspect.Parameter("yardstick", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None),
    inspect.Parameter("baseline", inspect.Parameter.POSITIONAL_OR_KEYWORD, default="mean"),
    inspect.Parameter("legacy", inspect.Parameter.VAR_KEYWORD),
])


def to_markdown(result: dict[str, Any]) -> str:
    """Render patient scores and shuffled baselines, followed by comparisons."""
    if "metrics" not in result:
        from .report import to_markdown_v1
        return to_markdown_v1(result)

    def fmt(value: Any) -> str:
        return f"{value:.3f}" if isinstance(value, (float, np.floating)) and np.isfinite(value) else str(value)

    headers = ("Metric", "Score (mean)", "Shuffled baseline (mean)", "Excess (mean)", "Excess (median [IQR])",
               "Excess > 0", "p < 0.05", "Patients", "Pooled score (for comparison)",
               "Per-patient score (mean; for comparison)")
    lines = ["| " + " | ".join(headers) + " |", "|---|" + "|".join(["---:"] * (len(headers) - 1)) + "|"]
    for name, metric in result["metrics"].items():
        summary, comparison = metric["summary"], metric["comparison"]
        q25, q75 = summary["iqr_excess"]
        row = [name, fmt(summary.get("mean_score", float("nan"))),
               fmt(summary.get("mean_shuffled_baseline", float("nan"))),
               fmt(summary.get("mean_excess", float("nan"))),
               f"{fmt(summary['median_excess'])} [{fmt(q25)}, {fmt(q75)}]",
               f"{summary['share_excess_positive']:.0%}" if np.isfinite(summary["share_excess_positive"]) else "nan",
               f"{summary['share_p_below_0_05']:.0%}" if np.isfinite(summary["share_p_below_0_05"]) else "nan",
               str(summary["n_eligible"]), fmt(comparison.get("pooled", float("nan"))),
               fmt(comparison.get("per_patient_mean", float("nan")))]
        if name == "calibration_slope" and "conditional_within_slope_shuffled_baseline" in comparison:
            row = ["Calibration slope (conditional, all patients)", fmt(comparison["conditional_within_slope"]),
                   fmt(comparison["conditional_within_slope_shuffled_baseline"]),
                   f"{fmt(comparison['conditional_within_slope_excess'])} (p = {fmt(comparison['conditional_within_slope_p'])})",
                   "", "", "", str(summary["n_eligible"]), fmt(comparison.get("pooled", float("nan"))), ""]
        lines.append("| " + " | ".join(row) + " |")
    settings = result.get("settings", {})
    lines.append("")
    lines.append(f"Threshold {settings.get('threshold', float('nan')):.3f}; {settings.get('n_shuffles')} within-patient "
                 f"shuffles per patient (seed {settings.get('seed')}); history rule: {settings.get('history_rule')}; "
                 f"yardstick: {settings.get('yardstick')}.")
    return "\n".join(lines)
