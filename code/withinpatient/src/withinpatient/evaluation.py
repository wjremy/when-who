"""Paper Section 3: patient-wise scoring against rebuilt permutation chance."""
from __future__ import annotations

import inspect
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd

from .metrics import _check_missing, _outcomes, _probabilities, auc, conditional_within_slope
from .nine import average_precision, naive_metrics, nine_metrics

METRICS = ("auc", "average_precision", "sensitivity", "specificity", "ppv", "npv",
           "net_benefit", "brier_skill", "calibration_slope")
PROBABILITY_METRICS = {"sensitivity", "specificity", "ppv", "npv", "net_benefit",
                       "brier_skill", "calibration_slope"}
ALIASES = {"ap": "average_precision", **{name: name for name in METRICS}}


# Excesses within this distance of zero count as zero when a sign is taken (floating-point noise).
SIGN_TOL = 1e-9

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
    if name not in {"running_rate", "window_count", "lagged_outcomes"}:
        raise ValueError(f"{label} must name running_rate, window_count, or lagged_outcomes")
    spec["name"] = name
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


def evaluate(data: pd.DataFrame, patient: str = "patient", time: Optional[str] = "time",
             outcome: str = "y", forecast: str = "forecast", runin: Optional[float] = None,
             threshold: Optional[float] = None, rebuild: Any = None, metrics: Any = None,
             n_perm: int = 200, seed: int = 0, yardstick: Any = None,
             chance: str = "mean") -> dict[str, Any]:
    """Apply the Section 3 fix to a long table of patient-time forecasts.

    ``runin`` is a time cutoff: earlier outcomes remain in history but are not
    scored. ``rebuild`` is ``None`` (fixed forecast), ``"running_rate"``, a
    mapping such as ``{"name": "window_count", "w": 7, "as_rate": True}``,
    ``{"name": "lagged_outcomes", "lags": L, "predict": fixed_model_fn}``,
    or a function f(y), f(y, scored) or f(y, scored, rows) taking one patient's
    time-ordered outcomes (plus the scored mask and that patient's data rows,
    e.g. for a fitted model with covariates). Rebuilt forecasts
    must reproduce the forecast column for the observed scored rows. Lagged
    prediction receives all non-outcome, non-forecast columns plus lag columns.

    ``yardstick=None`` uses that patient's scored event rate, recalculated in
    every shuffle. A column name/vector is fixed; a function or a ``running_rate``
    mapping is rebuilt. ``metrics`` defaults to the nine paper metrics and also
    accepts a list of their names or a mapping ``name -> f(y, p)``. ``n_perm``
    defaults to 200. Patient-specific random streams derive from ``seed`` and
    are stable if evaluation is later parallelized. ``chance`` is ``"mean"``
    (default) or ``"median"`` of each patient's shuffle distribution.
    """
    if chance not in {"mean", "median"}:
        raise ValueError("chance must be 'mean' or 'median'")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if int(n_perm) != n_perm or n_perm < 1:
        raise ValueError("n_perm must be a positive integer")
    if int(seed) != seed:
        raise ValueError("seed must be an integer")
    if runin is not None and time is None:
        raise ValueError("time is required when runin is given")
    columns = [patient, outcome, forecast] + ([time] if time is not None else [])
    missing = [col for col in columns if col not in data]
    if missing:
        raise ValueError(f"missing dataframe columns: {', '.join(missing)}")
    pid = data[patient].to_numpy()
    _check_missing(patient, pid)
    y_all = _outcomes(data[outcome].to_numpy())
    p_all = np.asarray(data[forecast], dtype=float)
    if p_all.ndim != 1 or len(p_all) != len(y_all):
        raise ValueError("forecast and outcome must have equal length")
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
    if not np.isfinite(p_all[score_all]).all():
        raise ValueError("forecast must be finite on scored rows")
    if threshold is None:
        # Paper convention: the cohort's scored event rate.
        threshold = float(np.mean(y_all[score_all]))
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between 0 and 1")

    custom, names = _select_metrics(metrics)
    rebuild_spec = _spec(rebuild, "rebuild")
    if rebuild_spec is not None and time is None:
        raise ValueError("time is required to rebuild a history-based forecast")
    window_count_mode = (rebuild_spec.get("as_rate", rebuild_spec.get("mode", "rate") != "count")
                         if rebuild_spec and rebuild_spec["name"] == "window_count" else True)
    count_forecast = bool(rebuild_spec and rebuild_spec["name"] == "window_count" and
                          (window_count_mode is False or str(window_count_mode).lower() == "count"))
    if count_forecast and any(name in PROBABILITY_METRICS for name in names):
        raise ValueError("window_count count mode supports AUC, average_precision, and user metrics; use as_rate=True for probability metrics")
    if not count_forecast:
        _probabilities("forecast", p_all[score_all])

    yardstick_value = yardstick
    static_all: Optional[np.ndarray] = None
    if isinstance(yardstick, str) and yardstick not in {"patient_rate", "running_rate", "window_count"}:
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
                                    stops, outcome, forecast, static=static_local)
        if len(reference) != len(ix):
            raise ValueError(f"yardstick for patient {key!r} must align with the patient's rows")
        if not np.isfinite(reference[local_score]).all() or ((reference[local_score] < 0) | (reference[local_score] > 1)).any():
            raise ValueError("yardstick must be finite probabilities in [0, 1]")
        refs_all[ix[local_score]] = reference[local_score]
        if rebuild_spec is not None:
            check = _history_forecast(rebuild_spec, frame, local_y, local_score,
                                      stops, outcome, forecast)[local_score]
            if not np.isfinite(check).all() or not np.allclose(check, p_all[ix][local_score], rtol=0, atol=1e-7):
                raise ValueError(f"rebuilt forecast for patient {key!r} does not match forecast on scored rows")

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
        y_draws = np.asarray([obs_y[rng.permutation(n_scored)] for _ in range(n_perm)], dtype=int)
        full_draws = np.repeat(local_y[None, :], n_perm, axis=0)
        full_draws[:, local_score] = y_draws
        if rebuild_spec is None:
            p_draws = np.repeat(obs_p[None, :], n_perm, axis=0)
        elif rebuild_spec["name"] in {"running_rate", "window_count"}:
            all_p = (_running_rate_batch(full_draws, stops, rebuild_spec)
                     if rebuild_spec["name"] == "running_rate"
                     else _window_count_batch(full_draws, stops, rebuild_spec))
            p_draws = all_p[:, local_score]
        else:
            p_draws = np.empty((n_perm, n_scored), dtype=float)
            for b in range(n_perm):
                p_draws[b] = _history_forecast(rebuild_spec, frame, full_draws[b],
                                               local_score, stops, outcome, forecast)[local_score]
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
                                stops, outcome, forecast)[local_score]
                for b in range(n_perm)
            ], dtype=float)
        else:
            local_ref = np.asarray(static_local, dtype=float)
            ref_draws = np.repeat(local_ref[local_score][None, :], n_perm, axis=0)
        if not count_forecast:
            _probabilities("rebuilt forecast", p_draws.ravel())
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
            chance_level = float(np.mean(null[finite]) if chance == "mean" else np.median(null[finite]))
            p_value = float((int(np.sum(null[finite] >= value)) + 1) / (n_perm + 1))
            target = output.setdefault(name, {"per_patient": {}, "comparison": {}})
            target["per_patient"][key] = {
                "observed": float(value), "chance_level": chance_level,
                "excess": float(value - chance_level), "p_value": p_value,
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
                "mean_observed": float(np.mean([entry["observed"] for entry in per])),
                "mean_chance_level": float(np.mean([entry["chance_level"] for entry in per])),
                "mean_excess": float(np.mean(excesses)),
                "n_eligible": len(excesses),
            }
        else:
            target["summary"] = {"median_excess": float("nan"), "iqr_excess": [float("nan"), float("nan")],
                                  "share_p_below_0_05": float("nan"), "share_excess_positive": float("nan"),
                                  "mean_observed": float("nan"), "mean_chance_level": float("nan"), "mean_excess": float("nan"),
                                  "n_eligible": 0}
        if name in observed_nine and name in observed_naive:
            naive = observed_naive[name]["summary"]
            target["comparison"] = {
                "pooled": observed_nine[name].get("pooled", float("nan")),
                "usual_per_patient_mean": naive.get("mean", float("nan")),
                "usual_per_patient_median": naive.get("median", float("nan")),
                "usual_per_patient_iqr": naive.get("iqr", [float("nan"), float("nan")]),
                "usual_per_patient_n": naive.get("n", 0),
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
                        for b in range(int(n_perm)):
                            fit = conditional_within_slope(all_pid, np.concatenate([d[b] for d in null_y]),
                                                           np.concatenate([d[b] for d in null_p]))
                            nulls.append(fit["slope"])
                    nulls = np.asarray(nulls, dtype=float)
                    ok = np.isfinite(nulls)
                    if ok.any() and np.isfinite(observed_slope):
                        level = float(np.mean(nulls[ok]) if chance == "mean" else np.median(nulls[ok]))
                        target["comparison"]["conditional_within_slope_chance"] = level
                        target["comparison"]["conditional_within_slope_excess"] = float(observed_slope - level)
                        target["comparison"]["conditional_within_slope_p"] = float(
                            (int(np.sum(nulls[ok] >= observed_slope)) + 1) / (int(n_perm) + 1))
        else:
            vals = [entry["observed"] for entry in per]
            if name in custom:
                pooled = float(custom[name](y_s.copy(), p_s.copy()))
            elif name == "auc":
                pooled = float(auc(y_s, p_s))
            else:
                pooled = float(average_precision(y_s, p_s))
            target["comparison"] = {
                "pooled": pooled,
                "usual_per_patient_mean": float(np.mean(vals)) if vals else float("nan"),
                "usual_per_patient_median": float(np.median(vals)) if vals else float("nan"),
                "usual_per_patient_iqr": [float(x) for x in np.percentile(vals, [25, 75])] if vals else [float("nan"), float("nan")],
                "usual_per_patient_n": len(vals),
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
            "settings": {"n_perm": int(n_perm), "seed": int(seed), "threshold": threshold,
                         "runin": runin, "rebuild": rebuild_spec["name"] if rebuild_spec else None, "chance": chance,
                         "yardstick": yardstick_label,
                         "scored_rows": int(score_all.sum()), "patients": len(groups)}}


def to_markdown(result: dict[str, Any]) -> str:
    """Render tailored results first (observed against rebuilt chance), then the comparisons."""
    if "metrics" not in result:
        from .report import to_markdown_v1
        return to_markdown_v1(result)

    def fmt(value: Any) -> str:
        return f"{value:.3f}" if isinstance(value, (float, np.floating)) and np.isfinite(value) else str(value)

    headers = ("Metric", "Mean observed", "Mean chance", "Median excess [IQR]",
               "Excess > 0", "p < 0.05", "Patients", "Pooled", "Usual per-patient mean")
    lines = ["| " + " | ".join(headers) + " |", "|---|" + "|".join(["---:"] * (len(headers) - 1)) + "|"]
    for name, metric in result["metrics"].items():
        summary, comparison = metric["summary"], metric["comparison"]
        q25, q75 = summary["iqr_excess"]
        row = [name, fmt(summary.get("mean_observed", float("nan"))),
               fmt(summary.get("mean_chance_level", float("nan"))),
               f"{fmt(summary['median_excess'])} [{fmt(q25)}, {fmt(q75)}]",
               f"{summary['share_excess_positive']:.0%}" if np.isfinite(summary["share_excess_positive"]) else "nan",
               f"{summary['share_p_below_0_05']:.0%}" if np.isfinite(summary["share_p_below_0_05"]) else "nan",
               str(summary["n_eligible"]), fmt(comparison.get("pooled", float("nan"))),
               fmt(comparison.get("usual_per_patient_mean", float("nan")))]
        if name == "calibration_slope" and "conditional_within_slope_chance" in comparison:
            # Calibration summary: one conditional within-patient slope against its own shuffle chance level.
            row = ["calibration_slope (conditional, all patients)", fmt(comparison["conditional_within_slope"]),
                   fmt(comparison["conditional_within_slope_chance"]),
                   f"{fmt(comparison['conditional_within_slope_excess'])} (p = {fmt(comparison['conditional_within_slope_p'])})",
                   "", "", str(summary["n_eligible"]), fmt(comparison.get("pooled", float("nan"))), ""]
        lines.append("| " + " | ".join(row) + " |")
    settings = result.get("settings", {})
    lines.append("")
    lines.append(f"Threshold {settings.get('threshold', float('nan')):.3f}; {settings.get('n_perm')} within-patient "
                 f"shuffles per patient (seed {settings.get('seed')}); rebuild: {settings.get('rebuild')}; "
                 f"yardstick: {settings.get('yardstick')}.")
    return "\n".join(lines)
