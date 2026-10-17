"""History features: inputs computed from a patient's earlier outcomes.

A model's inputs that are computed from the outcome (the last outcome, a rate over
recent occasions, time since the last event) must be recomputed from the shuffled
outcomes on every shuffle; otherwise the shuffled baseline misses the finite-record
bias. The recipes below name the common ones, so that ``evaluate(..., history=...)``
can rebuild them. Each recipe works on one patient's rows in time order and uses
only rows strictly before the current row (earlier occasions; run-in rows count as
history). Rows sharing a time stamp do not see each other.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


class HistoryFeature:
    """A feature computed from a patient's earlier outcomes."""

    def __init__(self, label: str, call: str, fill: float = np.nan):
        self.label = label
        self.call = call
        self.fill = float(fill)

    def __repr__(self) -> str:
        return self.call

    def compute(self, y: np.ndarray, stops: np.ndarray, times: Optional[np.ndarray]) -> np.ndarray:
        raise NotImplementedError


def _prefix(y: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.asarray(y, dtype=float))])


class _Lag(HistoryFeature):
    def __init__(self, k: int, fill: float):
        super().__init__(f"outcome {k} occasion{'s' if k > 1 else ''} earlier",
                         f"we.lag({k})" if np.isnan(fill) else f"we.lag({k}, fill={fill:g})", fill)
        self.k = k

    def compute(self, y, stops, times):
        pos = stops - self.k
        out = np.full(len(y), self.fill, dtype=float)
        ok = pos >= 0
        out[ok] = np.asarray(y, dtype=float)[pos[ok]]
        return out


class _Window(HistoryFeature):
    def __init__(self, w: Optional[int], kind: str, fill: float):
        what = "all earlier occasions" if w is None else f"the last {w} occasions"
        name = {"rate": "window_rate", "count": "window_count"}[kind] if w is not None else "rate_so_far"
        arg = "" if w is None else str(w)
        call = f"we.{name}({arg})" if np.isnan(fill) else f"we.{name}({arg}{', ' if arg else ''}fill={fill:g})"
        super().__init__(f"event {kind} over {what}", call, fill)
        self.w, self.kind = w, kind

    def compute(self, y, stops, times):
        cum = _prefix(y)
        starts = np.zeros_like(stops) if self.w is None else np.maximum(stops - self.w, 0)
        events = cum[stops] - cum[starts]
        n = (stops - starts).astype(float)
        if self.kind == "count":
            out = events.copy()
            if self.w is None:
                return out
            out[n == 0] = self.fill
            return out
        out = np.full(len(y), self.fill, dtype=float)
        np.divide(events, n, out=out, where=n > 0)
        return out


class _TimeSince(HistoryFeature):
    def __init__(self, fill: float):
        super().__init__("time since the last earlier event",
                         "we.time_since_event()" if np.isnan(fill) else f"we.time_since_event(fill={fill:g})", fill)

    def compute(self, y, stops, times):
        if times is None:
            raise ValueError("time_since_event needs the time column")
        t = np.asarray(times, dtype=float)
        y = np.asarray(y)
        out = np.full(len(y), self.fill, dtype=float)
        # rows are in time order; stops[i] is the first row with time >= time[i]
        event_rows = np.flatnonzero(y == 1)
        for i in range(len(y)):
            earlier = event_rows[event_rows < stops[i]]
            if len(earlier):
                out[i] = t[i] - t[earlier[-1]]
        return out


def _check_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def lag(k: int = 1, fill: float = np.nan) -> HistoryFeature:
    """The outcome ``k`` occasions earlier (``fill`` where there is none)."""
    return _Lag(_check_int(k, "lag k"), fill)


def rate_so_far(fill: float = np.nan) -> HistoryFeature:
    """Event rate over all earlier occasions, run-in included (``fill`` on the first)."""
    return _Window(None, "rate", fill)


def window_rate(w: int, fill: float = np.nan) -> HistoryFeature:
    """Event rate over the last ``w`` earlier occasions (fewer at the start)."""
    return _Window(_check_int(w, "window_rate w"), "rate", fill)


def window_count(w: int, fill: float = np.nan) -> HistoryFeature:
    """Number of events over the last ``w`` earlier occasions."""
    return _Window(_check_int(w, "window_count w"), "count", fill)


def time_since_event(fill: float = np.nan) -> HistoryFeature:
    """Time (in units of the time column) since the last earlier event."""
    return _TimeSince(fill)


def _candidates() -> list[HistoryFeature]:
    """Recipes tried when looking for undeclared outcome-derived columns."""
    out: list[HistoryFeature] = [lag(k) for k in (1, 2, 3)] + [rate_so_far()]
    for w in (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 21, 28, 30, 60, 90):
        out += [window_rate(w), window_count(w)]
    return out + [time_since_event()]


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    """Equal where both are defined, with at least a few defined values that vary."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    both = np.isfinite(a) & np.isfinite(b)
    if both.sum() < 3 or np.ptp(b[both]) == 0:
        return False
    return bool(np.allclose(a[both], b[both], rtol=0, atol=1e-7))
