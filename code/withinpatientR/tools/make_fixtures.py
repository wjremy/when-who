"""Create cross-language fixtures from the source Python withinpatient package."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "code" / "withinpatient" / "src"))

from withinpatient.metrics import (  # noqa: E402
    auc,
    brier_scores,
    brier_skill,
    brier_skill_summary,
    expanding_patient_rate,
    patient_calibration_in_the_large,
    per_patient_auc_summary,
    per_time_point_auc,
    pooled_auc,
    population_a0_moment_corrected,
    rate_benchmark,
    runin_patient_rate,
    within_patient_auc,
)
from withinpatient.report import evaluate_v1 as python_evaluate_v1  # noqa: E402
from withinpatient.evaluation import evaluate as python_evaluate_v2  # noqa: E402
import withinpatient.evaluation as evaluation_module  # noqa: E402


def history_forecast(outcomes: np.ndarray) -> np.ndarray:
    """A simple expanding event-rate forecast using strictly earlier outcomes."""
    outcomes = np.asarray(outcomes, dtype=float)
    previous = np.r_[0.0, np.cumsum(outcomes[:-1])]
    return (0.25 + previous) / (1.0 + np.arange(len(outcomes)))


def flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    """Flatten nested results to stable CSV paths, preserving NaN explicitly."""
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_numeric(item, path))
    elif isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value)
        if array.ndim == 0:
            return flatten_numeric(array.item(), prefix)
        names = getattr(value, "keys", None)
        for i, item in enumerate(value):
            path = f"{prefix}[{i + 1}]"
            out.update(flatten_numeric(item, path))
    elif isinstance(value, (bool, int, float, np.number)):
        out[prefix] = float(value)
    return out


def csv_value(value: float) -> str:
    if np.isnan(value):
        return "NaN"
    if np.isposinf(value):
        return "Inf"
    if np.isneginf(value):
        return "-Inf"
    return format(float(value), ".17g")


class _IndexedRng:
    """Small generator adapter so Python consumes the CSV permutation maps."""

    def __init__(self, draws: list[np.ndarray]) -> None:
        self.draws = iter(draws)

    def permutation(self, n: int) -> np.ndarray:
        draw = next(self.draws)
        if len(draw) != n:
            raise ValueError("permutation fixture length does not match the scored patient rows")
        return draw.copy()


def evaluate_with_maps(data: pd.DataFrame, maps: list[dict[str, np.ndarray]], **kwargs: Any) -> dict[str, Any]:
    """Run v2 evaluate while its patient streams consume the shared CSV maps."""
    keys = list(pd.unique(data[kwargs.get("patient", "patient")]))
    original = evaluation_module.np.random.default_rng

    def indexed_default_rng(seed_sequence: Any) -> _IndexedRng:
        group_no = int(seed_sequence.entropy[1])
        key = str(keys[group_no])
        return _IndexedRng([draw[key] - 1 for draw in maps])

    evaluation_module.np.random.default_rng = indexed_default_rng
    try:
        return python_evaluate_v2(data, **kwargs)
    finally:
        evaluation_module.np.random.default_rng = original


def explicit_permutation_results(
    patient_scored: np.ndarray,
    y_scored: np.ndarray,
    forecast_scored: np.ndarray,
    patient_full: np.ndarray,
    y_full: np.ndarray,
    time_full: np.ndarray,
    runin_days: int,
    maps: list[dict[str, np.ndarray]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use explicit within-patient index maps for the package's two permutation definitions."""
    keys = list(pd.unique(patient_scored))
    groups = {key: np.flatnonzero(patient_scored == key) for key in keys}
    observed_auc = auc(y_scored, forecast_scored)
    null_auc: list[float] = []
    patient_null: dict[str, list[float]] = {
        key: [] for key, ix in groups.items() if np.isfinite(auc(y_scored[ix], forecast_scored[ix]))
    }
    for draw in maps:
        shuffled = forecast_scored.copy()
        for key, ix in groups.items():
            shuffled[ix] = forecast_scored[ix[draw[key] - 1]]
            if key in patient_null:
                patient_null[key].append(auc(y_scored[ix], shuffled[ix]))
        null_auc.append(auc(y_scored, shuffled))
    pooled_result: dict[str, Any] = {
        "observed": observed_auc,
        "null_mean": float(np.mean(null_auc)),
        "null_percentiles": np.percentile(null_auc, [2.5, 97.5]),
        "one_sided_p_value": (int(np.sum(np.asarray(null_auc) >= observed_auc)) + 1) / (len(null_auc) + 1),
        "permutations": len(null_auc),
        "per_patient": {},
    }
    for key, draws in patient_null.items():
        ix = groups[key]
        pooled_result["per_patient"][key] = {
            "observed": auc(y_scored[ix], forecast_scored[ix]),
            "reference_mean": float(np.mean(draws)),
            "reference_percentiles": np.percentile(draws, [2.5, 97.5]),
        }

    observed_by: dict[str, float] = {}
    runin_by: dict[str, int] = {}
    scored_by: dict[str, np.ndarray] = {}
    full_groups = {key: np.flatnonzero(patient_full == key) for key in pd.unique(patient_full)}
    for key, ix in full_groups.items():
        ix = ix[np.argsort(time_full[ix], kind="mergesort")]
        k = int(np.sum(time_full[ix] < runin_days))
        runin_by[key] = k
        scored_by[key] = y_full[ix][k:]
        score = auc(scored_by[key], history_forecast(np.r_[y_full[ix][:k], scored_by[key]])[k:])
        if np.isfinite(score):
            observed_by[key] = float(score)

    history_draws: list[float] = []
    patient_history: dict[str, list[float]] = {key: [] for key in observed_by}
    for draw in maps:
        draw_values: dict[str, float] = {}
        for key, ix0 in full_groups.items():
            ix = ix0[np.argsort(time_full[ix0], kind="mergesort")]
            k = runin_by[key]
            shuffled_y = scored_by[key][draw[key] - 1]
            full = np.r_[y_full[ix][:k], shuffled_y]
            pred = history_forecast(full)
            score = auc(shuffled_y, pred[k:])
            if np.isfinite(score):
                draw_values[key] = float(score)
        history_draws.append(float(np.mean(list(draw_values.values()))) if draw_values else float("nan"))
        for key in patient_history:
            patient_history[key].append(draw_values.get(key, float("nan")))
    finite_draws = np.asarray(history_draws, dtype=float)
    finite_draws = finite_draws[np.isfinite(finite_draws)]
    observed = float(np.mean(list(observed_by.values()))) if observed_by else float("nan")
    ref = float(np.mean(finite_draws)) if len(finite_draws) else float("nan")
    history_result: dict[str, Any] = {
        "observed": observed,
        "reference_mean": ref,
        "difference": observed - ref,
        "reference_percentiles": np.percentile(finite_draws, [2.5, 97.5]) if len(finite_draws) else [np.nan, np.nan],
        "permutations": len(maps),
        "per_patient": {},
    }
    for key, observed_value in observed_by.items():
        vals = np.asarray(patient_history[key], dtype=float)
        vals = vals[np.isfinite(vals)]
        mean = float(np.mean(vals)) if len(vals) else float("nan")
        history_result["per_patient"][key] = {
            "observed": observed_value,
            "reference_mean": mean,
            "reference_percentiles": np.percentile(vals, [2.5, 97.5]) if len(vals) else [np.nan, np.nan],
            "difference": observed_value - mean if np.isfinite(mean) else float("nan"),
        }
    return pooled_result, history_result


def main() -> None:
    fixture_dir = PACKAGE_ROOT / "tests" / "testthat" / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261009)
    n_patients, n_time, runin_days = 12, 20, 4
    patient_ids = np.asarray([f"p{i:02d}" for i in range(n_patients)])
    patient = np.repeat(patient_ids, n_time)
    time = np.tile(np.arange(n_time, dtype=int), n_patients)
    latent_rate = rng.beta(2.5, 9.0, size=n_patients)
    y = rng.binomial(1, np.repeat(latent_rate, n_time)).astype(int)
    forecast = np.empty(len(y), dtype=float)
    for i, key in enumerate(patient_ids):
        ix = np.flatnonzero(patient == key)
        prior_count = np.r_[0.0, np.cumsum(y[ix][:-1])]
        history_rate = (0.25 + prior_count) / (1.0 + np.arange(n_time))
        forecast[ix] = np.clip(0.35 * latent_rate[i] + 0.65 * history_rate + rng.normal(0, 0.025, n_time), 0.01, 0.99)
    data = pd.DataFrame({"patient": patient, "time": time, "y": y, "forecast": forecast})
    scored = data.loc[data["time"] >= runin_days].reset_index(drop=True)

    permutation_rng = np.random.default_rng(8821)
    maps: list[dict[str, np.ndarray]] = []
    perm_rows: list[dict[str, Any]] = []
    n_perm = 23
    n_scored = n_time - runin_days
    for b in range(1, n_perm + 1):
        draw: dict[str, np.ndarray] = {}
        for key in patient_ids:
            order = permutation_rng.permutation(n_scored) + 1
            draw[str(key)] = order
            for position, source in enumerate(order, start=1):
                perm_rows.append({"iteration": b, "patient": str(key), "position": position, "source": int(source)})
        maps.append(draw)
    pd.DataFrame(perm_rows).to_csv(fixture_dir / "permutations.csv", index=False)
    data.to_csv(fixture_dir / "sim.csv", index=False, float_format="%.17g")

    runin_rate = runin_patient_rate(patient, y, time, runin_days)
    expanding_rate = expanding_patient_rate(patient, y, time)
    expanding_with_prior = expanding_patient_rate(patient, y, time, prior_rate=0.18, prior_weight=1.25)
    per_time = per_time_point_auc(scored, "time", min_patients=5)
    direct_results: dict[str, Any] = {
        "auc": auc(scored["y"].to_numpy(), scored["forecast"].to_numpy()),
        "pooled_auc": pooled_auc(scored["patient"].to_numpy(), scored["y"].to_numpy(), scored["forecast"].to_numpy()),
        "within_patient_auc": within_patient_auc(scored["patient"].to_numpy(), scored["y"].to_numpy(), scored["forecast"].to_numpy()),
        "within_patient_auc_pairs": within_patient_auc(scored["patient"].to_numpy(), scored["y"].to_numpy(), scored["forecast"].to_numpy(), weights="pairs"),
        "within_patient_auc_days": within_patient_auc(scored["patient"].to_numpy(), scored["y"].to_numpy(), scored["forecast"].to_numpy(), weights="days"),
        "per_patient_auc_summary": per_patient_auc_summary(scored["patient"].to_numpy(), scored["y"].to_numpy(), scored["forecast"].to_numpy()),
        "per_time_point_auc": per_time,
        "rate_benchmark": rate_benchmark(scored["patient"].to_numpy(), scored["y"].to_numpy()),
        "rate_benchmark_runin": rate_benchmark(scored["patient"].to_numpy(), scored["y"].to_numpy(), runin_rate[time >= runin_days]),
        "population_a0_moment_corrected": population_a0_moment_corrected(scored["patient"].to_numpy(), scored["y"].to_numpy()),
        "brier_scores": brier_scores(scored["patient"].to_numpy(), scored["y"].to_numpy(), scored["forecast"].to_numpy()),
        "brier_skill_runin": brier_skill(scored["y"].to_numpy(), scored["forecast"].to_numpy(), runin_rate[time >= runin_days]),
        "brier_skill_expanding": brier_skill(scored["y"].to_numpy(), scored["forecast"].to_numpy(), expanding_rate[time >= runin_days]),
        "brier_skill_summary_runin": brier_skill_summary(scored["patient"].to_numpy(), scored["y"].to_numpy(), scored["forecast"].to_numpy(), runin_rate[time >= runin_days]),
        "brier_skill_summary_expanding": brier_skill_summary(scored["patient"].to_numpy(), scored["y"].to_numpy(), scored["forecast"].to_numpy(), expanding_rate[time >= runin_days]),
        "patient_calibration_in_the_large": patient_calibration_in_the_large(scored["patient"].to_numpy(), scored["y"].to_numpy(), scored["forecast"].to_numpy()),
        "runin_patient_rate": runin_rate,
        "expanding_patient_rate": expanding_rate,
        "expanding_patient_rate_prior": expanding_with_prior,
    }
    pooled_perm, history_perm = explicit_permutation_results(
        scored["patient"].to_numpy(), scored["y"].to_numpy(), scored["forecast"].to_numpy(),
        patient, y, time, runin_days, maps,
    )
    direct_results["pooled_auc_permutation_test"] = pooled_perm
    direct_results["permutation_reference"] = history_perm

    wrapper = python_evaluate_v1(
        data, time="time", runin_days=runin_days, n_perm=0,
        return_references=True, _include_optional=False,
    )
    wrapper_results = {name: wrapper[name] for name in (
        "cohort", "auc", "brier", "calibration_in_the_large",
        "excluded_patients", "references",
    )}
    wrapper_results["skill"] = {
        name: wrapper["skill"][name]
        for name in ("runin_patient_rate", "expanding_patient_rate")
        if name in wrapper["skill"]
    }
    wrapper_results["references"] = {
        name: wrapper["references"][name]
        for name in ("runin_patient_rate", "expanding_patient_rate")
        if name in wrapper["references"]
    }
    all_results = {"direct": direct_results, "evaluate": wrapper_results}
    flattened = flatten_numeric(all_results)
    expected = pd.DataFrame(
        [{"path": path, "value": csv_value(value)} for path, value in sorted(flattened.items())]
    )
    expected.to_csv(fixture_dir / "expected.csv", index=False)

    # V2 API parity fixtures use the same maps, not matching RNG algorithms.
    eval_results: dict[str, Any] = {}
    fixed = data.copy()
    eval_results["fixed_default_yardstick"] = evaluate_with_maps(
        fixed, maps, time="time", runin=runin_days, n_perm=n_perm, seed=19,
    )
    eval_results["fixed_user_metric"] = evaluate_with_maps(
        fixed, maps, time="time", runin=runin_days, n_perm=n_perm, seed=19,
        metrics={"mean_forecast": lambda y, p: float(np.mean(p))},
    )
    running = data.copy()
    for key in patient_ids:
        ix = np.flatnonzero(patient == key)
        stops = np.searchsorted(time[ix], time[ix], side="left")
        previous = np.r_[0.0, np.cumsum(y[ix])][stops]
        running.loc[ix, "forecast"] = (0.5 + previous) / (1.0 + stops)
    eval_results["running_rate_yardstick"] = evaluate_with_maps(
        running, maps, time="time", runin=runin_days, rebuild="running_rate",
        yardstick="running_rate", n_perm=n_perm, seed=19,
        metrics=["auc", "average_precision", "calibration_slope"],
    )
    window = data.copy()
    width = 5
    for key in patient_ids:
        ix = np.flatnonzero(patient == key)
        stops = np.searchsorted(time[ix], time[ix], side="left")
        previous = np.r_[0.0, np.cumsum(y[ix])]
        starts = np.maximum(stops - width, 0)
        counts = np.minimum(stops, width)
        events = previous[stops] - previous[starts]
        window.loc[ix, "forecast"] = np.divide(events, counts, out=np.zeros_like(events), where=counts > 0)
    eval_results["window_count_default_yardstick"] = evaluate_with_maps(
        window, maps, time="time", runin=runin_days,
        rebuild={"name": "window_count", "w": width, "as_rate": True},
        n_perm=n_perm, seed=19, metrics=["auc", "average_precision"],
    )
    eval_flat: dict[str, float] = {}
    for scenario, result in eval_results.items():
        eval_flat.update(flatten_numeric(result, scenario))
    eval_expected = pd.DataFrame(
        [{"path": path, "value": csv_value(value)} for path, value in sorted(eval_flat.items())]
    )
    eval_expected.to_csv(fixture_dir / "evaluate_expected.csv", index=False)


if __name__ == "__main__":
    main()
