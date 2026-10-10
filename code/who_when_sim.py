"""Cohort-matched who/when simulation for Figure 3 and Table 3.

Run from any working directory with the project Python environment:
    python code/who_when_sim.py [--quick] [--reps N]

The cohort effect distributions and matching are imported from cohort_curves.py.
Each replication shares outcomes, patient effects, and independent noise draws
across all who lines and when-correlation settings.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
WITHINPATIENT_SRC = CODE / "withinpatient" / "src"
for import_path in (str(CODE), str(WITHINPATIENT_SRC)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

import cohort_curves as cohort_sim  # noqa: E402
from withinpatient.metrics import auc as withinpatient_auc  # noqa: E402
from withinpatient.metrics import _irls as metrics_irls  # noqa: E402
from withinpatient.metrics import within_patient_auc  # noqa: E402
from withinpatient.nine import nine_metrics  # noqa: E402


SEED = 20261010
X_SD = 2.0
CALIBRATION_NODES = 40
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite.hermgauss(CALIBRATION_NODES)
_GH_Z = _GH_NODES * math.sqrt(2.0)
_GH_WEIGHTS = _GH_WEIGHTS / math.sqrt(math.pi)
DEFAULT_REPS = 20
QUICK_REPS = 2
QUICK_GRID_POINTS = 3
FULL_GRID_POINTS = 11
MIN_CALIBRATION_ROWS = 200_000
MODEL_CASE_KEYS = {
    "REPLACE-BG": "replace_bg",
    "CTN-0094": "ctn_weekly_missing_excluded",
    "College Experience": "ces",
}
WHO_LINES = {
    "full": 1.0,
    "half": 0.5,
    "wrong": 0.0,
    "none": None,
    "reversed": -1.0,
}
METRIC_ORDER = (
    "auc", "average_precision", "sensitivity", "specificity", "ppv", "npv",
    "net_benefit", "brier_skill", "calibration_slope",
)
METRIC_LABELS = {
    "auc": "AUC",
    "average_precision": "Average precision",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "ppv": "PPV (per-patient excess over own event rate)",
    "npv": "NPV (per-patient excess over own non-event rate)",
    "net_benefit": "Net benefit",
    "brier_skill": "Brier skill",
    "calibration_slope": "Calibration slope (pooled: logistic; per patient: conditional within-patient)",
}
LINE_CODES = {"full": 1, "half": 2, "wrong": 3, "none": 4, "reversed": 5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="use 2 replications and a 3-point when grid")
    parser.add_argument("--reps", type=int, help="number of replications (default: 20; quick default: 2)")
    parser.add_argument("--supplement", action="store_true", help="include the reversed-who rho=-1 curve")
    parser.add_argument("--calibration", choices=("exact", "logistic"), default="exact",
                        help="conditional forecast method (default: exact; logistic reproduces the prior method)")
    args = parser.parse_args()
    args.reps = args.reps if args.reps is not None else (QUICK_REPS if args.quick else DEFAULT_REPS)
    if args.reps < 1:
        parser.error("--reps must be at least 1")
    args.grid_points = QUICK_GRID_POINTS if args.quick else FULL_GRID_POINTS
    return args


def expit(values: np.ndarray | float) -> np.ndarray:
    return cohort_sim.ss.expit(values)


def normal_expit_expectation(mean: np.ndarray | float,
                             variance: float) -> np.ndarray:
    """Integrate expit(N(mean, variance)) by Gauss-Hermite quadrature."""
    mean_array = np.asarray(mean, dtype=float)
    variance = max(0.0, float(variance))
    if variance == 0.0:
        return expit(mean_array)
    values = expit(mean_array[..., None] + math.sqrt(variance) * _GH_Z)
    return np.sum(values * _GH_WEIGHTS, axis=-1)


def logit_normal_level(rate: float, sd: float) -> float:
    """Find the location whose logistic-normal mean equals ``rate``."""
    lo, hi = -20.0, 12.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if float(normal_expit_expectation(mid, sd * sd)) < rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def logit(value: float) -> float:
    return math.log(value) - math.log1p(-value)


def make_replication(cohort: str, cohort_index: int, replication: int,
                     spec: dict[str, Any], fit: dict[str, Any], *,
                     n_patients: int | None = None,
                     recenter_samples: int = 512,
                     seed_parts: tuple[int, ...] | None = None) -> dict[str, np.ndarray | float]:
    """Draw true outcomes and the shared who/when noise for one replication."""
    seed = seed_parts if seed_parts is not None else (SEED, cohort_index, replication)
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    k, runin, n = n_patients or cohort_sim.PATIENTS, spec["runin"], spec["n"]
    x_all = X_SD * rng.standard_normal((k, runin + n))
    z_ref = rng.standard_normal(recenter_samples)
    base = cohort_sim.draw_effects(rng, fit, k)
    shift = cohort_sim.recenter(base, X_SD, spec["rate"], z_ref)
    b = base + shift
    p_all = expit(b[:, None] + x_all)
    y_all = (rng.random(p_all.shape) < p_all).astype(np.int8)

    # One common random draw per patient/day is reused for every rho setting.
    who_noise = rng.standard_normal(k)
    when_noise = rng.standard_normal((k, n))
    x = x_all[:, runin:]
    y = y_all[:, runin:]

    # The exact marginal patient risk integrates over the known N(0, X_SD^2)
    # daily signal, rather than using that patient's finite observed outcome rate.
    nodes, weights = np.polynomial.hermite.hermgauss(64)
    z = nodes * math.sqrt(2.0)
    weights = weights / math.sqrt(math.pi)
    patient_rate = np.sum(weights[None, :] * expit(b[:, None] + X_SD * z[None, :]), axis=1)
    return {
        "b": b,
        "x": x,
        "y": y,
        "who_noise": who_noise,
        "when_noise": when_noise,
        "patient_rate": patient_rate,
    }


def who_component(data: dict[str, np.ndarray | float], rho_who: float | None) -> np.ndarray:
    b = np.asarray(data["b"], dtype=float)
    if rho_who is None:
        return np.zeros_like(b)
    centered_b = b - np.mean(b)
    sd_b = float(np.std(b))
    noise = np.asarray(data["who_noise"], dtype=float)
    return rho_who * centered_b + math.sqrt(max(0.0, 1.0 - rho_who**2)) * sd_b * noise


def when_component(data: dict[str, np.ndarray | float], rho_when: float) -> np.ndarray:
    x = np.asarray(data["x"], dtype=float)
    noise = np.asarray(data["when_noise"], dtype=float)
    return rho_when * x + math.sqrt(max(0.0, 1.0 - rho_when**2)) * X_SD * noise


def _irls_with_offset(x: np.ndarray, y: np.ndarray, offset: np.ndarray) -> np.ndarray | None:
    """Small IRLS fit with a fixed linear-predictor offset."""
    design = np.column_stack([np.ones(len(y)), x])
    mean_y = float(np.mean(y))
    beta = np.zeros(design.shape[1], dtype=float)
    beta[0] = math.log(mean_y / (1.0 - mean_y)) if 0 < mean_y < 1 else 0.0
    for _ in range(100):
        eta = np.clip(offset + design @ beta, -35.0, 35.0)
        probability = 1.0 / (1.0 + np.exp(-eta))
        weight = np.maximum(probability * (1.0 - probability), 1e-10)
        hessian = design.T @ (weight[:, None] * design)
        try:
            step = np.linalg.solve(hessian + np.eye(len(beta)) * 1e-10,
                                   design.T @ (y - probability))
        except np.linalg.LinAlgError:
            return None
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            return beta
    return None


def fit_calibration(data: dict[str, np.ndarray | float], who_line: str,
                    rho_when: float) -> dict[str, Any]:
    """Fit the forecast recalibration coefficients on an independent cohort."""
    y = np.asarray(data["y"], dtype=float).ravel()
    who = who_component(data, WHO_LINES[who_line])
    when = when_component(data, rho_when)
    who_rows = np.broadcast_to(who[:, None], when.shape).ravel()
    when_rows = when.ravel()
    if who_line in {"wrong", "reversed"}:
        beta = _irls_with_offset(when_rows[:, None], y, who_rows)
        if beta is None:
            raise RuntimeError(f"offset calibration IRLS failed for {who_line}, rho_when={rho_when:g}")
        who_coefficient = 1.0
        intercept, when_coefficient = map(float, beta)
    elif who_line == "none":
        beta = metrics_irls(when_rows[:, None], y)
        if beta is None:
            raise RuntimeError(f"calibration IRLS failed for none, rho_when={rho_when:g}")
        intercept, when_coefficient = map(float, beta)
        who_coefficient = 0.0
    else:
        beta = metrics_irls(np.column_stack([who_rows, when_rows]), y)
        if beta is None:
            raise RuntimeError(f"calibration IRLS failed for {who_line}, rho_when={rho_when:g}")
        intercept, who_coefficient, when_coefficient = map(float, beta)
    return {
        "intercept": intercept,
        "who_coefficient": who_coefficient,
        "when_coefficient": when_coefficient,
        "who_coefficient_fixed": who_line in {"wrong", "reversed"},
        "n_calibration_rows": int(len(y)),
        "calibration_rho_when": float(rho_when),
    }


def get_calibration_model(cohort: str, cohort_index: int, spec: dict[str, Any],
                          fit: dict[str, Any], who_line: str, rho_when: float,
                          cache: dict[tuple[str, float], dict[str, Any]] | None = None,
                          method: str = "exact") -> dict[str, Any]:
    """Return an analytic conditional forecast or fit a logistic recalibration."""
    key = (who_line, round(float(rho_when), 12))
    if cache is not None and key in cache:
        return cache[key]
    if method == "exact":
        if fit["dist"] != "logit-normal":
            raise NotImplementedError(
                f"exact conditional calibration currently requires a logit-normal cohort; {cohort} uses {fit['dist']}"
            )
        baseline_sd = float(fit["tau"])
        # Wrong and reversed forecasts treat their observed W as a known baseline.
        # Choose its location so the assumed marginal risk equals the cohort rate.
        # W and the marginal daily signal are independent normal variables here.
        offset_level = (logit_normal_level(float(spec["rate"]),
                                           math.sqrt(baseline_sd**2 + X_SD**2))
                        if who_line in {"wrong", "reversed"} else 0.0)
        model: dict[str, Any] = {
            "method": "exact",
            "intercept": offset_level,
            "who_coefficient": 1.0 if who_line in {"wrong", "reversed", "full", "half"} else 0.0,
            "when_coefficient": 1.0,
            "who_coefficient_fixed": who_line in {"wrong", "reversed"},
            "calibration_rho_when": float(rho_when),
            "cohort_rate": float(spec["rate"]),
            "baseline_sd": baseline_sd,
            "quadrature_nodes": CALIBRATION_NODES,
            "distribution": fit["dist"],
        }
        if cache is not None:
            cache[key] = model
        return model
    if method != "logistic":
        raise ValueError(f"unknown calibration method: {method}")
    calibration_patients = max(1, math.ceil(MIN_CALIBRATION_ROWS / int(spec["n"])))
    rho_seed = int(round(float(rho_when) * 1_000_000_000))
    calibration_data = make_replication(
        cohort, cohort_index, 0, spec, fit,
        n_patients=calibration_patients,
        recenter_samples=64,
        seed_parts=(SEED, 91, cohort_index, LINE_CODES[who_line], rho_seed),
    )
    model = fit_calibration(calibration_data, who_line, float(rho_when))
    model["method"] = "logistic"
    model["n_calibration_patients"] = calibration_patients
    if cache is not None:
        cache[key] = model
    return model


def build_forecast(data: dict[str, np.ndarray | float], who_line: str,
                   rho_when: float, calibration: dict[str, Any]) -> tuple[np.ndarray, float]:
    who = who_component(data, WHO_LINES[who_line])[:, None]
    when = when_component(data, rho_when)
    if calibration.get("method") == "exact":
        b = np.asarray(data["b"], dtype=float)
        mean_b = float(np.mean(b))
        var_b = float(np.var(b))
        if who_line == "full":
            mean = b[:, None] + rho_when * when
            var_b_given_w = 0.0
        elif who_line == "half":
            rho_who = float(WHO_LINES["half"])
            mean = mean_b + rho_who * who + rho_when * when
            var_b_given_w = (1.0 - rho_who**2) * var_b
        elif who_line == "none":
            mean = mean_b + rho_when * when
            var_b_given_w = var_b
        else:
            # These two lines deliberately believe W is the patient's baseline.
            # For reversed, W is the negative of the centered true baseline.
            mean = float(calibration["intercept"]) + who + rho_when * when
            var_b_given_w = 0.0
        var_x_given_d = (1.0 - rho_when**2) * X_SD**2
        forecast = normal_expit_expectation(mean, var_b_given_w + var_x_given_d)
        return forecast, float(calibration["intercept"])
    linear_predictor = (float(calibration["intercept"])
                        + float(calibration["who_coefficient"]) * who
                        + float(calibration["when_coefficient"]) * when)
    return expit(linear_predictor), float(calibration["intercept"])


def auc_pair(y: np.ndarray, forecast: np.ndarray) -> tuple[float, float]:
    """Return pooled and equal-patient AUCs through the package functions."""
    k, n = y.shape
    patient_id = np.repeat(np.arange(k), n)
    pooled = withinpatient_auc(y.ravel(), forecast.ravel())
    within = within_patient_auc(patient_id, y.ravel(), forecast.ravel(), weights="patient")
    return float(pooled), float(within["value"])


def collect_curves(cohort: str, cohort_index: int, spec: dict[str, Any], fit: dict[str, Any],
                   rhos: np.ndarray, reps: int, include_reversed: bool,
                   calibrations: dict[tuple[str, float], dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    line_names = ["full", "half", "wrong", "none"] + (["reversed"] if include_reversed else [])
    auc_values: dict[tuple[str, float], dict[str, list[float]]] = {
        (line, float(rho)): {"pooled": [], "within": []}
        for line in line_names for rho in rhos
    }
    for rep in range(reps):
        data = make_replication(cohort, cohort_index, rep, spec, fit)
        y = np.asarray(data["y"], dtype=np.int8)
        for line in line_names:
            for rho_when in rhos:
                calibration = calibrations[(line, round(float(rho_when), 12))]
                forecast, _ = build_forecast(data, line, float(rho_when), calibration)
                pooled, within = auc_pair(y, forecast)
                values = auc_values[(line, float(rho_when))]
                values["pooled"].append(pooled)
                values["within"].append(within)

    rows: list[dict[str, Any]] = []
    for line in line_names:
        for rho_when in rhos:
            values = auc_values[(line, float(rho_when))]
            pooled = np.asarray(values["pooled"], dtype=float)
            within = np.asarray(values["within"], dtype=float)
            pooled_sd = float(np.std(pooled, ddof=1)) if reps > 1 else 0.0
            within_sd = float(np.std(within, ddof=1)) if reps > 1 else 0.0
            rows.append({
                "cohort": cohort,
                "who_line": line,
                "rho_who": WHO_LINES[line],
                "rho_when": float(rho_when),
                "x_sd": X_SD,
                "within_patient_auc_mean": float(np.mean(within)),
                "within_patient_auc_sd": within_sd,
                "within_patient_auc_sem": within_sd / math.sqrt(reps),
                "pooled_auc_mean": float(np.mean(pooled)),
                "pooled_auc_sd": pooled_sd,
                "pooled_auc_sem": pooled_sd / math.sqrt(reps),
                "n_reps": reps,
            })
    return rows, {"cohort": cohort, "cohort_index": cohort_index, "spec": spec,
                  "fit": fit, "calibrations": calibrations}


def matched_rho(curve_rows: list[dict[str, Any]], cohort: str, target: float) -> dict[str, float]:
    points = sorted((r["rho_when"], r["within_patient_auc_mean"]) for r in curve_rows
                    if r["cohort"] == cohort and r["who_line"] == "full")
    rhos = np.asarray([point[0] for point in points], dtype=float)
    aucs = np.asarray([point[1] for point in points], dtype=float)
    aucs = np.maximum.accumulate(aucs)
    if target < aucs[0] - 1e-8 or target > aucs[-1] + 1e-8:
        raise ValueError(
            f"{cohort}: GB per-patient AUC {target:.4f} is outside simulated full-who range "
            f"[{aucs[0]:.4f}, {aucs[-1]:.4f}]; increase X_SD or inspect the model target"
        )
    rho = float(np.interp(target, aucs, rhos))
    achieved = float(np.interp(rho, rhos, aucs))
    return {"rho_when": rho, "target_within_patient_auc": float(target),
            "curve_interpolated_within_patient_auc": achieved}


def json_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def summarize_table_metrics(result: dict[str, Any], metric: str) -> dict[str, Any]:
    entry = result[metric]
    summary = entry["summary"]
    iqr = summary.get("iqr", [None, None])
    return {
        "mean": json_number(summary.get("mean")),
        "median": json_number(summary.get("median")),
        "iqr": [json_number(iqr[0]), json_number(iqr[1])],
        "n": int(summary.get("n", 0)),
    }


def mc_sd_of_mean(values: list[Any]) -> float | None:
    finite = np.asarray([json_number(value) for value in values], dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return None
    return float(np.std(finite, ddof=1) / math.sqrt(len(finite)))


def make_table3_data(cohort: str, cohort_index: int, spec: dict[str, Any], fit: dict[str, Any],
                     rho_when: float, reps: int,
                     calibration: dict[str, Any]) -> tuple[tuple[np.ndarray, ...], dict[str, dict[str, list[float]]]]:
    ys: list[np.ndarray] = []
    true_forecasts: list[np.ndarray] = []
    rate_forecasts: list[np.ndarray] = []
    patient_rates: list[np.ndarray] = []
    patient_ids: list[np.ndarray] = []
    replicate_metrics: dict[str, dict[str, list[float]]] = {
        metric: {"true_pooled": [], "true_per_patient_mean": [], "rate_only_pooled": []}
        for metric in METRIC_ORDER
    }
    k, n = cohort_sim.PATIENTS, spec["n"]
    for rep in range(reps):
        data = make_replication(cohort, cohort_index, rep, spec, fit)
        y = np.asarray(data["y"], dtype=np.int8)
        full_forecast, _ = build_forecast(data, "full", rho_when, calibration)
        rates = np.asarray(data["patient_rate"], dtype=float)
        patient = np.arange(k)
        flat_y = y.ravel()
        flat_forecast = full_forecast.ravel()
        flat_rates = np.broadcast_to(rates[:, None], (k, n)).ravel()
        population_yardstick = np.full(len(flat_y), float(spec["rate"]), dtype=float)
        with warnings.catch_warnings():
            # Per-replication fits are used for Monte Carlo uncertainty; the
            # returned conditional-fit reason is retained in the aggregate result.
            warnings.simplefilter("ignore", RuntimeWarning)
            true_rep = nine_metrics(patient.repeat(n), flat_y, flat_forecast,
                                    float(spec["rate"]), yardstick=flat_rates)
            rate_rep = nine_metrics(patient.repeat(n), flat_y, flat_rates,
                                    float(spec["rate"]), yardstick=population_yardstick)
        for metric in METRIC_ORDER:
            true_pooled = true_rep[metric]["pooled"]
            if metric == "brier_skill":
                model_loss = float(np.mean((flat_y - flat_forecast) ** 2))
                population_loss = float(np.mean((flat_y - float(spec["rate"])) ** 2))
                true_pooled = 1.0 - model_loss / population_loss if population_loss > 0 else float("nan")
            replicate_metrics[metric]["true_pooled"].append(float(true_pooled))
            replicate_metrics[metric]["true_per_patient_mean"].append(
                float(true_rep[metric]["summary"]["mean"])
            )
            replicate_metrics[metric]["rate_only_pooled"].append(float(rate_rep[metric]["pooled"]))
        ys.append(y.ravel())
        true_forecasts.append(full_forecast.ravel())
        rate_forecasts.append(flat_rates)
        patient_rates.append(flat_rates)
        patient_ids.append(np.broadcast_to((rep * k + np.arange(k))[:, None], (k, n)).ravel())
    combined = (np.concatenate(patient_ids), np.concatenate(ys),
                np.concatenate(true_forecasts), np.concatenate(rate_forecasts),
                np.concatenate(patient_rates))
    return combined, replicate_metrics


def table3_for_cohort(cohort: str, cohort_index: int, spec: dict[str, Any], fit: dict[str, Any],
                      rho_when: float, reps: int,
                      calibration: dict[str, Any]) -> dict[str, dict[str, Any]]:
    (patient, y, true_forecast, rate_forecast, patient_rate), replicate_metrics = make_table3_data(
        cohort, cohort_index, spec, fit, rho_when, reps, calibration
    )
    threshold = float(spec["rate"])
    population_yardstick = np.full(len(y), threshold, dtype=float)
    true_patient_metrics = nine_metrics(patient, y, true_forecast, threshold, yardstick=patient_rate)
    rate_population_metrics = nine_metrics(patient, y, rate_forecast, threshold, yardstick=population_yardstick)

    cells: dict[str, dict[str, Any]] = {}
    for metric in METRIC_ORDER:
        true_pooled = true_patient_metrics[metric]["pooled"]
        if metric == "brier_skill":
            model_loss = float(np.mean((y - true_forecast) ** 2))
            population_loss = float(np.mean((y - threshold) ** 2))
            true_pooled = 1.0 - model_loss / population_loss if population_loss > 0 else float("nan")
        true_patient = summarize_table_metrics(true_patient_metrics, metric)
        calibration_fit = true_patient_metrics["calibration_slope"].get("conditional_fit", {})
        calibration_reason = str(calibration_fit.get("reason", ""))
        cells[metric] = {
            "true_risk_pooled": json_number(true_pooled),
            "true_risk_pooled_mc_sd": mc_sd_of_mean(replicate_metrics[metric]["true_pooled"]),
            "true_risk_per_patient": true_patient,
            "true_risk_per_patient_mc_sd": mc_sd_of_mean(
                replicate_metrics[metric]["true_per_patient_mean"]
            ),
            "rate_only_pooled": json_number(rate_population_metrics[metric]["pooled"]),
            "rate_only_pooled_mc_sd": mc_sd_of_mean(replicate_metrics[metric]["rate_only_pooled"]),
            "per_patient_calibration_reason": calibration_reason if metric == "calibration_slope" else None,
        }
    return cells


def fmt_num(value: Any, digits: int = 3) -> str:
    number = json_number(value)
    return "—" if number is None else f"{number:.{digits}f}"


def fmt_mc(value: Any) -> str:
    number = json_number(value)
    return f" (MC SD {number:.3f})" if number is not None else " (MC SD n/a)"


def pooled_cell_text(value: Any, mc_sd: Any, metric: str, rate: float) -> str:
    text = fmt_num(value)
    if metric == "ppv":
        text += f" [{fmt_num(rate)}]"
    elif metric == "npv":
        text += f" [{fmt_num(1.0 - rate)}]"
    return text + fmt_mc(mc_sd)


def write_table3(curve_rows: list[dict[str, Any]], info: dict[str, dict[str, Any]],
                 model_targets: dict[str, dict[str, float]], reps: int,
                 output_dir: Path) -> None:
    cohorts: dict[str, Any] = {}
    table_reps = reps
    for cohort_index, cohort in enumerate(cohort_sim.COHORTS):
        cohort_details = info[cohort]
        spec, fit = cohort_details["spec"], cohort_details["fit"]
        match = matched_rho(curve_rows, cohort, model_targets[cohort]["within_patient_auc"])
        calibration = get_calibration_model(
            cohort, cohort_details["cohort_index"], spec, fit, "full", match["rho_when"],
            cohort_details["calibrations"],
        )
        metrics = table3_for_cohort(cohort, cohort_index, spec, fit, match["rho_when"],
                                    table_reps, calibration)
        cohorts[cohort] = {
            "case_table_key": MODEL_CASE_KEYS[cohort],
            "gradient_boosting_auc": model_targets[cohort],
            "matched_ability": match,
            "threshold": float(spec["rate"]),
            "n_patients_per_replication": cohort_sim.PATIENTS,
            "n_table3_replications": table_reps,
            "matched_full_who_calibration": calibration,
            "metrics": metrics,
        }

    rows = []
    for metric in METRIC_ORDER:
        rows.append({
            "metric": metric,
            "label": METRIC_LABELS[metric],
            "cohorts": {cohort: cohorts[cohort]["metrics"][metric] for cohort in cohorts},
        })
    payload = {
        "title": "Cohort-matched who/when simulation (Table 3)",
        "seed": SEED,
        "curve_reps": reps,
        "curve_grid_points": len({r["rho_when"] for r in curve_rows if r["who_line"] == "full"}),
        "design": {
            "latent_risk_logit": "b_k + x_kt; b_k is the recentered cohort-matched patient logit baseline and x_kt = 2*z_kt, z_kt ~ N(0,1).",
            "forecast": ("Exact conditional expectation E[expit(L)|W,D], evaluated with 40-node Gauss-Hermite quadrature."
                         if calibration.get("method") == "exact" else
                         "expit(a + c1*W_k + c2*D_kt), with coefficients fit on an independent simulated calibration cohort for each cohort, who line, and rho_when."),
            "who_component": "For rho_who in {1, 0.5, 0, -1}, W_k = rho_who*(b_k-mean(b)) + sqrt(1-rho_who^2)*SD(b)*u_k, u_k ~ N(0,1); none uses W_k=0. The same u_k is used across lines and rho_when values.",
            "when_component": "D_kt = rho_when*x_kt + sqrt(1-rho_when^2)*SD(x)*v_kt, v_kt ~ N(0,1); the same v_kt is used across lines and rho_when values.",
            "calibration": ("Exact conditional expectation by 40-node Gauss-Hermite quadrature; the current cohort fits are all logit-normal."
                            if calibration.get("method") == "exact" else
                            f"Each calibration cohort uses a dedicated seed stream and at least {MIN_CALIBRATION_ROWS:,} scored rows; logistic IRLS fits the forecast conditional on its available components."),
            "wrong_line_exception": ("Wrong and reversed retain the specified forecast belief that W_k is a known baseline; their location is chosen to match the cohort event rate."
                                     if calibration.get("method") == "exact" else
                                     "For wrong and supplement-only reversed who lines, W_k is an offset with coefficient fixed at 1 and only the intercept and D_kt slope are fit; none fits the intercept and D_kt only."),
            "shared_records": "Outcomes, b, x, who noise, and when noise are shared across all lines and rho_when settings within a replication.",
            "patient_rate_yardstick": "Each patient's marginal true rate E[expit(b_k + X)] for X ~ N(0, 2^2), evaluated by 64-node Gauss-Hermite quadrature; used for per-patient Brier skill and the patient net-benefit policy.",
            "population_rate_yardstick": "The cohort target event rate from cohort_curves.COHORTS; used for pooled Brier skill.",
            "alarm_threshold": "The cohort target event rate.",
            "brier_skill": "Pooled Brier skill is versus the cohort population event rate; per-patient Brier skill is versus that patient's marginal true rate.",
            "per_patient_calibration": "The per-patient calibration cell is the shared conditional_within_slope from conditional logistic regression with patient intercepts.",
            "metric_aggregation": f"nine_metrics was applied to all {table_reps} independent replications, with patient IDs unique across replications; each replication has {cohort_sim.PATIENTS} patients. Monte Carlo SDs are replication-level standard deviations divided by sqrt(number of replications).",
            "cohort_matching": "Patient effect distributions, event rates, scored record lengths, and run-in lengths are imported from code/cohort_curves.py (fit_cohort, draw_effects, recenter).",
        },
        "model_keys_used": {
            cohort: {
                "case_key": MODEL_CASE_KEYS[cohort],
                "within_auc_key": f"{MODEL_CASE_KEYS[cohort]}.forecasts.gradient_boosting.within_patient_auc",
                "pooled_auc_key": f"{MODEL_CASE_KEYS[cohort]}.forecasts.gradient_boosting.pooled_auc",
                **model_targets[cohort],
            }
            for cohort in cohort_sim.COHORTS
        },
        "cohorts": cohorts,
        "rows": rows,
    }
    (output_dir / "table3.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    md = [
        "# Table 3. Cohort-matched who/when simulation",
        "",
        "Cells show pooled true-risk forecast; per-patient true-risk forecast (median [Q1, Q3], mean); and pooled rate-only forecast. Each mean is followed by its Monte Carlo SD (the across-replication SD divided by sqrt(reps)).",
        "PPV and NPV pooled cells show the population event and non-event rates in brackets, respectively; per-patient values are excess over each patient's own event or non-event rate.",
        "Brier skill compares the pooled forecast with the population event rate and the per-patient forecast with each patient's own marginal true rate.",
        "Per-patient calibration is the conditional within-patient slope with patient intercepts (exact conditional logistic regression).",
        ("Design: W_k = rho_who*(b_k - mean(b)) + sqrt(1-rho_who^2)*SD(b)*u_k (zero for none), and D_kt = rho_when*x_kt + sqrt(1-rho_when^2)*SD(x)*v_kt. Forecasts use the exact conditional expectation by 40-node Gauss-Hermite quadrature; wrong and reversed retain their stated belief that W_k is the baseline. "
         if calibration.get("method") == "exact" else
         "Design: W_k = rho_who*(b_k - mean(b)) + sqrt(1-rho_who^2)*SD(b)*u_k (zero for none), and D_kt = rho_when*x_kt + sqrt(1-rho_when^2)*SD(x)*v_kt. Each forecast is recalibrated by logistic regression on an independent simulated cohort; wrong and reversed fix the W_k coefficient at 1. ")
        + "Threshold = cohort event rate. Table 3 matches rho_when to the gradient-boosting target per-patient AUC shown for each cohort below.",
        "",
    ]
    for cohort in cohort_sim.COHORTS:
        md.extend([
            f"## {cohort}",
            "",
            f"Matched rho_when: {cohorts[cohort]['matched_ability']['rho_when']:.3f}; target gradient-boosting per-patient AUC: {cohorts[cohort]['matched_ability']['target_within_patient_auc']:.3f}; threshold: {cohorts[cohort]['threshold']:.3f}.",
            "",
            "| Metric | True risk, pooled | True risk, per patient | Rate-only, pooled |",
            "|---|---:|---:|---:|",
        ])
        for metric in METRIC_ORDER:
            cell = cohorts[cohort]["metrics"][metric]
            pp = cell["true_risk_per_patient"]
            q1, q3 = pp["iqr"]
            if metric == "calibration_slope":
                reason = cell["per_patient_calibration_reason"] or ""
                if "statsmodels is not installed" in reason:
                    pp_text = "n/a (no statsmodels)"
                elif json_number(pp["mean"]) is not None:
                    pp_text = fmt_num(pp["mean"]) + fmt_mc(cell["true_risk_per_patient_mc_sd"])
                else:
                    pp_text = f"n/a ({reason or 'not estimable'})"
            elif pp["n"]:
                pp_text = (f"{fmt_num(pp['median'])} [{fmt_num(q1)}, {fmt_num(q3)}]; "
                           f"mean {fmt_num(pp['mean'])}{fmt_mc(cell['true_risk_per_patient_mc_sd'])}")
            else:
                pp_text = "—"
            pooled_text = pooled_cell_text(cell["true_risk_pooled"],
                                           cell["true_risk_pooled_mc_sd"], metric,
                                           cohorts[cohort]["threshold"])
            rate_text = pooled_cell_text(cell["rate_only_pooled"],
                                         cell["rate_only_pooled_mc_sd"], metric,
                                         cohorts[cohort]["threshold"])
            md.append(f"| {METRIC_LABELS[metric]} | {pooled_text} | {pp_text} | {rate_text} |")
        md.append("")
    md.extend([
        "## Simulation details",
        "",
        f"Seed {SEED}; {reps} replications for curves and Table 3; {cohort_sim.PATIENTS} patients per replication; latent daily-logit SD {X_SD:.1f}.",
        "The cohort effect fits and cohort specifications are imported from `code/cohort_curves.py`.",
        "",
    ])
    (output_dir / "table3.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output_dir = ROOT / "open_data" / "results" / "v10"
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = ROOT / "open_data" / "cases_tables.json"
    with cases_path.open(encoding="utf-8") as handle:
        case_tables = json.load(handle)

    model_targets: dict[str, dict[str, float]] = {}
    print("Gradient-boosting AUC keys used:", flush=True)
    for cohort in cohort_sim.COHORTS:
        case_key = MODEL_CASE_KEYS[cohort]
        forecast = case_tables[case_key]["forecasts"]["gradient_boosting"]
        target = {
            "within_patient_auc": float(forecast["within_patient_auc"]),
            "pooled_auc": float(forecast["pooled_auc"]),
        }
        model_targets[cohort] = target
        print(f"  {cohort}: {case_key}.forecasts.gradient_boosting.within_patient_auc "
              f"(pooled_auc={target['pooled_auc']:.6f}, within_patient_auc={target['within_patient_auc']:.6f})", flush=True)

    rhos = np.linspace(0.0, 1.0, args.grid_points)
    all_rows: list[dict[str, Any]] = []
    cohort_info: dict[str, dict[str, Any]] = {}
    max_sem = 0.0
    for cohort_index, (cohort, spec) in enumerate(cohort_sim.COHORTS.items()):
        fit = cohort_sim.fit_cohort(spec)
        line_names = ["full", "half", "wrong", "none"] + (["reversed"] if args.supplement else [])
        calibrations = {
            (line, round(float(rho), 12)): get_calibration_model(
                cohort, cohort_index, spec, fit, line, float(rho), method=args.calibration
            )
            for line in line_names for rho in rhos
        }
        rows, info = collect_curves(cohort, cohort_index, spec, fit, rhos, args.reps,
                                    include_reversed=args.supplement, calibrations=calibrations)
        all_rows.extend(rows)
        cohort_info[cohort] = info
        for row in rows:
            max_sem = max(max_sem, row["within_patient_auc_sem"], row["pooled_auc_sem"])
        print(f"completed {cohort}: {len(rows)} curves, {args.reps} replications", flush=True)

    if args.calibration == "logistic":
        print("Calibration coefficient ranges (a, c1 for W, c2 for D):", flush=True)
        for cohort, details in cohort_info.items():
            models = list(details["calibrations"].values())
            ranges = []
            for key in ("intercept", "who_coefficient", "when_coefficient"):
                values = [float(model[key]) for model in models]
                ranges.append(f"{key}=[{min(values):.4f}, {max(values):.4f}]")
            full_at_one = details["calibrations"][("full", 1.0)]
            print(f"  {cohort}: " + ", ".join(ranges)
                  + f"; full rho_when=1 c1={full_at_one['who_coefficient']:.4f}, "
                  + f"c2={full_at_one['when_coefficient']:.4f}", flush=True)
    else:
        print(f"Calibration: exact conditional expectation, {CALIBRATION_NODES}-node Gauss-Hermite quadrature; "
              "cohort fits=" + ", ".join(
                  f"{name}:{details['fit']['dist']}" for name, details in cohort_info.items()
              ), flush=True)

    curve_path = output_dir / "who_when_curves.csv"
    columns = (
        "cohort", "who_line", "rho_who", "rho_when", "x_sd",
        "within_patient_auc_mean", "within_patient_auc_sd", "within_patient_auc_sem",
        "pooled_auc_mean", "pooled_auc_sd", "pooled_auc_sem", "n_reps",
    )
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)

    write_table3(all_rows, cohort_info, model_targets, args.reps, output_dir)
    elapsed = time.perf_counter() - started
    print(f"wrote {curve_path} ({len(all_rows)} rows), table3.json and table3.md; "
          f"max curve SEM={max_sem:.5f}; runtime={elapsed:.1f}s", flush=True)
    if not args.quick and max_sem >= 0.005:
        print("WARNING: at least one curve mean has SEM >= 0.005; increase --reps for the requested precision.", flush=True)


if __name__ == "__main__":
    main()
