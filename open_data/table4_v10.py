#!/usr/bin/env python3
"""Assemble the nine-metric clinical Table 4 and its permutation references.

Run from the repository root with the full modelling environment:
    python open_data/table4_v10.py [--quick]

The model-fixed fit and history-rebuild functions are imported from
``cases_model_fixed_permutation.py``. This script does not edit that file.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent

# Loaded only for model analysis and helper self-checks. In particular,
# --render-only stays independent of sklearn and the fitting import chain.
np = None
pd = None
nine_metrics = None

OUTPUT_DIR = ROOT / "results" / "v10"
CASES_TABLES = ROOT / "cases_tables.json"
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261010
METRIC_ORDER = (
    "auc", "average_precision", "sensitivity", "specificity", "ppv", "npv",
    "net_benefit", "brier_skill", "calibration_slope",
)
METRIC_LABELS = {
    "auc": "AUC",
    "average_precision": "Average precision",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "ppv": "PPV (pooled [base rate]; per patient: excess over own rate)",
    "npv": "NPV (pooled [base rate]; per patient: excess over own rate)",
    "net_benefit": "Net benefit",
    "brier_skill": "Brier skill",
    "calibration_slope": "Calibration slope",
}
COHORTS = (
    {
        "key": "replace_bg",
        "label": "REPLACE-BG",
        "cases_tables_key": "replace_bg",
        "permutation_key": "replace_bg",
        "forecast_file": "forecasts_replace_bg.csv",
        "updated_rate_column": "expanding_rate",
        "bundle_time_column": "night_index",
    },
    {
        "key": "ctn_weekly_missing_excluded",
        "label": "CTN-0094",
        "cases_tables_key": "ctn_weekly_missing_excluded",
        "permutation_key": "ctn0094",
        "forecast_file": "forecasts_ctn_weekly_missing_excluded.csv",
        "updated_rate_column": "updating_rate",
        "bundle_time_column": "study_week",
    },
    {
        "key": "ces",
        "label": "College Experience",
        "cases_tables_key": "ces",
        "permutation_key": "ces",
        "forecast_file": "forecasts_ces.csv",
        "updated_rate_column": "expanding_rate",
        "bundle_time_column": "date",
    },
)


def _load_modeling_dependencies():
    """Import numerical and metric code only for analysis, never for rendering."""
    global np, pd, nine_metrics
    import numpy
    import pandas

    sys.path.insert(0, str(ROOT.parent / "code" / "wheneval" / "src"))
    from wheneval.nine import nine_metrics as metrics_function

    np = numpy
    pd = pandas
    nine_metrics = metrics_function


def clean(value: Any) -> Any:
    """Convert NumPy values and undefined estimates to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def load_permutation_module():
    """Load the existing fixed-model fitting and history-rebuild machinery."""
    path = ROOT / "cases_model_fixed_permutation.py"
    name = "_table4_cases_model_fixed_permutation"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load permutation machinery from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sort_bundle_groups(bundle, time_column: str):
    """Return each patient's array positions in forecast-time order."""
    pid = np.asarray(bundle["pid"])
    test = bundle["test"]
    groups = []
    for key, ix in bundle_group_indices(pid):
        if time_column in test.columns:
            times = test.iloc[ix][time_column].to_numpy()
            ix = ix[np.argsort(times, kind="stable")]
        groups.append((key, ix))
    return groups


def bundle_group_indices(patient):
    """Stable NumPy-only equivalent of the permutation module's group helper."""
    pid = np.asarray(patient)
    return [(key, np.flatnonzero(pid == key)) for key in pd.unique(pid)]


def _sort_forecast_groups(frame: pd.DataFrame):
    groups = []
    patient_values = frame["patient"].to_numpy()
    for key in pd.unique(patient_values):
        ix = np.flatnonzero(patient_values == key)
        ix = ix[np.argsort(frame.iloc[ix]["t"].to_numpy(), kind="stable")]
        groups.append((key, ix))
    return groups


def _same_vector(left, right, *, atol=1e-10):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return False
    try:
        return bool(np.allclose(left.astype(float), right.astype(float), rtol=0, atol=atol,
                                equal_nan=True))
    except (TypeError, ValueError):
        return bool(np.array_equal(left, right))


def align_forecast_rows(frame: pd.DataFrame, bundle, config):
    """Map exported forecast rows to the fitted bundle, checking patient histories.

    The export uses integer patient labels while fitted bundles use cohort IDs.
    Matching is therefore done by each patient's ordered outcome/history values;
    group order is used only to break exact duplicate signatures.
    """
    bundle_pid = np.asarray(bundle["pid"])
    bundle_y = np.asarray(bundle["y"], dtype=int)
    bundle_groups = _sort_bundle_groups(bundle, config["bundle_time_column"])
    file_groups = _sort_forecast_groups(frame)
    if len(bundle_groups) != len(file_groups):
        raise AssertionError(
            f"{config['label']}: forecast export has {len(file_groups)} patients but "
            f"the fixed-model test set has {len(bundle_groups)}"
        )

    test = bundle["test"]
    bundle_runin = test["runin_rate"].to_numpy(float)
    rate_col = config["updated_rate_column"]
    bundle_rate_col = "expanding_rate"  # updating_rate aliases this in CTN's add_references.
    bundle_rate = test[bundle_rate_col].to_numpy(float)
    fitted_gb = np.asarray(bundle["predictions"]["gradient_boosting"], dtype=float)
    used = set()
    file_order = []
    bundle_order = []

    for group_number, (_, bix) in enumerate(bundle_groups):
        b_y = bundle_y[bix]
        candidates = []
        for file_number, (_, fix) in enumerate(file_groups):
            if file_number in used or len(fix) != len(bix):
                continue
            f = frame.iloc[fix]
            if not np.array_equal(f.y.to_numpy(int), b_y):
                continue
            score = 0
            if _same_vector(f["runin_rate"].to_numpy(float), bundle_runin[bix]):
                score += 1
            if _same_vector(f[rate_col].to_numpy(float), bundle_rate[bix]):
                score += 2
            if _same_vector(f["gradient_boosting"].to_numpy(float), fitted_gb[bix], atol=1e-8):
                score += 4
            candidates.append((score, file_number, fix))
        if not candidates:
            raise AssertionError(
                f"{config['label']}: could not align exported rows to fixed-model patient "
                f"group {group_number}; patient outcomes or ordered rows differ"
            )
        best_score = max(item[0] for item in candidates)
        best = [item for item in candidates if item[0] == best_score]
        # Files from the cohort exporters preserve the test-set patient order.
        chosen = next((item for item in best if item[1] == group_number), best[0])
        _, file_number, fix = chosen
        used.add(file_number)
        bundle_order.extend(bix.tolist())
        file_order.extend(fix.tolist())

    bundle_order = np.asarray(bundle_order, dtype=int)
    file_order = np.asarray(file_order, dtype=int)
    if not np.array_equal(bundle_y[bundle_order], frame.iloc[file_order].y.to_numpy(int)):
        raise AssertionError(f"{config['label']}: aligned outcomes do not match")
    return bundle_order, file_order


def _make_auxiliary_data(pm, config, bundle):
    """Prepare the full history frame required by the existing rebuild helpers."""
    if config["permutation_key"] == "replace_bg":
        panel = pd.read_csv(ROOT / "replace_bg_panel.csv")
        # Same types as fit_replace_bg in cases_model_fixed_permutation.py (pid as string).
        panel["pid"] = panel.pid.astype(str)
        panel["y"] = panel.y.astype(int)
        return {"panel": panel}
    if config["permutation_key"] == "ctn0094":
        raw = pd.read_csv(ROOT / "ctn0094_weekly_panel.csv")
        split = pm.CTN.split_people(raw)
        event_col = "event"
        train_y = raw.loc[
            raw.who.isin(split["train"]) & raw[event_col].notna(), event_col
        ].astype(int)
        train_rate = float(train_y.mean()) if len(train_y) else 0.5
        data = pm.CTN.add_references(raw, event_col, train_rate)
        return {"data": data, "event_col": event_col}
    return {}


def rebuild_updated_rate(pm, config, bundle, target, auxiliary):
    """Rebuild the yardstick aligned to one shuffled within-patient outcome draw."""
    key = config["permutation_key"]
    if key == "replace_bg":
        changed = pm.replace_build_histories(auxiliary["panel"], target, bundle["test"])
        return changed["expanding_rate"].to_numpy(float)
    if key == "ctn0094":
        changed = pm.ctn_rebuild_histories(
            auxiliary["data"], bundle["test"], target, auxiliary["event_col"]
        )
        # CTN.add_references defines updating_rate as an alias of expanding_rate, then fills missing
        # histories with the training rate and clips to ctn0094_analysis.CLIP (line 245 there).
        rate = changed.loc[bundle["test"].index, "expanding_rate"]
        fallback = auxiliary["data"].attrs.get("history_fallback")
        if fallback is not None:
            rate = rate.fillna(float(fallback))
        return rate.clip(1e-6, 1 - 1e-6).to_numpy(float)
    changed = pm.ces_rebuild_histories(bundle["test"], target)
    return changed["expanding_rate"].to_numpy(float)


def _summary(values):
    finite = np.asarray([value for value in values.values() if np.isfinite(value)], dtype=float)
    if len(finite):
        q25, q75 = np.percentile(finite, [25, 75])
        return {
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "iqr": [float(q25), float(q75)],
            "n": int(len(finite)),
        }
    return {"mean": None, "median": None, "iqr": [None, None], "n": 0}


def _chance_summary(draw_means):
    valid = np.asarray([value for value in draw_means if value is not None and np.isfinite(value)],
                       dtype=float)
    return {
        "mean_over_draws": float(np.mean(valid)) if len(valid) else None,
        "percentiles_2_5_97_5": (
            [float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))]
            if len(valid) else [None, None]
        ),
        "valid_draws": int(len(valid)),
        "draw_means": [float(value) if value is not None and np.isfinite(value) else None
                       for value in draw_means],
    }


def _pooled_brier_skill(y, forecast, reference):
    """Compute row-weighted Brier skill against a row-aligned reference rate."""
    y = np.asarray(y, dtype=int)
    forecast = np.asarray(forecast, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if not (len(y) == len(forecast) == len(reference)):
        raise ValueError("y, forecast, and reference must have the same length")
    if not len(y):
        return float("nan")
    denominator = float(np.mean((y - reference) ** 2))
    if denominator <= 0:
        return float("nan")
    return float(1 - np.mean((y - forecast) ** 2) / denominator)


def _set_brier_pooled_yardsticks(metric_result, y, forecast, training_cohort_rate):
    """Keep per-patient BSS on updated rates and use training rate for pooled BSS."""
    brier = metric_result["brier_skill"]
    updated_rate_pooled = float(brier["pooled"])
    training_rate_pooled = _pooled_brier_skill(y, forecast, training_cohort_rate)
    brier["pooled"] = training_rate_pooled
    brier["pooled_training_cohort_rate"] = training_rate_pooled
    brier["pooled_updated_rate"] = updated_rate_pooled
    brier["pooled_definition"] = (
        "1 - mean row Brier loss / mean row Brier loss against training_cohort_rate"
    )
    brier["pooled_updated_rate_definition"] = (
        "1 - mean row Brier loss / mean row Brier loss against updated rate"
    )
    brier["pooled_training_cohort_rate_value"] = (
        float(np.asarray(training_cohort_rate, dtype=float)[0])
        if len(training_cohort_rate) else None
    )
    return metric_result


def _paired_bootstrap_intervals(patient, y, forecast, yardstick,
                                training_cohort_rate, threshold, n_draws, rng):
    """Bootstrap pooled-minus-patient means; Brier's pool uses training rate."""
    pid = np.asarray(patient)
    y = np.asarray(y, dtype=int)
    forecast = np.asarray(forecast, dtype=float)
    yardstick = np.asarray(yardstick, dtype=float)
    training_cohort_rate = np.asarray(training_cohort_rate, dtype=float)
    groups = [ix for _, ix in bundle_group_indices(pid)]
    if not groups:
        raise ValueError("Cannot bootstrap an empty test set")
    draws = {metric: [] for metric in METRIC_ORDER}
    for _ in range(n_draws):
        selected = rng.integers(0, len(groups), size=len(groups))
        row_ix = np.concatenate([groups[j] for j in selected])
        # A duplicated sampled patient is a separate bootstrap patient.
        boot_pid = np.concatenate([
            np.full(len(groups[j]), draw_number, dtype=np.int64)
            for draw_number, j in enumerate(selected)
        ])
        values = nine_metrics(boot_pid, y[row_ix], forecast[row_ix], threshold,
                              yardstick=yardstick[row_ix])
        for metric in METRIC_ORDER:
            pooled = values[metric]["pooled"]
            if metric == "brier_skill":
                pooled = _pooled_brier_skill(
                    y[row_ix], forecast[row_ix], training_cohort_rate[row_ix]
                )
            patient_mean = values[metric]["summary"]["mean"]
            delta = float(pooled - patient_mean) if np.isfinite(pooled) and np.isfinite(patient_mean) else np.nan
            draws[metric].append(delta)

    result = {}
    for metric, values in draws.items():
        finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
        ci = ([float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]
              if len(finite) else [None, None])
        result[metric] = {
            "percentiles_2_5_97_5": ci,
            "valid_draws": int(len(finite)),
            "interval_includes_zero": bool(ci[0] <= 0 <= ci[1]) if ci[0] is not None else None,
        }
    return result


def _re_reference_ppv_npv(metric_result, patient, y, runin_rate, metric):
    """Express package PPV/NPV excesses against run-in rather than observed rates."""
    pid = np.asarray(patient)
    y = np.asarray(y, dtype=int)
    runin_rate = np.asarray(runin_rate, dtype=float)
    values = {}
    for key, own_excess in metric_result["per_patient"].items():
        ix = np.flatnonzero(pid == key)
        own_rate = float(np.mean(y[ix]))
        patient_runin = float(np.mean(runin_rate[ix]))
        if metric == "ppv":
            values[key] = float(own_excess + own_rate - patient_runin)
        else:
            values[key] = float(own_excess + patient_runin - own_rate)
    return {
        "pooled_raw_value": float(metric_result["pooled"]),
        "pooled_definition": "raw pooled predictive value; pooled nine_metrics has no PPV/NPV yardstick",
        "per_patient_definition": "predictive value excess over run-in event/non-event rate",
        "per_patient": values,
        "summary": _summary(values),
    }


def _sensitivity_metrics(pid, y, forecast, runin_rate, threshold):
    result = nine_metrics(pid, y, forecast, threshold, yardstick=runin_rate)
    return {
        "brier_skill": result["brier_skill"],
        "ppv": _re_reference_ppv_npv(result["ppv"], pid, y, runin_rate, "ppv"),
        "npv": _re_reference_ppv_npv(result["npv"], pid, y, runin_rate, "npv"),
        "net_benefit": result["net_benefit"],
    }


def _assert_auc_reproduction(actual, expected, cohort, metric_name):
    if expected is None or not np.isfinite(float(expected)):
        raise AssertionError(f"{cohort}: cases_tables.json has no finite {metric_name}")
    if not np.isclose(float(actual), float(expected), atol=1e-12, rtol=0):
        raise AssertionError(
            f"{cohort}: {metric_name}={float(actual):.15g} does not reproduce "
            f"cases_tables.json value {float(expected):.15g} within 1e-12"
        )


def _analyse_cohort(pm, config, bundle, auxiliary, frame, expected, threshold,
                    n_perm, rng_perm, n_bootstrap, rng_bootstrap):
    bundle_order, file_order = align_forecast_rows(frame, bundle, config)
    patient = np.asarray(bundle["pid"])[bundle_order]
    y = frame.iloc[file_order].y.to_numpy(int)
    model_forecast = frame.iloc[file_order].gradient_boosting.to_numpy(float)
    runin_forecast = frame.iloc[file_order].runin_rate.to_numpy(float)
    updated_rate = frame.iloc[file_order][config["updated_rate_column"]].to_numpy(float)
    training_cohort_rate = frame.iloc[file_order].training_cohort_rate.to_numpy(float)
    if not np.allclose(training_cohort_rate, threshold, rtol=0, atol=1e-12):
        raise AssertionError(
            f"{config['label']}: aligned training_cohort_rate does not equal the alarm threshold"
        )

    # Confirm the imported history builders reproduce the exported yardstick before shuffling.
    rebuilt_observed_rate = rebuild_updated_rate(pm, config, bundle, bundle["y"], auxiliary)
    if not _same_vector(rebuilt_observed_rate[bundle_order], updated_rate, atol=1e-10):
        raise AssertionError(
            f"{config['label']}: rebuilt observed updated-rate yardstick does not match "
            f"forecast column {config['updated_rate_column']}"
        )
    if not _same_vector(bundle["test"]["runin_rate"].to_numpy(float)[bundle_order], runin_forecast):
        raise AssertionError(f"{config['label']}: run-in rate does not align to forecast export")

    model_metrics = nine_metrics(patient, y, model_forecast, threshold, yardstick=updated_rate)
    runin_metrics = nine_metrics(patient, y, runin_forecast, threshold, yardstick=updated_rate)
    _set_brier_pooled_yardsticks(
        model_metrics, y, model_forecast, training_cohort_rate
    )
    _set_brier_pooled_yardsticks(
        runin_metrics, y, runin_forecast, training_cohort_rate
    )
    fitted_forecast = np.asarray(bundle["predictions"]["gradient_boosting"], dtype=float)[bundle_order]
    fitted_metrics = nine_metrics(patient, y, fitted_forecast, threshold, yardstick=updated_rate)
    model_auc = model_metrics["auc"]
    _assert_auc_reproduction(fitted_metrics["auc"]["pooled"], expected["pooled_auc"],
                             config["label"], "fixed-model gradient_boosting pooled_auc")
    _assert_auc_reproduction(fitted_metrics["auc"]["summary"]["mean"], expected["within_patient_auc"],
                             config["label"], "fixed-model gradient_boosting within_patient_auc")
    _assert_auc_reproduction(model_auc["pooled"], expected["pooled_auc"],
                             config["label"], "gradient_boosting pooled_auc")
    _assert_auc_reproduction(model_auc["summary"]["mean"], expected["within_patient_auc"],
                             config["label"], "gradient_boosting within_patient_auc")
    reproduction_checks = {
        "pooled_auc": {
            "cases_tables_json": float(expected["pooled_auc"]),
            "export_recomputed": float(model_auc["pooled"]),
            "fixed_model_recomputed": float(fitted_metrics["auc"]["pooled"]),
            "passed_at_atol_1e_12": True,
        },
        "within_patient_auc": {
            "cases_tables_json": float(expected["within_patient_auc"]),
            "export_recomputed": float(model_auc["summary"]["mean"]),
            "fixed_model_recomputed": float(fitted_metrics["auc"]["summary"]["mean"]),
            "passed_at_atol_1e_12": True,
        },
    }

    # Paired patient bootstrap for the pooled-minus-typical-patient marker.
    bootstrap = _paired_bootstrap_intervals(
        patient, y, model_forecast, updated_rate, training_cohort_rate,
        threshold, n_bootstrap, rng_bootstrap
    )
    differences = {}
    for metric in METRIC_ORDER:
        pooled = model_metrics[metric]["pooled"]
        per_patient_mean = model_metrics[metric]["summary"]["mean"]
        difference = (float(pooled - per_patient_mean)
                      if np.isfinite(pooled) and np.isfinite(per_patient_mean) else None)
        differences[metric] = {
            "pooled_minus_per_patient_mean": difference,
            **bootstrap[metric],
        }

    # Same fixed fitted GB model is rescored after every within-patient shuffle.
    permuted_targets = pm.generate_permuted_targets(
        bundle["y"], bundle["pid"], n_perm, rng_perm
    )
    chance_draws = {
        "model": {metric: [] for metric in METRIC_ORDER},
        "runin": {metric: [] for metric in METRIC_ORDER},
    }
    sensitivity_chance_draws = {
        "model": {metric: [] for metric in METRIC_ORDER},
        "runin": {metric: [] for metric in METRIC_ORDER},
    }
    for target in permuted_targets:
        rescored = bundle["score_permutation"](target)
        if "gradient_boosting" not in rescored:
            raise AssertionError(f"{config['label']}: permutation scorer omitted gradient_boosting")
        permuted_rate = rebuild_updated_rate(pm, config, bundle, target, auxiliary)
        permuted_y = np.asarray(target, dtype=int)[bundle_order]
        permuted_model = np.asarray(rescored["gradient_boosting"], dtype=float)[bundle_order]
        # The run-in forecast is unchanged; its outcomes and updated yardstick are paired to this draw.
        for cell, forecast in (("model", permuted_model), ("runin", runin_forecast)):
            result = nine_metrics(patient, permuted_y, forecast, threshold,
                                 yardstick=permuted_rate[bundle_order])
            for metric in METRIC_ORDER:
                mean_value = result[metric]["summary"]["mean"]
                chance_draws[cell][metric].append(
                    float(mean_value) if np.isfinite(mean_value) else None
                )
        for cell, forecast in (("model", permuted_model), ("runin", runin_forecast)):
            sensitivity = nine_metrics(patient, permuted_y, forecast, threshold,
                                       yardstick=runin_forecast)
            for metric in METRIC_ORDER:
                if metric in ("ppv", "npv"):
                    runin_reference_metric = _re_reference_ppv_npv(
                        sensitivity[metric], patient, permuted_y, runin_forecast, metric
                    )
                    mean_value = runin_reference_metric["summary"]["mean"]
                else:
                    mean_value = sensitivity[metric]["summary"]["mean"]
                sensitivity_chance_draws[cell][metric].append(
                    float(mean_value) if np.isfinite(mean_value) else None
                )

    permutation_chance = {
        cell: {metric: _chance_summary(values) for metric, values in by_metric.items()}
        for cell, by_metric in chance_draws.items()
    }
    sensitivity_chance = {
        cell: {metric: _chance_summary(values) for metric, values in by_metric.items()}
        for cell, by_metric in sensitivity_chance_draws.items()
    }
    runin_yardstick_sensitivity = {
        "model": _sensitivity_metrics(patient, y, model_forecast, runin_forecast, threshold),
        "runin": _sensitivity_metrics(patient, y, runin_forecast, runin_forecast, threshold),
        "permutation_chance": {
            cell: {metric: sensitivity_chance[cell][metric]
                   for metric in ("brier_skill", "ppv", "npv", "net_benefit")}
            for cell in ("model", "runin")
        },
    }
    return {
        "label": config["label"],
        "forecast_file": config["forecast_file"],
        "cases_tables_key": config["cases_tables_key"],
        "threshold": float(threshold),
        "training_cohort_rate": float(training_cohort_rate[0]),
        "test_patients": int(pd.unique(patient).size),
        "test_rows": int(len(y)),
        "reproduction_checks": reproduction_checks,
        "model": {"forecast_column": "gradient_boosting", "metrics": model_metrics},
        "runin": {"forecast_column": "runin_rate", "metrics": runin_metrics},
        "permutation_chance": permutation_chance,
        "pooled_minus_per_patient": differences,
        "runin_yardstick_sensitivity": runin_yardstick_sensitivity,
    }


def _fmt(value, digits=3):
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _summary_cell(summary):
    median = _fmt(summary.get("median"))
    q25, q75 = summary.get("iqr", [None, None])
    return f"{median} ({_fmt(q25)}–{_fmt(q75)})"


def _permutation_cell(result):
    mean = _fmt(result.get("mean_over_draws"))
    low, high = result.get("percentiles_2_5_97_5", [None, None])
    return f"{mean} ({_fmt(low)}–{_fmt(high)})"


def _population_event_rate(cohort):
    """Recover the test event rate from the pooled confusion-matrix metrics."""
    metrics = cohort["model"]["metrics"]
    sensitivity = float(metrics["sensitivity"]["pooled"])
    specificity = float(metrics["specificity"]["pooled"])
    ppv = float(metrics["ppv"]["pooled"])
    npv = float(metrics["npv"]["pooled"])

    candidates = []
    # PPV = sensitivity*p / (sensitivity*p + (1-specificity)*(1-p)).
    ppv_denominator = sensitivity * (1 - ppv) + ppv * (1 - specificity)
    if ppv_denominator > 0:
        candidates.append(ppv * (1 - specificity) / ppv_denominator)
    # NPV = specificity*(1-p) / (specificity*(1-p) + (1-sensitivity)*p).
    npv_denominator = specificity * (1 - npv) + npv * (1 - sensitivity)
    if npv_denominator > 0:
        candidates.append(specificity * (1 - npv) / npv_denominator)
    if not candidates or not all(math.isfinite(value) for value in candidates):
        raise ValueError(f"{cohort['label']}: pooled metrics cannot determine test event rate")
    if len(candidates) == 2 and not math.isclose(candidates[0], candidates[1],
                                                 rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError(f"{cohort['label']}: pooled metrics imply inconsistent event rates")
    return sum(candidates) / len(candidates)


def _pooled_value_cell(metric, value, event_rate):
    text = _fmt(value)
    if metric == "ppv":
        text += f" [{_fmt(event_rate)}]"
    elif metric == "npv":
        text += f" [{_fmt(1 - event_rate)}]"
    return text


def _per_patient_value(metric, metric_result):
    summary = metric_result["summary"]
    if metric == "brier_skill":
        # summary.mean is the equal-patient ratio of mean losses, not the
        # median/IQR of the individual patient skill ratios.
        return _fmt(summary.get("mean"))
    if metric == "calibration_slope":
        conditional_fit = metric_result.get("conditional_fit", {})
        return _fmt(conditional_fit.get("slope", summary.get("mean")))
    return _summary_cell(summary)


def build_markdown(result):
    lines = [
        "# Table 4. Pooled and per-patient performance in the clinical test sets",
        "",
        "The model is gradient boosting. Per-patient cells show the median (IQR), followed by the permutation chance mean (2.5th–97.5th percentile), except Brier skill (equal-patient ratio of mean losses) and calibration slope (one conditional within-patient slope). Run-in cells show pooled performance and the observed per-patient summary.",
        "",
    ]
    headers = ["Metric"]
    for cohort in COHORTS:
        headers += [
            f"{cohort['label']} model pooled",
            f"{cohort['label']} model per patient",
            f"{cohort['label']} run-in pooled (per-patient summary)",
        ]
    lines += ["| " + " | ".join(headers) + " |",
              "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|"]
    for metric in METRIC_ORDER:
        row = [METRIC_LABELS[metric]]
        for config in COHORTS:
            cohort = result["cohorts"][config["key"]]
            event_rate = _population_event_rate(cohort)
            model_metric = cohort["model"]["metrics"][metric]
            runin_metric = cohort["runin"]["metrics"][metric]
            chance = cohort["permutation_chance"]["model"][metric]
            delta = cohort["pooled_minus_per_patient"][metric]
            pooled_text = _pooled_value_cell(metric, model_metric["pooled"], event_rate)
            if delta["interval_includes_zero"]:
                pooled_text += "†"
            row.append(pooled_text)
            row.append(
                f"{_per_patient_value(metric, model_metric)} "
                f"[{_permutation_cell(chance)}]"
            )
            if metric == "calibration_slope":
                runin_patient = "not estimable (constant forecast)"
            else:
                runin_patient = _per_patient_value(metric, runin_metric)
            row.append(
                f"{_pooled_value_cell(metric, runin_metric['pooled'], event_rate)}; "
                f"{runin_patient}"
            )
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "† The paired patient-bootstrap 95% interval for pooled minus per-patient mean includes zero.",
        "",
        "Yardsticks: pooled Brier skill cells use `training_cohort_rate`; per-patient Brier skill uses the updated rate as 1 - equal-patient mean model Brier loss / equal-patient mean updated-rate Brier loss; per-patient net benefit is model NB minus treat-all NB `r̄ - (1-r̄)t/(1-t)` when that patient's mean row-level updated rate `r̄ >= t`, otherwise treat-none NB (0).",
        "Calibration slope is one conditional within-patient slope fitted with patient intercepts.",
        "",
        "Eligibility counts are shown as model/run-in. For AUC, average precision, sensitivity, specificity and calibration slope, eligibility follows the nine-metric package definitions; PPV requires at least one alarm and NPV at least one non-alarm. Net benefit includes all patients with usable forecasts; Brier skill excludes patients with a zero yardstick Brier score.",
        "",
    ]
    for metric in METRIC_ORDER:
        counts = []
        for config in COHORTS:
            cohort = result["cohorts"][config["key"]]
            model_n = cohort["model"]["metrics"][metric]["summary"]["n"]
            runin_n = cohort["runin"]["metrics"][metric]["summary"]["n"]
            counts.append(f"{config['label']} {model_n}/{runin_n}")
        lines.append(f"- **{METRIC_LABELS[metric]} eligible n (model/run-in):** "
                     + "; ".join(counts) + ".")
    thresholds = "; ".join(
        f"{config['label']} {_fmt(result['cohorts'][config['key']]['threshold'], 6)}"
        for config in COHORTS
    )
    lines += [
        "",
        f"Alarm thresholds equal each training-cohort event rate; values from the forecast files (and `cases_tables.json`): {thresholds}.",
        "The updated-rate yardsticks are REPLACE-BG `expanding_rate`, CTN-0094 `updating_rate` (an alias of the rebuilt `expanding_rate`), and College Experience `expanding_rate`.",
        "",
        "## Run-in-rate yardstick sensitivity",
        "",
        "Brier skill and net benefit are recomputed with run-in rate as the row-aligned yardstick. PPV/NPV patient excesses are re-expressed against each patient's run-in event/non-event rate; pooled PPV/NPV remain raw pooled predictive values and show the test-set population base rate in brackets.",
        "",
        "| Metric | REPLACE-BG model (pooled / patient mean) | REPLACE-BG run-in (pooled / patient mean) | CTN-0094 model (pooled / patient mean) | CTN-0094 run-in (pooled / patient mean) | College Experience model (pooled / patient mean) | College Experience run-in (pooled / patient mean) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    sensitivity_labels = {
        "brier_skill": "Brier skill",
        "ppv": "PPV excess vs run-in",
        "npv": "NPV excess vs run-in",
        "net_benefit": "Net benefit",
    }
    for metric in ("brier_skill", "ppv", "npv", "net_benefit"):
        row = [sensitivity_labels[metric]]
        for config in COHORTS:
            cohort = result["cohorts"][config["key"]]
            for cell in ("model", "runin"):
                value = cohort["runin_yardstick_sensitivity"][cell][metric]
                if metric in ("ppv", "npv"):
                    pooled = value["pooled_raw_value"]
                    mean = value["summary"]["mean"]
                else:
                    pooled = value["pooled"]
                    mean = value["summary"]["mean"]
                pooled_text = _pooled_value_cell(metric, pooled,
                                                 _population_event_rate(cohort))
                row.append(f"{pooled_text} / {_fmt(mean)}")
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "Run-in-yardstick per-patient permutation means (2.5th–97.5th percentile) for the model are in `table4.json`; the primary table's bracketed chance levels use the rebuilt updated-rate yardstick.",
        "",
    ]
    return "\n".join(lines)


def render_only():
    """Rewrite the Markdown from the final JSON without fitting or resampling."""
    json_path = OUTPUT_DIR / "table4.json"
    markdown_path = OUTPUT_DIR / "table4.md"
    result = json.loads(json_path.read_text(encoding="utf-8"))
    markdown_path.write_text(build_markdown(result), encoding="utf-8")
    print(f"Read {json_path}")
    print(f"Wrote {markdown_path}")


def run(quick=False):
    _load_modeling_dependencies()
    pm = load_permutation_module()
    n_perm = 5 if quick else pm.N_PERM
    n_bootstrap = 100 if quick else BOOTSTRAP_DRAWS
    source_results = json.loads(CASES_TABLES.read_text(encoding="utf-8"))
    forecast_frames = {}
    thresholds = {}
    for config in COHORTS:
        frame = pd.read_csv(ROOT / config["forecast_file"])
        required = {"patient", "t", "y", "threshold", "training_cohort_rate", "runin_rate",
                    "gradient_boosting", config["updated_rate_column"]}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{config['forecast_file']}: missing columns {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{config['forecast_file']}: no test forecasts")
        threshold_values = frame.threshold.to_numpy(float)
        threshold = float(threshold_values[0])
        if not np.allclose(threshold_values, threshold, rtol=0, atol=1e-12):
            raise ValueError(f"{config['forecast_file']}: threshold is not constant")
        if not np.allclose(frame.training_cohort_rate.to_numpy(float), threshold,
                           rtol=0, atol=1e-12):
            raise AssertionError(
                f"{config['label']}: cases_tables alarm threshold does not equal the "
                "training-cohort event rate"
            )
        table_threshold = float(source_results[config["cases_tables_key"]]["threshold"])
        if not np.isclose(threshold, table_threshold, rtol=0, atol=1e-12):
            raise AssertionError(f"{config['label']}: threshold differs from cases_tables.json")
        cases_training_rate = float(
            source_results[config["cases_tables_key"]]["forecasts"]
            ["training_cohort_rate"]["threshold"]["threshold"]
        )
        if not np.isclose(threshold, cases_training_rate, rtol=0, atol=1e-12):
            raise AssertionError(
                f"{config['label']}: forecast training_cohort_rate differs from the "
                "cases_tables.json training_cohort_rate threshold"
            )
        thresholds[config["key"]] = threshold
        forecast_frames[config["key"]] = frame

    prepared = {
        "replace_bg": pm.fit_replace_bg(),
        "ctn0094": pm.fit_ctn(),
        "ces": pm.fit_ces(),
    }

    rng_perm = np.random.default_rng(pm.SEED)
    rng_bootstrap = np.random.default_rng(BOOTSTRAP_SEED)
    output = {
        "metadata": {
            "title": "Pooled versus per-patient clinical forecast performance",
            "metric_order": list(METRIC_ORDER),
            "metric_source": "wheneval.nine.nine_metrics, with pooled Brier skill re-referenced to training_cohort_rate",
            "updated_rate_yardsticks": {
                config["key"]: config["updated_rate_column"] for config in COHORTS
            },
            "runin_forecast_column": "runin_rate",
            "training_cohort_rate_column": "training_cohort_rate",
            "training_cohort_rates": thresholds,
            "alarm_thresholds": thresholds,
            "permutation_draws": n_perm,
            "permutation_seed": int(pm.SEED),
            "permutation_method": "Within-patient outcome shuffles; imported fixed fitted models are rescored and the updated-rate history yardstick is rebuilt on every draw.",
            "bootstrap_draws": n_bootstrap,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_method": "Paired whole-patient resampling; repeated sampled patients are relabelled as separate bootstrap patients; the Brier-skill interval is pooled training-cohort-rate skill minus the updated-rate equal-patient ratio of mean losses, recomputed on each same draw.",
            "cases_tables_json": str(CASES_TABLES.relative_to(ROOT.parent)),
            "quick_mode": bool(quick),
            "notes": [
                "The exported forecast rows are aligned to model-fixed test rows by ordered outcomes and rate histories.",
                "For CTN-0094, updating_rate is an alias of expanding_rate; the existing history rebuild emits expanding_rate, which is used as the updated-rate yardstick.",
                "Sensitivity PPV/NPV patient excesses are re-referenced to run-in rates; pooled PPV/NPV remain raw pooled predictive values because the package has no pooled PPV/NPV yardstick definition.",
            ],
        },
        "cohorts": {},
    }
    for config in COHORTS:
        key = config["key"]
        expected = source_results[config["cases_tables_key"]]["forecasts"]["gradient_boosting"]
        auxiliary = _make_auxiliary_data(pm, config, prepared[config["permutation_key"]])
        output["cohorts"][key] = _analyse_cohort(
            pm, config, prepared[config["permutation_key"]], auxiliary,
            forecast_frames[key], expected, thresholds[key], n_perm, rng_perm,
            n_bootstrap, rng_bootstrap,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "table4.json"
    markdown_path = OUTPUT_DIR / "table4.md"
    json_path.write_text(json.dumps(clean(output), indent=2, allow_nan=False), encoding="utf-8")
    markdown_path.write_text(build_markdown(clean(output)), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    for config in COHORTS:
        cohort = output["cohorts"][config["key"]]
        print(f"{config['label']}: n={cohort['test_patients']}, threshold={cohort['threshold']:.6f}, "
              f"GB AUC pooled={cohort['model']['metrics']['auc']['pooled']:.3f}, "
              f"per-patient mean={cohort['model']['metrics']['auc']['summary']['mean']:.3f}")


def helper_selftest():
    """Exercise pure metric, bootstrap, chance-summary, and sensitivity helpers."""
    _load_modeling_dependencies()
    patient = np.asarray(["a"] * 4 + ["b"] * 4)
    y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    forecast = np.asarray([0.1, 0.9, 0.2, 0.8, 0.9, 0.8, 0.1, 0.2])
    yardstick = np.full(8, 0.5)
    observed = nine_metrics(patient, y, forecast, 0.5, yardstick=yardstick)
    assert tuple(observed) == METRIC_ORDER
    assert observed["auc"]["summary"]["n"] == 2
    assert np.isclose(observed["auc"]["summary"]["median"], 0.75)

    chance = _chance_summary([0.25, 0.5, 0.75])
    assert np.isclose(chance["mean_over_draws"], 0.5)
    assert chance["percentiles_2_5_97_5"] == [0.2625, 0.7375]

    re_referenced = _re_reference_ppv_npv(
        observed["ppv"], patient, y, np.asarray([0.0] * 4 + [1.0] * 4), "ppv"
    )
    # A's zero run-in rate raises its PPV excess by .5; B's one rate lowers it by .5.
    assert np.isclose(re_referenced["per_patient"]["a"], observed["ppv"]["per_patient"]["a"] + 0.5)
    assert np.isclose(re_referenced["per_patient"]["b"], observed["ppv"]["per_patient"]["b"] - 0.5)

    toy_bundle = {
        "pid": np.asarray(["b", "b", "a", "a"]),
        "y": np.asarray([0, 1, 1, 0]),
        "predictions": {"gradient_boosting": np.asarray([0.1, 0.2, 0.7, 0.6])},
        "test": pd.DataFrame({
            "pid": ["b", "b", "a", "a"], "t": [0, 1, 0, 1],
            "y": [0, 1, 1, 0], "runin_rate": [0.25, 0.25, 0.5, 0.5],
            "expanding_rate": [0.1, 0.2, 0.3, 0.4],
        }),
    }
    toy_export = pd.DataFrame({
        "patient": [9, 9, 3, 3], "t": [1, 0, 0, 1], "y": [1, 0, 1, 0],
        "runin_rate": [0.25, 0.25, 0.5, 0.5],
        "expanding_rate": [0.2, 0.1, 0.3, 0.4],
        "gradient_boosting": [0.2, 0.1, 0.7, 0.6],
    })
    bundle_order, export_order = align_forecast_rows(
        toy_export, toy_bundle,
        {"bundle_time_column": "t", "updated_rate_column": "expanding_rate", "label": "toy"},
    )
    assert bundle_order.tolist() == [0, 1, 2, 3]
    assert export_order.tolist() == [1, 0, 2, 3]

    bootstrap = _paired_bootstrap_intervals(
        patient, y, forecast, yardstick, np.full(8, 0.25),
        0.5, 40, np.random.default_rng(17)
    )
    assert bootstrap["auc"]["valid_draws"] == 40
    assert len(bootstrap["auc"]["percentiles_2_5_97_5"]) == 2

    repeated_patient = np.asarray(["a"] * 4 + ["b"] * 4)
    repeated_y = np.asarray([0, 1, 0, 1] * 2)
    repeated_forecast = np.asarray([0.1, 0.9, 0.2, 0.8] * 2)
    updated_rate = np.full(8, 0.5)
    training_rate = np.full(8, 0.25)
    brier = nine_metrics(repeated_patient, repeated_y, repeated_forecast, 0.5,
                         yardstick=updated_rate)
    updated_pooled = brier["brier_skill"]["pooled"]
    per_patient_mean = brier["brier_skill"]["summary"]["mean"]
    _set_brier_pooled_yardsticks(brier, repeated_y, repeated_forecast, training_rate)
    population_pooled = _pooled_brier_skill(
        repeated_y, repeated_forecast, training_rate
    )
    assert np.isclose(brier["brier_skill"]["pooled"], population_pooled)
    assert np.isclose(brier["brier_skill"]["pooled_updated_rate"], updated_pooled)
    assert not np.isclose(population_pooled, updated_pooled)
    assert np.isclose(brier["brier_skill"]["summary"]["mean"], per_patient_mean)
    paired_brier = _paired_bootstrap_intervals(
        repeated_patient, repeated_y, repeated_forecast, updated_rate, training_rate,
        0.5, 40, np.random.default_rng(17)
    )["brier_skill"]
    expected_delta = population_pooled - per_patient_mean
    assert np.allclose(paired_brier["percentiles_2_5_97_5"], expected_delta)
    print("Pure helper self-check passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="Use 5 permutations and 100 bootstrap draws; models are still fitted once")
    parser.add_argument("--render-only", action="store_true",
                        help="Rewrite table4.md from the existing table4.json; do not fit or draw")
    parser.add_argument("--selftest", action="store_true",
                        help="Run pure helper checks without fitting models")
    args = parser.parse_args()
    if args.render_only:
        render_only()
    elif args.selftest:
        helper_selftest()
    else:
        run(quick=args.quick)


if __name__ == "__main__":
    main()
