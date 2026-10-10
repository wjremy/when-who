#!/usr/bin/env python3
"""Build the v10 main table comparing pooled, naive, and tailored scoring.

Run from the repository root with the project Python environment:
    python open_data/main_table_v10.py [--quick]

The shuffled running-rate forecast rebuilds only the history rate on each
within-patient outcome permutation. It does not refit or rescore a predictive
model.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OPEN_DATA = ROOT / "open_data"
OUTPUT_DIR = OPEN_DATA / "results" / "v10"
METRIC_ORDER = (
    "auc", "average_precision", "sensitivity", "specificity", "ppv", "npv",
    "net_benefit", "brier_skill", "calibration_slope",
)
METRIC_LABELS = {
    "auc": "AUC",
    "average_precision": "Average precision",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "ppv": "PPV",
    "npv": "NPV",
    "net_benefit": "Net benefit",
    "brier_skill": "Brier skill",
    "calibration_slope": "Calibration slope",
}
SCORING_ORDER = ("pooled", "naive", "tailored")
FORECAST_ORDER = ("runin", "shuffled_running")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="use 10 within-patient permutations instead of 200")
    parser.add_argument("--draws", type=int, default=None,
                        help="number of within-patient permutations (default 200; quick 10)")
    args = parser.parse_args()
    args.draws = args.draws if args.draws is not None else (10 if args.quick else 200)
    if args.draws < 1:
        parser.error("--draws must be at least 1")
    return args


def _load_dependencies():
    """Load NumPy, pandas, metric functions, and Table 4 lazily."""
    if str(OPEN_DATA) not in sys.path:
        sys.path.insert(0, str(OPEN_DATA))
    import table4_v10 as table4

    # Table 4's dependency loader imports no estimators; model-fitting remains
    # local to the unused fit_* functions in cases_model_fixed_permutation.py.
    table4._load_modeling_dependencies()
    code_src = ROOT / "code" / "withinpatient" / "src"
    if str(code_src) not in sys.path:
        sys.path.insert(0, str(code_src))
    from withinpatient.nine import naive_metrics

    return table4, table4.np, table4.pd, table4.nine_metrics, naive_metrics


def _load_context(pm, table4, config, np, pd):
    """Recreate the same test panel and auxiliary history panel as Table 4."""
    key = config["permutation_key"]
    if key == "replace_bg":
        panel = pd.read_csv(OPEN_DATA / "replace_bg_panel.csv")
        panel["pid"] = panel.pid.astype(str)
        panel["y"] = panel.y.astype(int)
        scored = panel.loc[~panel.model_runin.astype(bool)].copy()
        _, _, test_ids = pm.CASES.split_replace(scored)
        test = scored.copy()
        test["day"] = pd.to_numeric(test.day, errors="coerce")
        test = test.loc[test.pid.isin(test_ids)].copy()
        auxiliary = {"panel": panel}
        patient = test.pid.to_numpy()
        y = test.y.to_numpy(int)
    elif key == "ctn0094":
        raw = pd.read_csv(OPEN_DATA / "ctn0094_weekly_panel.csv")
        split = pm.CTN.split_people(raw)
        event_col = "event"
        train_y = raw.loc[
            raw.who.isin(split["train"]) & raw[event_col].notna(), event_col
        ].astype(int)
        train_rate = float(train_y.mean()) if len(train_y) else 0.5
        data = pm.CTN.add_references(raw, event_col, train_rate)
        test = data.loc[
            data.who.isin(split["test"]) & data[event_col].notna()
        ].copy()
        auxiliary = {"data": data, "event_col": event_col}
        patient = test.who.to_numpy()
        y = test[event_col].astype(int).to_numpy()
    elif key == "ces":
        _, panel = pm.CES.load_reports(pm.CES.DATA, 3)
        split = pm.CES.split_people(panel)
        panel = panel.copy()
        panel["split"] = panel.pid.map(split)
        test = panel.loc[panel.split.eq("test")].copy().reset_index(drop=True)
        auxiliary = {}
        patient = test.pid.to_numpy()
        y = test.y.to_numpy(int)
    else:
        raise ValueError(f"Unsupported Table 4 cohort: {key}")

    bundle = {"test": test, "pid": patient, "y": y}
    updated = table4.rebuild_updated_rate(pm, config, bundle, y, auxiliary)
    return {"config": config, "bundle": bundle, "auxiliary": auxiliary,
            "updated_observed": np.asarray(updated, dtype=float)}


def _file_groups(frame, pd, np):
    groups = []
    for key in pd.unique(frame["patient"].to_numpy()):
        ix = np.flatnonzero(frame.patient.to_numpy() == key)
        times = frame.iloc[ix]["t"].to_numpy()
        ix = ix[np.argsort(times, kind="stable")]
        groups.append((key, ix))
    return groups


def _align_context(context, frame, np, pd):
    """Align CSV rows to the source test rows using Table 4's identity checks."""
    config = context["config"]
    bundle = context["bundle"]
    test = bundle["test"]
    bundle_groups = []
    pid = np.asarray(bundle["pid"])
    for key in pd.unique(pid):
        ix = np.flatnonzero(pid == key)
        if config["bundle_time_column"] in test.columns:
            times = test.iloc[ix][config["bundle_time_column"]].to_numpy()
            ix = ix[np.argsort(times, kind="stable")]
        bundle_groups.append((key, ix))
    file_groups = _file_groups(frame, pd, np)
    if len(bundle_groups) != len(file_groups):
        raise AssertionError(
            f"{config['label']}: CSV has {len(file_groups)} patients; source test set has "
            f"{len(bundle_groups)}"
        )

    file_rate_col = config["updated_rate_column"]
    panel_rate_col = "expanding_rate"
    bundle_runin = test.runin_rate.to_numpy(float)
    bundle_rate = context["updated_observed"]
    used: set[int] = set()
    bundle_order: list[int] = []
    file_order: list[int] = []
    for group_number, (_, bix) in enumerate(bundle_groups):
        candidates = []
        for file_number, (_, fix) in enumerate(file_groups):
            if file_number in used or len(fix) != len(bix):
                continue
            candidate = frame.iloc[fix]
            if not np.array_equal(candidate.y.to_numpy(int), bundle["y"][bix]):
                continue
            if not np.allclose(candidate.runin_rate.to_numpy(float), bundle_runin[bix],
                               rtol=0, atol=1e-10):
                continue
            if not np.allclose(candidate[file_rate_col].to_numpy(float), bundle_rate[bix],
                               rtol=0, atol=1e-10):
                continue
            candidates.append((file_number, fix))
        if not candidates:
            raise AssertionError(
                f"{config['label']}: could not align panel histories with Table 4 forecast rows"
            )
        chosen = next((item for item in candidates if item[0] == group_number), candidates[0])
        file_number, fix = chosen
        used.add(file_number)
        bundle_order.extend(bix.tolist())
        file_order.extend(fix.tolist())

    bundle_order = np.asarray(bundle_order, dtype=int)
    file_order = np.asarray(file_order, dtype=int)
    if len(bundle_order) != len(frame) or len(file_order) != len(frame):
        raise AssertionError(f"{config['label']}: alignment did not retain all scored rows")
    return bundle_order, file_order


def _as_scalar(value: Any) -> float | str | None:
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _draw_summary(values: list[Any], np) -> dict[str, Any]:
    finite = []
    text_value = None
    for value in values:
        scalar = _as_scalar(value)
        if isinstance(scalar, (int, float)) and math.isfinite(float(scalar)):
            finite.append(float(scalar))
        elif isinstance(scalar, str) and text_value is None:
            text_value = scalar
    if not finite:
        return {"mean": text_value, "percentiles_2_5_97_5": [None, None],
                "valid_draws": 0, "draw_values": values}
    return {
        "mean": float(np.mean(finite)),
        "percentiles_2_5_97_5": [float(np.percentile(finite, 2.5)),
                                   float(np.percentile(finite, 97.5))],
        "valid_draws": len(finite),
        "draw_values": values,
    }


def _eligible_summary(counts: list[Any]) -> Any:
    if not counts:
        return None
    if isinstance(counts[0], dict):
        return {key: _eligible_summary([item[key] for item in counts])
                for key in counts[0]}
    low, high = min(counts), max(counts)
    return low if low == high else [low, high]


def _single_cell(value: Any, chance: Any, eligible_n: Any,
                 chance_definition: str | None = None) -> dict[str, Any]:
    return {"mean": _as_scalar(value), "percentiles_2_5_97_5": [None, None],
            "valid_draws": 1 if _as_scalar(value) is not None else 0,
            "eligible_n": eligible_n, "chance": chance,
            "chance_definition": chance_definition}


def _draw_cell(values: list[Any], counts: list[Any], chance: Any,
               np, chance_definition: str | None = None) -> dict[str, Any]:
    summary = _draw_summary(values, np)
    summary["eligible_n"] = _eligible_summary(counts)
    summary["chance"] = chance
    summary["chance_definition"] = chance_definition
    return summary


def _pool_brier(y, forecast, training_rate, np) -> float:
    reference = np.asarray(training_rate, dtype=float)
    if reference.ndim == 0:
        reference = np.full(len(y), float(reference))
    denominator = float(np.mean((np.asarray(y, float) - reference) ** 2))
    if denominator <= 0:
        return float("nan")
    return float(1 - np.mean((np.asarray(y, float) - np.asarray(forecast, float)) ** 2)
                 / denominator)


def _metric_value(result, metric: str, scoring: str, y, forecast,
                  training_rate, np):
    if scoring == "pooled":
        value = result[metric]["pooled"]
        if metric == "brier_skill":
            value = _pool_brier(y, forecast, training_rate, np)
        return _as_scalar(value), len(y)
    summary = result[metric]["summary"]
    eligible = int(summary.get("n", 0))
    if metric == "calibration_slope":
        fit = result[metric].get("conditional_fit", {})
        eligible = int(fit.get("patients", eligible)) if fit.get("estimable") else 0
    return _as_scalar(summary.get("mean")), eligible


def _chance_levels(threshold: float, observed_y, np) -> dict[str, Any]:
    base_rate = float(np.mean(observed_y))
    treat_all = None
    if 0 < threshold < 1:
        treat_all = base_rate - (1 - base_rate) * threshold / (1 - threshold)
    return {
        "auc": 0.5,
        "average_precision": base_rate,
        "sensitivity": threshold,
        "specificity": 1 - threshold,
        "ppv": base_rate,
        "npv": 1 - base_rate,
        "net_benefit": {"treat_none": 0.0, "treat_all": treat_all},
        "brier_skill": 0.0,
        "calibration_slope": 1.0,
    }


def _analyse_context(context, frame, table4, pm, np, pd, nine_metrics,
                     naive_metrics, n_draws, rng_perm):
    config = context["config"]
    bundle = context["bundle"]
    bundle_order, file_order = _align_context(context, frame, np, pd)
    patient = np.asarray(bundle["pid"])[bundle_order]
    y = frame.iloc[file_order].y.to_numpy(int)
    runin = frame.iloc[file_order].runin_rate.to_numpy(float)
    updated = frame.iloc[file_order][config["updated_rate_column"]].to_numpy(float)
    threshold_values = frame.iloc[file_order].threshold.to_numpy(float)
    training_rates = frame.iloc[file_order].training_cohort_rate.to_numpy(float)
    threshold = float(threshold_values[0])
    if not np.allclose(threshold_values, threshold, rtol=0, atol=1e-12):
        raise ValueError(f"{config['label']}: threshold varies across rows")
    if not np.allclose(training_rates, threshold, rtol=0, atol=1e-12):
        raise AssertionError(f"{config['label']}: threshold is not the training-cohort rate")

    # This also confirms the current Table 4 rate column is reconstructed by
    # exactly the same history builder used for the shuffled draws.
    source_updated = context["updated_observed"][bundle_order]
    if not np.allclose(source_updated, updated, rtol=0, atol=1e-10):
        raise AssertionError(f"{config['label']}: observed updated-rate identity check failed")
    if not np.allclose(bundle["test"].runin_rate.to_numpy(float)[bundle_order], runin,
                       rtol=0, atol=1e-10):
        raise AssertionError(f"{config['label']}: run-in forecasts do not align")

    runin_result = nine_metrics(patient, y, runin, threshold, yardstick=updated)
    observed_results = {
        "pooled": runin_result,
        "naive": naive_metrics(patient, y, runin, threshold, population_rate=threshold),
        "tailored": runin_result,
    }
    chance = _chance_levels(threshold, y, np)
    cells: dict[str, dict[str, Any]] = {}
    youden: dict[str, Any] = {}
    observed_metrics = {}
    for scoring in SCORING_ORDER:
        per_metric = {}
        for metric in METRIC_ORDER:
            value, n_eligible = _metric_value(
                observed_results[scoring], metric, scoring, y, runin,
                training_rates, np,
            )
            if scoring == "naive":
                metric_chance = observed_results["naive"][metric].get("chance_level")
                if metric == "calibration_slope" and isinstance(value, (int, float)):
                    metric_chance = 1.0
            elif scoring == "tailored":
                metric_chance = value
            else:
                metric_chance = chance[metric]
            per_metric[metric] = _single_cell(
                value, metric_chance, n_eligible,
                "score of this no-within-patient-information forecast" if scoring == "tailored"
                else None,
            )
        observed_metrics[scoring] = per_metric
        if scoring == "naive":
            youden["runin"] = _single_cell(
                observed_results["naive"]["naive_youden"], 0.0,
                {
                    "sensitivity": observed_results["naive"]["sensitivity"]["summary"]["n"],
                    "specificity": observed_results["naive"]["specificity"]["summary"]["n"],
                },
            )

    # Seed the three cohorts in Table 4's existing cohort order, preserving its
    # 20261009 stream while the output is rendered in paper order.
    target_draws = pm.generate_permuted_targets(
        bundle["y"], bundle["pid"], n_draws, rng_perm
    )
    draw_values = {scoring: {metric: [] for metric in METRIC_ORDER}
                   for scoring in SCORING_ORDER}
    draw_counts = {scoring: {metric: [] for metric in METRIC_ORDER}
                   for scoring in SCORING_ORDER}
    draw_youden: list[Any] = []
    draw_youden_n: list[dict[str, int]] = []
    for target in target_draws:
        rebuilt = table4.rebuild_updated_rate(
            pm, config, bundle, target, context["auxiliary"]
        )
        forecast = np.asarray(rebuilt, dtype=float)[bundle_order]
        permuted_y = np.asarray(target, dtype=int)[bundle_order]
        result_rate = nine_metrics(patient, permuted_y, forecast, threshold,
                                   yardstick=forecast)
        result_naive = naive_metrics(patient, permuted_y, forecast, threshold,
                                     population_rate=threshold)
        results = {"pooled": result_rate, "naive": result_naive,
                   "tailored": result_rate}
        for scoring in SCORING_ORDER:
            for metric in METRIC_ORDER:
                value, n_eligible = _metric_value(
                    results[scoring], metric, scoring, permuted_y, forecast,
                    training_rates, np,
                )
                draw_values[scoring][metric].append(value)
                draw_counts[scoring][metric].append(n_eligible)
        draw_youden.append(result_naive["naive_youden"])
        draw_youden_n.append({
            "sensitivity": result_naive["sensitivity"]["summary"]["n"],
            "specificity": result_naive["specificity"]["summary"]["n"],
        })

    for scoring in SCORING_ORDER:
        cells[scoring] = observed_metrics[scoring]
        cells[scoring].update({
            "shuffled_running": _build_draw_metric_cells(
                draw_values[scoring], draw_counts[scoring], scoring, chance, np
            )
        })
    # Reshape to metric -> forecast -> scoring, the table's natural structure.
    metrics = {}
    for metric in METRIC_ORDER:
        metrics[metric] = {
            "label": METRIC_LABELS[metric],
            "cells": {
                "runin_pooled": observed_metrics["pooled"][metric],
                "runin_naive": observed_metrics["naive"][metric],
                "runin_tailored": observed_metrics["tailored"][metric],
                "shuffled_running_pooled": cells["pooled"]["shuffled_running"][metric],
                "shuffled_running_naive": cells["naive"]["shuffled_running"][metric],
                "shuffled_running_tailored": cells["tailored"]["shuffled_running"][metric],
            },
        }
    youden["shuffled_running"] = _draw_cell(draw_youden, draw_youden_n, 0.0, np)
    return {
        "label": config["label"],
        "forecast_file": config["forecast_file"],
        "updated_rate_column": config["updated_rate_column"],
        "threshold": threshold,
        "training_cohort_rate": float(training_rates[0]),
        "test_event_rate": float(np.mean(y)),
        "test_patients": int(pd.unique(patient).size),
        "scored_rows": int(len(y)),
        "permutation_draws": int(n_draws),
        "metrics": metrics,
        "naive_youden": youden,
    }


def _build_draw_metric_cells(draw_values, draw_counts, scoring, chance, np):
    output = {}
    for metric in METRIC_ORDER:
        if scoring == "naive":
            # All draws have the same Table 4 threshold/population-rate chance.
            naive_chance = {
                "auc": 0.5,
                "average_precision": None,
                "sensitivity": None,
                "specificity": None,
                "ppv": None,
                "npv": None,
                "net_benefit": 0.0,
                "brier_skill": 0.0,
                "calibration_slope": 1.0,
            }[metric]
            if metric in ("average_precision", "sensitivity", "ppv"):
                naive_chance = chance["sensitivity"]
            elif metric in ("specificity", "npv"):
                naive_chance = chance["specificity"]
            metric_chance = naive_chance
        elif scoring == "tailored":
            metric_chance = _draw_summary(draw_values[metric], np)
            metric_chance = {key: metric_chance[key] for key in
                             ("mean", "percentiles_2_5_97_5", "valid_draws")}
        else:
            metric_chance = chance[metric]
        output[metric] = _draw_cell(
            draw_values[metric], draw_counts[metric], metric_chance, np,
            "the shuffled running-rate forecast itself" if scoring == "tailored" else None,
        )
    return output


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def _format_stat(summary: dict[str, Any]) -> str:
    mean = summary.get("mean")
    if isinstance(mean, str) or mean is None:
        return _format_number(mean)
    low, high = summary.get("percentiles_2_5_97_5", [None, None])
    if low is None or high is None:
        return _format_number(mean)
    return f"{_format_number(mean)} ({_format_number(low)}, {_format_number(high)})"


def _format_chance(chance: Any) -> str:
    if isinstance(chance, dict) and "treat_none" in chance:
        return (f"{_format_number(chance.get('treat_none'))} / "
                f"{_format_number(chance.get('treat_all'))}")
    if isinstance(chance, dict) and "mean" in chance:
        return _format_stat(chance)
    return _format_number(chance)


def _format_cell(cell: dict[str, Any]) -> str:
    return f"{_format_stat(cell)} [{_format_chance(cell.get('chance'))}]"


def _format_eligible(value: Any) -> str:
    if isinstance(value, dict):
        return f"sens {value.get('sensitivity')}; spec {value.get('specificity')}"
    if isinstance(value, list) and len(value) == 2:
        return f"{value[0]}–{value[1]}"
    return str(value)


def build_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Main table: forecasts without knowledge of the day",
        "",
        "The run-in forecast is each patient's fixed run-in event rate. For the "
        "shuffled running-rate forecast, scored outcomes are shuffled separately "
        "within each patient, run-in outcomes stay fixed, and the expanding rate "
        "is rebuilt on each draw with Table 4's history builder.",
        "",
        "For shuffled forecasts, cells show the mean across draws and the "
        "2.5th–97.5th percentiles. Pooled values are row weighted; their Brier "
        "skill uses the training-cohort rate, as in Table 4. Pooled AUC uses 0.5; "
        "pooled AP, PPV, and NPV use the scored-set event/non-event rate; pooled "
        "sensitivity and specificity use the training-cohort rate and its "
        "complement; pooled net benefit shows treat-none / treat-all; pooled "
        "Brier skill uses 0; pooled calibration slope uses 1.",
        "",
        "Naive values are equal-patient means from `naive_metrics`; sensitivity "
        "and specificity use separate eligible patient groups, and Youden is "
        "their means added minus one, read against 0. Naive chance levels follow "
        "the training-cohort rate in `naive_metrics`; an estimable naive "
        "calibration slope is read against the calibrated-twin slope of 1, while "
        "the constant run-in slope is n/a. Tailored values use "
        "`nine_metrics`: run-in is referenced to Table 4's updated-rate column, "
        "while each shuffled draw uses its rebuilt updated rate. The tailored "
        "bracket is the score of that same no-within-patient-information forecast "
        "by construction.",
        "",
    ]
    headings = (
        "Run-in pooled", "Run-in naive patient", "Run-in tailored patient",
        "Shuffled running pooled", "Shuffled running naive patient",
        "Shuffled running tailored patient",
    )
    for label in ("CTN-0094", "REPLACE-BG", "College Experience"):
        cohort = result["datasets"][label]
        lines.extend([
            f"## {label}", "",
            f"{cohort['test_patients']} test patients; {cohort['scored_rows']} scored rows; "
            f"training-cohort rate / alarm threshold {cohort['threshold']:.6f}; "
            f"scored-set event rate {cohort['test_event_rate']:.6f}.", "",
            "| Metric | " + " | ".join(headings) + " |",
            "|---|" + "---:|" * len(headings),
        ])
        for metric in METRIC_ORDER:
            cells = cohort["metrics"][metric]["cells"]
            names = (
                "runin_pooled", "runin_naive", "runin_tailored",
                "shuffled_running_pooled", "shuffled_running_naive",
                "shuffled_running_tailored",
            )
            lines.append("| " + METRIC_LABELS[metric] + " | " +
                         " | ".join(_format_cell(cells[name]) for name in names) + " |")
        y_a = cohort["naive_youden"]["runin"]
        y_b = cohort["naive_youden"]["shuffled_running"]
        short_labels = {
            "auc": "AUC", "average_precision": "AP", "sensitivity": "sens",
            "specificity": "spec", "ppv": "PPV", "npv": "NPV",
            "net_benefit": "NB", "brier_skill": "Brier", "calibration_slope": "slope",
        }
        eligibility_lines = []
        for scoring in ("naive", "tailored"):
            entries = []
            for metric in METRIC_ORDER:
                metric_cells = cohort["metrics"][metric]["cells"]
                n_a = _format_eligible(metric_cells[f"runin_{scoring}"]["eligible_n"])
                n_b = _format_eligible(
                    metric_cells[f"shuffled_running_{scoring}"]["eligible_n"]
                )
                entries.append(f"{short_labels[metric]} {n_a}/{n_b}")
            eligibility_lines.append(f"{scoring}: " + ", ".join(entries))
        youden_na = _format_eligible(y_a.get("eligible_n"))
        youden_shuffle = _format_eligible(y_b.get("eligible_n"))
        lines.extend([
            f"*Eligible patient n (run-in / shuffled running):* {'; '.join(eligibility_lines)}. "
            f"Youden groups: run-in {youden_na}; shuffled {youden_shuffle}.",
            "",
        ])
        lines.extend([
            "",
            f"*Naive Youden (separate sensitivity/specificity groups; chance 0):* "
            f"run-in {_format_cell(y_a)}; shuffled running {_format_cell(y_b)}.",
            "",
            "*Eligibility and definitions:* pooled scores use all scored rows where "
            "the metric is defined. Per-patient eligible counts are in each JSON "
            "cell's `eligible_n` (for shuffled draws, a minimum–maximum if it "
            "varies). Naive sensitivity uses patients with at least one event; "
            "naive specificity uses patients with at least one non-event; naive "
            "AP uses patients with an event; naive PPV/NPV require an alarm/no "
            "alarm. Tailored AUC/AP/sensitivity/specificity require both outcome "
            "classes; tailored PPV/NPV require an alarm/no alarm; tailored Brier "
            "skill requires a nonzero yardstick loss; calibration slope requires "
            "within-patient forecast variation. PPV/NPV are raw for pooled and "
            "naive scoring, and tailored PPV/NPV are excesses over each patient's "
            "observed event/non-event rate. Net benefit is raw versus treat-none "
            "for pooled and naive scoring, and an excess over the yardstick "
            "policy for tailored scoring. Brier skill summaries use the ratio of "
            "mean patient losses.",
            "",
        ])
    return "\n".join(lines)


def _clean(value: Any, np):
    if isinstance(value, dict):
        return {str(key): _clean(item, np) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item, np) for item in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist(), np)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def run(n_draws: int, quick: bool) -> dict[str, Any]:
    started = time.perf_counter()
    table4, np, pd, nine_metrics, naive_metrics = _load_dependencies()
    pm = table4.load_permutation_module()
    frames = {}
    contexts = {}
    for config in table4.COHORTS:
        frame = pd.read_csv(OPEN_DATA / config["forecast_file"])
        required = {"patient", "t", "y", "threshold", "training_cohort_rate",
                    "runin_rate", config["updated_rate_column"]}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{config['forecast_file']}: missing {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{config['forecast_file']}: no scored rows")
        frames[config["key"]] = frame
        contexts[config["key"]] = _load_context(pm, table4, config, np, pd)

    rng_perm = np.random.default_rng(pm.SEED)
    datasets_by_table4_order = {}
    for config in table4.COHORTS:
        key = config["key"]
        datasets_by_table4_order[config["label"]] = _analyse_context(
            contexts[key], frames[key], table4, pm, np, pd, nine_metrics,
            naive_metrics, n_draws, rng_perm,
        )
    requested_order = ["CTN-0094", "REPLACE-BG", "College Experience"]
    datasets = {label: datasets_by_table4_order[label] for label in requested_order}
    result = {
        "title": "Pooled, naive, and tailored scoring of forecasts without knowledge of the day",
        "quick_mode": bool(quick),
        "metadata": {
            "metric_order": list(METRIC_ORDER),
            "dataset_output_order": requested_order,
            "permutation_draws": int(n_draws),
            "permutation_seed": int(pm.SEED),
            "permutation_rng_stream": "cases_model_fixed_permutation.generate_permuted_targets, with Table 4 cohort order REPLACE-BG, CTN-0094, College Experience",
            "permutation_method": "Within-patient shuffle of scored outcomes; run-in outcomes fixed; updated/expanding rate rebuilt with table4_v10.rebuild_updated_rate; no model fitting or model rescoring.",
            "metrics_source": "withinpatient.nine.nine_metrics and naive_metrics",
            "pooled_brier_skill_reference": "training_cohort_rate, as in Table 4",
            "pooled_chance_definitions": "AUC 0.5; AP/PPV scored-set event rate; NPV scored-set non-event rate; sensitivity/specificity training-cohort rate/complement; NB treat-none/treat-all; Brier skill 0; calibration slope 1.",
            "naive_slope_chance_definition": "n/a for the constant run-in forecast; slope 1 (calibrated twin) when the shuffled running forecast yields an estimable within-patient slope",
            "tailored_chance_definition": "the score of the no-within-patient-information forecast itself; for run-in by construction and for shuffled running rate each draw's rebuilt forecast",
            "elapsed_seconds": float(time.perf_counter() - started),
        },
        "datasets": datasets,
    }
    cleaned = _clean(result, np)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "main_table.json"
    markdown_path = OUTPUT_DIR / "main_table.md"
    json_path.write_text(json.dumps(cleaned, indent=2, allow_nan=False), encoding="utf-8")
    markdown_path.write_text(build_markdown(cleaned), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    print(f"Draws: {n_draws}; elapsed: {cleaned['metadata']['elapsed_seconds']:.1f} s")
    return cleaned


def main() -> None:
    args = parse_args()
    run(args.draws, args.quick)


if __name__ == "__main__":
    main()
