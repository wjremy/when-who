#!/usr/bin/env python3
"""Add patient-level mean excess beside median excess for the v10/v16 results.

Run from the repository root with the full project Python environment:
    python code/v19_mean_excess.py                 # stored results only
    python code/v19_mean_excess.py --rerun-missing  # fit/rescore missing patient baselines

The stored-only pass never fits models.  In the rerun pass the original
pipelines and their seed streams are imported and reused.  Patient-level
"pooled" scoring is not emitted for Section 5.2: pooled is a cohort-level,
row-weighted quantity, while this file summarizes patients.  The two
patient-level scoring definitions there are naive and tailored.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OPEN_DATA = ROOT / "open_data"
RESULTS = ROOT / "results" / "v19"
CSV_PATH = RESULTS / "A1_mean_excess.csv"
PATIENT_PATH = RESULTS / "A1_patient_excess.json"
TABLE4_PATH = OPEN_DATA / "results" / "v10" / "table4.json"
PACKAGE_PATH = OPEN_DATA / "results" / "v10" / "fitted_model_package.json"
BLOCK_PATH = OPEN_DATA / "results" / "v16" / "block_results.json"
MAIN_PATH = OPEN_DATA / "results" / "v10" / "main_table.json"
TASK1_PATIENT_PATH = ROOT / "work" / "2026-10-13_v18_analyses" / "task1_per_patient.json"
TASK1_CSV_PATH = ROOT / "work" / "2026-10-13_v18_analyses" / "task1_fitted_models.csv"

METRICS = (
    "auc", "average_precision", "sensitivity", "specificity", "ppv", "npv",
    "net_benefit", "brier_skill", "calibration_slope",
)
TABLE5_DATASETS = {
    "replace_bg": "REPLACE-BG",
    "ctn_weekly_missing_excluded": "CTN-0094",
    "ces": "College Experience Study",
}
BLOCK_DATASETS = {
    "replace_bg": "REPLACE-BG",
    "ctn0094": "CTN-0094",
    "ces": "College Experience Study",
}
CSV_FIELDS = (
    "set", "dataset", "model", "measure", "scoring_block", "n_patients",
    "median_excess", "iqr_q1", "iqr_q3", "mean_excess", "se_mean",
    "median_as_reported", "median_matches_reported", "availability", "source",
)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _key(*parts: str) -> tuple[str, ...]:
    return tuple(str(p) for p in parts)


def _add_group(groups: dict[tuple[str, ...], dict[str, Any]], group: dict[str, Any]) -> None:
    key = _key(group["set"], group["dataset"], group["model"],
               group["measure"], group["scoring_block"])
    groups[key] = group


def _missing_group(set_name: str, dataset: str, model: str, measure: str,
                   scoring_block: str, source: str) -> dict[str, Any]:
    return {
        "set": set_name, "dataset": dataset, "model": model,
        "measure": measure, "scoring_block": scoring_block,
        "patient_values": None, "median_as_reported": None,
        "availability": "missing_per_patient_values", "source": source,
    }


def _summary_group(set_name: str, dataset: str, model: str, measure: str,
                   scoring_block: str, n: int, median: float,
                   q1: float, q3: float, mean: float, source: str) -> dict[str, Any]:
    return {
        "set": set_name, "dataset": dataset, "model": model,
        "measure": measure, "scoring_block": scoring_block,
        "patient_values": None, "n_patients": int(n),
        "median_excess": float(median), "iqr_q1": float(q1), "iqr_q3": float(q3),
        "mean_excess": float(mean), "se_mean": None,
        "median_as_reported": float(median), "median_matches_reported": None,
        "availability": "aggregate_only_no_patient_values", "source": source,
    }


def _group_from_patient_values(set_name: str, dataset: str, model: str,
                               measure: str, scoring_block: str,
                               patient_values: list[dict[str, Any]], source: str,
                               median_as_reported: float | None = None) -> dict[str, Any]:
    return {
        "set": set_name, "dataset": dataset, "model": model,
        "measure": measure, "scoring_block": scoring_block,
        "patient_values": patient_values,
        "median_as_reported": median_as_reported,
        "availability": "per_patient_values", "source": source,
    }


def read_stored_results() -> dict[tuple[str, ...], dict[str, Any]]:
    """Read patient excesses where saved, and retain aggregate-only summaries."""
    groups: dict[tuple[str, ...], dict[str, Any]] = {}
    table4 = json.loads(TABLE4_PATH.read_text(encoding="utf-8"))

    # Table 5: only the CTN fitted-model package saved score, shuffled baseline,
    # and excess for each patient. table4.json has patient scores but only
    # cohort-level permutation summaries for the other two datasets.
    package = json.loads(PACKAGE_PATH.read_text(encoding="utf-8"))
    package_rows = {row["metric"]: row for row in package.get("rows", [])}
    for metric, patient_map in package.get("per_patient", {}).items():
        vals = []
        for patient, record in patient_map.items():
            excess = _finite(record.get("excess"))
            if excess is not None:
                vals.append({"patient_id": str(patient),
                             "observed": _finite(record.get("observed")),
                             "shuffled_baseline": _finite(record.get("chance_level")),
                             "excess": excess})
        report = package_rows.get(metric, {}).get("median_excess")
        _add_group(groups, _group_from_patient_values(
            "Table 5 fitted gradient boosting", "CTN-0094", "gradient_boosting",
            metric, "full_shuffle", vals,
            "open_data/results/v10/fitted_model_package.json:per_patient.<measure>.<patient>.excess",
            _finite(report),
        ))
    # REPLACE-BG and College Experience Study: the v18 task-1 run (Table 5 source, rerun 15 Oct with the
    # fixed p value) stores score, shuffled baseline and excess for each patient.
    task1 = json.loads(TASK1_PATIENT_PATH.read_text(encoding="utf-8"))["per_patient"]
    reported = {(r["dataset"], r["measure"]): _finite(r["median_excess"])
                for r in csv.DictReader(TASK1_CSV_PATH.open(encoding="utf-8"))}
    for dataset in ("REPLACE-BG", "College Experience Study"):
        for metric in METRICS:
            vals = []
            for patient, record in task1[dataset].get(metric, {}).items():
                excess = _finite(record.get("excess"))
                if excess is not None:
                    vals.append({"patient_id": str(patient), "observed": _finite(record.get("score")),
                                 "shuffled_baseline": _finite(record.get("shuffled_baseline")),
                                 "excess": excess})
            _add_group(groups, _group_from_patient_values(
                "Table 5 fitted gradient boosting", dataset, "gradient_boosting",
                metric, "full_shuffle", vals,
                "work/2026-10-13_v18_analyses/task1_per_patient.json:per_patient.<dataset>.<measure>.<patient>.excess",
                reported.get((dataset, metric)),
            ))

    # Supplement S10 / Table S16: block_results.json retains mean, median, and
    # quartiles but discards the patient vector, so SE must await rerunning it.
    block = json.loads(BLOCK_PATH.read_text(encoding="utf-8"))
    for cohort_key, cohort in block["cohorts"].items():
        dataset = BLOCK_DATASETS[cohort_key]
        for model, model_result in cohort["models"].items():
            for shuffle, summary in model_result["shuffles"].items():
                iqr = summary["excess_iqr"]
                _add_group(groups, _summary_group(
                    "Supplement S10 Table S16", dataset, model, "auc", shuffle,
                    summary["n_patients"], summary["median_excess"], iqr[0], iqr[1],
                    summary["mean_excess"],
                    "open_data/results/v16/block_results.json:cohorts.<cohort>.models.<model>.shuffles.<block>",
                ))

    # Section 5.2: no patientwise values/baselines are stored. Include explicit
    # rows so the CSV marks the work still needed rather than silently omitting it.
    main = json.loads(MAIN_PATH.read_text(encoding="utf-8"))
    for dataset in main["datasets"]:
        for model in ("run-in rate", "shuffled rate so far"):
            for scoring in ("naive", "tailored"):
                for metric in METRICS:
                    _add_group(groups, _missing_group(
                        "Section 5.2 no-skill forecasts", dataset, model, metric,
                        scoring,
                        "open_data/results/v10/main_table.json stores draw summaries, not patientwise baselines",
                    ))

    # Preserve raw patient vectors from an earlier explicit rerun, if present.
    if PATIENT_PATH.exists():
        saved = json.loads(PATIENT_PATH.read_text(encoding="utf-8"))
        for group in saved.get("groups", []):
            _add_group(groups, group)
    return groups


def _stats(values: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    finite = [number for item in values if (number := _finite(item.get("excess"))) is not None]
    if not finite:
        return {"n_patients": 0, "median_excess": None, "iqr_q1": None,
                "iqr_q3": None, "mean_excess": None, "se_mean": None}
    a = np.asarray(finite, dtype=float)
    q1, median, q3 = np.percentile(a, [25, 50, 75])
    sd = float(np.std(a, ddof=1)) if len(a) > 1 else float("nan")
    return {
        "n_patients": int(len(a)), "median_excess": float(median),
        "iqr_q1": float(q1), "iqr_q3": float(q3),
        "mean_excess": float(np.mean(a)),
        "se_mean": float(sd / math.sqrt(len(a))) if math.isfinite(sd) else None,
    }


def write_csv(groups: dict[tuple[str, ...], dict[str, Any]]) -> list[dict[str, Any]]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    output = []
    for key in sorted(groups):
        group = groups[key]
        if group.get("patient_values") is not None:
            values = group["patient_values"]
            row = _stats(values)
            reported = _finite(group.get("median_as_reported"))
            median = row["median_excess"]
            matches = (abs(median - reported) <= 1e-10) if median is not None and reported is not None else None
            availability = "per_patient_values"
        else:
            row = {name: group.get(name) for name in (
                "n_patients", "median_excess", "iqr_q1", "iqr_q3", "mean_excess", "se_mean"
            )}
            reported = _finite(group.get("median_as_reported"))
            matches = group.get("median_matches_reported")
            availability = group.get("availability", "missing_per_patient_values")
        output.append({
            "set": group["set"], "dataset": group["dataset"], "model": group["model"],
            "measure": group["measure"], "scoring_block": group["scoring_block"],
            **row, "median_as_reported": reported, "median_matches_reported": matches,
            "availability": availability, "source": group.get("source"),
        })
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(output)
    return output


def _import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _patient_values(observed: dict[Any, Any], baselines: dict[Any, list[Any]]) -> list[dict[str, Any]]:
    import numpy as np

    rows = []
    for patient, observed_score in observed.items():
        obs = _finite(observed_score)
        base_values = [_finite(x) for x in baselines.get(patient, [])]
        base_values = [x for x in base_values if x is not None]
        if obs is None or not base_values:
            continue
        baseline = float(np.mean(base_values))
        rows.append({"patient_id": str(patient), "observed": obs,
                     "shuffled_baseline": baseline, "excess": obs - baseline})
    return rows


def rerun_table5_missing(groups: dict[tuple[str, ...], dict[str, Any]]) -> None:
    """Rerun fixed-model within-patient full shuffles for Table 5 missing cohorts."""
    import numpy as np
    import pandas as pd

    if str(OPEN_DATA) not in sys.path:
        sys.path.insert(0, str(OPEN_DATA))
    table4 = importlib.import_module("table4_v10")
    main_table = importlib.import_module("main_table_v10")
    table4._load_modeling_dependencies()
    pm = table4.load_permutation_module()
    n_perm = int(json.loads(TABLE4_PATH.read_text(encoding="utf-8"))["metadata"]["permutation_draws"])
    frames = {cfg["key"]: pd.read_csv(OPEN_DATA / cfg["forecast_file"])
              for cfg in table4.COHORTS}
    contexts = {cfg["key"]: main_table._load_context(pm, table4, cfg, np, pd)
                for cfg in table4.COHORTS}
    # Fit only the two cohorts whose patientwise shuffled baselines were not saved.
    prepared = {"replace_bg": pm.fit_replace_bg(), "ces": pm.fit_ces()}
    rng = np.random.default_rng(pm.SEED)
    for cfg in table4.COHORTS:
        key = cfg["key"]
        context, frame = contexts[key], frames[key]
        if key in prepared:
            fitted = prepared[key]
            bundle = {
                "test": fitted["test"], "pid": fitted["pid"], "y": fitted["y"],
                "predictions": fitted["predictions"],
                "score_permutation": fitted["score_permutation"],
            }
        else:
            bundle = context["bundle"]
        targets = pm.generate_permuted_targets(bundle["y"], bundle["pid"], n_perm, rng)
        if key == "ctn_weekly_missing_excluded":
            # The CTN package already supplies the reported patientwise excesses;
            # these draws are still consumed to preserve the Table 4 RNG stream.
            continue
        bundle_order, file_order = table4.align_forecast_rows(frame, bundle, cfg)
        patient = np.asarray(bundle["pid"])[bundle_order]
        y = frame.iloc[file_order].y.to_numpy(int)
        forecast = frame.iloc[file_order].gradient_boosting.to_numpy(float)
        threshold = float(frame.iloc[file_order].threshold.iloc[0])
        auxiliary = table4._make_auxiliary_data(pm, cfg, bundle)
        observed_result = table4.nine_metrics(
            patient, y, forecast, threshold,
            yardstick=frame.iloc[file_order][cfg["updated_rate_column"]].to_numpy(float),
        )
        baseline_maps: dict[str, dict[Any, list[Any]]] = {
            metric: {} for metric in METRICS
        }
        for target in targets:
            rescored = bundle["score_permutation"](target)
            perm_y = np.asarray(target, dtype=int)[bundle_order]
            perm_forecast = np.asarray(rescored["gradient_boosting"], dtype=float)[bundle_order]
            perm_rate = table4.rebuild_updated_rate(pm, cfg, bundle, target, auxiliary)[bundle_order]
            result = table4.nine_metrics(
                patient, perm_y, perm_forecast, threshold, yardstick=perm_rate
            )
            for metric in METRICS:
                for pid, value in result[metric]["per_patient"].items():
                    baseline_maps[metric].setdefault(pid, []).append(value)
        dataset = TABLE5_DATASETS[key]
        for metric in METRICS:
            values = _patient_values(
                observed_result[metric]["per_patient"], baseline_maps[metric]
            )
            _add_group(groups, _group_from_patient_values(
                "Table 5 fitted gradient boosting", dataset, "gradient_boosting",
                metric, "full_shuffle", values,
                "rerun via table4_v10.load_permutation_module; seed and draws from Table 4",
                None,
            ))


def rerun_block_missing(groups: dict[tuple[str, ...], dict[str, Any]]) -> None:
    """Rerun S16 full and block AUC shuffles, saving each patient's excess."""
    import numpy as np
    import pandas as pd

    block = _import_file("_a1_block_shuffle", OPEN_DATA / "block_shuffle.py")
    analysis = block._load_analysis_module()
    existing = json.loads(BLOCK_PATH.read_text(encoding="utf-8"))
    n_perm = int(existing["metadata"]["permutations"])
    prepared = {
        "replace_bg": analysis.fit_replace_bg(),
        "ctn0094": analysis.fit_ctn(),
        "ces": analysis.fit_ces(),
    }
    rng_full = np.random.default_rng(analysis.SEED)
    rng_blocks = np.random.default_rng(analysis.SEED)
    for cohort_key in ("replace_bg", "ctn0094", "ces"):
        bundle = prepared[cohort_key]
        spec = block.COHORTS[cohort_key]
        time_values = bundle["test"][spec["time_column"]].to_numpy()
        shuffles = [("full", None)] + [
            (f"block_{length}", length) for length in spec["block_lengths"]
        ]
        for shuffle, block_len in shuffles:
            if block_len is None:
                targets = analysis.generate_permuted_targets(
                    bundle["y"], bundle["pid"], n_perm, rng_full
                )
            else:
                targets = block.generate_block_permuted_targets(
                    bundle["y"], bundle["pid"], time_values, block_len,
                    n_perm, rng_blocks,
                )
            rescored = analysis.feature_auc_matrix(
                bundle["y"], targets, bundle["pid"], bundle["predictions"],
                bundle["score_permutation"],
            )
            for model, model_data in rescored.items():
                baseline = np.mean(model_data["permutation_auc"], axis=0)
                excess = model_data["observed_auc"] - baseline
                values = [
                    {"patient_id": str(pid), "observed": float(obs),
                     "shuffled_baseline": float(base), "excess": float(delta)}
                    for pid, obs, base, delta in zip(
                        model_data["observed_ids"], model_data["observed_auc"], baseline, excess
                    )
                ]
                old = existing["cohorts"][cohort_key]["models"][model]["shuffles"][shuffle]
                _add_group(groups, _group_from_patient_values(
                    "Supplement S10 Table S16", BLOCK_DATASETS[cohort_key], model,
                    "auc", shuffle, values,
                    "rerun via block_shuffle.py / cases_model_fixed_permutation.feature_auc_matrix; original seeds",
                    _finite(old.get("median_excess")),
                ))


def _metric_patient_values(scoring: str, patient: Any, y: Any, forecast: Any,
                           threshold: float, yardstick: Any, nine_metrics: Any,
                           naive_metrics: Any) -> dict[str, dict[Any, Any]]:
    if scoring == "naive":
        result = naive_metrics(
            patient, y, forecast, threshold, population_rate=threshold
        )
    else:
        result = nine_metrics(patient, y, forecast, threshold, yardstick=yardstick)
    return {metric: result[metric]["per_patient"] for metric in METRICS}


def rerun_section52_missing(groups: dict[tuple[str, ...], dict[str, Any]]) -> None:
    """Rerun Section 5.2's patientwise baseline draws with main-table functions."""
    import numpy as np
    import pandas as pd

    main_table = importlib.import_module("main_table_v10")
    table4, np, pd, nine_metrics, naive_metrics = main_table._load_dependencies()
    pm = table4.load_permutation_module()
    existing = json.loads(MAIN_PATH.read_text(encoding="utf-8"))
    n_draws = int(existing["metadata"]["permutation_draws"])
    rng = np.random.default_rng(pm.SEED)
    for cfg in table4.COHORTS:
        frame = pd.read_csv(OPEN_DATA / cfg["forecast_file"])
        context = main_table._load_context(pm, table4, cfg, np, pd)
        bundle = context["bundle"]
        bundle_order, file_order = main_table._align_context(context, frame, np, pd)
        patient = np.asarray(bundle["pid"])[bundle_order]
        y = frame.iloc[file_order].y.to_numpy(int)
        runin = frame.iloc[file_order].runin_rate.to_numpy(float)
        updated = frame.iloc[file_order][cfg["updated_rate_column"]].to_numpy(float)
        threshold = float(frame.iloc[file_order].threshold.iloc[0])
        auxiliary = table4._make_auxiliary_data(pm, cfg, bundle)
        models = {
            "run-in rate": runin,
            "shuffled rate so far": updated,
        }
        observed: dict[tuple[str, str], dict[str, dict[Any, Any]]] = {}
        for model, forecast in models.items():
            for scoring in ("naive", "tailored"):
                observed[(model, scoring)] = _metric_patient_values(
                    scoring, patient, y, forecast, threshold, updated,
                    nine_metrics, naive_metrics,
                )
        baseline_maps: dict[tuple[str, str, str], dict[Any, list[Any]]] = {}
        for target in pm.generate_permuted_targets(
            bundle["y"], bundle["pid"], n_draws, rng
        ):
            perm_y = np.asarray(target, dtype=int)[bundle_order]
            perm_updated = table4.rebuild_updated_rate(
                pm, cfg, bundle, target, auxiliary
            )[bundle_order]
            draw_models = {
                "run-in rate": runin,
                "shuffled rate so far": perm_updated,
            }
            for model, forecast in draw_models.items():
                for scoring in ("naive", "tailored"):
                    draw_values = _metric_patient_values(
                        scoring, patient, perm_y, forecast, threshold, perm_updated,
                        nine_metrics, naive_metrics,
                    )
                    for metric in METRICS:
                        key = (model, scoring, metric)
                        bucket = baseline_maps.setdefault(key, {})
                        for pid, value in draw_values[metric].items():
                            bucket.setdefault(pid, []).append(value)
        for (model, scoring), by_metric in observed.items():
            for metric in METRICS:
                values = _patient_values(
                    by_metric[metric], baseline_maps.get((model, scoring, metric), {})
                )
                _add_group(groups, _group_from_patient_values(
                    "Section 5.2 no-skill forecasts", cfg["label"], model,
                    metric, scoring, values,
                    "rerun via main_table_v10 context and Table 4 history builder; original seed",
                    None,
                ))


def _write_patient_json(groups: dict[tuple[str, ...], dict[str, Any]]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    patient_groups = [group for group in groups.values()
                      if group.get("patient_values") is not None]
    PATIENT_PATH.write_text(json.dumps({
        "description": "Per-patient excess = observed score minus that patient's mean shuffled-baseline score.",
        "groups": patient_groups,
    }, indent=2, allow_nan=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rerun-missing", action="store_true",
        help="run the missing fixed-model, block-shuffle, and Section 5.2 pipelines",
    )
    args = parser.parse_args()
    groups = read_stored_results()
    if args.rerun_missing:
        rerun_table5_missing(groups)
        rerun_block_missing(groups)
        rerun_section52_missing(groups)
        _write_patient_json(groups)
    rows = write_csv(groups)
    available = [r for r in rows if r["median_excess"] is not None and r["mean_excess"] is not None]
    available.sort(key=lambda r: abs(r["mean_excess"] - r["median_excess"]), reverse=True)
    print(f"Wrote {CSV_PATH}")
    if args.rerun_missing:
        print(f"Wrote {PATIENT_PATH}")
    print(f"Patientwise results available: {sum(r['availability'] == 'per_patient_values' for r in rows)} groups")
    print("Largest absolute median/mean gaps among available groups:")
    for row in available[:10]:
        gap = row["mean_excess"] - row["median_excess"]
        print(f"{row['set']} | {row['dataset']} | {row['model']} | {row['measure']} | "
              f"{row['scoring_block']}: median={row['median_excess']:.6g}, "
              f"mean={row['mean_excess']:.6g}, mean-median={gap:+.6g}")


if __name__ == "__main__":
    main()
