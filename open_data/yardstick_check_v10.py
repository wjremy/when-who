#!/usr/bin/env python3
"""Compare the v10 run-in and shuffled running-rate forecasts with yardsticks.

Run from the repository root with the project Python environment:
    python open_data/yardstick_check_v10.py [--quick]

The shuffled forecast follows main_table_v10.py exactly: outcomes are shuffled
within patient, the expanding rate is rebuilt, and each draw is scored against
that draw's outcomes. Yardsticks a, b, and c1 are fixed; c2 is Table 4's
observed row-wise updated rate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import main_table_v10 as main_table


OPEN_DATA = Path(__file__).resolve().parent
OUTPUT_DIR = OPEN_DATA / "results" / "v10"
YARDSTICKS = (
    ("training_rate", "(a) Training-cohort population rate"),
    ("scored_rate", "(b) Patient rate over scored period (hindsight)"),
    ("runin_rate", "(c1) Patient run-in rate"),
    ("updated_rate", "(c2) Past-days-only updated rate (row-wise)"),
)


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


def _patient_groups(patient: Any, np, pd):
    values = np.asarray(patient)
    return [(key, np.flatnonzero(values == key)) for key in pd.unique(values)]


def _nb_all(rate: Any, odds: float, np):
    rate = np.asarray(rate, dtype=float)
    return rate - (1.0 - rate) * odds


def _score_draw(patient, y, forecast, threshold, yardsticks, np, pd):
    """Return patient-level metric values and their eligibility counts."""
    odds = threshold / (1.0 - threshold)
    groups = _patient_groups(patient, np, pd)
    output = {}
    for yardstick_key, _ in YARDSTICKS:
        patient_metrics = {
            "brier_skill_mean_patient": [],
            "brier_skill_ratio_mean_losses": [],
            "ppv_minus_yardstick": [],
            "net_benefit_minus_treat_all": [],
        }
        if yardstick_key == "scored_rate":
            patient_metrics["net_benefit_minus_best_static"] = []

        for _, ix in groups:
            truth = np.asarray(y, dtype=float)[ix]
            score = np.asarray(forecast, dtype=float)[ix]
            yard = np.asarray(yardsticks[yardstick_key], dtype=float)[ix]
            alarm = score >= threshold

            forecast_loss = float(np.mean((truth - score) ** 2))
            yardstick_loss = float(np.mean((truth - yard) ** 2))
            patient_metrics["brier_skill_ratio_mean_losses"].append(
                (forecast_loss, yardstick_loss)
            )
            if yardstick_loss > 0:
                patient_metrics["brier_skill_mean_patient"].append(
                    1.0 - forecast_loss / yardstick_loss
                )

            if alarm.any():
                ppv = float(np.mean(truth[alarm]))
                reference_rate = (float(np.mean(yard[alarm]))
                                  if yardstick_key == "updated_rate"
                                  else float(yard[0]))
                patient_metrics["ppv_minus_yardstick"].append(ppv - reference_rate)

            nb = float(np.mean(alarm & (truth == 1))
                       - odds * np.mean(alarm & (truth == 0)))
            if yardstick_key == "updated_rate":
                treat_all = float(np.mean(_nb_all(yard, odds, np)))
            else:
                treat_all = float(_nb_all(float(yard[0]), odds, np))
            patient_metrics["net_benefit_minus_treat_all"].append(nb - treat_all)
            if yardstick_key == "scored_rate":
                patient_metrics["net_benefit_minus_best_static"].append(
                    nb - max(treat_all, 0.0)
                )

        values = {}
        for metric_key, items in patient_metrics.items():
            if metric_key == "brier_skill_ratio_mean_losses":
                mean_forecast_loss = float(np.mean([pair[0] for pair in items]))
                mean_yardstick_loss = float(np.mean([pair[1] for pair in items]))
                ratio_skill = (1.0 - mean_forecast_loss / mean_yardstick_loss
                               if mean_yardstick_loss > 0 else None)
                values[metric_key] = {"values": [ratio_skill] if ratio_skill is not None else [],
                                      "eligible_n": len(items), "ratio": ratio_skill}
            else:
                values[metric_key] = {
                    "values": [float(x) for x in items if math.isfinite(float(x))],
                    "eligible_n": len(items),
                }
        output[yardstick_key] = values
    return output


def _finite_summary(values, np):
    finite = np.asarray([float(x) for x in values if x is not None and
                         math.isfinite(float(x))], dtype=float)
    if not len(finite):
        return {"mean": None, "median": None, "n": 0}
    return {"mean": float(np.mean(finite)), "median": float(np.median(finite)),
            "n": int(len(finite))}


def _row_summary(run_values, draw_summaries, np, is_ratio=False):
    run = _finite_summary(run_values, np)
    means = [item["mean"] for item in draw_summaries if item["mean"] is not None]
    if means:
        mean_array = np.asarray(means, dtype=float)
        shuffled_mean = {
            "mean": float(np.mean(mean_array)),
            "percentiles_2_5_97_5": [float(x) for x in np.percentile(mean_array, [2.5, 97.5])],
            "valid_draws": len(means),
        }
    else:
        shuffled_mean = {"mean": None, "percentiles_2_5_97_5": [None, None],
                         "valid_draws": 0}
    if is_ratio:
        shuffled_median = {"mean": None, "percentiles_2_5_97_5": [None, None],
                           "valid_draws": 0}
        run_median = None
    else:
        medians = [item["median"] for item in draw_summaries if item["median"] is not None]
        if medians:
            median_array = np.asarray(medians, dtype=float)
            shuffled_median = {
                "mean": float(np.mean(median_array)),
                "percentiles_2_5_97_5": [float(x) for x in np.percentile(median_array, [2.5, 97.5])],
                "valid_draws": len(medians),
            }
        else:
            shuffled_median = {"mean": None, "percentiles_2_5_97_5": [None, None],
                               "valid_draws": 0}
        run_median = run["median"]
    counts = [int(item["eligible_n"]) for item in draw_summaries]
    eligible = {
        "runin": int(run["n"]),
        "shuffled_min": min(counts) if counts else 0,
        "shuffled_max": max(counts) if counts else 0,
    }
    return {
        "runin_mean": run["mean"],
        "runin_median": run_median,
        "shuffled_mean": shuffled_mean,
        "shuffled_median": shuffled_median,
        "eligible_n": eligible,
        "shuffled_draw_summaries": draw_summaries,
    }


def _metric_label(key: str) -> str:
    return {
        "brier_skill_mean_patient": "Brier skill: mean individual skills",
        "brier_skill_ratio_mean_losses": "Brier skill: ratio of mean losses",
        "ppv_minus_yardstick": "PPV minus yardstick rate",
        "net_benefit_minus_treat_all": "Net benefit minus treat-all NB",
        "net_benefit_minus_best_static": "Net benefit minus max(treat-all, treat-none)",
    }[key]


def _metric_order(yardstick_key):
    result = ["brier_skill_mean_patient", "brier_skill_ratio_mean_losses",
              "ppv_minus_yardstick", "net_benefit_minus_treat_all"]
    if yardstick_key == "scored_rate":
        result.append("net_benefit_minus_best_static")
    return result


def _display(value, digits=4):
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def _draw_display(summary):
    lo, hi = summary["percentiles_2_5_97_5"]
    if summary["mean"] is None:
        return "n/a"
    return f"{_display(summary['mean'])} [{_display(lo)}, {_display(hi)}]"


def _eligible_display(summary):
    if summary["shuffled_min"] == summary["shuffled_max"]:
        shuffled = str(summary["shuffled_min"])
    else:
        shuffled = f"{summary['shuffled_min']}-{summary['shuffled_max']}"
    return f"{summary['runin']} / {shuffled}"


def _theory_checks(run_scores, dataset_label):
    """Compare the run-in results against the three explicit predictions."""
    b = run_scores["scored_rate"]
    ppv_values = b["ppv_minus_yardstick"]["values"]
    nb_treat_all = b["net_benefit_minus_treat_all"]["values"]
    nb_best = b["net_benefit_minus_best_static"]["values"]
    brier = b["brier_skill_mean_patient"]["values"]
    ppv_max_abs = max((abs(x) for x in ppv_values), default=None)
    ppv_mean = float(sum(ppv_values) / len(ppv_values)) if ppv_values else None
    nb_mean = float(sum(nb_treat_all) / len(nb_treat_all)) if nb_treat_all else None
    best_mean = float(sum(nb_best) / len(nb_best)) if nb_best else None
    brier_mean = float(sum(brier) / len(brier)) if brier else None
    zero_tol = 1e-12
    checks = [
        {
            "prediction": "Run-in PPV minus the patient's scored-period rate is exactly 0 for every eligible patient.",
            "estimate_mean": ppv_mean,
            "max_absolute_patient_value": ppv_max_abs,
            "status": "holds" if ppv_max_abs is not None and ppv_max_abs <= zero_tol else "fails",
        },
        {
            "prediction": "Run-in net benefit minus treat-all NB at the patient's scored-period rate has expected value 0.",
            "estimate_equal_patient_mean": nb_mean,
            "status": "holds at zero" if nb_mean is not None and abs(nb_mean) <= zero_tol else "fails at zero",
        },
        {
            "prediction": "Run-in net benefit minus max(treat-all, treat-none) at the patient's scored-period rate is not 0.",
            "estimate_equal_patient_mean": best_mean,
            "status": "nonzero as predicted" if best_mean is not None and abs(best_mean) > zero_tol else "fails (observed zero)",
        },
        {
            "prediction": "Run-in Brier skill against the patient's scored-period rate is biased low (negative).",
            "estimate_equal_patient_mean": brier_mean,
            "status": "negative as predicted" if brier_mean is not None and brier_mean < -zero_tol
                      else ("zero" if brier_mean is not None and abs(brier_mean) <= zero_tol
                            else "fails (positive)"),
        },
    ]
    return {"dataset": dataset_label, "forecast": "runin", "yardstick": "scored_rate",
            "checks": checks}


def _analyse_dataset(context, frame, table4, pm, np, pd, targets):
    config = context["config"]
    bundle = context["bundle"]
    bundle_order, file_order = main_table._align_context(context, frame, np, pd)
    ordered = frame.iloc[file_order]
    patient = np.asarray(bundle["pid"])[bundle_order]
    y_observed = ordered.y.to_numpy(int)
    runin = ordered.runin_rate.to_numpy(float)
    updated = ordered[config["updated_rate_column"]].to_numpy(float)
    threshold_values = ordered.threshold.to_numpy(float)
    threshold = float(threshold_values[0])
    if not np.allclose(threshold_values, threshold, rtol=0, atol=1e-12):
        raise ValueError(f"{config['label']}: threshold varies across rows")
    if not np.allclose(ordered.training_cohort_rate.to_numpy(float), threshold,
                       rtol=0, atol=1e-12):
        raise AssertionError(f"{config['label']}: threshold is not the training-cohort rate")

    patient_rate = np.zeros(len(y_observed), dtype=float)
    for _, ix in _patient_groups(patient, np, pd):
        patient_rate[ix] = float(np.mean(y_observed[ix]))
    yardsticks = {
        "training_rate": np.full(len(y_observed), threshold, dtype=float),
        "scored_rate": patient_rate,
        "runin_rate": runin.copy(),
        "updated_rate": updated.copy(),
    }
    run_scores = _score_draw(patient, y_observed, runin, threshold, yardsticks, np, pd)
    draw_scores = []
    for target in targets:
        rebuilt = table4.rebuild_updated_rate(pm, config, bundle, target, context["auxiliary"])
        shuffled_forecast = np.asarray(rebuilt, dtype=float)[bundle_order]
        shuffled_y = np.asarray(target, dtype=int)[bundle_order]
        # Verify that a within-patient shuffle preserves yardstick (b).
        draw_rates = np.zeros(len(shuffled_y), dtype=float)
        for _, ix in _patient_groups(patient, np, pd):
            draw_rates[ix] = float(np.mean(shuffled_y[ix]))
        if not np.allclose(draw_rates, patient_rate, rtol=0, atol=1e-12):
            raise AssertionError(f"{config['label']}: permutation changed a patient's scored-period rate")
        draw_yardsticks = dict(yardsticks)
        draw_yardsticks["scored_rate"] = draw_rates
        draw_scores.append(_score_draw(patient, shuffled_y, shuffled_forecast,
                                       threshold, draw_yardsticks, np, pd))

    metrics = {}
    for yardstick_key, yardstick_label in YARDSTICKS:
        metrics[yardstick_key] = {"label": yardstick_label, "metrics": {}}
        for metric_key in _metric_order(yardstick_key):
            run_values = run_scores[yardstick_key][metric_key]["values"]
            run_n = run_scores[yardstick_key][metric_key]["eligible_n"]
            draw_summaries = []
            for draw in draw_scores:
                detail = draw[yardstick_key][metric_key]
                if metric_key == "brier_skill_ratio_mean_losses":
                    values = [detail["ratio"]] if detail["ratio"] is not None else []
                else:
                    values = detail["values"]
                draw_summary = _finite_summary(values, np)
                draw_summaries.append({
                    "mean": draw_summary["mean"],
                    "median": (None if metric_key == "brier_skill_ratio_mean_losses"
                               else draw_summary["median"]),
                    "eligible_n": int(detail["eligible_n"]),
                })
            summary = _row_summary(run_values, draw_summaries, np,
                                   metric_key == "brier_skill_ratio_mean_losses")
            # The ratio row is one aggregate value, but its losses use every
            # patient; report that patient count rather than the scalar count.
            summary["eligible_n"]["runin"] = int(run_n)
            summary["label"] = _metric_label(metric_key)
            summary["runin_eligible_n"] = int(run_n)
            metrics[yardstick_key]["metrics"][metric_key] = summary

    return {
        "label": config["label"],
        "threshold": threshold,
        "training_cohort_rate": float(ordered.training_cohort_rate.iloc[0]),
        "test_patients": int(pd.unique(patient).size),
        "scored_rows": int(len(y_observed)),
        "permutation_draws": len(targets),
        "metrics": metrics,
        "theory_checks": _theory_checks(run_scores, config["label"]),
    }


def build_markdown(result):
    lines = [
        "# V10 yardstick check", "",
        "Both forecasts are scored per patient. Run-in uses the observed scored-period outcomes. "
        "For each shuffled draw, the outcomes are permuted within patient, the running rate is "
        "rebuilt, and that draw is scored against its permuted outcomes, matching "
        "`main_table_v10.py`. Yardsticks (a), (b), and (c1) use the training rate, the "
        "patient's unchanged scored-period rate, and the fixed run-in rate; (c2) is the "
        "observed Table 4 past-days-only rate, row by row.",
        "",
        f"Draws: {result['metadata']['permutation_draws']} (seed {result['metadata']['permutation_seed']}; "
        f"Table 4 cohort order). Values are equal-patient summaries, rounded to four decimals. "
        "Shuffled intervals are the 2.5th-97.5th percentiles across draw-level summaries.",
        "",
        "Brier skill has two rows: mean of patient skills (only patients with positive yardstick "
        "Brier loss) and the ratio of equal-patient mean losses (all patients; its patient median "
        "is not defined). PPV rows include patients with at least one alarm. For (c2), PPV's "
        "reference rate is averaged over that patient's alarmed rows; treat-all NB is averaged "
        "over all rows. All alarms use forecast >= the training-cohort threshold.",
        "",
        "## Theory predictions for run-in against (b)", "",
        "Statuses compare the observed equal-patient summaries with the stated zero/negative "
        "prediction. The PPV identity is also checked patient by patient. These are dataset "
        "checks, not repeated-sample uncertainty estimates.", "",
        "| Dataset | Prediction | Estimate | Status |", "|---|---|---:|---|",
    ]
    for label in ("CTN-0094", "REPLACE-BG", "College Experience"):
        checks = result["datasets"][label]["theory_checks"]["checks"]
        for check in checks:
            value = check.get("estimate_equal_patient_mean", check.get("estimate_mean"))
            if "max_absolute_patient_value" in check:
                value_text = (f"mean {_display(check['estimate_mean'], 6)}; max |patient| "
                              f"{_display(check['max_absolute_patient_value'], 6)}")
            else:
                value_text = _display(value, 6)
            lines.append(f"| {label} | {check['prediction']} | {value_text} | {check['status']} |")
    for label in ("CTN-0094", "REPLACE-BG", "College Experience"):
        dataset = result["datasets"][label]
        lines.extend(["", f"## {label}", "",
                      f"{dataset['test_patients']} patients; {dataset['scored_rows']} rows; "
                      f"training rate / threshold {_display(dataset['threshold'], 6)}.", "",
                      "| Metric x yardstick | Run-in mean | Run-in median | Shuffled mean [2.5-97.5] | "
                      "Shuffled median [2.5-97.5] | Eligible n (run-in / shuffled) |",
                      "|---|---:|---:|---:|---:|---:|"])
        for yardstick_key, _ in YARDSTICKS:
            yardstick = dataset["metrics"][yardstick_key]
            for metric_key in _metric_order(yardstick_key):
                cell = yardstick["metrics"][metric_key]
                run_median = "n/a" if cell["runin_median"] is None else _display(cell["runin_median"])
                if cell["shuffled_median"]["mean"] is None:
                    shuffled_median = "n/a"
                else:
                    shuffled_median = _draw_display(cell["shuffled_median"])
                lines.append(
                    f"| {cell['label']} x {yardstick['label']} | "
                    f"{_display(cell['runin_mean'])} | {run_median} | "
                    f"{_draw_display(cell['shuffled_mean'])} | {shuffled_median} | "
                    f"{_eligible_display(cell['eligible_n'])} |"
                )
    lines.append("")
    return "\n".join(lines)


def run(n_draws: int, quick: bool):
    table4, np, pd, _, _ = main_table._load_dependencies()
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
        contexts[config["key"]] = main_table._load_context(pm, table4, config, np, pd)

    # Keep the same single RNG stream and Table 4 cohort order as main_table_v10.py.
    rng_perm = np.random.default_rng(pm.SEED)
    targets_by_key = {}
    for config in table4.COHORTS:
        bundle = contexts[config["key"]]["bundle"]
        targets_by_key[config["key"]] = pm.generate_permuted_targets(
            bundle["y"], bundle["pid"], n_draws, rng_perm
        )

    by_label = {}
    for config in table4.COHORTS:
        key = config["key"]
        by_label[config["label"]] = _analyse_dataset(
            contexts[key], frames[key], table4, pm, np, pd, targets_by_key[key]
        )
    requested_order = ["CTN-0094", "REPLACE-BG", "College Experience"]
    result = {
        "title": "V10 forecast yardstick checks",
        "quick_mode": bool(quick),
        "metadata": {
            "permutation_draws": int(n_draws),
            "permutation_seed": int(pm.SEED),
            "permutation_rng_stream": "cases_model_fixed_permutation.generate_permuted_targets, Table 4 cohort order",
            "shuffled_scoring": "Each rebuilt shuffled running rate is scored against the corresponding within-patient permuted outcomes, as in main_table_v10.py.",
            "yardstick_c2": "Observed Table 4 updated rate aligned row-wise; it is not rebuilt on the shuffled records.",
            "alarm_rule": "forecast >= training-cohort rate",
            "patient_summary": "Equal patient weighting. Shuffled summaries aggregate the per-draw equal-patient means and medians; intervals are percentiles across draws.",
            "brier_skill": "Individual skill is 1 - patient forecast Brier / patient yardstick Brier, eligible only for positive yardstick loss. Ratio skill is 1 - equal-patient mean forecast loss / equal-patient mean yardstick loss.",
        },
        "datasets": {label: by_label[label] for label in requested_order},
    }
    cleaned = main_table._clean(result, np)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "yardstick_check.json"
    markdown_path = OUTPUT_DIR / "yardstick_check.md"
    json_path.write_text(json.dumps(cleaned, indent=2, allow_nan=False), encoding="utf-8")
    markdown_path.write_text(build_markdown(cleaned), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    print(f"Draws: {n_draws}")
    return cleaned


def main():
    args = parse_args()
    run(args.draws, args.quick)


if __name__ == "__main__":
    main()
