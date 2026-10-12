#!/usr/bin/env python3
"""Build B2: patient-level baselines for the two rate-only forecasts.

Run from the repository root:
    python open_data/baseline_table_v14.py [--quick] [--draws N]

The v10 panel construction, row alignment, within-patient shuffles, and
expanding-rate history rebuild are imported rather than reimplemented.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OPEN_DATA = ROOT / "open_data"
OUTPUT_DIR = ROOT / "results" / "v14"
V10_JSON = OPEN_DATA / "results" / "v10" / "main_table.json"

MEASURES = (
    "auc",
    "average_precision",
    "average_precision_any_event",
    "sensitivity_specificity_minus_one",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "net_benefit",
    "net_benefit_forecast_level",
    "brier_skill",
    "calibration_slope",
)
MEASURE_LABELS = {
    "auc": "AUC",
    "average_precision": "Average precision (patients with both outcomes)",
    "average_precision_any_event": "Average precision (patients with an event)",
    "sensitivity_specificity_minus_one": "Sensitivity + specificity - 1",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "ppv": "PPV",
    "npv": "NPV",
    "net_benefit": "Net benefit (baseline: all or none by own rate)",
    "net_benefit_forecast_level": "Net benefit (baseline: all or none by the forecast's level)",
    "brier_skill": "Brier skill",
    "calibration_slope": "Calibration slope",
}
FORECASTS = ("runin_rate", "shuffled_rate_so_far")
CSV_FIELDS = (
    "dataset", "forecast", "measure", "eligible_n", "score", "score_lo",
    "score_hi", "usual_baseline", "within_baseline", "own_rate",
    "shuffled_baseline", "shuffled_lo", "shuffled_hi", "diff_within",
    "diff_shuffled",
)

_np = None
_average_precision = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=None,
                        help="number of draws (default 200; quick default 10)")
    parser.add_argument("--quick", action="store_true",
                        help="use 10 draws unless --draws is also supplied")
    parser.add_argument("--seed", type=int, default=20261012,
                        help="seed for score draws; independent baseline draws use seed+1")
    args = parser.parse_args()
    args.draws = args.draws if args.draws is not None else (10 if args.quick else 200)
    if args.draws < 1:
        parser.error("--draws must be at least 1")
    return args


def _load_dependencies():
    """Import v10's data and history helpers and the package scoring functions."""
    global _np, _average_precision
    if str(OPEN_DATA) not in sys.path:
        sys.path.insert(0, str(OPEN_DATA))
    import main_table_v10 as main_table

    table4, np, pd, nine_metrics, naive_metrics = main_table._load_dependencies()
    code_src = ROOT / "code" / "wheneval" / "src"
    if str(code_src) not in sys.path:
        sys.path.insert(0, str(code_src))
    from wheneval.nine import average_precision

    _np = np
    _average_precision = average_precision
    return main_table, table4, np, pd, nine_metrics, naive_metrics


def _random_ranking_ap(T: int, n: int) -> float:
    """Expected AP for a uniformly random strict ranking of T rows with n events."""
    if T < 1 or n < 0 or n > T:
        return float("nan")
    if T == 1:
        return float(n)
    return float(sum((1.0 + (r - 1) * (n - 1) / (T - 1)) / r
                     for r in range(1, T + 1)) / T)


def _check_ap_ties() -> None:
    """Check the repository's tie convention and the strict-ranking AP formula."""
    np = _np
    ap = _average_precision
    if np is None or ap is None:
        raise RuntimeError("load scoring dependencies before checking AP")

    constant = ap(np.array([1, 1, 1] + [0] * 7), np.full(10, 0.4))
    if not np.isclose(constant, 0.3, rtol=0, atol=1e-12):
        raise AssertionError(f"constant-score AP should equal n/T=0.3; got {constant}")

    T, n = 6, 2
    brute = []
    y = np.array([1, 1, 0, 0, 0, 0], dtype=int)
    for ranking in itertools.permutations(range(T)):
        # Give each position in this ranking a distinct descending forecast.
        score = np.empty(T, dtype=float)
        for rank, row in enumerate(ranking):
            score[row] = T - rank
        brute.append(float(ap(y, score)))
    exact = _random_ranking_ap(T, n)
    if not np.isclose(exact, float(np.mean(brute)), rtol=0, atol=1e-12):
        raise AssertionError(
            f"R(6,2)={exact} does not match brute-force strict-ranking mean {np.mean(brute)}"
        )


def _scalar(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _groups(patient, np, pd):
    ids = np.asarray(patient)
    for key in pd.unique(ids):
        yield key, np.flatnonzero(ids == key)


def _summarize(values, np) -> tuple[float | None, float | None, float | None]:
    finite = np.asarray([value for value in (_scalar(v) for v in values)
                         if value is not None], dtype=float)
    if not len(finite):
        return None, None, None
    return (float(np.mean(finite)), float(np.percentile(finite, 2.5)),
            float(np.percentile(finite, 97.5)))


def _metric_result(patient, y, forecast, threshold, population_rate,
                   naive_metrics, np, pd) -> dict[str, dict[str, Any]]:
    """Return equal-patient values, eligibility, and own rates for one forecast."""
    result = naive_metrics(patient, y, forecast, threshold,
                          population_rate=population_rate)
    group_data = list(_groups(patient, np, pd))

    ap_values: dict[Any, float] = {}
    for key, ix in group_data:
        if 0 < int(np.sum(y[ix])) < len(ix):
            ap_values[key] = float(_average_precision(y[ix], forecast[ix]))

    common_keys = set(result["sensitivity"]["per_patient"]).intersection(
        result["specificity"]["per_patient"])
    youden_values = {
        key: float(result["sensitivity"]["per_patient"][key]
                   + result["specificity"]["per_patient"][key] - 1.0)
        for key in common_keys
    }

    def entry(values, mean=None, n=None):
        finite = {key: float(value) for key, value in values.items()
                  if _scalar(value) is not None}
        mean_value = (_scalar(mean) if mean is not None else
                      (float(np.mean(list(finite.values()))) if finite else None))
        return {"mean": mean_value,
                "n": int(n if n is not None else len(finite)),
                "keys": set(finite)}

    output = {
        "auc": entry(result["auc"]["per_patient"]),
        "average_precision": entry(ap_values),
        # Eligibility of v10 and of the printed supplement tables: at least one event.
        "average_precision_any_event": entry(result["average_precision"]["per_patient"]),
        "sensitivity_specificity_minus_one": entry(youden_values),
        "sensitivity": entry(result["sensitivity"]["per_patient"]),
        "specificity": entry(result["specificity"]["per_patient"]),
        "ppv": entry(result["ppv"]["per_patient"]),
        "npv": entry(result["npv"]["per_patient"]),
        "net_benefit": entry(result["net_benefit"]["per_patient"]),
        "net_benefit_forecast_level": entry(result["net_benefit"]["per_patient"]),
        # The package defines this summary as a ratio of equal-patient mean losses.
        "brier_skill": entry(result["brier_skill"]["per_patient"],
                             result["brier_skill"]["summary"]["mean"],
                             result["brier_skill"]["summary"]["n"]),
        "calibration_slope": entry(
            result["calibration_slope"]["per_patient"],
            result["calibration_slope"]["summary"]["mean"],
            result["calibration_slope"]["summary"]["n"],
        ),
    }
    rates = {key: float(np.mean(y[ix])) for key, ix in group_data}
    for metric, item in output.items():
        item["own_rate"] = (float(np.mean([rates[key] for key in item["keys"]]))
                            if item["keys"] else None)
    return output


def _within_baseline(metric: str, patient, y, eligible_keys: set[Any],
                     forecast, threshold, population_rate, np, pd) -> float | None:
    """Level-only baseline on the exact patient group eligible for the score."""
    if metric == "calibration_slope":
        # The level-only forecast has no within-patient variation; its
        # conditional within-patient slope baseline is zero by definition.
        return 0.0
    grouped = {key: ix for key, ix in _groups(patient, np, pd)}
    eligible = [key for key in grouped if key in eligible_keys]
    if not eligible:
        return None
    if metric == "auc":
        return 0.5
    if metric in ("average_precision", "average_precision_any_event"):
        vals = [_random_ranking_ap(len(grouped[key]), int(np.sum(y[grouped[key]])))
                for key in eligible]
        return float(np.mean(vals))
    if metric == "sensitivity_specificity_minus_one":
        return 0.0
    if metric in ("sensitivity", "specificity"):
        return None
    if metric == "ppv":
        return float(np.mean([np.mean(y[grouped[key]]) for key in eligible]))
    if metric == "npv":
        return float(np.mean([1.0 - np.mean(y[grouped[key]]) for key in eligible]))
    if metric == "net_benefit":
        odds = threshold / (1.0 - threshold)
        values = []
        for key in eligible:
            q = float(np.mean(y[grouped[key]]))
            # nine_metrics uses (yardstick >= threshold) per row; a repeated own
            # rate therefore selects treat-all or treat-none for the patient.
            values.append(q - (1.0 - q) * odds if q >= threshold else 0.0)
        return float(np.mean(values))
    if metric == "net_benefit_forecast_level":
        # Table 4: alarm on all days or on none, as the forecast's level selects.
        # The level is the patient's mean forecast over the scored time points.
        odds = threshold / (1.0 - threshold)
        values = []
        for key in eligible:
            ix = grouped[key]
            q = float(np.mean(y[ix]))
            values.append(q - (1.0 - q) * odds
                          if float(np.mean(forecast[ix])) >= threshold else 0.0)
        return float(np.mean(values))
    if metric == "brier_skill":
        model_losses, reference_losses = [], []
        for key in eligible:
            ix = grouped[key]
            q = float(np.mean(y[ix]))
            model_losses.append(float(np.mean((y[ix] - q) ** 2)))
            reference_losses.append(float(np.mean((y[ix] - population_rate) ** 2)))
        denominator = float(np.mean(reference_losses))
        return (float(1.0 - float(np.mean(model_losses)) / denominator)
                if denominator > 0 else None)
    raise KeyError(metric)


def _usual_baseline(metric: str, threshold: float, population_rate: float,
                    scored_rate: float) -> float | str:
    if metric == "auc":
        return 0.5
    if metric in ("average_precision", "average_precision_any_event"):
        return (f"scored_set={scored_rate:.12g}; "
                f"training_cohort={population_rate:.12g}")
    if metric == "sensitivity_specificity_minus_one":
        return 0.0
    if metric == "sensitivity":
        return threshold
    if metric == "specificity":
        return 1.0 - threshold
    if metric == "ppv":
        return population_rate
    if metric == "npv":
        return 1.0 - population_rate
    if metric in ("net_benefit", "net_benefit_forecast_level"):
        return (population_rate - (1.0 - population_rate) * threshold / (1.0 - threshold)
                if population_rate >= threshold else 0.0)
    if metric == "brier_skill":
        return 0.0
    if metric == "calibration_slope":
        return 1.0
    raise KeyError(metric)


def _count_range(score_counts: list[int], shuffle_counts: list[int]) -> int | str:
    counts = score_counts + shuffle_counts
    if not counts:
        return 0
    low, high = min(counts), max(counts)
    return low if low == high else f"{low} to {high}"


def _format_num(value: Any) -> str:
    number = _scalar(value)
    return "n/a" if number is None else f"{number:.3f}"


def _format_range(low: Any, high: Any) -> str:
    low_num, high_num = _scalar(low), _scalar(high)
    if low_num is None or high_num is None:
        return "n/a"
    if math.isclose(low_num, high_num, rel_tol=0, abs_tol=1e-12):
        return _format_num(low_num)
    return f"{low_num:.3f} to {high_num:.3f}"


def _format_baseline(metric: str, value: Any) -> str:
    if metric.startswith("average_precision") and isinstance(value, str):
        pieces = dict(part.split("=", 1) for part in value.split("; "))
        return (f"{float(pieces['scored_set']):.3f} (scored); "
                f"{float(pieces['training_cohort']):.3f} (training)")
    return _format_num(value)


def _analyze_dataset(config, frame, context, table4, main_table,
                     naive_metrics, np, pd, pm, n_draws: int,
                     rng_score, rng_baseline) -> tuple[dict[str, Any], dict[str, list[float]]]:
    bundle_order, file_order = main_table._align_context(context, frame, np, pd)
    bundle = context["bundle"]
    patient = np.asarray(bundle["pid"])[bundle_order]
    y = frame.iloc[file_order].y.to_numpy(int)
    runin = frame.iloc[file_order].runin_rate.to_numpy(float)
    threshold = float(frame.iloc[file_order].threshold.iloc[0])
    population_rate = float(frame.iloc[file_order].training_cohort_rate.iloc[0])
    if not np.allclose(frame.iloc[file_order].threshold.to_numpy(float), threshold,
                       rtol=0, atol=1e-12):
        raise ValueError(f"{config['label']}: threshold varies across rows")
    if not np.allclose(frame.iloc[file_order].training_cohort_rate.to_numpy(float),
                       population_rate, rtol=0, atol=1e-12):
        raise ValueError(f"{config['label']}: training-cohort rate varies across rows")
    if not np.isclose(threshold, population_rate, rtol=0, atol=1e-12):
        raise ValueError(f"{config['label']}: alarm threshold is not training-cohort rate")
    if not np.array_equal(y, np.asarray(bundle["y"])[bundle_order]):
        raise AssertionError(f"{config['label']}: aligned source outcomes differ from v10 rows")

    observed = _metric_result(patient, y, runin, threshold, population_rate,
                              naive_metrics, np, pd)
    observed_within = {
        metric: _within_baseline(metric, patient, y, observed[metric]["keys"],
                                 runin, threshold, population_rate, np, pd)
        for metric in MEASURES
    }

    score_targets = pm.generate_permuted_targets(bundle["y"], bundle["pid"],
                                                  n_draws, rng_score)
    baseline_targets = pm.generate_permuted_targets(bundle["y"], bundle["pid"],
                                                     n_draws, rng_baseline)
    score_draws = {metric: [] for metric in MEASURES}
    score_counts = {metric: [] for metric in MEASURES}
    score_within = {metric: [] for metric in MEASURES}
    score_own_rates = {metric: [] for metric in MEASURES}
    baseline_draws = {forecast: {metric: [] for metric in MEASURES}
                      for forecast in FORECASTS}
    baseline_counts = {forecast: {metric: [] for metric in MEASURES}
                       for forecast in FORECASTS}

    def collect(target, forecast, outputs, counts, within_outputs=None,
                own_rates=None):
        target_aligned = np.asarray(target, dtype=int)[bundle_order]
        metrics = _metric_result(patient, target_aligned, forecast, threshold,
                                 population_rate, naive_metrics, np, pd)
        for metric in MEASURES:
            item = metrics[metric]
            outputs[metric].append(item["mean"])
            counts[metric].append(item["n"])
            if within_outputs is not None:
                within_outputs[metric].append(_within_baseline(
                    metric, patient, target_aligned, item["keys"], forecast,
                    threshold, population_rate, np, pd,
                ))
            if own_rates is not None:
                own_rates[metric].append(item["own_rate"])

    for target in score_targets:
        updated = table4.rebuild_updated_rate(
            pm, config, bundle, target, context["auxiliary"]
        )
        forecast = np.asarray(updated, dtype=float)[bundle_order]
        collect(target, forecast, score_draws, score_counts,
                score_within, score_own_rates)
    for target in baseline_targets:
        collect(target, runin, baseline_draws["runin_rate"],
                baseline_counts["runin_rate"])
        updated = table4.rebuild_updated_rate(
            pm, config, bundle, target, context["auxiliary"]
        )
        forecast = np.asarray(updated, dtype=float)[bundle_order]
        collect(target, forecast, baseline_draws["shuffled_rate_so_far"],
                baseline_counts["shuffled_rate_so_far"])

    records: dict[str, Any] = {}
    draw_values_for_compare: dict[str, list[float]] = {}
    for forecast_name in FORECASTS:
        for metric in MEASURES:
            if forecast_name == "runin_rate":
                score_series = [observed[metric]["mean"]]
                score_n = [observed[metric]["n"]]
                within_series = [observed_within[metric]]
                own_series = [observed[metric]["own_rate"]]
            else:
                score_series = score_draws[metric]
                score_n = score_counts[metric]
                within_series = score_within[metric]
                own_series = score_own_rates[metric]

            score_mean, score_lo, score_hi = _summarize(score_series, np)
            within_mean, _, _ = _summarize(within_series, np)
            own_rate, _, _ = _summarize(own_series, np)
            baseline_forecast = ("runin_rate" if forecast_name == "runin_rate"
                                 else "shuffled_rate_so_far")
            shuffled_mean, shuffled_lo, shuffled_hi = _summarize(
                baseline_draws[baseline_forecast][metric], np
            )
            # Count ranges cover both the score sample and the independent
            # shuffled-baseline sample so varying eligibility is visible.
            eligible_n = _count_range(
                score_n, baseline_counts[baseline_forecast][metric]
            )
            usual = _usual_baseline(metric, threshold, population_rate,
                                    float(np.mean(y)))
            diff_within = (score_mean - within_mean
                           if score_mean is not None and within_mean is not None else None)
            diff_shuffled = (score_mean - shuffled_mean
                             if score_mean is not None and shuffled_mean is not None else None)
            row = {
                "dataset": ("College Experience Study" if config["label"] == "College Experience"
                            else config["label"]),
                "forecast": forecast_name,
                "measure": metric,
                "eligible_n": eligible_n,
                "score": score_mean,
                "score_lo": score_lo if forecast_name == "shuffled_rate_so_far" else None,
                "score_hi": score_hi if forecast_name == "shuffled_rate_so_far" else None,
                "usual_baseline": usual,
                "within_baseline": within_mean,
                "own_rate": own_rate,
                "shuffled_baseline": shuffled_mean,
                "shuffled_lo": shuffled_lo,
                "shuffled_hi": shuffled_hi,
                "diff_within": diff_within,
                "diff_shuffled": diff_shuffled,
            }
            records[f"{forecast_name}:{metric}"] = row
            if forecast_name == "shuffled_rate_so_far":
                draw_values_for_compare[metric] = [
                    float(value) for value in score_series if _scalar(value) is not None
                ]
    return records, draw_values_for_compare


def _monte_carlo_tolerance(new_values, old_values, new_n: int, old_n: int,
                           np) -> float | None:
    new = np.asarray([v for v in new_values if _scalar(v) is not None], dtype=float)
    old = np.asarray([v for v in old_values if _scalar(v) is not None], dtype=float)
    if len(new) < 2 or len(old) < 2:
        return None
    new_se = float(np.std(new, ddof=1)) / math.sqrt(max(1, new_n))
    old_se = float(np.std(old, ddof=1)) / math.sqrt(max(1, old_n))
    return 3.0 * math.sqrt(new_se * new_se + old_se * old_se)


def _eligible_bounds(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (int, float)):
        return int(value), int(value)
    if isinstance(value, str) and " to " in value:
        left, right = value.split(" to ", 1)
        try:
            return int(left), int(right)
        except ValueError:
            return None
    return None


def _reference_comparison(records, draws_by_dataset, v10, args, np) -> list[str]:
    lines = ["", "Comparison with `open_data/results/v10/main_table.json` (naive patient means):"]
    old_draws = int(v10.get("metadata", {}).get("permutation_draws", 0))
    aliases = {
        "sensitivity_specificity_minus_one": "naive_youden",
        "average_precision_any_event": "average_precision",
        "net_benefit_forecast_level": "net_benefit",
    }
    for config in records["_configs"]:
        old_label = config["label"]
        new_label = ("College Experience Study" if old_label == "College Experience"
                     else old_label)
        old_cohort = v10["datasets"][old_label]
        for forecast, cell_name in (("runin_rate", "runin_naive"),
                                    ("shuffled_rate_so_far", "shuffled_running_naive")):
            lines.append(f"- {new_label}, {forecast}:")
            for metric in MEASURES:
                row = records[new_label][f"{forecast}:{metric}"]
                old_metric = aliases.get(metric, metric)
                if old_metric == "naive_youden":
                    old_cell = old_cohort["naive_youden"][
                        "runin" if forecast == "runin_rate" else "shuffled_running"
                    ]
                else:
                    old_cell = old_cohort["metrics"][old_metric]["cells"][cell_name]
                old_mean = _scalar(old_cell.get("mean"))
                new_mean = _scalar(row["score"])
                difference = (new_mean - old_mean
                              if new_mean is not None and old_mean is not None else None)
                if metric == "sensitivity_specificity_minus_one":
                    status = "definition differs (see notes)"
                elif metric.startswith("average_precision"):
                    old_n = _eligible_bounds(old_cell.get("eligible_n"))
                    new_n = _eligible_bounds(row.get("eligible_n"))
                    if old_n != new_n or (new_n and new_n[0] != new_n[1]):
                        status = "eligibility differs (v10 includes event-only patients)"
                    elif forecast == "runin_rate":
                        status = ("match" if difference is not None
                                  and abs(difference) <= 1e-10 else "different")
                    else:
                        tolerance = _monte_carlo_tolerance(
                            draws_by_dataset[new_label].get(metric, []),
                            old_cell.get("draw_values", []), args.draws,
                            old_draws, np,
                        )
                        status = (f"within 3-SE Monte Carlo tolerance {tolerance:.4g}"
                                  if difference is not None and tolerance is not None
                                  and abs(difference) <= tolerance else
                                  (f"outside 3-SE Monte Carlo tolerance {tolerance:.4g}"
                                   if tolerance is not None else
                                   "Monte Carlo comparison; unavailable SE"))
                elif forecast == "runin_rate":
                    status = ("both n/a (not estimable)" if old_mean is None and new_mean is None
                              else ("match" if difference is not None
                                    and abs(difference) <= 1e-10 else "different"))
                else:
                    old_values = old_cell.get("draw_values", [])
                    tolerance = _monte_carlo_tolerance(
                        draws_by_dataset[new_label].get(metric, []), old_values,
                        args.draws, old_draws, np,
                    )
                    status = (f"within 3-SE Monte Carlo tolerance {tolerance:.4g}"
                              if difference is not None and tolerance is not None
                              and abs(difference) <= tolerance else
                              (f"outside 3-SE Monte Carlo tolerance {tolerance:.4g}"
                               if tolerance is not None else "Monte Carlo comparison; unavailable SE"))
                lines.append(
                    f"  - {MEASURE_LABELS[metric]}: B2 {_format_num(new_mean)}, "
                    f"v10 {_format_num(old_mean)}, difference {_format_num(difference)}; {status}."
                )
    return lines


def _markdown(records: dict[str, dict[str, Any]], metadata: dict[str, Any]) -> str:
    lines = [
        "# B2. Baselines for the rate-only forecasts",
        "",
        "For each dataset and forecast, cells give the equal-patient score; shuffled-rate "
        "scores also give the 2.5th to 97.5th percentiles across draws. `eligible_n` is "
        "the minimum-to-maximum patient count across score and independent shuffled-baseline "
        "draws (a single count when fixed).",
        "",
        "Usual baselines are AUC 0.5; AP at both the scored-set event rate and the "
        "training-cohort rate; sensitivity at the training-cohort rate; specificity at "
        "its complement; PPV/NPV at the training-cohort event/non-event rates; net benefit "
        "of the population-rate treat-all or treat-none policy; Brier skill 0; and slope 1. "
        "Within-patient baselines use the same eligible patients: AUC 0.5, strict-random-ranking "
        "AP, Youden 0, own event/non-event rates for PPV/NPV, for net benefit the policy of "
        "alarming on all days or on none (selected by the patient's own scored rate, which is the "
        "better of the two in hindsight, or by the forecast's level, its mean over the scored time "
        "points), and the own-rate-only forecast's Brier skill against the "
        "training-cohort rate. Sensitivity and specificity have no within-patient baseline; "
        "the level-only conditional slope baseline is 0 even where the run-in slope is not "
        "estimable. "
        "For shuffled-rate scores, each independent shuffle keeps run-in outcomes fixed and "
        "rebuilds the expanding rate. The independent shuffled-baseline draws use a separate "
        f"seed; score seed {metadata['seed']}, baseline seed {metadata['seed'] + 1}, "
        f"{metadata['draws']} draws each.",
        "",
        "The repository's `wheneval.nine.average_precision` groups tied forecast "
        "values at one threshold; a constant forecast therefore has AP equal to the patient's "
        "scored-set event rate n/T. The within-patient AP baseline instead uses the exact "
        "expected AP of a uniformly random strict ranking.",
        "",
    ]
    for dataset in ("CTN-0094", "REPLACE-BG", "College Experience Study"):
        lines.extend([f"## {dataset}", ""])
        lines.extend([
            "| Forecast | Measure | Eligible n | Score | Usual baseline | Within-patient baseline | Own rate | Shuffled baseline | Score - within | Score - shuffled |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for forecast in FORECASTS:
            for metric in MEASURES:
                row = records[dataset][f"{forecast}:{metric}"]
                score_text = _format_num(row["score"])
                if forecast == "shuffled_rate_so_far":
                    score_text += (f" ({_format_range(row['score_lo'], row['score_hi'])})")
                baseline_text = _format_range(row["shuffled_lo"], row["shuffled_hi"])
                if _scalar(row["shuffled_lo"]) is not None:
                    baseline_text = (f"{_format_num(row['shuffled_baseline'])} "
                                     f"({_format_range(row['shuffled_lo'], row['shuffled_hi'])})")
                lines.append(
                    f"| {forecast} | {MEASURE_LABELS[metric]} | {row['eligible_n']} | "
                    f"{score_text} | {_format_baseline(metric, row['usual_baseline'])} | "
                    f"{_format_num(row['within_baseline'])} | {_format_num(row['own_rate'])} | "
                    f"{baseline_text} | {_format_num(row['diff_within'])} | "
                    f"{_format_num(row['diff_shuffled'])} |"
                )
        lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if value is None else value)
                             for key, value in row.items()})


def run(args: argparse.Namespace) -> dict[str, Any]:
    main_table, table4, np, pd, _nine_metrics, naive_metrics = _load_dependencies()
    _check_ap_ties()
    if not V10_JSON.exists():
        raise FileNotFoundError(f"required comparison file is missing: {V10_JSON}")
    with V10_JSON.open(encoding="utf-8") as handle:
        v10 = json.load(handle)

    pm = table4.load_permutation_module()
    frames, contexts = {}, {}
    for config in table4.COHORTS:
        frame = pd.read_csv(OPEN_DATA / config["forecast_file"])
        required = {"patient", "t", "y", "threshold", "training_cohort_rate",
                    "runin_rate", config["updated_rate_column"]}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{config['forecast_file']}: missing columns {sorted(missing)}")
        frames[config["key"]] = frame
        contexts[config["key"]] = main_table._load_context(pm, table4, config, np, pd)

    rng_score = np.random.default_rng(args.seed)
    rng_baseline = np.random.default_rng(args.seed + 1)
    ordered: dict[str, dict[str, Any]] = {}
    compare_draws: dict[str, dict[str, list[float]]] = {}
    config_order = []
    for config in table4.COHORTS:
        label = config["label"]
        config_order.append(config)
        cohort_records, draws = _analyze_dataset(
            config, frames[config["key"]], contexts[config["key"]], table4,
            main_table, naive_metrics, np, pd, pm, args.draws,
            rng_score, rng_baseline,
        )
        dataset = ("College Experience Study" if label == "College Experience" else label)
        ordered[dataset] = cohort_records
        compare_draws[dataset] = draws

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [ordered[dataset][f"{forecast}:{metric}"]
            for dataset in ("CTN-0094", "REPLACE-BG", "College Experience Study")
            for forecast in FORECASTS for metric in MEASURES]
    csv_path = OUTPUT_DIR / "B2_baselines.csv"
    md_path = OUTPUT_DIR / "B2_baselines.md"
    _write_csv(csv_path, rows)
    md_text = _markdown(ordered, {"seed": args.seed, "draws": args.draws})
    comparison = _reference_comparison(
        {**ordered, "_configs": config_order}, compare_draws, v10, args, np
    )
    md_path.write_text(md_text + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print("\n".join(comparison))
    return ordered


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
