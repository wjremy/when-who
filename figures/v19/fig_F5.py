"""Figure 5: CTN-0094 forecasts under four scorings."""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from style import COLORS, TWO_COLUMNS, mpl_or_report, read_json, save_figure, write_csv
DARK_AMBER = "#B07800"  # darker than the house amber, for contrast on white (rubric r08)

# This is the only list that controls the figure's column order.
COLUMN_ORDER = ["pooled", "per_patient", "per_patient_own_rate", "shuffled_baseline"]
METRIC_ORDER = ["auc", "average_precision", "ppv", "npv", "youden_j",
                "net_benefit", "brier_skill", "calibration_slope"]
METRIC_LABELS = {
    "auc": "AUC", "average_precision": "Average precision", "ppv": "PPV",
    "npv": "NPV", "youden_j": "Sensitivity + specificity \u2212 1",
    "net_benefit": "Net benefit", "brier_skill": "Brier skill",
    "calibration_slope": "Calibration slope",
}
SCORING_CELL = {"pooled": "pooled", "per_patient": "naive", "shuffled_baseline": "tailored"}
OWN_RATE_SHUFFLE_METRICS = ("net_benefit", "brier_skill")
HINDSIGHT_METRICS = ("ppv", "npv", "average_precision", "net_benefit", "brier_skill")
HINDSIGHT_FORECASTS = (
    ("shuffled_rate_so_far", "shuffled rate so far"),
    ("runin_rate", "run-in rate"),
)


def number(value):
    if isinstance(value, dict):
        if "mean" in value:
            return number(value["mean"])
        if "treat_none" in value or "treat_all" in value:
            vals = [number(value[k]) for k in ("treat_none", "treat_all") if value.get(k) is not None]
            return max(vals) if vals else float("nan")
        return float("nan")
    if value is None or value == "n/a" or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def get_cell(metrics, metric, forecast, scoring):
    return metrics[metric]["cells"][f"{forecast}_{SCORING_CELL[scoring]}"]


def value_for(metrics, measure, forecast, scoring, field):
    if measure == "youden_j":
        a = get_cell(metrics, "sensitivity", forecast, scoring)
        b = get_cell(metrics, "specificity", forecast, scoring)
        va = number(a["mean"] if field == "score" else a.get("chance"))
        vb = number(b["mean"] if field == "score" else b.get("chance"))
        return va + vb - 1 if math.isfinite(va) and math.isfinite(vb) else float("nan")
    cell = get_cell(metrics, measure, forecast, scoring)
    return number(cell.get("mean") if field == "score" else cell.get("chance"))


def baseline_for(metrics, measure, forecast, scoring):
    """Return the reference used for excess in each scoring column."""
    if measure == "calibration_slope" and scoring in ("pooled", "per_patient"):
        return 1.0
    return value_for(metrics, measure, forecast, scoring, "baseline")


def read_hindsight():
    """Read the CTN-0094 hindsight comparisons keyed by forecast and measure."""
    source = ROOT / "work" / "2026-10-13_v18_analyses" / "task2_hindsight.csv"
    records = {}
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == "CTN-0094" and row["forecast"] in {
                    forecast for forecast, _ in HINDSIGHT_FORECASTS}:
                records[(row["forecast"], row["measure"])] = row
    for forecast, _ in HINDSIGHT_FORECASTS:
        for measure in HINDSIGHT_METRICS:
            assert (forecast, measure) in records, f"Missing hindsight row: {forecast}, {measure}"
    return records


def hindsight_pair(record, measure):
    if measure == "brier_skill":
        # against_own_rate is the skill plotted against the zero-skill baseline.
        return number(record["against_own_rate"]), 0.0
    return number(record["score"]), number(record["own_rate_baseline"])


def collect_data():
    data = read_json(ROOT / "open_data" / "results" / "v10" / "main_table.json")
    hindsight = read_hindsight()
    rows = []
    ctn = data["datasets"]["CTN-0094"]["metrics"]
    primary = {}
    runin = {}
    own_rate = {}
    for measure in METRIC_ORDER:
        primary[measure] = {}
        runin[measure] = {}
        own_rate[measure] = {}
        for scoring in COLUMN_ORDER:
            if scoring == "per_patient_own_rate":
                for source_forecast, display_forecast in HINDSIGHT_FORECASTS:
                    if measure in HINDSIGHT_METRICS:
                        score, baseline = hindsight_pair(hindsight[(source_forecast, measure)], measure)
                        own_rate[measure][source_forecast] = (score, baseline)
                        excess = score - baseline if math.isfinite(score) and math.isfinite(baseline) else ""
                    else:
                        score, baseline, excess = "", "", ""
                        own_rate[measure][source_forecast] = (float("nan"), float("nan"))
                    rows.append({"dataset": "CTN-0094", "measure": measure,
                                 "measure_label": METRIC_LABELS[measure],
                                 "scoring": "per_patient_own_rate", "forecast": display_forecast,
                                 "score": score, "baseline": baseline, "excess": excess})
                continue
            score = value_for(ctn, measure, "shuffled_running", scoring, "score")
            baseline = baseline_for(ctn, measure, "shuffled_running", scoring)
            primary[measure][scoring] = (score, baseline)
            runin_score = value_for(ctn, measure, "runin", scoring, "score")
            runin_baseline = baseline_for(ctn, measure, "runin", scoring)
            if scoring == "shuffled_baseline" and measure in OWN_RATE_SHUFFLE_METRICS:
                # As in Table 5: net benefit and Brier skill against the own rate, rebuilt in each
                # shuffle. A within-patient shuffle keeps the event count, so the own rate is the
                # same in every shuffle, and the shuffled baseline of each no-skill forecast equals
                # its score (the shuffled rate so far's score is itself a mean over shuffles).
                score = baseline = float(hindsight[("shuffled_rate_so_far", measure)]["against_own_rate"])
                runin_score = runin_baseline = float(hindsight[("runin_rate", measure)]["against_own_rate"])
                primary[measure][scoring] = (score, baseline)
            runin[measure][scoring] = (runin_score, runin_baseline)
            rows.append({"dataset": "CTN-0094", "measure": measure, "measure_label": METRIC_LABELS[measure],
                         "scoring": scoring, "forecast": "shuffled rate so far",
                         "score": score, "baseline": baseline,
                         "excess": score - baseline if math.isfinite(score) and math.isfinite(baseline) else ""})
            rows.append({"dataset": "CTN-0094", "measure": measure, "measure_label": METRIC_LABELS[measure],
                         "scoring": scoring, "forecast": "run-in rate",
                         "score": runin_score, "baseline": runin_baseline,
                         "excess": runin_score - runin_baseline
                         if math.isfinite(runin_score) and math.isfinite(runin_baseline) else ""})

    marker_values = {}
    for dataset_key, short_label in (("REPLACE-BG", "REPLACE-BG"), ("College Experience", "CES")):
        metric_map = data["datasets"][dataset_key]["metrics"]
        marker_values[short_label] = {}
        for measure in ("auc", "calibration_slope"):
            marker_values[short_label][measure] = {}
            for scoring in COLUMN_ORDER:
                if scoring == "per_patient_own_rate":
                    continue
                score = value_for(metric_map, measure, "shuffled_running", scoring, "score")
                baseline = baseline_for(metric_map, measure, "shuffled_running", scoring)
                marker_values[short_label][measure][scoring] = (
                    score - baseline if math.isfinite(score) and math.isfinite(baseline)
                    else float("nan")
                )
                rows.append({"dataset": short_label, "measure": measure, "measure_label": METRIC_LABELS[measure],
                             "scoring": scoring, "forecast": "shuffled rate so far",
                             "score": score, "baseline": baseline,
                             "excess": score - baseline if math.isfinite(score) and math.isfinite(baseline) else ""})
    for quantity, value in (("test_patients", data["datasets"]["CTN-0094"]["test_patients"]),
                            ("scored_rows", data["datasets"]["CTN-0094"]["scored_rows"]),
                            ("permutation_draws", data["metadata"]["permutation_draws"]),
                            ("seed", data["metadata"]["permutation_seed"])):
        rows.append({"dataset": "CTN-0094", "forecast": "metadata", "quantity": quantity, "value": value})
    return data, primary, runin, own_rate, marker_values, rows


def main():
    data, primary, runin, own_rate, markers, rows = collect_data()
    write_csv(ROOT / "results" / "v19" / "F5_numbers.csv",
              ["dataset", "measure", "measure_label", "scoring", "forecast", "score", "baseline", "excess",
               "quantity", "value"], rows)
    plt = mpl_or_report()
    if plt is None:
        return
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    fig, axes = plt.subplots(len(METRIC_ORDER), len(COLUMN_ORDER),
                             figsize=(TWO_COLUMNS, 184 / 25.4), squeeze=False)
    fig.subplots_adjust(left=0.255, right=0.99, top=0.91, bottom=0.215,
                        hspace=0.46, wspace=0.16)
    marker_styles = {"REPLACE-BG": ("s", "#3A3A3A"), "CES": ("D", "#3A3A3A")}
    title_labels = {
        "pooled": "Pooled vs\nusual",
        "per_patient": "Per-patient vs\nusual",
        "per_patient_own_rate": "Per-patient vs\nown rate",
        "shuffled_baseline": "Per-patient vs\nshuffled",
    }
    scoring_colors = {
        "pooled": COLORS["pooled"],
        "per_patient": DARK_AMBER,
        "per_patient_own_rate": DARK_AMBER,
        "shuffled_baseline": COLORS["shuffled_baseline"],
    }
    no_own_rate = {"auc", "youden_j", "calibration_slope"}

    def subtract(score, baseline):
        return (score - baseline
                if math.isfinite(score) and math.isfinite(baseline)
                else float("nan"))

    def nice_step(max_abs):
        raw = max(max_abs * 1.12, 0.05)
        power = 10 ** math.floor(math.log10(raw))
        fraction = raw / power
        for candidate in (1.0, 2.0, 2.5, 5.0, 10.0):
            if fraction <= candidate:
                return candidate * power
        return 10.0 * power

    def tick_text(value, _position):
        return "0" if abs(value) < 1e-12 else f"{value:.2g}"

    cell_excess = {}
    row_steps = {}
    for measure in METRIC_ORDER:
        magnitudes = []
        for scoring in COLUMN_ORDER:
            if scoring == "per_patient_own_rate":
                primary_pair = own_rate[measure]["shuffled_rate_so_far"]
                runin_pair = own_rate[measure]["runin_rate"]
            else:
                primary_pair = primary[measure][scoring]
                runin_pair = runin[measure][scoring]
            values = (subtract(*primary_pair), subtract(*runin_pair))
            cell_excess[(measure, scoring)] = values
            magnitudes.extend(abs(value) for value in values if math.isfinite(value))
        row_steps[measure] = nice_step(max(magnitudes, default=0.0))

    for col, scoring in enumerate(COLUMN_ORDER):
        color = scoring_colors[scoring]
        axes[0, col].set_title(title_labels[scoring], color=color, fontsize=9.5,
                               fontweight="bold", pad=4, linespacing=1.02)
        for row_i, measure in enumerate(METRIC_ORDER):
            ax = axes[row_i, col]
            if scoring == "per_patient_own_rate" and measure in no_own_rate:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9,
                        color=COLORS["muted"])
                continue

            step = row_steps[measure]
            ax.set_xlim(-1.22 * step, 1.22 * step)
            ax.set_ylim(-0.56, 0.56)
            ax.axvline(0, color=COLORS["ink"], lw=1.35, zorder=1)
            ax.set_axisbelow(True)
            ax.grid(axis="x", color=COLORS["grid"], lw=0.45, zorder=0)

            shuffled_excess, runin_excess = cell_excess[(measure, scoring)]
            if math.isfinite(shuffled_excess):
                ax.scatter([shuffled_excess], [0.15], marker="o", s=34,
                           color=color, edgecolors=COLORS["paper"],
                           linewidths=0.45, zorder=4)
            if math.isfinite(runin_excess):
                ax.scatter([runin_excess], [-0.15], marker="^", s=36,
                           color=color, edgecolors=COLORS["paper"],
                           linewidths=0.45, zorder=4)

            ax.set_yticks([])
            ax.set_xticks((-step, 0.0, step))
            ax.xaxis.set_major_formatter(FuncFormatter(tick_text))
            ax.tick_params(axis="x", labelbottom=True, labelsize=8.0, pad=1.5, length=2)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(COLORS["grid"])
            if col == 0:
                ax.set_ylabel(METRIC_LABELS[measure], rotation=0, ha="right",
                              va="center", labelpad=7, fontsize=9.0)

    handles = [
        Line2D([], [], marker="o", ls="", color=COLORS["ink"],
               label="Shuffled rate so far"),
        Line2D([], [], marker="^", ls="", color=COLORS["ink"],
               label="Run-in rate"),
        Line2D([], [], color=COLORS["ink"], lw=1.35, label="Zero excess"),
    ]
    fig.supxlabel("Excess (score minus baseline)", x=0.62, y=0.145, fontsize=9.0)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.62, 0.045), handlelength=1.2,
               columnspacing=1.6, labelspacing=0.45, fontsize=8.6)
    save_figure(fig, 5)
    plt.close(fig)


if __name__ == "__main__":
    main()
