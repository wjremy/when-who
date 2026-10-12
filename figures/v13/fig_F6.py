"""F6: CTN-0094 result 1 with the three scorings and run-in markers."""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from style import (COLORS, SCORING_LABELS, SCORING_ORDER, TWO_COLUMNS, dot_ring_gap,
                   mpl_or_report, read_json, save_figure, write_csv)

SOURCE = ROOT / "open_data/results/v10/main_table.json"
METRIC_ORDER = ["auc", "average_precision", "ppv", "npv", "youden_j",
                "net_benefit", "brier_skill", "calibration_slope"]
METRIC_LABELS = {
    "auc": "AUC", "average_precision": "Average precision", "ppv": "PPV",
    "npv": "NPV", "youden_j": "Sensitivity + specificity − 1",
    "net_benefit": "Net benefit", "brier_skill": "Brier skill",
    "calibration_slope": "Calibration slope",
}
SCORING_CELL = {"pooled": "pooled", "per_patient": "naive", "shuffled_baseline": "tailored"}
COLUMN_LIMITS = {
    "pooled": (-0.6, 1.1),
    "per_patient": (-0.6, 1.1),
    "shuffled_baseline": (-0.6, 0.6),
}
COLUMN_TICKS = {
    "pooled": (-0.5, 0.0, 0.5, 1.0),
    "per_patient": (-0.5, 0.0, 0.5, 1.0),
    "shuffled_baseline": (-0.5, 0.0, 0.5),
}


def number(value):
    if isinstance(value, dict):
        if "mean" in value:
            return number(value["mean"])
        if "treat_none" in value or "treat_all" in value:
            vals = [number(value[k]) for k in ("treat_none", "treat_all") if value.get(k) is not None]
            return max(vals) if vals else float("nan")
        return float("nan")
    if value is None or value == "n/a":
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


def assert_source_x_coordinates(ax, line_start, collection_start, source_values,
                                copies, context):
    """Assert that newly plotted x-data are the unscaled values read from the table."""
    expected = [float(value) for value in source_values if math.isfinite(value)]
    observed = []
    for line in ax.lines[line_start:]:
        observed.extend(float(value) for value in line.get_xdata()
                        if math.isfinite(float(value)))
    for collection in ax.collections[collection_start:]:
        observed.extend(float(point[0]) for point in collection.get_offsets()
                        if math.isfinite(float(point[0])))
    assert sorted(observed) == sorted(expected * copies), (
        f"{context}: plotted x coordinates {sorted(observed)} do not match "
        f"source values {sorted(expected)}"
    )


def collect_data():
    data = read_json(SOURCE)
    rows = []
    ctn = data["datasets"]["CTN-0094"]["metrics"]
    base_metric = {
        "youden_j": "youden_j", "average_precision": "average_precision",
        "auc": "auc", "ppv": "ppv", "npv": "npv", "net_benefit": "net_benefit",
        "brier_skill": "brier_skill", "calibration_slope": "calibration_slope",
    }
    primary = {}
    runin = {}
    for measure in METRIC_ORDER:
        primary[measure] = {}
        runin[measure] = {}
        for scoring in SCORING_ORDER:
            score = value_for(ctn, measure, "shuffled_running", scoring, "score")
            baseline = value_for(ctn, measure, "shuffled_running", scoring, "baseline")
            primary[measure][scoring] = (score, baseline)
            runin_score = value_for(ctn, measure, "runin", scoring, "score")
            runin_baseline = value_for(ctn, measure, "runin", scoring, "baseline")
            runin[measure][scoring] = (runin_score, runin_baseline)
            rows.append({"dataset": "CTN-0094", "measure": measure, "measure_label": METRIC_LABELS[measure],
                         "scoring": scoring, "forecast": "shuffled rate so far",
                         "score": score, "baseline": baseline,
                         "excess": score - baseline if math.isfinite(score) and math.isfinite(baseline) else ""})
            rows.append({"dataset": "CTN-0094", "measure": measure, "measure_label": METRIC_LABELS[measure],
                         "scoring": scoring, "forecast": "run-in rate",
                         "score": runin_score, "baseline": runin_baseline,
                         "excess": runin_score - runin_baseline if math.isfinite(runin_score) and math.isfinite(runin_baseline) else ""})

    marker_values = {}
    for dataset_key, short_label in (("REPLACE-BG", "REPLACE-BG"), ("College Experience", "CES")):
        metric_map = data["datasets"][dataset_key]["metrics"]
        marker_values[short_label] = {}
        for measure in ("auc", "calibration_slope"):
            marker_values[short_label][measure] = {}
            for scoring in SCORING_ORDER:
                score = value_for(metric_map, measure, "shuffled_running", scoring, "score")
                baseline = value_for(metric_map, measure, "shuffled_running", scoring, "baseline")
                marker_values[short_label][measure][scoring] = score
                rows.append({"dataset": short_label, "measure": measure, "measure_label": METRIC_LABELS[measure],
                             "scoring": scoring, "forecast": "shuffled rate so far",
                             "score": score, "baseline": baseline,
                             "excess": score - baseline if math.isfinite(score) and math.isfinite(baseline) else ""})
    for quantity, value in (("test_patients", data["datasets"]["CTN-0094"]["test_patients"]),
                            ("scored_rows", data["datasets"]["CTN-0094"]["scored_rows"]),
                            ("permutation_draws", data["metadata"]["permutation_draws"]),
                            ("seed", data["metadata"]["permutation_seed"])):
        rows.append({"dataset": "CTN-0094", "forecast": "metadata", "quantity": quantity, "value": value})
    return data, primary, runin, marker_values, rows


def main():
    data, primary, runin, markers, rows = collect_data()
    write_csv(ROOT / "results/v13/F6_numbers.csv",
              ["dataset", "measure", "measure_label", "scoring", "forecast", "score", "baseline", "excess",
               "quantity", "value"], rows)
    plt = mpl_or_report()
    if plt is None:
        return
    from matplotlib.lines import Line2D
    from matplotlib.transforms import offset_copy

    fig, axes = plt.subplots(len(METRIC_ORDER), 3,
                             figsize=(TWO_COLUMNS, 184 / 25.4))
    fig.subplots_adjust(left=0.255, right=0.99, top=0.94, bottom=0.225,
                        hspace=0.40, wspace=0.10)
    fig.suptitle("CTN-0094 · shuffled rate so far", x=0.01, ha="left",
                 fontsize=9.5, fontweight="bold")
    marker_styles = {"REPLACE-BG": ("s", "#3A3A3A"), "CES": ("D", "#3A3A3A")}
    def triangle_gap(ax, y, score, baseline, color):
        gap_color = COLORS["better"] if score >= baseline else COLORS["worse"]
        ax.plot([baseline, score], [y, y], color=gap_color, lw=1.35,
                solid_capstyle="round", zorder=2)
        base_transform = offset_copy(ax.transData, fig=fig, x=-1.1, units="points")
        score_transform = offset_copy(ax.transData, fig=fig, x=1.1, units="points")
        ax.scatter([baseline], [y], marker="^", s=22, facecolors=COLORS["paper"],
                   edgecolors=color, linewidths=0.9, zorder=4, transform=base_transform)
        ax.scatter([score], [y], marker="^", s=19, color=color,
                   edgecolors=color, linewidths=0.4, zorder=5, transform=score_transform)

    for col, scoring in enumerate(SCORING_ORDER):
        axes[0, col].set_title(SCORING_LABELS[scoring], color=COLORS[scoring],
                               fontweight="bold", pad=5)
        for row_i, measure in enumerate(METRIC_ORDER):
            ax = axes[row_i, col]
            ax.set_xlim(*COLUMN_LIMITS[scoring])
            ax.set_ylim(-0.42, 0.42)
            ax.axvline(0, color=COLORS["grid"], lw=0.65, zorder=0)
            ax.axhline(0, color="#F1F2F3", lw=0.45, zorder=0)
            score, baseline = primary[measure][scoring]
            if math.isfinite(score) and math.isfinite(baseline):
                line_start, collection_start = len(ax.lines), len(ax.collections)
                dot_ring_gap(ax, 0, score, baseline, score_color=COLORS[scoring],
                             marker_size=4.8, marker_offset_points=1.35)
                assert_source_x_coordinates(ax, line_start, collection_start,
                                            (baseline, score), 2,
                                            f"{measure} {scoring} primary glyph")
            rs, rb = runin[measure][scoring]
            if math.isfinite(rs) and math.isfinite(rb):
                line_start, collection_start = len(ax.lines), len(ax.collections)
                triangle_gap(ax, 0.22, rs, rb, COLORS[scoring])
                assert_source_x_coordinates(ax, line_start, collection_start,
                                            (rb, rs), 2,
                                            f"{measure} {scoring} run-in glyph")
            if measure in ("auc", "calibration_slope"):
                marker_y = -0.19
                for name in ("REPLACE-BG", "CES"):
                    value = markers[name][measure][scoring]
                    marker, edge = marker_styles[name]
                    if math.isfinite(value):
                        line_start, collection_start = len(ax.lines), len(ax.collections)
                        ax.scatter([value], [marker_y], marker=marker, s=17,
                                   facecolors=COLORS[scoring], edgecolors=edge,
                                   linewidths=0.65, zorder=5)
                        assert_source_x_coordinates(ax, line_start, collection_start,
                                                    (value,), 1,
                                                    f"{measure} {scoring} {name} marker")
                    marker_y -= 0.13
            ax.set_yticks([])
            ax.grid(axis="x", color=COLORS["grid"], lw=0.4)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(COLORS["grid"])
            if col == 0:
                ax.set_ylabel(METRIC_LABELS[measure], rotation=0, ha="right", va="center", labelpad=7)
            ax.set_xticks(COLUMN_TICKS[scoring])
            ax.tick_params(axis="x", labelbottom=True, labelsize=7.5, pad=2, length=2)
    handles = [
        Line2D([], [], marker="o", ls="", color=COLORS["pooled"], label="Shuffled rate so far"),
        Line2D([], [], marker="^", ls="", color=COLORS["ink"], label="Run-in rate"),
        Line2D([], [], marker="s", ls="", markerfacecolor="white", markeredgecolor=COLORS["ink"], label="REPLACE-BG"),
        Line2D([], [], marker="D", ls="", markerfacecolor="white", markeredgecolor=COLORS["ink"], label="College Experience (CES)"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=COLORS["ink"], label="Baseline"),
        Line2D([], [], color=COLORS["better"], lw=1.8, label="Looks better"),
        Line2D([], [], color=COLORS["worse"], lw=1.8, label="Looks worse"),
    ]
    fig.supxlabel("Score / baseline", y=0.174, fontsize=8.2)
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.62, 0.045), handlelength=1.4,
               columnspacing=1.2, labelspacing=0.5)
    save_figure(fig, 6)
    plt.close(fig)


if __name__ == "__main__":
    main()
