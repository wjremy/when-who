"""F7: fitted-model scores, with shuffled mean and median excess separated."""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from style import (COLORS, SCORING_LABELS, SCORING_ORDER, TWO_COLUMNS, dot_ring_gap,
                   mpl_or_report, read_json, save_figure, write_csv)

MODEL_SOURCE = ROOT / "open_data/results/v10/fitted_model_package.json"
TABLE_SOURCE = ROOT / "open_data/results/v10/main_table.json"
ORDER = ["auc", "average_precision", "sensitivity", "specificity", "ppv", "npv",
         "net_benefit", "brier_skill", "calibration_slope"]
LABELS = {"auc": "AUC", "average_precision": "Average precision", "sensitivity": "Sensitivity",
          "specificity": "Specificity", "ppv": "PPV", "npv": "NPV", "net_benefit": "Net benefit",
          "brier_skill": "Brier skill", "calibration_slope": "Conditional within-\npatient slope"}


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


def collect_data():
    model = read_json(MODEL_SOURCE)
    table = read_json(TABLE_SOURCE)
    model_rows = {r["metric"]: r for r in model["rows"]}
    baselines = table["datasets"]["CTN-0094"]["metrics"]
    result = {}
    csv_rows = []
    for metric in ORDER:
        r = model_rows[metric]
        pooled_baseline = number(baselines[metric]["cells"]["shuffled_running_pooled"].get("chance"))
        naive_baseline = number(baselines[metric]["cells"]["shuffled_running_naive"].get("chance"))
        values = {
            "pooled": (number(r["pooled"]), pooled_baseline),
            "per_patient": (number(r["usual_mean"]), naive_baseline),
            "shuffled_baseline": (number(r["mean_observed"]), number(r["mean_chance"])),
        }
        conditional = {}
        if metric == "calibration_slope":
            conditional = {
                "score": number(r.get("conditional_slope")),
                "baseline": number(r.get("conditional_slope_chance")),
                "excess": number(r.get("conditional_slope_excess")),
                "p": number(r.get("conditional_slope_p")),
            }
            values["shuffled_baseline"] = (conditional["score"], conditional["baseline"])
        result[metric] = {"values": values,
                          "median_excess": number(r["median_excess"]),
                          "iqr_excess": [number(v) for v in r["iqr_excess"]],
                          "n_eligible": int(r["n_eligible"]),
                          "conditional": conditional}
        for scoring, (score, baseline) in values.items():
            csv_rows.append({"measure": metric, "measure_label": LABELS[metric], "scoring": scoring,
                             "score": score, "baseline": baseline,
                             "excess": score - baseline if math.isfinite(score) and math.isfinite(baseline) else "",
                             "excess_kind": "mean minus mean" if scoring == "shuffled_baseline" else "score minus usual baseline",
                             "median_excess": result[metric]["median_excess"],
                             "iqr_low": result[metric]["iqr_excess"][0],
                             "iqr_high": result[metric]["iqr_excess"][1],
                             "conditional_slope": conditional.get("score", ""),
                             "conditional_slope_baseline": conditional.get("baseline", ""),
                             "conditional_slope_excess": conditional.get("excess", ""),
                             "conditional_slope_p": conditional.get("p", ""),
                             "n_eligible": result[metric]["n_eligible"]})
    for quantity, value in (("patients", model["settings"]["patients"]),
                            ("scored_rows", model["settings"]["scored_rows"]),
                            ("shuffles", model["settings"]["n_perm"]),
                            ("seed", model["settings"]["seed"])):
        csv_rows.append({"measure": "", "measure_label": "", "scoring": "", "score": "",
                         "baseline": "", "excess": "", "excess_kind": "metadata",
                         "median_excess": "", "iqr_low": "", "iqr_high": "",
                         "conditional_slope": "", "conditional_slope_baseline": "",
                         "conditional_slope_excess": "", "conditional_slope_p": "",
                         "n_eligible": "", "quantity": quantity, "value": value})
    return model, result, csv_rows


def main():
    model, result, rows = collect_data()
    write_csv(ROOT / "results/v13/F7_numbers.csv",
              ["measure", "measure_label", "scoring", "score", "baseline", "excess",
               "excess_kind", "median_excess", "iqr_low", "iqr_high",
               "conditional_slope", "conditional_slope_baseline", "conditional_slope_excess",
               "conditional_slope_p", "n_eligible",
               "quantity", "value"], rows)
    plt = mpl_or_report()
    if plt is None:
        return
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(len(ORDER), 4, figsize=(TWO_COLUMNS, 186 / 25.4))
    fig.subplots_adjust(left=0.235, right=0.99, top=0.92, bottom=0.245,
                        hspace=0.34, wspace=0.10)
    fig.suptitle("Fitted model scores versus the shuffled baseline", x=0.01,
                 ha="left", fontsize=9.5, fontweight="bold")
    headings = ["Pooled", "Per-patient", "Shuffled-baseline\n(means)", "Excess summary\n(median [IQR])"]
    for col, title in enumerate(headings):
        axes[0, col].set_title(title, color=COLORS[SCORING_ORDER[col]] if col < 3 else COLORS["ink"],
                               fontweight="bold", pad=5)
    xmin, xmax = -1.12, 1.85
    for row_i, metric in enumerate(ORDER):
        for col, scoring in enumerate(SCORING_ORDER):
            ax = axes[row_i, col]
            score, baseline = result[metric]["values"][scoring]
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(-0.42, 0.42)
            ax.axvline(0, color=COLORS["grid"], lw=0.6, zorder=0)
            if math.isfinite(score) and math.isfinite(baseline):
                dot_ring_gap(ax, 0, score, baseline, score_color=COLORS[scoring], marker_size=4.6)
            ax.set_yticks([])
            ax.grid(axis="x", color=COLORS["grid"], lw=0.35)
            ax.set_axisbelow(True)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color(COLORS["grid"])
            if col == 0:
                ax.set_ylabel(LABELS[metric], rotation=0, ha="right", va="center", labelpad=7)
            if row_i != len(ORDER) - 1:
                ax.tick_params(axis="x", labelbottom=False, length=0)
        ax = axes[row_i, 3]
        conditional = result[metric]["conditional"]
        is_conditional_slope = metric == "calibration_slope"
        q1, q3 = result[metric]["iqr_excess"]
        med = conditional["excess"] if is_conditional_slope else result[metric]["median_excess"]
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.42, 0.42)
        ax.axvline(0, color=COLORS["muted"], lw=0.6, ls=(0, (2, 2)), zorder=0)
        gap_color = COLORS["better"] if med >= 0 else COLORS["worse"]
        if not is_conditional_slope and math.isfinite(q1) and math.isfinite(q3) and math.isfinite(med):
            ax.plot([q1, q3], [0, 0], color=gap_color, lw=2.0, solid_capstyle="round", zorder=2)
            ax.plot([q1, q1], [-0.07, 0.07], color=gap_color, lw=0.9, zorder=2)
            ax.plot([q3, q3], [-0.07, 0.07], color=gap_color, lw=0.9, zorder=2)
            ax.scatter([med], [0], marker="D", s=18, color=gap_color, zorder=4)
        elif is_conditional_slope and math.isfinite(med):
            ax.scatter([med], [0], marker="D", s=18, color=gap_color, zorder=4)
            ax.text(0.98, 0.88, f"p = {conditional['p']:.3f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8.0, color=COLORS["muted"])
        ax.set_yticks([])
        ax.grid(axis="x", color=COLORS["grid"], lw=0.35)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(COLORS["grid"])
        if row_i != len(ORDER) - 1:
            ax.tick_params(axis="x", labelbottom=False, length=0)
    for col in range(4):
        axes[-1, col].set_xticks([-1, 0, 1])
        axes[-1, col].tick_params(axis="x", labelsize=8.0, pad=2)
    handles = [
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=COLORS["ink"], label="Baseline"),
        Line2D([], [], marker="o", ls="", color=COLORS["ink"], label="Score"),
        Line2D([], [], color=COLORS["better"], lw=1.8, label="Looks better / positive excess"),
        Line2D([], [], color=COLORS["worse"], lw=1.8, label="Looks worse / negative excess"),
        Line2D([], [], marker="D", ls="", color=COLORS["better"], label="Excess; bar = IQR when shown"),
    ]
    fig.supxlabel("Metric value / excess", y=0.188, fontsize=8.2)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.58, 0.075), handlelength=1.45, columnspacing=1.0,
               labelspacing=0.45)
    fig.text(0.99, 0.025, f"{model['settings']['patients']} patients · {model['settings']['scored_rows']:,} scored rows · {model['settings']['n_perm']} shuffles",
             ha="right", va="bottom", fontsize=8.0, color=COLORS["muted"])
    save_figure(fig, 7)
    plt.close(fig)


if __name__ == "__main__":
    main()
