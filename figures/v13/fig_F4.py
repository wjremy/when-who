"""F4: schematic 2×2 of errors moving scores or baselines."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from style import COLORS, TWO_COLUMNS, dot_ring_gap, mpl_or_report, save_figure, write_csv

CELLS = [
    (1, "Between-patient bias", "Moves the score", ["AUC", "AP", "Sensitivity + specificity − 1", "PPV", "NPV", "Calibration slope"], "score_up"),
    (2, "Between-patient bias", "Moves the baseline", ["AP", "PPV", "NPV", "Net benefit", "Brier skill"], "baseline_varies"),
    (3, "Finite-record bias", "Moves the score", ["AUC", "AP", "Sensitivity + specificity − 1", "PPV", "NPV", "Net benefit", "Brier skill", "Calibration slope"], "score_down"),
    (4, "Finite-record bias", "Moves the baseline", ["Net benefit", "Brier skill"], "baseline_up"),
]


def build_rows():
    rows = []
    for error_id, bias, location, measures, direction in CELLS:
        for measure in measures:
            rows.append({"error_id": error_id, "bias": bias, "location": location,
                         "measure": measure, "direction": direction,
                         "magnitude": "schematic only"})
    return rows


def main():
    rows = build_rows()
    write_csv(ROOT / "results/v13/F4_numbers.csv",
              ["error_id", "bias", "location", "measure", "direction", "magnitude"], rows)
    plt = mpl_or_report()
    if plt is None:
        return
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(TWO_COLUMNS, 112 / 25.4))
    fig.subplots_adjust(left=0.23, right=0.985, top=0.82, bottom=0.205,
                        wspace=0.12, hspace=0.40)
    fig.suptitle("Four errors: when the score or baseline moves", x=0.01,
                 ha="left", fontsize=9.5, fontweight="bold")
    fig.text(0.408, 0.865, "Moves the score", ha="center", va="bottom",
             fontsize=8.5, fontweight="bold")
    fig.text(0.807, 0.865, "Moves the baseline", ha="center", va="bottom",
             fontsize=8.5, fontweight="bold")
    fig.text(0.015, 0.69, "Between-patient bias", ha="left", va="center",
             fontsize=8.0, fontweight="bold", color=COLORS["muted"])
    fig.text(0.015, 0.33, "Finite-record bias", ha="left", va="center",
             fontsize=8.0, fontweight="bold", color=COLORS["muted"])

    def shifted_glyph(ax, y, score_x, baseline_x, score_y, baseline_y):
        color = COLORS["better"] if score_x >= baseline_x else COLORS["worse"]
        ax.plot([baseline_x, score_x], [baseline_y, score_y], color=color,
                lw=1.5, solid_capstyle="round", zorder=2)
        ax.scatter([baseline_x], [baseline_y], s=14, marker="o", facecolors="white",
                   edgecolors=COLORS["pooled"], linewidths=1.0, zorder=3)
        ax.scatter([score_x], [score_y], s=13, marker="o", color=COLORS["pooled"],
                   edgecolors=COLORS["pooled"], linewidths=0.45, zorder=4)

    for ax, (error_id, bias, location, measures, direction) in zip(axes.flat, CELLS):
        ax.text(0.02, 1.015, f"Error {error_id}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8.0, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.55, len(measures) - 0.45)
        for i, measure in enumerate(measures):
            y = len(measures) - 1 - i
            ax.text(0.01, y, measure, ha="left", va="center", fontsize=8.0,
                    color=COLORS["ink"], clip_on=False)
            center = y
            score_x = 0.82
            baseline_x = 0.74
            if direction == "score_up":
                shifted_glyph(ax, center, 0.92, 0.78, center + 0.12, center - 0.12)
            elif direction == "score_down":
                shifted_glyph(ax, center, 0.72, 0.88, center - 0.12, center + 0.12)
            elif direction == "baseline_up":
                shifted_glyph(ax, center, 0.79, 0.91, center - 0.12, center + 0.12)
            else:
                # Error 2 can shift the baseline either way while the score stays fixed.
                shifted_glyph(ax, center + 0.10, score_x, 0.72, center + 0.10, center + 0.10)
                shifted_glyph(ax, center - 0.10, score_x, 0.92, center - 0.10, center - 0.10)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines[:].set_visible(False)
        ax.set_axisbelow(True)
    handles = [
        Line2D([], [], marker="o", ls="", markerfacecolor=COLORS["pooled"], markeredgecolor=COLORS["pooled"], label="Score"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=COLORS["pooled"], label="Baseline"),
        Line2D([], [], color=COLORS["better"], lw=1.8, label="Looks better"),
        Line2D([], [], color=COLORS["worse"], lw=1.8, label="Looks worse"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.50, 0.105), handlelength=1.5, columnspacing=1.3)
    fig.text(0.99, 0.025, "Direction shown; shifts are schematic, not to scale.",
             ha="right", va="bottom", fontsize=8.0, color=COLORS["muted"])
    save_figure(fig, 4)
    plt.close(fig)


if __name__ == "__main__":
    main()
