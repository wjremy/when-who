"""Figure 6: cohort distributions in the Goldenholz reconstruction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
from style import COLORS, TWO_COLUMNS, mpl_or_report, save_figure, write_csv  # noqa: E402


PRIMARY_SCENARIO = "primary_midpoint_lengths_spread_counts"
PUBLISHED_PER_PATIENT_AUC = 0.43
PUBLISHED_POOLED_AUC = 0.82
USUAL_BASELINE_AUC = 0.5


def collect_data():
    cohorts = pd.read_csv(ROOT / "results" / "v19" / "F6_cohorts.csv")
    pooled = pd.read_csv(ROOT / "results" / "v14" / "B1_goldenholz_pooled.csv")
    primary_rows = pooled.loc[pooled["scenario"] == PRIMARY_SCENARIO]
    other_rows = pooled.loc[pooled["scenario"] != PRIMARY_SCENARIO]
    assert len(primary_rows) == 1, f"Expected one v14 primary row, found {len(primary_rows)}"
    assert len(other_rows) > 0, "B1_goldenholz_pooled.csv has no non-primary scenarios"
    assert len(cohorts) == 5000, f"Expected 5,000 cohorts, found {len(cohorts)}"

    per_patient = cohorts["per_patient_auc"].to_numpy(dtype=float)
    pooled_auc = cohorts["pooled_auc"].to_numpy(dtype=float)
    primary = primary_rows.iloc[0]
    other_min = float(other_rows["per_patient_mean"].min())
    other_max = float(other_rows["per_patient_mean"].max())
    a0_mean = float(primary["a0_mean"])

    bins_a = np.linspace(0.25, 0.55, 31)
    bins_b = np.linspace(0.48, 0.90, 43)
    counts_a, edges_a = np.histogram(per_patient, bins=bins_a)
    counts_b, edges_b = np.histogram(pooled_auc, bins=bins_b)

    rows = []
    for panel, counts, edges in (("a", counts_a, edges_a), ("b", counts_b, edges_b)):
        for i, count in enumerate(counts):
            rows.append({
                "panel": panel,
                "element": "histogram_bin",
                "label": "cohort count",
                "value": int(count),
                "lower": float(edges[i]),
                "upper": float(edges[i + 1]),
                "count": int(count),
            })
    for label, value in (
        ("published 0.43", PUBLISHED_PER_PATIENT_AUC),
        ("usual baseline 0.5", USUAL_BASELINE_AUC),
    ):
        rows.append({"panel": "a", "element": "reference_line", "label": label,
                     "value": value, "lower": "", "upper": "", "count": ""})
    rows.append({"panel": "a", "element": "reference_band", "label": "other reconstructions",
                 "value": "", "lower": other_min, "upper": other_max, "count": ""})
    for label, value in (
        ("A0, Equation 1", a0_mean),
        ("published 0.82", PUBLISHED_POOLED_AUC),
        ("usual baseline 0.5", USUAL_BASELINE_AUC),
    ):
        rows.append({"panel": "b", "element": "reference_line", "label": label,
                     "value": value, "lower": "", "upper": "", "count": ""})

    return per_patient, pooled_auc, (other_min, other_max), a0_mean, bins_a, bins_b, rows


def main() -> None:
    per_patient, pooled_auc, other_range, a0_mean, bins_a, bins_b, rows = collect_data()
    write_csv(ROOT / "results" / "v19" / "F6_numbers.csv",
              ["panel", "element", "label", "value", "lower", "upper", "count"], rows)

    plt = mpl_or_report()
    if plt is None:
        return
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 2, figsize=(TWO_COLUMNS, 83 / 25.4), squeeze=False)
    ax_a, ax_b = axes[0]
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.17, top=0.90, wspace=0.34)

    ax_a.axvspan(other_range[0], other_range[1], color=COLORS["muted"], alpha=0.16,
                 zorder=0)
    ax_a.hist(per_patient, bins=bins_a, color=COLORS["per_patient"], alpha=0.78,
              edgecolor=COLORS["paper"], linewidth=0.45, zorder=2)
    ax_a.axvline(PUBLISHED_PER_PATIENT_AUC, color=COLORS["worse"], lw=1.5, zorder=4)
    ax_a.axvline(USUAL_BASELINE_AUC, color=COLORS["muted"], lw=1.0,
                 linestyle=(0, (3, 2)), zorder=4)
    ax_a.set_xlim(0.21, 0.56)
    ax_a.set_xlabel("Mean per-patient AUC of a forecast without skill")
    ax_a.set_ylabel("Cohorts")
    ax_a.text(0.0, 1.055, "a", transform=ax_a.transAxes, ha="left", va="bottom",
              fontsize=10, fontweight="bold", color=COLORS["ink"], clip_on=False)
    ax_a.legend(handles=[
        Line2D([], [], color=COLORS["worse"], lw=1.5, label="published 0.43"),
        Line2D([], [], color=COLORS["muted"], lw=1.0, linestyle=(0, (3, 2)),
               label="usual baseline 0.5"),
        Patch(facecolor=COLORS["muted"], alpha=0.16, edgecolor="none",
              label="other reconstructions"),
    ], loc="upper left", frameon=False, fontsize=7.0,
        handlelength=1.5, labelspacing=0.35, borderaxespad=0.25)

    ax_b.hist(pooled_auc, bins=bins_b, color=COLORS["pooled"], alpha=0.78,
              edgecolor=COLORS["paper"], linewidth=0.45, zorder=2)
    ax_b.axvline(a0_mean, color=COLORS["shuffled_baseline"], lw=1.5, zorder=4)
    ax_b.axvline(PUBLISHED_POOLED_AUC, color=COLORS["worse"], lw=1.5, zorder=4)
    ax_b.axvline(USUAL_BASELINE_AUC, color=COLORS["muted"], lw=1.0,
                 linestyle=(0, (3, 2)), zorder=4)
    ax_b.set_xlim(0.48, 0.90)
    ax_b.set_xlabel("Pooled AUC")
    ax_b.set_ylabel("Cohorts")
    ax_b.text(0.0, 1.055, "b", transform=ax_b.transAxes, ha="left", va="bottom",
              fontsize=10, fontweight="bold", color=COLORS["ink"], clip_on=False)
    ax_b.legend(handles=[
        Line2D([], [], color=COLORS["shuffled_baseline"], lw=1.5,
               label="A0, Equation 1"),
        Line2D([], [], color=COLORS["worse"], lw=1.5, label="published 0.82"),
        Line2D([], [], color=COLORS["muted"], lw=1.0, linestyle=(0, (3, 2)),
               label="usual baseline 0.5"),
    ], loc="upper left", bbox_to_anchor=(0.06, 1.0), frameon=False, fontsize=7.0,
        handlelength=1.5, labelspacing=0.35, borderaxespad=0.25)

    for ax in (ax_a, ax_b):
        ax.set_ylim(0, ax.get_ylim()[1] * 1.4)  # headroom: legends above the bars
        ax.grid(axis="y", color=COLORS["grid"], lw=0.45, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=2, pad=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(COLORS["grid"])
        ax.spines["bottom"].set_color(COLORS["grid"])

    save_figure(fig, 6)
    plt.close(fig)


if __name__ == "__main__":
    main()
