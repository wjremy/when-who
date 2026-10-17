"""Figure 9 (Section 5.3): the fitted gradient-boosting model, one point per test patient:
x = the patient's shuffled baseline (mean of 200 shuffles with rebuilt inputs), y = the patient's
per-patient AUC. Points above the diagonal show within-patient skill under the correction; the
dashed line is the usual baseline 0.5. Filled points lie above the diagonal, open points on or below
it (grayscale-safe). The cross marks the cohort means.

Input: results/v19/F9_patients.csv (code/v19_fitted_patient_baselines.py).
Output: figures/v19/F9.pdf/.png, results/v19/F9_numbers.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
from style import COLORS, TWO_COLUMNS, mpl_or_report, save_figure, write_csv  # noqa: E402

COHORTS = ("CTN-0094", "REPLACE-BG", "College Experience Study")


def numbers(df: pd.DataFrame) -> list[dict]:
    out = []
    for name in COHORTS:
        d = df[df.cohort == name]
        out.append({
            "cohort": name, "patients": len(d),
            "mean_auc": round(d.auc.mean(), 4), "mean_shuffled_baseline": round(d.shuffled_baseline.mean(), 4),
            "share_above_diagonal": round(float((d.auc > d.shuffled_baseline).mean()), 4),
            "share_baseline_below_0_5": round(float((d.shuffled_baseline < 0.5).mean()), 4),
            "share_auc_below_0_5": round(float((d.auc < 0.5).mean()), 4),
            "baseline_min": round(d.shuffled_baseline.min(), 4), "baseline_median": round(d.shuffled_baseline.median(), 4),
            "baseline_max": round(d.shuffled_baseline.max(), 4),
        })
    return out


def panel(ax, d: pd.DataFrame, title: str, big: bool) -> None:
    above = d.auc > d.shuffled_baseline
    size = 16 if big else 9
    ax.plot([0, 1], [0, 1], color=COLORS["ink"], lw=0.9, zorder=1)
    ax.axhline(0.5, color=COLORS["muted"], lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.scatter(d.shuffled_baseline[above], d.auc[above], s=size, color=COLORS["pooled"], alpha=0.75,
               linewidths=0, zorder=3)
    ax.scatter(d.shuffled_baseline[~above], d.auc[~above], s=size, facecolors=COLORS["paper"],
               edgecolors=COLORS["pooled"], linewidths=0.8, zorder=3)
    mx, my = d.shuffled_baseline.mean(), d.auc.mean()
    ax.scatter([mx], [my], marker="X", s=70 if big else 40, color=COLORS["worse"], edgecolors=COLORS["paper"],
               linewidths=0.6, zorder=5)
    ax.annotate(f"mean ({mx:.2f}, {my:.2f})", xy=(mx, my), xytext=(8, -2) if big else (5, -9),
                textcoords="offset points", fontsize=7.4 if big else 6.4, color=COLORS["worse"],
                ha="left", va="top", zorder=6,
                bbox=dict(boxstyle="square,pad=0.15", facecolor=COLORS["paper"], edgecolor="none", alpha=0.85))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ticks = [0, 0.25, 0.5, 0.75, 1] if big else [0, 0.5, 1]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.set_title(title, loc="left", fontweight="bold", fontsize=8.5 if big else 7.6, pad=4)
    ax.grid(color=COLORS["grid"], lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["grid"])
    ax.tick_params(length=2, pad=2, labelsize=7.5 if big else 6.5)


def main() -> None:
    df = pd.read_csv(ROOT / "results" / "v19" / "F9_patients.csv")
    rows = numbers(df)
    write_csv(ROOT / "results" / "v19" / "F9_numbers.csv", list(rows[0].keys()), rows)
    plt = mpl_or_report()
    if plt is None:
        return
    height = 108 / 25.4
    fig = plt.figure(figsize=(TWO_COLUMNS, height))
    # Square panels sized in inches, then the group is centred: a is 80% of the height, b and c 36% each.
    big, small, gap = 0.80 * height, 0.36 * height, 0.60
    left = (TWO_COLUMNS - (big + gap + small)) / 2 + 0.12
    ax_a = fig.add_axes([left / TWO_COLUMNS, 0.11, big / TWO_COLUMNS, 0.80])
    x_bc = (left + big + gap) / TWO_COLUMNS
    ax_b = fig.add_axes([x_bc, 0.55, small / TWO_COLUMNS, 0.36])
    ax_c = fig.add_axes([x_bc, 0.11, small / TWO_COLUMNS, 0.36])
    panel(ax_a, df[df.cohort == "CTN-0094"], "a  CTN-0094", True)
    panel(ax_b, df[df.cohort == "REPLACE-BG"], "b  REPLACE-BG", False)
    panel(ax_c, df[df.cohort == "College Experience Study"], "c  College Experience Study", False)
    ax_a.set_xlabel("Patient's shuffled baseline (per-patient AUC)")
    ax_a.set_ylabel("Patient's per-patient AUC")
    ax_a.text(0.97, 0.985, "y = x: no within-patient skill", rotation=45, rotation_mode="anchor",
              ha="right", va="bottom", fontsize=6.8, color=COLORS["ink"], transform=ax_a.transData)
    ax_a.text(0.995, 0.515, "usual baseline 0.5", ha="right", va="bottom", fontsize=6.8, color=COLORS["muted"])
    save_figure(fig, 9)
    plt.close(fig)


if __name__ == "__main__":
    main()
