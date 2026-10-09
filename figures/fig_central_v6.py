"""Central figure: pooled metrics against within-patient AUC, reference curves and three cohorts.

Reads code/cohort_curves_results.csv (simulation matched to each cohort) and code/cohort_points.csv
(observed test-set results). Run from this project with: python figures/fig_central_v6.py
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
COHORTS = (("REPLACE-BG", "REPLACE-BG: nocturnal hypoglycaemia"),
           ("CTN-0094", "CTN-0094: weekly opioid use"),
           ("College Experience", "College Experience Study: symptoms"))
POOLED = "#0072B2"   # blue
WITHIN = "#D55E00"   # vermilion
GREY = "#8a8a8a"
MODEL_MARKERS = {"gradient_boosting": "o", "logistic_regression": "o"}
RATE_MARKERS = {"runin_rate": "D", "retrospective_rate": "D"}


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def curve(rows, cohort, forecast, metric, version):
    """(within-patient AUC, metric) pairs along the signal grid, ordered by signal strength."""
    x = {r["omega"]: num(r["mean"]) for r in rows
         if r["cohort"] == cohort and r["forecast"] == forecast and r["metric"] == "auc" and r["version"] == "within"}
    pts = [(num(r["omega"]), x.get(r["omega"]), num(r["mean"])) for r in rows
           if r["cohort"] == cohort and r["forecast"] == forecast and r["metric"] == metric and r["version"] == version]
    pts = sorted(p for p in pts if None not in p)
    return [p[1] for p in pts], [p[2] for p in pts]


def main():
    plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "axes.linewidth": 0.7})
    curves = read_rows(ROOT / "code" / "cohort_curves_results.csv")
    points = read_rows(ROOT / "code" / "cohort_points.csv")
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 7.4), sharex=True)

    for i, (cohort, title) in enumerate(COHORTS):
        ax_auc, ax_j = axes[i]
        obs = [r for r in points if r["cohort"] == cohort]

        # Left: pooled AUC against within-patient AUC.
        ax_auc.plot([0.5, 0.9], [0.5, 0.9], color="#A6A6A6", lw=0.8)
        xs, ys = curve(curves, cohort, "true_risk", "auc", "pooled")
        ax_auc.plot(xs, ys, color=POOLED, lw=2.0)
        a0 = ys[xs.index(0.5)]
        ax_auc.axhspan(a0 - 0.006, a0 + 0.006, color=POOLED, alpha=0.10, lw=0, zorder=0)
        ax_auc.plot([0.5, 0.525], [a0, a0], color=POOLED, lw=2.0, zorder=3)
        ax_auc.text(0.532, a0, "A0", color=POOLED, fontsize=8.5, va="center", ha="left",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.0}, zorder=4)
        for r in obs:
            x, y = num(r["within_auc"]), num(r["pooled_auc"])
            if x is None or y is None:
                continue
            if r["forecast"] in MODEL_MARKERS:
                ax_auc.scatter([x], [y], s=26, marker=MODEL_MARKERS[r["forecast"]], color=POOLED,
                               edgecolors="black", linewidths=0.5, zorder=5)
            elif r["forecast"] in RATE_MARKERS:
                ax_auc.scatter([x], [y], s=26, marker=RATE_MARKERS[r["forecast"]], facecolors="white",
                               edgecolors=POOLED, linewidths=1.0, zorder=5, clip_on=False)
        ax_auc.set_ylim(0.45, 1.0)
        ax_auc.set_ylabel("Pooled AUC", fontsize=9.5)
        cohort_label = {"REPLACE-BG": "REPLACE-BG · nocturnal hypoglycaemia",
                        "CTN-0094": "CTN-0094 · weekly opioid use",
                        "College Experience": "College Experience Study · symptoms"}[cohort]
        ax_auc.text(0.015, 0.975, cohort_label, transform=ax_auc.transAxes,
                    fontsize=8.5, fontweight="bold", ha="left", va="top",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.2}, zorder=8)

        # Right: Youden index, pooled and within-patient, against within-patient AUC.
        xs, ys = curve(curves, cohort, "true_risk", "youden", "pooled")
        ax_j.plot(xs, ys, color=POOLED, lw=2.0)
        xs, ys = curve(curves, cohort, "true_risk", "youden", "within")
        ax_j.plot(xs, ys, color=WITHIN, lw=2.0)
        for r in obs:
            x, yp, yw = num(r["within_auc"]), num(r["pooled_youden"]), num(r["within_youden"])
            if x is None:
                continue
            if r["forecast"] in MODEL_MARKERS:
                m = MODEL_MARKERS[r["forecast"]]
                if yp is not None and yw is not None:
                    ax_j.plot([x, x], [yw, yp], color=GREY, lw=0.6, zorder=3)
                if yp is not None:
                    ax_j.scatter([x], [yp], s=26, marker=m, color=POOLED, edgecolors="black", linewidths=0.5, zorder=5)
                if yw is not None:
                    ax_j.scatter([x], [yw], s=26, marker=m, color=WITHIN, edgecolors="black", linewidths=0.5, zorder=5)
            elif r["forecast"] in RATE_MARKERS and yp is not None:
                ax_j.scatter([x], [yp], s=26, marker=RATE_MARKERS[r["forecast"]], facecolors="white",
                             edgecolors=POOLED, linewidths=1.0, zorder=5, clip_on=False)
        ax_j.scatter([0.5], [0.0], s=26, marker="o", facecolors="white", edgecolors=WITHIN, linewidths=1.0,
                     zorder=5, clip_on=False)
        ax_j.set_ylim(-0.05, 0.85)
        ax_j.set_ylabel("Youden index", fontsize=9.5)

        for ax in (ax_auc, ax_j):
            ax.set_xlim(0.5, 0.9)
            ax.tick_params(labelsize=7, width=0.6, length=2.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel("Within-patient AUC of the forecast", fontsize=9.5)

    # A concise callout identifies the CTN-0094 gradient-boosting model result.
    axes[1, 0].annotate("Model: pooled 0.93\nper patient 0.64", xy=(0.638, 0.929),
                        xytext=(0.69, 0.84), textcoords="data", fontsize=8.5,
                        ha="left", va="top", color="#222222",
                        arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.8},
                        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white",
                              "edgecolor": "#AAAAAA", "linewidth": 0.6}, zorder=10)

    handles = [
        Line2D([], [], color=POOLED, lw=2.0, label="Pooled value"),
        Line2D([], [], color=WITHIN, lw=2.0, label="Per-patient value"),
        Line2D([], [], marker="o", ls="", color=POOLED, markeredgecolor="black", markeredgewidth=0.4,
               markersize=5.5, label="Observed model"),
        Line2D([], [], marker="D", ls="", markerfacecolor="white", markeredgecolor=POOLED,
               markersize=5, label="Observed patient rate"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               handlelength=1.8, columnspacing=1.4, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.975, bottom=0.115, wspace=0.30, hspace=0.28)
    out = Path(__file__)
    fig.savefig(out.with_name("fig_central_v6.png"), dpi=300)
    fig.savefig(out.with_name("fig_central_v6.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
