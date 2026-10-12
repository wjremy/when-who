"""F8: ranking the closer pair under pooled, per-patient, and shuffled-baseline scoring."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from style import (COLORS, SCORING_ORDER, TWO_COLUMNS, mpl_or_report,
                   read_json, save_figure, write_csv)

SOURCE = ROOT / "open_data/results/v10/ranking_sim.json"
PAIR = "A daily rho=0.4 vs B daily rho=0.3"
SCORINGS = {
    "pooled": ("pooled", "Pooled"),
    "naive": ("per_patient", "Per-patient"),
    "tailored_excess": ("shuffled_baseline", "Shuffled-baseline"),
}


def collect_data():
    data = read_json(SOURCE)
    settings = data["run"]
    selected = [r for r in data["ranking"] if r["pair"] == PAIR and r["metric"] == "auc"
                and r["scoring"] in SCORINGS]
    rows = []
    for r in selected:
        scoring, label = SCORINGS[r["scoring"]]
        rows.append({
            "cohort": r["cohort"], "patient_spread": r["patient_spread"],
            "record_days": r["record_days"], "patients": r["patients"],
            "pair": r["pair"], "metric": r["metric"], "scoring": scoring,
            "scoring_label": label, "replications": r["valid_replications"],
            "share_a_ranks_above_b": r["share_a_ranks_above_b"],
            "equal_ranking_reference": 0.5,
            "mean_a_score": r.get("mean_a_score", ""), "mean_b_score": r.get("mean_b_score", ""),
            "seed": settings["seed"],
            "chance_shuffles_per_replication": settings["chance_shuffles_per_replication"],
        })
    return settings, rows


def main():
    settings, rows = collect_data()
    write_csv(ROOT / "results/v13/F8_numbers.csv",
              ["cohort", "patient_spread", "record_days", "patients", "pair", "metric",
               "scoring", "scoring_label", "replications", "share_a_ranks_above_b",
               "equal_ranking_reference",
               "mean_a_score", "mean_b_score", "seed", "chance_shuffles_per_replication"], rows)
    plt = mpl_or_report()
    if plt is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(TWO_COLUMNS, 112 / 25.4), sharey=True)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.88, bottom=0.22,
                        wspace=0.32, hspace=0.47)
    fig.suptitle("Which scoring ranks A above B? · closer pair: daily ρ = 0.4 versus 0.3",
                 x=0.01, ha="left", fontsize=9.5, fontweight="bold")
    panels = [
        ("low", 50, "Low spread · REPLACE-BG · 50 patients"),
        ("low", 150, "Low spread · REPLACE-BG · 150 patients"),
        ("high", 50, "High spread · CTN-0094 · 50 patients"),
        ("high", 150, "High spread · CTN-0094 · 150 patients"),
    ]
    color_map = {"pooled": COLORS["pooled"], "per_patient": COLORS["per_patient"],
                 "shuffled_baseline": COLORS["shuffled_baseline"]}
    category_positions = {30: 0, 100: 1, 365: 2}
    for ax, (spread, n, title) in zip(axes.flat, panels):
        subset = [r for r in rows if r["patient_spread"] == spread and int(r["patients"]) == n]
        for method in SCORING_ORDER:
            values = sorted((r for r in subset if r["scoring"] == method),
                            key=lambda r: int(r["record_days"]))
            if values:
                xs = [category_positions[int(r["record_days"])] for r in values]
                ys = [float(r["share_a_ranks_above_b"]) for r in values]
                ax.scatter(xs, ys, color=color_map[method], marker="o", s=22,
                           label={"pooled": "Pooled", "per_patient": "Per-patient",
                                  "shuffled_baseline": "Shuffled-baseline"}[method], zorder=3)
        ax.axhline(0.5, color=COLORS["muted"], lw=0.65, ls=(0, (2, 2)), zorder=0)
        ax.set_xticks([0, 1, 2], ["30", "100", "365"])
        ax.set_xlim(-0.35, 2.35)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(title, loc="left", fontsize=8.0, fontweight="bold", pad=4)
        ax.set_xlabel("Record length (days)")
        ax.set_ylabel("Share ranking A first")
        ax.grid(axis="y", color=COLORS["grid"], lw=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_color(COLORS["grid"])
        ax.spines["bottom"].set_color(COLORS["grid"])
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", ls="", color=color_map["pooled"], label="Pooled"),
        Line2D([], [], marker="o", ls="", color=color_map["per_patient"], label="Per-patient"),
        Line2D([], [], marker="o", ls="", color=color_map["shuffled_baseline"], label="Shuffled-baseline"),
        Line2D([], [], color=COLORS["muted"], lw=0.8, ls=(0, (2, 2)), label="Equal ranking (0.5)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.06), handlelength=1.5, columnspacing=1.2)
    save_figure(fig, 8)
    plt.close(fig)


if __name__ == "__main__":
    main()
