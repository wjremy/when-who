"""Figure 7: ranking the closer pair under pooled, per-patient, and shuffled-baseline scoring."""
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
def collect_data(patients: int):
    data = read_json(SOURCE)
    settings = data["run"]
    selected = [
        r for r in data["ranking"]
        if r["pair"] == PAIR and r["metric"] == "auc"
        and r["scoring"] in SCORINGS and int(r["patients"]) == patients
    ]
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
            "mean_a_score": r.get("mean_a_score", ""),
            "mean_b_score": r.get("mean_b_score", ""),
            "seed": settings["seed"],
            "chance_shuffles_per_replication": settings["chance_shuffles_per_replication"],
        })
    return rows


def plot_ranking_figure(rows, number: int, name: str | None = None):
    plt = mpl_or_report()
    if plt is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(TWO_COLUMNS, 79 / 25.4), sharey=True)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.88, bottom=0.27, wspace=0.20)
    panels = [
        ("low", "a  Low spread (REPLACE-BG)"),
        ("high", "b  High spread (CTN-0094)"),
    ]
    color_map = {"pooled": COLORS["pooled"], "per_patient": COLORS["per_patient"],
                 "shuffled_baseline": COLORS["shuffled_baseline"]}
    x_positions = {30: 0, 100: 1, 365: 2}
    x_offsets = {"pooled": -0.24, "per_patient": 0.0, "shuffled_baseline": 0.24}
    bar_width = 0.22
    hatches = {"pooled": None, "per_patient": "////", "shuffled_baseline": "...."}  # grayscale-safe (rubric r07)

    for ax, (spread, panel_label) in zip(axes, panels):
        subset = [r for r in rows if r["patient_spread"] == spread]
        for days, group_x in x_positions.items():
            for method in SCORING_ORDER:
                matching = [r for r in subset
                            if int(r["record_days"]) == days and r["scoring"] == method]
                if not matching:
                    continue
                value = float(matching[0]["share_a_ranks_above_b"])
                x = group_x + x_offsets[method]
                ax.bar(x, value, width=bar_width, color=color_map[method], hatch=hatches[method],
                       edgecolor=COLORS["paper"], linewidth=0.35, zorder=3)
                if value == 0:
                    # Mark zero-height bars with a colored tick on the baseline.
                    ax.plot([x - bar_width * 0.42, x + bar_width * 0.42], [0, 0],
                            color=color_map[method], linewidth=1.8,
                            solid_capstyle="butt", zorder=5, clip_on=False)

        ax.axhline(0.5, color=COLORS["muted"], lw=0.65, ls=(0, (2, 2)), zorder=4)
        ax.set_xticks([0, 1, 2], ["30", "100", "365"])
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0, 1)
        ax.set_title(panel_label, loc="left", fontsize=8.0, fontweight="bold", pad=5)
        ax.set_xlabel("Record length (days)")
        ax.set_ylabel("Share of replications ranking A above B" if spread == "low" else "")
        ax.grid(axis="y", color=COLORS["grid"], lw=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(COLORS["grid"])
        ax.spines["bottom"].set_color(COLORS["grid"])

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=color_map["pooled"], edgecolor=COLORS["paper"], label="Pooled"),
        Patch(facecolor=color_map["per_patient"], edgecolor=COLORS["paper"], hatch=hatches["per_patient"],
              label="Per-patient"),
        Patch(facecolor=color_map["shuffled_baseline"], hatch=hatches["shuffled_baseline"],
              edgecolor=COLORS["paper"], label="Shuffled-baseline"),
        Line2D([], [], color=COLORS["muted"], lw=0.8, ls=(0, (2, 2)),
               label="Equal ranking (0.5)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.045), handlelength=1.5, columnspacing=1.2)
    save_figure(fig, number, name=name)
    plt.close(fig)


def main():
    patients = 50
    rows = collect_data(patients)
    write_csv(ROOT / "results/v19/F7_numbers.csv",
              ["cohort", "patient_spread", "record_days", "patients", "pair", "metric",
               "scoring", "scoring_label", "replications", "share_a_ranks_above_b",
               "equal_ranking_reference", "mean_a_score", "mean_b_score", "seed",
               "chance_shuffles_per_replication"], rows)
    plot_ranking_figure(rows, 7)


if __name__ == "__main__":
    main()
