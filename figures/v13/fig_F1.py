"""F1: who versus when, with the exact Ben and Anna pair decomposition."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from style import COLORS, TWO_COLUMNS, dot_ring_gap, mpl_or_report, save_figure, write_csv

GRID_SEED = 20261013


def data_rows():
    patients, days = 8, 21
    rng = np.random.default_rng(GRID_SEED)
    patient_rates = rng.uniform(0.10, 0.50, size=patients)
    grid = rng.binomial(1, patient_rates[:, None], size=(patients, days)).astype(int)
    events = int(grid.sum())
    non_events = patients * days - events
    within = int(sum(int(row.sum()) * (days - int(row.sum())) for row in grid))
    same_day = int(sum(int(grid[:, d].sum()) * (patients - int(grid[:, d].sum()))
                       for d in range(days)))
    total = events * non_events
    other = total - within - same_day

    rows = []
    for p, row in enumerate(grid, start=1):
        for d, outcome in enumerate(row, start=1):
            rows.append({"panel": "a", "record_type": "grid_day", "patient": p,
                         "day": d, "outcome": int(outcome), "quantity": "",
                         "value": "", "seed": GRID_SEED})
        rows.append({"panel": "a", "record_type": "patient_rate", "patient": p,
                     "day": "", "outcome": "", "quantity": "whole-record rate",
                     "value": float(patient_rates[p - 1]), "seed": GRID_SEED})
    for name, value in (("patients", patients), ("days_per_patient", days),
                        ("events", events), ("event_free_days", non_events),
                        ("pooled_event_non_event_pairs", total),
                        ("same_patient_pairs", within),
                        ("same_day_different_patients_pairs", same_day),
                        ("other_pairs", other),
                        ("same_patient_pair_share", within / total),
                        ("same_day_different_patients_pair_share", same_day / total),
                        ("other_pair_share", other / total),
                        ("seed", GRID_SEED)):
        rows.append({"panel": "a", "record_type": "summary", "patient": "",
                     "day": "", "outcome": "", "quantity": name,
                     "value": value, "seed": GRID_SEED})

    ben_events = {1, 3, 4, 7, 9}
    anna_events = {4}
    ben_rate, anna_rate = len(ben_events) / 10, len(anna_events) / 10
    ben_within = len(ben_events) * (10 - len(ben_events))
    anna_within = len(anna_events) * (10 - len(anna_events))
    between_right = len(ben_events) * (10 - len(anna_events))
    between_wrong = len(anna_events) * (10 - len(ben_events))
    between_pairs = between_right + between_wrong
    pairs = between_right + between_wrong + ben_within + anna_within
    pooled_auc_numerator = between_right + 0.5 * (ben_within + anna_within)
    pooled_auc = pooled_auc_numerator / pairs
    summary = {
        "Ben_events": len(ben_events), "Anna_events": len(anna_events),
        "days_each": 10, "Ben_whole_record_rate": ben_rate,
        "Anna_whole_record_rate": anna_rate, "between_patient_right": between_right,
        "between_patient_wrong": between_wrong, "between_patient_pairs": between_pairs,
        "within_Ben_ties": ben_within,
        "within_Anna_ties": anna_within, "total_pairs": pairs,
        "pooled_auc_numerator": pooled_auc_numerator, "pooled_auc": pooled_auc,
        "per_patient_auc_Ben": 0.5,
        "per_patient_auc_Anna": 0.5,
    }
    for quantity, value in summary.items():
        rows.append({"panel": "b", "record_type": "summary", "patient": "",
                     "day": "", "outcome": "", "quantity": quantity,
                     "value": value, "seed": GRID_SEED})
    for patient, event_days, rate in (("Ben", ben_events, ben_rate),
                                      ("Anna", anna_events, anna_rate)):
        for day in range(1, 11):
            rows.append({"panel": "b", "record_type": "patient_day", "patient": patient,
                         "day": day, "outcome": int(day in event_days),
                         "quantity": "whole-record rate", "value": rate,
                         "seed": GRID_SEED})
    return grid, patient_rates, rows, {
        "events": events, "non_events": non_events, "within": within,
        "same_day": same_day, "other": other, "pairs": total,
        "between_right": between_right, "between_wrong": between_wrong,
        "ben_within": ben_within, "anna_within": anna_within,
        "pooled_auc": pooled_auc, "ben_rate": ben_rate, "anna_rate": anna_rate,
    }


def main():
    grid, patient_rates, rows, s = data_rows()
    write_csv(ROOT / "results/v13/F1_numbers.csv",
              ["panel", "record_type", "patient", "day", "outcome", "quantity", "value", "seed"], rows)
    plt = mpl_or_report()
    if plt is None:
        return
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(TWO_COLUMNS, 126 / 25.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.0], width_ratios=[1.1, 1.0],
                          hspace=0.38, wspace=0.22)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.09, top=0.93)
    ax_grid = fig.add_subplot(gs[0, :])
    ax_pairs = fig.add_subplot(gs[1, 0])
    right = gs[1, 1].subgridspec(3, 1, height_ratios=[2.0, 1.0, 0.9], hspace=0.42)
    ax_example = fig.add_subplot(right[0])
    ax_pair_counts = fig.add_subplot(right[1])
    ax_auc = fig.add_subplot(right[2])

    ax_grid.set_title("a  Patients are rows; days are columns", loc="left", fontweight="bold", pad=5)
    for p, row in enumerate(grid):
        for d, outcome in enumerate(row):
            if outcome:
                ax_grid.scatter(d, p, s=13, marker="o", color=COLORS["ink"], zorder=3)
            else:
                ax_grid.scatter(d, p, s=13, marker="o", facecolors=COLORS["paper"],
                                edgecolors="#C7CCCF", linewidths=0.45, zorder=2)
    ax_grid.add_patch(Rectangle((-0.42, 2 - 0.42), 21 - 0.16, 0.84, fill=False,
                                edgecolor=COLORS["pooled"], linewidth=1.15))
    ax_grid.add_patch(Rectangle((9 - 0.42, -0.42), 0.84, 8 - 0.16, fill=False,
                                edgecolor=COLORS["worse"], linewidth=1.15))
    ax_grid.set_xlim(-0.7, 26.0)
    ax_grid.set_ylim(8.4, -0.55)
    ax_grid.set_xticks([0, 4, 9, 14, 20], [1, 5, 10, 15, 21])
    ax_grid.set_yticks(range(8), [f"P{i}" for i in range(1, 9)])
    ax_grid.set_xlabel("Day")
    ax_grid.set_ylabel("Patient")
    ax_grid.grid(axis="x", color=COLORS["grid"], lw=0.45)
    ax_grid.tick_params(length=0)
    ax_grid.spines[["top", "right", "left"]].set_visible(False)
    ax_grid.spines["bottom"].set_color(COLORS["grid"])
    ax_grid.text(21.15, 2, "Outlined row: when", ha="left", va="center",
                 fontsize=8.0, color=COLORS["pooled"], clip_on=False)
    ax_grid.text(9, 7.85, "Outlined column: who", ha="center", va="top",
                 fontsize=8.0, color=COLORS["worse"])
    ax_grid.text(0.99, 0.98, f"Random fixed grid · seed {GRID_SEED}",
                 transform=ax_grid.transAxes, ha="right", va="top",
                 fontsize=8.0, color=COLORS["muted"])

    ax_pairs.set_title("Pooled event–event-free pairs", loc="left", fontweight="bold", pad=5)
    segments = [
        ("Same patient", s["within"], COLORS["pooled"]),
        ("Same day, different patients", s["same_day"], COLORS["per_patient"]),
        ("Other pairs", s["other"], "#B7BEC2"),
    ]
    bar_y, left = 0.0, 0
    for label, value, color in segments:
        ax_pairs.barh(bar_y, value, left=left, height=0.34, color=color,
                      edgecolor=COLORS["paper"], linewidth=0.6, zorder=2)
        left += value
    for i, (label, value, color) in enumerate(segments):
        y = 2.05 - i * 0.49
        ax_pairs.scatter([s["pairs"] * 0.025], [y], marker="s", s=24,
                         color=color, edgecolors="none", zorder=3)
        ax_pairs.text(s["pairs"] * 0.055, y,
                      f"{value:,}  ({value / s['pairs']:.1%})  {label}",
                      ha="left", va="center", fontsize=8.0, color=COLORS["ink"])
    ax_pairs.set_xlim(0, max(5000, s["pairs"]))
    ax_pairs.set_ylim(-0.65, 2.55)
    ticks = [0, 1000, 2000, 3000, 4000, 5000]
    ax_pairs.set_xticks(ticks, [f"{v:,}" for v in ticks])
    ax_pairs.set_yticks([])
    ax_pairs.set_xlabel("Number of event–event-free pairs", labelpad=3)
    ax_pairs.spines[["top", "right", "left"]].set_visible(False)
    ax_pairs.spines["bottom"].set_color(COLORS["grid"])
    ax_pairs.grid(axis="x", color=COLORS["grid"], lw=0.45, zorder=0)
    ax_pairs.set_axisbelow(True)

    ax_example.set_title("b  Whole-record rate: pooled vs per-patient",
                         loc="left", fontweight="bold", pad=4)
    ax_example.set_xlim(0, 10)
    ax_example.set_ylim(0.0, 2.25)
    for y, name, events, rate, color in (
        (1.55, "Ben", {1, 3, 4, 7, 9}, s["ben_rate"], COLORS["pooled"]),
        (0.7, "Anna", {4}, s["anna_rate"], COLORS["per_patient"]),
    ):
        for day in range(1, 11):
            filled = day in events
            ax_example.scatter(day - 0.5, y, s=22, marker="o",
                               facecolors=COLORS["ink"] if filled else COLORS["paper"],
                               edgecolors=COLORS["ink"] if filled else "#ABB2B6",
                               linewidths=0.75, zorder=2)
    ax_example.set_yticks([1.55, 0.7], ["Ben · 50%", "Anna · 10%"])
    ax_example.set_xticks(np.arange(0.5, 10, 1), [str(d) for d in range(1, 11)])
    ax_example.set_xlabel("Day (filled = event)", labelpad=1)
    ax_example.grid(axis="x", color=COLORS["grid"], lw=0.4)
    ax_example.tick_params(axis="both", length=0, labelsize=8.0)
    ax_example.spines[["top", "right", "left"]].set_visible(False)
    ax_example.spines["bottom"].set_color(COLORS["grid"])
    ax_example.set_axisbelow(True)

    ax_pair_counts.axis("off")
    ax_pair_counts.text(0.02, 0.15,
                        "Between-patient pairs (50): 45 wins →, 5 losses ←\nWithin-patient ties: 34",
                        transform=ax_pair_counts.transAxes, ha="left", va="center",
                        fontsize=8.0, color=COLORS["muted"])
    ax_auc.set_xlim(0, 1)
    ax_auc.set_ylim(-0.5, 1.5)
    dot_ring_gap(ax_auc, 1, s["pooled_auc"], 0.5, score_color=COLORS["pooled"],
                 marker_size=4.2, line_width=1.5)
    dot_ring_gap(ax_auc, 0, 0.5, 0.5, score_color=COLORS["per_patient"],
                 marker_size=4.2, line_width=1.5, marker_offset_points=1.5)
    ax_auc.set_yticks([1, 0], ["Pooled AUC 0.74", "Per-patient AUC 0.50"])
    ax_auc.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax_auc.set_xlabel("AUC", labelpad=0)
    ax_auc.tick_params(axis="y", length=0, labelsize=8.0, pad=2)
    ax_auc.tick_params(axis="x", length=2, labelsize=8.0, pad=1)
    ax_auc.spines[["top", "right", "left"]].set_visible(False)
    ax_auc.spines["bottom"].set_color(COLORS["grid"])
    ax_auc.grid(axis="x", color=COLORS["grid"], lw=0.4)
    ax_auc.set_axisbelow(True)
    save_figure(fig, 1)
    plt.close(fig)


if __name__ == "__main__":
    main()
