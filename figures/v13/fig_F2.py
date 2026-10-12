"""F2: finite-record bias and the flipped rate-so-far forecast."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from style import COLORS, TWO_COLUMNS, dot_ring_gap, mpl_or_report, read_json, save_figure, write_csv


def auc_value(y, score):
    sys.path.insert(0, str(ROOT / "code/wheneval/src"))
    from wheneval import auc
    return float(auc(y, score))


def data_rows():
    event_days = {2, 5, 9}
    outcome = [int(day in event_days) for day in range(1, 11)]
    total_events = sum(outcome)
    rows = []
    rate_so_far = []
    other_nine = []
    for idx, y in enumerate(outcome):
        n_before = idx
        rate = (sum(outcome[:idx]) / n_before) if n_before else ""
        rate_so_far.append(float(rate) if rate != "" else float("nan"))
        loo = (total_events - y) / 9
        other_nine.append(loo)
        rows.append({"panel": "c", "day": idx + 1, "outcome": y,
                     "rate_so_far": rate, "other_nine_rate": loo,
                     "days_scored": "", "auc_flipped_mean": "",
                     "auc_flipped_mcse": "", "replications": ""})

    scored = [i for i in range(1, 10)]
    c_auc = auc_value([outcome[i] for i in scored],
                      [rate_so_far[i] for i in scored])
    event_mean = sum(other_nine[i] for i, y in enumerate(outcome) if y) / total_events
    free_mean = sum(other_nine[i] for i, y in enumerate(outcome) if not y) / (10 - total_events)
    for quantity, value in (("event_days", total_events), ("event_free_days", 10-total_events),
                            ("other_nine_event_day_mean", event_mean),
                            ("other_nine_event_free_day_mean", free_mean),
                            ("other_nine_auc", auc_value(outcome, other_nine)),
                            ("rate_so_far_auc", c_auc),
                            ("clara_record_days", 10), ("event_day_1_unscored", 1),
                            ("auc_baseline", 0.5), ("continuing_record_daily_event_probability", 0.30)):
        rows.append({"panel": "summary", "day": "", "outcome": "", "rate_so_far": "",
                     "other_nine_rate": "", "days_scored": "", "auc_flipped_mean": "",
                     "auc_flipped_mcse": "", "replications": "",
                     "quantity": quantity, "value": value})

    a2_path = ROOT / "results/v13/A2b_flip_curve.csv"
    if a2_path.exists():
        with a2_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                rows.append({"panel": "d", **row})
    else:
        rows.append({"panel": "d_status", "quantity": "A2b_flip_curve.csv not present yet"})
    return outcome, rate_so_far, other_nine, rows, {
        "event_mean": event_mean, "free_mean": free_mean, "other_auc": auc_value(outcome, other_nine),
        "rate_auc": c_auc, "a2_path": a2_path,
    }


def main():
    outcome, rate, loo, rows, s = data_rows()
    write_csv(ROOT / "results/v13/F2_numbers.csv",
              ["panel", "day", "outcome", "rate_so_far", "other_nine_rate", "days_scored",
               "auc_flipped_mean", "auc_flipped_mcse", "replications", "quantity", "value"], rows)
    plt = mpl_or_report()
    if plt is None:
        return

    fig = plt.figure(figsize=(TWO_COLUMNS, 130 / 25.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08], width_ratios=[1, 1],
                          hspace=0.48, wspace=0.32)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.13, top=0.90)
    fig.suptitle("Finite-record bias and rate-so-far forecasts", x=0.01, ha="left",
                 fontsize=9.5, fontweight="bold")
    ax_windows = fig.add_subplot(gs[0, 0])
    b_axes = gs[0, 1].subgridspec(2, 1, height_ratios=[1.25, 0.85], hspace=0.30)
    ax_other = fig.add_subplot(b_axes[0])
    ax_other_auc = fig.add_subplot(b_axes[1])
    ax_rate = fig.add_subplot(gs[1, 0])
    ax_flip = fig.add_subplot(gs[1, 1])

    ax_windows.set_title("a  Earlier days versus the whole record", loc="left", fontweight="bold", pad=4)
    ax_windows.set_xlim(0, 12)
    ax_windows.set_ylim(-0.12, 2.15)
    ax_windows.axis("off")
    scored_x = 7.7
    ax_windows.text(0.1, 1.45, "Real forecast", ha="left", va="center", fontweight="bold")
    arrow_start_x = 4.8
    ax_windows.annotate("", xy=(scored_x, 1.45), xytext=(arrow_start_x, 1.45),
                        arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": COLORS["muted"]})
    ax_windows.text(6.25, 1.64, "days before", ha="center", fontsize=8.0, color=COLORS["muted"])
    ax_windows.scatter([scored_x], [1.45], s=31, marker="o", facecolors=COLORS["paper"],
                       edgecolors=COLORS["ink"], linewidths=0.9, zorder=3)
    ax_windows.text(scored_x, 1.18, "scored day", ha="center", fontsize=8.0, color=COLORS["muted"])
    ax_windows.text(10.2, 1.45, "later days\nnot used", ha="center", va="center",
                    fontsize=8.0, color=COLORS["better"])
    ax_windows.text(0.1, 0.55, "Whole-record rate", ha="left", va="center", fontweight="bold")
    ax_windows.annotate("", xy=(10.8, 0.55), xytext=(arrow_start_x, 0.55),
                        arrowprops={"arrowstyle": "<->", "lw": 0.9, "color": COLORS["muted"]})
    ax_windows.scatter([scored_x], [0.55], s=31, marker="o", facecolors=COLORS["ink"],
                       edgecolors=COLORS["ink"], linewidths=0.9, zorder=3)
    ax_windows.text(6.25, 0.77, "days before", ha="center", fontsize=8.0, color=COLORS["muted"])
    ax_windows.text(9.2, 0.77, "days after", ha="center", fontsize=8.0, color=COLORS["muted"])
    ax_windows.text(scored_x, 0.27, "scored day included", ha="center",
                    fontsize=8.0, color=COLORS["worse"])

    ax_other.set_title("b  Clara: forecast from the other nine days", loc="left", fontweight="bold", pad=4)
    ax_other.set_xlim(0, 0.5)
    ax_other.set_ylim(-0.5, 1.5)
    for yy, label, val, color in ((1, "Event days", s["event_mean"], COLORS["worse"]),
                                  (0, "Event-free days", s["free_mean"], COLORS["pooled"])):
        ax_other.scatter([val], [yy], marker="o", s=29, color=color, zorder=3)
        ax_other.text(val + 0.018, yy, f"{val:.0%}", va="center", fontsize=8, color=color)
    ax_other.axvline(0.5, color=COLORS["grid"], lw=0.7, zorder=0)
    ax_other.set_yticks([1, 0], ["Event days", "Event-free days"])
    ax_other.set_xticks([0, 0.25, 0.5], ["0%", "25%", "50%"])
    ax_other.set_xlabel("Mean forecast", labelpad=1)
    ax_other.tick_params(axis="y", length=0, labelsize=8.0)
    ax_other.spines[["top", "right", "left"]].set_visible(False)
    ax_other.spines["bottom"].set_color(COLORS["grid"])
    ax_other.grid(axis="x", color=COLORS["grid"], lw=0.4)
    ax_other.set_axisbelow(True)
    ax_other_auc.set_xlim(0, 1)
    ax_other_auc.set_ylim(-0.4, 0.4)
    dot_ring_gap(ax_other_auc, 0, s["other_auc"], 0.5, score_color=COLORS["pooled"],
                 marker_size=4.2, line_width=1.5)
    ax_other_auc.set_yticks([])
    ax_other_auc.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax_other_auc.set_xlabel("AUC score 0 versus baseline 0.5", labelpad=1)
    ax_other_auc.spines[["top", "right", "left"]].set_visible(False)
    ax_other_auc.spines["bottom"].set_color(COLORS["grid"])
    ax_other_auc.grid(axis="x", color=COLORS["grid"], lw=0.4)
    ax_other_auc.set_axisbelow(True)

    ax_rate.set_title("c  Clara: rate so far", loc="left", fontweight="bold", pad=4)
    days = list(range(1, 11))
    valid_x = [d for d, r in zip(days, rate) if r == r]
    valid_y = [r for r in rate if r == r]
    ax_rate.plot(valid_x, valid_y, color=COLORS["pooled"], lw=0.9, zorder=1)
    for d, r, y in zip(days, rate, outcome):
        if r == r:
            ax_rate.scatter([d], [r], s=25, marker="o",
                            facecolors=COLORS["pooled"] if y else COLORS["paper"],
                            edgecolors=COLORS["pooled"], linewidths=0.9, zorder=3)
    for event_day, next_day in ((2, 3), (5, 6), (9, 10)):
        ax_rate.annotate("", xy=(next_day, rate[next_day - 1]),
                         xytext=(event_day, rate[event_day - 1]),
                         arrowprops={"arrowstyle": "->", "lw": 0.9,
                                     "color": COLORS["worse"], "shrinkA": 4, "shrinkB": 4},
                         zorder=2)
    ax_rate.set_xlim(0.5, 10.5)
    ax_rate.set_ylim(-0.06, 0.62)
    ax_rate.set_xticks(days)
    ax_rate.set_yticks([0, 0.25, 0.5], ["0%", "25%", "50%"])
    ax_rate.set_xlabel("Day")
    ax_rate.set_ylabel("Rate so far")
    ax_rate.text(0.98, 0.97, "AUC 0", transform=ax_rate.transAxes,
                 ha="right", va="top", fontsize=8.2, fontweight="bold", color=COLORS["worse"])
    ax_rate.grid(axis="y", color=COLORS["grid"], lw=0.5)
    ax_rate.set_axisbelow(True)
    ax_rate.spines[["top", "right"]].set_visible(False)
    ax_rate.spines["left"].set_color(COLORS["grid"])
    ax_rate.spines["bottom"].set_color(COLORS["grid"])

    ax_flip.set_title("d  Forecast AUC against days scored", loc="left", fontweight="bold", pad=4)
    if s["a2_path"].exists():
        with s["a2_path"].open(newline="", encoding="utf-8-sig") as handle:
            a2 = list(csv.DictReader(handle))
        a2 = sorted(a2, key=lambda r: float(r["days_scored"]))
        days_scored = [float(r["days_scored"]) for r in a2]
        xs = list(range(len(a2)))
        flipped = [float(r["auc_flipped_mean"]) for r in a2]
        rate_mirror = [float(r["auc_rate_so_far_mean"]) for r in a2]
        errors = [float(r["auc_flipped_mcse"]) for r in a2]
        ax_flip.errorbar(xs, flipped, yerr=errors, color=COLORS["pooled"], lw=0.9,
                         marker="o", markersize=3.0, capsize=1.5, label="Flipped forecast")
        ax_flip.plot(xs, rate_mirror, color=COLORS["per_patient"], lw=1.0,
                     marker="s", markersize=2.8, label="Rate so far")
        ax_flip.axhline(0.5, color=COLORS["ink"], lw=0.8, ls=(0, (3, 2)), zorder=0)
        ax_flip.set_xticks(xs, [f"{int(d):,}" for d in days_scored])
        ax_flip.set_xlim(-0.25, len(xs) - 0.75)
        ax_flip.set_ylim(0.25, 0.75)
        ax_flip.set_yticks([0.25, 0.5, 0.75])
        ax_flip.grid(axis="y", color=COLORS["grid"], lw=0.4)
        ax_flip.legend(frameon=False, fontsize=8.0, loc="upper right", ncol=1,
                       borderaxespad=0.15, handlelength=1.3)
        ax_flip.spines[["top", "right"]].set_visible(False)
    else:
        ax_flip.text(0.5, 0.5, "Waiting for flip-curve data", ha="center", va="center",
                     transform=ax_flip.transAxes, fontsize=8, color=COLORS["muted"])
        ax_flip.set_xticks([])
        ax_flip.set_yticks([])
    ax_flip.set_xlabel("Days scored", labelpad=2)
    ax_flip.set_ylabel("AUC", labelpad=2)
    save_figure(fig, 2)
    plt.close(fig)


if __name__ == "__main__":
    main()
