"""F5: one simulated patient and the shuffled baseline."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from style import COLORS, TWO_COLUMNS, dot_ring_gap, mpl_or_report, save_figure, write_csv

SEED = 20261012
RECORD_DAYS = 100
EVENT_PROBABILITY = 0.30
SHUFFLES = 200


def patient_numbers():
    sys.path.insert(0, str(ROOT / "code/wheneval/src"))
    from wheneval import auc

    rng = np.random.default_rng(SEED)
    y = rng.binomial(1, EVENT_PROBABILITY, size=RECORD_DAYS).astype(int)

    def history_rate(outcomes):
        return np.r_[np.nan, np.cumsum(outcomes[:-1]) / np.arange(1, len(outcomes))]

    observed_rates = history_rate(y)
    observed_auc = float(auc(y[1:], observed_rates[1:]))
    shuffle_results = []
    shuffled_records = []
    for shuffle_id in range(1, SHUFFLES + 1):
        permuted = rng.permutation(y)
        scores = history_rate(permuted)
        shuffled_auc = float(auc(permuted[1:], scores[1:]))
        shuffle_results.append(shuffled_auc)
        if shuffle_id <= 3:
            shuffled_records.append((permuted, scores))
    baseline = float(np.mean(shuffle_results))
    excess = observed_auc - baseline
    return y, observed_rates, shuffled_records, shuffle_results, {
        "seed": SEED, "record_days": RECORD_DAYS, "event_probability": EVENT_PROBABILITY,
        "event_days": int(y.sum()), "event_rate": float(y.mean()), "scored_days": RECORD_DAYS - 1,
        "shuffle_replications": SHUFFLES, "auc_observed": observed_auc,
        "shuffled_baseline_mean": baseline, "excess": excess,
    }


def main():
    y, rates, shuffled_records, shuffled_aucs, summary = patient_numbers()
    rows = []
    for day, (outcome, rate) in enumerate(zip(y, rates), start=1):
        rows.append({"row_type": "record_day", "seed": SEED, "record_days": RECORD_DAYS,
                     "event_probability": EVENT_PROBABILITY, "event_rate": summary["event_rate"],
                     "event_days": summary["event_days"], "day": day, "outcome": int(outcome),
                     "rate_so_far": "" if np.isnan(rate) else float(rate),
                     "shuffle_id": "", "auc": "", "score": "", "baseline": "", "excess": "",
                     "replications": SHUFFLES})
    for shuffle_id, (record, forecast) in enumerate(shuffled_records, start=1):
        for day, (outcome, rate) in enumerate(zip(record, forecast), start=1):
            rows.append({"row_type": "shuffle_day", "seed": SEED, "record_days": RECORD_DAYS,
                         "event_probability": EVENT_PROBABILITY, "event_rate": summary["event_rate"],
                         "event_days": summary["event_days"], "day": day, "outcome": int(outcome),
                         "rate_so_far": "" if np.isnan(rate) else float(rate),
                         "shuffle_id": shuffle_id, "auc": "", "score": "", "baseline": "",
                         "excess": "", "replications": SHUFFLES})
    for shuffle_id, auc_value in enumerate(shuffled_aucs, start=1):
        rows.append({"row_type": "shuffle_auc", "seed": SEED, "record_days": RECORD_DAYS,
                     "event_probability": EVENT_PROBABILITY, "event_rate": summary["event_rate"],
                     "event_days": summary["event_days"],
                     "day": "", "outcome": "", "rate_so_far": "", "shuffle_id": shuffle_id,
                     "auc": auc_value, "score": "", "baseline": "", "excess": "",
                     "replications": SHUFFLES})
    rows.append({"row_type": "summary", **summary, "day": "", "outcome": "",
                 "rate_so_far": "", "shuffle_id": "", "auc": "",
                 "score": summary["auc_observed"], "baseline": summary["shuffled_baseline_mean"],
                 "replications": SHUFFLES})
    fields = ["row_type", "seed", "record_days", "event_probability", "event_rate", "event_days", "day",
              "outcome", "rate_so_far", "shuffle_id", "auc", "score", "baseline", "excess", "replications"]
    write_csv(ROOT / "results/v13/F5_patient.csv", fields, rows)
    plt = mpl_or_report()
    if plt is None:
        return
    fig = plt.figure(figsize=(TWO_COLUMNS, 78 / 25.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.31)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.88, bottom=0.17)
    ax_record = fig.add_subplot(gs[0, 0])
    hist_gs = gs[0, 1].subgridspec(2, 1, height_ratios=[0.52, 1.0], hspace=0.12)
    ax_glyph = fig.add_subplot(hist_gs[0])
    ax_hist = fig.add_subplot(hist_gs[1], sharex=ax_glyph)
    record_versions = [(y, "Observed")] + [(s[0], f"Shuffle {i}") for i, s in enumerate(shuffled_records, 1)]
    for row_i, (record, label) in enumerate(record_versions):
        for d, outcome in enumerate(record, start=1):
            if outcome:
                ax_record.scatter(d, row_i, marker="s", s=10, color=COLORS["ink"], zorder=3)
            else:
                ax_record.scatter(d, row_i, marker="s", s=10, facecolors="white",
                                  edgecolors="#C7CCCF", linewidths=0.45, zorder=2)
    ax_record.set_xlim(0.5, RECORD_DAYS + 0.5)
    ax_record.set_ylim(3.6, -0.6)
    ax_record.set_xticks([1, 50, 100])
    ax_record.set_xlabel("Day (filled square = event)")
    ax_record.set_yticks(range(len(record_versions)), [label for _, label in record_versions])
    ax_record.tick_params(axis="y", length=0, pad=4)
    ax_record.set_title(f"One patient · {summary['event_days']} events in {RECORD_DAYS} days",
                        loc="left", fontsize=8, fontweight="bold", pad=4)
    ax_record.grid(axis="x", color=COLORS["grid"], lw=0.4)
    ax_record.set_axisbelow(True)
    ax_record.spines[["top", "right", "left"]].set_visible(False)

    ax_glyph.set_xlim(0, 1)
    ax_glyph.set_ylim(-0.35, 1.05)
    dot_ring_gap(ax_glyph, 0.05, summary["auc_observed"], summary["shuffled_baseline_mean"],
                 score_color=COLORS["pooled"], marker_size=5.0, line_width=1.7)
    ax_glyph.text(0.5, 0.94,
                  f"Observed {summary['auc_observed']:.3f}  ·  Baseline {summary['shuffled_baseline_mean']:.3f}\nExcess {summary['excess']:.3f}",
                  transform=ax_glyph.transAxes, ha="center", va="top", fontsize=8.0,
                  color=COLORS["worse"] if summary["excess"] < 0 else COLORS["better"])
    ax_glyph.set_title("Observed score, shuffled baseline, and excess", loc="left",
                        fontsize=8, fontweight="bold", pad=3)
    ax_glyph.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_glyph.set_yticks([])
    ax_glyph.spines[["top", "right", "left", "bottom"]].set_visible(False)

    bins = ax_hist.hist(shuffled_aucs, bins=18, range=(0, 1), color="#A9C7D7",
                        edgecolor="white", linewidth=0.55)
    max_bin = max(bins[0]) if len(bins[0]) else 1
    ax_hist.set_ylim(0, max_bin * 1.12)
    ax_hist.set_title("Shuffled AUC distribution", loc="left", fontsize=8, fontweight="bold", pad=3)
    ax_hist.set_xlim(0, 1)
    ax_hist.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
    ax_hist.set_xlabel("AUC")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(axis="y", color=COLORS["grid"], lw=0.5)
    ax_hist.set_axisbelow(True)
    ax_hist.spines[["top", "right"]].set_visible(False)
    save_figure(fig, 5)
    plt.close(fig)


if __name__ == "__main__":
    main()
