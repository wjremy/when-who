"""F4: one simulated patient and the shuffled baseline."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from style import COLORS, TWO_COLUMNS, mpl_or_report, save_figure, write_csv

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
    shuffled_aucs = []
    shuffled_records = []
    for shuffle_id in range(1, SHUFFLES + 1):
        permuted = rng.permutation(y)
        forecasts = history_rate(permuted)
        shuffled_auc = float(auc(permuted[1:], forecasts[1:]))
        shuffled_aucs.append(shuffled_auc)
        if shuffle_id <= 3:
            shuffled_records.append((shuffle_id, permuted, forecasts, shuffled_auc))

    baseline = float(np.mean(shuffled_aucs))
    excess = observed_auc - baseline
    summary = {
        "seed": SEED,
        "record_days": RECORD_DAYS,
        "event_probability": EVENT_PROBABILITY,
        "event_days": int(y.sum()),
        "event_rate": float(y.mean()),
        "scored_days": RECORD_DAYS - 1,
        "shuffle_replications": SHUFFLES,
        "auc_observed": observed_auc,
        "shuffled_baseline_mean": baseline,
        "excess": excess,
    }
    assert summary["event_days"] == 34, summary["event_days"]
    assert f"{observed_auc:.3f}" == "0.330", observed_auc
    assert f"{baseline:.3f}" == "0.421", baseline
    return y, observed_rates, shuffled_records, shuffled_aucs, summary


def write_numbers_csv(shuffled_aucs, summary):
    fields = [
        "row_type", "seed", "record_days", "event_probability", "event_rate",
        "event_days", "scored_days", "shuffle_replications", "shuffle_id", "auc",
        "auc_observed", "shuffled_baseline_mean", "excess",
    ]
    rows = []
    for shuffle_id, auc_value in enumerate(shuffled_aucs, start=1):
        rows.append({
            "row_type": "shuffle_auc", "seed": SEED, "record_days": RECORD_DAYS,
            "event_probability": EVENT_PROBABILITY, "event_rate": summary["event_rate"],
            "event_days": summary["event_days"], "scored_days": RECORD_DAYS - 1,
            "shuffle_replications": SHUFFLES, "shuffle_id": shuffle_id, "auc": auc_value,
        })
    rows.append({
        "row_type": "summary", "seed": SEED, "record_days": RECORD_DAYS,
        "event_probability": EVENT_PROBABILITY, "event_rate": summary["event_rate"],
        "event_days": summary["event_days"], "scored_days": RECORD_DAYS - 1,
        "shuffle_replications": SHUFFLES, "auc_observed": summary["auc_observed"],
        "shuffled_baseline_mean": summary["shuffled_baseline_mean"],
        "excess": summary["excess"],
    })
    write_csv(ROOT / "results/v19/F4_numbers.csv", fields, rows)


def write_patient_csv(y, rates, shuffled_records, summary):
    fields = [
        "row_type", "seed", "record_days", "event_probability", "event_rate",
        "event_days", "day", "outcome", "rate_so_far", "shuffle_id", "auc",
        "auc_observed", "shuffled_baseline_mean", "excess", "scored_days",
        "shuffle_replications",
    ]
    rows = []
    displayed_records = [(0, y, rates, summary["auc_observed"])] + shuffled_records
    for shuffle_id, record, forecasts, row_auc in displayed_records:
        row_type = "record_day" if shuffle_id == 0 else "shuffle_day"
        for day, (outcome, rate) in enumerate(zip(record, forecasts), start=1):
            rows.append({
                "row_type": row_type, "seed": SEED, "record_days": RECORD_DAYS,
                "event_probability": EVENT_PROBABILITY, "event_rate": summary["event_rate"],
                "event_days": summary["event_days"], "day": day, "outcome": int(outcome),
                "rate_so_far": "" if np.isnan(rate) else float(rate),
                "shuffle_id": "" if shuffle_id == 0 else shuffle_id, "auc": row_auc,
                "scored_days": RECORD_DAYS - 1, "shuffle_replications": SHUFFLES,
            })
    rows.append({
        "row_type": "summary", "seed": SEED, "record_days": RECORD_DAYS,
        "event_probability": EVENT_PROBABILITY, "event_rate": summary["event_rate"],
        "event_days": summary["event_days"], "auc_observed": summary["auc_observed"],
        "shuffled_baseline_mean": summary["shuffled_baseline_mean"],
        "excess": summary["excess"], "scored_days": RECORD_DAYS - 1,
        "shuffle_replications": SHUFFLES,
    })
    write_csv(ROOT / "results/v19/F4_patient.csv", fields, rows)


def main():
    y, observed_rates, shuffled_records, shuffled_aucs, summary = patient_numbers()
    write_numbers_csv(shuffled_aucs, summary)
    write_patient_csv(y, observed_rates, shuffled_records, summary)

    plt = mpl_or_report()
    if plt is None:
        return

    fig = plt.figure(figsize=(TWO_COLUMNS, 90 / 25.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.34)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.82, bottom=0.19)
    ax_record = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])

    record_versions = [(y, observed_rates, "Observed", summary["auc_observed"], 0)]
    record_versions += [
        (record, forecasts, f"Shuffle {shuffle_id}", row_auc, shuffle_id)
        for shuffle_id, record, forecasts, row_auc in shuffled_records
    ]
    for row_i, (record, forecasts, _, row_auc, shuffle_id) in enumerate(record_versions):
        days = np.arange(1, RECORD_DAYS + 1)
        event_days = days[np.asarray(record, dtype=bool)]
        non_event_days = days[~np.asarray(record, dtype=bool)]
        ax_record.scatter(event_days, np.full(len(event_days), row_i), marker="s", s=6,
                          color=COLORS["ink"], zorder=3)
        ax_record.scatter(non_event_days, np.full(len(non_event_days), row_i), marker="s", s=6,
                          facecolors="white", edgecolors="#C7CCCF", linewidths=0.45, zorder=2)

        scored_days = days[1:]
        scaled_height = np.clip(np.asarray(forecasts[1:]) / 0.5, 0, 1)
        line_y = row_i - 0.22 - 0.31 * scaled_height
        line_color = COLORS["pooled"] if shuffle_id == 0 else COLORS["shuffled_baseline"]
        ax_record.plot(scored_days, line_y, color=line_color, lw=0.7, zorder=4)
        ax_record.text(104.0, row_i, f"{row_auc:.3f}", ha="left", va="center", fontsize=7.8)

    ax_record.set_xlim(0.5, 111.5)
    ax_record.set_ylim(3.55, -0.85)
    ax_record.set_xticks([1, 50, 100], ["1", "50", "100"])
    ax_record.set_xlabel("Day (filled square = event)")
    ax_record.set_yticks(range(len(record_versions)), [row[2] for row in record_versions])
    ax_record.tick_params(axis="y", length=0, pad=4)
    ax_record.tick_params(axis="x", pad=3)
    ax_record.grid(axis="x", color=COLORS["grid"], lw=0.4)
    ax_record.set_axisbelow(True)
    ax_record.spines[["top", "right", "left"]].set_visible(False)
    ax_record.text(-0.16, 1.10, "a", transform=ax_record.transAxes, ha="left", va="bottom",
                   fontsize=10, fontweight="bold", clip_on=False)
    ax_record.text(0.935, 1.10, "AUC", transform=ax_record.transAxes,
                   ha="right", va="bottom", fontsize=7.5, fontweight="bold")

    bins = ax_hist.hist(shuffled_aucs, bins=18, range=(0, 1), color="#A9C7D7",
                        edgecolor="white", linewidth=0.55)
    max_bin = max(bins[0]) if len(bins[0]) else 1
    ax_hist.set_ylim(0, max_bin * 1.85)
    ax_hist.set_xlim(0, 1)
    ax_hist.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
    ax_hist.set_xlabel("AUC")
    ax_hist.set_ylabel("Count")
    arrow_y = max_bin * 1.34
    observed_line, = ax_hist.plot(
        [summary["auc_observed"], summary["auc_observed"]], [0, arrow_y],
        color=COLORS["pooled"], lw=1.4,
        label=f"Observed {summary['auc_observed']:.3f}", zorder=4)
    baseline_line, = ax_hist.plot(
        [summary["shuffled_baseline_mean"], summary["shuffled_baseline_mean"]],
        [0, arrow_y], color=COLORS["shuffled_baseline"], lw=1.4,
        label=f"Shuffled baseline {summary['shuffled_baseline_mean']:.3f}", zorder=4)
    ax_hist.axvline(0.5, color="#AEB4B8", lw=1.0, linestyle="--", zorder=3)
    ax_hist.text(0.51, max_bin * 1.81, "0.5", ha="left", va="top", fontsize=6.8,
                 color=COLORS["muted"])
    ax_hist.legend(handles=[observed_line, baseline_line], loc="upper center",
                   bbox_to_anchor=(0.5, -0.20), frameon=False, fontsize=6.4,
                   handlelength=1.4, labelspacing=0.2, borderaxespad=0.0)
    arrow_x0 = summary["auc_observed"]
    arrow_x1 = summary["shuffled_baseline_mean"]
    ax_hist.annotate("", xy=(arrow_x1, arrow_y), xytext=(arrow_x0, arrow_y),
                     arrowprops={"arrowstyle": "<->", "color": COLORS["worse"], "lw": 0.8})
    ax_hist.text(arrow_x0 - 0.02, arrow_y,
                 f"excess {summary['excess']:.3f}", ha="right", va="center",
                 fontsize=6.6, color=COLORS["worse"])
    ax_hist.grid(axis="y", color=COLORS["grid"], lw=0.5)
    ax_hist.set_axisbelow(True)
    ax_hist.spines[["top", "right"]].set_visible(False)
    ax_hist.text(-0.12, 1.10, "b", transform=ax_hist.transAxes, ha="left", va="bottom",
                 fontsize=10, fontweight="bold", clip_on=False)

    save_figure(fig, 4)
    plt.close(fig)


if __name__ == "__main__":
    main()
