"""F3: record length and the two dials, from the existing grid and A3 schema."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from style import COLORS, TWO_COLUMNS, mpl_or_report, read_json, save_figure, write_csv

HISTORY = ROOT / "results/history_bias_grid.json"
A3 = ROOT / "results/v13/A3_dials.csv"
SAFE_LINE_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]


def collect_data():
    history = [r for r in read_json(HISTORY) if r["feature"] == "expanding"]
    history.sort(key=lambda r: (float(r["p"]), int(r["T"])))
    rows = [{"panel": "a", **r} for r in history]
    rows.append({"panel": "a_reference", "feature": "no-skill AUC baseline", "within_auc": 0.5})
    a3_rows = []
    if A3.exists():
        with A3.open(newline="", encoding="utf-8-sig") as handle:
            a3_rows = list(csv.DictReader(handle))
        rows.extend({"panel": "b", **r} for r in a3_rows)
    else:
        rows.append({"panel": "b_status", "status": "A3_dials.csv not present yet"})
    return history, a3_rows, rows


def main():
    history, a3_rows, rows = collect_data()
    fields = ["panel", "T", "p", "feature", "within_auc", "approx", "patients", "K",
              "share_within_patient_pairs", "one_over_K", "auc_rate_so_far_mean",
              "auc_rate_so_far_mcse", "replications", "status"]
    write_csv(ROOT / "results/v13/F3_numbers.csv", fields, rows)
    plt = mpl_or_report()
    if plt is None:
        return
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(TWO_COLUMNS, 84 / 25.4))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.20, top=0.87,
                        wspace=0.38)
    fig.suptitle("Record length and the two dials", x=0.01, ha="left",
                 fontsize=9.5, fontweight="bold")
    ax = axes[0]
    ts_history = sorted({int(r["T"]) for r in history})
    x_history = list(range(len(ts_history)))
    for p_i, p in enumerate(sorted({float(r["p"]) for r in history})):
        subset = sorted((r for r in history if float(r["p"]) == p), key=lambda r: int(r["T"]))
        ys = [float(r["within_auc"]) for r in subset]
        ax.plot(x_history, ys, color=SAFE_LINE_COLORS[p_i], marker="o", markersize=3.2,
                lw=1.1, label=f"{p:.0%}")
        vertical_offsets = {0.05: -12, 0.15: 0, 0.30: 12}
        ax.annotate(f"{p:.0%}", xy=(x_history[-1], ys[-1]),
                    xytext=(5, vertical_offsets.get(p, 0)),
                    textcoords="offset points", ha="left", va="center", fontsize=8.0,
                    color=SAFE_LINE_COLORS[p_i])
    ax.axhline(0.5, color=COLORS["muted"], lw=0.6, ls=(0, (2, 2)), zorder=0)
    ax.set_xlim(-0.15, len(ts_history) - 0.15)
    ax.set_xticks(x_history, [str(t) for t in ts_history])
    ax.set_ylim(0.1, 0.53)
    ax.set_title("a  Rate so far", loc="left", fontweight="bold", pad=4)
    ax.set_xlabel("Record length (days)")
    ax.set_ylabel("Per-patient AUC")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    for ax, field, title, ylabel in (
        (axes[1], "share_within_patient_pairs", "b  Same-patient pairs", "Share of pooled pairs"),
        (axes[2], "auc_rate_so_far_mean", "c  Per-patient score", "Mean AUC"),
    ):
        if not a3_rows:
            ax.text(0.5, 0.5, "Waiting for A3 dial data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8.0, color=COLORS["muted"])
            ax.set_axis_off()
            ax.set_title(title, loc="left", fontweight="bold", pad=4)
            continue
        ts = sorted({int(float(r["T"])) for r in a3_rows})
        ks = sorted({int(float(r["K"])) for r in a3_rows})
        xpos = list(range(len(ks)))
        for ti, t in enumerate(ts):
            sub = sorted((r for r in a3_rows if int(float(r["T"])) == t),
                         key=lambda r: int(float(r["K"])))
            ys = [float(r[field]) for r in sub]
            ax.plot(xpos, ys, color=SAFE_LINE_COLORS[ti % len(SAFE_LINE_COLORS)],
                    marker="o", markersize=3.0, lw=1.0, zorder=2)
            if field == "auc_rate_so_far_mean":
                ax.annotate(f"T={t}", xy=(xpos[-1], ys[-1]), xytext=(5, 0),
                            textcoords="offset points", ha="left", va="center",
                            fontsize=8.0, color=SAFE_LINE_COLORS[ti % len(SAFE_LINE_COLORS)])
        if field == "share_within_patient_pairs":
            ax.plot(xpos, [1 / k for k in ks], color=COLORS["ink"], lw=1.35,
                    ls=(0, (3, 2)), zorder=6)
            ax.text(0.97, 0.06, "1/K", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8.0, fontweight="bold", color=COLORS["ink"],
                    bbox={"facecolor": COLORS["paper"], "edgecolor": "none", "pad": 0.8})
            ax.legend(handles=[
                Line2D([], [], color=SAFE_LINE_COLORS[i], marker="o", lw=1.0,
                       markersize=3.0, label=f"T={t}")
                for i, t in enumerate(ts)
            ], frameon=False, fontsize=8.0, ncol=2, loc="upper right",
               borderaxespad=0.1, handlelength=1.2, columnspacing=0.8)
        ax.set_xlim(-0.1, len(ks) + (0.55 if field == "auc_rate_so_far_mean" else -0.1))
        ax.set_xticks(xpos, [str(k) for k in ks])
        ax.set_title(title, loc="left", fontweight="bold", pad=4)
        ax.set_xlabel("Number of patients, K")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=COLORS["grid"], lw=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[2].set_ylim(0.25, 0.49)
    save_figure(fig, 3)
    plt.close(fig)


if __name__ == "__main__":
    main()
