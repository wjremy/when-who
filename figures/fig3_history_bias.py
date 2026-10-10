from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
GRID_FILE = PROJECT / "results" / "history_bias_grid.json"
FIX_FILE = PROJECT / "results" / "history_bias_fix.json"

# Okabe-Ito colors and compact journal styling.
BLUE = "#0072B2"
GREEN = "#009E73"
PINK = "#CC79A7"
ORANGE = "#E69F00"
BLACK = "#000000"
GRAY = "#888888"
MARKER_SIZE = 3.5
LINE_WIDTH = 1.0

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

with GRID_FILE.open(encoding="utf-8") as f:
    grid = json.load(f)
with FIX_FILE.open(encoding="utf-8") as f:
    fix = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

# Panel a: no-skill history forecast by diary length.
ax = axes[0]
colors = {0.05: BLUE, 0.15: GREEN, 0.30: ORANGE}
for p in (0.05, 0.15, 0.30):
    rows = sorted(
        (r for r in grid if r["feature"] == "expanding" and r["p"] == p),
        key=lambda r: r["T"],
    )
    diary_days = np.array([r["T"] for r in rows], dtype=float)
    auc = np.array([r["within_auc"] for r in rows], dtype=float)
    ax.plot(diary_days, auc, marker="o", markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH, color=colors[p], label=f"Event rate {p:.0%}")
ax.axhline(0.5, color=BLACK, linestyle="--", linewidth=LINE_WIDTH)
ax.set_xscale("log")
ax.set_xlabel("Diary length (days)")
ax.set_ylabel("Within-patient AUC of a running\nevent rate, independent events")
ax.set_xlim(50, 800)
ax.set_ylim(0.10, 0.53)
ticks_a = [60, 100, 180, 365, 730]
ax.set_xticks(ticks_a, labels=[str(x) for x in ticks_a])
ax.get_xaxis().set_minor_locator(plt.NullLocator())
ax.text(730, 0.503, "0.5", ha="right", va="bottom", fontsize=7)
ax.legend(frameon=False, loc="lower right", handlelength=1.8,
          borderaxespad=0.4)

# Panel b: paired raw and permutation-null-corrected estimates.
ax = axes[1]
rows = [r for r in fix if r["signal_b"] == 0.0]
groups = [(7, 100), (30, 100), (7, 365), (30, 365)]
raw_values = []
corrected_values = []
for window, diary_length in groups:
    row = next(r for r in rows
               if r["window"] == window and r["T"] == diary_length)
    raw_values.append(float(row["raw"]))
    corrected_values.append(float(row["corrected"]))

x = np.arange(len(groups), dtype=float)
for xpos, raw, corrected in zip(x, raw_values, corrected_values):
    ax.vlines(xpos, min(raw, corrected), max(raw, corrected),
              color=GRAY, linewidth=0.7, zorder=1)
    ax.plot(xpos, raw, marker="o", linestyle="none", markersize=MARKER_SIZE,
            markerfacecolor="white", markeredgecolor=PINK,
            markeredgewidth=LINE_WIDTH, zorder=2)
    ax.plot(xpos, corrected, marker="o", linestyle="none",
            markersize=MARKER_SIZE, markerfacecolor=BLUE,
            markeredgecolor=BLUE, markeredgewidth=LINE_WIDTH, zorder=3)
ax.set_ylabel("Within-patient AUC")
ax.set_ylim(0.40, 0.52)
ax.set_yticks(np.arange(0.40, 0.5201, 0.02))
ax.set_xlim(-0.5, 3.5)
ax.axhline(0.5, color=BLACK, linestyle="--", linewidth=LINE_WIDTH)
ax.text(3.45, 0.502, "0.5", ha="right", va="bottom", fontsize=7)
ax.set_xticks(x, labels=[
    "7-day\n100 d",
    "30-day\n100 d",
    "7-day\n365 d",
    "30-day\n365 d",
])
ax.tick_params(axis="x", labelsize=8, pad=3)
ax.set_xlabel("History window and diary length")

legend_handles = [
    Line2D([], [], marker="o", linestyle="none", markersize=MARKER_SIZE,
           markerfacecolor="white", markeredgecolor=PINK,
           markeredgewidth=LINE_WIDTH, label="Observed"),
    Line2D([], [], marker="o", linestyle="none", markersize=MARKER_SIZE,
           markerfacecolor=BLUE, markeredgecolor=BLUE,
           markeredgewidth=LINE_WIDTH,
           label="Observed minus chance level, plus 0.5"),
]

# Put panel labels above the axes, outside the data regions.
for ax, label in zip(axes, ("a", "b")):
    ax.text(-0.12, 1.045, label, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9, fontweight="bold",
            clip_on=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.5, width=0.7)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)

axes[1].legend(handles=legend_handles, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 1.08), ncol=2, handlelength=1.2,
               handletextpad=0.5, borderaxespad=0.0)
fig.subplots_adjust(left=0.13, right=0.99, bottom=0.22, top=0.88, wspace=0.38)
fig.savefig(HERE / "fig3_history_bias.pdf", bbox_inches="tight")
fig.savefig(HERE / "fig3_history_bias.png", dpi=300, bbox_inches="tight")
