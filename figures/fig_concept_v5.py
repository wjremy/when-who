"""Conceptual figure (v5): three ways to score a patient-by-time grid.

Rows are patients, columns are time points. (1) Split the grid into rows, score each row (one patient's days:
when?), and average over patients. (2) Split it into columns, score each column (all patients at one time point:
who, now?), and average over time points. (3) Pool: score all cells at once; most pairs then share neither the
patient nor the time point. The pair shares in the footer are computed for a 200 x 180 grid.

Usage: python figures/fig_concept_v5.py  -> figures/fig_concept_v5.png, .pdf
"""
import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = pathlib.Path(__file__).resolve().parent
WHEN = "#D55E00"     # vermilion: within patient (as in Figures 1-2)
WHO = "#009E73"      # bluish green: within time point
POOLED = "#0072B2"   # blue: pooled (as in Figures 1-2)
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#c8c8c8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "text.color": INK})

NP, NT = 4, 8
EVENTS = {(0, 0), (0, 1), (0, 3), (0, 4), (0, 6), (0, 7), (1, 2), (1, 5), (1, 6), (2, 4), (3, 1)}
C = 0.9  # cell size


def share(P, T):
    n = P * T
    total = n * (n - 1) / 2
    row, col = P * T * (T - 1) / 2, T * P * (P - 1) / 2
    return row / total, col / total, 1 - (row + col) / total


def cell(ax, x, y, p, t):
    ev = (p, t) in EVENTS
    ax.add_patch(Rectangle((x, y), C, C, facecolor=INK if ev else "white", edgecolor=INK if ev else GRID,
                           linewidth=0.6))


def arrow(ax, a, b, colour):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, color=colour, linewidth=1.0,
                                 shrinkA=0, shrinkB=1))


def box(ax, x, y, w, h, text, colour):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.25",
                                facecolor=colour, edgecolor=colour))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="white", fontsize=7.5, fontweight="bold")


fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.1))
fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.22, wspace=0.08)
for ax in axes:
    ax.set_xlim(-1.3, 12.2)
    ax.set_ylim(-3.6, 6.4)
    ax.set_aspect("equal")
    ax.axis("off")

# ---- 1: rows ----------------------------------------------------------------------------------------
ax = axes[0]
ROWGAP = 1.45
for p in range(NP):
    y = 5.0 - p * ROWGAP
    ax.text(-0.2, y + C / 2, f"P{p + 1}", ha="right", va="center", fontsize=7.5, color=MUTED)
    for t in range(NT):
        cell(ax, t, y, p, t)
    ax.add_patch(Rectangle((-0.08, y - 0.08), NT - 0.02, C + 0.16, fill=False, edgecolor=WHEN, linewidth=1.3))
    arrow(ax, (NT + 0.05, y + C / 2), (NT + 1.25, 2.85), WHEN)
box(ax, NT + 1.25, 2.3, 2.6, 1.1, "mean", WHEN)
ax.text(-1.3, -0.6, "time →", fontsize=7.5, color=MUTED)

# ---- 2: columns -------------------------------------------------------------------------------------
ax = axes[1]
COLGAP = 1.35
for t in range(NT):
    x = t * COLGAP
    for p in range(NP):
        cell(ax, x, 5.0 - p * 1.0, p, t)
    ax.add_patch(Rectangle((x - 0.08, 5.0 - (NP - 1) * 1.0 - 0.08), C + 0.16, (NP - 1) * 1.0 + C + 0.16,
                           fill=False, edgecolor=WHO, linewidth=1.3))
    arrow(ax, (x + C / 2, 1.9), (NT * COLGAP / 2 - 0.4 - 1.0 + 2.0 * t / (NT - 1), -1.5), WHO)
for p in range(NP):
    ax.text(-0.25, 5.0 - p * 1.0 + C / 2, f"P{p + 1}", ha="right", va="center", fontsize=7.5, color=MUTED)
box(ax, NT * COLGAP / 2 - 1.7, -2.65, 2.6, 1.1, "mean", WHO)

# ---- 3: pooled --------------------------------------------------------------------------------------
ax = axes[2]
for p in range(NP):
    ax.text(-0.25, 5.0 - p * 1.0 + C / 2, f"P{p + 1}", ha="right", va="center", fontsize=7.5, color=MUTED)
    for t in range(NT):
        cell(ax, t * 1.0, 5.0 - p * 1.0, p, t)
ax.add_patch(Rectangle((-0.1, 1.9), NT + 0.1, NP + 0.1 - 0.1 + 0.1, fill=False, edgecolor=POOLED, linewidth=1.6))
arrow(ax, (NT / 2, 1.85), (NT / 2, -1.55), POOLED)
box(ax, NT / 2 - 2.3, -2.65, 4.6, 1.1, "one number", POOLED)

titles = [("1  Per patient: when?", WHEN, "score each row, then average:\nwhich of my days are risky?"),
          ("2  Per time point: who?", WHO, "score each column, then average:\nwhich patient is at risk now?"),
          ("3  Pooled", POOLED, "score all cells at once: most pairs\nshare neither patient nor time")]
for ax, (title, colour, sub) in zip(axes, titles):
    ax.text(-1.3, 6.6, title, fontsize=9.5, fontweight="bold", color=colour, va="bottom")
    ax.text(-1.3, -3.0 if ax is axes[0] else -3.15, sub, fontsize=7.5, color=MUTED, va="top")

fig.text(0.01, 0.045, "In three clinical cohorts, 95-98% of the event/event-free pairs behind a pooled AUC shared neither "
         "the patient nor the time point.", fontsize=8)
fig.text(0.01, -0.01, "Score 1 and 2 separately for the use at hand; 3 alone cannot show either ability.",
         fontsize=8, color=MUTED)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig_concept_v5.{ext}", dpi=300, bbox_inches="tight")
