"""Shared print style and score/baseline glyphs for the v13 figures."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PDF_DPI = 300
MM = 1 / 25.4
ONE_COLUMN = 89 * MM
TWO_COLUMNS = 183 * MM

# Okabe-Ito colours keep the three scoring methods distinct in print.
COLORS = {
    "pooled": "#0072B2",
    "per_patient": "#E69F00",
    "shuffled_baseline": "#CC79A7",
    "better": "#009E73",
    "worse": "#D55E00",
    "ink": "#252525",
    "muted": "#727272",
    "grid": "#E5E7E8",
    "paper": "#FFFFFF",
    "soft": "#F4F6F7",
}

SCORING_ORDER = ("pooled", "per_patient", "shuffled_baseline")
SCORING_LABELS = {
    "pooled": "Pooled",
    "per_patient": "Per-patient",
    "shuffled_baseline": "Shuffled-baseline",
}


def setup_mpl():
    """Import and configure Matplotlib lazily so data generation works without it."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.edgecolor": "#AEB4B8",
        "axes.linewidth": 0.65,
        "axes.facecolor": COLORS["paper"],
        "figure.facecolor": COLORS["paper"],
        "savefig.facecolor": COLORS["paper"],
        "text.color": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    return plt


def save_figure(fig, number: int) -> None:
    """Write fixed-size vector PDF and 300-dpi PNG for one figure."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"F{number}.pdf", dpi=PDF_DPI)
    fig.savefig(OUT / f"F{number}.png", dpi=PDF_DPI)


def dot_ring_gap(ax, y: float, score: float, baseline: float, *,
                 score_color: str = COLORS["ink"], height: float = 0.0,
                 marker_size: float = 5.0, line_width: float = 1.8,
                 zorder: int = 4, orientation: str = "horizontal",
                 marker_offset_points: float = 0.0):
    """Draw score as a filled dot, baseline as an open ring, and colour their gap.

    Positive excess is green (looks better); negative excess is vermilion
    (looks worse). The scoring method's fixed colour is used for both markers.
    """
    if orientation not in {"horizontal", "vertical"}:
        raise ValueError("orientation must be horizontal or vertical")
    yy = y + height
    gap_color = COLORS["better"] if score >= baseline else COLORS["worse"]
    # A tiny display-space offset keeps a zero or near-zero score-baseline gap
    # legible without changing either value or the connector's data position.
    baseline_transform = ax.transData
    score_transform = ax.transData
    if marker_offset_points:
        from matplotlib.transforms import offset_copy
        if orientation == "horizontal":
            baseline_transform = offset_copy(ax.transData, fig=ax.figure,
                                             y=marker_offset_points, units="points")
            score_transform = offset_copy(ax.transData, fig=ax.figure,
                                          y=-marker_offset_points, units="points")
        else:
            baseline_transform = offset_copy(ax.transData, fig=ax.figure,
                                             x=-marker_offset_points, units="points")
            score_transform = offset_copy(ax.transData, fig=ax.figure,
                                          x=marker_offset_points, units="points")
    if orientation == "horizontal":
        ax.plot([baseline, score], [yy, yy], color=gap_color, lw=line_width,
                solid_capstyle="round", zorder=zorder - 1)
        ax.scatter([baseline], [yy], s=marker_size ** 2, marker="o",
                   facecolors=COLORS["paper"], edgecolors=score_color,
                   linewidths=1.15, zorder=zorder + 1, transform=baseline_transform)
        ax.scatter([score], [yy], s=marker_size ** 2 * 0.8, marker="o",
                   color=score_color, edgecolors=score_color,
                   linewidths=0.5, zorder=zorder + 2, transform=score_transform)
    else:
        ax.plot([yy, yy], [baseline, score], color=gap_color, lw=line_width,
                solid_capstyle="round", zorder=zorder - 1)
        ax.scatter([yy], [baseline], s=marker_size ** 2, marker="o",
                   facecolors=COLORS["paper"], edgecolors=score_color,
                   linewidths=1.15, zorder=zorder + 1, transform=baseline_transform)
        ax.scatter([yy], [score], s=marker_size ** 2 * 0.8, marker="o",
                   color=score_color, edgecolors=score_color,
                   linewidths=0.5, zorder=zorder + 2, transform=score_transform)
    return gap_color


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Write a stable, tidy CSV with an explicit header, including empty tables."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path):
    import json

    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def mpl_or_report():
    """Return pyplot, or None with a concise message when Matplotlib is absent."""
    try:
        return setup_mpl()
    except ImportError:
        print("Matplotlib is not installed in this Python; data CSV was written, rendering skipped.")
        return None
