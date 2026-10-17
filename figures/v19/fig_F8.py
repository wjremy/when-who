"""Figure 8: growth of the literature that the audit query finds (Discussion 7.1; ledger AUD6).

Input: results/v19/F8_numbers.csv (code/v19_pubmed_totals.py). Years run from October to September
and are labelled by the year they end in; 2026 was retrieved on 15 Oct 2026 and can still grow
with indexing, so it is drawn open.
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
from style import COLORS, TWO_COLUMNS, mpl_or_report, save_figure  # noqa: E402

SOURCE = ROOT / "results" / "v19" / "F8_numbers.csv"
LAST = 2026


def main() -> None:
    with SOURCE.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    years = [int(r["year_ending_sep"]) for r in rows]
    records = [int(r["records"]) for r in rows]
    share = [float(r["share_percent"]) for r in rows]
    plt = mpl_or_report()
    if plt is None:
        return

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(TWO_COLUMNS, 62 / 25.4))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.17, top=0.88, wspace=0.30)

    for y, n in zip(years, records):
        last = y == LAST
        ax_a.bar(y, n / 1000, width=0.75, color=COLORS["paper"] if last else COLORS["pooled"],
                 edgecolor=COLORS["pooled"], linewidth=0.9 if last else 0, hatch="////" if last else None,
                 zorder=2)
    ax_a.set_ylabel("Records per year (thousands)")
    ax_a.set_ylim(0, 72)

    ax_b.plot(years[:-1], share[:-1], color=COLORS["pooled"], lw=1.4, marker="o", ms=3, zorder=3)
    ax_b.plot(years[-2:], share[-2:], color=COLORS["pooled"], lw=1.2, ls=(0, (2, 1.5)), zorder=3)
    ax_b.plot([years[-1]], [share[-1]], marker="o", ms=4, mfc=COLORS["paper"], mec=COLORS["pooled"],
              mew=1.1, ls="none", zorder=4)
    ax_b.set_ylabel("Share of all PubMed records (%)")
    ax_b.set_ylim(0, 3.6)

    note = f"{LAST}: retrieved 15 Oct {LAST},\nmay still grow with indexing"
    ax_a.annotate(note, xy=(LAST, records[-1] / 1000), xytext=(LAST - 1.6, 63), fontsize=6.6,
                  color=COLORS["muted"], ha="right", va="center",
                  arrowprops={"arrowstyle": "-", "color": COLORS["muted"], "lw": 0.6})
    for y in (2016, 2021, LAST):
        k = years.index(y)
        ax_b.annotate(f"{share[k]:.1f}%", xy=(y, share[k]), xytext=(-4, 6), textcoords="offset points",
                      ha="right", va="bottom", fontsize=6.8, color=COLORS["ink"])

    for ax, letter in ((ax_a, "a"), (ax_b, "b")):
        ax.set_xlim(2000.3, 2026.7)
        ax.set_xticks([2001, 2006, 2011, 2016, 2021, 2026])
        ax.set_xlabel("Publication year (October to September)")
        ax.text(0.0, 1.06, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=10,
                fontweight="bold", color=COLORS["ink"], clip_on=False)
        ax.grid(axis="y", color=COLORS["grid"], lw=0.45, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=2, pad=2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(COLORS["grid"])

    save_figure(fig, 8)
    plt.close(fig)


if __name__ == "__main__":
    main()
