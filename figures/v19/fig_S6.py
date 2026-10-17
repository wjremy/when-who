"""Supplement Figure S6: ranking the closer pair with 150 patients."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from fig_F7 import plot_ranking_figure
from style import read_json, write_csv

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


def main():
    patients = 150
    rows = collect_data(patients)
    write_csv(ROOT / "results/v19/S6_numbers.csv",
              ["cohort", "patient_spread", "record_days", "patients", "pair", "metric",
               "scoring", "scoring_label", "replications", "share_a_ranks_above_b",
               "equal_ranking_reference", "mean_a_score", "mean_b_score", "seed",
               "chance_shuffles_per_replication"], rows)
    plot_ranking_figure(rows, 6, name="FS6")


if __name__ == "__main__":
    main()
