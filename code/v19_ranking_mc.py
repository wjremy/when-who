"""Monte Carlo errors and undefined counts for the ranking simulation (Supplement S6, Table 6).

Reruns only the ranking part of code/ranking_sim.py (same cells, seeds, tasks and worker; the power
part is not needed) and keeps what the stored JSON drops: the per-replication scores, for the Monte
Carlo standard error of each mean and share, and per replication the number of patients on whom a
measure is undefined. The means and shares are checked against open_data/results/v10/ranking_sim.json;
that file is not touched.

Run from the repository root (about 40-60 min with 6 workers):
    python code/v19_ranking_mc.py [--workers 6] [--reps 500]
Outputs: results/v19/S6_mc.csv (one row per cell, pair, scoring and measure) and
results/v19/S6_undefined.csv (one row per cell, pair and model).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import ranking_sim as sim  # noqa: E402

SOURCE_JSON = ROOT / "open_data" / "results" / "v10" / "ranking_sim.json"
OUT_MC = ROOT / "results" / "v19" / "S6_mc.csv"
OUT_UNDEFINED = ROOT / "results" / "v19" / "S6_undefined.csv"
SCORINGS = ("pooled", "naive", "tailored_excess")


def worker(task: dict[str, Any]) -> dict[str, Any]:
    """ranking_sim's worker, plus per-patient undefined counts on the observed record."""
    result = sim._main_worker(task)
    y, x, noise, b = task["y"], task["x"], task["noise"], task["b"]
    pop = task["population_rate"]
    events = y.sum(axis=1)
    days = y.shape[1]
    undefined: dict[str, dict[str, int]] = {}
    for pair in task["pairs"]:
        fa = sim._a_forecast(y, sim._daily_proxy(x, noise, pair["rho_a"]), pop,
                             task["calibrations"][str(pair["rho_a"])])
        fb = sim._b_forecast(b, sim._daily_proxy(x, noise, pair["rho_b"]), pair["rho_b"])
        undefined[pair["name"]] = {
            "no_event": int(np.sum(events == 0)),
            "all_events": int(np.sum(events == days)),
            "a_no_alarm": int(np.sum((fa >= pop).sum(axis=1) == 0)),
            "b_no_alarm": int(np.sum((fb >= pop).sum(axis=1) == 0)),
        }
    result["undefined"] = undefined
    return result


def pair_values(scores: dict[str, Any], scoring: str, metric: str) -> tuple[float, float]:
    """Model A's and B's score in one replication, as ranking_sim._ranking_rows forms them."""
    if scoring == "tailored_excess":
        return (scores["tailored_a_actual"][metric] - scores["tailored_a_chance"][metric],
                scores["tailored_b_actual"][metric] - scores["tailored_b_chance_shuffle"][metric])
    key = "pooled" if scoring == "pooled" else "naive"
    return scores[f"{key}_a"][metric], scores[f"{key}_b"][metric]


def se(values: np.ndarray) -> float:
    return float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--cells", type=int, help="first N cells only (smoke test; no check against the JSON)")
    args = parser.parse_args()
    started = time.perf_counter()
    stored = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    run = stored["run"]
    assert run["seed"] == sim.BASE_SEED and run["b_chance"] == "shuffle" and run["extra_pair_included"]
    n_perm = int(run["chance_shuffles_per_replication"])
    cells = [(c["cohort"], int(c["record_days"]), int(c["patients"])) for c in run["cells"]]
    if args.cells:
        cells = cells[:args.cells]
    pairs = [dict(sim.BASE_PAIR), dict(sim.EXTRA_PAIR)]
    specs, fits, calibrations = sim._context(cells, True)
    results = sim._parent_generated_map(
        worker, sim._main_tasks(cells, args.reps, n_perm, specs, fits, calibrations, pairs), args.workers)
    results.sort(key=lambda row: (sim._cell_index(tuple(row["cell"])), row["rep"]))
    print(f"ranking part done in {time.perf_counter() - started:.0f} s", flush=True)

    check = args.reps == int(run["ranking_replications_per_cell"]) and not args.cells
    stored_rows = {(r["cohort"], int(r["record_days"]), int(r["patients"]), r["pair"], r["scoring"], r["metric"]): r
                   for r in stored["ranking"]}
    mc_rows, undefined_rows, checked, mismatches = [], [], 0, []
    for cell in cells:
        cell_results = [r for r in results if tuple(r["cell"]) == cell]
        for pair in pairs:
            name = pair["name"]
            for scoring in SCORINGS:
                metrics = cell_results[0]["pairs"][name]["tailored_a_actual" if scoring == "tailored_excess"
                                                        else f"{'pooled' if scoring == 'pooled' else 'naive'}_a"].keys()
                for metric in metrics:
                    ab = np.array([pair_values(r["pairs"][name], scoring, metric) for r in cell_results], dtype=float)
                    ok = np.isfinite(ab).all(axis=1)
                    a, b = ab[ok, 0], ab[ok, 1]
                    share = float(np.mean(a > b)) if a.size else float("nan")
                    row = {"cohort": cell[0], "spread": "low" if cell[0] == "REPLACE-BG" else "high",
                           "record_days": cell[1], "patients": cell[2], "pair": name, "scoring": scoring,
                           "metric": metric, "replications": len(cell_results), "valid_replications": int(a.size),
                           "share_a_ranks_above_b": share,
                           "share_mc_se": math.sqrt(share * (1 - share) / a.size) if a.size else float("nan"),
                           "mean_a": float(a.mean()) if a.size else float("nan"), "mean_a_mc_se": se(a),
                           "mean_b": float(b.mean()) if b.size else float("nan"), "mean_b_mc_se": se(b),
                           "sd_a": float(a.std(ddof=1)) if a.size > 1 else float("nan"),
                           "sd_b": float(b.std(ddof=1)) if b.size > 1 else float("nan")}
                    key = (*cell, name, scoring, metric)
                    if check and key in stored_rows:
                        s = stored_rows[key]
                        if s["valid_replications"] != row["valid_replications"]:
                            mismatches.append((key, "valid_replications", row["valid_replications"], s["valid_replications"]))
                        for mine, theirs in (("mean_a", "mean_a_score"), ("mean_b", "mean_b_score"),
                                             ("share_a_ranks_above_b", "share_a_ranks_above_b")):
                            if s[theirs] is not None and not math.isclose(row[mine], s[theirs], rel_tol=0, abs_tol=1e-9):
                                mismatches.append((key, mine, row[mine], s[theirs]))
                        checked += 1
                    mc_rows.append(row)
            counts = np.array([[r["undefined"][name][k] for k in ("no_event", "all_events", "a_no_alarm", "b_no_alarm")]
                               for r in cell_results], dtype=float)
            for model, alarm_col in (("A", 2), ("B", 3)):
                undefined_rows.append({
                    "cohort": cell[0], "spread": "low" if cell[0] == "REPLACE-BG" else "high",
                    "record_days": cell[1], "patients": cell[2], "pair": name, "model": model,
                    "mean_patients_no_event": counts[:, 0].mean(),
                    "mean_patients_all_events": counts[:, 1].mean(),
                    "mean_patients_auc_undefined": (counts[:, 0] + counts[:, 1]).mean(),
                    "max_patients_auc_undefined": (counts[:, 0] + counts[:, 1]).max(),
                    "mean_patients_ppv_undefined_no_alarm": counts[:, alarm_col].mean(),
                    "max_patients_ppv_undefined_no_alarm": counts[:, alarm_col].max(),
                })
    if check:
        expected = sum(1 for k in stored_rows if k[4] in SCORINGS)
        print(f"checked {checked} of {expected} stored ranking rows; {len(mismatches)} mismatches")
        for m in mismatches[:20]:
            print("MISMATCH", m)
    OUT_MC.parent.mkdir(parents=True, exist_ok=True)
    for path, rows in ((OUT_MC, mc_rows), (OUT_UNDEFINED, undefined_rows)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")
    print(f"done in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
