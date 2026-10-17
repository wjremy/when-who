"""Finite-horizon values of models A and B on the ranked measures other than the AUC (ASTRA18, A2/A3).

The long-record truth (code/v19_long_record_truth.py) applies model A's short-record calibration to
records of 5,000 days. This script instead gives each measure's expected per-patient value on records
of the simulated length itself: many independent patients, one record each, with the same patient
generation, forecasts, threshold and per-patient definitions as code/ranking_sim.py
(_patient_statistics, then the mean over patients). Two values per measure:
- actual: the record's forecasts scored on the record's own outcomes (what raw per-patient scoring
  estimates, averaged over records instead of 50 patients); for model A the forecasts are built from
  those outcomes, so ranking measures carry the record dependence;
- ghost: the same forecasts scored on fresh outcomes drawn from the same patient level and the same
  daily components, independent of the record's outcomes given them. This is the model's skill on the
  record at hand without the feedback of outcomes into later forecasts. The Brier reference (the rate
  so far) stays built from the record's own outcomes, as the forecasts are. For day-by-day sums the two
  values have the same expectation.

Run from the repository root:
    python code/v19_finite_horizon_measures.py [--patients 20000]
Output: results/v19/A3_finite_horizon_measures.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import ranking_sim as sim  # noqa: E402

SOURCE_JSON = ROOT / "open_data" / "results" / "v10" / "ranking_sim.json"
OUTPUT_CSV = ROOT / "results" / "v19" / "A3_finite_horizon_measures.csv"
MEASURES = ("auc", "average_precision", "brier_skill_updated_rate", "brier_skill_oracle", "ppv", "net_benefit")
CHUNK = 2_000
OUTCOMES = ("actual", "ghost")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", type=int, default=20_000)
    args = parser.parse_args()
    started = time.perf_counter()
    payload = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    fits, calibrations = payload["run"]["cohort_fits"], payload["run"]["calibrations"]
    pairs = [dict(sim.BASE_PAIR), dict(sim.EXTRA_PAIR)]
    rows = []
    for ci, cohort in enumerate(sim.COHORTS):
        spec = sim.cohort_sim.COHORTS[cohort]
        pop = float(spec["rate"])
        for days in sim.LENGTHS:
            values = {(p["name"], m, o, k): [] for p in pairs for m in ("A", "B") for o in OUTCOMES for k in MEASURES}
            rng = sim._rng(940, ci, days)
            for start in range(0, args.patients, CHUNK):
                n = min(CHUNK, args.patients - start)
                b = sim._recentered_levels(rng, fits[cohort], spec, n)
                x, y = sim._draw_outcomes(rng, b, days)
                noise = rng.standard_normal(x.shape)
                patient_rate = sim.who_when.normal_expit_expectation(b, sim.X_SD * sim.X_SD)
                y_ghost = (rng.random(x.shape) < sim._expit(b[:, None] + x)).astype(np.int8)
                updated = sim._running_rate(y, pop)
                for pair in pairs:
                    cal = calibrations[f"{cohort}|{days}|{pair['rho_a']:.1f}"]
                    fa = sim._a_forecast(y, sim._daily_proxy(x, noise, pair["rho_a"]), pop, cal)
                    fb = sim._b_forecast(b, sim._daily_proxy(x, noise, pair["rho_b"]), pair["rho_b"])
                    for model, f in (("A", fa), ("B", fb)):
                        for outcome, yy in (("actual", y), ("ghost", y_ghost)):
                            stats = sim._patient_statistics(yy, f, pop, pop, patient_rate, updated)
                            for k in MEASURES:
                                values[(pair["name"], model, outcome, k)].append(sim._patient_means(stats[k]))
            for pair in pairs:
                for k in MEASURES:
                    for outcome in OUTCOMES:
                        est = {}
                        for model in ("A", "B"):
                            v = np.concatenate(values[(pair["name"], model, outcome, k)])
                            v = v[np.isfinite(v)]
                            est[model] = (float(v.mean()), float(v.std(ddof=1) / math.sqrt(v.size)), int(v.size))
                        for model in ("A", "B"):
                            rows.append({"pair": pair["name"], "spread": "low" if cohort == "REPLACE-BG" else "high",
                                         "days": days, "model": model, "measure": k, "outcomes": outcome,
                                         "value": est[model][0], "se": est[model][1],
                                         "patients_defined": est[model][2],
                                         "a_better": est["A"][0] > est["B"][0]})
            print(f"{cohort} {days} d done ({time.perf_counter() - started:.0f} s)", flush=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
