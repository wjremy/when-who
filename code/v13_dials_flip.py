"""A2 flip and A3 two-dials simulations for the v13 pooled-metrics paper.

Usage (project Python with numpy and pandas):
    python code/v13_dials_flip.py

The bulk AUC calculation is vectorized and uses the same midrank/tie rule as
wheneval.auc. A small explicit check against that package function runs
before simulation. A2c deliberately uses the same draw order and seed as
code/history_bias.py so its rate-so-far values can be checked against
results/history_bias_grid.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WHENEVAL_SRC = ROOT / "code" / "wheneval" / "src"
sys.dont_write_bytecode = True
sys.path.insert(0, str(WHENEVAL_SRC))
from wheneval import auc as package_auc  # noqa: E402

OUT = ROOT / "results" / "v13"
FLIP_SEED = 20261012
DIALS_SEED = 20261012
GENERIC_SEED = 1  # exact source stream from code/history_bias.py

FLIP_HORIZONS = (10, 30, 100, 300, 1000, 3000, 5000)  # scored days, after one unscored day
FLIP_REPS = 2500
GENERIC_DAYS = (60, 100, 180, 365, 730)
EVENT_RATES = (0.05, 0.15, 0.30)
GENERIC_PATIENTS = 3000
RUN_IN_DAYS = 30
DIAL_K = (4, 8, 16, 32, 100)
DIAL_T = (10, 20, 40, 80, 365)
DIAL_REPS_INITIAL = 4000
DIAL_REPS_INCREMENT = 1000
DIAL_REPS_MAX = 10000
DIAL_BATCH = 64


def row_auc(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Per-row AUC for arrays whose last axis is days; ties receive half credit."""
    yy = np.asarray(y, dtype=np.uint8)
    ss = np.asarray(score, dtype=float)
    order = np.argsort(ss, axis=-1, kind="stable")
    sorted_s = np.take_along_axis(ss, order, axis=-1)
    sorted_y = np.take_along_axis(yy, order, axis=-1).astype(float)
    n = ss.shape[-1]
    pos_idx = np.broadcast_to(np.arange(n, dtype=float), ss.shape)
    starts = np.ones(ss.shape, dtype=bool)
    starts[..., 1:] = sorted_s[..., 1:] != sorted_s[..., :-1]
    ends = np.ones(ss.shape, dtype=bool)
    ends[..., :-1] = sorted_s[..., :-1] != sorted_s[..., 1:]
    left = np.maximum.accumulate(np.where(starts, pos_idx, 0.0), axis=-1)
    right = np.minimum.accumulate(
        np.where(ends, pos_idx, float(n - 1))[..., ::-1], axis=-1
    )[..., ::-1]
    ranks = (left + right + 2.0) / 2.0  # 1-based midranks
    n1 = np.sum(yy, axis=-1).astype(float)
    n0 = n - n1
    rank_sum = np.sum(ranks * sorted_y, axis=-1)
    denom = n1 * n0
    return np.divide(
        rank_sum - n1 * (n1 + 1.0) / 2.0,
        denom,
        out=np.full(n1.shape, np.nan, dtype=float),
        where=denom > 0,
    )


def check_auc_definition() -> None:
    y = np.array([0, 1, 1, 0, 1, 0], dtype=np.uint8)
    s = np.array([0.2, 0.4, 0.4, 0.4, 0.9, 0.2], dtype=float)
    got = float(row_auc(y[None, :], s[None, :])[0])
    expected = package_auc(y, s)
    if not np.isclose(got, expected, atol=1e-14, rtol=0):
        raise AssertionError(f"Vectorized AUC {got} differs from wheneval.auc {expected}")


def write_clara() -> float:
    event_days = {2, 5, 9}
    outcomes = np.array([int(day in event_days) for day in range(1, 11)], dtype=np.uint8)
    prior_events = np.r_[0, np.cumsum(outcomes[:-1])]
    day_numbers = np.arange(1, 11)
    rate = np.full(10, np.nan, dtype=float)
    rate[1:] = prior_events[1:] / (day_numbers[1:] - 1)
    flipped = 1.0 - rate
    other_nine = (outcomes.sum() - outcomes) / 9.0
    table = pd.DataFrame({
        "day": day_numbers,
        "outcome": outcomes.astype(int),
        "rate_so_far": rate,
        "flipped": flipped,
        "other_nine_rate": other_nine,
    })
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "A2a_clara.csv", index=False, float_format="%.10g")
    scored_y, scored_rate = outcomes[1:], rate[1:]
    return package_auc(scored_y, scored_rate)


def run_flip_curve() -> pd.DataFrame:
    rng = np.random.default_rng(FLIP_SEED)
    max_scored = max(FLIP_HORIZONS)
    batch_size = 64
    auc_store = {h: [] for h in FLIP_HORIZONS}
    event_score_sum = {h: 0.0 for h in FLIP_HORIZONS}
    event_n = {h: 0 for h in FLIP_HORIZONS}
    free_score_sum = {h: 0.0 for h in FLIP_HORIZONS}
    free_n = {h: 0 for h in FLIP_HORIZONS}

    for start in range(0, FLIP_REPS, batch_size):
        reps = min(batch_size, FLIP_REPS - start)
        # One independent day initializes the history; the next h days are the h scored days.
        y = (rng.random((reps, max_scored + 1)) < 0.30).astype(np.uint8)
        prior = np.cumsum(y, axis=1) - y
        denom = np.arange(max_scored + 1, dtype=float)
        rate = np.divide(prior, denom[None, :], out=np.zeros_like(prior, dtype=float),
                         where=denom[None, :] > 0)
        flipped = 1.0 - rate
        for h in FLIP_HORIZONS:
            yy = y[:, 1:h + 1]
            ff = flipped[:, 1:h + 1]
            auc_flip = row_auc(yy, ff)
            auc_store[h].append(auc_flip)
            ev = yy == 1
            fr = ~ev
            event_score_sum[h] += float(np.sum(ff[ev]))
            event_n[h] += int(np.sum(ev))
            free_score_sum[h] += float(np.sum(ff[fr]))
            free_n[h] += int(np.sum(fr))

    rows = []
    for h in FLIP_HORIZONS:
        auc_values = np.concatenate(auc_store[h])
        valid = auc_values[np.isfinite(auc_values)]
        mean_flip = float(np.mean(valid))
        mcse = float(np.std(valid, ddof=1) / np.sqrt(len(valid)))
        rows.append({
            "days_scored": h,
            "auc_flipped_mean": mean_flip,
            "auc_flipped_mcse": mcse,
            "auc_rate_so_far_mean": 1.0 - mean_flip,
            "mean_flipped_event_days": event_score_sum[h] / event_n[h],
            "mean_flipped_eventfree_days": free_score_sum[h] / free_n[h],
            "replications": len(valid),
        })
    return pd.DataFrame(rows)


def history_features(y: np.ndarray) -> np.ndarray:
    """Expanding exclusive rate, exactly as history_bias.py's expanding feature."""
    prior = np.concatenate([np.zeros((y.shape[0], 1)), np.cumsum(y, axis=1)[:, :-1]], axis=1)
    return prior / np.maximum(np.arange(y.shape[1], dtype=float), 1.0)[None, :]


def run_generic_curve() -> tuple[pd.DataFrame, bool, list[dict[str, float]]]:
    # Keep the reference script's generator, cell order, and one draw per full cell.
    rng = np.random.default_rng(GENERIC_SEED)
    rows: list[dict[str, float]] = []
    checks = []
    for days in GENERIC_DAYS:
        for p in EVENT_RATES:
            y = (rng.random((GENERIC_PATIENTS, days)) < p).astype(np.uint8)
            rate = history_features(y)
            y_scored = y[:, RUN_IN_DAYS:]
            rate_scored = rate[:, RUN_IN_DAYS:]
            auc_rate = row_auc(y_scored, rate_scored)
            auc_flip = row_auc(y_scored, 1.0 - rate_scored)
            valid = np.isfinite(auc_rate)
            mean_rate = float(np.mean(auc_rate[valid]))
            mean_flip = float(np.mean(auc_flip[valid]))
            mcse = float(np.std(auc_rate[valid], ddof=1) / np.sqrt(np.sum(valid)))
            rows.append({
                "record_days": days,
                "event_rate": p,
                "auc_rate_so_far": mean_rate,
                "auc_flipped": mean_flip,
                "mcse": mcse,
            })
            checks.append({"T": days, "p": p, "feature": "expanding", "within_auc": round(mean_rate, 4)})

    ref_path = ROOT / "results" / "history_bias_grid.json"
    reference = json.loads(ref_path.read_text(encoding="utf-8"))
    ref_expanding = {(r["T"], r["p"]): r["within_auc"] for r in reference if r["feature"] == "expanding"}
    all_match = len(ref_expanding) == len(checks) and all(
        ref_expanding.get((r["T"], r["p"])) == r["within_auc"] for r in checks
    )
    return pd.DataFrame(rows), all_match, checks


def dials_cell(k: int, t: int, target_reps: int, seed_cell: int) -> tuple[float, float, float, float, int, int]:
    """Return means and MCSEs for one (K,T), accumulating independent batches."""
    rng = np.random.default_rng(np.random.SeedSequence((DIALS_SEED, seed_cell)))
    share_values: list[np.ndarray] = []
    auc_values: list[np.ndarray] = []
    reps_done = 0
    scored_days = t - 1  # day 1 is the unscored initial observation
    while reps_done < target_reps:
        b = min(DIAL_BATCH, target_reps - reps_done)
        y_all = (rng.random((b, k, t)) < 0.30).astype(np.uint8)
        y = y_all[:, :, 1:]
        prior = (np.cumsum(y_all, axis=2) - y_all)[:, :, 1:]
        rate = prior / np.arange(1, t, dtype=float)[None, None, :]
        aucs = row_auc(y, rate)
        eligible = np.isfinite(aucs)
        auc_rep = np.divide(
            np.nansum(aucs, axis=1), eligible.sum(axis=1),
            out=np.full(b, np.nan, dtype=float), where=eligible.sum(axis=1) > 0,
        )
        n1 = np.sum(y, axis=2)
        n0 = scored_days - n1
        within = np.sum(n1 * n0, axis=1)
        pooled = np.sum(n1, axis=1) * np.sum(n0, axis=1)
        share_rep = np.divide(within, pooled, out=np.full(b, np.nan), where=pooled > 0)
        auc_values.append(auc_rep)
        share_values.append(share_rep)
        reps_done += b

    shares = np.concatenate(share_values)
    aucs = np.concatenate(auc_values)
    finite_s = shares[np.isfinite(shares)]
    finite_a = aucs[np.isfinite(aucs)]
    share_mean = float(np.mean(finite_s))
    share_mcse = float(np.std(finite_s, ddof=1) / np.sqrt(len(finite_s)))
    auc_mean = float(np.mean(finite_a))
    auc_mcse = float(np.std(finite_a, ddof=1) / np.sqrt(len(finite_a)))
    return share_mean, share_mcse, auc_mean, auc_mcse, len(finite_a), len(finite_s)


def run_dials() -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    cell_number = 0
    for k in DIAL_K:
        for t in DIAL_T:
            nrep = DIAL_REPS_INITIAL
            while True:
                share, share_se, auc_mean, auc_se, auc_n, share_n = dials_cell(k, t, nrep, cell_number)
                if share_se < 0.005 and auc_se < 0.005:
                    break
                nrep += DIAL_REPS_INCREMENT
                if nrep > DIAL_REPS_MAX:
                    raise RuntimeError(
                        f"Could not meet MCSE < 0.005 for K={k}, T={t}; "
                        f"share MCSE={share_se:.6g}, AUC MCSE={auc_se:.6g}"
                    )
            rows.append({
                "K": k,
                "T": t,
                "share_within_patient_pairs": share,
                "one_over_K": 1.0 / k,
                "auc_rate_so_far_mean": auc_mean,
                "auc_rate_so_far_mcse": auc_se,
                "replications": nrep,
            })
            print(f"A3 K={k}, T={t}: reps={nrep}, share MCSE={share_se:.4g}, AUC MCSE={auc_se:.4g}", flush=True)
            cell_number += 1
    return pd.DataFrame(rows)


def main() -> None:
    check_auc_definition()
    OUT.mkdir(parents=True, exist_ok=True)
    clara_auc = write_clara()
    print(f"A2a Clara rate-so-far AUC (days 2-10; day 1 unscored): {clara_auc:.6f}; flipped AUC: {1.0-clara_auc:.6f}")

    flip = run_flip_curve()
    flip.to_csv(OUT / "A2b_flip_curve.csv", index=False, float_format="%.10g")
    print(f"A2b: {FLIP_REPS} records simulated per horizon; eligible AUC replications are listed in the CSV; seed={FLIP_SEED}.")

    generic, matches, checks = run_generic_curve()
    generic.to_csv(OUT / "A2c_flip_generic.csv", index=False, float_format="%.10g")
    print("A2c rate-so-far values match history_bias_grid.json to four decimals: " + ("yes" if matches else "NO"))
    if not matches:
        reference = json.loads((ROOT / "results" / "history_bias_grid.json").read_text(encoding="utf-8"))
        ref = {(r["T"], r["p"]): r["within_auc"] for r in reference if r["feature"] == "expanding"}
        for r in checks:
            expected = ref.get((r["T"], r["p"]))
            if expected != r["within_auc"]:
                print(f"  mismatch T={r['T']}, p={r['p']}: {r['within_auc']} vs {expected}")

    dials = run_dials()
    dials.to_csv(OUT / "A3_dials.csv", index=False, float_format="%.10g")
    print(f"A3: day 1 initializes the history and is unscored; scored days are 2..T; seed={DIALS_SEED}.")
    print(f"Wrote A2a/A2b/A2c/A3 CSVs under {OUT}")


if __name__ == "__main__":
    main()
