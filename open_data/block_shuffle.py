#!/usr/bin/env python3
"""Model-fixed block-shuffle sensitivity analysis for per-patient AUC.

The fitted models, data loaders, seeds, feature builders, and history rebuilders
are reused from open_data/cases_model_fixed_permutation.py. Block lengths count
sorted scored rows: CTN-0094 uses 4 or 8 scored study weeks; REPLACE-BG uses 7
or 14 scored nights (night_index); the College Experience Study uses 7 or 14
date-ordered scored report rows (reports are nominally daily, with gaps between
complete reports retained in the ordering). The final block for each patient
may be shorter. Full shuffles are also run as the reference and use the
original generator and seed stream.

For each fitted model, shuffled AUCs are rescored after the existing cohort
callback rebuilds outcome-history inputs. The model itself stays fixed.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "open_data" / "results" / "v16"
N_PERM = 200
QUICK_N_PERM = 5

COHORTS = {
    "replace_bg": {
        "time_column": "night_index",
        "time_unit": "scored night (night_index; daily/nightly steps)",
        "block_lengths": (7, 14),
    },
    "ctn0094": {
        "time_column": "study_week",
        "time_unit": "observed scored study week",
        "block_lengths": (4, 8),
    },
    "ces": {
        "time_column": "date",
        "time_unit": "scored complete-report row (date-ordered; nominally daily)",
        "block_lengths": (7, 14),
    },
}


def group_indices(pid):
    """Return stable patient groups as (identifier, row indices)."""
    pid = np.asarray(pid)
    return [(key, np.flatnonzero(pid == key)) for key in pd.unique(pid)]


def generate_block_permuted_targets(y, pid, time, block_len, n_perm, rng):
    """Permute outcomes within patient and consecutive time-sorted row blocks.

    ``time`` is used only to order each patient's scored rows. Blocks are made
    from that order in chunks of ``block_len`` rows, with a shorter final chunk
    when needed. The returned integer array has shape ``(n_perm, len(y))``, as
    does the full-shuffle generator in the existing analysis.
    """
    y = np.asarray(y, dtype=int)
    pid = np.asarray(pid)
    time = np.asarray(time)
    if y.ndim != 1 or pid.ndim != 1 or time.ndim != 1:
        raise ValueError("y, pid, and time must be one-dimensional")
    if len(y) != len(pid) or len(y) != len(time):
        raise ValueError("y, pid, and time must have equal lengths")
    if not isinstance(block_len, (int, np.integer)) or block_len < 1:
        raise ValueError("block_len must be a positive integer")
    if not isinstance(n_perm, (int, np.integer)) or n_perm < 0:
        raise ValueError("n_perm must be a nonnegative integer")

    out = np.empty((int(n_perm), len(y)), dtype=int)
    for _, ix in group_indices(pid):
        ordered = ix[np.argsort(time[ix], kind="mergesort")]
        for start in range(0, len(ordered), int(block_len)):
            block = ordered[start:start + int(block_len)]
            for b in range(int(n_perm)):
                out[b, block] = rng.permutation(y[block])
    return out


def summarize_shuffle(observed_auc, permutation_auc):
    """Summarize fixed-model per-patient AUCs and one-sided empirical p-values."""
    observed_auc = np.asarray(observed_auc, dtype=float)
    permutation_auc = np.asarray(permutation_auc, dtype=float)
    if permutation_auc.ndim != 2 or permutation_auc.shape[1] != len(observed_auc):
        raise ValueError("Permutation AUC matrix does not match observed patients")
    if permutation_auc.shape[0] < 1:
        raise ValueError("At least one permutation is required")

    patient_baseline = np.mean(permutation_auc, axis=0)
    patient_excess = observed_auc - patient_baseline
    # Inclusive upper tail with the plus-one correction; share uses strict p < .05.
    patient_p = (1 + np.sum(permutation_auc >= observed_auc[None, :], axis=0)) / (
        permutation_auc.shape[0] + 1
    )
    q1, median, q3 = np.percentile(patient_excess, [25, 50, 75])
    return {
        "n_patients": int(len(observed_auc)),
        "n_permutations": int(permutation_auc.shape[0]),
        "mean_per_patient_auc": float(np.mean(observed_auc)),
        "mean_shuffled_baseline": float(np.mean(patient_baseline)),
        "mean_excess": float(np.mean(patient_excess)),
        "median_excess": float(median),
        "excess_iqr": [float(q1), float(q3)],
        "share_permutation_p_below_0_05": float(np.mean(patient_p < 0.05)),
        "permutation_p_value_convention": "(1 + count(shuffled patient AUC >= observed patient AUC)) / (N_PERM + 1); one-sided upper tail",
    }


def _load_analysis_module():
    """Import the existing fit/load/rebuild code only for a real run."""
    open_data = ROOT / "open_data"
    sys.path.insert(0, str(open_data))
    return importlib.import_module("cases_model_fixed_permutation")


def _full_shuffle_check(row):
    """Check the published full-shuffle CTN-0094 gradient-boosting summary."""
    actual = row
    observed = (
        round(actual["mean_per_patient_auc"], 2),
        round(actual["mean_shuffled_baseline"], 2),
        round(actual["median_excess"], 2),
        round(actual["excess_iqr"][0], 2),
        round(actual["excess_iqr"][1], 2),
        int(round(100 * actual["share_permutation_p_below_0_05"])),
    )
    expected = (0.64, 0.40, 0.26, 0.05, 0.46, 36)
    if observed != expected:
        raise AssertionError(
            "CTN-0094 gradient-boosting full-shuffle check failed: "
            f"rounded values {observed}, expected {expected}"
        )
    return {
        "passed": True,
        "rounded_observed": list(observed),
        "expected": list(expected),
    }


def build_markdown(result):
    lines = [
        "# Block-shuffle model-fixed permutation results",
        "",
        f"Seed: `{result['metadata']['seed']}`; permutations per shuffle: `{result['metadata']['permutations']}`.",
        "",
        "Blocks contain consecutive scored rows after sorting by the cohort time index. The full-shuffle rows use the original analysis generator. Each patient's baseline is its mean shuffled AUC; excess is observed AUC minus that baseline. The p-value share uses the one-sided plus-one convention described in the JSON.",
        "",
        "| Cohort | Time unit | Model | Shuffle | Patients | Mean patient AUC | Mean shuffled baseline | Mean excess | Median excess [Q1, Q3] | Patients with p < 0.05 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cohort_key, cohort in result["cohorts"].items():
        for model_name, model in cohort["models"].items():
            for shuffle_name, summary in model["shuffles"].items():
                iqr = summary["excess_iqr"]
                lines.append(
                    f"| {cohort['label']} | {cohort['time_unit']} | {model_name.replace('_', ' ').title()} | "
                    f"{shuffle_name} | {summary['n_patients']} | {summary['mean_per_patient_auc']:.3f} | "
                    f"{summary['mean_shuffled_baseline']:.3f} | {summary['mean_excess']:.3f} | "
                    f"{summary['median_excess']:.3f} [{iqr[0]:.3f}, {iqr[1]:.3f}] | "
                    f"{summary['share_permutation_p_below_0_05']:.1%} |"
                )
    lines += [
        "",
        "Full-shuffle reproduction check (CTN-0094 gradient boosting): "
        + ("passed." if (result["metadata"].get("ctn_full_shuffle_check") or {}).get("passed") else "not run (quick mode)."),
        "",
        "For every shuffle, outcomes are assigned back to their original scored rows. The existing cohort-specific callback then rebuilds outcome-history inputs from those outcomes while keeping run-in/origin inputs and all non-history inputs fixed; no model is refit.",
        "",
    ]
    return "\n".join(lines)


def run(quick=False):
    analysis = _load_analysis_module()
    n_perm = QUICK_N_PERM if quick else N_PERM
    existing_path = analysis.ROOT / "cases_extra_results.json"
    existing = json.loads(existing_path.read_text(encoding="utf-8"))
    prepared = {
        "replace_bg": analysis.fit_replace_bg(),
        "ctn0094": analysis.fit_ctn(),
        "ces": analysis.fit_ces(),
    }

    # This stream is advanced exactly once per cohort, in the original order,
    # by the original full-shuffle generator. Separate block draws cannot alter it.
    rng_full = np.random.default_rng(analysis.SEED)
    rng_blocks = np.random.default_rng(analysis.SEED)
    output = {
        "metadata": {
            "seed": analysis.SEED,
            "permutations": n_perm,
            "quick": bool(quick),
            "model_fit_policy": "Reuse the existing fit functions, splits, model seeds, preprocessing, and fitted estimators; fit once and keep models fixed for all shuffles.",
            "p_value_convention": "One-sided upper-tail empirical p=(1 + count(shuffled patient AUC >= observed patient AUC))/(N_PERM + 1); the reported share is the fraction strictly below 0.05.",
            "block_definition": "Within each patient, sort scored rows by the cohort time index and chunk into consecutive blocks of the specified scored-row length; the last block may be shorter. Outcomes are shuffled only within each block.",
            "history_rebuild": "Each shuffled target vector is passed to the existing score_permutation callback, which invokes the unchanged cohort-specific history rebuild function and rescoring pipeline with fixed fitted estimators.",
            "history_rebuild_functions": {
                "replace_bg": "replace_build_histories",
                "ctn0094": "ctn_rebuild_histories",
                "ces": "ces_rebuild_histories",
            },
            "shuffle_units": {
                key: {
                    "time_column": spec["time_column"],
                    "time_unit": spec["time_unit"],
                    "block_lengths": list(spec["block_lengths"]),
                }
                for key, spec in COHORTS.items()
            },
            "ctn_full_shuffle_check": None,
        },
        "cohorts": {},
    }

    for cohort_key in ("replace_bg", "ctn0094", "ces"):
        bundle = prepared[cohort_key]
        spec = COHORTS[cohort_key]
        checks = analysis.verify_against_existing(bundle, cohort_key, existing)
        time_values = bundle["test"][spec["time_column"]].to_numpy()
        shuffle_targets = [("full", None)] + [
            (f"block_{length}", length) for length in spec["block_lengths"]
        ]
        cohort_result = {
            "label": analysis.COHORTS[cohort_key]["label"],
            "time_column": spec["time_column"],
            "time_unit": spec["time_unit"],
            "test_patients": int(pd.unique(bundle["pid"]).size),
            "test_scored_rows": int(len(bundle["y"])),
            "selected_gb_iterations": bundle["selected_gb_iterations"],
            "observed_reproduction_checks": checks,
            "models": {},
        }

        for shuffle_name, block_len in shuffle_targets:
            if block_len is None:
                targets = analysis.generate_permuted_targets(
                    bundle["y"], bundle["pid"], n_perm, rng_full
                )
            else:
                targets = generate_block_permuted_targets(
                    bundle["y"], bundle["pid"], time_values, block_len,
                    n_perm, rng_blocks,
                )
            rescored = analysis.feature_auc_matrix(
                bundle["y"], targets, bundle["pid"], bundle["predictions"],
                bundle["score_permutation"],
            )
            for model_name, model_data in rescored.items():
                model_entry = cohort_result["models"].setdefault(
                    model_name, {"eligible_patients": len(model_data["observed_ids"]), "shuffles": {}}
                )
                model_entry["shuffles"][shuffle_name] = summarize_shuffle(
                    model_data["observed_auc"], model_data["permutation_auc"]
                )
        output["cohorts"][cohort_key] = cohort_result

    if not quick:
        full_ctn_gb = output["cohorts"]["ctn0094"]["models"]["gradient_boosting"]["shuffles"]["full"]
        output["metadata"]["ctn_full_shuffle_check"] = _full_shuffle_check(full_ctn_gb)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "block_results.json"
    md_path = OUTPUT_DIR / "block_results.md"
    json_path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    md_path.write_text(build_markdown(output), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    if not quick:
        print("CTN-0094 gradient-boosting full-shuffle reference check passed.")
    for cohort_key, cohort in output["cohorts"].items():
        for model_name, model in cohort["models"].items():
            for shuffle_name, summary in model["shuffles"].items():
                print(
                    f"{cohort_key} {model_name} {shuffle_name}: "
                    f"mean AUC={summary['mean_per_patient_auc']:.3f}, "
                    f"baseline={summary['mean_shuffled_baseline']:.3f}, "
                    f"median excess={summary['median_excess']:.3f}, "
                    f"p<.05 share={summary['share_permutation_p_below_0_05']:.1%}, "
                    f"n={summary['n_patients']}"
                )


def _unrestricted_sorted_shuffle(y, pid, time, n_perm, rng):
    """Self-test helper: full within-patient shuffle in sorted row order."""
    y = np.asarray(y, dtype=int)
    pid = np.asarray(pid)
    time = np.asarray(time)
    out = np.empty((n_perm, len(y)), dtype=int)
    for _, ix in group_indices(pid):
        ordered = ix[np.argsort(time[ix], kind="mergesort")]
        for b in range(n_perm):
            out[b, ordered] = rng.permutation(y[ordered])
    return out


def helper_selftest():
    """Test block boundaries and the unrestricted-shuffle special case."""
    # Deliberately unsorted input verifies ordering by the supplied time index.
    y = np.asarray([1, 0, 1, 0, 0, 1, 1, 0, 1])
    pid = np.asarray(["a", "a", "a", "a", "b", "b", "b", "b", "b"])
    time = np.asarray([3, 1, 4, 2, 12, 10, 14, 11, 13])
    block_len = 2
    n_perm = 60
    perms = generate_block_permuted_targets(
        y, pid, time, block_len, n_perm, np.random.default_rng(41)
    )
    for key, ix in group_indices(pid):
        ordered = ix[np.argsort(time[ix], kind="mergesort")]
        if not np.all(perms[:, ix].sum(axis=1) == y[ix].sum()):
            raise AssertionError(f"Patient total event count changed for {key}")
        for start in range(0, len(ordered), block_len):
            block = ordered[start:start + block_len]
            original = np.sort(y[block])
            for b in range(n_perm):
                if not np.array_equal(np.sort(perms[b, block]), original):
                    raise AssertionError(f"Outcome escaped patient/block for {key}, block {start}")

    # With one block per patient, block shuffle is the same unrestricted
    # within-patient permutation distribution; identical streams give identical draws.
    large_block = max(np.bincount(pd.factorize(pid)[0]))
    block_full = generate_block_permuted_targets(
        y, pid, time, int(large_block), n_perm, np.random.default_rng(81)
    )
    unrestricted = _unrestricted_sorted_shuffle(
        y, pid, time, n_perm, np.random.default_rng(81)
    )
    if not np.array_equal(block_full, unrestricted):
        raise AssertionError("A block at least as long as each record differs from full shuffle")
    print("Block-shuffle selftest passed (patient/block counts and full-shuffle limit).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use 5 permutations instead of 200")
    parser.add_argument("--selftest", action="store_true", help="Run the NumPy/pandas-only permutation selftest")
    args = parser.parse_args()
    if args.selftest:
        helper_selftest()
    else:
        run(quick=args.quick)


if __name__ == "__main__":
    main()
