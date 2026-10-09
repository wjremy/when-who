"""Three synthetic feature-importance designs for the methods paper.

Design A reproduces the noisy-trait experiment in importance_2x2.py, including
its RNG stream, and compares its tau=1.5 results with importance_2x2.csv.
Design B uses the same simulated patients, outcomes, and daily features as A,
but removes the daily measurement noise from both trait features. Design C is
an interaction design in which a patient-level binary trait selects which
daily feature predicts events.

Requires NumPy, pandas, and scikit-learn for a full run. ``--self-test`` only
uses NumPy and pandas. No patient data are read or used.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


TAU = 1.5
TAUS_TO_RUN = (0.5, 1.0, TAU)  # Keep A's reference random stream unchanged.
N_TRAIN_PATIENTS = 300
N_TEST_PATIENTS = 200
DAYS_PER_PATIENT = 200
N_PERMUTATION_REPEATS = 5
FEATURES_AB = (
    "trait_A", "trait_B", "trigger_1 (b=0.6)", "trigger_2 (b=0.4)",
    "trigger_3 (b=0.2)",
)
FEATURES_C = ("g", "z1", "z2")
SCHEMES = ("across_all_days", "within_patient")


def _expit(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function using NumPy only."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    nonnegative = x >= 0
    out[nonnegative] = 1.0 / (1.0 + np.exp(-x[nonnegative]))
    z = np.exp(x[~nonnegative])
    out[~nonnegative] = z / (1.0 + z)
    return out


def simulate_ab_pair(
    rng: np.random.Generator, k: int, t: int, tau: float, mu: float = 0.15
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return noisy and constant-trait X matrices with shared y and pid.

    Draw order exactly matches ``sim_traits`` in importance_2x2.py. Daily
    noise is still drawn in design B, then omitted, so the trigger features,
    outcomes, and the next RNG state remain matched to design A.
    """
    alpha = math.log(mu / (1.0 - mu)) + rng.normal(0, tau, k)
    za = (alpha - alpha.mean()) / alpha.std()
    pid = np.repeat(np.arange(k), t)

    patient_a = 0.9 * za + 0.45 * rng.normal(0, 1, k)
    daily_a = 0.3 * rng.normal(0, 1, k * t)
    patient_b = 0.7 * za + 0.7 * rng.normal(0, 1, k)
    daily_b = 0.3 * rng.normal(0, 1, k * t)
    trait_constant = np.column_stack(
        [np.repeat(patient_a, t), np.repeat(patient_b, t)]
    )
    trait_noisy = trait_constant + np.column_stack([daily_a, daily_b])

    trig = rng.normal(0, 1, (k * t, 3))
    beta = np.array([0.6, 0.4, 0.2])
    eta = np.repeat(alpha, t) + trig @ beta
    y = (rng.random(k * t) < _expit(eta)).astype(int)
    return (
        np.column_stack([trait_noisy, trig]),
        np.column_stack([trait_constant, trig]),
        y,
        pid,
    )


def simulate_interaction(
    rng: np.random.Generator, k: int, t: int, tau: float, mu: float = 0.15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate g-selected daily effects, with g independent of alpha."""
    alpha = math.log(mu / (1.0 - mu)) + rng.normal(0, tau, k)
    # A single independent patient-level Bernoulli trait, repeated daily.
    g_patient = rng.integers(0, 2, size=k)
    g = np.repeat(g_patient, t)
    z = rng.normal(0, 1, (k * t, 2))
    pid = np.repeat(np.arange(k), t)
    eta = np.repeat(alpha, t) + 0.8 * np.where(g == 1, z[:, 0], z[:, 1])
    y = (rng.random(k * t) < _expit(eta)).astype(int)
    return np.column_stack([g, z]), y, pid


def permute_feature(
    x: np.ndarray, j: int, pid: np.ndarray, scheme: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Copy X and permute one column across all days or within each patient."""
    xp = x.copy()
    if scheme == "across_all_days":
        xp[:, j] = rng.permutation(xp[:, j])
    elif scheme == "within_patient":
        for patient in np.unique(pid):
            idx = np.flatnonzero(pid == patient)
            xp[idx, j] = rng.permutation(xp[idx, j])
    else:
        raise ValueError(f"Unknown permutation scheme: {scheme}")
    return xp


def within_patient_auc_values(
    y: np.ndarray, score: np.ndarray, pid: np.ndarray, roc_auc_score
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid patient AUCs and their patient IDs, in sorted order."""
    values, patients = [], []
    for patient in np.unique(pid):
        idx = np.flatnonzero(pid == patient)
        yk = y[idx]
        if 0 < int(yk.sum()) < len(yk):
            values.append(float(roc_auc_score(yk, score[idx])))
            patients.append(patient)
    return np.asarray(values, dtype=float), np.asarray(patients)


def _mean_se(values: list[float] | np.ndarray) -> tuple[float, float]:
    """Mean and sample standard error; one observation has SE zero."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    se = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return float(np.mean(arr)), se


def _permutation_rows(
    design: str, x: np.ndarray, y: np.ndarray, pid: np.ndarray, model,
    roc_auc_score, rng: np.random.Generator, feature_names: tuple[str, ...],
) -> list[dict]:
    """Compute permutation drops for both AUC summaries and both schemes."""
    base_score = model.predict_proba(x)[:, 1]
    base_pooled = float(roc_auc_score(y, base_score))
    base_patient_values, base_patients = within_patient_auc_values(
        y, base_score, pid, roc_auc_score
    )
    base_within = float(np.mean(base_patient_values))
    base_by_patient = dict(zip(base_patients.tolist(), base_patient_values.tolist()))
    rows = []

    for j, feature in enumerate(feature_names):
        drops = {(scheme, metric): [] for scheme in SCHEMES
                 for metric in ("pooled_auc", "mean_within_patient_auc")}
        for _ in range(N_PERMUTATION_REPEATS):
            for scheme in SCHEMES:
                xp = permute_feature(x, j, pid, scheme, rng)
                score = model.predict_proba(xp)[:, 1]
                drops[(scheme, "pooled_auc")].append(
                    base_pooled - float(roc_auc_score(y, score))
                )
                perm_values, perm_patients = within_patient_auc_values(
                    y, score, pid, roc_auc_score
                )
                perm_by_patient = dict(zip(perm_patients.tolist(), perm_values.tolist()))
                common = [p for p in base_by_patient if p in perm_by_patient]
                # All AUC-valid patients are determined by y, so these match.
                drop = base_within - float(np.mean([perm_by_patient[p] for p in common]))
                drops[(scheme, "mean_within_patient_auc")].append(drop)

        for (scheme, metric), vals in drops.items():
            estimate, se = _mean_se(vals)
            rows.append({
                "design": design,
                "tau": TAU,
                "result_type": "permutation_drop",
                "feature": feature,
                "permutation_scheme": scheme,
                "evaluation": metric,
                "estimate": estimate,
                "standard_error": se,
                "n_repeats": N_PERMUTATION_REPEATS,
                "se_basis": "permutation_repeats",
            })
    return rows


def _ablation_rows(y: np.ndarray, pid: np.ndarray, full_score: np.ndarray,
                   no_g_score: np.ndarray, roc_auc_score) -> list[dict]:
    """Paired within-patient AUC comparison of full and no-g models."""
    full_values, full_patients = within_patient_auc_values(y, full_score, pid, roc_auc_score)
    no_g_values, no_g_patients = within_patient_auc_values(y, no_g_score, pid, roc_auc_score)
    full_map = dict(zip(full_patients.tolist(), full_values.tolist()))
    no_g_map = dict(zip(no_g_patients.tolist(), no_g_values.tolist()))
    patients = [p for p in full_map if p in no_g_map]
    full = np.asarray([full_map[p] for p in patients], dtype=float)
    no_g = np.asarray([no_g_map[p] for p in patients], dtype=float)
    diff = full - no_g
    output = []
    for label, vals in (("full_model", full), ("model_without_g", no_g),
                        ("full_minus_without_g", diff)):
        estimate, se = _mean_se(vals)
        output.append({
            "design": "C",
            "tau": TAU,
            "result_type": "model_comparison",
            "feature": label,
            "permutation_scheme": "not_applicable",
            "evaluation": "mean_within_patient_auc",
            "estimate": estimate,
            "standard_error": se,
            "n_repeats": len(patients),
            "se_basis": "paired_test_patients",
        })
    return output


def _fit_model(x: np.ndarray, y: np.ndarray):
    """Fit the same HistGradientBoosting model used by importance_2x2.py."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, random_state=0
    ).fit(x, y)


def run_experiments() -> pd.DataFrame:
    """Run A, B, and C and return a tidy results frame."""
    from sklearn.metrics import roc_auc_score

    rng_ab = np.random.default_rng(3)
    rng_b_perm = np.random.default_rng(2301)
    rng_c = np.random.default_rng(1701)
    rows: list[dict] = []

    # Pair A and B data generation. The exact draws and fit sequence for A
    # match importance_2x2.py through tau=1.5; only tau=1.5 is reported.
    for tau in TAUS_TO_RUN:
        x_train_a, x_train_b, y_train, _ = simulate_ab_pair(
            rng_ab, N_TRAIN_PATIENTS, DAYS_PER_PATIENT, tau
        )
        x_test_a, x_test_b, y_test, pid_test = simulate_ab_pair(
            rng_ab, N_TEST_PATIENTS, DAYS_PER_PATIENT, tau
        )
        model_a = _fit_model(x_train_a, y_train)
        model_b = _fit_model(x_train_b, y_train)
        # In A the earlier tau permutation calls advance the reference RNG
        # before tau=1.5 is simulated. Keep that stream exact.
        rows_a = _permutation_rows("A", x_test_a, y_test, pid_test,
                                   model_a, roc_auc_score, rng_ab, FEATURES_AB)
        if tau == TAU:
            rows.extend(rows_a)
            rows.extend(_permutation_rows("B", x_test_b, y_test, pid_test,
                                          model_b, roc_auc_score, rng_b_perm, FEATURES_AB))

    x_train_c, y_train_c, _ = simulate_interaction(
        rng_c, N_TRAIN_PATIENTS, DAYS_PER_PATIENT, TAU
    )
    x_test_c, y_test_c, pid_test_c = simulate_interaction(
        rng_c, N_TEST_PATIENTS, DAYS_PER_PATIENT, TAU
    )
    model_c = _fit_model(x_train_c, y_train_c)
    rows.extend(_permutation_rows("C", x_test_c, y_test_c, pid_test_c,
                                  model_c, roc_auc_score, rng_c, FEATURES_C))
    no_g_model = _fit_model(x_train_c[:, 1:], y_train_c)
    rows.extend(_ablation_rows(
        y_test_c, pid_test_c, model_c.predict_proba(x_test_c)[:, 1],
        no_g_model.predict_proba(x_test_c[:, 1:])[:, 1], roc_auc_score
    ))

    result = pd.DataFrame.from_records(rows)
    try:
        check_against_reference(result)
    except Exception as exc:  # the reference check must not discard the results
        print(f"A reference check failed to run: {type(exc).__name__}: {exc}")
    return result


def check_against_reference(result: pd.DataFrame) -> bool:
    """Check design A against all corresponding rows in the saved v1 CSV."""
    path = Path(__file__).resolve().parent / "importance_2x2.csv"
    if not path.exists():
        print(f"A reference check skipped: {path.name} not found")
        return False
    ref = pd.read_csv(path)
    tau_col = next((c for c in ref.columns if c.lower() == "tau"), None)
    if tau_col is not None:
        ref = ref.loc[np.isclose(ref[tau_col], TAU)]
    actual = result.loc[(result["design"] == "A")
                        & (result["result_type"] == "permutation_drop")]
    keys = ["feature", "permutation_scheme", "evaluation"]
    merged = actual.merge(ref, on=keys, suffixes=("_new", "_old"), validate="one_to_one")
    if len(merged) != len(ref):
        print(f"A reference check: CHECK ({len(merged)}/{len(ref)} rows matched)")
        return False
    mean_ok = np.allclose(merged["estimate"], merged["mean_drop"], atol=1e-7, rtol=0)
    se_ok = np.allclose(merged["standard_error_new"], merged["standard_error_old"],
                        atol=1e-7, rtol=0)
    passed = bool(mean_ok and se_ok)
    print(f"A vs importance_2x2.csv: {'PASS' if passed else 'CHECK'}")
    return passed


def self_test() -> None:
    """Unit-test NumPy/pandas helpers without importing scikit-learn."""
    rng = np.random.default_rng(3)
    xa, xb, y, pid = simulate_ab_pair(rng, 8, 20, TAU)
    assert xa.shape == xb.shape == (160, 5)
    assert y.shape == pid.shape == (160,)
    assert set(np.unique(y)).issubset({0, 1})
    assert np.array_equal(xb[:, 0], np.repeat(xb[::20, 0], 20))
    assert np.array_equal(xb[:, 1], np.repeat(xb[::20, 1], 20))
    assert np.array_equal(xa[:, 2:], xb[:, 2:])
    assert np.array_equal(_expit(np.array([-1000.0, 0.0, 1000.0])),
                          np.array([0.0, 0.5, 1.0]))

    x, yc, pidc = simulate_interaction(np.random.default_rng(4), 8, 20, TAU)
    assert x.shape == (160, 3)
    assert np.array_equal(x[:, 0], np.repeat(x[::20, 0], 20))
    # g and alpha are generated by independent RNG draws; the helper is
    # deterministic and g has exactly one binary value for each patient.
    assert set(np.unique(x[:, 0])).issubset({0.0, 1.0})
    assert yc.shape == pidc.shape == (160,)

    original = xa.copy()
    across = permute_feature(xa, 0, pid, "across_all_days", np.random.default_rng(5))
    within = permute_feature(xa, 0, pid, "within_patient", np.random.default_rng(6))
    assert np.array_equal(np.sort(across[:, 0]), np.sort(original[:, 0]))
    assert np.array_equal(across[:, 1:], original[:, 1:])
    for patient in np.unique(pid):
        idx = np.flatnonzero(pid == patient)
        assert np.array_equal(np.sort(within[idx, 0]), np.sort(original[idx, 0]))
    assert np.array_equal(within[:, 1:], original[:, 1:])
    try:
        permute_feature(xa, 0, pid, "invalid", np.random.default_rng(0))
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid scheme was not rejected")

    assert _mean_se([2.0, 4.0]) == (3.0, 1.0)
    assert _mean_se([2.0]) == (2.0, 0.0)
    print("NumPy/pandas helper checks passed (A/B pairing, C shapes, permutations, mean/SE).")


def print_conclusions(result: pd.DataFrame) -> None:
    """Print one concise numeric conclusion for each design."""
    p = result[(result.result_type == "permutation_drop")]

    def value(design: str, feature: str, scheme: str, metric: str) -> float:
        q = p[(p.design == design) & (p.feature == feature)
              & (p.permutation_scheme == scheme) & (p.evaluation == metric)]
        return float(q.estimate.iloc[0])

    pooled = "pooled_auc"
    within = "mean_within_patient_auc"
    print(
        "A (noisy traits): trait_A across-day drops were "
        f"{value('A', 'trait_A', 'across_all_days', pooled):.3f} pooled and "
        f"{value('A', 'trait_A', 'across_all_days', within):.3f} within-patient; "
        "within-patient shuffle drops were "
        f"{value('A', 'trait_A', 'within_patient', pooled):.3f} and "
        f"{value('A', 'trait_A', 'within_patient', within):.3f}."
    )
    print(
        "B (constant traits): trait_A across-day drops were "
        f"{value('B', 'trait_A', 'across_all_days', pooled):.3f} pooled and "
        f"{value('B', 'trait_A', 'across_all_days', within):.3f} within-patient; "
        "within-patient shuffle drops were "
        f"{value('B', 'trait_A', 'within_patient', pooled):.3f} and "
        f"{value('B', 'trait_A', 'within_patient', within):.3f}."
    )
    ab = result[(result.design == "C") & (result.result_type == "model_comparison")]
    full = float(ab.loc[ab.feature == "full_model", "estimate"].iloc[0])
    no_g = float(ab.loc[ab.feature == "model_without_g", "estimate"].iloc[0])
    diff = float(ab.loc[ab.feature == "full_minus_without_g", "estimate"].iloc[0])
    print(f"C (interaction): within-patient AUC was {full:.3f} full vs {no_g:.3f} without g (difference {diff:+.3f}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true",
                        help="run NumPy/pandas helper checks only")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    result = run_experiments()
    output = Path(__file__).resolve().parents[1] / "results" / "importance_v2.csv"
    result.to_csv(output, index=False, float_format="%.8f")
    print("Permutation drops and model comparison:")
    print(result.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"\nWrote {output}")
    print_conclusions(result)


if __name__ == "__main__":
    main()
