"""Regression tests for leakage, scored-period handling, checklist quantities, and release metadata."""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from withinpatient import (
    auc,
    evaluate,
    expanding_patient_rate,
    pooled_auc_permutation_test,
    population_a0_moment_corrected,
)


def small_frame():
    rows = []
    for patient in range(8):
        for time in range(6):
            y = (patient + time) % 2
            rows.append((patient, time, y, 0.15 + 0.1 * ((patient * 3 + time) % 7)))
    return pd.DataFrame(rows, columns=["patient", "time", "y", "forecast"])


def running_rate(outcomes):
    outcomes = np.asarray(outcomes, dtype=float)
    earlier = np.r_[0.0, np.cumsum(outcomes[:-1])]
    return (0.25 + earlier) / (1.0 + np.arange(len(outcomes)))


class TestLeakageAndRunin(unittest.TestCase):
    def test_expanding_rate_never_uses_current_or_later_outcomes(self):
        patient = np.array(["a"] * 5 + ["b"] * 4)
        time = np.array([0, 1, 1, 2, 3, 0, 2, 2, 4])
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1])
        reference = expanding_patient_rate(patient, y, time)
        self.assertTrue(np.isnan(reference[0]))
        self.assertEqual(reference[1], 1.0)
        self.assertEqual(reference[2], 1.0)  # same-time row is not earlier
        changed = y.copy()
        changed[time >= 3] = 1 - changed[time >= 3]
        changed_reference = expanding_patient_rate(patient, changed, time)
        self.assertTrue(np.array_equal(reference[time <= 3], changed_reference[time <= 3], equal_nan=True))
        prior = expanding_patient_rate(["a", "a", "a"], [1, 0, 1], [0, 1, 2], prior_rate=0.25, prior_weight=2)
        np.testing.assert_allclose(prior, [0.25, 0.5, 0.375])

    def test_evaluate_expanding_skill_has_no_future_seed_and_skips_undefined_rows(self):
        frame = small_frame()
        base = evaluate(frame, time="time", n_perm=9, return_references=True)
        changed = frame.copy()
        changed.loc[changed.time >= 3, "y"] = 1 - changed.loc[changed.time >= 3, "y"]
        again = evaluate(changed, time="time", n_perm=9, return_references=True)
        unaffected = frame.time.to_numpy() <= 3
        np.testing.assert_array_equal(
            base["references"]["expanding_patient_rate"][unaffected],
            again["references"]["expanding_patient_rate"][unaffected],
        )
        skill = base["skill"]["expanding_patient_rate"]
        self.assertEqual(skill["excluded_rows"], frame.patient.nunique())
        self.assertEqual(skill["scored_rows"], len(frame) - frame.patient.nunique())

    def test_runin_rows_are_excluded_from_scored_quantities(self):
        frame = small_frame()
        result = evaluate(frame, time="time", runin_days=2, threshold=0.4, n_perm=9, return_references=True)
        scored_only = evaluate(frame.loc[frame.time >= 2].copy(), time="time", threshold=0.4, n_perm=9)
        self.assertEqual(result["cohort"]["days"], int((frame.time >= 2).sum()))
        self.assertEqual(result["cohort"]["rows_excluded_runin"], int((frame.time < 2).sum()))
        scored = frame.loc[frame.time >= 2]
        scored_positions = np.flatnonzero(frame.time.to_numpy() >= 2)
        for patient, rows in scored.groupby("patient", sort=False):
            observed_rate = frame.loc[(frame.patient == patient) & (frame.time < 2), "y"].mean()
            aligned_position = int(np.flatnonzero(scored_positions == rows.index[0])[0])
            self.assertAlmostEqual(result["references"]["expanding_patient_rate"][aligned_position], observed_rate)
        for key in ("pooled", "within_patient"):
            self.assertAlmostEqual(result["auc"][key]["value"], scored_only["auc"][key]["value"])
        self.assertEqual(result["auc"]["per_time_point"]["per_time"], scored_only["auc"]["per_time_point"]["per_time"])
        self.assertAlmostEqual(result["auc"]["A0"], scored_only["auc"]["A0"])
        self.assertAlmostEqual(result["brier"]["pooled"]["brier"], scored_only["brier"]["pooled"]["brier"])
        n_scored = int((frame.time >= 2).sum())
        for name in ("population_rate", "retrospective_patient_rate", "runin_patient_rate", "expanding_patient_rate"):
            self.assertEqual(result["skill"][name]["scored_rows"], n_scored)
        self.assertEqual(scored_only["skill"]["expanding_patient_rate"]["scored_rows"], n_scored - frame.patient.nunique())
        self.assertAlmostEqual(result["calibration_in_the_large"]["mean"], scored_only["calibration_in_the_large"]["mean"])
        self.assertAlmostEqual(result["calibration"]["pooled_slope"], scored_only["calibration"]["pooled_slope"])
        self.assertAlmostEqual(result["calibration"]["between_slope"], scored_only["calibration"]["between_slope"])
        self.assertAlmostEqual(result["calibration"]["within_slope"], scored_only["calibration"]["within_slope"])
        self.assertEqual(result["threshold"]["alarm_counts"], scored_only["threshold"]["alarm_counts"])


class TestChecklistOutputs(unittest.TestCase):
    def test_population_a0_is_close_to_true_population_value(self):
        rng = np.random.default_rng(92)
        n_patients, n_rows = 2400, 300
        rates = rng.beta(5, 15, n_patients)
        patient = np.repeat(np.arange(n_patients), n_rows)
        y = rng.binomial(1, np.repeat(rates, n_rows))
        actual = population_a0_moment_corrected(patient, y)
        left = rng.integers(0, n_patients, size=400_000)
        right = rng.integers(0, n_patients, size=400_000)
        mu = float(np.mean(rates))
        true_value = 0.5 + float(np.mean(np.abs(rates[left] - rates[right]))) / (4 * mu * (1 - mu))
        self.assertLess(abs(actual - true_value), 0.035)

    def test_pooled_auc_forecast_shuffle_test_and_patient_references(self):
        frame = small_frame()
        patient, y, forecast = (frame[column].to_numpy() for column in ("patient", "y", "forecast"))
        out = pooled_auc_permutation_test(patient, y, forecast, n_perm=31, seed=3)
        rng = np.random.default_rng(3)
        expected_null = []
        for _ in range(31):
            shuffled = forecast.copy()
            for value in pd.unique(patient):
                ix = np.flatnonzero(patient == value)
                shuffled[ix] = forecast[ix[rng.permutation(len(ix))]]
            expected_null.append(auc(y, shuffled))
        expected_null = np.asarray(expected_null)
        expected_p = (int(np.sum(expected_null >= auc(y, forecast))) + 1) / 32
        self.assertAlmostEqual(out["null_mean"], float(np.mean(expected_null)))
        np.testing.assert_allclose(out["null_percentiles"], np.percentile(expected_null, [2.5, 97.5]))
        self.assertEqual(out["one_sided_p_value"], expected_p)
        self.assertEqual(out["permutations"], 31)
        self.assertEqual(len(out["null_percentiles"]), 2)
        self.assertGreaterEqual(out["one_sided_p_value"], 1 / 32)
        self.assertLessEqual(out["one_sided_p_value"], 1)
        self.assertEqual(set(out["per_patient"]), set(frame.patient.unique()))
        for patient_result in out["per_patient"].values():
            self.assertIn("observed", patient_result)
            self.assertIn("reference_mean", patient_result)
            self.assertEqual(len(patient_result["reference_percentiles"]), 2)

    def test_patient_mean_share_and_patient_level_runin_and_updated_skill(self):
        frame = small_frame()
        frame["forecast"] = 0.1 + 0.75 * frame["y"] + 0.01 * frame["patient"] / frame["patient"].max()
        result = evaluate(frame, time="time", runin_days=2, threshold=0.4, n_perm=11, return_references=True)
        scored = frame.loc[frame.time >= 2]
        means = scored.groupby("patient", sort=False).forecast.transform("mean")
        expected_mean_auc = auc(scored.y.to_numpy(), means.to_numpy())
        expected_share = (expected_mean_auc - 0.5) / (result["auc"]["pooled"]["value"] - 0.5)
        self.assertAlmostEqual(result["auc"]["patient_mean"]["pooled_auc"], expected_mean_auc)
        self.assertAlmostEqual(result["auc"]["patient_mean"]["share_of_pooled_auc_above_0_5"], expected_share)
        for reference_name in ("runin_patient_rate", "expanding_patient_rate"):
            summary = result["skill"][reference_name]
            self.assertIn("pooled", summary)
            self.assertIn("patient_mean", summary)
            self.assertIn("per_patient", summary)
            self.assertGreaterEqual(summary["eligible_patients"], 1)
        for patient, rows in scored.groupby("patient", sort=False):
            baseline = float(frame.loc[(frame.patient == patient) & (frame.time < 2), "y"].mean())
            expected = 1 - float(np.sum((rows.y.to_numpy() - rows.forecast.to_numpy()) ** 2)) / float(np.sum((rows.y.to_numpy() - baseline) ** 2))
            self.assertAlmostEqual(result["skill"]["runin_patient_rate"]["per_patient"][patient], expected)
        self.assertEqual(len(result["references"]["expanding_patient_rate"]), len(scored))

    def test_elapsed_time_alarm_rate_and_counts(self):
        frame = pd.DataFrame({
            "patient": ["a", "a", "a", "b", "b"],
            "time": [0, 2, 5, 10, 20],
            "y": [1, 0, 1, 0, 1],
            "forecast": [0.9, 0.1, 0.8, 0.7, 0.2],
        })
        default = evaluate(frame, time="time", threshold=0.5, n_perm=5)["threshold"]
        self.assertNotIn("alarm_counts_per_patient", default)
        self.assertNotIn("per_patient", default["alarms_per_patient_per_time_unit"])
        self.assertEqual(default["alarm_counts"]["total"], 3)
        self.assertAlmostEqual(default["alarms_per_patient_30_days"]["mean"], (2 * 30 / 3 + 1 * 30 / 2) / 2)
        from withinpatient.metrics import threshold_metrics
        threshold = threshold_metrics(frame.patient, frame.y, frame.forecast, 0.5, time=frame.time, per_patient=True)
        self.assertEqual(threshold["alarm_counts_per_patient"], {"a": 2, "b": 1})
        self.assertAlmostEqual(threshold["alarms_per_patient_per_time_unit"]["per_patient"]["a"], 2 / 5)
        self.assertAlmostEqual(threshold["alarms_per_patient_per_time_unit"]["per_patient"]["b"], 1 / 10)
        self.assertAlmostEqual(threshold["alarms_per_patient_per_time_unit"]["mean"], 0.25)

    def test_history_permutation_returns_each_patient_beside_reference(self):
        frame = small_frame()
        result = evaluate(
            frame, time="time", runin_days=2, history_forecast_fn=running_rate,
            n_perm=7, seed=4,
        )
        reference = result["permutation_reference"]
        self.assertIsNotNone(reference)
        self.assertEqual(set(reference["per_patient"]), set(frame.patient.unique()))
        self.assertTrue(np.isfinite(reference["observed"]))
        for values in reference["per_patient"].values():
            self.assertEqual(set(values), {"observed", "reference_mean", "reference_percentiles", "difference"})

    def test_patient_bootstrap_intervals_resample_complete_records(self):
        frame = small_frame()
        first = evaluate(frame, time="time", runin_days=2, threshold=0.4, n_perm=5, n_boot=12, seed=7)
        second = evaluate(frame, time="time", runin_days=2, threshold=0.4, n_perm=5, n_boot=12, seed=7)
        intervals = first["bootstrap"]["intervals"]
        self.assertEqual(first["bootstrap"]["n_boot"], 12)
        self.assertEqual(intervals, second["bootstrap"]["intervals"])
        for key in ("auc.pooled", "auc.within_patient", "auc.per_time_point_mean", "skill.runin_patient_rate.patient_mean"):
            self.assertEqual(len(intervals[key]), 2)


class TestReleaseMetadata(unittest.TestCase):
    def test_distribution_metadata_and_license(self):
        root = Path(__file__).resolve().parents[1]
        project = (root / "pyproject.toml").read_text()
        self.assertIn('version = "1.0.0"', project)
        self.assertIn('readme = "README.md"', project)
        self.assertTrue((root / "LICENSE").is_file())
        self.assertIn("Copyright (c) 2026 Wesley Remy", (root / "LICENSE").read_text())
        readme = (root / "README.md").read_text()
        self.assertIn("Minimal simulated example", readme)
        self.assertIn("expanding_patient_rate", readme)


if __name__ == "__main__":
    unittest.main()
