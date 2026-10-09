"""Small deterministic checks for withinpatient."""
import sys
import runpy
import warnings
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from withinpatient import (auc, calibration_slopes, conditional_within_slope, evaluate, patient_calibration_in_the_large,
                           per_patient_auc_summary, per_time_point_auc, permutation_reference, pooled_auc,
                           predictive_value_excess, threshold_metrics, to_markdown, within_patient_auc)


class TestWithinPatient(unittest.TestCase):
    def test_auc_and_within_patient_definition(self):
        y = [0, 1, 0, 1]
        score = [0.1, 0.9, 0.8, 0.2]
        patient = ["a", "a", "b", "b"]
        self.assertEqual(auc(y, score), 0.75)
        summary = within_patient_auc(patient, y, score)
        self.assertEqual(summary["value"], 0.5)
        self.assertEqual(summary["excluded"], 0)
        const = threshold_metrics(patient, y, [0.4] * 4, 0.5)
        self.assertEqual(const["within_patient"]["youden"], 0)
        self.assertIn("both outcomes", threshold_metrics.__doc__)
        self.assertIn("descriptive only", threshold_metrics.__doc__)

    def test_validation_and_score_auc(self):
        self.assertEqual(auc([0, 1], [-4, 8]), 1)
        for fn, args, message in [
            (auc, ([0, 2], [0.1, 0.9]), "only 0 and 1"),
            (auc, ([0, 1], [0.1]), "equal length"),
            (auc, ([0, np.nan], [0.1, 0.9]), "missing"),
            (within_patient_auc, (["a", None], [0, 1], [0.1, 0.9]), "missing"),
            (threshold_metrics, (["a", "a"], [0, 1], [0.1, 1.1], 0.5), r"\[0, 1\]"),
            (threshold_metrics, (["a", "a"], [0, 2], [0.1, 0.9], 0.5), "only 0 and 1"),
            (threshold_metrics, (["a", "a"], [0], [0.1, 0.9], 0.5), "equal length"),
            (threshold_metrics, (["a", "a"], [0, 1], [0.1, np.nan], 0.5), "missing"),
        ]:
            with self.assertRaisesRegex(ValueError, message):
                fn(*args)

    def test_evaluate_markdown_and_cli(self):
        import withinpatient.report as report
        fake_calibration = lambda *_: {"pooled_slope": 1.0, "between_slope": 0.8, "within_slope": 1.2,
                                       "pooled_intercept": 0.0, "joint_intercept": 0.0}
        frame = pd.DataFrame({"patient": ["a", "a", "b", "b"], "day": [0, 1, 0, 1],
                              "y": [0, 1, 0, 0], "forecast": [0.1, 0.7, 0.2, 0.3]})
        with patch.object(report, "calibration_slopes", fake_calibration):
            result = evaluate(frame, time="day", threshold=0.2, runin_days=1)
            self.assertNotIn("references", result)
            self.assertEqual(set(result["skill"]), {"population_rate", "retrospective_patient_rate", "runin_patient_rate", "expanding_patient_rate"})
            report_text = to_markdown(result)
            for term in ("Patients", "Scored observations", "Scored events", "Scored event rate", "Mean patient event rate", "SD of patient event rates",
                         "Within-patient pair share", "Pooled AUC", "Within-patient AUC",
                         "Patients excluded from within-patient AUC", "A0", "Pooled sensitivity", "Pooled specificity",
                         "Pooled PPV", "Pooled Youden index", "Within-patient sensitivity", "Within-patient specificity",
                         "Within-patient Youden index", "Patients with both outcomes", "Alarms per patient (mean count)",
                         "Pooled Brier score", "Patient-mean Brier score", "Brier skill vs population_rate",
                         "Brier skill vs retrospective_patient_rate", "Brier skill vs runin_patient_rate",
                         "Brier skill vs expanding_patient_rate", "Pooled calibration slope",
                         "Between-patient calibration slope", "Within-patient calibration slope"):
                self.assertIn(term, report_text)
            self.assertIn("Population A0 (noise-corrected)", report_text)
            self.assertNotIn("Pooled sensitivity", to_markdown(evaluate(frame)))
            self.assertIn("references", evaluate(frame, return_references=True))
            with tempfile.TemporaryDirectory() as directory:
                csv_path = Path(directory) / "cohort.csv"
                frame.to_csv(csv_path, index=False)
                argv = ["withinpatient", str(csv_path), "--time", "day", "--threshold", "0.2"]
                output = StringIO()
                with patch.object(sys, "argv", argv), redirect_stdout(output):
                    runpy.run_module("withinpatient", run_name="__main__")
                self.assertIn("| Quantity | Value |", output.getvalue())


class TestChecklistAdditions(unittest.TestCase):
    @staticmethod
    def two_patient_example():
        # Patient A: 30 days, 20 events, forecast 20/30 every day. Patient B: 30 days, 2 events, forecast 2/30.
        patient = np.repeat(["A", "B"], 30)
        y = np.r_[[1] * 20 + [0] * 10, [1] * 2 + [0] * 28]
        forecast = np.r_[np.full(30, 20 / 30), np.full(30, 2 / 30)]
        return patient, y, forecast

    def test_two_patient_example(self):
        patient, y, forecast = self.two_patient_example()
        self.assertAlmostEqual(pooled_auc(patient, y, forecast)["value"], 0.8229665, places=6)
        self.assertEqual(within_patient_auc(patient, y, forecast)["per_patient"], {"A": 0.5, "B": 0.5})
        ppv = predictive_value_excess(patient, y, forecast, threshold=0.5)
        self.assertAlmostEqual(ppv["ppv_excess"]["mean"], 0.0, places=12)
        self.assertAlmostEqual(ppv["npv_excess"]["mean"], 0.0, places=12)
        cil = patient_calibration_in_the_large(patient, y, forecast)
        self.assertAlmostEqual(cil["mean"], 0.0, places=12)
        self.assertEqual(cil["share_within_tolerance"], 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            slopes = calibration_slopes(patient, y, forecast)
            self.assertAlmostEqual(slopes["pooled_slope"], 1.0, places=6)
            self.assertAlmostEqual(slopes["pooled_intercept"], 0.0, places=6)
            with self.assertWarnsRegex(RuntimeWarning, "not estimable"):
                cond = conditional_within_slope(patient, y, forecast)
        self.assertTrue(np.isnan(cond["slope"]) and np.isnan(cond["se"]))
        self.assertFalse(cond["estimable"])
        self.assertEqual(cond["patients"], 2)
        self.assertIn("does not vary", cond["reason"])

    def test_per_time_point_auc_weights_and_exclusions(self):
        rows = []
        def add(time, patients, y, s):
            rows.extend((time, p, yy, ss) for p, yy, ss in zip(patients, y, s))
        # Time 0: five patients, perfect separation (AUC 1, 4 pairs).
        add(0, range(5), [1, 0, 0, 0, 0], [0.9, 0.1, 0.2, 0.3, 0.4])
        # Time 1: six patients, AUC 5/8 (8 pairs).
        add(1, range(6), [1, 1, 0, 0, 0, 0], [0.5, 0.1, 0.4, 0.3, 0.2, 0.0])
        # Time 2: only three patients, excluded by min_patients=5.
        add(2, range(3), [1, 0, 0], [0.9, 0.1, 0.2])
        # Time 3: six patients with no events, excluded for lacking both outcomes.
        add(3, range(6), [0] * 6, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        df = pd.DataFrame(rows, columns=["day", "patient", "y", "forecast"])
        out = per_time_point_auc(df, "day", patient="patient", y="y", forecast="forecast")
        self.assertAlmostEqual(out["value"], (1 + 0.625) / 2, places=12)
        self.assertAlmostEqual(out["comparison_weighted"], (1 * 4 + 0.625 * 8) / 12, places=12)
        self.assertEqual(out["time_points_used"], 2)
        self.assertEqual(out["time_points_excluded"], 2)
        self.assertEqual(set(out["per_time"]), {0, 1})
        with self.assertRaisesRegex(ValueError, "missing dataframe columns: nope"):
            per_time_point_auc(df, "nope")

    def test_predictive_value_excess_hand_example(self):
        # Patient A: alarms on days 0-2 (events 1, 1, 0), non-alarm on day 3 (non-event). Patient B: never alarmed.
        patient = np.array(["A"] * 4 + ["B"] * 4)
        y = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        forecast = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.2, 0.3, 0.4])
        out = predictive_value_excess(patient, y, forecast, threshold=0.5)
        self.assertEqual(out["ppv_excess"]["patients"], 1)
        self.assertAlmostEqual(out["ppv_excess"]["per_patient"]["A"], 2 / 3 - 0.5, places=12)
        self.assertEqual(out["npv_excess"]["patients"], 2)
        self.assertAlmostEqual(out["npv_excess"]["per_patient"]["A"], 1 - 0.5, places=12)
        self.assertAlmostEqual(out["npv_excess"]["per_patient"]["B"], 0.0, places=12)
        self.assertAlmostEqual(out["npv_excess"]["mean"], 0.25, places=12)
        self.assertAlmostEqual(out["pooled"]["ppv"], 2 / 3, places=12)
        self.assertAlmostEqual(out["pooled"]["npv"], 1.0, places=12)
        self.assertAlmostEqual(out["pooled"]["cohort_rate"], 0.25, places=12)

    def test_patient_calibration_in_the_large_hand_example(self):
        patient = np.array(["A", "A", "B", "B", "C", "C"])
        y = np.array([0, 1, 1, 1, 0, 0])
        forecast = np.array([0.4, 0.6, 0.5, 0.5, 0.1, 0.1])
        out = patient_calibration_in_the_large(patient, y, forecast)
        # Differences: A 0, B 0.5, C -0.1.
        self.assertAlmostEqual(out["mean"], (0 + 0.5 - 0.1) / 3, places=12)
        self.assertAlmostEqual(out["median"], 0.0, places=12)
        self.assertAlmostEqual(out["iqr"][0], -0.05, places=12)
        self.assertAlmostEqual(out["iqr"][1], 0.25, places=12)
        self.assertAlmostEqual(out["share_within_tolerance"], 1 / 3, places=12)
        self.assertEqual(out["patients"], 3)

    def test_conditional_within_slope_recovers_within_slope(self):
        rng = np.random.default_rng(5)
        n_patients, n_days = 200, 80
        patient = np.repeat(np.arange(n_patients), n_days)
        x = rng.normal(size=patient.size)
        intercept = np.repeat(rng.normal(-1.0, 1.0, n_patients), n_days)
        y = rng.binomial(1, 1 / (1 + np.exp(-(intercept + x))))
        # Forecast logit = patient intercept + x / 4: within-patient slope of the forecast is 4 times too small.
        forecast = 1 / (1 + np.exp(-(intercept + x / 4)))
        out = conditional_within_slope(patient, y, forecast)
        self.assertTrue(out["estimable"])
        self.assertEqual(out["patients"], n_patients)
        self.assertLess(abs(out["slope"] - 4), 0.5)
        self.assertGreater(out["se"], 0)

    def test_conditional_within_slope_no_both_outcomes(self):
        with self.assertWarnsRegex(RuntimeWarning, "no patient has both outcomes"):
            out = conditional_within_slope(["a", "a", "b"], [0, 0, 1], [0.1, 0.5, 0.9])
        self.assertTrue(np.isnan(out["slope"]))
        self.assertEqual(out["patients"], 0)
        self.assertFalse(out["estimable"])

    def test_per_patient_auc_summary(self):
        patient = np.repeat(["A", "B", "C", "D"], 2)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 0])
        forecast = np.array([0.1, 0.9, 0.9, 0.1, 0.5, 0.5, 0.2, 0.3])
        out = per_patient_auc_summary(patient, y, forecast)
        self.assertEqual(out["patients"], 3)
        self.assertEqual(out["excluded"], 1)
        self.assertAlmostEqual(out["median"], 0.5, places=12)
        self.assertEqual(out["iqr"], [0.25, 0.75])
        self.assertAlmostEqual(out["share_above_0_5"], 1 / 3, places=12)

    def test_permutation_reference_keeps_runin_fixed(self):
        rng = np.random.default_rng(2)
        patient = np.repeat(np.arange(8), 12)
        day = np.tile(np.arange(12), 8)
        y = rng.binomial(1, 0.3, size=patient.size)
        seen = []

        def running_rate(seq):
            seq = np.asarray(seq, float)
            seen.append(seq.copy())
            prior = np.r_[0.0, np.cumsum(seq[:-1])]
            return (prior + 0.3) / (np.arange(seq.size) + 1.0)

        out = permutation_reference(patient, y, day, running_rate, n_perm=5, seed=1, runin_days=4)
        self.assertTrue(np.isfinite(out["difference"]))
        self.assertEqual(len(seen), 8 * (1 + 5))
        observed_runin = [y[patient == k][:4] for k in range(8)]
        for call, k in zip(seen, np.tile(np.arange(8), 6)):
            self.assertEqual(len(call), 12)
            self.assertTrue(np.array_equal(call[:4], observed_runin[k]))

    def test_evaluate_wires_new_quantities(self):
        rng = np.random.default_rng(9)
        frame = pd.DataFrame({"patient": np.repeat([f"p{i}" for i in range(6)], 10),
                              "day": np.tile(np.arange(10), 6)})
        frame["y"] = rng.binomial(1, 0.3, size=len(frame))
        frame["forecast"] = np.clip(0.3 + 0.2 * frame["y"] + rng.normal(0, 0.2, len(frame)), 0.01, 0.99)
        plain = evaluate(frame)
        self.assertIsNone(plain["auc"]["per_time_point"])
        self.assertIsNone(plain["threshold"])
        self.assertIsNone(plain["permutation_reference"])
        self.assertIn("per_patient_summary", plain["auc"])
        self.assertIn("conditional_within_slope", plain)
        self.assertIn("calibration_in_the_large", plain)
        full = evaluate(frame, time="day", threshold=0.5, runin_days=3, n_perm=3, seed=1,
                        history_forecast_fn=lambda seq: np.full(len(seq), 0.3))
        self.assertIsNotNone(full["auc"]["per_time_point"])
        self.assertIn("predictive_value_excess", full["threshold"])
        self.assertIsNotNone(full["permutation_reference"])
        text = to_markdown(full)
        for term in ("Per-time-point AUC (equal weight)", "PPV excess over own rate", "History permutation difference",
                     "Calibration in the large, mean patient difference", "Conditional within-patient calibration slope"):
            self.assertIn(term, text)
        self.assertIn("Within-patient calibration slope (joint model)", text)


if __name__ == "__main__":
    unittest.main()
