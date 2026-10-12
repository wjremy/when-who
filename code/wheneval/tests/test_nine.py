"""Checks for the paper's nine metric helpers."""
import itertools
import unittest

import numpy as np

from wheneval.metrics import calibration_slopes, conditional_within_slope
from wheneval.nine import (
    accuracy,
    ap_chance,
    average_precision,
    f1,
    naive_metrics,
    net_benefit,
    nine_metrics,
)


def brute_average_precision(y, scores):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    ap = 0.0
    previous_recall = 0.0
    for threshold in np.sort(np.unique(scores))[::-1]:
        selected = scores >= threshold
        tp = int(np.sum(y[selected] == 1))
        recall = tp / positives
        precision = tp / int(selected.sum())
        ap += (recall - previous_recall) * precision
        previous_recall = recall
    return ap


class TestAveragePrecision(unittest.TestCase):
    def test_matches_threshold_brute_force_with_random_ties(self):
        rng = np.random.default_rng(173)
        for n in range(1, 18):
            for _ in range(20):
                y = rng.integers(0, 2, size=n)
                scores = rng.integers(0, 5, size=n) / 4
                actual = average_precision(y, scores)
                expected = brute_average_precision(y, scores)
                if np.isnan(expected):
                    self.assertTrue(np.isnan(actual))
                else:
                    self.assertAlmostEqual(actual, expected, places=14)

    def test_chance_matches_exhaustive_random_orderings_through_n_seven(self):
        for n in range(1, 8):
            descending_scores = np.arange(n, 0, -1, dtype=float)
            for m in range(1, n + 1):
                outcomes = []
                for positive_positions in itertools.combinations(range(n), m):
                    y_ordered = np.zeros(n, dtype=int)
                    y_ordered[list(positive_positions)] = 1
                    outcomes.append(average_precision(y_ordered, descending_scores))
                expected = float(np.mean(outcomes))
                self.assertAlmostEqual(ap_chance([1] * m + [0] * (n - m)), expected, places=14)


class TestDecisionAndNineMetrics(unittest.TestCase):
    @staticmethod
    def toy_data():
        patient = np.array(["A"] * 4 + ["B"] * 4)
        y = np.array([0, 1, 0, 1] * 2)
        forecast = np.array([0.1, 0.8, 0.4, 0.6, 0.7, 0.6, 0.2, 0.1])
        yardstick = np.array([0.25] * 4 + [0.9] * 4)
        return patient, y, forecast, yardstick

    def test_net_benefit_hand_computation(self):
        # Two true positives and one false positive among four rows; t/(1-t)=2/3.
        self.assertAlmostEqual(net_benefit([1, 0, 1, 0], [0.8, 0.6, 0.5, 0.2], 0.4), 1 / 3)

    def test_nine_metrics_two_patient_example(self):
        patient, y, forecast, yardstick = self.toy_data()
        result = nine_metrics(patient, y, forecast, 0.5, yardstick)

        self.assertAlmostEqual(result["auc"]["pooled"], 0.65625)
        self.assertEqual(result["auc"]["per_patient"], {"A": 1.0, "B": 0.25})
        self.assertEqual(result["average_precision"]["per_patient"], {"A": 1.0, "B": 0.5})
        self.assertAlmostEqual(result["average_precision"]["pooled"], 0.75)
        self.assertEqual(result["sensitivity"]["summary"]["mean"], 0.75)
        self.assertEqual(result["specificity"]["summary"]["mean"], 0.75)
        self.assertEqual(result["ppv"]["pooled"], 0.75)
        self.assertAlmostEqual(result["ppv"]["summary"]["mean"], 0.25)
        self.assertEqual(result["npv"]["pooled"], 0.75)
        self.assertAlmostEqual(result["npv"]["summary"]["mean"], 0.25)
        self.assertAlmostEqual(result["net_benefit"]["pooled"], 0.25)
        self.assertAlmostEqual(result["net_benefit"]["pooled_treat_all"], 0.0)
        # A: model NB 2/4 - 0 = .5, policy (.25 < .5) treats none: 0. B: model NB (1-1)/4 = 0, policy treats all: (2-2)/4 = 0.
        self.assertAlmostEqual(result["net_benefit"]["summary"]["mean"], 0.25)

        # Patient A: model Brier .0925, yardstick .3125. Patient B: .375 and .41.
        expected_equal_patient_skill = 1 - ((0.0925 + 0.375) / 2) / ((0.3125 + 0.41) / 2)
        individual_skills = result["brier_skill"]["per_patient"]
        self.assertAlmostEqual(individual_skills["A"], 1 - 0.0925 / 0.3125)
        self.assertAlmostEqual(individual_skills["B"], 1 - 0.375 / 0.41)
        self.assertAlmostEqual(result["brier_skill"]["summary"]["mean"], expected_equal_patient_skill)
        self.assertNotAlmostEqual(result["brier_skill"]["summary"]["mean"],
                                  np.mean(list(individual_skills.values())))

        self.assertEqual(accuracy(patient, y, forecast, 0.5)["pooled"], 0.75)
        self.assertEqual(f1(patient, y, forecast, 0.5)["pooled"], 0.75)
        self.assertEqual(f1(patient, y, forecast, 0.5)["per_patient"], {"A": 1.0, "B": 0.5})
        self.assertAlmostEqual(result["calibration_slope"]["pooled"],
                               calibration_slopes(patient, y, forecast)["pooled_slope"])
        conditional = conditional_within_slope(patient, y, forecast)
        reported = result["calibration_slope"]["summary"]["mean"]
        if np.isfinite(conditional["slope"]):
            self.assertAlmostEqual(reported, conditional["slope"])
        else:
            self.assertTrue(np.isnan(reported))

    def test_naive_metrics_two_patient_example_hand_computed(self):
        patient, y, forecast, _ = self.toy_data()
        result = naive_metrics(patient, y, forecast, 0.5, population_rate=0.5)

        self.assertEqual(result["auc"]["per_patient"], {"A": 1.0, "B": 0.25})
        self.assertAlmostEqual(result["auc"]["summary"]["mean"], 0.625)
        # AP includes each patient with at least one event; B's ranks yield 0.5.
        self.assertEqual(result["average_precision"]["per_patient"],
                         {"A": 1.0, "B": 0.5})
        self.assertAlmostEqual(result["average_precision"]["summary"]["mean"], 0.75)

        # Both patients have events and non-events; A is perfect and B is 0.5/0.5.
        for metric in ("sensitivity", "specificity"):
            self.assertEqual(result[metric]["per_patient"], {"A": 1.0, "B": 0.5})
            self.assertAlmostEqual(result[metric]["summary"]["mean"], 0.75)
        self.assertAlmostEqual(result["naive_youden"], 0.5)

        # Raw predictive values retain the patients' own rates; at t=.5, NB
        # is .5 for A and zero for B, with treat-none as the zero comparator.
        self.assertEqual(result["ppv"]["per_patient"], {"A": 1.0, "B": 0.5})
        self.assertEqual(result["npv"]["per_patient"], {"A": 1.0, "B": 0.5})
        self.assertAlmostEqual(result["net_benefit"]["per_patient"]["A"], 0.5)
        self.assertAlmostEqual(result["net_benefit"]["per_patient"]["B"], 0.0)
        self.assertAlmostEqual(result["net_benefit"]["summary"]["mean"], 0.25)

        # Each patient's population-rate Brier reference is .25. The forecast
        # losses are .0925 and .375, so the equal-patient ratio is 1-.23375/.25.
        self.assertAlmostEqual(result["brier_skill"]["per_patient"]["A"], 0.63)
        self.assertAlmostEqual(result["brier_skill"]["per_patient"]["B"], -0.5)
        self.assertAlmostEqual(result["brier_skill"]["summary"]["mean"], 0.065)
        self.assertTrue(np.isfinite(
            result["calibration_slope"]["summary"]["mean"]))

    def test_naive_calibration_slope_is_not_applicable_for_patient_constant_forecast(self):
        patient, y, _, _ = self.toy_data()
        forecast = np.array([0.25] * 4 + [0.75] * 4)
        result = naive_metrics(patient, y, forecast, 0.5, population_rate=0.5)
        self.assertEqual(result["calibration_slope"]["summary"]["mean"], "n/a")
        self.assertEqual(result["calibration_slope"]["per_patient"], {})


if __name__ == "__main__":
    unittest.main()
