"""Tests for the Section 3 evaluator and its compatibility boundary."""
import unittest
import warnings

import numpy as np
import pandas as pd

from withinpatient import auc, evaluate, evaluate_v1
from withinpatient.evaluation import _ap_batch, _auc_batch
from withinpatient.nine import average_precision


def level_only_frame(n_patients=20, n_days=12):
    patient = np.repeat(np.arange(n_patients), n_days)
    time = np.tile(np.arange(n_days), n_patients)
    y = np.tile(np.r_[np.ones(n_days // 2), np.zeros(n_days - n_days // 2)], n_patients)
    rates = np.repeat(np.linspace(0.15, 0.85, n_patients), n_days)
    return pd.DataFrame({"patient": patient, "time": time, "y": y,
                         "forecast": rates.astype(float)})


def running_rate(y, times, prior_rate=0.3, prior_weight=4.0):
    stops = np.searchsorted(times, times, side="left")
    cumulative = np.r_[0.0, np.cumsum(np.asarray(y, dtype=float))]
    return (prior_rate * prior_weight + cumulative[stops]) / (prior_weight + stops)


class TestEvaluateV2(unittest.TestCase):
    def test_vectorized_ranking_metrics_match_scalar_definitions_with_ties(self):
        rng = np.random.default_rng(41)
        outcomes = rng.integers(0, 2, size=(25, 18))
        scores = rng.integers(0, 5, size=(25, 18)).astype(float) / 4
        np.testing.assert_allclose(_auc_batch(outcomes, scores),
                                   [auc(y, p) for y, p in zip(outcomes, scores)],
                                   equal_nan=True, atol=1e-14)
        np.testing.assert_allclose(_ap_batch(outcomes, scores),
                                   [average_precision(y, p) for y, p in zip(outcomes, scores)],
                                   equal_nan=True, atol=1e-14)

    def test_level_only_forecast_has_zero_excess(self):
        frame = level_only_frame()
        result = evaluate(frame, n_perm=31, seed=4)
        for name, metric in result["metrics"].items():
            for value in metric["per_patient"].values():
                self.assertAlmostEqual(value["observed"], value["chance_level"], places=12, msg=name)
                self.assertAlmostEqual(value["excess"], 0.0, places=12, msg=name)
        self.assertEqual(result["metrics"]["sensitivity"]["summary"]["n_eligible"],
                         result["metrics"]["specificity"]["summary"]["n_eligible"])
        self.assertIn("Median excess", __import__("withinpatient").to_markdown(result))

    def test_running_rate_on_shuffled_records_is_at_chance(self):
        rng = np.random.default_rng(212)
        n_patients, n_days = 60, 36
        patient = np.repeat(np.arange(n_patients), n_days)
        time = np.tile(np.arange(n_days), n_patients)
        y = rng.binomial(1, 0.3, size=len(patient))
        frame = pd.DataFrame({"patient": patient, "time": time, "y": y})
        frame["forecast"] = np.concatenate([
            running_rate(y[patient == k], time[patient == k]) for k in range(n_patients)
        ])
        result = evaluate(frame, runin=4,
                          rebuild={"name": "running_rate", "prior_rate": 0.3, "prior_weight": 4},
                          metrics=["auc"], n_perm=100, seed=15)
        values = [x["excess"] for x in result["metrics"]["auc"]["per_patient"].values()]
        self.assertGreater(len(values), 50)
        self.assertLess(abs(float(np.mean(values))), 0.06)

    def test_outcome_based_yardstick_is_rebuilt_with_runin_fixed(self):
        frame = level_only_frame(3, 10)
        frame["forecast"] = np.concatenate([
            running_rate(group.y.to_numpy(), group.time.to_numpy(), 0.5, 1.0)
            for _, group in frame.groupby("patient", sort=False)
        ])
        seen = []

        def updated_rate(y, scored):
            seen.append((y.copy(), scored.copy()))
            previous = np.r_[0.0, np.cumsum(y[:-1])]
            return (0.5 + previous) / (1 + np.arange(len(y)))

        result = evaluate(frame, runin=2,
                          rebuild={"name": "running_rate", "prior_rate": 0.5, "prior_weight": 1},
                          yardstick=updated_rate, metrics=["brier_skill"], n_perm=7, seed=6)
        self.assertEqual(len(seen), 3 * (1 + 7))
        for patient in range(3):
            self.assertTrue(np.array_equal(seen[patient][0], frame.loc[frame.patient == patient, "y"]))
        offset = 3
        for patient in range(3):
            observed_y = frame.loc[frame.patient == patient, "y"].to_numpy()
            for _ in range(7):
                y_seen, scored_seen = seen[offset]
                offset += 1
                np.testing.assert_array_equal(y_seen[:2], observed_y[:2])
                np.testing.assert_array_equal(np.sort(y_seen[2:]), np.sort(observed_y[2:]))
                self.assertEqual(int(scored_seen.sum()), 8)
        self.assertEqual(result["metrics"]["brier_skill"]["summary"]["n_eligible"], 3)

    def test_custom_auc_matches_builtin_for_observed_and_permutation(self):
        result = evaluate(level_only_frame(12, 12),
                          metrics=["auc", ("auc_copy", auc)], n_perm=25, seed=8)
        built = result["metrics"]["auc"]
        copied = result["metrics"]["auc_copy"]
        self.assertEqual(built["per_patient"], copied["per_patient"])
        self.assertEqual(built["summary"], copied["summary"])

    def test_average_precision_uses_its_own_eligibility(self):
        frame = pd.DataFrame({
            "patient": np.repeat(["all_positive", "all_negative", "mixed"], 4),
            "time": np.tile(np.arange(4), 3),
            "y": [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0],
            "forecast": [0.4, 0.5, 0.6, 0.7, 0.4, 0.5, 0.6, 0.7, 0.4, 0.5, 0.6, 0.7],
        })
        result = evaluate(frame, metrics=["average_precision"], n_perm=5, seed=2)
        self.assertEqual(set(result["metrics"]["average_precision"]["per_patient"]),
                         {"all_positive", "mixed"})

    def test_seed_reproducibility_and_p_values_in_bounds(self):
        frame = level_only_frame(10, 12)
        first = evaluate(frame, metrics=["auc"], n_perm=37, seed=29)
        again = evaluate(frame, metrics=["auc"], n_perm=37, seed=29)
        self.assertEqual(first, again)
        for patient in first["metrics"]["auc"]["per_patient"].values():
            self.assertGreaterEqual(patient["p_value"], 1 / 38)
            self.assertLessEqual(patient["p_value"], 1.0)

    def test_fixed_runin_and_lagged_fixed_model_rebuild(self):
        rng = np.random.default_rng(9)
        y = rng.binomial(1, 0.35, size=80)
        frame = pd.DataFrame({"patient": np.repeat(np.arange(4), 20),
                              "time": np.tile(np.arange(20), 4), "y": y,
                              "x": rng.normal(size=80)})

        def predict(design):
            lag = design["y_lag1"].fillna(0.4).to_numpy()
            return 0.1 + 0.7 * lag

        frame["forecast"] = np.concatenate([
            predict(pd.DataFrame({"patient": frame.patient[frame.patient == k].to_numpy(),
                                  "time": frame.time[frame.patient == k].to_numpy(),
                                  "x": frame.x[frame.patient == k].to_numpy(),
                                  "y_lag1": np.r_[np.nan, y[frame.patient.to_numpy() == k][:-1]]}))
            for k in range(4)
        ])
        result = evaluate(frame, runin=3, rebuild={"name": "lagged_outcomes", "lags": 1,
                                                   "predict": predict},
                          metrics=["auc"], n_perm=9, seed=2)
        self.assertEqual(result["settings"]["scored_rows"], 4 * 17)

    def test_evaluate_v1_is_available_with_deprecation_warning(self):
        frame = level_only_frame(4, 6)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            result = evaluate_v1(frame, time="time", n_perm=1)
        self.assertIn("cohort", result)
        self.assertTrue(any(item.category is DeprecationWarning for item in seen))


if __name__ == "__main__":
    unittest.main()


class RebuildWithRowsTest(unittest.TestCase):
    def test_three_argument_rebuild_receives_patient_rows(self):
        rng = np.random.default_rng(5)
        patient = np.repeat(np.arange(12), 20)
        time = np.tile(np.arange(20), 12)
        covariate = rng.normal(size=len(patient))
        y = rng.binomial(1, 0.3, len(patient))
        frame = pd.DataFrame({"patient": patient, "time": time, "y": y, "c": covariate})

        def build(outcomes, scored, rows):
            past = np.concatenate([[0], np.cumsum(outcomes)[:-1]]) / (1 + np.arange(len(outcomes)))
            return 1 / (1 + np.exp(-(rows["c"].to_numpy() + past - 1)))

        frame["forecast"] = np.concatenate([
            build(g.y.to_numpy(), None, g) for _, g in frame.groupby("patient", sort=False)])
        result = evaluate(frame, rebuild=build, metrics=["auc"], n_perm=20, seed=1)
        self.assertEqual(result["metrics"]["auc"]["summary"]["n_eligible"], 12)
        with self.assertRaises(ValueError):
            evaluate(frame.assign(forecast=frame.forecast * 0.9), rebuild=build, metrics=["auc"], n_perm=5)
