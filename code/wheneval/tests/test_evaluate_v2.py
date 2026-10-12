"""Tests for the one-call evaluator and its compatibility boundary."""
import unittest
import warnings

import numpy as np
import pandas as pd

import wheneval as we
from wheneval.evaluation import _ap_batch, _auc_batch
from wheneval.nine import average_precision


def level_only_frame(n_patients=20, n_days=12):
    patient_id = np.repeat(np.arange(n_patients), n_days)
    time = np.tile(np.arange(n_days), n_patients)
    outcome = np.tile(np.r_[np.ones(n_days // 2), np.zeros(n_days - n_days // 2)], n_patients)
    rates = np.repeat(np.linspace(0.15, 0.85, n_patients), n_days)
    return pd.DataFrame({"patient_id": patient_id, "time": time, "outcome": outcome,
                         "prediction": rates.astype(float)})


def running_rate(y, times, prior_rate=0.3, prior_weight=4.0):
    stops = np.searchsorted(times, times, side="left")
    cumulative = np.r_[0.0, np.cumsum(np.asarray(y, dtype=float))]
    return (prior_rate * prior_weight + cumulative[stops]) / (prior_weight + stops)


def smoothed_training_rate(train_rows, test_rows):
    rate = (float(train_rows["outcome"].sum()) + 1) / (len(train_rows) + 2)
    return np.full(len(test_rows), rate)


def cross_validated_predictions(frame, fit_predict, folds):
    predictions = np.full(len(frame), np.nan)
    for _, rows in frame.groupby("patient_id", sort=False):
        rows = rows.sort_values("time", kind="stable")
        positions = np.arange(len(rows))
        fold_id = positions % folds
        for fold in range(folds):
            test_ix = np.flatnonzero(fold_id == fold)
            train_ix = np.flatnonzero(fold_id != fold)
            test_rows = rows.iloc[test_ix].copy()
            test_rows["outcome"] = np.nan
            predictions[rows.index.to_numpy()[test_ix]] = fit_predict(rows.iloc[train_ix], test_rows)
    return predictions


class TestEvaluateV3(unittest.TestCase):
    def test_vectorized_ranking_metrics_match_scalar_definitions_with_ties(self):
        rng = np.random.default_rng(41)
        outcomes = rng.integers(0, 2, size=(25, 18))
        scores = rng.integers(0, 5, size=(25, 18)).astype(float) / 4
        np.testing.assert_allclose(_auc_batch(outcomes, scores),
                                   [we.auc(y, p) for y, p in zip(outcomes, scores)],
                                   equal_nan=True, atol=1e-14)
        np.testing.assert_allclose(_ap_batch(outcomes, scores),
                                   [average_precision(y, p) for y, p in zip(outcomes, scores)],
                                   equal_nan=True, atol=1e-14)

    def test_level_only_forecast_has_zero_excess(self):
        result = we.evaluate(level_only_frame(), n_shuffles=31, seed=4)
        for name, metric in result["metrics"].items():
            for value in metric["per_patient"].values():
                self.assertAlmostEqual(value["score"], value["shuffled_baseline"], places=12, msg=name)
                self.assertAlmostEqual(value["excess"], 0.0, places=12, msg=name)
        self.assertEqual(result["metrics"]["sensitivity"]["summary"]["n_eligible"],
                         result["metrics"]["specificity"]["summary"]["n_eligible"])
        report = we.to_markdown(result)
        for label in ("Score", "Shuffled baseline", "Excess", "p", "Pooled score (for comparison)",
                      "Per-patient score (mean; for comparison)"):
            self.assertIn(label, report)

    def test_new_window_count_example_call(self):
        rng = np.random.default_rng(25)
        patient_id = np.repeat(np.arange(8), 14)
        day = np.tile(np.arange(14), 8)
        outcome = rng.binomial(1, 0.3, len(patient_id))
        prediction = np.zeros(len(outcome), dtype=float)
        for pid in np.unique(patient_id):
            ix = np.flatnonzero(patient_id == pid)
            yy = outcome[ix]
            stops = np.arange(len(ix))
            cumulative = np.r_[0, np.cumsum(yy)]
            starts = np.maximum(stops - 7, 0)
            counts = np.minimum(stops, 7)
            events = cumulative[stops] - cumulative[starts]
            prediction[ix] = np.divide(events, counts, out=np.zeros_like(events, dtype=float), where=counts > 0)
        df = pd.DataFrame({"patient_id": patient_id, "day": day,
                           "outcome": outcome, "prediction": prediction})
        result = we.evaluate(data=df, patient_id="patient_id", time="day", outcome="outcome",
                             prediction="prediction", history_rule="window_count", window=7,
                             n_shuffles=200)
        self.assertEqual(result["settings"]["n_shuffles"], 200)
        self.assertEqual(result["settings"]["history_rule"], "window_count")

    def test_legacy_keywords_are_identical_and_conflicts_are_errors(self):
        frame = level_only_frame(8, 12)
        current = we.evaluate(frame, metrics=["auc"], n_shuffles=17, baseline="median", seed=8)
        legacy = we.evaluate(frame, patient="patient_id", forecast="prediction", rebuild=None,
                             metrics=["auc"], n_perm=17, chance="median", seed=8)
        self.assertEqual(current, legacy)
        with self.assertRaises(TypeError):
            we.evaluate(frame, patient_id="patient_id", patient="patient_id", metrics=["auc"])
        with self.assertRaises(TypeError):
            we.evaluate(frame, made_up=True)

    def test_window_prediction_none_and_width_validation(self):
        frame = level_only_frame(8, 12)
        with self.assertRaisesRegex(ValueError, "requires a width"):
            we.evaluate(frame, history_rule="window_count", metrics=["auc"])
        with self.assertRaisesRegex(ValueError, "conflict"):
            we.evaluate(frame, history_rule={"name": "window_count", "w": 5}, window=7,
                        metrics=["auc"])
        with self.assertRaisesRegex(ValueError, "prediction=None"):
            we.evaluate(frame, prediction=None, history_rule=None)

    def test_rate_so_far_alias_and_legacy_output_names(self):
        rng = np.random.default_rng(212)
        n_patients, n_days = 12, 36
        patient_id = np.repeat(np.arange(n_patients), n_days)
        time = np.tile(np.arange(n_days), n_patients)
        outcome = rng.binomial(1, 0.3, size=len(patient_id))
        frame = pd.DataFrame({"patient_id": patient_id, "time": time, "outcome": outcome})
        frame["prediction"] = np.concatenate([
            running_rate(outcome[patient_id == k], time[patient_id == k])
            for k in range(n_patients)
        ])
        result = we.evaluate(frame, runin=4,
                             history_rule={"name": "rate_so_far", "prior_rate": 0.3, "prior_weight": 4},
                             metrics=["auc"], n_shuffles=40, seed=15)
        values = [x["excess"] for x in result["metrics"]["auc"]["per_patient"].values()]
        self.assertGreater(len(values), 10)
        self.assertLess(abs(float(np.mean(values))), 0.1)
        self.assertEqual(result["settings"]["history_rule"], "running_rate")
        self.assertEqual(set(result["metrics"]["auc"]["per_patient"][0]),
                         {"score", "shuffled_baseline", "excess", "p_value", "n_scored"})
        self.assertIn("mean_score", result["metrics"]["auc"]["summary"])
        self.assertIn("per_patient_mean", result["metrics"]["auc"]["comparison"])

    def test_outcome_based_yardstick_is_rebuilt_with_runin_fixed(self):
        frame = level_only_frame(3, 10)
        frame["prediction"] = np.concatenate([
            running_rate(group.outcome.to_numpy(), group.time.to_numpy(), 0.5, 1.0)
            for _, group in frame.groupby("patient_id", sort=False)
        ])
        seen = []

        def updated_rate(y, scored):
            seen.append((y.copy(), scored.copy()))
            previous = np.r_[0.0, np.cumsum(y[:-1])]
            return (0.5 + previous) / (1 + np.arange(len(y)))

        result = we.evaluate(frame, runin=2,
                             history_rule={"name": "running_rate", "prior_rate": 0.5, "prior_weight": 1},
                             yardstick=updated_rate, metrics=["brier_skill"], n_shuffles=7, seed=6)
        self.assertEqual(len(seen), 3 * (1 + 7))
        for patient in range(3):
            self.assertTrue(np.array_equal(seen[patient][0], frame.loc[frame.patient_id == patient, "outcome"]))
        offset = 3
        for patient in range(3):
            observed_y = frame.loc[frame.patient_id == patient, "outcome"].to_numpy()
            for _ in range(7):
                y_seen, scored_seen = seen[offset]
                offset += 1
                np.testing.assert_array_equal(y_seen[:2], observed_y[:2])
                np.testing.assert_array_equal(np.sort(y_seen[2:]), np.sort(observed_y[2:]))
                self.assertEqual(int(scored_seen.sum()), 8)
        self.assertEqual(result["metrics"]["brier_skill"]["summary"]["n_eligible"], 3)

    def test_custom_auc_matches_builtin_for_observed_and_permutation(self):
        result = we.evaluate(level_only_frame(12, 12),
                             metrics=["auc", ("auc_copy", we.auc)], n_shuffles=25, seed=8)
        built = result["metrics"]["auc"]
        copied = result["metrics"]["auc_copy"]
        self.assertEqual(built["per_patient"], copied["per_patient"])
        self.assertEqual(built["summary"], copied["summary"])

    def test_average_precision_uses_its_own_eligibility(self):
        frame = pd.DataFrame({
            "patient_id": np.repeat(["all_positive", "all_negative", "mixed"], 4),
            "time": np.tile(np.arange(4), 3),
            "outcome": [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0],
            "prediction": [0.4, 0.5, 0.6, 0.7, 0.4, 0.5, 0.6, 0.7, 0.4, 0.5, 0.6, 0.7],
        })
        result = we.evaluate(frame, metrics=["average_precision"], n_shuffles=5, seed=2)
        self.assertEqual(set(result["metrics"]["average_precision"]["per_patient"]),
                         {"all_positive", "mixed"})

    def test_seed_reproducibility_and_p_values_in_bounds(self):
        frame = level_only_frame(10, 12)
        first = we.evaluate(frame, metrics=["auc"], n_shuffles=37, seed=29)
        again = we.evaluate(frame, metrics=["auc"], n_shuffles=37, seed=29)
        self.assertEqual(first, again)
        for patient in first["metrics"]["auc"]["per_patient"].values():
            self.assertGreaterEqual(patient["p_value"], 1 / 38)
            self.assertLessEqual(patient["p_value"], 1.0)

    def test_fixed_runin_and_lagged_fixed_model_rebuild(self):
        rng = np.random.default_rng(9)
        outcome = rng.binomial(1, 0.35, size=80)
        frame = pd.DataFrame({"patient_id": np.repeat(np.arange(4), 20),
                              "time": np.tile(np.arange(20), 4), "outcome": outcome,
                              "x": rng.normal(size=80)})

        def predict(design):
            lag = design["outcome_lag1"].fillna(0.4).to_numpy()
            return 0.1 + 0.7 * lag

        frame["prediction"] = np.concatenate([
            predict(pd.DataFrame({"patient_id": frame.patient_id[frame.patient_id == k].to_numpy(),
                                  "time": frame.time[frame.patient_id == k].to_numpy(),
                                  "x": frame.x[frame.patient_id == k].to_numpy(),
                                  "outcome_lag1": np.r_[np.nan, outcome[frame.patient_id.to_numpy() == k][:-1]]}))
            for k in range(4)
        ])
        result = we.evaluate(frame, runin=3,
                             history_rule={"name": "lagged_outcomes", "lags": 1, "predict": predict},
                             metrics=["auc"], n_shuffles=9, seed=2)
        self.assertEqual(result["settings"]["scored_rows"], 4 * 17)

    def test_refit_predictions_match_and_prediction_none_is_identical(self):
        rng = np.random.default_rng(73)
        patient_id = np.repeat(np.arange(18), 20)
        time = np.tile(np.arange(20), 18)
        outcome = rng.binomial(1, 0.3, len(patient_id))
        frame = pd.DataFrame({"patient_id": patient_id, "time": time, "outcome": outcome})
        frame["prediction"] = cross_validated_predictions(frame, smoothed_training_rate, 5)
        rule = we.refit(smoothed_training_rate, folds=5)
        explicit = we.evaluate(frame, history_rule=rule, metrics=["auc"], n_shuffles=15, seed=8)
        automatic = we.evaluate(frame.drop(columns="prediction"), prediction=None, history_rule=rule,
                                 metrics=["auc"], n_shuffles=15, seed=8)
        self.assertEqual(explicit, automatic)
        self.assertEqual(explicit["settings"]["history_rule"], "refit")

    def test_refit_reaches_shuffled_baseline_for_finite_records(self):
        rng = np.random.default_rng(20261012)
        n_patients, n_days, folds = 300, 40, 5
        patient_id = np.repeat(np.arange(n_patients), n_days)
        time = np.tile(np.arange(n_days), n_patients)
        outcome = rng.binomial(1, 0.3, len(patient_id))
        frame = pd.DataFrame({"patient_id": patient_id, "time": time, "outcome": outcome})
        frame["prediction"] = cross_validated_predictions(frame, smoothed_training_rate, folds)
        refit_rule = we.refit(smoothed_training_rate, folds=folds)
        refitted = we.evaluate(frame, history_rule=refit_rule, metrics=["auc"],
                               n_shuffles=99, seed=17)

        def frozen_prediction(y, scored, rows):
            return rows["prediction"].to_numpy()

        frozen = we.evaluate(frame, history_rule=frozen_prediction, metrics=["auc"],
                             n_shuffles=99, seed=17)
        refit_summary = refitted["metrics"]["auc"]["summary"]
        frozen_summary = frozen["metrics"]["auc"]["summary"]
        self.assertLess(refit_summary["mean_score"], 0.5)
        self.assertLess(abs(refit_summary["mean_excess"]), 0.02)
        self.assertLess(abs(frozen_summary["mean_shuffled_baseline"] - 0.5), 0.03)
        self.assertGreater(abs(refit_summary["mean_shuffled_baseline"] -
                               frozen_summary["mean_shuffled_baseline"]), 0.03)

    def test_refit_validation(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            we.refit(smoothed_training_rate, folds=1)
        with self.assertRaisesRegex(ValueError, "at most"):
            we.evaluate(level_only_frame(1, 4), history_rule=we.refit(smoothed_training_rate, folds=5),
                        metrics=["auc"], n_shuffles=1)

    def test_evaluate_v1_is_available_with_deprecation_warning(self):
        frame = pd.DataFrame({"patient": np.repeat(np.arange(4), 6),
                              "time": np.tile(np.arange(6), 4),
                              "y": np.tile([0, 1, 0, 1, 0, 1], 4),
                              "forecast": np.repeat(np.linspace(0.2, 0.8, 4), 6)})
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            result = we.evaluate_v1(frame, time="time", n_perm=1)
        self.assertIn("cohort", result)
        self.assertTrue(any(item.category is DeprecationWarning for item in seen))

    def test_unknown_and_duplicate_legacy_keywords(self):
        frame = level_only_frame(4, 6)
        with self.assertRaises(TypeError):
            we.evaluate(frame, prediction="prediction", forecast="prediction", metrics=["auc"])
        with self.assertRaises(TypeError):
            we.evaluate(frame, n_shuffles=3, n_perm=3, metrics=["auc"])


if __name__ == "__main__":
    unittest.main()
