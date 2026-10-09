"""Checks of the paper's identities on a small simulated cohort (fixed seed, runs in a few seconds)."""
import unittest

import numpy as np

from withinpatient import (auc, calibration_slopes, expanding_patient_rate, permutation_reference,
                           rate_benchmark, threshold_metrics, within_patient_auc)


def simulate(n_patients=150, n_days=120, tau=0.65, omega=0.9, seed=7):
    rng = np.random.default_rng(seed)
    patient = np.repeat(np.arange(n_patients), n_days)
    day = np.tile(np.arange(n_days), n_patients)
    signal = rng.normal(size=n_patients * n_days)
    intercept = np.repeat(rng.normal(-2.0, tau, n_patients), n_days)
    logit = intercept + omega * signal
    risk = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, risk)
    return patient, day, y, risk, logit


class TestIdentities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patient, cls.day, cls.y, cls.risk, cls.logit = simulate()
        rates = np.array([cls.y[cls.patient == k].mean() for k in np.unique(cls.patient)])
        cls.rate_forecast = rates[cls.patient]

    def test_rate_only_forecast(self):
        # A forecast that is constant within every patient cannot tell that patient's days apart.
        self.assertEqual(within_patient_auc(self.patient, self.y, self.rate_forecast)["value"], 0.5)
        tm = threshold_metrics(self.patient, self.y, self.rate_forecast, float(np.mean(self.y)))
        self.assertEqual(tm["within_patient"]["youden"], 0.0)
        self.assertGreater(tm["pooled"]["youden"], 0.05)
        # Its pooled AUC is the benchmark A0.
        self.assertAlmostEqual(auc(self.y, self.rate_forecast), rate_benchmark(self.patient, self.y), places=12)

    def test_auc_matches_pair_count_with_ties(self):
        y = np.array([1, 0, 1, 0, 0, 1])
        s = np.array([0.4, 0.4, 0.9, 0.1, 0.9, 0.2])
        wins = sum((a > b) + 0.5 * (a == b) for a in s[y == 1] for b in s[y == 0])
        self.assertAlmostEqual(auc(y, s), wins / (3 * 3), places=12)

    def test_no_heterogeneity_pooled_equals_within(self):
        patient, _, y, risk, _ = simulate(tau=0.0, seed=11)
        self.assertLess(abs(auc(y, risk) - within_patient_auc(patient, y, risk)["value"]), 0.02)

    def test_calibration_slopes(self):
        slopes = calibration_slopes(self.patient, self.y, self.risk)
        for key in ("pooled_slope", "between_slope", "within_slope"):
            self.assertLess(abs(slopes[key] - 1), 0.15, key)
        # Daily variation of the forecast four times too small: within-patient slope near four.
        mean_logit = np.array([self.logit[self.patient == k].mean() for k in np.unique(self.patient)])[self.patient]
        damped = 1 / (1 + np.exp(-(mean_logit + (self.logit - mean_logit) / 4)))
        self.assertLess(abs(calibration_slopes(self.patient, self.y, damped)["within_slope"] - 4), 0.6)

    def test_expanding_rate_uses_only_earlier_days(self):
        base = expanding_patient_rate(self.patient, self.y, self.day, 0.15, 10)
        changed = self.y.copy()
        last = np.flatnonzero(self.patient == 0)[-1]
        changed[last] = 1 - changed[last]
        again = expanding_patient_rate(self.patient, changed, self.day, 0.15, 10)
        self.assertTrue(np.array_equal(base, again))

    def test_history_forecast_is_below_half_under_independence(self):
        # Independent events: a running rate has a within-patient AUC below 0.5, and the
        # permutation reference reproduces it (difference near zero).
        rng = np.random.default_rng(3)
        patient = np.repeat(np.arange(120), 60)
        day = np.tile(np.arange(60), 120)
        y = rng.binomial(1, 0.2, size=patient.size)

        def running_rate(seq):
            seq = np.asarray(seq, float)
            prior = np.r_[0.0, np.cumsum(seq[:-1])]
            return (prior + 0.2) / (np.arange(seq.size) + 1.0)

        out = permutation_reference(patient, y, day, running_rate, n_perm=40, seed=1)
        self.assertLess(out["observed"], 0.47)
        self.assertLess(out["reference_mean"], 0.47)
        self.assertLess(abs(out["difference"]), 0.03)


if __name__ == "__main__":
    unittest.main()
