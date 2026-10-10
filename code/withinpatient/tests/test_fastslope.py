"""Checks for the exact one-covariate conditional logistic implementation."""
from importlib.util import find_spec
from itertools import combinations
import unittest

import numpy as np

from withinpatient.metrics import (
    _conditional_logit_1d,
    _conditional_loglik_terms_1d,
    conditional_within_slope,
)


def brute_terms(y, x, groups, beta):
    """Enumerate every fixed-event subset to calculate exact likelihood terms."""
    loglik = score = information = 0.0
    for ix in groups:
        yg, xg = y[ix], x[ix]
        m = int(yg.sum())
        if m == 0 or m == len(ix):
            continue
        subset_sums = np.array([sum(xg[list(s)]) for s in combinations(range(len(ix)), m)], dtype=float)
        values = beta * subset_sums
        top = float(values.max())
        weights = np.exp(values - top)
        total = float(weights.sum())
        mean = float(np.dot(weights, subset_sums) / total)
        variance = float(np.dot(weights, (subset_sums - mean) ** 2) / total)
        observed = float(np.dot(yg, xg))
        loglik += beta * observed - top - np.log(total)
        score += observed - mean
        information += variance
    return loglik, score, information


class TestFastConditionalSlope(unittest.TestCase):
    def test_exact_terms_match_subset_enumeration(self):
        y = np.array([1, 0, 0, 1, 1, 0, 0, 1, 0], dtype=int)
        x = np.array([-1.3, 0.2, 1.1, 0.7, -0.5, 2.0, -0.8, 0.4, 1.6])
        groups = [np.arange(0, 4), np.arange(4, 9)]
        expected = brute_terms(y, x, groups, beta=0.73)
        actual = _conditional_loglik_terms_1d(y, x, groups, beta=0.73)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_mle_matches_dense_grid(self):
        rng = np.random.default_rng(29)
        y_parts, x_parts, groups = [], [], []
        offset = 0
        for _ in range(8):
            xg = rng.normal(size=6)
            yg = np.zeros(6, dtype=int)
            yg[rng.choice(6, size=2, replace=False)] = 1
            x_parts.append(xg)
            y_parts.append(yg)
            groups.append(np.arange(offset, offset + 6))
            offset += 6
        x = np.concatenate(x_parts)
        y = np.concatenate(y_parts)
        fit = _conditional_logit_1d(y, x, groups)
        self.assertIsNotNone(fit)
        beta, se = fit
        grid = np.linspace(-10.0, 10.0, 4001)
        values = np.array([brute_terms(y, x, groups, b)[0] for b in grid])
        grid_beta = float(grid[int(np.argmax(values))])
        self.assertAlmostEqual(beta, grid_beta, delta=0.0051)
        self.assertGreater(se, 0.0)

    def test_complement_and_large_logits_are_stable(self):
        # Both groups have predictors spanning +/-15; the first has more events than non-events.
        x = np.array([-15.0, 0.0, 15.0, 5.0, -5.0, -15.0, -5.0, 5.0, 15.0])
        y = np.array([1, 0, 1, 1, 0, 1, 1, 0, 0], dtype=int)
        groups = [np.arange(5), np.arange(5, 9)]
        actual = _conditional_loglik_terms_1d(y, x, groups, beta=0.6)
        expected = brute_terms(y, x, groups, beta=0.6)
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-10)
        self.assertTrue(np.isfinite(actual).all())
        fit = _conditional_logit_1d(y, x, groups)
        self.assertIsNotNone(fit)
        self.assertTrue(np.isfinite(fit).all())

    @unittest.skipUnless(find_spec("statsmodels"), "statsmodels is not installed")
    def test_statsmodels_cross_check(self):
        rng = np.random.default_rng(811)
        n_groups, n_each = 24, 10
        patient = np.repeat(np.arange(n_groups), n_each)
        x = rng.normal(size=len(patient))
        y = np.zeros(len(patient), dtype=int)
        for group in range(n_groups):
            chosen = rng.choice(np.arange(group * n_each, (group + 1) * n_each), size=3, replace=False)
            y[chosen] = 1
        forecast = 1.0 / (1.0 + np.exp(-x))
        exact = conditional_within_slope(patient, y, forecast)
        reference = conditional_within_slope(patient, y, forecast, method="statsmodels")
        self.assertTrue(exact["estimable"], exact["reason"])
        self.assertTrue(reference["estimable"], reference["reason"])
        self.assertAlmostEqual(exact["slope"], reference["slope"], places=5)  # statsmodels BFGS stops at ~1e-6
        self.assertAlmostEqual(exact["se"], reference["se"], places=6)


if __name__ == "__main__":
    unittest.main()
