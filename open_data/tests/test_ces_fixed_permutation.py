import unittest
import numpy as np

from open_data import ces_analysis as a


class CESFixedPermutationTest(unittest.TestCase):
    def test_identity_and_fixed_report_21(self):
        ev = np.array([1, 0, 1, 0, 1, 1], dtype=int)  # reports 21..26
        totals = np.array([4., 2., 5., 1., 6., 3.])
        class IdentityRNG:
            def permutation(self, values):
                return np.asarray(values)
        evp, tp = a.permute_scored_reports(ev, totals, IdentityRNG())
        np.testing.assert_array_equal(evp, ev)
        np.testing.assert_array_equal(tp, totals)
        self.assertEqual(evp[0], ev[0])
        self.assertEqual(evp[1:].sum(), ev[1:].sum())
        # A real shuffle can reorder scored reports, but not the origin or totals.
        evq, tq = a.permute_scored_reports(ev, totals, np.random.default_rng(21))
        self.assertEqual(evq[0], ev[0])
        self.assertEqual(tq[0], totals[0])
        np.testing.assert_array_equal(np.sort(evq[1:]), np.sort(ev[1:]))
        np.testing.assert_array_equal(np.sort(tq[1:]), np.sort(totals[1:]))
        # The first forecast (report 22) is based on the fixed run-in + report 21.
        runin = np.array([0, 1] * 10)
        self.assertEqual(np.mean(np.r_[runin, evp[0]]), np.mean(np.r_[runin, ev[0]]))

    def test_identity_reference_equals_observed(self):
        reports, panel = a.load_reports(str(a.DATA))
        split = a.split_people(panel)
        panel['split'] = panel.pid.map(split)
        test = panel[panel.split == 'test'].copy().reset_index(drop=True)
        class IdentityRNG:
            def permutation(self, values):
                return np.asarray(values)
        real_rng = a.np.random.default_rng
        # The same RNG seed is used inside permutation_bias; its identity stub
        # leaves the scored target order unchanged for all 200 permutations.
        class IdentityRNG2:
            def permutation(self, values):
                return np.asarray(values)
        a.np.random.default_rng = lambda seed=None: IdentityRNG2() if seed == a.SEED + 50 else real_rng(seed)
        try:
            result = a.permutation_bias(test, ['expanding_rate', 'last12_rate', 'persistence'], nperm=2)
        finally:
            a.np.random.default_rng = real_rng
        for metric, item in result.items():
            # Averaging the identical per-permutation AUCs can differ in the last bit.
            self.assertAlmostEqual(item['observed_within_auc'], item['permutation_reference_within_auc'], places=12)
            self.assertAlmostEqual(item['observed_minus_reference'], 0., places=12)


if __name__ == '__main__':
    unittest.main()
