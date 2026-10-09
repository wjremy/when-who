import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from open_data import replace_bg_analysis as a


PANEL = Path(__file__).resolve().parents[1] / "replace_bg_panel.csv"


def load():
    panel = pd.read_csv(PANEL, float_precision="round_trip")
    panel["pid"] = panel.pid.astype(str)
    panel["persistence"] = panel.groupby("pid").y.shift(1)
    return panel


class IdentityTest(unittest.TestCase):
    def test_all_observed_forecasts_exact(self):
        panel = load()
        checked = 0
        for _, g in panel.sort_values(["pid", "night_index"]).groupby("pid", sort=False):
            target, rebuilt = a.rebuild_history(g)
            observed = g.loc[~g.model_runin.astype(bool)]
            np.testing.assert_array_equal(target, observed.y.to_numpy(int))
            for name, column in [
                ("recent_28_night_rate", "history_rate_28"),
                ("expanding_rate", "expanding_rate"),
                ("persistence", "persistence"),
            ]:
                np.testing.assert_array_equal(rebuilt[name], observed[column].to_numpy(float))
            checked += len(target)
        print(f"Exact identity: {checked} scored rows, 224 patients, 3 forecasts")

    def test_shuffle_preserves_runin_and_scored_counts(self):
        rng = np.random.default_rng(42)
        for _, g in load().sort_values(["pid", "night_index"]).groupby("pid", sort=False):
            original, identity = a.rebuild_history(g)
            target, shuffled = a.rebuild_history(g, rng)
            np.testing.assert_array_equal(np.sort(target), np.sort(original))
            for name in identity:
                self.assertEqual(shuffled[name][0], identity[name][0])


if __name__ == "__main__":
    unittest.main()
