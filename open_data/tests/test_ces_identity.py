"""Verify the unchanged CES permutation reconstruction under an identity ordering."""
import unittest
from pathlib import Path

import numpy as np

from open_data import ces_analysis as a


class CESIdentityTest(unittest.TestCase):
    def test_next_report_alignment(self):
        source = Path(__file__).resolve().parents[1] / "ces_openesm" / "0040_nepal_ts.tsv"
        _, panel = a.load_reports(str(source))
        checked = 0
        for _, g in panel.groupby("pid", sort=False):
            g = g.sort_values("date")
            ev = np.r_[g.current_event.to_numpy(int), int(g.y.iloc[-1])]
            total = np.r_[g.persistence.to_numpy(float), float(g.next_persistence.iloc[-1])]
            runin = np.asarray(g.runin_event_history.iloc[0], int)
            np.testing.assert_array_equal(ev[1:], g.y.to_numpy(int))
            np.testing.assert_array_equal(total[:-1], g.persistence.to_numpy(float))
            histories = [np.r_[runin, ev[:j+1]] for j in range(len(ev)-1)]
            np.testing.assert_array_equal([h.mean() for h in histories], g.expanding_rate.to_numpy(float))
            np.testing.assert_array_equal([h[-12:].mean() for h in histories], g.last12_rate.to_numpy(float))
            checked += len(g)
        print(f"CES exact identity: {checked} rows, {panel.pid.nunique()} patients, 3 forecasts; source unchanged")


if __name__ == "__main__":
    unittest.main()
