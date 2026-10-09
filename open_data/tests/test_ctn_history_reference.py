import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "open_data"
PORT = ROOT / "out_permport"
sys.path.insert(0, str(SOURCE))

from open_data import ctn0094_analysis as a
from history_kernel import history_plan, rebuild_history, NAMES


VARIANTS = ["weekly_missing_excluded", "weekly_missing_positive", "daily_self_reported"]


def load_variant(variant):
    weekly = variant.startswith("weekly")
    stem = "weekly" if weekly else "daily"
    raw = pd.read_csv(SOURCE / f"ctn0094_{stem}_panel.csv", float_precision="round_trip")
    slot_cols = ["who", "history_initial_events"]
    if weekly:
        slot_cols.append("history_initial_screens")
    seeds = pd.read_csv(PORT / f"ctn0094_{stem}_panel_rebuilt.csv", usecols=slot_cols)
    seeds = seeds.drop_duplicates("who").set_index("who")
    for col in seeds:
        raw[col] = raw.who.map(seeds[col])
    event_col = "event_missing_positive" if variant == "weekly_missing_positive" else "event"
    if event_col == "event_missing_positive":
        raw = a.rebuild_missing_positive_history(raw)
    split = a.split_people(raw)
    train = raw.loc[raw.who.isin(split["train"]) & raw[event_col].notna(), event_col].astype(int)
    data = a.add_references(raw, event_col, float(train.mean()))
    scored = data.loc[data[event_col].notna()].copy()
    test = scored.loc[scored.who.isin(split["test"])].copy()
    return scored, test, event_col


class CTNIdentityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panels = {v: load_variant(v) for v in VARIANTS}

    def test_existing_panel_columns_unchanged(self):
        for stem in ["weekly", "daily"]:
            original = pd.read_csv(SOURCE / f"ctn0094_{stem}_panel.csv", float_precision="round_trip")
            rebuilt = pd.read_csv(PORT / f"ctn0094_{stem}_panel_rebuilt.csv", float_precision="round_trip")
            pd.testing.assert_frame_equal(original, rebuilt[original.columns], check_exact=True)

    def test_all_variants_identity_exact(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                data, _, event_col = self.panels[variant]
                count = 0
                for _, g in data.groupby("who", sort=False):
                    plan = history_plan(g, event_col, data.attrs["history_fallback"])
                    target = g[event_col].to_numpy(int)
                    scores = rebuild_history(plan, target)
                    for name in NAMES:
                        np.testing.assert_array_equal(scores[name], g[name].to_numpy(float),
                                                      err_msg=f"{variant} patient {g.who.iloc[0]} {name}")
                    count += len(g)
                print(f"{variant}: exact identity {count} rows, {data.who.nunique()} patients, 3 forecasts")

    def test_identity_through_actual_history_bias(self):
        real_rng = np.random.default_rng

        class IdentityRNG:
            def shuffle(self, values):
                pass

        for variant in VARIANTS:
            data, _, event_col = self.panels[variant]
            ids = [pid for pid, g in data.groupby("who", sort=False) if g[event_col].nunique() == 2][:5]
            subset = data.loc[data.who.isin(ids)].copy()
            seed = a.SEED + 101
            with patch.object(a.np.random, "default_rng",
                              side_effect=lambda z: IdentityRNG() if z == seed else real_rng(z)):
                out = a.history_bias(subset, subset[event_col].astype(int), subset.who, n_perm=2, seed=seed)
            for name in NAMES:
                self.assertEqual(out[name]["permutation_reference_mean"], out[name]["observed_within_auc"])
                self.assertEqual(out[name]["observed_minus_reference"], 0.)

    def test_shuffle_uses_same_calendar_slots_and_fixed_initialisation(self):
        rng = np.random.default_rng(42)
        for variant in VARIANTS:
            data, _, event_col = self.panels[variant]
            for _, g in data.groupby("who", sort=False):
                plan = history_plan(g, event_col, data.attrs["history_fallback"])
                target = g[event_col].to_numpy(int)
                shuffled = target.copy()
                rng.shuffle(shuffled)
                scores = rebuild_history(plan, shuffled)
                identity = rebuild_history(plan, target)
                for name in NAMES:
                    self.assertEqual(scores[name][0], identity[name][0])
                np.testing.assert_array_equal(np.sort(shuffled), np.sort(target))
                self.assertTrue(all(len(scores[name]) == len(target) for name in NAMES))


if __name__ == "__main__":
    unittest.main()
