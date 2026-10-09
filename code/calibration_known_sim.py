"""Known within-patient calibration simulation used for Supplement Table S16."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from calibration_known_core import StableConditionalLogit, irls


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=7182026)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    G, T = 150, 40
    groups = np.repeat(np.arange(G), T)
    scenarios = [
        ('Calibrated', 1, 1, 0),
        ('Between miscalibrated only', .35, 1, 0),
        ('Within miscalibrated only', 1, .5, 0),
        ('Both scales miscalibrated', .35, 2, 0),
        ('Uncaptured patient intercept; within calibrated', .35, 1, 1.5),
    ]
    rows = []
    for rep in range(args.reps):
        z = rng.normal(0, 1.3, G)
        u = rng.normal(size=G)
        x = rng.normal(size=(G, T))
        x -= x.mean(axis=1, keepdims=True)
        uniform = rng.uniform(size=(G, T))
        for label, d, c, sigma in scenarios:
            a = -1 + z + sigma * u
            ap = -1 + d * z
            y = (uniform < expit(a[:, None] + x)).ravel().astype(float)
            lp = (ap[:, None] + c * x).ravel()
            mu = np.repeat(ap, T)
            dev = c * x.ravel()
            pooled = irls(np.column_stack([np.ones(len(y)), lp]), y, np.ones(len(y)))
            joint = irls(np.column_stack([np.ones(len(y)), mu, dev]), y, np.ones(len(y)))
            conditional = StableConditionalLogit(y, lp, groups).estimate(start=1 / c)
            rows.append(dict(rep=rep, scenario=label, between_scale=d, within_scale=c,
                             residual_intercept_sd=sigma, true_within=1 / c,
                             pooled=pooled[1], current=joint[2], conditional=conditional))

    draws = pd.DataFrame(rows)
    summary = []
    for label, d, c, sigma in scenarios:
        subset = draws[draws.scenario == label]
        row = dict(scenario=label, between_scale=d, within_scale=c,
                   residual_intercept_sd=sigma, true_within=1 / c, reps=args.reps)
        for key in ['pooled', 'current', 'conditional']:
            values = subset[key].dropna()
            row[key] = values.mean()
            row[key + '_mcse'] = values.std(ddof=1) / np.sqrt(len(values))
            row[key + '_bias'] = values.mean() - 1 / c
            row[key + '_rmse'] = np.sqrt(np.mean((values - 1 / c) ** 2))
            row[key + '_failed'] = args.reps - len(values)
        summary.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(OUT / 'calibration_known_simulation_results.csv', index=False)
    metadata = dict(seed=args.seed, reps=args.reps, patients=G, observations_per_patient=T,
                    b=1, x='independent standard normal, centred to exact zero mean per patient',
                    patient_z_sd=1.3)
    (OUT / 'calibration_known_simulation_metadata.json').write_text(
        json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
