"""Show why a history forecast needs a rebuilt chance level."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from withinpatient import evaluate

SEED = 20261006
N_PATIENTS = 250
RECORD_LENGTH = 100
RUNIN = 30
EVENT_RATE = 0.15
N_PERM = 49


def running_rate(outcomes):
    """Use earlier outcomes with one prior observation at the known rate."""
    y = np.asarray(outcomes, dtype=float)
    earlier = np.r_[0.0, np.cumsum(y[:-1])]
    return (EVENT_RATE + earlier) / (1.0 + np.arange(len(y)))


def main():
    rng = np.random.default_rng(SEED)
    y = rng.binomial(1, EVENT_RATE, size=(N_PATIENTS, RECORD_LENGTH))
    data = pd.DataFrame({
        "patient": np.repeat(np.arange(N_PATIENTS), RECORD_LENGTH),
        "time": np.tile(np.arange(RECORD_LENGTH), N_PATIENTS),
        "y": y.ravel(),
        "forecast": np.vstack([running_rate(row) for row in y]).ravel(),
    })
    result = evaluate(
        data, time="time", runin=RUNIN,
        rebuild={"name": "running_rate", "prior_rate": EVENT_RATE, "prior_weight": 1},
        yardstick="running_rate", metrics=["auc"], n_perm=N_PERM, seed=SEED,
    )
    tailored = result["metrics"]["auc"]
    per_patient = list(tailored["per_patient"].values())
    print("Independent Bernoulli(0.15) outcomes; 250 finished 100-day records; first 30 days are run-in.")
    print("Each scored-outcome shuffle rebuilds the running forecast and yardstick.")
    print(f"Mean observed patient AUC: {_fmt(np.mean([x['observed'] for x in per_patient]))}")
    print(f"Mean rebuilt-history chance: {_fmt(np.mean([x['chance_level'] for x in per_patient]))}")
    print(f"Median patient excess: {_fmt(tailored['summary']['median_excess'])}")
    print(f"Patient excess IQR: {_fmt(tailored['summary']['iqr_excess'][0])} to {_fmt(tailored['summary']['iqr_excess'][1])}")


def _fmt(value):
    return "NA" if not np.isfinite(value) else f"{value:.3f}"


if __name__ == "__main__":
    main()
