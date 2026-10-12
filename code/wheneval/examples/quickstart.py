"""Run the tailored report for fixed, externally built probability forecasts."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import wheneval as we


def main():
    rng = np.random.default_rng(17)
    n_patients, n_days = 40, 35
    patient = np.repeat(np.arange(n_patients), n_days)
    time = np.tile(np.arange(n_days), n_patients)
    rates = np.repeat(rng.beta(3, 12, n_patients), n_days)
    daily_signal = 0.08 * np.sin(2 * np.pi * time / 12)
    forecast = np.clip(rates + daily_signal, 0.01, 0.99)
    y = rng.binomial(1, forecast)
    data = pd.DataFrame({"patient_id": patient, "time": time, "outcome": y, "prediction": forecast})
    result = we.evaluate(data, n_shuffles=99, seed=17)
    print(we.to_markdown(result))


if __name__ == "__main__":
    main()
