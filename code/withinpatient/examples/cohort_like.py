"""Synthetic cohorts matched to the published cohort summary inputs."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from withinpatient import evaluate


SEED = 20261009
# Event rates, SDs, scored lengths, and run-in lengths use Supplement Table S20;
# test-patient counts use main-text Table 4. The rates and SDs in Table 4 are
# rounded versions of the supplement values.
COHORTS = (
    {"name": "REPLACE-BG", "rate": 0.059, "rate_sd": 0.052,
     "patients": 45, "scored": 208, "runin": 14},
    {"name": "CTN-0094", "rate": 0.312, "rate_sd": 0.340,
     "patients": 246, "scored": 17, "runin": 4},
    {"name": "College Experience", "rate": 0.382, "rate_sd": 0.314,
     "patients": 38, "scored": 163, "runin": 20},
)


def _expit(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -35, 35)))


def _logit(probability):
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


def _patient_rates(rng, spec):
    """Calibrate a logistic-normal rate distribution to the target moments.

    The target SD describes observed patient rates, so subtract the expected
    binomial sampling variance for the stated scored-record length first.
    The generated outcomes then add that sampling variation back.
    """
    mean = spec["rate"]
    target_observed_variance = spec["rate_sd"] ** 2
    binomial_variance = mean * (1 - mean) / spec["scored"]
    finite_factor = 1 - (spec["patients"] - 1) / (spec["patients"] * spec["scored"])
    target_rate_variance = max(
        (target_observed_variance - binomial_variance) / finite_factor, 0.0
    )
    normal_draws = rng.normal(size=spec["patients"])

    def probabilities_for(sd):
        low, high = -30.0, 30.0
        for _ in range(60):
            intercept = (low + high) / 2
            probabilities = _expit(intercept + sd * normal_draws)
            if probabilities.mean() < mean:
                low = intercept
            else:
                high = intercept
        return _expit((low + high) / 2 + sd * normal_draws)

    low, high = 0.0, 16.0
    for _ in range(60):
        sd = (low + high) / 2
        probabilities = probabilities_for(sd)
        if probabilities.var(ddof=1) < target_rate_variance:
            low = sd
        else:
            high = sd
    rates = probabilities_for((low + high) / 2)
    return rates


def _simulate(spec, rng):
    rates = _patient_rates(rng, spec)
    total = spec["runin"] + spec["scored"]
    time = np.arange(total)
    phase = np.sin(2 * np.pi * time / 12)

    # A common temporal signal is known to the second forecast. Shift its
    # intercept so the panel's expected event rate matches the cohort target.
    latent = _logit(rates)[:, None] + 0.75 * phase[None, :]
    low, high = -10.0, 10.0
    for _ in range(60):
        shift = (low + high) / 2
        if _expit(latent - shift).mean() > spec["rate"]:
            low = shift
        else:
            high = shift
    rate_and_time = _expit(latent - (low + high) / 2)
    y = rng.binomial(1, rate_and_time)
    return pd.DataFrame({
        "patient": np.repeat(np.arange(spec["patients"]), total),
        "time": np.tile(time, spec["patients"]),
        "y": y.ravel(),
        "rate_only": np.repeat(rates, total),
        "rate_and_time": rate_and_time.ravel(),
    })


def _fmt(value):
    return "—" if not np.isfinite(value) else f"{value:.3f}"


def _evaluate(data, forecast, runin):
    return evaluate(data, forecast=forecast, time="time", runin=runin,
                    metrics=["auc"], n_perm=19, seed=SEED)


def main():
    rng = np.random.default_rng(SEED)
    print("Simulated data only; no provider records are loaded.")
    print("Rates / patient-rate SDs / lengths: Supplement S20; test counts: Table 4.")
    print(f"{'cohort':<20} {'forecast':<15} {'N':>4} {'scored':>7} {'rate':>7} "
          f"{'rate SD':>8} {'median excess':>14} {'IQR':>20} {'p<.05':>8} {'pooled':>8}")

    for spec in COHORTS:
        data = _simulate(spec, rng)
        for label, forecast in (("rate only", "rate_only"),
                                ("rate + time", "rate_and_time")):
            result = _evaluate(data, forecast, spec["runin"])
            auc_result = result["metrics"]["auc"]
            summary = auc_result["summary"]
            print(f"{spec['name']:<20} {label:<15} {spec['patients']:>4} "
                  f"{spec['scored']:>7} {spec['rate']:>7.3f} {spec['rate_sd']:>8.3f} "
                  f"{_fmt(summary['median_excess']):>14} "
                  f"{_fmt(summary['iqr_excess'][0])}–{_fmt(summary['iqr_excess'][1]):>17} "
                  f"{_fmt(summary['share_p_below_0_05']):>8} "
                  f"{_fmt(auc_result['comparison']['pooled']):>8}")


if __name__ == "__main__":
    main()
