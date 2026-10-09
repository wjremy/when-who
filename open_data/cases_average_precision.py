"""Average precision (area under the precision-recall curve), pooled and within patients, in the three cohorts.

Pooled: average precision of the model, of the run-in rate and of the rate over the scored period, against the
cohort event rate (the value of an uninformative forecast). Within patients: the model's average precision per
patient with at least one event, beside its expectation under a random ordering of that patient's days
(Supplement S6, untied form), averaged over patients.

Usage: python open_data/cases_average_precision.py   -> open_data/cases_average_precision.md
"""
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HERE = pathlib.Path(__file__).resolve().parent
COHORTS = {"REPLACE-BG": ("forecasts_replace_bg.csv", "retrospective_rate_reference"),
           "CTN-0094": ("forecasts_ctn_weekly_missing_excluded.csv", "retrospective_rate"),
           "College Experience": ("forecasts_ces.csv", "retrospective_rate")}


def null_ap(n: int, T: int) -> float:
    if T == 1:
        return 1.0
    h = np.sum(1.0 / np.arange(1, T + 1))
    return h / T + (n - 1) / (T - 1) * (1 - h / T)


def main() -> None:
    rng = np.random.default_rng(20261007)
    lines = ["# Average precision, pooled and within patients (test patients)", "",
             "| Cohort | Event rate | Pooled AP: rate over scored period | Pooled AP: run-in rate | Pooled AP: model | Patients with an event | Mean patient event rate | Within-patient AP, model | Expected under random order | Difference (95% CI) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, (file, retro) in COHORTS.items():
        df = pd.read_csv(HERE / file)
        y = df["y"].to_numpy()
        rows = []
        for _, g in df.groupby("patient"):
            n, T = int(g["y"].sum()), len(g)
            if n == 0:
                continue
            rows.append((average_precision_score(g["y"], g["gradient_boosting"]), null_ap(n, T), n / T))
        a = np.array(rows)
        diff = a[:, 0] - a[:, 1]
        bs = [rng.choice(diff, len(diff), replace=True).mean() for _ in range(2000)]
        lines.append(f"| {label} | {y.mean():.2f} | {average_precision_score(y, df[retro]):.2f} | {average_precision_score(y, df['runin_rate']):.2f} | "
                     f"{average_precision_score(y, df['gradient_boosting']):.2f} | {len(a)} | {a[:, 2].mean():.2f} | {a[:, 0].mean():.2f} | {a[:, 1].mean():.2f} | "
                     f"{diff.mean():.2f} ({np.percentile(bs, 2.5):.2f} to {np.percentile(bs, 97.5):.2f}) |")
    text = "\n".join(lines) + "\n"
    (HERE / "cases_average_precision.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
