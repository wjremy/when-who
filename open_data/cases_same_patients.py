"""Pooled metrics restricted to the patients who enter the within-patient metrics.

Within-patient AUC, sensitivity and specificity are defined for test patients with both outcomes in the
scored period. This script recomputes the pooled AUC, the benchmark A0 and the pooled threshold metrics of
the gradient-boosting model on those patients only, so that the pooled and within-patient values can be
compared on one set of patients. It also gives the equally weighted mean over patients of the Brier
improvement of the model on each rate reference (the skill in Table 1 uses summed losses).

Usage: python open_data/cases_same_patients.py   -> open_data/cases_same_patients.md
"""
import json
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
COHORTS = {"REPLACE-BG": ("replace_bg", "forecasts_replace_bg.csv", "retrospective_rate_reference", "expanding_rate"),
           "CTN-0094": ("ctn_weekly_missing_excluded", "forecasts_ctn_weekly_missing_excluded.csv", "retrospective_rate", "updating_rate"),
           "College Experience": ("ces", "forecasts_ces.csv", "retrospective_rate", "expanding_rate")}


def auc(y: np.ndarray, s: np.ndarray) -> float:
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pooled(df: pd.DataFrame, col: str, threshold: float) -> dict:
    y = df["y"].to_numpy()
    alarm = df[col].to_numpy() >= threshold
    sens = float(alarm[y == 1].mean())
    spec = float((~alarm)[y == 0].mean())
    return {"auc": auc(y, df[col].to_numpy()), "sens": sens, "spec": spec, "youden": sens + spec - 1}


def within(df: pd.DataFrame, col: str, threshold: float) -> dict:
    rows = []
    for _, g in df.groupby("patient"):
        y = g["y"].to_numpy()
        alarm = g[col].to_numpy() >= threshold
        rows.append((auc(y, g[col].to_numpy()), alarm[y == 1].mean(), (~alarm)[y == 0].mean()))
    a = np.array(rows, dtype=float)
    return {"auc": float(a[:, 0].mean()), "youden": float((a[:, 1] + a[:, 2] - 1).mean())}


def main() -> None:
    tables = json.loads((HERE / "cases_tables.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(20261007)
    out = ["# Pooled metrics on the patients who enter the within-patient metrics (gradient-boosting model, test patients)", "",
           "| Cohort | Patients: all / both outcomes | Event rate: all / restricted | A0: all / restricted | Pooled AUC: all / restricted (95% CI) | Within-patient AUC | Pooled sens; spec, restricted | Pooled Youden: all / restricted (95% CI) | Within-patient Youden |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    brier = ["", "## Brier improvement of the model, equally weighted over patients", "",
             "Mean over all test patients of (Brier score of the reference minus Brier score of the model), each patient's score being the mean over that patient's observations; 95% interval from 2,000 resamples of patients. The last column is the share of patients with a positive difference.", "",
             "| Cohort | Reference | Mean per-patient improvement (95% CI) | Mean per-patient Brier of reference | Patients improved |", "| --- | --- | ---: | ---: | ---: |"]
    for label, (key, file, retro, updated) in COHORTS.items():
        df = pd.read_csv(HERE / file)
        thr = tables[key]["forecasts"]["gradient_boosting"]["threshold"]["threshold"]
        g = df.groupby("patient")["y"]
        both = g.transform("min").ne(g.transform("max"))
        r = df[both].copy()
        pa, pr, w = pooled(df, "gradient_boosting", thr), pooled(r, "gradient_boosting", thr), within(r, "gradient_boosting", thr)
        ids = r["patient"].unique()
        groups = {k: v for k, v in r.groupby("patient")}
        boot = []
        for _ in range(1000):
            b = pd.concat([groups[k] for k in rng.choice(ids, len(ids), replace=True)])
            p = pooled(b, "gradient_boosting", thr)
            boot.append((p["auc"], p["youden"]))
        lo, hi = np.percentile(np.array(boot), [2.5, 97.5], axis=0)
        out.append(f"| {label} | {df['patient'].nunique()} / {len(ids)} | {df['y'].mean():.2f} / {r['y'].mean():.2f} | "
                   f"{auc(df['y'].to_numpy(), df[retro].to_numpy()):.2f} / {auc(r['y'].to_numpy(), r[retro].to_numpy()):.2f} | "
                   f"{pa['auc']:.2f} / {pr['auc']:.2f} ({lo[0]:.2f} to {hi[0]:.2f}) | {w['auc']:.2f} | {pr['sens']:.2f}; {pr['spec']:.2f} | "
                   f"{pa['youden']:.2f} / {pr['youden']:.2f} ({lo[1]:.2f} to {hi[1]:.2f}) | {w['youden']:.2f} |")
        for name, col in (("run-in rate", "runin_rate"), ("updated rate", updated), ("rate over the scored period", retro)):
            d = df.assign(ref=(df[col] - df["y"]) ** 2, mod=(df["gradient_boosting"] - df["y"]) ** 2).groupby("patient")[["ref", "mod"]].mean()
            diff = (d["ref"] - d["mod"]).to_numpy()
            bs = [rng.choice(diff, len(diff), replace=True).mean() for _ in range(2000)]
            brier.append(f"| {label} | {name} | {diff.mean():.3f} ({np.percentile(bs, 2.5):.3f} to {np.percentile(bs, 97.5):.3f}) | {d['ref'].mean():.3f} | {100 * (diff > 0).mean():.0f}% |")
    text = "\n".join(out + brier) + "\n"
    (HERE / "cases_same_patients.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
