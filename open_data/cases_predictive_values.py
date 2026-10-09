"""Predictive values of an alarm, pooled and within patients, in the three cohorts.

For the gradient-boosting model and for the alarm from the run-in rate, at the threshold used in Table 1:
pooled PPV and NPV against the cohort event rate, and for every patient the PPV minus that patient's own
event rate over the scored period (and the NPV minus one minus that rate), averaged over patients with at
least one alarm (one non-alarm). Intervals: 2,000 bootstrap resamples of patients.

Usage: python open_data/cases_predictive_values.py   -> open_data/cases_predictive_values.md
"""
import json
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
COHORTS = {"REPLACE-BG": ("replace_bg", "forecasts_replace_bg.csv"),
           "CTN-0094": ("ctn_weekly_missing_excluded", "forecasts_ctn_weekly_missing_excluded.csv"),
           "College Experience": ("ces", "forecasts_ces.csv")}
FORECASTS = {"model": "gradient_boosting", "run-in rate": "runin_rate"}


def per_patient(df: pd.DataFrame, col: str, threshold: float) -> pd.DataFrame:
    d = df.assign(alarm=df[col] >= threshold)
    g = d.groupby("patient")
    out = pd.DataFrame({"n": g.size(), "events": g["y"].sum(), "alarms": g["alarm"].sum(),
                        "tp": (d["alarm"] & (d["y"] == 1)).groupby(d["patient"]).sum(),
                        "tn": ((~d["alarm"]) & (d["y"] == 0)).groupby(d["patient"]).sum()})
    out["rate"] = out["events"] / out["n"]
    return out


def summary(pp: pd.DataFrame) -> dict:
    n, ev, al = pp["n"].sum(), pp["events"].sum(), pp["alarms"].sum()
    a = pp[pp["alarms"] > 0]
    q = pp[pp["alarms"] < pp["n"]]
    return {"rate": ev / n,
            "ppv_pooled": pp["tp"].sum() / al if al else np.nan,
            "npv_pooled": pp["tn"].sum() / (n - al) if n > al else np.nan,
            "ppv_gain_within": float((a["tp"] / a["alarms"] - a["rate"]).mean()),
            "npv_gain_within": float((q["tn"] / (q["n"] - q["alarms"]) - (1 - q["rate"])).mean()),
            "patients_with_alarm": int(len(a)), "patients_with_non_alarm": int(len(q))}


def main() -> None:
    tables = json.loads((HERE / "cases_tables.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(20261007)
    lines = ["# Predictive values, pooled and within patients (test patients)", "",
             "Gain = predictive value minus the event rate (PPV) or minus one minus the event rate (NPV). Pooled: against the cohort rate. "
             "Within: each patient's value against that patient's own rate over the scored period, averaged over patients with at least one alarm (PPV) or one non-alarm (NPV). "
             "Intervals from 2,000 bootstrap resamples of patients.", "",
             "| Cohort | Alarm | Event rate | Pooled PPV | Pooled PPV gain | Within-patient PPV gain | Pooled NPV | Pooled NPV gain | Within-patient NPV gain | Patients with alarm / non-alarm |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, (key, file) in COHORTS.items():
        df = pd.read_csv(HERE / file)
        pcol = "patient" if "patient" in df.columns else [c for c in df.columns if c.lower() in ("pid", "patient_id", "id", "who", "ptid")][0]
        ycol = "y" if "y" in df.columns else [c for c in df.columns if c.lower() in ("event", "outcome", "label")][0]
        df = df.rename(columns={pcol: "patient", ycol: "y"})
        for name, col in FORECASTS.items():
            threshold = tables[key]["forecasts"][col]["threshold"]["threshold"]
            pp = per_patient(df, col, threshold)
            s = summary(pp)
            boot = []
            ids = np.arange(len(pp))
            for _ in range(2000):
                b = summary(pp.iloc[rng.choice(ids, len(ids), replace=True)])
                boot.append((b["ppv_pooled"] - b["rate"], b["ppv_gain_within"], b["npv_pooled"] - (1 - b["rate"]), b["npv_gain_within"]))
            lo, hi = np.nanpercentile(np.array(boot, dtype=float), [2.5, 97.5], axis=0)
            f = lambda v, i: f"{v:.2f} ({lo[i]:.2f} to {hi[i]:.2f})"
            lines.append(f"| {label} | {name} | {s['rate']:.2f} | {s['ppv_pooled']:.2f} | {f(s['ppv_pooled'] - s['rate'], 0)} | {f(s['ppv_gain_within'], 1)} | "
                         f"{s['npv_pooled']:.2f} | {f(s['npv_pooled'] - (1 - s['rate']), 2)} | {f(s['npv_gain_within'], 3)} | "
                         f"{s['patients_with_alarm']} / {s['patients_with_non_alarm']} |")
    (HERE / "cases_predictive_values.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
