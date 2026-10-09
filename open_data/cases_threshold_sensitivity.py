import json, sys, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code" / "withinpatient" / "src"))
from withinpatient import threshold_metrics
out = {}
for c in ["replace_bg", "ctn_weekly_missing_excluded", "ces"]:
    df = pd.read_csv(HERE / f"forecasts_{c}.csv")
    base = float(df["threshold"].iloc[0]); rows = []
    for mult in [0.5, 0.75, 1.0, 1.5, 2.0]:
        thr = min(base * mult, 0.99)
        for f in ["gradient_boosting", "runin_rate"]:
            tm = threshold_metrics(df["patient"], df["y"], df[f], thr)
            rows.append({"multiple_of_training_rate": mult, "threshold": round(thr, 4), "forecast": f,
                         "pooled_youden": round(tm["pooled"]["youden"], 3), "within_youden": round(tm["within_patient"]["youden"], 3),
                         "pooled_sens": round(tm["pooled"]["sensitivity"], 3), "pooled_spec": round(tm["pooled"]["specificity"], 3),
                         "within_sens": round(tm["within_patient"]["sensitivity"], 3), "within_spec": round(tm["within_patient"]["specificity"], 3),
                         "alarms_per_30": round(tm["alarms_per_patient_30_days"]["mean"], 1)})
    out[c] = rows
    print("==", c, "training rate", round(base, 3))
    for r in rows: print("  x%.2f %-18s pooled J %.3f  within J %.3f  alarms/30 %.1f" % (r["multiple_of_training_rate"], r["forecast"], r["pooled_youden"], r["within_youden"], r["alarms_per_30"]))
json.dump(out, open(HERE / "cases_threshold_sensitivity.json", "w"), indent=1)
