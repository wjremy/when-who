"""Re-render cases_tables.md and cases_table7.md from the saved cases_tables.json and print the Table 1 values."""
import json
import pathlib

import cases_tables as ct

results = json.loads(pathlib.Path("cases_tables.json").read_text(encoding="utf-8"))
pathlib.Path("cases_tables.md").write_text(ct.render_tables(results), encoding="utf-8")
pathlib.Path("cases_table7.md").write_text(ct.table7(results), encoding="utf-8")


def r3(value):
    return None if value is None else round(value, 3)


for cohort in ["replace_bg", "ctn_weekly_missing_excluded", "ces"]:
    model = results[cohort]["forecasts"]["gradient_boosting"]
    runin = results[cohort]["forecasts"]["runin_rate"]
    th = model["threshold"]
    print("==", cohort, "patients", results[cohort]["patients"], "both", th["eligible"]["both"],
          "rate", r3(results[cohort]["event_rate"]), "sd", r3(results[cohort]["patient_event_rate_sd"]))
    print(" AUC pooled/within", r3(model["pooled_auc"]), r3(model["within_patient_auc"]),
          "| Youden", r3(th["pooled"]["youden"]), r3(th["within_patient"]["youden"]),
          "| sens/spec pooled", r3(th["pooled"]["sensitivity"]), r3(th["pooled"]["specificity"]),
          "within", r3(th["within_patient"]["sensitivity"]), r3(th["within_patient"]["specificity"]),
          "| alarms", r3(th["alarms_per_patient_30_days"]["mean"]))
    print(" run-in rate: AUC", r3(runin["pooled_auc"]), "sens/spec", r3(runin["threshold"]["pooled"]["sensitivity"]),
          r3(runin["threshold"]["pooled"]["specificity"]), "Youden", r3(runin["threshold"]["pooled"]["youden"]))
    print(" skill", {k: r3(v) for k, v in model["brier_skill"].items()},
          "slopes", {k: r3(v) for k, v in model["calibration_slopes"].items() if "slope" in k})
    for key, value in model.items():
        if "boot" in key or "interval" in key or key.endswith("_ci"):
            print(" ", key, json.dumps(value)[:900])
