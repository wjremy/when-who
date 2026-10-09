# Which patients or which days? Separating within-patient and between-patient performance of clinical forecasts

[![DOI](https://zenodo.org/badge/1412370860.svg)](https://doi.org/10.5281/zenodo.23269933)

This repository accompanies the paper “Which patients or which days? Separating within-patient and between-patient performance of clinical forecasts” by Wesley Remy (Department of Computer Science, ETH Zurich, Zurich, Switzerland; ORCID 0009-0007-6431-4200). The paper shows how pooled scores of repeated clinical forecasts mix comparisons within and between patients, and gives quantities for evaluating whether a forecast distinguishes a patient's higher-risk days from other patients' risk at a given time.

## Repository contents

| Path | Contents |
| --- | --- |
| code/ | Synthetic methods, simulations, supporting analyses, environment requirements, and the withinpatient package. |
| figures/ | Scripts and rendered files for the three main-text figures. |
| open_data/ | Cohort preparation and analysis scripts, plus aggregate tables and reports. Provider data, patient-day panels, and forecast exports are not included. |
| results/ | Frozen aggregate outputs used by the synthetic analyses and figures. |
| Repository root | This README, the MIT license, and citation metadata. |

## Public cohort data

The cohort source files are not part of this repository. Obtain them from their public sources and place them at the paths below before running the cohort analyses.

- **REPLACE-BG:** download the public dataset from the [Jaeb Center dataset page](https://public.jaeb.org/dataset/546) (study NCT02258373). Put HDeviceCGM.txt, HDeviceBolus.txt, HDeviceWizard.txt, and HDeviceBasal.txt under open_data/replace_bg/Data Tables/.
- **NIDA CTN-0094:** obtain the harmonized public tables from the [public.ctn0094data package on CRAN](https://cran.r-project.org/package=public.ctn0094data) and its [documentation](https://ctn-0094.github.io/public.ctn0094data/). Export randomization, everybody, uds_temp, uds, tlfb, and treatment as CSV files under open_data/ctn0094/csv/.
- **College Experience Study:** download the harmonized dataset from its [openESM page](https://openesmdata.org/datasets/0040_nepal/) and place 0040_nepal_ts.tsv at open_data/ces_openesm/0040_nepal_ts.tsv.

## Environment and reproduction

Run commands from the repository root in PowerShell. The analysis environment uses Python 3.12; install the listed dependencies and the package:

    python -m pip install -r code/requirements.txt
    python -m pip install ./code/withinpatient

The calibration-history routines build a small C helper on first use and require a GCC-compatible compiler on PATH (MinGW-w64 GCC on Windows). Run compute-intensive jobs sequentially.

First rebuild the REPLACE-BG and CTN-0094 panels from the provider files, then run the three cohort analyses. The College Experience analysis reads the openESM TSV directly.

    python open_data/prepare_replace_bg_panel_v2.py
    Copy-Item open_data/replace_bg_panel_v2.csv open_data/replace_bg_panel.csv -Force

    python open_data/prep_ctn0094_panels.py
    Copy-Item open_data/ctn0094_daily_panel_rebuilt.csv open_data/ctn0094_daily_panel.csv -Force
    Copy-Item open_data/ctn0094_weekly_panel_rebuilt.csv open_data/ctn0094_weekly_panel.csv -Force

    python open_data/replace_bg_analysis.py
    python open_data/ctn0094_analysis.py
    python open_data/ces_analysis.py

Those analyses create local panel and forecast files used by the aggregate producers. These person-level intermediate files are not included in the public repository.

Run the cohort tables and supplementary analyses after the cohort scripts:

    python open_data/cases_tables.py
    python open_data/cases_extra.py
    python open_data/cases_average_precision.py
    python open_data/cases_predictive_values.py
    python open_data/cases_same_patients.py
    python open_data/cases_resplit.py --cohort replace_bg
    python open_data/cases_resplit.py --cohort ctn_weekly_missing_excluded
    python open_data/cases_resplit.py --cohort ces
    python open_data/cases_threshold_sensitivity.py
    python open_data/realtime_analysis.py
    python open_data/skill_analysis.py
    python open_data/cases_information_audit.py
    python open_data/cases_review_v71.py
    python open_data/cases_model_fixed_permutation.py

The last three commands produce the v7 information audit, rate-reference and calibration-noise summaries, and model-fixed permutation results. cases_model_fixed_permutation.py uses the aggregate output from cases_extra.py.

For the calibration-history supplement, run the producers in this order:

    python open_data/calhist/fit_core.py
    python open_data/calhist/cohort_analysis.py --nboot 2000 --seed 20261007
    python open_data/calhist/exact_counterexample.py
    python open_data/calhist/simulation.py
    python open_data/calhist/clipping_sensitivity.py
    python open_data/calhist/figures.py
    python open_data/calhist/verify.py

The synthetic methods and figure commands are:

    python code/theory_check.py
    python code/supp_tables.py --table all
    python code/calibration.py
    python code/history_bias.py
    python code/history_bias_fix.py
    python figures/fig_concept_v5.py
    python figures/fig3_history_bias.py
    python code/cohort_curves.py
    python figures/fig_central_v6.py
    python code/importance_v2.py

Tables 1, 2, 5, S1, S8, and S31 are manuscript-authored summaries; code/theory_check.py checks calculations supporting Table 3. The current source tree does not contain generators for the exact robustness scenarios in Table S12 or the known-calibration scenarios in Table S16, so those two tables cannot be regenerated from the listed commands. Other commands regenerate the script-produced tables and figures from their required source data and aggregate inputs.

## The withinpatient package

Install from the repository root:

    python -m pip install ./code/withinpatient

The package computes pooled, within-patient, and per-time-point quantities from a patient identifier, time, binary outcome, and forecast probability. The synthetic example has five patients so each time point meets the default minimum size for per-time-point AUC.

    import pandas as pd
    from withinpatient import evaluate

    outcomes = {
        "A": [0, 1, 0, 1],
        "B": [0, 0, 1, 1],
        "C": [0, 1, 1, 0],
        "D": [1, 0, 1, 0],
        "E": [1, 1, 0, 0],
    }
    forecasts = {
        "A": [0.1, 0.8, 0.2, 0.7],
        "B": [0.2, 0.3, 0.6, 0.7],
        "C": [0.3, 0.8, 0.7, 0.2],
        "D": [0.8, 0.2, 0.7, 0.3],
        "E": [0.7, 0.8, 0.2, 0.3],
    }
    data = pd.DataFrame([
        {"patient_id": patient, "time": day, "event": outcomes[patient][day - 1],
         "risk": forecasts[patient][day - 1]}
        for patient in outcomes
        for day in range(1, 5)
    ])
    result = evaluate(
        data,
        patient="patient_id",
        time="time",
        y="event",
        forecast="risk",
        threshold=0.5,
    )
    print(result["auc"]["pooled"]["value"])
    print(result["auc"]["within_patient"]["value"])
    print(result["auc"]["per_time_point"]["value"])
    print(result["auc"]["A0"])

## License

Code is licensed under the MIT License; see LICENSE.

## Methods citation

Paste-ready sentence:

> Following Remy (2026), we report discrimination per patient and per time point, the pooled AUC beside $A_0$ (the pooled AUC that patients' event rates alone attain), Brier skill against each patient's own event rate and calibration-in-the-large per patient, computed with the withinpatient package (version 1.0.0; https://github.com/wjremy/when-who).

Reference: Remy W. Which patients or which days? Separating within-patient and between-patient performance of clinical forecasts. Preprint, 2026.

Software: Remy W. when-who: code for “Which patients or which days?” Version 1.0.0. Zenodo; 2026. doi:10.5281/zenodo.23269934. The concept DOI 10.5281/zenodo.23269933 always resolves to the latest version.

The preprint DOI will be added once the preprint is posted.
