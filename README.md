# Which patients or which days? Pooling distorts every common measure of repeated clinical forecasts, and per-patient scores need a chance level of their own

Code for the paper by Wesley Remy (2026): the Python package `withinpatient` and its R port, the simulations, the cohort analyses, and the scripts that produce every table and figure.

A forecast that is issued every day for the same patients can score well because it knows which patients have events often, without knowing on which days. Scoring each patient separately is necessary but not sufficient. The fix has two steps:

1. **Score within each patient.** Compute every measure on each patient's own days, so that every comparison is between days of the same patient.
2. **Compare with that patient's chance level.** Shuffle the patient's scored outcomes over the days, rebuild the forecast and the reference from the shuffled outcomes (the fitted model stays fixed), and score again. The mean over a few hundred shuffles is the chance level; the excess over it is what the forecast knows about the days.

## Five-minute example

```
python -m pip install ./code/withinpatient
```

```python
import numpy as np, pandas as pd
from withinpatient import evaluate, to_markdown

# 60 patients, 100 days each; outcomes are independent of the day (no daily signal)
rng = np.random.default_rng(1)
rate = rng.beta(2, 10, 60)
rows = []
for k, p in enumerate(rate):
    y = rng.binomial(1, p, 100)
    past = np.concatenate([[0], np.cumsum(y)[:-1]])
    running = (past + 0.5) / (np.arange(100) + 1)          # running event rate of earlier days
    rows.append(pd.DataFrame({"patient": k, "time": np.arange(100), "y": y, "forecast": running}))
data = pd.concat(rows)

# Score days 30-99; rebuild the running rate in every within-patient shuffle
result = evaluate(data, runin=30, rebuild={"name": "running_rate", "prior_rate": 0.5, "prior_weight": 1},
                  metrics=["auc", "average_precision", "brier_skill"], n_perm=200, seed=1)
print(to_markdown(result))
```

```
| Metric | Mean observed | Mean chance | Median excess [IQR] | Excess > 0 | p < 0.05 | Patients | Pooled | Usual per-patient mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| auc | 0.356 | 0.341 | 0.002 [-0.032, 0.061] | 52% | 5% | 58 | 0.653 | 0.356 |
| average_precision | 0.139 | 0.138 | -0.001 [-0.010, 0.011] | 43% | 3% | 58 | 0.233 | 0.139 |
| brier_skill | -0.034 | -0.033 | -0.000 [-0.001, 0.001] | 47% | 5% | 58 | 0.033 | 0.033 |
```

The running rate knows nothing about the days. Pooled, its AUC is 0.65; per patient it is 0.36, below 0.5 because the record is finished; against its chance level the excess is zero. The default measures are the AUC, average precision, sensitivity, specificity, PPV, NPV, net benefit, Brier skill and the calibration slope; any function `f(y, p)` can be added. A fitted model with covariates is rebuilt by a function `f(y, scored, rows)` that recomputes its history features from the shuffled outcomes (`open_data/fitted_model_package.py` does this for the CTN-0094 model). See `code/withinpatient/README.md` and, for R, `code/withinpatientR/README.md`.

## Tables and figures

Outputs are in `open_data/results/v10/` (the analysis version used for paper v11) unless stated otherwise.

| Item | Content | Script | Output |
|---|---|---|---|
| Table 1 | Two patients, nine measures | `code/table1_two_patients.py` | `table1.md` |
| Table 2 | The four errors | analytical | — |
| Table 3 | Per-patient chance levels | analytical (Appendix A); expected average precision of a random ranking: `withinpatient.ap_chance` | — |
| Table 4 | Two forecasts without knowledge of the weeks, CTN-0094 | `open_data/main_table_v10.py` | `main_table.md` |
| Table 5 | Ranking simulation, AUC | `code/ranking_sim.py --extra-pair` | `ranking_sim.md` |
| Table 6 | Gradient-boosting model, CTN-0094, read three ways | `open_data/fitted_model_package.py` | `fitted_model_package.md` |
| Figure 1 | Finite-record effect | `code/history_bias.py`, `code/history_bias_fix.py`, then `figures/fig3_history_bias.py` | `results/history_bias_*.json`, `figures/fig3_history_bias.pdf` |
| Table S1 | The three datasets | panel builders and cohort analyses (counts) | `open_data/*_results.md`, `main_table.md` |
| Tables S2–S4 | Main table for CTN-0094, REPLACE-BG, College Experience Study | `open_data/main_table_v10.py` | `main_table.md` |
| Table S5 | Run-in rate against scored-period references | `open_data/yardstick_check_v10.py` | `yardstick_check.md` |
| Table S6 | Per-patient AUC of history-built forecasts | `code/history_bias.py` | `results/history_bias_grid.json` |
| Table S7 | Recent-window counts with clustered events | `code/history_bias_fix.py` | `results/history_bias_fix.json` |
| Tables S8–S10 | Ranking simulation, both pairs, shortcut effect | `code/ranking_sim.py --extra-pair` | `ranking_sim.md` |
| Table S11 | Power and size of the per-patient permutation test | `code/ranking_sim.py` | `ranking_sim.md` |
| Table S12 | Recent mean without time structure | `code/lj_baseline_check.py` (prints the table) | `lj_baseline_check.md` |

`open_data/table4_v10.py` produces the nine-measure comparison of the fitted models (`table4.md`), whose model-fixed chance levels `fitted_model_package.py` checks against.

## Cohort data

No data are redistributed. Obtain the provider files and place them as follows.

- **REPLACE-BG:** the public dataset of the [Jaeb Center](https://public.jaeb.org/dataset/546) (NCT02258373). Put HDeviceCGM.txt, HDeviceBolus.txt, HDeviceWizard.txt and HDeviceBasal.txt under `open_data/replace_bg/Data Tables/`.
- **CTN-0094:** the harmonized tables of the [public.ctn0094data package on CRAN](https://cran.r-project.org/package=public.ctn0094data). Export randomization, everybody, uds_temp, uds, tlfb and treatment as CSV files under `open_data/ctn0094/csv/`.
- **College Experience Study:** the harmonized dataset on [openESM](https://openesmdata.org/datasets/0040_nepal/); place `0040_nepal_ts.tsv` at `open_data/ces_openesm/0040_nepal_ts.tsv`.

## Reproduction

Environment: `code/environment.md`. Install with `python -m pip install -r code/requirements.txt` and `python -m pip install ./code/withinpatient`. Run from the repository root.

```
# 1. Panels and cohort analyses (forecasts and fitted models)
python open_data/prepare_replace_bg_panel_v2.py
cp open_data/replace_bg_panel_v2.csv open_data/replace_bg_panel.csv
python open_data/prep_ctn0094_panels.py
cp open_data/ctn0094_daily_panel_rebuilt.csv open_data/ctn0094_daily_panel.csv
cp open_data/ctn0094_weekly_panel_rebuilt.csv open_data/ctn0094_weekly_panel.csv
python open_data/replace_bg_analysis.py
python open_data/ctn0094_analysis.py
python open_data/ces_analysis.py
python open_data/cases_tables.py
python open_data/cases_extra.py

# 2. Cohort tables (table4_v10 and main_table_v10 take up to about an hour each)
python open_data/table4_v10.py
python open_data/main_table_v10.py
python open_data/yardstick_check_v10.py
python open_data/fitted_model_package.py

# 3. Simulations and Figure 1
python code/table1_two_patients.py
python code/history_bias.py
python code/history_bias_fix.py
python figures/fig3_history_bias.py
python code/ranking_sim.py --extra-pair --workers 4
python code/lj_baseline_check.py
```

`ranking_sim.py` gives identical results for any `--workers`; the full run takes about 2 hours on four cores. Tests: `python -m unittest discover code/withinpatient/tests` and, in R, `testthat::test_dir("code/withinpatientR/tests/testthat")`.

## License

MIT; see LICENSE.

## Citation

Remy W. Which patients or which days? Pooling distorts every common measure of repeated clinical forecasts, and per-patient scores need a chance level of their own. Preprint, 2026.

Software: Remy W. when-who: code for "Which patients or which days?". Zenodo; 2026. doi:10.5281/zenodo.23269933 (all versions; each release has its own DOI). See CITATION.cff.
