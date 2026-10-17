# Evaluating repeated clinical forecasts from longitudinal data: between-patient bias, finite-record bias, and how to correct them

Code for the paper by Wesley Remy (2026): the Python package `wheneval` and its R port, the simulations, the cohort analyses, and the scripts that produce every table and figure.

A forecast that is issued every day for the same patients can score well because it knows which patients have events often, without knowing on which days (the between-patient bias). Scoring each patient separately removes that, but on a finished record a forecast built from the patient's other days runs low on event days, so its per-patient score falls below the usual baseline (the finite-record bias). The correction has two steps:

1. **Score within each patient.** Compute every measure on each patient's own days, so that every comparison is between days of the same patient.
2. **Compare with that patient's shuffled baseline.** Shuffle the patient's scored outcomes over the days, predict again from each shuffled record, and score again. Everything that depends on the shuffled outcomes is built again: a history rule is rebuilt, and a model fitted on the patient's own record is refitted; a model trained on other patients is applied as it is. The mean over a few hundred shuffles is the shuffled baseline; the excess over it is what the forecast knows about the days.

## Five-minute example

```
python -m pip install ./code/wheneval
```

```python
import numpy as np, pandas as pd
import wheneval as we

# 60 patients, 100 days each; outcomes are independent of the day (no daily signal)
rng = np.random.default_rng(1)
rate = rng.beta(2, 10, 60)
rows = []
for k, p in enumerate(rate):
    y = rng.binomial(1, p, 100)
    past = np.concatenate([[0], np.cumsum(y)[:-1]])
    rate_so_far = (past + 0.5) / (np.arange(100) + 1)      # event rate of the earlier days
    rows.append(pd.DataFrame({"patient_id": k, "day": np.arange(100), "outcome": y,
                              "prediction": rate_so_far}))
data = pd.concat(rows)

# Score days 30-99; predict again from every within-patient shuffle of the scored outcomes
result = we.evaluate(data, patient_id="patient_id", time="day", outcome="outcome",
                     prediction="prediction", runin=30,
                     history_rule={"name": "rate_so_far", "prior_rate": 0.5, "prior_weight": 1},
                     metrics=["auc", "average_precision", "brier_skill"], n_shuffles=200, seed=1)
print(we.to_markdown(result))
```

```
| Metric | Score (mean) | Shuffled baseline (mean) | Excess (median [IQR]) | Excess > 0 | p < 0.05 | Patients | Pooled score (for comparison) | Per-patient score (mean; for comparison) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| auc | 0.356 | 0.341 | 0.002 [-0.032, 0.061] | 52% | 5% | 58 | 0.653 | 0.356 |
| average_precision | 0.139 | 0.138 | -0.001 [-0.010, 0.011] | 43% | 3% | 58 | 0.233 | 0.139 |
| brier_skill | -0.034 | -0.033 | -0.000 [-0.001, 0.001] | 47% | 5% | 58 | 0.033 | 0.033 |

Threshold 0.153; 200 within-patient shuffles per patient (seed 1); history rule: running_rate; yardstick: patient_rate.
```

The rate so far knows nothing about the days. Pooled, its AUC is 0.65; per patient it is 0.36, below 0.5 because the record is finished; against its shuffled baseline the excess is zero.

## A fitted model in one call

For a fitted model (scikit-learn or anything with `predict_proba`), give the data, the model and its input columns. Name every input computed from the patient's earlier outcomes in `history`; wheneval checks each against the data, recomputes it on every shuffle and keeps all other inputs fixed.

```python
result = we.evaluate(data, patient_id="patient_id", time="day", outcome="outcome",
                     model=fitted_model, features=["glucose", "steps", "rate_7", "last"],
                     history={"rate_7": we.window_rate(7), "last": we.lag(1)})
```

Recipes: `we.lag(k)`, `we.rate_so_far()`, `we.window_rate(w)`, `we.window_count(w)`, `we.time_since_event()`. `history` is optional; an undeclared column that equals a standard recipe raises a warning. Inputs no recipe covers: pass a feature function, `features=make_features`, which receives one patient's rows with the shuffled outcomes and returns the model's inputs. See `code/wheneval/README.md`.

`history_rule` takes the built-in rules `"rate_so_far"`, `"window_count"` (with `window=`) and `"lagged_outcomes"` (history inputs for a model trained on other patients); `we.refit(fit_predict, folds=5)` for a model fitted on the patient's own record by cross-validation, which is refitted in every shuffle; or any function `f(y, scored, rows)` that predicts again from one patient's shuffled outcomes (`open_data/fitted_model_package.py` does this for the CTN-0094 model). With `prediction=None` the package makes the predictions itself from the rule. The default measures are the AUC, average precision, sensitivity, specificity, PPV, NPV, net benefit, Brier skill and the calibration slope; any function `f(y, p)` can be added. Pooled and per-patient scores are reported for comparison only. See `code/wheneval/README.md` and, for R (`library(wheneval)`), `code/whenevalR/README.md`.

## Figures and tables of the posted version (v32)

| Paper item | Script | Inputs and outputs |
|---|---|---|
| Figure 1 (who and when, invented example) | `figures/v19/fig_F1.py`, then `figures/v22/revise_figures.py` | `results/v19/F1_numbers.csv`; `figures/v22/F1` |
| Figure 2 (Clara's record, rate so far) | `figures/v19/fig_F2.py`, then `figures/v21/make_figures.py` | `results/v19/F2_numbers.csv`, `results/v13/A2b_flip_curve.csv`; `figures/v21/F2` |
| Figure 3 (rebuilding procedure, worked example) | `figures/v19/fig_F4.py`, then `figures/v20/draw_figures.py` | `results/v19/F4_numbers.csv`; `figures/v20/F3` |
| Figure 4 (two no-skill forecasts, four references, CTN-0094) | `figures/v19/fig_F5.py`, then `figures/v22/revise_figures.py` | `results/v19/F5_numbers.csv` (from the cohort outputs in `open_data/results/v10/`); `figures/v22/F4` |
| Figure 5 (fitted model and its patient baselines) | `code/v19_fitted_patient_baselines.py` (needs the provider data), then `figures/v23/draw_f5.py` | `results/v19/F9_patients.csv` (not distributed: one row per patient), `figures/v23/mean_auc_numbers.json`; `figures/v23/F5` |
| Figure 6, Supplement Figure S3 (seizure forecast reconstruction) | `code/v19_goldenholz_cohorts.py`, then `figures/v20/draw_figures.py` | `results/v19/F6_cohorts.csv`, `results/v14/B1_goldenholz_pooled.csv`; `figures/v20/F6`, `S3` |
| Figures 7 and 8, Supplement Tables S10-S13 (ranking simulation) | `code/ranking_sim.py --extra-pair`, `code/v19_long_record_truth.py`, `code/v19_finite_horizon.py`, `code/v19_finite_horizon_measures.py`, `code/v19_ranking_mc.py`; then `figures/v20/draw_figures.py` | `open_data/results/v10/ranking_sim.json`, `results/v19/A2_*`, `A3_*`, `A4_*`, `S6_*`; `figures/v20/F7`, `F8` |
| Mean excess with standard error (Table S14) | `code/v19_mean_excess.py` | `results/v19/A1_mean_excess.csv` |
| Block shuffles (Table S15) | `open_data/block_shuffle.py` | `open_data/results/v16/block_results.md` |
| Supplement Figure S1 (size of the finite-record bias) | `figures/v19/fig_F3.py` | `results/history_bias_grid.json`, `results/v19/F3_numbers.csv`; `figures/v19/F3` |
| Supplement Figure S2 (flipped rate so far) | `figures/v20/draw_figures.py` | `results/v13/A2b_flip_curve.csv`; `figures/v20/S2` |
| PubMed counts (Supplement S9) | `code/v19_pubmed_totals.py` | `results/v19/F8_numbers.csv` |

Earlier analyses, listed by content:

## Analyses, tables and figures

Table and figure numbers change between versions of the paper, so the analyses are listed by content. Cohort outputs are in `open_data/results/v10/`, newer analyses in `results/v13/` and `results/v14/`, and figures in `figures/v13/` (each figure script also writes the numbers it plots to `results/v13/F<n>_numbers.csv`).

| Analysis | Script | Output |
|---|---|---|
| Two-patient example (who and when) | `code/table1_two_patients.py`; figure `figures/v13/fig_F1.py` | `table1.md`; `F1` |
| Expected average precision of a random ranking | analytical; `wheneval.ap_chance` | — |
| Size of the finite-record bias (independent days) | `code/history_bias.py`, `code/v13_dials_flip.py`; figures `fig_F2.py`, `fig_F3.py` | `results/history_bias_grid.json`, `results/v13/A2*.csv`, `A3_dials.csv`; `F2`, `F3` |
| Recent-window counts with clustered events | `code/history_bias_fix.py` | `results/history_bias_fix.json` |
| Exact enumeration for Proposition 1 and its extension | `code/v14_prop1_enumeration.py` | `results/v14/B4_prop1_enumeration.md` |
| Four errors (schematic) | `figures/v13/fig_F4.py` | `F4` |
| Two no-skill forecasts on nine measures, three cohorts | `open_data/main_table_v10.py`; figure `fig_F6.py` | `main_table.md`; `F6` |
| Usual, within-patient and shuffled baselines of those forecasts | `open_data/baseline_table_v14.py` | `results/v14/B2_baselines.md` |
| Run-in rate against scored-period baselines | `open_data/yardstick_check_v10.py` | `yardstick_check.md` |
| Who-only per-week AUC and patient selection, CTN-0094 | `open_data/v13_ctn_who.py` | `results/v13/A1_who_column.csv`, `A6_selection.csv` |
| One simulated record and its shuffled baseline | `figures/v13/fig_F5.py` | `results/v13/F5_patient.csv`; `F5` |
| Gradient-boosting model, CTN-0094, three scorings | `open_data/fitted_model_package.py` (needs `open_data/table4_v10.py`); figure `fig_F7.py` | `fitted_model_package.md`, `table4.md`; `F7` |
| Ranking simulation, power and size of the permutation test | `code/ranking_sim.py --extra-pair`; figure `fig_F8.py` | `ranking_sim.md`; `F8` |
| Mean excess against long-record skill | `code/v13_excess_vs_skill.py` | `results/v13/A4_excess_vs_skill.csv` |
| Published seizure forecast without within-patient skill | `code/v13_goldenholz.py`, then `code/v14_goldenholz_pooled.py` | `results/v13/A5_goldenholz.csv`, `results/v14/B1_goldenholz_pooled.md` |
| Who, when and a shared time course in pooled, per-patient and per-day AUC | `code/v14_who_when_auc.py` | `results/v14/B3_simulations.md` (B3a) |
| Frozen against refitted model in the shuffle | `code/v14_refit_shuffle.py` | `results/v14/B3_simulations.md` (B3b) |
| Shared time course and block shuffle | `code/v14_shared_course.py` | `results/v14/B3_simulations.md` (B3c) |
| Langener and Jacobson's recent-mean baseline | `code/lj_baseline_check.py` | `lj_baseline_check.md` |
| Cohort counts | panel builders and cohort analyses | `open_data/*_results.md` |

## Cohort data

No data are redistributed. Obtain the provider files and place them as follows.

- **REPLACE-BG:** the public dataset of the [Jaeb Center](https://public.jaeb.org/dataset/546) (NCT02258373). Put HDeviceCGM.txt, HDeviceBolus.txt, HDeviceWizard.txt and HDeviceBasal.txt under `open_data/replace_bg/Data Tables/`.
- **CTN-0094:** the harmonized tables of the [public.ctn0094data package on CRAN](https://cran.r-project.org/package=public.ctn0094data). Export randomization, everybody, uds_temp, uds, tlfb and treatment as CSV files under `open_data/ctn0094/csv/`.
- **College Experience Study:** the harmonized dataset on [openESM](https://openesmdata.org/datasets/0040_nepal/); place `0040_nepal_ts.tsv` at `open_data/ces_openesm/0040_nepal_ts.tsv`.

## Reproduction

Environment: `code/environment.md`. Install with `python -m pip install -r code/requirements.txt` and `python -m pip install ./code/wheneval`. Run from the repository root.

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

# 3. Simulations
python code/table1_two_patients.py
python code/history_bias.py
python code/history_bias_fix.py
python figures/fig3_history_bias.py
python code/ranking_sim.py --extra-pair --workers 4
python code/lj_baseline_check.py

# 4. Analyses added for the current paper (after steps 1-3)
python open_data/v13_ctn_who.py
python open_data/baseline_table_v14.py
python code/v13_dials_flip.py
python code/v13_excess_vs_skill.py
python code/v13_goldenholz.py
python code/v14_goldenholz_pooled.py
python code/v14_who_when_auc.py
python code/v14_refit_shuffle.py
python code/v14_shared_course.py
python code/v14_b3_report.py
python code/v14_prop1_enumeration.py
for n in 1 2 3 4 5 6 7 8; do python figures/v13/fig_F$n.py; done
```

`ranking_sim.py` gives identical results for any `--workers`; the full run takes about 2 hours on four cores. Tests: `python -m unittest discover code/wheneval/tests` and, in R, `testthat::test_dir("code/whenevalR/tests/testthat")`.

## License

MIT; see LICENSE.

## Citation

Remy W. Evaluating repeated clinical forecasts from longitudinal data: between-patient bias, finite-record bias, and how to correct them. Preprint, 2026.

Software: Remy W. when-eval: code for "Evaluating repeated clinical forecasts from longitudinal data: between-patient bias, finite-record bias, and how to correct them". Zenodo; 2026. doi:10.5281/zenodo.23269933 (all versions; each release has its own DOI). See CITATION.cff.
