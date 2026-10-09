"""Recompute the synthetic training-experiment summaries from saved CSVs.

Only NumPy and pandas are needed. This reads existing outputs; it does not run
the training experiment or regenerate any synthetic data.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _fmt(value, digits):
    return f"{float(value):.{digits}f}"


def _fmt_grid(values):
    labels = []
    for value in sorted(values):
        if float(value).is_integer():
            labels.append(str(int(value)))
        else:
            labels.append(str(float(value)))
    return "{" + ", ".join(labels) + "}"


def _spearman(x, y):
    """Spearman correlation using pandas average ranks and NumPy only."""
    rank_x = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    rank_y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def _static_assignments(path, names):
    """Read literal assignments from source without importing or running it."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        for target in targets:
            if not isinstance(target, ast.Name) or target.id not in names:
                continue
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "dict":
                found[target.id] = {
                    keyword.arg: ast.literal_eval(keyword.value)
                    for keyword in value.keywords
                    if keyword.arg is not None
                }
            else:
                found[target.id] = ast.literal_eval(value)
    missing = set(names) - set(found)
    if missing:
        raise ValueError(f"Missing static definitions in {path.name}: {sorted(missing)}")
    return found


def _add(rows, quantity, printed, recomputed, matches):
    rows.append(
        {
            "quantity": quantity,
            "printed": printed,
            "recomputed": recomputed,
            "match": matches,
        }
    )


def _add_numeric(rows, quantity, printed_value, recomputed_value, digits):
    printed = _fmt(printed_value, digits)
    recomputed = _fmt(recomputed_value, digits)
    _add(rows, quantity, printed, recomputed, "Yes" if printed == recomputed else "No")


def main():
    main_dir = ROOT / "results" / "kaggle_runs" / "main"
    followup_dir = ROOT / "results" / "kaggle_runs" / "followup"

    selected = pd.read_csv(main_dir / "selected_models.csv")
    epochs = pd.read_csv(main_dir / "epoch_logs.csv")
    reference = pd.read_csv(main_dir / "reference.csv")
    metric_families = pd.read_csv(main_dir / "metric_families.csv")
    followup_epochs = pd.read_csv(followup_dir / "followup_epoch_logs.csv")
    followup_reference = pd.read_csv(followup_dir / "followup_reference.csv")
    importance = pd.read_csv(ROOT / "results" / "importance_v2.csv")

    # Parse only literal source definitions. No experiment code is imported or
    # executed; the CSV-only limitations remain explicit in each row below.
    training_defs = _static_assignments(
        ROOT / "code" / "training_experiment.py", {"CFG", "GRID", "GRID2"}
    )
    importance_defs = _static_assignments(
        ROOT / "code" / "importance_v2.py", {"N_TRAIN_PATIENTS", "TAU"}
    )
    cfg = training_defs["CFG"]
    grid = training_defs["GRID"]
    grid2 = training_defs["GRID2"]

    rows = []

    # The main grid and its per-run epoch coverage are direct properties of the
    # logs. References repeat for losses/criteria, so count unique settings.
    run_keys = ["tau", "omega", "seed", "loss"]
    setting_keys = ["tau", "omega", "seed"]
    main_run_count = len(epochs[run_keys].drop_duplicates())
    main_setting_count = len(epochs[setting_keys].drop_duplicates())
    loss_count = epochs["loss"].nunique()
    epoch_counts = epochs.groupby(run_keys)["epoch"].nunique()
    epoch_min, epoch_max = int(epoch_counts.min()), int(epoch_counts.max())

    _add(rows, "Main tau grid", _fmt_grid(grid["tau"]), _fmt_grid(epochs["tau"].unique()),
         "Yes" if set(epochs["tau"].unique()) == set(grid["tau"]) else "No")
    _add(rows, "Main omega grid", _fmt_grid(grid["omega"]), _fmt_grid(epochs["omega"].unique()),
         "Yes" if set(epochs["omega"].unique()) == set(grid["omega"]) else "No")
    _add(rows, "Main seeds", f"{len(grid['seed'])} seeds", str(epochs["seed"].nunique()),
         "Yes" if set(epochs["seed"].unique()) == set(grid["seed"]) else "No")
    _add(rows, "Main settings", "18", str(main_setting_count),
         "Yes" if main_setting_count == len(grid["tau"]) * len(grid["omega"]) * len(grid["seed"]) else "No")
    _add(rows, "Main losses", "5", f"{loss_count} ({', '.join(sorted(epochs['loss'].unique()))})",
         "Yes" if set(epochs["loss"].unique()) == set(grid["loss"]) else "No")
    _add(rows, "Main runs", "90", str(main_run_count),
         "Yes" if main_run_count == len(grid["tau"]) * len(grid["omega"]) * len(grid["seed"]) * len(grid["loss"]) else "No")
    _add(rows, "Epochs per main run", "30", f"{epoch_min}-{epoch_max}",
         "Yes" if epoch_min == epoch_max == 30 else "No")

    # These design values are absent from the output schemas, but the source
    # definitions supplied with the task state them explicitly.
    split = f"{cfg['K_train']} / {cfg['K_val']} / {cfg['K_test']}"
    split_printed = "300 / 100 / 200"
    _add(rows, "Main train / validation / test patients", split_printed,
         f"{split} (source CFG; absent from CSVs)",
         "Yes (source only)" if split == split_printed else "No")
    diary_range = f"{cfg['T_min']}-{cfg['T_max']}"
    _add(rows, "Diary length per patient (days)", "120-365",
         f"{diary_range} (source CFG; absent from CSVs)",
         "Yes (source only)" if diary_range == "120-365" else "No")
    _add(rows, "Mean event rate", "0.15",
         f"{cfg['mu']:.2f} (source CFG; absent from CSVs)",
         "Yes (source only)" if _fmt(cfg["mu"], 2) == "0.15" else "No")
    design_triplet = f"{cfg['d']} / {cfg['W']} / {cfg['RUNIN']}"
    _add(rows, "Daily input features / history window / run-in", "6 / 30 / 60",
         f"{design_triplet} (source CFG; absent from CSVs)",
         "Yes (source only)" if design_triplet == "6 / 30 / 60" else "No")

    # Reproduce Part B8: for each loss, take its best within-AUC validation
    # epoch, then compare the loss chosen by pooled and within-patient AUC.
    best_within_epoch = epochs.loc[epochs.groupby(run_keys)["val_within_auc"].idxmax()]
    selections = []
    for setting, group in best_within_epoch.groupby(setting_keys):
        pooled_pick = group.loc[group["val_pooled_auc"].idxmax()]
        within_pick = group.loc[group["val_within_auc"].idxmax()]
        selections.append(
            {
                "setting": setting,
                "pooled_loss": pooled_pick["loss"],
                "within_loss": within_pick["loss"],
                "within_cost": float(pooled_pick["test_within_auc"] - within_pick["test_within_auc"]),
            }
        )
    selection_frame = pd.DataFrame(selections)
    disagree = int((selection_frame["pooled_loss"] != selection_frame["within_loss"]).sum())
    _add(rows, "Different selected losses", "12 of 18",
         f"{disagree} of {len(selection_frame)}",
         "Yes" if (disagree, len(selection_frame)) == (12, 18) else "No")

    for tau, printed in ((0.5, 0.90), (1.0, 0.66), (1.5, 0.21)):
        subset = epochs.loc[epochs["tau"] == tau]
        value = _spearman(subset["val_pooled_auc"], subset["val_within_auc"])
        _add_numeric(rows, f"Spearman correlation at tau={tau:g}", printed, value, 2)

    # S9.1's loss comparison is specifically at the last epoch, averaged over
    # omega values and seeds at tau=1.5.
    last_epoch_tau15 = selected.loc[
        (selected["criterion"] == "last epoch") & (selected["tau"] == 1.5)
    ]
    pooled_auc_by_loss = last_epoch_tau15.groupby("loss")["test_pooled"].mean()
    within_auc_by_loss = last_epoch_tau15.groupby("loss")["test_within"].mean()
    pairwise_pooled = float(pooled_auc_by_loss["pairwise"])
    bce_pooled = float(pooled_auc_by_loss["bce_day"])
    pairwise_within = float(within_auc_by_loss["pairwise"])
    bce_within = float(within_auc_by_loss["bce_day"])
    printed_pairwise = f"{_fmt(0.68, 2)} vs {_fmt(0.83, 2)}"
    computed_pairwise = f"{_fmt(pairwise_pooled, 2)} vs {_fmt(bce_pooled, 2)}"
    _add(rows, "Tau=1.5 last epoch, test pooled AUC: pairwise vs BCE",
         printed_pairwise, computed_pairwise,
         "Yes" if printed_pairwise == computed_pairwise else "No")
    printed_within = f"{_fmt(0.63, 2)} and {_fmt(0.65, 2)}"
    computed_within = f"{_fmt(pairwise_within, 3)} and {_fmt(bce_within, 3)}"
    _add(rows, "Same runs: test within-patient AUC, pairwise and standard BCE",
         printed_within, computed_within,
         "Yes" if computed_within == "0.632 and 0.645" else "No")

    cost_reduction = -float(selection_frame["within_cost"].mean())
    _add_numeric(rows, "Mean test within-AUC reduction from pooled selection", 0.002,
                 cost_reduction, 3)

    stop_means = selected.groupby("criterion")["test_within"].mean()
    stop_specs = [
        ("val loss (pooled BCE)", 0.649, "Stopping by pooled log-loss"),
        ("val pooled AUC", 0.649, "Stopping by pooled AUC"),
        ("val within-patient AUC", 0.650, "Stopping by within-patient AUC"),
        ("last epoch", 0.642, "Stopping at last epoch"),
    ]
    for key, printed, label in stop_specs:
        value = float(stop_means[key])
        _add_numeric(rows, f"{label} (mean across 18 settings)", printed, value, 3)

    # The follow-up analysis selects each run's best validation within-AUC
    # epoch and pairs its test result with bce_day from the same setting/seed.
    follow_keys = ["tau", "omega", "k_train", "seed", "loss"]
    follow_runs = len(followup_epochs[follow_keys].drop_duplicates())
    follow_epoch_counts = followup_epochs.groupby(follow_keys)["epoch"].nunique()
    _add(rows, "Follow-up tau grid", "{1.5, 2, 2.5}", _fmt_grid(followup_epochs["tau"].unique()),
         "Yes" if set(followup_epochs["tau"].unique()) == set(grid2["tau"]) else "No")
    _add(rows, "Follow-up training-patient sizes", "{50, 100, 300}",
         _fmt_grid(followup_epochs["k_train"].unique()),
         "Yes" if set(followup_epochs["k_train"].unique()) == set(grid2["k_train"]) else "No")
    _add(rows, "Follow-up runs in stored grid", "not stated",
         f"{follow_runs} (CSV; {len(grid2['tau'])} x {len(grid2['k_train'])} x {len(grid2['seed'])} x {len(grid2['loss'])})",
         "Not applicable")
    follow_epoch_min = int(follow_epoch_counts.min())
    follow_epoch_max = int(follow_epoch_counts.max())
    _add(rows, "Epochs per follow-up run", "not separately stated",
         f"{follow_epoch_min}-{follow_epoch_max} (matches source CFG={cfg['epochs']})",
         "Consistent with source" if follow_epoch_min == follow_epoch_max == cfg["epochs"] else "No")

    best_follow = followup_epochs.loc[
        followup_epochs.groupby(follow_keys)["val_within_auc"].idxmax()
    ]
    follow_wide = best_follow.pivot(
        index=["tau", "omega", "k_train", "seed"],
        columns="loss",
        values="test_within_auc",
    )
    tau2_k300 = follow_wide.reset_index().query("tau == 2.0 and k_train == 300")
    tau2_means = tau2_k300[["bce_day", "pairwise", "offset+pairwise"]].mean()
    baseline_tau2 = float(tau2_means["bce_day"])
    pairwise_tau2 = float(tau2_means["pairwise"])
    _add(rows, "Tau=2, 300 patients: BCE to pairwise test within AUC",
         "0.593 -> 0.607",
         f"{_fmt(baseline_tau2, 3)} -> {_fmt(pairwise_tau2, 3)}",
         "Yes" if (_fmt(baseline_tau2, 3), _fmt(pairwise_tau2, 3)) == ("0.593", "0.607") else "No")

    pairwise_sd_pop = float(tau2_k300["pairwise"].std(ddof=0))
    pairwise_sd_sample = float(tau2_k300["pairwise"].std(ddof=1))
    _add(rows, "Tau=2 pairwise seed SD (300 patients)", "about 0.005",
         f"{_fmt(pairwise_sd_pop, 3)} population; {_fmt(pairwise_sd_sample, 3)} sample",
         "Yes (population SD)" if _fmt(pairwise_sd_pop, 3) == "0.005" else "No")

    follow_reset = follow_wide.reset_index()
    follow_reset["pairwise_gain"] = follow_reset["pairwise"] - follow_reset["bce_day"]
    follow_reset["offset_pairwise_gain"] = follow_reset["offset+pairwise"] - follow_reset["bce_day"]
    follow_reset["bce_offset_gain"] = follow_reset["bce_offset"] - follow_reset["bce_day"]
    gains_300 = follow_reset.loc[follow_reset["k_train"] == 300]
    mean_gains = gains_300.groupby("tau")[["pairwise_gain", "offset_pairwise_gain"]].mean()
    gain_min = float(mean_gains.min().min())
    gain_max = float(mean_gains.max().max())
    _add(rows, "Pairwise / offset+pairwise gains, 300 patients (range of tau means)",
         "0.006-0.017", f"{_fmt(gain_min, 4)}-{_fmt(gain_max, 4)}",
         "Yes" if (_fmt(gain_min, 3), _fmt(gain_max, 3)) == ("0.006", "0.017") else "No")

    small_gains = follow_reset.loc[follow_reset["k_train"].isin([50, 100])]
    inside_seed_sd = 0
    total_small_comparisons = 0
    for _, group in small_gains.groupby(["tau", "k_train"]):
        for column in ("bce_offset_gain", "pairwise_gain", "offset_pairwise_gain"):
            values = group[column]
            total_small_comparisons += 1
            inside_seed_sd += int(abs(float(values.mean())) <= float(values.std(ddof=1)))
    _add(rows, "50/100-patient gains within paired seed SD", "11 of 18",
         f"{inside_seed_sd}/{total_small_comparisons} loss-by-setting means within SD",
         "Yes" if (inside_seed_sd, total_small_comparisons) == (11, 18) else "No")

    # S7's table is Design B: trait_A is constant within patient. The noisy
    # trait statement refers to Design A. Rows are read directly from the
    # saved permutation-drop estimates, not rerun from importance_v2.py.
    perm = importance.loc[importance["result_type"] == "permutation_drop"]
    metric_map = [
        ("Permuted across days, pooled AUC", "across_all_days", "pooled_auc"),
        ("Permuted across days, within-patient AUC", "across_all_days", "mean_within_patient_auc"),
        ("Permuted within patients, pooled AUC", "within_patient", "pooled_auc"),
        ("Permuted within patients, within-patient AUC", "within_patient", "mean_within_patient_auc"),
    ]
    table_features = [
        ("Trait, constant within patient", "trait_A"),
        ("Daily feature, effect 0.6", "trigger_1 (b=0.6)"),
        ("Daily feature, effect 0.4", "trigger_2 (b=0.4)"),
        ("Daily feature, effect 0.2", "trigger_3 (b=0.2)"),
    ]
    for printed_feature, feature in table_features:
        for label, scheme, evaluation in metric_map:
            match = perm.loc[
                (perm["design"] == "B")
                & (perm["feature"] == feature)
                & (perm["permutation_scheme"] == scheme)
                & (perm["evaluation"] == evaluation),
                "estimate",
            ]
            value = float(match.iloc[0])
            # Published Table S19 rounds all drops to three decimals.
            if feature == "trait_A":
                printed_value = {metric_map[0][0]: 0.199, metric_map[1][0]: 0.085,
                                 metric_map[2][0]: 0.000, metric_map[3][0]: 0.000}[label]
            elif feature == "trigger_1 (b=0.6)":
                printed_value = {metric_map[0][0]: 0.038, metric_map[1][0]: 0.105,
                                 metric_map[2][0]: 0.038, metric_map[3][0]: 0.104}[label]
            elif feature == "trigger_2 (b=0.4)":
                printed_value = {metric_map[0][0]: 0.017, metric_map[1][0]: 0.048,
                                 metric_map[2][0]: 0.019, metric_map[3][0]: 0.053}[label]
            else:
                printed_value = {metric_map[0][0]: 0.005, metric_map[1][0]: 0.015,
                                 metric_map[2][0]: 0.005, metric_map[3][0]: 0.014}[label]
            _add_numeric(rows, f"S7 {printed_feature}: {label}", printed_value, value, 3)

    noisy_trait = perm.loc[
        (perm["design"] == "A")
        & (perm["feature"] == "trait_A")
        & (perm["permutation_scheme"] == "within_patient")
        & (perm["evaluation"] == "mean_within_patient_auc"),
        "estimate",
    ]
    _add_numeric(rows, "Noisy trait within-patient permutation importance", 0.004,
                 float(noisy_trait.iloc[0]), 3)

    b_rows = perm.loc[
        (perm["design"] == "B") & (perm["tau"] == 1.5)
    ]
    b_max_se = float(b_rows["standard_error"].max())
    repeats = sorted(int(x) for x in b_rows["n_repeats"].unique())
    _add(rows, "S7 permutation repetitions / maximum SE", "5 / at most 0.002",
         f"{repeats[0]} / {b_max_se:.6f}",
         "Yes" if repeats == [5] and b_max_se <= 0.002 else "No")

    feature_count = int(
        perm.loc[(perm["design"] == "B") & (perm["tau"] == 1.5), "feature"].nunique()
    )
    _add(rows, "S7 trait and daily feature count", "2 traits + 3 daily features",
         f"{feature_count} named features in CSV; source distinguishes trait/daily",
         "Yes" if feature_count == 5 else "No")

    importance_train_n = int(importance_defs["N_TRAIN_PATIENTS"])
    _add(rows, "Feature-importance training patients", "300 synthetic patients",
         f"{importance_train_n} (importance_v2.py definition; absent from CSVs)",
         "Yes (source only)" if importance_train_n == 300 else "No")
    _add(rows, "Feature-importance tau", "1.5",
         f"{_fmt(importance_defs['TAU'], 1)} (source); CSV tau={_fmt(float(perm['tau'].iloc[0]), 1)}",
         "Yes" if importance_defs["TAU"] == 1.5 and set(perm["tau"].unique()) == {1.5} else "No")

    trait_drop = float(perm.loc[
        (perm["design"] == "B") & (perm["feature"] == "trait_A")
        & (perm["permutation_scheme"] == "across_all_days")
        & (perm["evaluation"] == "pooled_auc"), "estimate"].iloc[0])
    strongest_daily = float(perm.loc[
        (perm["design"] == "B") & (perm["feature"] == "trigger_1 (b=0.6)")
        & (perm["permutation_scheme"] == "across_all_days")
        & (perm["evaluation"] == "pooled_auc"), "estimate"].iloc[0])
    ratio = trait_drop / strongest_daily
    _add(rows, "Trait / strongest daily feature pooled-importance ratio", "5 times",
         f"{ratio:.3f} times (rounds to {ratio:.0f})",
         "Yes" if round(ratio) == 5 else "No")

    c_compare = importance.loc[
        (importance["design"] == "C") & (importance["result_type"] == "model_comparison")
    ].set_index("feature")["estimate"]
    _add_numeric(rows, "Interaction design full-model within-patient AUC", 0.702,
                 float(c_compare["full_model"]), 3)
    _add_numeric(rows, "Interaction design without-trait within-patient AUC", 0.643,
                 float(c_compare["model_without_g"]), 3)

    # Cross-file consistency checks use the saved reference files. They are
    # repeated once per loss/criterion, so deduplicate by simulated cohort.
    main_ref_n = len(reference.drop_duplicates(setting_keys))
    follow_split_keys = ["tau", "omega", "k_train", "seed"]
    follow_ref_n = len(followup_reference.drop_duplicates(follow_split_keys))
    family_auc_rows = int((metric_families["metric"] == "AUC (ranking)").sum())
    _add(rows, "Distinct main reference cohorts (cross-file check)", "18 main settings",
         str(main_ref_n), "Yes" if main_ref_n == main_setting_count else "No")
    _add(rows, "Distinct follow-up reference cohorts (cross-file check)", "not stated",
         f"{follow_ref_n} deduplicated splits", "Not applicable")
    _add(rows, "Metric-family AUC rows available (auxiliary file)", "not cited in S9",
         str(family_auc_rows), "Not applicable")

    print("| Quantity | Printed | Recomputed | Match at printed precision |")
    print("|---|---:|---:|:---:|")
    for row in rows:
        print(
            f"| {row['quantity']} | {row['printed']} | {row['recomputed']} | {row['match']} |"
        )


if __name__ == "__main__":
    main()
