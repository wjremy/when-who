"""High-level reporting for repeated binary forecasts."""
from __future__ import annotations

import inspect
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd

from .metrics import (
    _check_missing,
    _outcomes,
    _probabilities,
    brier_scores,
    brier_skill_summary,
    calibration_slopes,
    conditional_within_slope,
    expanding_patient_rate,
    patient_calibration_in_the_large,
    per_patient_auc_summary,
    per_time_point_auc,
    permutation_reference,
    pooled_auc,
    pooled_auc_permutation_test,
    population_a0_moment_corrected,
    population_rate,
    predictive_value_excess,
    rate_benchmark,
    retrospective_patient_rate,
    runin_patient_rate,
    threshold_metrics,
    within_patient_auc,
)


def evaluate_v1(
    df: pd.DataFrame,
    patient: str = "patient",
    y: str = "y",
    forecast: str = "forecast",
    time: Optional[str] = None,
    threshold: Optional[float] = None,
    runin_days: Optional[float] = None,
    return_references: bool = False,
    time_col: Optional[str] = None,
    history_forecast_fn: Optional[Any] = None,
    n_perm: int = 2000,
    seed: int = 0,
    prior_rate: Optional[float] = None,
    prior_weight: float = 1.0,
    n_boot: int = 0,
    _include_optional: bool = True,
) -> dict[str, Any]:
    """Deprecated version 1 report API; use :func:`evaluate` for the paper's fix.

    If ``runin_days`` is given, rows with ``time < runin_days`` are history only and
    are excluded from every reported score. Run-in and expanding-rate references may
    use that history. ``prior_rate`` is an optional probability from outside the
    evaluation cohort used only to seed the expanding reference; its weight is
    ``prior_weight`` pseudo-observations. Without it, the expanding rate uses only
    strictly earlier outcomes (including run-in outcomes when present), and is
    undefined until such an outcome exists. The scored-cohort and retrospective
    rates are descriptive hindsight references, not prospective forecasts.

    ``n_perm`` controls the within-patient forecast-shuffle test of pooled AUC and,
    when ``history_forecast_fn`` is supplied, the history-rebuild reference. ``n_boot``
    defaults to zero; positive values add percentile intervals from whole-patient
    bootstrap resamples. ``time_col`` selects the grouping column for per-time-point
    AUC and defaults to ``time``.

    ``history_forecast_fn`` receives one patient's outcomes in time order, with
    run-in first, and returns one forecast per outcome. Its per-patient reference
    scores only post-run-in rows.
    """
    warnings.warn(
        "evaluate_v1() preserves the wheneval 1.x report API; use evaluate() "
        "for the within-patient permutation procedure.",
        DeprecationWarning,
        stacklevel=2,
    )
    if n_perm < 1 and _include_optional:
        raise ValueError("n_perm must be at least 1")
    if n_boot < 0:
        raise ValueError("n_boot must be nonnegative")
    if runin_days is not None and time is None:
        raise ValueError("time is required when runin_days is given")
    if prior_rate is not None:
        prior_rate = float(prior_rate)
        if not np.isfinite(prior_rate) or not 0 <= prior_rate <= 1:
            raise ValueError("prior_rate must be a probability in [0, 1]")
        if time is None:
            raise ValueError("time is required when prior_rate is given")
    if not np.isfinite(prior_weight) or prior_weight < 0:
        raise ValueError("prior_weight must be finite and nonnegative")

    columns = [patient, y, forecast] + ([time] if time is not None else [])
    missing_columns = [column for column in columns if column not in df]
    if missing_columns:
        raise ValueError(f"missing dataframe columns: {', '.join(missing_columns)}")
    pid = df[patient].to_numpy()
    yy = _outcomes(df[y].to_numpy())
    pp = _probabilities("forecast", df[forecast].to_numpy())
    _check_missing("patient", pid)
    if not (len(pid) == len(yy) == len(pp)):
        raise ValueError("patient, y, and forecast must have equal length")

    tt: Optional[np.ndarray] = None
    if time is not None:
        _check_missing(time, df[time].to_numpy())
        try:
            tt = df[time].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("time must be numeric elapsed time") from exc
        if not np.isfinite(tt).all():
            raise ValueError("time must be finite elapsed time")

    score_mask = np.ones(len(yy), dtype=bool) if runin_days is None else tt >= float(runin_days)
    if not score_mask.any():
        raise ValueError("no scored rows remain after the run-in period")
    scored_df = df.iloc[np.flatnonzero(score_mask)].copy()
    pid_s, yy_s, pp_s = pid[score_mask], yy[score_mask], pp[score_mask]
    tt_s = tt[score_mask] if tt is not None else None

    pooled = pooled_auc(pid_s, yy_s, pp_s)
    within = within_patient_auc(pid_s, yy_s, pp_s)
    per_patient = per_patient_auc_summary(pid_s, yy_s, pp_s)
    patient_mean_forecast = np.empty(len(yy_s), dtype=float)
    patient_rate_mean: dict[Any, float] = {}
    for key in pd.unique(pid_s):
        ix = np.flatnonzero(pid_s == key)
        patient_mean_forecast[ix] = float(np.mean(pp_s[ix]))
        patient_rate_mean[key] = float(np.mean(yy_s[ix]))
    patient_mean_auc = pooled_auc(pid_s, yy_s, patient_mean_forecast)["value"]
    share_denominator = pooled["value"] - 0.5
    patient_mean_share = ((patient_mean_auc - 0.5) / share_denominator
                          if np.isfinite(share_denominator) and abs(share_denominator) > 1e-15
                          else float("nan"))

    rates = pd.Series(patient_rate_mean, dtype=float)
    n_events = int(np.sum(yy_s == 1))
    n_nonevents = int(len(yy_s) - n_events)
    within_pairs = sum(int(np.sum(yy_s[np.flatnonzero(pid_s == key)] == 1) *
                             np.sum(yy_s[np.flatnonzero(pid_s == key)] == 0))
                       for key in pd.unique(pid_s))
    pair_share = within_pairs / (n_events * n_nonevents) if n_events and n_nonevents else float("nan")

    result: dict[str, Any] = {
        "cohort": {
            "patients": int(pd.Series(pid_s).nunique()),
            "days": int(len(yy_s)),
            "events": n_events,
            "overall_rate": float(np.mean(yy_s)),
            "patient_rate_mean": float(rates.mean()),
            "patient_rate_sd": float(rates.std(ddof=1)) if len(rates) > 1 else float("nan"),
            "within_patient_pair_share": pair_share,
            "rows_excluded_runin": int(len(yy) - len(yy_s)),
            "runin_days": runin_days,
        },
        "auc": {
            "pooled": pooled,
            "within_patient": within,
            "per_patient_summary": per_patient,
            "per_time_point": None,
            "A0": rate_benchmark(pid_s, yy_s),
            "A0_population": population_a0_moment_corrected(pid_s, yy_s),
            "patient_mean": {
                "pooled_auc": float(patient_mean_auc),
                "share_of_pooled_auc_above_0_5": float(patient_mean_share),
            },
        },
        "threshold": threshold_metrics(pid_s, yy_s, pp_s, threshold, time=tt_s)
                     if threshold is not None else None,
        "brier": brier_scores(pid_s, yy_s, pp_s),
        "skill": {},
        "calibration": calibration_slopes(pid_s, yy_s, pp_s),
        "excluded_patients": {
            "pooled_auc": pooled["excluded"],
            "within_auc": within["excluded"],
        },
    }

    references: dict[str, np.ndarray] = {
        "population_rate": population_rate(yy_s),
        "retrospective_patient_rate": retrospective_patient_rate(pid_s, yy_s),
    }
    if tt is not None:
        if runin_days is not None:
            references["runin_patient_rate"] = runin_patient_rate(pid, yy, tt, float(runin_days))[score_mask]
        # The reference is aligned with every input row before scoring is applied, so
        # later forecasts can use run-in observations but never same-time/future outcomes.
        references["expanding_patient_rate"] = expanding_patient_rate(
            pid, yy, tt, prior_rate=prior_rate, prior_weight=prior_weight
        )[score_mask]
    result["skill"] = {
        name: brier_skill_summary(pid_s, yy_s, pp_s, ref)
        for name, ref in references.items()
    }

    time_for_points = time_col if time_col is not None else time
    if time_for_points is not None:
        if time_for_points not in scored_df:
            raise ValueError(f"missing dataframe columns: {time_for_points}")
        result["auc"]["per_time_point"] = per_time_point_auc(
            scored_df, time_for_points, patient=patient, y=y, forecast=forecast
        )
    if threshold is not None:
        result["threshold"]["predictive_value_excess"] = predictive_value_excess(
            pid_s, yy_s, pp_s, threshold
        )
    result["calibration_in_the_large"] = patient_calibration_in_the_large(pid_s, yy_s, pp_s)
    if _include_optional:
        result["conditional_within_slope"] = conditional_within_slope(pid_s, yy_s, pp_s)
        result["pooled_auc_permutation"] = pooled_auc_permutation_test(
            pid_s, yy_s, pp_s, n_perm=n_perm, seed=seed
        )
        result["permutation_reference"] = None
        if history_forecast_fn is not None and time is not None and runin_days is not None:
            result["permutation_reference"] = permutation_reference(
                pid, yy, tt, history_forecast_fn, n_perm=n_perm, seed=seed, runin_days=runin_days
            )
    else:
        result["conditional_within_slope"] = None
        result["pooled_auc_permutation"] = None
        result["permutation_reference"] = None
    if return_references:
        result["references"] = references
    result["excluded_patients"].update({
        "threshold_sensitivity": len(pd.unique(pid_s)) - result["threshold"]["eligible"]["sensitivity"]
        if threshold is not None else None,
        "threshold_specificity": len(pd.unique(pid_s)) - result["threshold"]["eligible"]["specificity"]
        if threshold is not None else None,
    })
    if n_boot:
        result["bootstrap"] = _patient_bootstrap_intervals(
            df, patient, y, forecast, time, threshold, runin_days, time_col,
            prior_rate, prior_weight, n_boot, seed,
        )
    return result


def _patient_bootstrap_intervals(
    df: pd.DataFrame,
    patient: str,
    y: str,
    forecast: str,
    time: Optional[str],
    threshold: Optional[float],
    runin_days: Optional[float],
    time_col: Optional[str],
    prior_rate: Optional[float],
    prior_weight: float,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap evaluate's numeric summaries by resampling complete patient records."""
    patient_values = pd.unique(df[patient].to_numpy())
    groups = [np.flatnonzero(df[patient].to_numpy() == value) for value in patient_values]
    if not len(groups):
        return {"n_boot": 0, "intervals": {}}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}

    def add(path: str, value: Any) -> None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        if np.isfinite(number):
            samples.setdefault(path, []).append(number)

    for _ in range(n_boot):
        chosen = rng.integers(0, len(groups), size=len(groups))
        parts = [groups[i] for i in chosen]
        ix = np.concatenate(parts)
        boot = df.iloc[ix].copy().reset_index(drop=True)
        boot[patient] = np.concatenate([np.full(len(part), j) for j, part in enumerate(parts)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            draw = evaluate_v1(
                boot, patient=patient, y=y, forecast=forecast, time=time, threshold=threshold,
                runin_days=runin_days, time_col=time_col, n_perm=0, seed=seed,
                prior_rate=prior_rate, prior_weight=prior_weight, n_boot=0, _include_optional=False,
            )
        add("auc.pooled", draw["auc"]["pooled"]["value"])
        add("auc.within_patient", draw["auc"]["within_patient"]["value"])
        add("auc.per_patient_mean", draw["auc"]["per_patient_summary"]["mean"])
        add("auc.per_patient_median", draw["auc"]["per_patient_summary"]["median"])
        add("auc.per_patient_iqr25", draw["auc"]["per_patient_summary"]["iqr"][0])
        add("auc.per_patient_iqr75", draw["auc"]["per_patient_summary"]["iqr"][1])
        add("auc.per_time_point_mean", draw["auc"]["per_time_point"]["value"] if draw["auc"]["per_time_point"] else np.nan)
        add("auc.per_time_point_median", draw["auc"]["per_time_point"]["median"] if draw["auc"]["per_time_point"] else np.nan)
        add("auc.per_time_point_iqr25", draw["auc"]["per_time_point"]["iqr"][0] if draw["auc"]["per_time_point"] else np.nan)
        add("auc.per_time_point_iqr75", draw["auc"]["per_time_point"]["iqr"][1] if draw["auc"]["per_time_point"] else np.nan)
        add("auc.A0", draw["auc"]["A0"])
        add("auc.A0_population", draw["auc"]["A0_population"])
        add("auc.pooled_minus_A0", draw["auc"]["pooled"]["value"] - draw["auc"]["A0"])
        add("auc.patient_mean_share", draw["auc"]["patient_mean"]["share_of_pooled_auc_above_0_5"])
        add("brier.pooled", draw["brier"]["pooled"]["brier"])
        add("brier.patient_mean", draw["brier"]["patient_mean"]["brier"])
        add("calibration_in_the_large.mean", draw["calibration_in_the_large"]["mean"])
        add("calibration.pooled_slope", draw["calibration"]["pooled_slope"])
        add("calibration.between_slope", draw["calibration"]["between_slope"])
        add("calibration.within_slope", draw["calibration"]["within_slope"])
        for name, skill in draw["skill"].items():
            add(f"skill.{name}.pooled", skill["pooled"])
            add(f"skill.{name}.patient_mean", skill["patient_mean"])
        if threshold is not None:
            tm = draw["threshold"]
            for name in ("sensitivity", "specificity", "ppv", "npv", "youden"):
                add(f"threshold.pooled.{name}", tm["pooled"][name])
            add("threshold.within_patient.youden", tm["within_patient"]["youden"])
            pv = tm.get("predictive_value_excess")
            if pv is not None:
                add("threshold.ppv_excess_patient_mean", pv["ppv_excess"]["mean"])
                add("threshold.npv_excess_patient_mean", pv["npv_excess"]["mean"])
            alarm = tm["alarms_per_patient_per_time_unit"]
            if alarm is not None:
                add("threshold.alarms_per_patient_per_time_unit.mean", alarm["mean"])

    intervals = {
        name: [float(v) for v in np.percentile(values, [2.5, 97.5])]
        for name, values in samples.items() if values
    }
    return {"n_boot": int(n_boot), "seed": int(seed), "intervals": intervals}


def to_markdown_v1(result: dict[str, Any]) -> str:
    """Render the main evaluate results as a compact three-decimal Markdown table."""
    c, a, b = result["cohort"], result["auc"], result["brier"]
    rows: list[tuple[str, Any]] = [
        ("Patients", c["patients"]), ("Scored observations", c["days"]), ("Scored events", c["events"]),
        ("Rows excluded as run-in", c["rows_excluded_runin"]), ("Scored event rate", c["overall_rate"]),
        ("Mean patient event rate", c["patient_rate_mean"]), ("SD of patient event rates", c["patient_rate_sd"]),
        ("Within-patient pair share", c["within_patient_pair_share"]),
        ("Pooled AUC", a["pooled"]["value"]), ("Per-patient AUC", a["within_patient"]["value"]),
        ("Patients excluded from per-patient AUC", a["within_patient"]["excluded"]),
        ("Retrospective A0", a["A0"]), ("Population A0 (noise-corrected)", a["A0_population"]),
        ("Patient-mean forecast pooled AUC", a["patient_mean"]["pooled_auc"]),
        ("Patient-mean share of pooled AUC above 0.5", a["patient_mean"]["share_of_pooled_auc_above_0_5"]),
        ("Pooled Brier score", b["pooled"]["brier"]), ("Patient-mean Brier score", b["patient_mean"]["brier"]),
    ]
    for name, skill in result["skill"].items():
        rows.extend([(f"Brier skill vs {name} (pooled)", skill["pooled"]),
                     (f"Brier skill vs {name} (patient mean)", skill["patient_mean"]),
                     (f"Patients with usable skill vs {name}", skill["eligible_patients"])])
    if result["threshold"] is not None:
        t = result["threshold"]
        rows.extend([
            ("Threshold", t["threshold"]), ("Pooled sensitivity", t["pooled"]["sensitivity"]),
            ("Pooled specificity", t["pooled"]["specificity"]), ("Pooled PPV", t["pooled"]["ppv"]),
            ("Pooled Youden index", t["pooled"]["youden"]),
            ("Within-patient sensitivity", t["within_patient"]["sensitivity"]),
            ("Within-patient specificity", t["within_patient"]["specificity"]),
            ("Within-patient Youden index", t["within_patient"]["youden"]),
            ("Patients with both outcomes", t["eligible"]["both"]),
            ("Alarms per patient (mean count)", t["alarm_counts"]["mean"]),
            ("Alarms per patient per 30 rows (mean)", t["alarms_per_patient_30_days"]["mean"]),
        ])
        alarm_rate = t["alarms_per_patient_per_time_unit"]
        if alarm_rate is not None:
            rows.append(("Alarms per patient per elapsed time unit (mean)", alarm_rate["mean"]))
    rows.extend([
        ("Pooled calibration slope", result["calibration"]["pooled_slope"]),
        ("Between-patient calibration slope", result["calibration"]["between_slope"]),
        ("Within-patient calibration slope (joint model)", result["calibration"]["within_slope"]),
        ("Calibration in the large, mean patient difference", result["calibration_in_the_large"]["mean"]),
        ("Calibration in the large, median patient difference", result["calibration_in_the_large"]["median"]),
        ("Per-patient AUC median", a["per_patient_summary"]["median"]),
        ("Per-patient AUC IQR", a["per_patient_summary"]["iqr"]),
        ("Share of patient AUCs above 0.5", a["per_patient_summary"]["share_above_0_5"]),
    ])
    conditional = result.get("conditional_within_slope")
    if conditional is not None:
        rows.extend([
            ("Conditional within-patient calibration slope", conditional["slope"]),
            ("Conditional slope SE", conditional["se"]),
            ("Patients in conditional slope", conditional["patients"]),
        ])
        if not conditional["estimable"]:
            rows.append(("Conditional slope note", conditional["reason"]))
    tp = a["per_time_point"]
    if tp is not None:
        rows.extend([
            ("Per-time-point AUC (equal weight)", tp["value"]),
            ("Per-time-point AUC median", tp["median"]), ("Per-time-point AUC IQR", tp["iqr"]),
            ("Per-time-point AUC (comparison-weighted)", tp["comparison_weighted"]),
            ("Time points used", tp["time_points_used"]), ("Time points excluded", tp["time_points_excluded"]),
        ])
    if result["threshold"] is not None and "predictive_value_excess" in result["threshold"]:
        pv = result["threshold"]["predictive_value_excess"]
        rows.extend([("PPV excess over own rate (patient mean)", pv["ppv_excess"]["mean"]),
                     ("NPV excess over own non-event rate (patient mean)", pv["npv_excess"]["mean"])])
    perm = result.get("pooled_auc_permutation")
    if perm is not None:
        rows.extend([("Pooled AUC permutation null mean", perm["null_mean"]),
                     ("Pooled AUC permutation null percentiles", perm["null_percentiles"]),
                     ("Pooled AUC permutation one-sided p", perm["one_sided_p_value"])])
    hist_perm = result.get("permutation_reference")
    if hist_perm is not None:
        rows.extend([("History permutation observed patient-mean AUC", hist_perm["observed"]),
                     ("History permutation reference mean", hist_perm["reference_mean"]),
                     ("History permutation difference", hist_perm["difference"])])
    bootstrap = result.get("bootstrap")
    if bootstrap is not None:
        rows.extend((f"Patient-bootstrap 95% interval: {name}", interval)
                    for name, interval in bootstrap["intervals"].items())

    def display(value: Any) -> str:
        if isinstance(value, (dict, list, tuple)):
            return str(value)
        return f"{value:.3f}" if isinstance(value, (int, float, np.integer, np.floating)) else str(value)

    lines = ["| Quantity | Value |", "|---|---:|"]
    lines.extend(f"| {key} | {display(value)} |" for key, value in rows)
    return "\n".join(lines)


# Keep the v1 report callable under its explicit name while making Section 3 the default.
from .evaluation import evaluate, to_markdown  # noqa: E402
