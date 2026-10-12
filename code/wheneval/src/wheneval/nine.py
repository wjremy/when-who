"""The paper's nine pooled and within-patient performance metrics.

Patient-level summaries use equal patient weights. ``per_patient`` is a mapping
from patient ID to its eligible value. For PPV, NPV, and net benefit those
values are excesses over the patient-specific reference described below.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .metrics import (
    _arrays,
    _groups,
    _probabilities,
    auc,
    calibration_slopes,
    conditional_within_slope,
    threshold_metrics,
)


def _threshold(value: Any, *, strict: bool = False) -> float:
    try:
        t = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be a finite number") from exc
    if not np.isfinite(t):
        raise ValueError("threshold must be a finite number")
    if strict and not 0 < t < 1:
        raise ValueError("net-benefit threshold must be strictly between 0 and 1")
    if not strict and not 0 <= t <= 1:
        raise ValueError("threshold must be in [0, 1]")
    return t


def _summary(values: dict[Any, float], mean_override: float | None = None,
             n_override: int | None = None) -> dict[str, Any]:
    finite = np.asarray([v for v in values.values() if np.isfinite(v)], dtype=float)
    if len(finite):
        q25, q75 = np.percentile(finite, [25, 75])
        median = float(np.median(finite))
        mean = float(np.mean(finite)) if mean_override is None else float(mean_override)
        iqr = [float(q25), float(q75)]
    else:
        mean = float("nan") if mean_override is None else float(mean_override)
        median = float("nan")
        iqr = [float("nan"), float("nan")]
    return {"mean": mean, "median": median, "iqr": iqr,
            "n": int(len(finite) if n_override is None else n_override)}


def _result(pooled: float, values: dict[Any, float], *, mean_override: float | None = None,
            n_override: int | None = None, **extra: Any) -> dict[str, Any]:
    return {"pooled": float(pooled), "summary": _summary(values, mean_override, n_override),
            "per_patient": values, **extra}


def average_precision(y: Any, s: Any) -> float:
    """Return non-interpolated, step-wise average precision.

    Each distinct score defines a threshold, so tied observations enter together
    as in ``sklearn.metrics.average_precision_score``. Returns NaN when there
    are no positive outcomes (AP is undefined in that case).
    """
    score = np.asarray(s, dtype=float)
    _, yy, ss = _arrays(np.zeros(len(score), dtype=int), y, score)
    positives = int(yy.sum())
    if positives == 0:
        return float("nan")

    order = np.argsort(-ss, kind="mergesort")
    sorted_scores = ss[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    ends = np.r_[starts[1:], len(yy)]
    true_positives = 0
    ap = 0.0
    for lo, hi in zip(starts, ends):
        block_positives = int(np.sum(yy[order[lo:hi]]))
        true_positives += block_positives
        precision = true_positives / hi
        ap += (block_positives / positives) * precision
    return float(ap)


def ap_chance(y: Any) -> float:
    """Return exact expected AP under a uniformly random ordering of rows.

    For ``n`` rows and ``m`` positives this is
    ``(H_n + ((m - 1) / (n - 1)) * (n - H_n)) / n`` for ``n > 1``,
    where ``H_n`` is the nth harmonic number. The equivalent position-wise
    sum is evaluated deterministically, with no random sampling. Returns NaN
    when there are no positives or no rows.
    """
    yy = np.asarray(y)
    _, yy = _arrays(np.arange(len(yy)), yy)
    n = len(yy)
    m = int(yy.sum())
    if n == 0 or m == 0:
        return float("nan")
    if n == 1:
        return 1.0
    harmonic = math.fsum(1.0 / k for k in range(1, n + 1))
    return float((harmonic + ((m - 1) / (n - 1)) * (n - harmonic)) / n)


def net_benefit(y: Any, s: Any, t: float) -> float:
    """Return pooled decision-curve net benefit for alarms ``s >= t``."""
    threshold = _threshold(t, strict=True)
    score = np.asarray(s, dtype=float)
    _, yy, pp = _arrays(np.zeros(len(score), dtype=int), y, score)
    pp = _probabilities("forecast", pp)
    if not len(yy):
        return float("nan")
    alarm = pp >= threshold
    tp = int(np.sum(alarm & (yy == 1)))
    fp = int(np.sum(alarm & (yy == 0)))
    odds = threshold / (1 - threshold)
    return float(tp / len(yy) - fp / len(yy) * odds)


def _classification(patient: Any, y: Any, s: Any, t: float, metric: str) -> dict[str, Any]:
    threshold = _threshold(t)
    pid, yy, pp = _arrays(patient, y, s)
    pp = _probabilities("forecast", pp)
    pred = pp >= threshold

    def score(ix: np.ndarray) -> float:
        truth, alarm = yy[ix] == 1, pred[ix]
        if metric == "accuracy":
            return float(np.mean(truth == alarm)) if len(ix) else float("nan")
        tp = int(np.sum(truth & alarm))
        fp = int(np.sum(~truth & alarm))
        fn = int(np.sum(truth & ~alarm))
        denominator = 2 * tp + fp + fn
        return float(2 * tp / denominator) if denominator else float("nan")

    pooled = score(np.arange(len(yy)))
    per: dict[Any, float] = {}
    for key, ix in _groups(pid):
        if metric == "f1" and not (np.any(yy[ix] == 1) and np.any(yy[ix] == 0)):
            continue
        value = score(ix)
        if np.isfinite(value):
            per[key] = value
    return _result(pooled, per)


def accuracy(patient: Any, y: Any, s: Any, t: float) -> dict[str, Any]:
    """Return pooled and per-patient accuracy at threshold ``t``."""
    return _classification(patient, y, s, t, "accuracy")


def f1(patient: Any, y: Any, s: Any, t: float) -> dict[str, Any]:
    """Return pooled and per-patient F1 at threshold ``t``.

    Per-patient values use patients with both outcomes. A zero-denominator F1
    is omitted as undefined.
    """
    return _classification(patient, y, s, t, "f1")


def _net_benefit_patient(y: np.ndarray, s: np.ndarray, t: float) -> float:
    if not len(y):
        return float("nan")
    alarm = s >= t
    odds = t / (1 - t)
    return float(np.sum(alarm & (y == 1)) / len(y)
                 - np.sum(alarm & (y == 0)) / len(y) * odds)


def nine_metrics(patient: Any, y: Any, forecast: Any, threshold: float,
                 yardstick: Any = None) -> dict[str, dict[str, Any]]:
    """Calculate pooled and per-patient summaries for the paper's nine metrics.

    The optional row-aligned ``yardstick`` defaults to each patient's observed
    event rate repeated over their rows. Sensitivity and specificity share the
    group with both outcomes. PPV and NPV patient values are excesses over that
    patient's observed event and non-event rates. Net-benefit patient values
    are excesses over the own-rate policy (treat a row when its yardstick rate
    is at least the threshold), both scored on the observed outcomes. Brier-skill ``summary.mean`` uses the
    equal-patient ratio of mean Brier scores; its median and IQR describe the
    individual patient skill values. Calibration's conditional estimate is a
    single shared slope; its value is repeated for contributing patients in
    ``per_patient`` to retain the common result structure, not as individual
    patient-specific slope estimates.
    """
    t = _threshold(threshold, strict=True)
    pid, yy, pp = _arrays(patient, y, forecast)
    pp = _probabilities("forecast", pp)
    groups = _groups(pid)
    if yardstick is None:
        reference = np.empty(len(yy), dtype=float)
        for _, ix in groups:
            reference[ix] = float(np.mean(yy[ix]))
    else:
        reference = np.asarray(yardstick, dtype=float)
        if reference.ndim != 1 or len(reference) != len(yy):
            raise ValueError("yardstick must have the same length as y")
    reference = _probabilities("yardstick", reference)

    # AUC and AP: pooled rows, and patient-level values only with both outcomes.
    both_groups = [(key, ix) for key, ix in groups
                   if np.any(yy[ix] == 1) and np.any(yy[ix] == 0)]
    auc_per = {key: auc(yy[ix], pp[ix]) for key, ix in both_groups}
    ap_per = {key: average_precision(yy[ix], pp[ix]) for key, ix in both_groups}
    result: dict[str, dict[str, Any]] = {
        "auc": _result(auc(yy, pp), auc_per),
        "average_precision": _result(average_precision(yy, pp), ap_per),
    }

    # threshold_metrics supplies the package's pooled operating-point definitions.
    operating = threshold_metrics(pid, yy, pp, t)
    sens_per: dict[Any, float] = {}
    spec_per: dict[Any, float] = {}
    ppv_per: dict[Any, float] = {}
    npv_per: dict[Any, float] = {}
    for key, ix in groups:
        truth, alarm = yy[ix] == 1, pp[ix] >= t
        rate = float(np.mean(yy[ix]))
        if truth.any() and (~truth).any():
            sens_per[key] = float(np.mean(alarm[truth]))
            spec_per[key] = float(np.mean(~alarm[~truth]))
        if alarm.any():
            ppv_per[key] = float(np.mean(truth[alarm]) - rate)
        if (~alarm).any():
            npv_per[key] = float(np.mean(~truth[~alarm]) - (1 - rate))
    result["sensitivity"] = _result(operating["pooled"]["sensitivity"], sens_per)
    result["specificity"] = _result(operating["pooled"]["specificity"], spec_per)
    result["ppv"] = _result(operating["pooled"]["ppv"], ppv_per,
                            per_patient_definition="PPV minus own event rate")
    result["npv"] = _result(operating["pooled"]["npv"], npv_per,
                            per_patient_definition="NPV minus own non-event rate")

    # Pooled NB is day-weighted. Patient values compare the model's NB with the NB of the
    # own-rate policy (treat a row when its yardstick rate >= t), both scored on the observed
    # outcomes. With the default yardstick (the patient's observed rate) the policy is the
    # better of treat-all and treat-none for that patient.
    pooled_nb = net_benefit(yy, pp, t)
    pooled_all = (float(np.mean(yy) - (1 - np.mean(yy)) * t / (1 - t))
                  if len(yy) else float("nan"))
    nb_per: dict[Any, float] = {}
    for key, ix in groups:
        policy = (reference[ix] >= t).astype(float)
        nb_per[key] = _net_benefit_patient(yy[ix], pp[ix], t) - _net_benefit_patient(yy[ix], policy, t)
    result["net_benefit"] = _result(pooled_nb, nb_per, pooled_treat_all=pooled_all,
                                    per_patient_definition="model NB minus NB of the own-rate policy (treat when yardstick >= t), on observed outcomes")

    # Brier skills: pooled row weights and equal-patient weights as in the design.
    model_brier = float(np.mean((yy - pp) ** 2)) if len(yy) else float("nan")
    reference_brier = float(np.mean((yy - reference) ** 2)) if len(yy) else float("nan")
    pooled_skill = (float(1 - model_brier / reference_brier)
                    if reference_brier > 0 else float("nan"))
    model_by_patient: list[float] = []
    reference_by_patient: list[float] = []
    brier_per: dict[Any, float] = {}
    for key, ix in groups:
        model_loss = float(np.mean((yy[ix] - pp[ix]) ** 2))
        reference_loss = float(np.mean((yy[ix] - reference[ix]) ** 2))
        model_by_patient.append(model_loss)
        reference_by_patient.append(reference_loss)
        if reference_loss > 0:
            brier_per[key] = float(1 - model_loss / reference_loss)
    equal_reference_brier = float(np.mean(reference_by_patient)) if reference_by_patient else float("nan")
    equal_patient_skill = (float(1 - np.mean(model_by_patient) / equal_reference_brier)
                          if equal_reference_brier > 0 else float("nan"))
    result["brier_skill"] = _result(
        pooled_skill, brier_per, mean_override=equal_patient_skill,
        definition="per-patient mean is 1 - mean(patient Brier model) / mean(patient Brier yardstick)",
    )

    # Pooled slope and the paper's single conditional within-patient slope.
    pooled_calibration = calibration_slopes(pid, yy, pp)["pooled_slope"]
    conditional = conditional_within_slope(pid, yy, pp)
    slope_per = {key: float(conditional["slope"]) for key, _ in both_groups}
    result["calibration_slope"] = _result(
        pooled_calibration, slope_per, mean_override=float(conditional["slope"]),
        n_override=int(conditional["patients"]), conditional_fit=conditional,
        per_patient_definition="shared conditional within-patient slope",
    )
    return result


def naive_metrics(patient: Any, y: Any, forecast: Any, threshold: float,
                  population_rate: float) -> dict[str, Any]:
    """Return the usual per-patient metrics against population-level reference values.

    Unlike :func:`nine_metrics`, this deliberately uses the customary pooled
    comparison values. AUC uses 0.5; AP, sensitivity, and PPV use the supplied
    population event rate; specificity and NPV use its complement; raw net
    benefit uses treat-none (zero); and Brier skill uses the population rate as
    its row-level reference. Sensitivity and specificity use separate eligible
    patient sets, matching their usual standalone definitions. ``naive_youden``
    is therefore the sensitivity mean plus the specificity mean minus one.

    Brier-skill individual values are patient-specific ratios, while its
    ``summary.mean`` is the equal-patient ratio of mean forecast and population-
    rate Brier losses. Calibration slope is the shared conditional
    within-patient slope; it is reported as ``"n/a"`` when it is not estimable.
    """
    t = _threshold(threshold, strict=True)
    pid, yy, pp = _arrays(patient, y, forecast)
    pp = _probabilities("forecast", pp)
    rate_array = _probabilities("population_rate", [population_rate])
    pop_rate = float(rate_array[0])
    groups = _groups(pid)

    def entry(values: dict[Any, float], chance_level: float,
              *, mean_override: float | None = None) -> dict[str, Any]:
        return {"per_patient": values,
                "summary": _summary(values, mean_override=mean_override),
                "chance_level": float(chance_level)}

    # Ranking metrics have different patient eligibility rules: AUC needs both
    # classes; AP is defined for every patient with at least one event.
    auc_per: dict[Any, float] = {}
    ap_per: dict[Any, float] = {}
    for key, ix in groups:
        if np.any(yy[ix] == 1) and np.any(yy[ix] == 0):
            auc_per[key] = auc(yy[ix], pp[ix])
        if np.any(yy[ix] == 1):
            ap_per[key] = average_precision(yy[ix], pp[ix])

    alarm = pp >= t
    truth = yy == 1
    sens_per: dict[Any, float] = {}
    spec_per: dict[Any, float] = {}
    ppv_per: dict[Any, float] = {}
    npv_per: dict[Any, float] = {}
    nb_per: dict[Any, float] = {}
    brier_per: dict[Any, float] = {}
    model_losses: list[float] = []
    population_losses: list[float] = []

    for key, ix in groups:
        patient_truth = truth[ix]
        patient_alarm = alarm[ix]
        n_events = int(patient_truth.sum())
        n_non_events = len(ix) - n_events
        if n_events:
            sens_per[key] = float(np.mean(patient_alarm[patient_truth]))
        if n_non_events:
            spec_per[key] = float(np.mean(~patient_alarm[~patient_truth]))
        if patient_alarm.any():
            ppv_per[key] = float(np.mean(patient_truth[patient_alarm]))
        if (~patient_alarm).any():
            npv_per[key] = float(np.mean(~patient_truth[~patient_alarm]))

        nb_per[key] = _net_benefit_patient(yy[ix], pp[ix], t)
        model_loss = float(np.mean((yy[ix] - pp[ix]) ** 2))
        population_loss = float(np.mean((yy[ix] - pop_rate) ** 2))
        model_losses.append(model_loss)
        population_losses.append(population_loss)
        if population_loss > 0:
            brier_per[key] = float(1.0 - model_loss / population_loss)

    sens_mean = float(np.mean(list(sens_per.values()))) if sens_per else float("nan")
    spec_mean = float(np.mean(list(spec_per.values()))) if spec_per else float("nan")
    naive_youden = (sens_mean + spec_mean - 1.0
                    if np.isfinite(sens_mean) and np.isfinite(spec_mean)
                    else float("nan"))

    mean_population_loss = float(np.mean(population_losses)) if population_losses else float("nan")
    mean_model_loss = float(np.mean(model_losses)) if model_losses else float("nan")
    equal_patient_brier_skill = (
        float(1.0 - mean_model_loss / mean_population_loss)
        if mean_population_loss > 0 else float("nan")
    )

    conditional = conditional_within_slope(pid, yy, pp)
    slope = conditional["slope"]
    if np.isfinite(slope):
        # As in nine_metrics, one conditional estimate is repeated for the
        # patients with both outcomes; it is not a set of patient-specific fits.
        both_keys = [key for key, ix in groups
                     if np.any(yy[ix] == 1) and np.any(yy[ix] == 0)]
        slope_per = {key: float(slope) for key in both_keys}
        slope_result: dict[str, Any] = {
            "per_patient": slope_per,
            "summary": _summary(slope_per, mean_override=float(slope),
                                n_override=int(conditional["patients"])),
            "chance_level": "n/a (no within-patient forecast variation)",
            "conditional_fit": conditional,
        }
    else:
        slope_result = {
            "per_patient": {},
            "summary": {"mean": "n/a", "median": "n/a",
                        "iqr": ["n/a", "n/a"], "n": 0},
            "chance_level": "n/a (no within-patient forecast variation)",
            "conditional_fit": conditional,
        }

    result: dict[str, Any] = {
        "auc": entry(auc_per, 0.5),
        "average_precision": entry(ap_per, pop_rate),
        "sensitivity": entry(sens_per, pop_rate),
        "specificity": entry(spec_per, 1.0 - pop_rate),
        "ppv": entry(ppv_per, pop_rate),
        "npv": entry(npv_per, 1.0 - pop_rate),
        "net_benefit": entry(nb_per, 0.0),
        "brier_skill": {
            "per_patient": brier_per,
            "summary": _summary(brier_per,
                                mean_override=equal_patient_brier_skill),
            "chance_level": 0.0,
            "definition": (
                "1 - mean patient Brier(forecast) / mean patient "
                "Brier(population rate)"
            ),
        },
        "calibration_slope": slope_result,
        "naive_youden": naive_youden,
        "population_rate": pop_rate,
    }
    return result
