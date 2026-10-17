"""evaluate(model=, features=, history=): the one-call interface for a fitted model."""
import warnings

import numpy as np
import pandas as pd
import pytest

import wheneval as we


class Logistic:
    """A fixed logistic model with predict_proba, as a fitted scikit-learn model would offer."""

    def __init__(self, coef, intercept=0.0):
        self.coef = np.asarray(coef, dtype=float)
        self.intercept = intercept

    def predict_proba(self, X):
        z = self.intercept + np.asarray(X, dtype=float) @ self.coef
        p = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p, p])


def cohort(n_patients=30, n_days=40, seed=1):
    """Independent days (no within-patient skill possible) with a sensor column and history columns."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_patients):
        rate = rng.uniform(0.15, 0.5)
        y = (rng.random(n_days) < rate).astype(int)
        sensor = rng.normal(size=n_days)
        for t in range(n_days):
            past = y[:t]
            rows.append({"pid": k, "day": t, "event": y[t], "sensor": sensor[t],
                         "rate_so_far": past.mean() if t else 0.3,
                         "last": past[-1] if t else 0.0,
                         "count_7": past[-7:].sum()})
    return pd.DataFrame(rows)


def test_sensor_only_model_equals_fixed_predictions():
    d = cohort()
    m = Logistic([0.8])
    a = we.evaluate(d, patient_id="pid", time="day", outcome="event", model=m,
                    features=["sensor"], metrics=["auc"], n_shuffles=50, seed=3)
    d2 = d.assign(p=m.predict_proba(d[["sensor"]])[:, 1])
    b = we.evaluate(d2, patient_id="pid", time="day", outcome="event", prediction="p",
                    metrics=["auc"], n_shuffles=50, seed=3)
    assert a["metrics"]["auc"]["per_patient"] == b["metrics"]["auc"]["per_patient"]


def test_declared_history_is_rebuilt_and_centres_the_excess():
    d = cohort(n_patients=60, n_days=30, seed=2)
    m = Logistic([3.0, 0.2], intercept=-1.0)  # forecast rises with the rate so far
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = we.evaluate(d, patient_id="pid", time="day", outcome="event", model=m,
                        features=["rate_so_far", "sensor"],
                        history={"rate_so_far": we.rate_so_far()},
                        metrics=["auc"], n_shuffles=100, seed=5)
    pp = r["metrics"]["auc"]["per_patient"].values()
    scores = np.array([v["score"] for v in pp])
    excess = np.array([v["excess"] for v in pp])
    baselines = np.array([v["shuffled_baseline"] for v in pp])
    # Finite-record bias: no-skill AUC below 0.5, and the rebuilt baseline follows it.
    assert scores.mean() < 0.47
    assert baselines.mean() < 0.47
    assert abs(excess.mean()) < 3 * excess.std(ddof=1) / np.sqrt(len(excess))


def test_same_result_as_an_explicit_rule():
    d = cohort(n_patients=10, n_days=25, seed=4)
    m = Logistic([2.0, 1.0, 0.5])
    feats = ["rate_so_far", "last", "sensor"]
    a = we.evaluate(d, patient_id="pid", time="day", outcome="event", model=m, features=feats,
                    history={"rate_so_far": we.rate_so_far(), "last": we.lag(1)},
                    metrics=["auc", "brier_skill"], n_shuffles=30, seed=9)

    def rule(y, scored, rows):
        y = np.asarray(y, dtype=float)
        X = rows[feats].copy()
        X["rate_so_far"] = [y[:t].mean() if t else 0.3 for t in range(len(y))]
        X["last"] = [y[t - 1] if t else 0.0 for t in range(len(y))]
        return m.predict_proba(X)[:, 1]

    b = we.evaluate(d, patient_id="pid", time="day", outcome="event", prediction=None,
                    history_rule=rule, metrics=["auc", "brier_skill"], n_shuffles=30, seed=9)
    for name in ("auc", "brier_skill"):
        for k, v in a["metrics"][name]["per_patient"].items():
            w = b["metrics"][name]["per_patient"][k]
            assert v["score"] == pytest.approx(w["score"])
            assert v["shuffled_baseline"] == pytest.approx(w["shuffled_baseline"])


def test_features_function():
    d = cohort(n_patients=8, n_days=20, seed=6)
    m = Logistic([2.0, 0.5])

    def make(rows):
        y = rows["event"].to_numpy(dtype=float)
        rate = [y[:t].mean() if t else 0.3 for t in range(len(y))]
        return pd.DataFrame({"rate": rate, "sensor": rows["sensor"].to_numpy()})

    a = we.evaluate(d, patient_id="pid", time="day", outcome="event", model=m, features=make,
                    metrics=["auc"], n_shuffles=20, seed=1)
    b = we.evaluate(d, patient_id="pid", time="day", outcome="event", model=m,
                    features=["rate_so_far", "sensor"], history={"rate_so_far": we.rate_so_far()},
                    metrics=["auc"], n_shuffles=20, seed=1)
    assert a["metrics"]["auc"]["per_patient"] == b["metrics"]["auc"]["per_patient"]


def test_undeclared_history_column_warns():
    d = cohort(n_patients=6, n_days=20)
    with pytest.warns(UserWarning, match=r"count_7.*we\.window_count\(7\)"):
        we.evaluate(d, patient_id="pid", time="day", outcome="event", model=Logistic([0.5, 0.1]),
                    features=["count_7", "sensor"], metrics=["auc"], n_shuffles=5)


def test_wrong_recipe_is_an_error():
    d = cohort(n_patients=4, n_days=15)
    with pytest.raises(ValueError, match="is not we.window_count\\(3\\)"):
        we.evaluate(d, patient_id="pid", time="day", outcome="event", model=Logistic([0.5]),
                    features=["count_7"], history={"count_7": we.window_count(3)},
                    metrics=["auc"], n_shuffles=5)


def test_argument_errors():
    d = cohort(n_patients=3, n_days=10)
    with pytest.raises(ValueError, match="need a model"):
        we.evaluate(d, patient_id="pid", time="day", outcome="event", prediction="sensor",
                    features=["sensor"])
    with pytest.raises(ValueError, match="requires features"):
        we.evaluate(d, patient_id="pid", time="day", outcome="event", model=Logistic([1.0]))
    with pytest.raises(ValueError, match="not among the features"):
        we.evaluate(d, patient_id="pid", time="day", outcome="event", model=Logistic([1.0]),
                    features=["sensor"], history={"last": we.lag(1)})


def test_recipes():
    y = np.array([1, 0, 1, 1, 0], dtype=float)
    stops = np.arange(5)
    t = np.array([0, 1, 2, 5, 6], dtype=float)
    np.testing.assert_allclose(we.lag(1).compute(y, stops, t), [np.nan, 1, 0, 1, 1])
    np.testing.assert_allclose(we.rate_so_far().compute(y, stops, t), [np.nan, 1, .5, 2 / 3, .75])
    np.testing.assert_allclose(we.window_rate(2).compute(y, stops, t), [np.nan, 1, .5, .5, 1])
    np.testing.assert_allclose(we.window_count(2).compute(y, stops, t), [np.nan, 1, 1, 1, 2])
    np.testing.assert_allclose(we.time_since_event().compute(y, stops, t), [np.nan, 1, 2, 3, 1])
