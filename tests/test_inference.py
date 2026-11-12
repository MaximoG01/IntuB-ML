from types import SimpleNamespace

import numpy as np

from app import inference
from training import run_vitaldb


def test_exported_model_matches_sklearn():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(55, 15, 400), rng.integers(0, 2, 400), rng.normal(25, 4, 400),
                         rng.integers(1, 4, 400), rng.integers(0, 2, 400)]).astype(float)
    y = (rng.random(400) < 1 / (1 + np.exp(-(-3 + 0.04 * (X[:, 0] - 55))))).astype(int)
    pipe = run_vitaldb.make_models()["lr"].fit(X, y)
    model = run_vitaldb.export_lr(pipe, ["age", "sex_male", "bmi", "asa", "emergency"], {})

    r = SimpleNamespace(age=70, sex="M", weight=90.0, height=175.0, asa=2, emergency=False)
    bmi = 90.0 / 1.75 ** 2
    expected = pipe.predict_proba(np.array([[70, 1.0, bmi, 2, 0.0]]))[0, 1]
    proba, contributions = inference.predict(r, model)
    assert abs(proba - expected) < 1e-4
    assert {c["feature"] for c in contributions} == set(model["features"])


def test_form_fields_map_to_padova_items():
    r = SimpleNamespace(age=50, sex="F", weight=80.0, height=160.0, asa=None, emergency=None, mallampati=3,
                        dii=3.5, dtm=6.0, neck_movement=0, can_prognath=False, history_difficult=1)
    f = inference.record_features(r)
    assert (f["mallampati_ge3"], f["mouth_lt4"], f["dtm"], f["neck_limited"]) == (1.0, 1.0, 6.0, 0.0)
    assert (f["no_prognath"], f["history_difficult_any"]) == (1.0, 1.0)
    empty = inference.record_features(SimpleNamespace(age=50, weight=80.0))
    assert all(empty[k] is None for k in ("mallampati_ge3", "mouth_lt4", "dtm", "neck_limited", "no_prognath"))


def test_fixed_features_override_input():
    model = {"features": ["video_laryngoscope"], "impute": {"video_laryngoscope": 1.0},
             "mean": {"video_laryngoscope": 0.5}, "scale": {"video_laryngoscope": 0.5},
             "coef": {"video_laryngoscope": -1.0}, "intercept": 0.0, "fixed": {"video_laryngoscope": 0.0}}
    proba, contributions = inference.predict_features({"video_laryngoscope": 1.0}, model)
    assert abs(proba - 1 / (1 + np.exp(-1.0))) < 1e-9
    assert contributions[0]["fixed"] is True and contributions[0]["value"] == 0.0


def test_missing_values_use_training_median():
    model = {"features": ["bmi"], "impute": {"bmi": 23.0}, "mean": {"bmi": 23.0},
             "scale": {"bmi": 3.0}, "coef": {"bmi": 0.5}, "intercept": 0.0}
    r = SimpleNamespace(age=50, sex=None, weight=70.0, height=None, asa=None, emergency=None)
    proba, contributions = inference.predict(r, model)
    assert proba == 0.5 and contributions[0]["imputed"] is True


def test_missing_sign_uses_submodel_not_median():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from training import submodels

    rng = np.random.default_rng(1)
    X = np.column_stack([rng.integers(0, 2, 300), rng.normal(7.5, 2, 300)]).astype(float)
    y = (rng.random(300) < 1 / (1 + np.exp(-(-2 + 1.2 * X[:, 0])))).astype(int)
    feats = ["mallampati_ge3", "dtm"]
    sc = StandardScaler().fit(X)
    full = LogisticRegression(C=1.0).fit(sc.transform(X), y)
    model = {"features": feats, "impute": {"mallampati_ge3": 0.0, "dtm": 7.5},
             "mean": dict(zip(feats, sc.mean_)), "scale": dict(zip(feats, sc.scale_)),
             "coef": dict(zip(feats, full.coef_[0])), "intercept": float(full.intercept_[0]),
             "submodels": submodels.pattern_submodels(X, y, feats, 1.0)}
    proba, contributions = inference.predict_features({"mallampati_ge3": None, "dtm": 6.0}, model)
    only_dtm = LogisticRegression(C=1.0).fit(sc.transform(X)[:, [1]], y)
    expected = only_dtm.predict_proba(np.array([[(6.0 - sc.mean_[1]) / sc.scale_[1]]]))[0, 1]
    assert abs(proba - expected) < 1e-4
    assert [c["feature"] for c in contributions if c["omitted"]] == ["mallampati_ge3"]
    nothing, _ = inference.predict_features({}, model)
    assert abs(nothing - y.mean()) < 1e-4            # nothing given: the prevalence
