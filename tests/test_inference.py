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


def test_missing_values_use_training_median():
    model = {"features": ["bmi"], "impute": {"bmi": 23.0}, "mean": {"bmi": 23.0},
             "scale": {"bmi": 3.0}, "coef": {"bmi": 0.5}, "intercept": 0.0}
    r = SimpleNamespace(age=50, sex=None, weight=70.0, height=None, asa=None, emergency=None)
    proba, contributions = inference.predict(r, model)
    assert proba == 0.5 and contributions[0]["imputed"] is True
