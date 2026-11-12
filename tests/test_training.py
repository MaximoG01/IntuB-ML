import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from training import thyroid, evaluate, laryngoscope


def test_fold_auc_of_null_model_is_one_half():
    # pooled out-of-fold AUC of a constant model drops below 0.5 with few events; per fold it must not
    y = np.r_[np.ones(18), np.zeros(81)].astype(int)
    X = np.zeros((len(y), 1))
    assert evaluate.fold_auc(DummyClassifier(strategy="prior"), X, y, n_repeats=3)["mean"] == 0.5


def test_riley_needs_more_patients_with_more_predictors():
    few = evaluate.riley_sample_size(0.18, 2)
    many = evaluate.riley_sample_size(0.18, 10)
    assert many["n_required"] > few["n_required"] >= few["n_intercept"]
    assert few["n_intercept"] == np.ceil(1.96 ** 2 / 0.05 ** 2 * 0.18 * 0.82)


@pytest.mark.skipif(not thyroid.DEFAULT_PATH.exists(), reason="dataset not downloaded")
def test_thyroid_coding():
    df = thyroid.load()
    assert len(df) == 500 and df["difficult"].sum() == 48
    assert 0.2 < df["sex_male"].mean() < 0.25  # thyroid surgery: mostly women
    assert df[thyroid.FEATURES].notna().all().all()


@pytest.mark.skipif(not laryngoscope.DEFAULT_PATH.exists(), reason="dataset not downloaded")
def test_laryngoscope_outcome_coding():
    df = laryngoscope.load()
    assert len(df) == 99 and df["difficult"].sum() == 18
