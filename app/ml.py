# app/ml.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.models import IntubationRecord

# sklearn is required for LR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DIFFICULT_THRESHOLD = 0.15  # classification threshold used in UI - lowered from 0.5 due to poor calibration on imbalanced data


@dataclass
class Confusion:
    tn: int
    fp: int
    fn: int
    tp: int


@dataclass
class LogisticMetrics:
    n_samples: int
    accuracy: float
    auc: Optional[float]
    threshold: float
    confusion: Confusion
    fpr: List[float]
    tpr: List[float]
    data_source: str  # 'database'


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x, default=0) -> int:
    try:
        if x is None:
            return int(default)
        return int(x)
    except Exception:
        return int(default)


def build_feature_vector(r: IntubationRecord) -> List[float]:
    """
    Feature engineering is deliberately simple and stable.

    Sex encoding:
      M -> 1, F -> 0, O/None -> 0.5
    """
    sex = (r.sex or "").strip().upper()
    sex_code = 0.5
    if sex == "M":
        sex_code = 1.0
    elif sex == "F":
        sex_code = 0.0

    age = _safe_float(r.age)
    weight = _safe_float(r.weight)
    height = _safe_float(r.height, default=np.nan)
    dtm = _safe_float(r.dtm, default=np.nan)
    dii = _safe_float(r.dii, default=np.nan)

    # Derived BMI (if height in cm)
    bmi = np.nan
    if not (height is None or math.isnan(height) or height <= 0):
        bmi = weight / ((height / 100.0) ** 2)

    mall = _safe_float(r.mallampati, default=np.nan)
    stop = _safe_float(r.stop_bang, default=np.nan)
    alg = _safe_float(r.alganzouri, default=np.nan)

    # replace NaNs with median-like constants (keeps model stable even with missing)
    def fill(v, fallback):
        return fallback if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)

    return [
        fill(age, 50.0),
        fill(weight, 75.0),
        fill(height, 170.0),
        fill(bmi, 26.0),
        sex_code,
        fill(dtm, 6.5),
        fill(dii, 4.0),
        fill(mall, 2.0),
        fill(stop, 3.0),
        fill(alg, 6.0),
    ]


class InsufficientDataError(RuntimeError):
    """Not enough labeled cases (or only one class) to train a model."""


def _load_xy(min_samples: int = 50) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load labeled records (difficult_binary not None) from the database.
    No synthetic fallback: raises InsufficientDataError if data are not enough.
    Returns: (X, y, data_source) with data_source = 'database'.
    """
    records = IntubationRecord.query.filter(IntubationRecord.difficult_binary.isnot(None)).all()
    n_difficult = sum(1 for r in records if r.difficult_binary)
    if len(records) < min_samples or n_difficult < 2 or len(records) - n_difficult < 2:
        raise InsufficientDataError(
            f"Not enough labeled cases to train a model: {len(records)} cases "
            f"({n_difficult} difficult). At least {min_samples} are needed, with both outcomes present."
        )
    X = np.array([build_feature_vector(r) for r in records], dtype=float)
    y = np.array([1.0 if r.difficult_binary else 0.0 for r in records], dtype=float)
    return X, y, "database"


def train_logistic_model(
    min_samples: int = 50,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, float]]:
    X, y, _ = _load_xy(min_samples=min_samples)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, solver="lbfgs")),
    ])
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_val)[:, 1]
    pred = (proba >= DIFFICULT_THRESHOLD).astype(int)

    metrics = {
        "n_train": float(len(X_train)),
        "n_val": float(len(X_val)),
        "accuracy_val": float(accuracy_score(y_val, pred)),
        "auc_val": float(roc_auc_score(y_val, proba)) if len(np.unique(y_val)) > 1 else float("nan"),
    }
    return model, metrics


def evaluate_logistic(min_samples: int = 50) -> LogisticMetrics:
    X, y, data_source = _load_xy(min_samples=min_samples)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, solver="lbfgs")),
    ])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_val)[:, 1]
    pred = (proba >= DIFFICULT_THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_val.astype(int), pred, labels=[0, 1]).ravel()

    auc = None
    fpr_list, tpr_list = [], []
    if len(np.unique(y_val)) > 1:
        fpr, tpr, _ = roc_curve(y_val, proba)
        fpr_list, tpr_list = fpr.tolist(), tpr.tolist()
        auc = float(roc_auc_score(y_val, proba))

    return LogisticMetrics(
        n_samples=int(len(X)),
        accuracy=float(accuracy_score(y_val, pred)),
        auc=auc,
        threshold=float(DIFFICULT_THRESHOLD),
        confusion=Confusion(int(tn), int(fp), int(fn), int(tp)),
        fpr=fpr_list,
        tpr=tpr_list,
        data_source=data_source,
    )
