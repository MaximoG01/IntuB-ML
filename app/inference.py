# app/inference.py
"""
Load the exported model (models/<ACTIVE>/model.json) and predict with numpy only.
Training happens offline in training/; the app never trains the active model.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

FEATURE_LABELS = {
    "age": "Età",
    "sex_male": "Sesso maschile",
    "bmi": "BMI",
    "asa": "ASA",
    "emergency": "Urgenza",
}


@lru_cache(maxsize=1)
def active_model() -> Optional[dict]:
    """The model named in models/ACTIVE, or None if there is no exported model."""
    try:
        name = (MODELS_DIR / "ACTIVE").read_text(encoding="utf-8").strip()
        return json.loads((MODELS_DIR / name / "model.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def record_features(r) -> Dict[str, Optional[float]]:
    """Map an IntubationRecord-like object to the model's feature names (None = missing)."""
    bmi = None
    if r.weight and r.height:
        bmi = r.weight / ((r.height / 100.0) ** 2)
    sex = (r.sex or "").upper()
    return {
        "age": float(r.age) if r.age is not None else None,
        "sex_male": 1.0 if sex == "M" else 0.0 if sex == "F" else None,
        "bmi": bmi,
        "asa": float(r.asa) if getattr(r, "asa", None) is not None else None,
        "emergency": None if getattr(r, "emergency", None) is None else float(bool(r.emergency)),
    }


def predict(r, model: Optional[dict] = None) -> Optional[Tuple[float, List[dict]]]:
    """
    Return (probability, contributions) or None without an active model.
    Contribution = coefficient x standardized value (log-odds); missing values use the
    training median and are flagged as imputed.
    """
    model = model or active_model()
    if model is None:
        return None
    values = record_features(r)
    z = model["intercept"]
    contributions = []
    for f in model["features"]:
        v = values.get(f)
        imputed = v is None
        if imputed:
            v = model["impute"][f]
        c = model["coef"][f] * (v - model["mean"][f]) / model["scale"][f]
        z += c
        contributions.append({"feature": f, "label": FEATURE_LABELS.get(f, f), "value": v,
                              "contribution": c, "imputed": imputed})
    contributions.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    return 1.0 / (1.0 + math.exp(-z)), contributions
