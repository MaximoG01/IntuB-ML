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
    "mallampati": "Mallampati",
    "video_laryngoscope": "Videolaringoscopio",
    "mallampati_ge3": "Mallampati III–IV",
    "mouth_lt4": "Apertura bocca < 4 cm",
    "dtm": "Distanza tiromentoniera",
    "neck_limited": "Mobilità del collo ≤ 90°",
    "no_prognath": "Prognatismo non possibile",
    "history_difficult_any": "Storia di intubazione difficile",
}


@lru_cache(maxsize=1)
def active_model() -> Optional[dict]:
    """The model named in models/ACTIVE, or None if there is no exported model."""
    try:
        name = (MODELS_DIR / "ACTIVE").read_text(encoding="utf-8").strip()
        return load_model(name)
    except (OSError, ValueError):
        return None


def load_model(name: str) -> dict:
    return json.loads((MODELS_DIR / name / "model.json").read_text(encoding="utf-8"))


STUDY_MODELS = ("decassai-lr-v1", "laryngoscope-lr-v1")


@lru_cache(maxsize=1)
def study_models() -> Dict[str, dict]:
    """Exported models of the project's studies (for the 'Impara' pages), keyed by name."""
    out = {}
    for name in STUDY_MODELS:
        try:
            out[name] = load_model(name)
        except (OSError, ValueError):
            pass
    return out


@lru_cache(maxsize=1)
def teaching_data() -> Optional[dict]:
    """Precomputed material for the 'Impara' pages (models/<ACTIVE>/teaching.json), if any."""
    try:
        name = (MODELS_DIR / "ACTIVE").read_text(encoding="utf-8").strip()
        return json.loads((MODELS_DIR / name / "teaching.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def record_features(r) -> Dict[str, Optional[float]]:
    """Map an IntubationRecord-like object to the model's feature names (None = missing)."""
    bmi = None
    if r.weight and getattr(r, "height", None):
        bmi = r.weight / ((r.height / 100.0) ** 2)
    sex = (getattr(r, "sex", None) or "").upper()
    mallampati = getattr(r, "mallampati", None)
    dii, dtm = getattr(r, "dii", None), getattr(r, "dtm", None)
    neck, prognath = getattr(r, "neck_movement", None), getattr(r, "can_prognath", None)
    history = getattr(r, "history_difficult", None)
    flag = lambda cond, value: None if value is None else float(cond)  # noqa: E731
    return {
        # yes/no items as in the Padova dataset (SPEC §11)
        "mallampati_ge3": flag(mallampati is not None and mallampati >= 3, mallampati),
        "mouth_lt4": flag(dii is not None and dii < 4, dii),
        "dtm": float(dtm) if dtm is not None else None,
        "neck_limited": flag(neck is not None and neck >= 1, neck),  # 0 = > 90°
        "no_prognath": flag(prognath is False, prognath),
        "history_difficult_any": flag(history is not None and history >= 1, history),  # doubtful or certain

        "age": float(r.age) if r.age is not None else None,
        "sex_male": 1.0 if sex == "M" else 0.0 if sex == "F" else None,
        "bmi": bmi,
        "asa": float(r.asa) if getattr(r, "asa", None) is not None else None,
        "emergency": None if getattr(r, "emergency", None) is None else float(bool(r.emergency)),
        "mallampati": float(mallampati) if mallampati is not None else None,
    }


def predict_features(values: Dict[str, Optional[float]], model: dict) -> Tuple[float, List[dict]]:
    """
    Probability from a dict of feature values. Features in model["fixed"] (e.g. the device:
    prediction is for direct laryngoscopy) override the input and are flagged as fixed.
    Contribution = coefficient x standardized value (log-odds); missing values use the
    training median and are flagged as imputed.
    """
    fixed = model.get("fixed", {})
    z = model["intercept"]
    contributions = []
    for f in model["features"]:
        v = fixed.get(f, values.get(f))
        imputed = v is None
        if imputed:
            v = model["impute"][f]
        c = model["coef"][f] * (v - model["mean"][f]) / model["scale"][f]
        z += c
        contributions.append({"feature": f, "label": FEATURE_LABELS.get(f, f), "value": v,
                              "contribution": c, "imputed": imputed, "fixed": f in fixed})
    contributions.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    return 1.0 / (1.0 + math.exp(-z)), contributions


def predict(r, model: Optional[dict] = None) -> Optional[Tuple[float, List[dict]]]:
    """Return (probability, contributions) for a record, or None without an active model."""
    model = model or active_model()
    if model is None:
        return None
    return predict_features(record_features(r), model)
