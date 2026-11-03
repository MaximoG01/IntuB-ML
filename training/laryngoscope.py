"""
Laryngoscope trial (R package medicaldata, MIT licence).
Abdallah et al., Anesth Analg 2011;113:1082-7: 99 obese adults, Macintosh vs Pentax AWS.
Nowacki AS, 'Laryngoscope Dataset', TSHS Resources Portal (2017).

Outcome coding: the package documentation contradicts itself on `view`. The data show that
view == 0 goes with harder, longer, less successful intubations, so view == 0 = Cormack 3-4
(see SPEC §10).
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

COMMIT = "be27ec88ac8955b4babe1456e6e6d6a7f3c117cb"
URL = f"https://raw.githubusercontent.com/higgi13425/medicaldata/{COMMIT}/data/laryngoscope.rda"
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "external" / "laryngoscope" / "laryngoscope.rda"
SOURCE = "medicaldata::laryngoscope (Abdallah et al., Anesth Analg 2011; MIT licence)"

FEATURES = ["mallampati", "bmi", "age", "sex_male", "video_laryngoscope"]


def download(path: Path = DEFAULT_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    return path


def sha256(path: Path = DEFAULT_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    """One row per patient: FEATURES (NaN = missing), 'asa' and 'difficult' (1 = Cormack 3-4)."""
    import pyreadr  # training-only dependency

    raw = next(iter(pyreadr.read_r(str(download(path))).values()))
    return pd.DataFrame({
        "mallampati": raw["Mallampati"].astype(float),
        "bmi": raw["BMI"].astype(float),
        "age": raw["age"].astype(float),
        "sex_male": raw["gender"].astype(float),
        "video_laryngoscope": raw["Randomization"].astype(float),
        "asa": raw["asa"].astype(float),
        "difficult": (raw["view"] == 0).astype(int),
    })
