"""
Thyroid surgery cohort, Padova (De Cassai A et al., PLoS ONE 2019;14:e0212976, CC-BY 4.0).
500 adults scheduled for thyroidectomy; patients with an expected difficult intubation excluded.
Outcome: Cormack-Lehane 3-4 with a conventional curved blade (SPEC §11).

Sex is coded M = 0, so 1 = female. PROGNAT = 1 means the patient cannot protrude the jaw.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0212976.s004"
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "external" / "decassai2019" / "S1_Database.xlsx"
SOURCE = "De Cassai et al., PLoS ONE 2019 (Padova, thyroid surgery; CC-BY 4.0)"

FEATURES = ["mallampati_ge3", "mouth_lt4", "dtm", "neck_limited", "no_prognath", "history_difficult_any", "bmi"]

COLUMNS = {  # original header (whitespace-normalized) -> our name
    "MALLAMP (<3=0)": "mallampati_ge3",
    "AP MOUTH (<4=1)": "mouth_lt4",
    "TMD": "dtm",
    "NECK MOV (>90=0)": "neck_limited",
    "PROGNAT (NO=0)": "no_prognath",
    "PAST DI (NO=0)": "history_difficult_any",
    "BMI": "bmi",
    "AGE": "age",
    "GOITER CIRC.": "neck_circumference",
    "El-GANZURI (<4=0)": "egri_ge4",
    "CORMACK (<3=0)": "difficult",
}


def download(path: Path = DEFAULT_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    return path


def sha256(path: Path = DEFAULT_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    raw = pd.read_excel(download(path))
    raw.columns = [" ".join(str(c).split()) for c in raw.columns]
    df = raw[list(COLUMNS)].rename(columns=COLUMNS).astype(float)
    df["sex_male"] = 1.0 - raw["SEX (M=O)"].astype(float)
    df["difficult"] = df["difficult"].astype(int)
    return df
