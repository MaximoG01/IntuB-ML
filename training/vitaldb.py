"""
VitalDB (Seoul National University Hospital), PhysioNet, CC-BY 4.0.
Lee HC et al., Sci Data 2022;9:279. https://physionet.org/content/vitaldb/1.0.0/

Only demographic / pre-operative variables are available: no Mallampati or other
airway examination. Outcome: Cormack-Lehane III (a/b) or IV.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://physionet.org/files/vitaldb/1.0.0/clinical_data.csv"
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "external" / "vitaldb" / "clinical_data.csv"

# height and weight are left out: BMI is computed from them, and with all three the
# coefficients offset each other (collinearity) and cannot be interpreted.
FEATURES = ["age", "sex_male", "bmi", "asa", "emergency"]
CORMACK_MAP = {"I": 1, "II": 2, "IIIa": 3, "IIIb": 3, "IV": 4}


def download(path: Path = DEFAULT_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    return path


def load(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    """Return one row per patient with FEATURES, 'cormack' (1-4) and 'difficult' (0/1)."""
    raw = pd.read_csv(download(path), encoding="utf-8-sig")
    df = raw[raw["cormack"].isin(CORMACK_MAP)].copy()
    df = df[(df["ane_type"] == "General") & (df["airway"] == "Oral") & (df["age"].apply(_is_number))]
    df["age"] = df["age"].astype(float)
    df = df[df["age"] >= 18]
    df = df[df["asa"].isin([1, 2, 3, 4, 5])]  # ASA 6 = organ donor
    # first surgery of each patient: repeated cases of the same person are not independent
    df = df.sort_values(["subjectid", "casestart"]).drop_duplicates("subjectid", keep="first")

    out = pd.DataFrame({
        "age": df["age"],
        "sex_male": (df["sex"] == "M").astype(float),
        "height": pd.to_numeric(df["height"], errors="coerce"),
        "weight": pd.to_numeric(df["weight"], errors="coerce"),
        "bmi": pd.to_numeric(df["bmi"], errors="coerce"),
        "asa": df["asa"].astype(float),
        "emergency": (df["emop"] == 1).astype(float),
        "cormack": df["cormack"].map(CORMACK_MAP),
    })
    out["difficult"] = (out["cormack"] >= 3).astype(int)
    return out.reset_index(drop=True)


def _is_number(x) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False
