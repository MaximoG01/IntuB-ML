# app/egri.py
"""
El-Ganzouri Risk Index (EGRI), el-Ganzouri et al., Anesth Analg 1996;82:1197-204.
7 items, total 0-12, positive if >= 4. Pure Python: used by the app and by training/.

Mallampati: the 1996 paper used the original 3-class Mallampati (I=0, II=1, III=2).
With the modified 4-class version, III and IV both score 2 here. Some secondary
sources score I-II=0, III=1, IV=2 instead: verify against the original paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

EGRI_POSITIVE = 4

# Coded categories stored in the database (see app/models.py)
NECK_MOVEMENT_CHOICES = {0: "> 90°", 1: "80–90°", 2: "< 80°"}
HISTORY_CHOICES = {0: "None", 1: "Questionable", 2: "Definite"}


@dataclass
class EgriResult:
    total: Optional[int]          # None if any item is missing
    partial: int                  # sum of the items that are available
    items: Dict[str, Optional[int]]
    missing: list

    @property
    def positive(self) -> Optional[bool]:
        return None if self.total is None else self.total >= EGRI_POSITIVE


def _mouth_opening(cm: Optional[float]) -> Optional[int]:
    if cm is None:
        return None
    return 0 if cm >= 4.0 else 1


def _thyromental(cm: Optional[float]) -> Optional[int]:
    if cm is None:
        return None
    if cm > 6.5:
        return 0
    return 1 if cm >= 6.0 else 2


def _mallampati(cls: Optional[int]) -> Optional[int]:
    if cls is None:
        return None
    return {1: 0, 2: 1}.get(int(cls), 2)


def _weight(kg: Optional[float]) -> Optional[int]:
    if kg is None:
        return None
    if kg < 90:
        return 0
    return 1 if kg <= 110 else 2


def egri_score(
    mouth_opening_cm: Optional[float] = None,
    thyromental_cm: Optional[float] = None,
    mallampati: Optional[int] = None,
    neck_movement: Optional[int] = None,       # coded 0/1/2, see NECK_MOVEMENT_CHOICES
    can_prognath: Optional[bool] = None,
    weight_kg: Optional[float] = None,
    history_difficult: Optional[int] = None,   # coded 0/1/2, see HISTORY_CHOICES
) -> EgriResult:
    items = {
        "mouth_opening": _mouth_opening(mouth_opening_cm),
        "thyromental": _thyromental(thyromental_cm),
        "mallampati": _mallampati(mallampati),
        "neck_movement": None if neck_movement is None else int(neck_movement),
        "prognathism": None if can_prognath is None else (0 if can_prognath else 1),
        "weight": _weight(weight_kg),
        "history": None if history_difficult is None else int(history_difficult),
    }
    missing = [k for k, v in items.items() if v is None]
    partial = sum(v for v in items.values() if v is not None)
    return EgriResult(total=None if missing else partial, partial=partial, items=items, missing=missing)


def egri_from_record(r) -> EgriResult:
    """Score any object with the IntubationRecord attribute names."""
    return egri_score(
        mouth_opening_cm=getattr(r, "dii", None),
        thyromental_cm=getattr(r, "dtm", None),
        mallampati=getattr(r, "mallampati", None),
        neck_movement=getattr(r, "neck_movement", None),
        can_prognath=getattr(r, "can_prognath", None),
        weight_kg=getattr(r, "weight", None),
        history_difficult=getattr(r, "history_difficult", None),
    )
