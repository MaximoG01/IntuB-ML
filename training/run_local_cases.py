"""
Check, and if there are enough events update, the active model on the cases collected in the app.

    1. On the site: log in as administrator -> Users -> "Download all cases (CSV)"
    2. Save the file as data/local/intub-cases.csv (data/ is never committed)
    3. venv\\Scripts\\python.exe -m training.run_local_cases            # validation only
       venv\\Scripts\\python.exe -m training.run_local_cases --export   # also write an updated model

Only cases with a Cormack grade seen with direct laryngoscopy (DL) are used: that is the outcome the model
predicts. Features are computed with app.inference.record_features, exactly as on the Assess page.

Updating follows Steyerberg (Clinical Prediction Models, ch. 20), never a full refit on a few local cases:
  < 10 events   validation only
  10-49 events  recalibration in the large (new intercept, same weights)
  >= 50 events  logistic recalibration (new intercept and one common slope for all weights)
The updated model is written next to the others but NOT made active: compare it, then edit models/ACTIVE.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from app.inference import MODELS_DIR, active_model, predict_features, record_features

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "local" / "intub-cases.csv"
INTS = {"age", "asa", "dii", "mallampati", "neck_movement", "history_difficult", "stop_bang", "cormack"}
FLOATS = {"weight", "height", "dtm", "alganzouri", "predicted_proba"}
BOOLS = {"emergency", "can_prognath", "success", "difficult_binary"}


def _value(col: str, raw: str):
    raw = (raw or "").strip()
    if raw == "":
        return None  # empty = missing, never 0
    if col in INTS:
        return int(float(raw))
    if col in FLOATS:
        return float(raw)
    if col in BOOLS:
        return raw.lower() in ("true", "1", "yes")
    return raw


def load_cases(path: Path) -> tuple[list[SimpleNamespace], dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        blank = dict.fromkeys(INTS | FLOATS | BOOLS | {"sex", "cormack_device"})  # absent column = missing
        rows = [SimpleNamespace(**{**blank, **{k: _value(k, v) for k, v in r.items()}}) for r in csv.DictReader(fh)]
    usable = [r for r in rows if r.cormack is not None and (r.cormack_device or "DL") == "DL"]
    counts = {"rows": len(rows), "no_outcome": sum(r.cormack is None for r in rows),
              "video": sum(r.cormack is not None and r.cormack_device == "VL" for r in rows), "used": len(usable)}
    return usable, counts


def linear_predictor(cases, model) -> np.ndarray:
    p = np.array([predict_features(record_features(r), model)[0] for r in cases])
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def validate(lp: np.ndarray, y: np.ndarray) -> dict:
    p = 1 / (1 + np.exp(-lp))
    out = {"n": int(len(y)), "events": int(y.sum()), "observed": float(y.mean()), "mean_predicted": float(p.mean()),
           "o_e": float(y.sum() / p.sum()), "brier": float(brier_score_loss(y, p))}
    if 0 < y.sum() < len(y):
        out["auc"] = float(roc_auc_score(y, p))
    if y.sum() >= 10 and len(y) - y.sum() >= 10:
        fit = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y)
        out["cal_slope"] = float(fit.coef_[0, 0])
    return out


def intercept_shift(lp: np.ndarray, y: np.ndarray) -> float:
    """Recalibration in the large: a such that mean(sigmoid(lp + a)) = mean(y) (Newton, lp as offset)."""
    a = 0.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-(lp + a)))
        step = (y.sum() - p.sum()) / max((p * (1 - p)).sum(), 1e-9)
        a += step
        if abs(step) < 1e-8:
            break
    return float(a)


def updated_model(model: dict, lp: np.ndarray, y: np.ndarray, source: str) -> tuple[dict, str] | None:
    events = int(y.sum())
    if events < 10:
        return None
    new = json.loads(json.dumps(model))
    if events < 50:
        a, b, how = intercept_shift(lp, y), 1.0, "recalibration in the large (intercept only)"
    else:
        fit = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y)
        a, b, how = float(fit.intercept_[0]), float(fit.coef_[0, 0]), "logistic recalibration (intercept + slope)"
    # z' = a + b*z, with z = intercept + sum(coef * standardized value)
    new["intercept"] = a + b * model["intercept"]
    new["coef"] = {f: b * c for f, c in model["coef"].items()}
    for sm in new.get("submodels", {}).values():
        sm["intercept"] = a + b * sm["intercept"]
        sm["coef"] = {f: b * c for f, c in sm["coef"].items()}
    new["name"] = f"{model['name']}-local-{date.today():%Y%m%d}"
    new["created"] = date.today().isoformat()
    new["updated_from"] = {"model": model["name"], "method": how, "a": a, "b": b,
                           "local_cases": int(len(y)), "local_events": events, "source": source}
    return new, how


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--export", action="store_true", help="write the updated model to models/ (not made active)")
    args = ap.parse_args()
    if not args.csv.exists():
        raise SystemExit(f"{args.csv} not found: download it from the site (Users page) and save it there.")

    model = active_model()
    cases, counts = load_cases(args.csv)
    print(f"Rows {counts['rows']}: {counts['no_outcome']} without Cormack, {counts['video']} with video "
          f"laryngoscope (excluded), {counts['used']} used.")
    if not cases:
        raise SystemExit("No usable case yet.")
    y = np.array([int(r.cormack >= 3) for r in cases])
    lp = linear_predictor(cases, model)

    v = validate(lp, y)
    print(f"\nActive model {model['name']} on local cases: {v['n']} patients, {v['events']} Cormack 3-4 "
          f"({v['observed'] * 100:.1f}%), mean predicted {v['mean_predicted'] * 100:.1f}%, O/E {v['o_e']:.2f}")
    print(f"  AUC {v['auc']:.2f}" if "auc" in v else "  AUC: needs both easy and difficult cases")
    print(f"  calibration slope {v['cal_slope']:.2f} (1 = ideal)" if "cal_slope" in v
          else "  calibration slope: needs at least 10 events and 10 non-events")

    result = updated_model(model, lp, y, args.csv.name)
    if result is None:
        print(f"\n{v['events']} events: too few to update the model (need >= 10). Keep collecting.")
        return
    new, how = result
    print(f"\nUpdate possible: {how}. New intercept {new['intercept']:.3f}"
          f" (was {model['intercept']:.3f}), weights x {new['updated_from']['b']:.2f}.")
    if args.export:
        out = MODELS_DIR / new["name"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.json").write_text(json.dumps(new, indent=1), encoding="utf-8")
        teaching = MODELS_DIR / model["name"] / "teaching.json"  # the Learn pages read it from the active model
        if teaching.exists():
            shutil.copy(teaching, out / "teaching.json")
        print(f"Written {out}. To use it in the app: write '{new['name']}' in models/ACTIVE, commit and push.")
    else:
        print("Run again with --export to write it.")


if __name__ == "__main__":
    main()
