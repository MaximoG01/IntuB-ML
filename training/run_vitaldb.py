"""
Compare models on VitalDB and export the pre-specified primary model (penalized LR).

    venv\\Scripts\\python.exe -m training.run_vitaldb

Writes models/vitaldb-demo-lr-v1/ (model.json, model_card.md) and models/ACTIVE.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from training import evaluate, vitaldb

MODEL_NAME = "vitaldb-demo-lr-v1"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
LABELS = {
    "null": "No model (prevalence)",
    "lr": "Penalized logistic (L2)",
    "hgb": "Gradient boosting",
    "mlp": "Neural network (MLP 8)",
}


def make_models(seed: int = 42):
    lr = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        # penalty strength chosen by inner CV inside each outer fold (nested CV)
        ("lr", LogisticRegressionCV(Cs=10, cv=5, scoring="neg_log_loss", max_iter=2000)),
    ])
    return {
        "null": DummyClassifier(strategy="prior"),
        "lr": lr,
        "hgb": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                              l2_regularization=1.0, random_state=seed),
        "mlp": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(8,), alpha=1.0, max_iter=3000, random_state=seed)),
        ]),
    }


def export_lr(pipe: Pipeline, features, info: dict, name: str = MODEL_NAME) -> dict:
    lr = pipe.named_steps["lr"]
    return {
        "name": name,
        "type": "logistic",
        "created": date.today().isoformat(),
        "features": list(features),
        "impute": dict(zip(features, pipe.named_steps["impute"].statistics_.round(6).tolist())),
        "mean": dict(zip(features, pipe.named_steps["scale"].mean_.round(6).tolist())),
        "scale": dict(zip(features, pipe.named_steps["scale"].scale_.round(6).tolist())),
        "coef": dict(zip(features, lr.coef_[0].round(6).tolist())),
        "intercept": round(float(lr.intercept_[0]), 6),
        "C": float(lr.C_[0]),
        "threshold": evaluate.THRESHOLD,
        **info,
    }


def model_card(model: dict, results: dict) -> str:
    d = model["training_data"]
    lines = [
        f"# Model card — {model['name']}",
        "",
        f"Created: {model['created']} · type: penalized logistic regression (L2, C={model['C']:.3g})",
        "",
        "## Data",
        f"- {d['source']} — {d['n']} adult patients (first surgery per patient), "
        f"{d['events']} with Cormack 3–4 ({d['prevalence'] * 100:.1f}%).",
        f"- Variables: {', '.join(model['features'])}. **No airway examination** (not available in VitalDB).",
        "- Korean population, non-cardiac surgery, one centre: transferability unknown.",
        "- Height and weight excluded after a first try: together with BMI (derived from them) they gave "
        "opposite, unstable coefficients (collinearity). Performance was the same (AUC ~0.59).",
        "",
        "## Internal validation (stratified 5-fold CV × 10 repeats)",
        "Values: mean and 2.5–97.5% range across repeats.",
        "",
        "| Model | AUC | AUPRC | Brier | Calibr. slope | Sens. @15% | PPV @15% | Flagged @15% |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, r in results.items():
        m = r["metrics"]
        f = lambda k, pct=False: (f"{m[k]['mean'] * 100:.1f}%" if pct else f"{m[k]['mean']:.3f}") + \
            (f" ({m[k]['lo'] * 100:.1f}–{m[k]['hi'] * 100:.1f})" if pct else f" ({m[k]['lo']:.3f}–{m[k]['hi']:.3f})")
        lines.append(f"| {LABELS[key]} | {f('auc')} | {f('auprc')} | {f('brier')} | {f('cal_slope')} | "
                     f"{f('sensitivity', True)} | {f('ppv', True)} | {f('flagged', True)} |")
    lines += [
        "",
        "## Limitations",
        "- No Mallampati, TMD, mouth opening or neck movement: the model uses demographics only.",
        "- The Cormack grade in VitalDB comes from the electronic record; whether a direct blade or a video laryngoscope was used is unknown.",
        "- Not externally validated. **Do not use for clinical decisions.**",
        "",
        "Data source: Lee HC et al., VitalDB, Sci Data 2022 (CC-BY 4.0), https://physionet.org/content/vitaldb/1.0.0/",
    ]
    return "\n".join(lines) + "\n"


def main():
    df = vitaldb.load()
    X = df[vitaldb.FEATURES].to_numpy(dtype=float)
    y = df["difficult"].to_numpy(dtype=int)
    print(f"VitalDB: {len(y)} patients, {y.sum()} difficult ({y.mean() * 100:.2f}%)")

    models = make_models()
    results = evaluate.compare(models, X, y)
    for key, r in results.items():
        m = r["metrics"]
        print(f"{LABELS[key]:32s} AUC {m['auc']['mean']:.3f} ({m['auc']['lo']:.3f}-{m['auc']['hi']:.3f})"
              f"  Brier {m['brier']['mean']:.4f}  slope {m['cal_slope']['mean']:.2f}"
              f"  sens@15% {m['sensitivity']['mean']:.2f}  flagged@15% {m['flagged']['mean']:.3f}")

    final = models["lr"].fit(X, y)
    info = {
        "training_data": {"source": "VitalDB 1.0.0 (PhysioNet, CC-BY 4.0)", "n": int(len(y)),
                          "events": int(y.sum()), "prevalence": float(y.mean())},
        "validation": {"method": "repeated stratified 5-fold CV x10 (nested for C)",
                       "results": {LABELS[k]: v for k, v in results.items()},
                       "dca_reference": evaluate.dca_reference(y)},
    }
    model = export_lr(final, vitaldb.FEATURES, info)

    out = MODELS_DIR / MODEL_NAME
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(json.dumps(model, indent=1), encoding="utf-8")
    (out / "model_card.md").write_text(model_card(model, results), encoding="utf-8")
    (MODELS_DIR / "ACTIVE").write_text(MODEL_NAME + "\n", encoding="utf-8")
    print(f"Exported {out}")


if __name__ == "__main__":
    main()
