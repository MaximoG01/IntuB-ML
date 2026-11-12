"""
Pre-specified study on the Padova thyroid-surgery cohort (SPEC §11) and export of the primary model.

    venv\\Scripts\\python.exe -m training.run_thyroid

Writes models/thyroid-lr-v1/ (model.json, model_card.md, teaching.json) and models/ACTIVE.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app import inference
from training import thyroid, evaluate, run_vitaldb, teaching
from training.run_laryngoscope import (N_REPEATS, coef_bootstrap, coefficient_lines, cv_with_ci, fmt_ci,
                                       optimism_table, probast_and_limitations)

MODEL_NAME = "thyroid-lr-v1"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
LABELS = {
    "null": "No model (prevalence)",
    "egri": "El-Ganzouri ≥ 4 (recalibrated)",
    "mallampati": "Mallampati ≥ III only",
    "lr": "Penalized logistic (L2)",
    "hgb": "Gradient boosting",
    "mlp": "Neural network (MLP 8)",
}
PRIMARY = LABELS["lr"]
POPULATION = "adults scheduled for thyroidectomy, expected difficult intubation excluded"
METHOD = f"stratified 5-fold CV × {N_REPEATS} (nested CV for the penalty)"


def limitations(events: int) -> list:
    return [
        f"{events} events: estimates still unstable; sample size below Riley's criteria.",
        "Thyroid surgery only, one centre; patients with an expected difficult intubation excluded.",
        "Mallampati, neck movement and history of difficult intubation available only as yes/no.",
        "Overall risk of bias high (PROBAST). Not externally validated. **Do not use for clinical decisions.**",
    ]


def unpenalized():
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("lr", LogisticRegression(C=1e6, max_iter=2000))])


def egri_rule(df):
    """M0: the 1996 rule as it is (positive if score >= 4), no fitting."""
    y, r = df["difficult"].to_numpy(), df["egri_ge4"].to_numpy()
    m = evaluate.threshold_metrics(y, r, t=0.5)
    return {"auc": float(roc_auc_score(y, r)), **{k: m[k] for k in ("sensitivity", "specificity", "ppv", "npv", "flagged")}}


def external_vitaldb(df):
    model = inference.load_model(run_vitaldb.MODEL_NAME)
    y = df["difficult"].to_numpy()
    p = np.array([inference.predict_features({"age": r.age, "sex_male": r.sex_male, "bmi": r.bmi,
                                              "asa": None, "emergency": 0.0}, model)[0] for r in df.itertuples()])
    slope, _ = evaluate.calibration_slope_intercept(y, p)
    return {"model": model["name"], "auc_ci": evaluate.bootstrap_auc_ci(y, p), "mean_predicted": float(p.mean()),
            "observed": float(y.mean()), "cal_slope": slope, "flagged": float((p >= evaluate.THRESHOLD).mean())}


PROBAST = [
    {"domain": "1. Participants", "risk": "high", "applicability": "high",
     "why": "Patients with an expected difficult intubation were excluded by protocol: the most obvious cases "
            "are missing, and the ability of the predictors to discriminate is distorted (spectrum bias).",
     "applicability_why": "Thyroid surgery only (77% women, a quarter with a large goitre), one centre."},
    {"domain": "2. Predictors", "risk": "low", "applicability": "low",
     "why": "Measured at the anaesthesia visit, before intubation, with standard definitions.",
     "applicability_why": "Almost all the El-Ganzouri items, the same as in the app's form."},
    {"domain": "3. Outcome", "risk": "unclear", "applicability": "low",
     "why": "Cormack with a conventional curved blade, standard definition; the intubating clinician probably "
            "knows the pre-operative assessment (not blinded).",
     "applicability_why": "Exactly the project's outcome (direct laryngoscopy, Cormack 3–4)."},
    {"domain": "4. Analysis", "risk": "high", "applicability": None,
     "why": "48 events for 7 predictors (about 7 per predictor), below Riley's sample size; Mallampati, neck and "
            "history arrive already cut into yes/no. In favour: pre-specified predictors, no missing values, "
            "penalization, calibration and optimism measured.",
     "applicability_why": None},
]
PROBAST_OVERALL = {"risk": "high", "applicability": "high"}


def model_card(model: dict) -> str:
    d, v = model["training_data"], model["validation"]
    m0 = v["egri_rule"]
    lines = [
        f"# Model card — {model['name']}", "",
        f"Created: {model['created']} · penalized logistic regression (L2, C={model['C']:.3g}) · protocol: SPEC §11", "",
        "## Data",
        f"- {d['source']}, sha256 `{d['sha256'][:12]}…`.",
        f"- {d['n']} adults scheduled for thyroidectomy; {d['events']} with Cormack 3–4 ({d['prevalence'] * 100:.1f}%), "
        "conventional curved blade.",
        "- Patients with an expected difficult intubation excluded by protocol.",
        f"- Predictors: {', '.join(inference.FEATURE_LABELS[f] for f in model['features'])}.", "",
        f"## Internal validation (stratified 5-fold CV × {N_REPEATS} repeats)",
        "AUC computed within each fold; mean and 2.5–97.5% range across repeats (variability, not a CI).", "",
        "| Model | AUC per fold | Brier | Calibr. slope | Sens. @15% | Spec. @15% | Flagged @15% |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in v["results"].items():
        m = r["metrics"]
        lines.append(f"| {name} | {fmt_ci(r['auc_fold'])} | {fmt_ci(m['brier'])} | {fmt_ci(m['cal_slope'])} | "
                     f"{fmt_ci(m['sensitivity'], True)} | {fmt_ci(m['specificity'], True)} | {fmt_ci(m['flagged'], True)} |")
    lines += [
        "", f"**El-Ganzouri ≥ 4 as a fixed rule (M0)**: AUC {m0['auc']:.3f}, sensitivity {m0['sensitivity'] * 100:.1f}%, "
        f"specificity {m0['specificity'] * 100:.1f}%, PPV {m0['ppv'] * 100:.1f}%, flagged {m0['flagged'] * 100:.1f}%.",
        *optimism_table(v),
        *coefficient_lines(model),
    ]
    t, e, n = v["thyroid_model"], v["external_vitaldb"], v["sample_size"]
    lines += [
        "", "## Secondary analyses",
        f"- **+ neck circumference** (thyroid-only model): AUC per fold {fmt_ci(t['auc_fold'])}, "
        f"bootstrap-corrected {t['optimism']['auc']['corrected']:.3f}.",
        f"- **VitalDB model on these patients**: AUC {e['auc_ci']['auc']:.3f} ({e['auc_ci']['lo']:.3f}–{e['auc_ci']['hi']:.3f}); "
        f"mean predicted risk {e['mean_predicted'] * 100:.1f}% vs {e['observed'] * 100:.1f}% observed.",
        f"- **Minimum sample size (Riley 2020)** with {len(model['features'])} predictors: {n['n_required']} patients "
        f"({n['events_required']} events); available {d['n']} ({d['events']}).",
        *probast_and_limitations(model),
        "Source: De Cassai A et al., PLoS ONE 2019;14:e0212976, S1 Database (CC-BY 4.0).",
    ]
    return "\n".join(lines) + "\n"


def write_teaching(X, y, chosen_C):
    """teaching.json for the Learn pages; also callable alone (python -m training.run_thyroid --teaching-only)."""
    labels = [inference.FEATURE_LABELS[f] for f in thyroid.FEATURES]
    material = teaching.material(X, y, labels, 0, "Mallampati ≥ III", thyroid.SOURCE, chosen_C=chosen_C)
    (MODELS_DIR / MODEL_NAME / "teaching.json").write_text(json.dumps(material), encoding="utf-8")


def teaching_only():
    df = thyroid.load()
    model = json.loads((MODELS_DIR / MODEL_NAME / "model.json").read_text(encoding="utf-8"))
    write_teaching(df[thyroid.FEATURES].to_numpy(dtype=float), df["difficult"].to_numpy(dtype=int), model.get("C"))
    print("Rewrote teaching.json")


def main():
    df = thyroid.load()
    y = df["difficult"].to_numpy(dtype=int)
    X = df[thyroid.FEATURES].to_numpy(dtype=float)
    print(f"Padova: {len(y)} patients, {y.sum()} Cormack 3-4 ({y.mean() * 100:.1f}%)")

    shared = run_vitaldb.make_models()
    plain = lambda: LogisticRegression(max_iter=1000)  # noqa: E731
    candidates = {  # key -> (model, columns)
        "null": (DummyClassifier(strategy="prior"), X),
        "egri": (plain(), df[["egri_ge4"]].to_numpy()),
        "mallampati": (plain(), df[["mallampati_ge3"]].to_numpy()),
        "lr": (shared["lr"], X), "hgb": (shared["hgb"], X), "mlp": (shared["mlp"], X),
    }
    results = {}
    for key, (m, Xk) in candidates.items():
        results[LABELS[key]] = r = cv_with_ci(m, Xk, y)
        a = r["auc_fold"]
        print(f"{LABELS[key]:32s} AUC {a['mean']:.3f} ({a['lo']:.3f}-{a['hi']:.3f})  "
              f"slope {r['metrics']['cal_slope']['mean']:.2f}  sens@15% {r['metrics']['sensitivity']['mean']:.2f}  "
              f"flagged@15% {r['metrics']['flagged']['mean']:.2f}")
    m0 = egri_rule(df)
    print("EGRI >= 4 rule:", {k: round(v, 3) for k, v in m0.items()})

    optimism = evaluate.optimism_corrected(shared["lr"], X, y)
    optimism_unpen = evaluate.optimism_corrected(unpenalized(), X, y)
    for title, o in (("penalized", optimism), ("unpenalized", optimism_unpen)):
        print(f"Optimism {title}:", {k: (round(v["apparent"], 3), round(v["corrected"], 3)) for k, v in o.items() if k != "n_boot"})

    Xt = df[thyroid.FEATURES + ["neck_circumference"]].to_numpy(dtype=float)
    neck_model = {"auc_fold": evaluate.fold_auc(shared["lr"], Xt, y, n_repeats=N_REPEATS),
               "optimism": evaluate.optimism_corrected(shared["lr"], Xt, y)}
    print("+ neck circumference:", round(neck_model["auc_fold"]["mean"], 3), round(neck_model["optimism"]["auc"]["corrected"], 3))
    external = external_vitaldb(df)
    print("VitalDB model:", {k: v for k, v in external.items() if k != "model"})
    sample_size = evaluate.riley_sample_size(float(y.mean()), len(thyroid.FEATURES))
    print("Riley:", sample_size["n_required"], sample_size["events_required"])

    final = clone(shared["lr"]).fit(X, y)
    info = {
        "training_data": {"source": thyroid.SOURCE, "sha256": thyroid.sha256(), "n": int(len(y)),
                          "events": int(y.sum()), "prevalence": float(y.mean()),
                          "population": POPULATION},
        "primary": PRIMARY,
        "validation": {"method": METHOD,
                       "results": results, "dca_reference": evaluate.dca_reference(y), "egri_rule": m0,
                       "optimism": optimism, "optimism_unpenalized": optimism_unpen,
                       "thyroid_model": neck_model, "external_vitaldb": external, "sample_size": sample_size},
        "coef_ci": coef_bootstrap(shared["lr"], X, y, thyroid.FEATURES),
        "probast": {"tool": "PROBAST (Wolff et al., Ann Intern Med 2019)", "domains": PROBAST, "overall": PROBAST_OVERALL},
        "limitations": limitations(int(y.sum())),
    }
    model = run_vitaldb.export_lr(final, thyroid.FEATURES, info, name=MODEL_NAME)

    out = MODELS_DIR / MODEL_NAME
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(json.dumps(model, indent=1), encoding="utf-8")
    (out / "model_card.md").write_text(model_card(model), encoding="utf-8")
    write_teaching(X, y, model.get("C"))
    # SPEC §11: the primary model becomes the app model if it discriminates (it does: AUC ~0.74)
    (MODELS_DIR / "ACTIVE").write_text(MODEL_NAME + "\n", encoding="utf-8")
    print(f"Exported {out} and set it as the active model")


if __name__ == "__main__":
    teaching_only() if "--teaching-only" in sys.argv else main()
