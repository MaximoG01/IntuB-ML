"""
Pre-specified study on the laryngoscope trial (SPEC §10) and export of the primary model.

    venv\\Scripts\\python.exe -m training.run_laryngoscope

Writes models/laryngoscope-lr-v1/ (model.json, model_card.md, teaching.json); models/ACTIVE is left as is.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from app import inference
from training import evaluate, laryngoscope, run_vitaldb, teaching

MODEL_NAME = "laryngoscope-lr-v1"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
N_REPEATS = 20
LABELS = {
    "null": "No model (prevalence)",
    "mallampati": "Mallampati only",
    "lr": "Penalized logistic (L2)",
    "hgb": "Gradient boosting",
    "mlp": "Neural network (MLP 8)",
}
POPULATION = "obese adults (BMI 30–50), elective surgery"
METHOD = f"stratified 5-fold CV × {N_REPEATS} (nested CV for the penalty)"


def limitations(events: int) -> list:
    return [
        f"Only {events} events: unstable estimates, wide confidence intervals.",
        "Only obese patients in elective surgery, one centre: not transferable to other populations without validation.",
        "The only airway examination available is Mallampati. El-Ganzouri cannot be computed.",
        "Half of the Cormack grades were obtained with a video laryngoscope: the model uses the device as a "
        "predictor; in the app it is fixed to a Macintosh blade.",
        "Overall risk of bias high (PROBAST). Not externally validated. **Do not use for clinical decisions.**",
    ]
PRIMARY = LABELS["lr"]


def _first_column(X):
    return X[:, :1]


def make_models(seed: int = 42):
    """Same algorithms as the VitalDB study, plus Mallampati alone as clinical reference."""
    shared = run_vitaldb.make_models(seed)
    return {
        "null": DummyClassifier(strategy="prior"),
        "mallampati": Pipeline([  # FEATURES[0] is Mallampati
            ("select", FunctionTransformer(_first_column)),
            ("impute", SimpleImputer(strategy="median")),
            ("lr", LogisticRegression(max_iter=1000)),
        ]),
        "lr": shared["lr"],
        "hgb": shared["hgb"],
        "mlp": shared["mlp"],
    }


def cv_with_ci(model, X, y):
    P = evaluate.oof_predictions(model, X, y, n_repeats=N_REPEATS)
    out = evaluate.summarize(y, P)
    out["auc_fold"] = evaluate.fold_auc(model, X, y, n_repeats=N_REPEATS)
    return out


def coef_bootstrap(model, X, y, features, n_boot=500, seed=42):
    """Percentile 95% CI of the standardized coefficients (odds ratio per 1 SD)."""
    rng = np.random.default_rng(seed)
    coefs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if 5 <= y[i].sum() <= len(i) - 5:
            coefs.append(clone(model).fit(X[i], y[i]).named_steps["lr"].coef_[0])
    coefs = np.array(coefs)
    return {f: {"lo": float(np.percentile(coefs[:, j], 2.5)), "hi": float(np.percentile(coefs[:, j], 97.5))}
            for j, f in enumerate(features)}


def external_vitaldb(df):
    """Transportability: the VitalDB model on these (all elective) patients."""
    model = inference.load_model(run_vitaldb.MODEL_NAME)
    y = df["difficult"].to_numpy()
    p = np.array([inference.predict_features(
        {"age": r.age, "sex_male": r.sex_male, "bmi": None if np.isnan(r.bmi) else r.bmi,
         "asa": r.asa, "emergency": 0.0}, model)[0] for r in df.itertuples()])
    slope, _ = evaluate.calibration_slope_intercept(y, p)
    return {"model": model["name"], "auc_ci": evaluate.bootstrap_auc_ci(y, p),
            "mean_predicted": float(p.mean()), "observed": float(y.mean()), "cal_slope": slope,
            "flagged": float((p >= evaluate.THRESHOLD).mean())}


PROBAST = [
    {"domain": "1. Participants", "risk": "unclear", "applicability": "high",
     "why": "RCT data (an acceptable source), but the exclusion criteria must be checked in the original "
            "paper: if known difficult airways were excluded, the prevalence is distorted.",
     "applicability_why": "Only obese patients (BMI 30–50), elective surgery, one centre: not the general "
                          "population of our question (§1)."},
    {"domain": "2. Predictors", "risk": "low", "applicability": "high",
     "why": "Predictors measured before laryngoscopy, standard definitions, chosen before the analysis.",
     "applicability_why": "Almost all of El-Ganzouri is missing (TMD, mouth opening, neck, jaw protrusion, "
                          "history): only Mallampati is available."},
    {"domain": "3. Outcome", "risk": "high", "applicability": "high",
     "why": "Cormack graded by the intubating clinician, who knows the Mallampati (not blinded); half of "
            "the patients assessed with a video laryngoscope, which improves the view: the outcome is not "
            "the same for everyone.",
     "applicability_why": "Our question is direct laryngoscopy; here the outcome is only binary (1–2 vs 3–4)."},
    {"domain": "4. Analysis", "risk": "high", "applicability": None,
     "why": "18 events for 5 predictors (3.6 events per predictor), far below Riley's minimum sample size. "
            "In favour: pre-specified predictors, no univariable selection, missing values imputed within "
            "the folds, penalization, calibration and optimism measured.",
     "applicability_why": None},
]
PROBAST_OVERALL = {"risk": "high", "applicability": "high"}


def fmt_ci(d, pct=False):
    s = (lambda v: f"{v * 100:.1f}%") if pct else (lambda v: f"{v:.3f}")
    return f"{s(d['mean'])} ({s(d['lo'])}–{s(d['hi'])})"


def model_card(model: dict) -> str:
    d, v = model["training_data"], model["validation"]
    lines = [
        f"# Model card — {model['name']}", "",
        f"Created: {model['created']} · type: penalized logistic regression (L2, C={model['C']:.3g}) · "
        "protocol: SPEC §10", "",
        "## Data",
        f"- {d['source']}, commit `{d['commit'][:7]}`, sha256 `{d['sha256'][:12]}…`.",
        f"- {d['n']} obese adults, elective surgery; {d['events']} with Cormack 3–4 ({d['prevalence'] * 100:.1f}%).",
        "- Outcome: `view == 0` (the package documentation has the coding inverted; see SPEC §10).",
        f"- Predictors: {', '.join(model['features'])}. In the app the video laryngoscope is fixed to 0 "
        "(prediction for a Macintosh blade).", "",
        f"## Internal validation (stratified 5-fold CV × {N_REPEATS} repeats)",
        "Mean and 2.5–97.5% range across repeats (CV variability, **not** a CI). "
        "AUC computed within each fold: on pooled out-of-fold predictions, with few events, weak models "
        "fall artificially below 0.5 (the null model gave 0.31). For the same reason the CV calibration "
        "slope is distorted for near-constant models: for calibration use the bootstrap below.", "",
        "| Model | AUC per fold | Brier | Calibr. slope | Sens. @15% | Spec. @15% | Flagged @15% |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in v["results"].items():
        m = r["metrics"]
        lines.append(f"| {name} | {fmt_ci(r['auc_fold'])} | {fmt_ci(m['brier'])} | "
                     f"{fmt_ci(m['cal_slope'])} | {fmt_ci(m['sensitivity'], True)} | "
                     f"{fmt_ci(m['specificity'], True)} | {fmt_ci(m['flagged'], True)} |")
    lines += optimism_table(v)
    lines += [
        "",
        "The corrected slope of the penalized model cannot be interpreted: its predictions are almost "
        "constant, and a small difference between patients produces a large slope. The unpenalized model "
        "shows typical overfitting instead: corrected slope well below 1.",
        *coefficient_lines(model),
    ]
    s, e, n = v["sensitivity_macintosh"], v["external_vitaldb"], v["sample_size"]
    lines += [
        "", "## Secondary analyses",
        f"- **Macintosh blade only** (n = {s['n']}, {s['events']} events): AUC per fold {fmt_ci(s['auc_fold'])}.",
        f"- **VitalDB model on these patients**: AUC {e['auc_ci']['auc']:.3f} "
        f"({e['auc_ci']['lo']:.3f}–{e['auc_ci']['hi']:.3f}); mean predicted risk "
        f"{e['mean_predicted'] * 100:.1f}% vs {e['observed'] * 100:.1f}% observed; flagged @15% "
        f"{e['flagged'] * 100:.1f}%.",
        f"- **Minimum sample size (Riley 2020)** with {len(model['features'])} predictors and prevalence "
        f"{d['prevalence'] * 100:.0f}%: {n['n_required']} patients ({n['events_required']} events); "
        f"available {d['n']} ({d['events']}).",
        *probast_and_limitations(model),
        "Source: Abdallah et al., Anesth Analg 2011;113:1082-7; Nowacki AS, TSHS Resources Portal 2017; "
        "R package medicaldata (MIT).",
    ]
    return "\n".join(lines) + "\n"


def optimism_table(v: dict) -> list:
    lines = ["", "## Optimism (Harrell's bootstrap)", "",
             "| Model | Metric | Apparent | Optimism | Corrected |", "|---|---|---|---|---|"]
    for title, o in (("Penalized logistic (primary)", v["optimism"]),
                     ("Unpenalized logistic", v["optimism_unpenalized"])):
        lines += [f"| {title} | {k} | {o[k]['apparent']:.3f} | {o[k]['optimism']:.3f} | {o[k]['corrected']:.3f} |"
                  for k in ("auc", "brier", "cal_slope")]
    return lines


def coefficient_lines(model: dict) -> list:
    return ["", "## Coefficients (odds ratio per 1 standard deviation, bootstrap 95% CI)", "",
            *[f"- {inference.FEATURE_LABELS[f]}: OR {np.exp(model['coef'][f]):.2f} "
              f"({np.exp(model['coef_ci'][f]['lo']):.2f}–{np.exp(model['coef_ci'][f]['hi']):.2f})"
              for f in model["features"]]]


def probast_and_limitations(model: dict) -> list:
    pb = model["probast"]
    return ["", "## Risk of bias (PROBAST)", "",
            "| Domain | Risk of bias | Why | Applicability (concern) | Why |", "|---|---|---|---|---|",
            *[f"| {p['domain']} | {p['risk']} | {p['why']} | {p['applicability'] or '—'} | "
              f"{p['applicability_why'] or '—'} |" for p in pb["domains"]],
            f"| **Overall** | **{pb['overall']['risk']}** | | **{pb['overall']['applicability']}** | |",
            "", "## Limitations", *[f"- {x}" for x in model["limitations"]], ""]


def main():
    df = laryngoscope.load()
    X = df[laryngoscope.FEATURES].to_numpy(dtype=float)
    y = df["difficult"].to_numpy(dtype=int)
    print(f"Laryngoscope: {len(y)} patients, {y.sum()} Cormack 3-4 ({y.mean() * 100:.1f}%)")

    models = make_models()
    results = {}
    for key, m in models.items():
        results[LABELS[key]] = r = cv_with_ci(m, X, y)
        a = r["auc_fold"]
        print(f"{LABELS[key]:30s} AUC {a['mean']:.3f} ({a['lo']:.3f}-{a['hi']:.3f})  "
              f"slope {r['metrics']['cal_slope']['mean']:.2f}  flagged@15% {r['metrics']['flagged']['mean']:.2f}")

    optimism = evaluate.optimism_corrected(models["lr"], X, y)
    unpenalized = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                            ("lr", LogisticRegression(C=1e6, max_iter=2000))])
    optimism_unpen = evaluate.optimism_corrected(unpenalized, X, y)
    for title, o in (("penalized", optimism), ("unpenalized", optimism_unpen)):
        print(f"Optimism {title}:", {k: (round(v["apparent"], 3), round(v["corrected"], 3))
                                     for k, v in o.items() if k != "n_boot"})

    mac = df["video_laryngoscope"] == 0
    Xm, ym = X[mac.to_numpy()][:, :4], y[mac.to_numpy()]  # device column dropped: constant here
    sens = {"n": int(len(ym)), "events": int(ym.sum()),
            "auc_fold": evaluate.fold_auc(models["lr"], Xm, ym, n_repeats=N_REPEATS)}
    print("Macintosh only:", sens)
    external = external_vitaldb(df)
    print("VitalDB model:", external)
    sample_size = evaluate.riley_sample_size(float(y.mean()), len(laryngoscope.FEATURES))
    print("Riley:", sample_size)

    final = clone(models["lr"]).fit(X, y)
    info = {
        "training_data": {"source": laryngoscope.SOURCE, "commit": laryngoscope.COMMIT,
                          "sha256": laryngoscope.sha256(), "n": int(len(y)), "events": int(y.sum()),
                          "prevalence": float(y.mean()), "population": POPULATION},
        "fixed": {"video_laryngoscope": 0.0},
        "primary": PRIMARY,
        "validation": {"method": METHOD,
                       "results": results, "dca_reference": evaluate.dca_reference(y),
                       "optimism": optimism, "optimism_unpenalized": optimism_unpen,
                       "sensitivity_macintosh": sens,
                       "external_vitaldb": external, "sample_size": sample_size},
        "coef_ci": coef_bootstrap(models["lr"], X, y, laryngoscope.FEATURES),
        "probast": {"tool": "PROBAST (Wolff et al., Ann Intern Med 2019)",
                    "domains": PROBAST, "overall": PROBAST_OVERALL},
        "limitations": limitations(int(y.sum())),
    }
    model = run_vitaldb.export_lr(final, laryngoscope.FEATURES, info, name=MODEL_NAME)

    out = MODELS_DIR / MODEL_NAME
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(json.dumps(model, indent=1), encoding="utf-8")
    (out / "model_card.md").write_text(model_card(model), encoding="utf-8")
    (out / "teaching.json").write_text(json.dumps(teaching.material(X, y, [inference.FEATURE_LABELS[f] for f in laryngoscope.FEATURES], 0, "Mallampati", laryngoscope.SOURCE)), encoding="utf-8")
    print(f"Exported {out}")


if __name__ == "__main__":
    main()
