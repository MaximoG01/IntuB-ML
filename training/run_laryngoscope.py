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
    "null": "Nessun modello (prevalenza)",
    "mallampati": "Solo Mallampati",
    "lr": "Logistica penalizzata (L2)",
    "hgb": "Gradient boosting",
    "mlp": "Rete neurale (MLP 8)",
}
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
    {"domain": "1. Partecipanti", "risk": "incerto", "applicability": "alta",
     "why": "Dati di un RCT (fonte accettabile), ma criteri di esclusione da verificare sull'articolo "
            "originale: se escludeva vie aeree difficili note, la prevalenza è distorta.",
     "applicability_why": "Solo obesi (BMI 30–50), chirurgia elettiva, un centro: non rappresenta la "
                          "popolazione generale della domanda (§1)."},
    {"domain": "2. Predittori", "risk": "basso", "applicability": "alta",
     "why": "Predittori misurati prima della laringoscopia, definizioni standard, scelti prima dell'analisi.",
     "applicability_why": "Manca quasi tutto l'El-Ganzouri (DTM, apertura bocca, collo, prognatismo, "
                          "storia): c'è solo il Mallampati."},
    {"domain": "3. Esito", "risk": "alto", "applicability": "alta",
     "why": "Cormack dato da chi intuba, che conosce il Mallampati (non in cieco); metà dei pazienti "
            "valutata con videolaringoscopio, che migliora la visione: l'esito non è uguale per tutti.",
     "applicability_why": "La domanda è la laringoscopia diretta; qui l'esito è solo binario (1–2 vs 3–4)."},
    {"domain": "4. Analisi", "risk": "alto", "applicability": None,
     "why": "18 eventi per 5 predittori (3,6 eventi per predittore), molto sotto la numerosità minima di "
            "Riley. A favore: predittori pre-specificati, niente selezione univariata, mancanti imputati "
            "dentro i fold, penalizzazione, calibrazione e ottimismo misurati.",
     "applicability_why": None},
]
PROBAST_OVERALL = {"risk": "alto", "applicability": "alta"}


def fmt_ci(d, pct=False):
    s = (lambda v: f"{v * 100:.1f}%") if pct else (lambda v: f"{v:.3f}")
    return f"{s(d['mean'])} ({s(d['lo'])}–{s(d['hi'])})"


def model_card(model: dict) -> str:
    d, v = model["training_data"], model["validation"]
    lines = [
        f"# Model card — {model['name']}", "",
        f"Creato: {model['created']} · tipo: regressione logistica penalizzata (L2, C={model['C']:.3g}) · "
        "protocollo: SPEC §10", "",
        "## Dati",
        f"- {d['source']}, commit `{d['commit'][:7]}`, sha256 `{d['sha256'][:12]}…`.",
        f"- {d['n']} adulti obesi, chirurgia elettiva; {d['events']} con Cormack 3–4 ({d['prevalence'] * 100:.1f}%).",
        "- Esito: `view == 0` (la documentazione del pacchetto ha la codifica invertita; vedi SPEC §10).",
        f"- Predittori: {', '.join(model['features'])}. Nell'app il videolaringoscopio è fissato a 0 "
        "(predizione per lama Macintosh).", "",
        f"## Validazione interna (CV stratificata 5 fold × {N_REPEATS} ripetizioni)",
        "Media e intervallo 2,5–97,5% tra le ripetizioni (variabilità della CV, **non** un IC). "
        "AUC calcolata dentro ogni fold: sulle predizioni out-of-fold riunite, con pochi eventi, "
        "i modelli deboli scendono artificialmente sotto 0,5 (il modello nullo dava 0,31). Per lo stesso "
        "motivo la pendenza di calibrazione della CV è falsata per i modelli quasi costanti: per la "
        "calibrazione vale il bootstrap qui sotto.", "",
        "| Modello | AUC per fold | Brier | Pendenza calibr. | Sens. @15% | Spec. @15% | Segnalati @15% |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in v["results"].items():
        m = r["metrics"]
        lines.append(f"| {name} | {fmt_ci(r['auc_fold'])} | {fmt_ci(m['brier'])} | "
                     f"{fmt_ci(m['cal_slope'])} | {fmt_ci(m['sensitivity'], True)} | "
                     f"{fmt_ci(m['specificity'], True)} | {fmt_ci(m['flagged'], True)} |")
    lines += ["", "## Ottimismo (bootstrap di Harrell)", "",
              "| Modello | Metrica | Apparente | Ottimismo | Corretto |", "|---|---|---|---|---|"]
    for title, o in (("Logistica penalizzata (principale)", v["optimism"]),
                     ("Logistica senza penalizzazione", v["optimism_unpenalized"])):
        lines += [f"| {title} | {k} | {o[k]['apparent']:.3f} | {o[k]['optimism']:.3f} | {o[k]['corrected']:.3f} |"
                  for k in ("auc", "brier", "cal_slope")]
    lines += [
        "",
        "La pendenza corretta del modello penalizzato non è interpretabile: le sue predizioni sono quasi "
        "costanti, e una piccola differenza tra pazienti produce una pendenza grande. Il modello senza "
        "penalizzazione mostra invece l'overfitting tipico: pendenza corretta molto sotto 1.",
        "", "## Coefficienti (odds ratio per 1 deviazione standard, IC 95% bootstrap)", "",
        *[f"- {inference.FEATURE_LABELS[f]}: OR {np.exp(model['coef'][f]):.2f} "
          f"({np.exp(model['coef_ci'][f]['lo']):.2f}–{np.exp(model['coef_ci'][f]['hi']):.2f})"
          for f in model["features"]],
    ]
    s, e, n = v["sensitivity_macintosh"], v["external_vitaldb"], v["sample_size"]
    lines += [
        "", "## Analisi secondarie",
        f"- **Sola lama Macintosh** (n = {s['n']}, {s['events']} eventi): AUC per fold {fmt_ci(s['auc_fold'])}.",
        f"- **Modello VitalDB su questi pazienti**: AUC {e['auc_ci']['auc']:.3f} "
        f"({e['auc_ci']['lo']:.3f}–{e['auc_ci']['hi']:.3f}); rischio medio previsto "
        f"{e['mean_predicted'] * 100:.1f}% contro {e['observed'] * 100:.1f}% osservato; segnalati @15% "
        f"{e['flagged'] * 100:.1f}%.",
        f"- **Numerosità minima (Riley 2020)** con {len(model['features'])} predittori e prevalenza "
        f"{d['prevalence'] * 100:.0f}%: {n['n_required']} pazienti ({n['events_required']} eventi); "
        f"disponibili {d['n']} ({d['events']}).",
        "", "## Rischio di bias (PROBAST)", "",
        "| Dominio | Rischio di bias | Perché | Applicabilità (preoccupazione) | Perché |",
        "|---|---|---|---|---|",
        *[f"| {p['domain']} | {p['risk']} | {p['why']} | {p['applicability'] or '—'} | "
          f"{p['applicability_why'] or '—'} |" for p in model["probast"]["domains"]],
        f"| **Complessivo** | **{model['probast']['overall']['risk']}** | | "
        f"**{model['probast']['overall']['applicability']}** | |",
        "", "## Limiti", *[f"- {x}" for x in model["limitations"]], "",
        "Fonte: Abdallah et al., Anesth Analg 2011;113:1082-7; Nowacki AS, TSHS Resources Portal 2017; "
        "pacchetto R medicaldata (MIT).",
    ]
    return "\n".join(lines) + "\n"


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
                          "prevalence": float(y.mean()), "population": "adulti obesi (BMI 30–50), chirurgia elettiva"},
        "fixed": {"video_laryngoscope": 0.0},
        "primary": PRIMARY,
        "validation": {"method": f"CV stratificata 5 fold × {N_REPEATS} (CV annidata per la penalizzazione)",
                       "results": results, "dca_reference": evaluate.dca_reference(y),
                       "optimism": optimism, "optimism_unpenalized": optimism_unpen,
                       "sensitivity_macintosh": sens,
                       "external_vitaldb": external, "sample_size": sample_size},
        "coef_ci": coef_bootstrap(models["lr"], X, y, laryngoscope.FEATURES),
        "probast": {"tool": "PROBAST (Wolff et al., Ann Intern Med 2019)",
                    "domains": PROBAST, "overall": PROBAST_OVERALL},
        "limitations": [
            f"Solo {int(y.sum())} eventi: stime instabili, intervalli di confidenza larghi.",
            "Solo obesi in chirurgia elettiva, un centro: non trasferibile ad altre popolazioni senza validazione.",
            "Unico esame delle vie aeree disponibile: Mallampati. El-Ganzouri non calcolabile.",
            "Metà dei Cormack con videolaringoscopio: il modello lo usa come predittore; nell'app è fissato a lama Macintosh.",
            "Rischio di bias complessivo alto (PROBAST). Non validato esternamente. **Non usare per decisioni cliniche.**",
        ],
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
