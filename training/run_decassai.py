"""
Pre-specified study on the Padova thyroid-surgery cohort (SPEC §11) and export of the primary model.

    venv\\Scripts\\python.exe -m training.run_decassai

Writes models/decassai-lr-v1/ (model.json, model_card.md, teaching.json) and models/ACTIVE.
"""
from __future__ import annotations

import json
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
from training import decassai, evaluate, run_vitaldb, teaching
from training.run_laryngoscope import N_REPEATS, coef_bootstrap, cv_with_ci, fmt_ci

MODEL_NAME = "decassai-lr-v1"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
LABELS = {
    "null": "Nessun modello (prevalenza)",
    "egri": "El-Ganzouri ≥ 4 (ricalibrato)",
    "mallampati": "Solo Mallampati ≥ III",
    "lr": "Logistica penalizzata (L2)",
    "hgb": "Gradient boosting",
    "mlp": "Rete neurale (MLP 8)",
}
PRIMARY = LABELS["lr"]


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
    {"domain": "1. Partecipanti", "risk": "alto", "applicability": "alta",
     "why": "Esclusi per protocollo i pazienti con intubazione difficile prevista: mancano proprio i casi più "
            "evidenti, e la capacità dei predittori di distinguere è distorta (bias di spettro).",
     "applicability_why": "Solo chirurgia tiroidea (77% donne, un quarto con gozzo voluminoso), un centro."},
    {"domain": "2. Predittori", "risk": "basso", "applicability": "bassa",
     "why": "Misurati alla visita anestesiologica, prima dell'intubazione, con definizioni standard.",
     "applicability_why": "Sono quasi tutti gli item di El-Ganzouri, gli stessi del form dell'app."},
    {"domain": "3. Esito", "risk": "incerto", "applicability": "bassa",
     "why": "Cormack con lama curva convenzionale, definizione standard; chi intuba probabilmente conosce la "
            "valutazione pre-operatoria (non in cieco).",
     "applicability_why": "È esattamente l'esito del progetto (laringoscopia diretta, Cormack 3–4)."},
    {"domain": "4. Analisi", "risk": "alto", "applicability": None,
     "why": "48 eventi per 7 predittori (circa 7 per predittore), sotto la numerosità di Riley; Mallampati, collo e "
            "storia arrivano già tagliati in sì/no. A favore: predittori pre-specificati, nessun mancante, "
            "penalizzazione, calibrazione e ottimismo misurati.",
     "applicability_why": None},
]
PROBAST_OVERALL = {"risk": "alto", "applicability": "alta"}


def model_card(model: dict) -> str:
    d, v = model["training_data"], model["validation"]
    m0 = v["egri_rule"]
    lines = [
        f"# Model card — {model['name']}", "",
        f"Creato: {model['created']} · regressione logistica penalizzata (L2, C={model['C']:.3g}) · protocollo: SPEC §11", "",
        "## Dati",
        f"- {d['source']}, sha256 `{d['sha256'][:12]}…`.",
        f"- {d['n']} adulti per tiroidectomia; {d['events']} con Cormack 3–4 ({d['prevalence'] * 100:.1f}%), lama curva convenzionale.",
        "- Esclusi per protocollo i pazienti con intubazione difficile prevista.",
        f"- Predittori: {', '.join(inference.FEATURE_LABELS[f] for f in model['features'])}.", "",
        f"## Validazione interna (CV stratificata 5 fold × {N_REPEATS} ripetizioni)",
        "AUC calcolata dentro ogni fold; media e intervallo 2,5–97,5% tra le ripetizioni (variabilità, non IC).", "",
        "| Modello | AUC per fold | Brier | Pendenza calibr. | Sens. @15% | Spec. @15% | Segnalati @15% |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in v["results"].items():
        m = r["metrics"]
        lines.append(f"| {name} | {fmt_ci(r['auc_fold'])} | {fmt_ci(m['brier'])} | {fmt_ci(m['cal_slope'])} | "
                     f"{fmt_ci(m['sensitivity'], True)} | {fmt_ci(m['specificity'], True)} | {fmt_ci(m['flagged'], True)} |")
    lines += [
        "", f"**El-Ganzouri ≥ 4 come regola fissa (M0)**: AUC {m0['auc']:.3f}, sensibilità {m0['sensitivity'] * 100:.1f}%, "
        f"specificità {m0['specificity'] * 100:.1f}%, VPP {m0['ppv'] * 100:.1f}%, segnalati {m0['flagged'] * 100:.1f}%.",
        "", "## Ottimismo (bootstrap di Harrell)", "",
        "| Modello | Metrica | Apparente | Ottimismo | Corretto |", "|---|---|---|---|---|",
    ]
    for title, o in (("Logistica penalizzata (principale)", v["optimism"]),
                     ("Logistica senza penalizzazione", v["optimism_unpenalized"])):
        lines += [f"| {title} | {k} | {o[k]['apparent']:.3f} | {o[k]['optimism']:.3f} | {o[k]['corrected']:.3f} |"
                  for k in ("auc", "brier", "cal_slope")]
    lines += ["", "## Coefficienti (odds ratio per 1 deviazione standard, IC 95% bootstrap)", "",
              *[f"- {inference.FEATURE_LABELS[f]}: OR {np.exp(model['coef'][f]):.2f} "
                f"({np.exp(model['coef_ci'][f]['lo']):.2f}–{np.exp(model['coef_ci'][f]['hi']):.2f})"
                for f in model["features"]]]
    t, e, n = v["thyroid_model"], v["external_vitaldb"], v["sample_size"]
    lines += [
        "", "## Analisi secondarie",
        f"- **+ circonferenza del collo** (modello solo per la tiroide): AUC per fold {fmt_ci(t['auc_fold'])}, "
        f"corretta con bootstrap {t['optimism']['auc']['corrected']:.3f}.",
        f"- **Modello VitalDB su questi pazienti**: AUC {e['auc_ci']['auc']:.3f} ({e['auc_ci']['lo']:.3f}–{e['auc_ci']['hi']:.3f}); "
        f"rischio medio previsto {e['mean_predicted'] * 100:.1f}% contro {e['observed'] * 100:.1f}% osservato.",
        f"- **Numerosità minima (Riley 2020)** con {len(model['features'])} predittori: {n['n_required']} pazienti "
        f"({n['events_required']} eventi); disponibili {d['n']} ({d['events']}).",
        "", "## Rischio di bias (PROBAST)", "",
        "| Dominio | Rischio di bias | Perché | Applicabilità (preoccupazione) | Perché |", "|---|---|---|---|---|",
        *[f"| {p['domain']} | {p['risk']} | {p['why']} | {p['applicability'] or '—'} | {p['applicability_why'] or '—'} |"
          for p in model["probast"]["domains"]],
        f"| **Complessivo** | **{model['probast']['overall']['risk']}** | | **{model['probast']['overall']['applicability']}** | |",
        "", "## Limiti", *[f"- {x}" for x in model["limitations"]], "",
        "Fonte: De Cassai A et al., PLoS ONE 2019;14:e0212976, S1 Database (CC-BY 4.0).",
    ]
    return "\n".join(lines) + "\n"


def main():
    df = decassai.load()
    y = df["difficult"].to_numpy(dtype=int)
    X = df[decassai.FEATURES].to_numpy(dtype=float)
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

    Xt = df[decassai.FEATURES + ["neck_circumference"]].to_numpy(dtype=float)
    thyroid = {"auc_fold": evaluate.fold_auc(shared["lr"], Xt, y, n_repeats=N_REPEATS),
               "optimism": evaluate.optimism_corrected(shared["lr"], Xt, y)}
    print("+ neck circumference:", round(thyroid["auc_fold"]["mean"], 3), round(thyroid["optimism"]["auc"]["corrected"], 3))
    external = external_vitaldb(df)
    print("VitalDB model:", {k: v for k, v in external.items() if k != "model"})
    sample_size = evaluate.riley_sample_size(float(y.mean()), len(decassai.FEATURES))
    print("Riley:", sample_size["n_required"], sample_size["events_required"])

    final = clone(shared["lr"]).fit(X, y)
    info = {
        "training_data": {"source": decassai.SOURCE, "sha256": decassai.sha256(), "n": int(len(y)),
                          "events": int(y.sum()), "prevalence": float(y.mean()),
                          "population": "adulti per tiroidectomia, esclusa l'intubazione difficile prevista"},
        "primary": PRIMARY,
        "validation": {"method": f"CV stratificata 5 fold × {N_REPEATS} (CV annidata per la penalizzazione)",
                       "results": results, "dca_reference": evaluate.dca_reference(y), "egri_rule": m0,
                       "optimism": optimism, "optimism_unpenalized": optimism_unpen,
                       "thyroid_model": thyroid, "external_vitaldb": external, "sample_size": sample_size},
        "coef_ci": coef_bootstrap(shared["lr"], X, y, decassai.FEATURES),
        "probast": {"tool": "PROBAST (Wolff et al., Ann Intern Med 2019)", "domains": PROBAST, "overall": PROBAST_OVERALL},
        "limitations": [
            f"{int(y.sum())} eventi: stime ancora instabili; numerosità sotto i criteri di Riley.",
            "Solo chirurgia tiroidea, un centro; esclusi i pazienti con intubazione difficile prevista.",
            "Mallampati, mobilità del collo e storia di intubazione difficile disponibili solo come sì/no.",
            "Rischio di bias complessivo alto (PROBAST). Non validato esternamente. **Non usare per decisioni cliniche.**",
        ],
    }
    model = run_vitaldb.export_lr(final, decassai.FEATURES, info, name=MODEL_NAME)

    out = MODELS_DIR / MODEL_NAME
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(json.dumps(model, indent=1), encoding="utf-8")
    (out / "model_card.md").write_text(model_card(model), encoding="utf-8")
    labels = [inference.FEATURE_LABELS[f] for f in decassai.FEATURES]
    (out / "teaching.json").write_text(json.dumps(teaching.material(X, y, labels, 0, "Mallampati ≥ III", decassai.SOURCE)),
                                       encoding="utf-8")
    # SPEC §11: the primary model becomes the app model if it discriminates (it does: AUC ~0.74)
    (MODELS_DIR / "ACTIVE").write_text(MODEL_NAME + "\n", encoding="utf-8")
    print(f"Exported {out} and set it as the active model")


if __name__ == "__main__":
    main()
