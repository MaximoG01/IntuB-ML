import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from app.models import IntubationRecord



DIFFICULT_THRESHOLD = 0.20   # 20% risk => consider as "difficult"

# elenco delle feature in ordine fisso
FEATURES = [
    "age",
    "weight",
    "height",
    "sex_male",
    "dtm",
    "dii",
    "mallampati",
    "stop_bang",
    "alganzouri",
]

# valori "ragionevoli" da usare se manca qualcosa
DEFAULTS = {
    "height": 170.0,
    "dtm": 6.0,
    "dii": 4.0,
    "mallampati": 2,
    "stop_bang": 3,
    "alganzouri": 3,
}


def build_feature_vector(record):
    """
    Converte un record (dal DB o da un 'dummy' con stessi attributi)
    in un vettore numerico pronto per scikit-learn.
    """
    sex_male = 1 if getattr(record, "sex", None) == "M" else 0

    height = getattr(record, "height", None) or DEFAULTS["height"]
    dtm = getattr(record, "dtm", None) or DEFAULTS["dtm"]
    dii = getattr(record, "dii", None) or DEFAULTS["dii"]
    mallampati = getattr(record, "mallampati", None) or DEFAULTS["mallampati"]
    stop_bang = getattr(record, "stop_bang", None) or DEFAULTS["stop_bang"]
    alganzouri = getattr(record, "alganzouri", None) or DEFAULTS["alganzouri"]

    return [
        record.age,
        record.weight,
        height,
        sex_male,
        dtm,
        dii,
        mallampati,
        stop_bang,
        alganzouri,
    ]


def train_logistic_model(min_samples: int = 50):
    """
    Allena una regressione logistica sui record con esito noto (difficult_binary != None).
    Ritorna (model, metrics_dict).

    Lancia ValueError se i casi completi sono troppo pochi.
    """
    q = IntubationRecord.query.filter(IntubationRecord.difficult_binary.isnot(None))
    records = q.all()

    if len(records) < min_samples:
        raise ValueError(f"Not enough complete cases for training (found {len(records)}, need {min_samples}).")

    X = np.array([build_feature_vector(r) for r in records], dtype=float)
    y = np.array([1 if r.difficult_binary else 0 for r in records], dtype=int)

    # modello semplice, ma con class_weight='balanced' per classi sbilanciate
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, y)

    # metriche base (training interno, per ora)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    acc = float(accuracy_score(y, y_pred))
    try:
        auc = float(roc_auc_score(y, y_proba))
    except ValueError:
        auc = None

    metrics = {
        "n_samples": len(records),
        "accuracy": acc,
        "auc": auc,
    }

    return model, metrics

def evaluate_logistic(min_samples: int = 50):
    """
    Train logistic model and return detailed metrics + ROC curve.
    Used by /analytics.
    """
    model, base_metrics = train_logistic_model(min_samples=min_samples)

    q = IntubationRecord.query.filter(IntubationRecord.difficult_binary.isnot(None))
    records = q.all()
    X = np.array([build_feature_vector(r) for r in records], dtype=float)
    y = np.array([1 if r.difficult_binary else 0 for r in records], dtype=int)

    proba = model.predict_proba(X)[:, 1]
    y_pred = (proba >= DIFFICULT_THRESHOLD).astype(int)

    fpr, tpr, _ = roc_curve(y, proba)
    cm = confusion_matrix(y, y_pred)

    metrics = dict(base_metrics)  # n_samples, accuracy, auc
    metrics.update(
        {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "threshold": DIFFICULT_THRESHOLD,
            "confusion": {
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            },
        }
    )
    return metrics
