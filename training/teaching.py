"""
Real-data material for the 'Impara' pages (teaching.json). Computed offline; the app only draws it.
"""
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def gradient_descent(x: np.ndarray, y: np.ndarray, start=(1.5, -2.5), steps=60, lr=1.0) -> dict:
    """Logistic with 2 parameters (intercept, one standardized predictor): exact 2D loss surface."""
    x = np.where(np.isnan(x), np.nanmedian(x), x)
    xs = (x - x.mean()) / x.std()

    def loss(b0, b1):
        p = np.clip(1 / (1 + np.exp(-(b0 + b1 * xs))), 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    b0s = np.linspace(-4, 2, 49); b1s = np.linspace(-3, 3, 49)
    grid = [[round(loss(a, b), 4) for a in b0s] for b in b1s]
    path, b = [], np.array(start, dtype=float)
    for _ in range(steps):
        path.append({"b0": round(float(b[0]), 4), "b1": round(float(b[1]), 4), "loss": round(loss(*b), 4)})
        p = 1 / (1 + np.exp(-(b[0] + b[1] * xs)))
        b = b - lr * np.array([np.mean(p - y), np.mean((p - y) * xs)])
    return {"b0": b0s.round(3).tolist(), "b1": b1s.round(3).tolist(), "loss": grid, "path": path,
            "learning_rate": lr, "prevalence": float(y.mean()), "n": int(len(y))}


def neural_net_curves(X: np.ndarray, y: np.ndarray, inputs, seed=42, epochs=400) -> dict:
    """Small MLP, loss per epoch on training and held-out patients, without and with weight penalty."""
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    prep = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]).fit(Xtr)
    Xtr, Xva = prep.transform(Xtr), prep.transform(Xva)
    curves = {}
    for label, alpha in (("without regularization", 1e-4), ("with regularization", 3.0)):
        net = MLPClassifier(hidden_layer_sizes=(8,), alpha=alpha, learning_rate_init=0.01, random_state=seed)
        tr, va = [], []
        for _ in range(epochs):
            net.partial_fit(Xtr, ytr, classes=[0, 1])
            tr.append(round(log_loss(ytr, net.predict_proba(Xtr)[:, 1], labels=[0, 1]), 4))
            va.append(round(log_loss(yva, net.predict_proba(Xva)[:, 1], labels=[0, 1]), 4))
        curves[label] = {"train": tr, "val": va,
                         "w1": net.coefs_[0].T.round(3).tolist(), "w2": net.coefs_[1][:, 0].round(3).tolist()}
    return {"curves": curves, "n_train": int(len(ytr)), "n_val": int(len(yva)),
            "events_train": int(ytr.sum()), "events_val": int(yva.sum()), "inputs": list(inputs),
            "n_params": int(X.shape[1] * 8 + 8 + 8 + 1)}


def regularization_path(X: np.ndarray, y: np.ndarray, inputs, chosen_C: float | None = None,
                        n_points=25, n_repeats=5, seed=42) -> dict:
    """Penalized logistic over a grid of L2 strengths: coefficients, apparent vs cross-validated AUC,
    spread of the predicted risks. Only aggregates leave this function (no patient rows)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import RepeatedStratifiedKFold

    def pipe(C):
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                         ("lr", LogisticRegression(C=C, max_iter=2000))])

    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed).split(X, y))
    steps = []
    for C in np.logspace(-3, 2, n_points):
        full = pipe(C).fit(X, y)
        p = full.predict_proba(X)[:, 1]
        cv = [roc_auc_score(y[te], pipe(C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]) for tr, te in folds]
        steps.append({"C": float(C), "coef": full[-1].coef_[0].round(4).tolist(),
                      "auc_train": round(float(roc_auc_score(y, p)), 4), "auc_cv": round(float(np.mean(cv)), 4),
                      "p_min": round(float(p.min()), 4), "p_max": round(float(p.max()), 4)})
    return {"inputs": list(inputs), "steps": steps, "chosen_C": chosen_C, "prevalence": float(y.mean()),
            "n": int(len(y)), "events": int(y.sum()), "cv": f"5-fold x {n_repeats}"}


def material(X: np.ndarray, y: np.ndarray, inputs, gd_column: int, gd_label: str, source: str,
             chosen_C: float | None = None) -> dict:
    gd = gradient_descent(X[:, gd_column], y)
    gd["predictor"] = gd_label
    return {"source": source, "gradient_descent": gd, "neural_net": neural_net_curves(X, y, inputs),
            "regularization": regularization_path(X, y, inputs, chosen_C)}
