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


def material(X: np.ndarray, y: np.ndarray, inputs, gd_column: int, gd_label: str, source: str) -> dict:
    gd = gradient_descent(X[:, gd_column], y)
    gd["predictor"] = gd_label
    return {"source": source, "gradient_descent": gd, "neural_net": neural_net_curves(X, y, inputs)}
