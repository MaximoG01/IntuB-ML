"""
What to do when a sign is missing at prediction time? Simulation on the Padova data (no missing values).

    venv\\Scripts\\python.exe -m training.missing_sim

In each cross-validation fold the model is trained on complete data; values are then deleted from the test
patients only, and two strategies are compared:
  median    fill the gap with the training median (what the app does now; for yes/no signs = "absent")
  submodel  ignore the sign: use a model refitted on the training patients without it
            (pattern submodels, Mercaldo & Blume, Biostatistics 2020)
Scenarios: one sign missing in every test patient, and 20% / 40% of all test values missing at random.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.inference import FEATURE_LABELS, active_model
from training import thyroid

N_REPEATS = 10


def fit(X, y, C):
    return Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(C=C, max_iter=2000))]).fit(X, y)


def metrics(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(lp, y).coef_[0, 0]
    return {"auc": roc_auc_score(y, p), "o_e": y.sum() / p.sum(), "slope": slope}


def run(X, y, C, seed=0):
    rng = np.random.default_rng(seed)
    k = X.shape[1]
    scenarios = [(f"no {FEATURE_LABELS[f]}", [j]) for j, f in enumerate(thyroid.FEATURES)] + \
                [("20% of values at random", 0.2), ("40% of values at random", 0.4)]
    preds = {(s, m): [] for s, _ in scenarios for m in ("median", "submodel")}
    preds[("complete data", "-")] = []
    ys = []
    folds = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=seed).split(X, y)
    for tr, te in folds:
        Xtr, ytr, Xte = X[tr], y[tr], X[te]
        med = np.median(Xtr, axis=0)
        full = fit(Xtr, ytr, C)

        @lru_cache(maxsize=None)
        def sub(observed: tuple):
            return fit(Xtr[:, list(observed)], ytr, C) if observed else None

        def predict_sub(mask):  # mask[i, j] True = missing
            out = np.empty(len(mask))
            for i, row in enumerate(mask):
                obs = tuple(j for j in range(k) if not row[j])
                m = sub(obs)
                out[i] = m.predict_proba(Xte[i:i + 1, list(obs)])[0, 1] if m else ytr.mean()
            return out

        ys.append(y[te])
        preds[("complete data", "-")].append(full.predict_proba(Xte)[:, 1])
        for name, spec in scenarios:
            mask = np.zeros_like(Xte, dtype=bool)
            if isinstance(spec, list):
                mask[:, spec] = True
            else:
                mask = rng.random(Xte.shape) < spec
            preds[(name, "median")].append(full.predict_proba(np.where(mask, med, Xte))[:, 1])
            preds[(name, "submodel")].append(predict_sub(mask))

    # metrics per fold, then averaged (pooled AUC across folds is biased with few events)
    out = {}
    for key, ps in preds.items():
        per_fold = [metrics(yy, pp) for yy, pp in zip(ys, ps)]
        out[key] = {m: float(np.mean([d[m] for d in per_fold])) for m in per_fold[0]}
    return out


def main():
    df = thyroid.load()
    X, y = df[thyroid.FEATURES].to_numpy(dtype=float), df["difficult"].to_numpy(dtype=int)
    C = active_model()["C"]
    res = run(X, y, C)
    print(f"Padova, {len(y)} patients, {y.sum()} events; 5-fold CV x {N_REPEATS}; C = {C:.3f}")
    print(f"{'scenario':38s} {'method':9s} {'AUC':>6s} {'O/E':>6s} {'slope':>6s}")
    for (s, m), r in res.items():
        print(f"{s:38s} {m:9s} {r['auc']:6.3f} {r['o_e']:6.2f} {r['slope']:6.2f}")
    print("O/E = observed / expected events (1 = right average risk); slope 1 = ideal spread.")


if __name__ == "__main__":
    main()
