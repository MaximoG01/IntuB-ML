"""
Model comparison with the same folds and the same metrics for every model (SPEC §6).

Repeated stratified K-fold: every preprocessing step lives inside the sklearn Pipeline,
so it is fitted on the training folds only (no leakage).
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold

THRESHOLD = 0.15
DCA_THRESHOLDS = np.round(np.arange(0.01, 0.31, 0.01), 2)


def oof_predictions(model, X: np.ndarray, y: np.ndarray, n_splits=5, n_repeats=10, seed=42) -> np.ndarray:
    """Out-of-fold probabilities, shape (n_repeats, n_samples)."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    out = np.full((n_repeats, len(y)), np.nan)
    for i, (tr, te) in enumerate(cv.split(X, y)):
        m = clone(model).fit(X[tr], y[tr])
        out[i // n_splits, te] = m.predict_proba(X[te])[:, 1]
    return out


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray):
    """Slope: logistic regression of y on logit(p) (1 = ideal). In-the-large: mean observed - mean predicted."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y).coef_[0, 0]
    return float(slope), float(y.mean() - p.mean())


def threshold_metrics(y: np.ndarray, p: np.ndarray, t: float = THRESHOLD) -> Dict[str, float]:
    pred = p >= t
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1))); tn = int(np.sum(~pred & (y == 0)))
    div = lambda a, b: float(a / b) if b else float("nan")
    return {
        "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
        "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn),
        "flagged": div(tp + fp, len(y)), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def net_benefit(y: np.ndarray, p: np.ndarray, thresholds=DCA_THRESHOLDS) -> List[float]:
    n = len(y)
    out = []
    for t in thresholds:
        pred = p >= t
        tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
        out.append(float(tp / n - fp / n * t / (1 - t)))
    return out


def calibration_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    """Quantile bins: mean predicted vs observed rate, with bin size."""
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    return [{"pred": float(p[b].mean()), "obs": float(y[b].mean()), "n": int(len(b))} for b in bins if len(b)]


def summarize(y: np.ndarray, P: np.ndarray) -> Dict:
    """Metrics per repeat → mean and 2.5–97.5 percentile across repeats; curves from repeat 0."""
    per = {k: [] for k in ("auc", "auprc", "brier", "cal_slope", "cal_in_large",
                           "sensitivity", "specificity", "ppv", "npv", "flagged")}
    for p in P:
        per["auc"].append(roc_auc_score(y, p))
        per["auprc"].append(average_precision_score(y, p))
        per["brier"].append(brier_score_loss(y, p))
        s, i = calibration_slope_intercept(y, p)
        per["cal_slope"].append(s); per["cal_in_large"].append(i)
        for k, v in threshold_metrics(y, p).items():
            if k in per:
                per[k].append(v)
    stats = {k: {"mean": float(np.nanmean(v)),
                 "lo": float(np.nanpercentile(v, 2.5)),
                 "hi": float(np.nanpercentile(v, 97.5))} for k, v in per.items()}
    p0 = P[0]
    fpr, tpr, _ = roc_curve(y, p0)
    idx = np.linspace(0, len(fpr) - 1, min(len(fpr), 200)).astype(int)  # keep JSON small
    return {
        "metrics": stats,
        "roc": {"fpr": fpr[idx].round(4).tolist(), "tpr": tpr[idx].round(4).tolist()},
        "calibration": calibration_curve(y, p0),
        "dca": {"thresholds": DCA_THRESHOLDS.tolist(), "net_benefit": net_benefit(y, p0)},
    }


def dca_reference(y: np.ndarray) -> Dict[str, List[float]]:
    prev = float(y.mean())
    return {"thresholds": DCA_THRESHOLDS.tolist(),
            "treat_all": [prev - (1 - prev) * t / (1 - t) for t in DCA_THRESHOLDS],
            "treat_none": [0.0] * len(DCA_THRESHOLDS)}


def compare(models: Dict[str, object], X: np.ndarray, y: np.ndarray, **cv) -> Dict[str, Dict]:
    return {name: summarize(y, oof_predictions(m, X, y, **cv)) for name, m in models.items()}


def fold_auc(model, X: np.ndarray, y: np.ndarray, n_splits=5, n_repeats=10, seed=42) -> Dict[str, float]:
    """
    AUC computed inside each test fold, averaged per repeat. With few events the AUC of pooled
    out-of-fold predictions is biased below 0.5 for weak models (fold-to-fold shifts in training
    prevalence anti-correlate with the outcome); within one fold that shift cancels out.
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    aucs = np.array([roc_auc_score(y[te], clone(model).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])
                     for tr, te in cv.split(X, y)]).reshape(n_repeats, n_splits).mean(axis=1)
    return {"mean": float(aucs.mean()), "lo": float(np.percentile(aucs, 2.5)), "hi": float(np.percentile(aucs, 97.5))}


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 2000, seed: int = 42) -> Dict[str, float]:
    """Point AUC and percentile bootstrap 95% CI (resampling patients)."""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            aucs.append(roc_auc_score(y[i], p[i]))
    return {"auc": float(roc_auc_score(y, p)),
            "lo": float(np.percentile(aucs, 2.5)), "hi": float(np.percentile(aucs, 97.5))}


def _apparent(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    slope, _ = calibration_slope_intercept(y, p)
    return {"auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p), "cal_slope": slope}


def optimism_corrected(model, X: np.ndarray, y: np.ndarray, n_boot: int = 500, seed: int = 42) -> Dict:
    """
    Harrell's bootstrap: fit on each bootstrap sample, measure how much better it looks on that
    sample than on the original data (optimism); corrected = apparent - mean optimism.
    The corrected calibration slope estimates the shrinkage the model would need.
    """
    rng = np.random.default_rng(seed)
    apparent = _apparent(y, clone(model).fit(X, y).predict_proba(X)[:, 1])
    optimism = {k: [] for k in apparent}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() < 5 or y[i].sum() > len(i) - 5:  # inner CV needs events in every fold
            continue
        m = clone(model).fit(X[i], y[i])
        on_boot = _apparent(y[i], m.predict_proba(X[i])[:, 1])
        on_orig = _apparent(y, m.predict_proba(X)[:, 1])
        for k in apparent:
            optimism[k].append(on_boot[k] - on_orig[k])
    return {k: {"apparent": float(apparent[k]), "optimism": float(np.mean(optimism[k])),
                "corrected": float(apparent[k] - np.mean(optimism[k]))} for k in apparent} | \
        {"n_boot": len(optimism["auc"])}


def riley_sample_size(prevalence: float, n_params: int, r2_cs: float | None = None,
                      shrinkage: float = 0.9, delta: float = 0.05) -> Dict[str, float]:
    """
    Minimum n for a binary-outcome prediction model (Riley et al., BMJ 2020;368:m441).
    Without prior information R2_CS is set to 15% of its maximum, as the authors suggest.
    """
    phi = prevalence
    ln_l0 = phi * np.log(phi) + (1 - phi) * np.log(1 - phi)  # null log-likelihood per patient
    max_r2 = 1 - np.exp(2 * ln_l0)
    r2 = r2_cs if r2_cs is not None else 0.15 * max_r2
    n1 = n_params / ((shrinkage - 1) * np.log(1 - r2 / shrinkage))
    s2 = r2 / (r2 + delta * max_r2)
    n2 = n_params / ((s2 - 1) * np.log(1 - r2 / s2))
    n3 = (1.96 / 0.05) ** 2 * phi * (1 - phi)
    n = int(np.ceil(max(n1, n2, n3)))
    return {"r2_cs": float(r2), "max_r2_cs": float(max_r2), "n_shrinkage": float(np.ceil(n1)),
            "n_optimism": float(np.ceil(n2)), "n_intercept": float(np.ceil(n3)),
            "n_required": n, "events_required": int(np.ceil(n * phi))}
