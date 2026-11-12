"""
Pattern submodels (Mercaldo & Blume, Biostatistics 2020): one logistic model for every combination of
observed signs, so that a sign left empty in the app is ignored instead of replaced by the median.
Why: training/missing_sim.py shows that the median (= "sign absent" for yes/no items) underestimates the
risk when a sign is missing (O/E up to 1.17), while submodels stay calibrated with the same AUC.

Requires complete training data. All submodels share the full model's standardization (mean and scale of
each sign do not depend on which other signs are in the model), so only intercept and weights are stored.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def key(observed) -> str:
    return "|".join(observed)


def pattern_submodels(X: np.ndarray, y: np.ndarray, features, C: float) -> dict:
    if np.isnan(X).any():
        raise ValueError("pattern submodels need complete training data")
    features = list(features)
    Xs = StandardScaler().fit_transform(X)
    p = float(y.mean())
    out = {"": {"intercept": round(math.log(p / (1 - p)), 6), "coef": {}}}  # nothing given: prevalence
    for r in range(1, len(features)):  # the full model (all signs) is the main model itself
        for idx in combinations(range(len(features)), r):
            lr = LogisticRegression(C=C, max_iter=2000).fit(Xs[:, idx], y)
            out[key(features[j] for j in idx)] = {
                "intercept": round(float(lr.intercept_[0]), 6),
                "coef": {features[j]: round(float(c), 6) for j, c in zip(idx, lr.coef_[0])}}
    return out
