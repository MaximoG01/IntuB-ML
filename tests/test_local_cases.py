import csv

import numpy as np

from app.inference import active_model
from training import run_local_cases as rlc

COLS = ["case_id", "mallampati", "dii", "dtm", "neck_movement", "can_prognath", "history_difficult",
        "weight", "height", "sex", "cormack", "cormack_device"]


def _write(path, n=200, seed=0):
    rng = np.random.default_rng(seed)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        for i in range(n):
            m = int(rng.integers(1, 5))
            w.writerow([i, m, 4.5, "" if i % 7 == 0 else 7.5, 0, "True", 0, 70, 170, "F",
                        3 if rng.random() < (0.35 if m >= 3 else 0.12) else 1, "VL" if i % 20 == 0 else "DL"])


def test_local_cases_validation_and_update(tmp_path):
    f = tmp_path / "cases.csv"
    _write(f)
    cases, counts = rlc.load_cases(f)
    assert counts["video"] > 0 and counts["used"] == counts["rows"] - counts["video"]
    assert any(r.dtm is None for r in cases)          # empty cell stays missing
    model = active_model()
    y = np.array([int(r.cormack >= 3) for r in cases])
    lp = rlc.linear_predictor(cases, model)
    v = rlc.validate(lp, y)
    assert 0 <= v["auc"] <= 1
    new, how = rlc.updated_model(model, lp, y, "test")
    # recalibrated model reproduces the local event rate
    p = 1 / (1 + np.exp(-rlc.linear_predictor(cases, new)))
    assert abs(p.mean() - y.mean()) < 0.02 or "slope" in how


def test_too_few_events_means_no_update():
    lp = np.zeros(30)
    y = np.array([1] * 5 + [0] * 25)
    assert rlc.updated_model(active_model(), lp, y, "test") is None
