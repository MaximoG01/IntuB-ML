from app.egri import egri_score


def test_all_low_risk_is_zero():
    r = egri_score(5.0, 7.0, 1, 0, True, 70, 0)
    assert r.total == 0 and r.positive is False


def test_all_high_risk_is_twelve():
    r = egri_score(3.0, 5.5, 4, 2, False, 120, 2)
    assert r.total == 12 and r.positive is True


def test_cutoffs():
    assert egri_score(mouth_opening_cm=4.0).items["mouth_opening"] == 0
    assert egri_score(thyromental_cm=6.5).items["thyromental"] == 1
    assert egri_score(thyromental_cm=6.0).items["thyromental"] == 1
    assert egri_score(thyromental_cm=5.9).items["thyromental"] == 2
    assert egri_score(weight_kg=90).items["weight"] == 1
    assert egri_score(weight_kg=110).items["weight"] == 1
    assert egri_score(weight_kg=110.1).items["weight"] == 2
    assert [egri_score(mallampati=m).items["mallampati"] for m in (1, 2, 3, 4)] == [0, 1, 2, 2]


def test_missing_item_gives_no_total():
    r = egri_score(3.0, 5.5, 4, None, False, 120, 2)
    assert r.total is None and r.partial == 10 and r.missing == ["neck_movement"]
