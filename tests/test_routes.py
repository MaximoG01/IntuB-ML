from app.models import IntubationRecord

ASSESSMENT = {"age": "62", "sex": "M", "weight": "95", "height": "175", "asa": "2", "emergency": "0",
              "dii": "3,5", "dtm": "5.8", "mallampati": "3", "neck_movement": "1",
              "can_prognath": "0", "history_difficult": "0"}


def test_pages_respond(client):
    for url in ("/", "/predict", "/analytics", "/blog", "/dashboard"):
        assert client.get(url).status_code == 200, url
    assert client.get("/seed-db-secret-key-12345").status_code == 404


def test_learn_pages_respond(client):
    from app.routes import LEARN_POSTS
    for p in LEARN_POSTS:
        assert client.get(f"/blog/{p['slug']}").status_code == 200, p["slug"]
    assert client.get("/blog/non-esiste").status_code == 404


def test_register_then_login(client):
    data = {"name": "Nuovo", "email": "nuovo@example.org", "password": "segreta1", "password2": "segreta1"}
    assert client.post("/register", data=data).status_code == 302
    body = client.post("/register", data=data).get_data(as_text=True)
    assert "already registered" in body
    resp = client.post("/login", data={"email": "nuovo@example.org", "password": "segreta1"})
    assert resp.status_code == 302


def test_predict_shows_model_and_egri(client):
    body = client.post("/predict", data=ASSESSMENT).get_data(as_text=True)
    # 1 (mouth) + 2 (TMD) + 2 (Mallampati III) + 1 (neck) + 1 (prognath) + 1 (weight) + 0 = 8
    assert "8 / 12" in body
    assert "Statistical model" in body


def test_predict_requires_age_and_weight(client):
    body = client.post("/predict", data={"age": "", "weight": ""}).get_data(as_text=True)
    assert "Age and weight are required." in body


def test_save_pending_then_complete(app, logged_in):
    logged_in.post("/predict", data={**ASSESSMENT, "action": "save"})
    rec = IntubationRecord.query.one()
    assert rec.alganzouri == 8 and rec.dii == 3.5 and rec.difficult_binary is None
    assert rec.predicted_proba is not None and rec.model_version

    logged_in.post(f"/case/{rec.id}/complete", data={"cormack": "3", "cormack_device": "DL", "success": "1"})
    rec = IntubationRecord.query.one()
    assert rec.difficult_binary is True and rec.cormack_device == "DL"


def test_empty_fields_stay_none(app, logged_in):
    logged_in.post("/predict", data={"age": "40", "weight": "70", "action": "save"})
    rec = IntubationRecord.query.one()
    assert rec.height is None and rec.dtm is None and rec.alganzouri is None


def test_new_case_with_outcome(app, logged_in):
    logged_in.post("/case/new", data={**ASSESSMENT, "cormack": "1", "success": "1"})
    rec = IntubationRecord.query.one()
    assert rec.difficult_binary is False and rec.success is True


def test_public_mode_closes_signup_and_serves_assets():
    from app import create_app
    from tests.conftest import TestConfig

    class PublicConfig(TestConfig):
        ALLOW_REGISTRATION = False
    c = create_app(PublicConfig).test_client()
    assert c.get("/register").status_code == 302
    home = c.get("/").get_data(as_text=True)
    assert "Sign up" not in home and "/assets/logo.svg?v=" in home
    assert c.get("/assets/logo.svg").status_code == 200
