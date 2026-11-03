from app import create_app, db
from app.models import CustomField, CustomValue, IntubationRecord, User
from tests.conftest import TestConfig

NECK = {"label": "Neck circumference", "key": "neck_circ", "kind": "number", "unit": "cm",
        "min_value": "25", "max_value": "70"}
ULBT = {"label": "Upper lip bite test", "key": "ulbt", "kind": "choice", "choices": "I | II | III"}
CASE = {"age": "50", "weight": "80", "action": "save"}


def make_admin(app):
    # same session as the requests (the app fixture keeps one app context open)
    User.query.filter_by(email="clinico@example.org").one().is_admin = True
    db.session.commit()


def test_only_admin_can_add_variables(app, logged_in):
    assert logged_in.post("/variables", data=NECK).status_code == 403
    assert logged_in.get("/export.csv").status_code == 403
    make_admin(app)
    assert logged_in.post("/variables", data=NECK).status_code == 302
    assert CustomField.query.filter_by(key="neck_circ").one().unit == "cm"


def test_custom_values_saved_validated_and_exported(app, logged_in):
    make_admin(app)
    logged_in.post("/variables", data=NECK)
    logged_in.post("/variables", data=ULBT)
    neck, ulbt = CustomField.query.order_by(CustomField.id).all()
    assert ulbt.choice_list == ["I", "II", "III"]

    body = logged_in.get("/predict").get_data(as_text=True)
    assert "Neck circumference" in body and "Upper lip bite test" in body

    out_of_range = logged_in.post("/predict", data={**CASE, f"cf_{neck.id}": "200"}).get_data(as_text=True)
    assert "outside the allowed range" in out_of_range and IntubationRecord.query.count() == 0

    logged_in.post("/predict", data={**CASE, f"cf_{neck.id}": "41,5", f"cf_{ulbt.id}": ""})
    rec = IntubationRecord.query.one()
    assert [(cv.field.key, cv.value) for cv in rec.custom_values] == [("neck_circ", 41.5)]  # empty = no row

    csv = logged_in.get("/export.csv").get_data(as_text=True).splitlines()
    header, row = csv[0].split(","), csv[1].split(",")
    assert row[header.index("x_neck_circ")] == "41.5" and row[header.index("x_ulbt")] == ""


def test_deactivated_variable_hidden_but_values_kept(app, logged_in):
    make_admin(app)
    logged_in.post("/variables", data=NECK)
    field = CustomField.query.one()
    logged_in.post("/predict", data={**CASE, f"cf_{field.id}": "38"})
    logged_in.post(f"/variables/{field.id}/toggle")
    assert CustomField.query.one().active is False
    assert f'name="cf_{field.id}"' not in logged_in.get("/predict").get_data(as_text=True)
    assert CustomValue.query.count() == 1


def test_invite_code_required_when_set():
    class InviteConfig(TestConfig):
        INVITE_CODE = "segreto-reparto"
    app = create_app(InviteConfig)
    with app.app_context():
        db.create_all()
        c = app.test_client()
        data = {"name": "A", "email": "a@example.org", "password": "abcdef1", "password2": "abcdef1"}
        assert "Invalid invite code" in c.post("/register", data=data).get_data(as_text=True)
        assert c.post("/register", data={**data, "invite_code": "segreto-reparto"}).status_code == 302
        db.drop_all()


def test_post_without_csrf_token_is_rejected():
    class CsrfConfig(TestConfig):
        WTF_CSRF_ENABLED = True
    app = create_app(CsrfConfig)
    with app.app_context():
        db.create_all()
        assert app.test_client().post("/predict", data={"age": "50", "weight": "80"}).status_code == 400
        db.drop_all()
