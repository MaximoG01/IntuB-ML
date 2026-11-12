import pytest

from app import create_app, db
from app.models import User


class TestConfig:
    SECRET_KEY = "test"
    SQLALCHEMY_DATABASE_URI = "sqlite://"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    WTF_CSRF_ENABLED = False
    TESTING = True
    ALLOW_REGISTRATION = True


@pytest.fixture
def app():
    app = create_app(TestConfig)
    with app.app_context():
        db.create_all()
        u = User(email="clinico@example.org", name="Clinico")
        u.set_password("password-di-test")
        db.session.add(u)
        db.session.commit()
        yield app
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def logged_in(client):
    client.post("/login", data={"email": "clinico@example.org", "password": "password-di-test"})
    return client
