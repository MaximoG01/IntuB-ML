
import os
import secrets
from pathlib import Path

basedir = Path(__file__).resolve().parent

class Config:
    # Without SECRET_KEY a random key is used: sessions are lost at every restart.
    SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        "sqlite:///" + str(basedir / "app.db"),
    )
    # Heroku may provide postgres://, SQLAlchemy expects postgresql://
    if SQLALCHEMY_DATABASE_URI.startswith("postgres://"):
        SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace("postgres://", "postgresql://", 1)

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # If set, registration requires this code (shared app). Unset = open registration (local development).
    INVITE_CODE = os.environ.get("INVITE_CODE") or None
    WTF_CSRF_ENABLED = True
