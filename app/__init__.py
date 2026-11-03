import math

import click

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect
from config import Config

db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
login_manager.login_view = "main.login"
login_manager.login_message = "Accedi per continuare."
csrf = CSRFProtect()  # every POST needs a token, also plain HTML forms (AGENTS.md rule 3)

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    csrf.init_app(app)
    app.jinja_env.filters["exp"] = math.exp  # odds ratios from log-odds coefficients
    from app.inference import FEATURE_LABELS
    app.jinja_env.filters["feature_label"] = lambda f: FEATURE_LABELS.get(f, f)
    app.jinja_env.filters["pct"] = lambda p, d=1: f"{p * 100:.{d}f}".replace(".", ",") + "%"  # 0.46 -> "46,0%"

    from app import models  # noqa: F401
    from app.routes import bp as main_bp
    app.register_blueprint(main_bp)

    @app.cli.command("make-admin")
    @click.argument("email")
    def make_admin(email):
        """Give the admin role to a registered user: flask make-admin nome@esempio.it"""
        user = models.User.query.filter_by(email=email).first()
        if user is None:
            raise click.ClickException(f"Nessun utente registrato con {email}.")
        user.is_admin = True
        db.session.commit()
        click.echo(f"{email} è ora amministratore.")

    return app
