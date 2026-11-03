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
login_manager.login_message = "Please log in to continue."
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
    app.jinja_env.filters["pct"] = lambda p, d=1: f"{p * 100:.{d}f}%"  # 0.46 -> "46.0%"

    from app import models  # noqa: F401
    from app.routes import bp as main_bp
    app.register_blueprint(main_bp)

    @app.cli.command("make-admin")
    @click.argument("email")
    def make_admin(email):
        """Give the admin role to a registered user: flask make-admin name@example.org"""
        user = models.User.query.filter_by(email=email).first()
        if user is None:
            raise click.ClickException(f"No registered user with {email}.")
        user.is_admin = True
        db.session.commit()
        click.echo(f"{email} is now an administrator.")

    return app
