import math

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from config import Config

db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
login_manager.login_view = "main.login"

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    app.jinja_env.filters["exp"] = math.exp  # odds ratios from log-odds coefficients

    from app import models  # noqa: F401
    from app.routes import bp as main_bp
    app.register_blueprint(main_bp)

    return app
