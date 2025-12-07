
import numpy as np
from flask import render_template, redirect, url_for, flash, abort, request
from flask_login import login_required, current_user

from app import db
from app.models import IntubationRecord
from app.forms import IntubationForm, PredictionForm, OutcomeForm
from app.ml import evaluate_logistic  # NON importiamo più evaluate_nn qui
# ⚠️ togli o commenta:
# from app.ml_nn import evaluate_nn
from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from sqlalchemy import func

from app import db
from app.models import User, IntubationRecord
from app.forms import LoginForm, RegisterForm, IntubationForm
from app.ml import train_logistic_model, build_feature_vector, DIFFICULT_THRESHOLD


# PRIMA (probabile)
# from app.ml_nn import evaluate_nn, TORCH_AVAILABLE

# DOPO
try:
    from app.ml_nn import evaluate_nn, TORCH_AVAILABLE
except Exception:
    evaluate_nn = None
    TORCH_AVAILABLE = False





bp = Blueprint("main", __name__)

@bp.route("/")
def landing():
    return render_template("index.html")

@bp.route("/dashboard")
@login_required
def dashboard():
    total = IntubationRecord.query.count()
    user_total = IntubationRecord.query.filter_by(operator_id=current_user.id).count()
    difficult = IntubationRecord.query.filter_by(difficult_binary=True).count()
    return render_template(
        "dashboard.html",
        total=total,
        user_total=user_total,
        difficult=difficult,
    )

@bp.route("/blog")
def blog():
    return render_template("blog.html")

# app/routes.py (o dove hai il blueprint bp)

from app.ml import evaluate_logistic
# ATTENZIONE: per Heroku al momento NON importiamo evaluate_nn
# from app.ml_nn import evaluate_nn

@bp.route("/analytics")
@login_required
def analytics():
    log_metrics = None
    nn_metrics = None
    error = None

    # Logistic regression
    try:
        log_metrics = evaluate_logistic(min_samples=50)
    except ValueError as e:
        error = str(e)

    # PyTorch NN – se qualcosa va storto NON rompere la pagina
    try:
        nn_metrics = evaluate_nn(min_samples=50)
    except Exception as e:
        print("NN error:", e)

    return render_template(
        "analytics.html",
        log_metrics=log_metrics,
        nn_metrics=nn_metrics,
        error=error,
    )





@bp.route("/analytics/json")
@login_required
def analytics_json():
    try:
        _, metrics = train_logistic_model()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(metrics)

@bp.route("/new", methods=["GET", "POST"])
@login_required
def new_record():
    form = IntubationForm()
    if form.validate_on_submit():
        difficult = False
        if form.cormack.data is not None and form.cormack.data >= 3:
            difficult = True
        if not form.success.data:
            difficult = True

        record = IntubationRecord(
            age=form.age.data,
            weight=form.weight.data,
            dtm=form.dtm.data,
            dii=form.dii.data,
            mallampati=form.mallampati.data,
            stop_bang=form.stop_bang.data,
            alganzouri=form.alganzouri.data,
            drug_used=form.drug_used.data,
            technique=form.technique.data,
            success=form.success.data,
            cormack=form.cormack.data,
            difficult_binary=difficult,
            operator=current_user,
        )
        db.session.add(record)
        db.session.commit()
        flash("Record saved.")
        return redirect(url_for("main.dashboard"))
    return render_template("new_record.html", form=form)

import numpy as np
from app.ml import train_logistic_model, build_feature_vector, DIFFICULT_THRESHOLD
from app.forms import PredictionForm
from app.models import IntubationRecord
from app import db

@bp.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    form = PredictionForm()
    prediction = None
    proba = None
    pending_record_id = None
    error = None

    if form.validate_on_submit():
        try:
            model, metrics = train_logistic_model(min_samples=50)
        except ValueError as e:
            error = str(e)
            return render_template(
                "predict.html",
                form=form,
                prediction=prediction,
                proba=proba,
                pending_record_id=pending_record_id,
                error=error,
            )

        class Dummy:
            pass

        dummy = Dummy()
        dummy.age = form.age.data
        dummy.weight = form.weight.data
        dummy.height = form.height.data
        dummy.sex = form.sex.data or None
        dummy.dtm = form.dtm.data
        dummy.dii = form.dii.data
        dummy.mallampati = form.mallampati.data
        dummy.stop_bang = form.stop_bang.data
        dummy.alganzouri = form.alganzouri.data

        x = np.array([build_feature_vector(dummy)], dtype=float)
        proba = float(model.predict_proba(x)[0][1])

        prediction = "Difficile" if proba >= DIFFICULT_THRESHOLD else "Facile"

        if form.save_case.data:
            rec = IntubationRecord(
                operator_id=current_user.id,
                age=form.age.data,
                weight=form.weight.data,
                height=form.height.data,
                sex=form.sex.data or None,
                dtm=form.dtm.data,
                dii=form.dii.data,
                mallampati=form.mallampati.data,
                stop_bang=form.stop_bang.data,
                alganzouri=form.alganzouri.data,
                drug_used=form.drug_used.data,
                technique=form.technique.data,
                success=None,
                cormack=None,
                difficult_binary=None,
            )
            db.session.add(rec)
            db.session.commit()
            pending_record_id = rec.id

    return render_template(
        "predict.html",
        form=form,
        prediction=prediction,
        proba=proba,
        pending_record_id=pending_record_id,
        error=error,
    )


@bp.route("/records/<int:record_id>/outcome", methods=["GET", "POST"])
@login_required
def record_outcome(record_id):
    record = IntubationRecord.query.get_or_404(record_id)
    if record.operator_id != current_user.id:
        abort(403)

    form = OutcomeForm(obj=record)
    if form.validate_on_submit():
        record.success = form.success.data
        record.cormack = form.cormack.data
        record.difficult_binary = (record.cormack >= 3) or (not record.success)
        db.session.commit()
        flash("Esito salvato e registrato per il training.", "success")
        return redirect(url_for("main.dashboard"))

    return render_template("record_outcome.html", form=form, record=record)



@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data.lower()).first()
        if user is None or not user.check_password(form.password.data):
            flash("Invalid email or password.")
            return redirect(url_for("main.login"))
        login_user(user)
        return redirect(url_for("main.dashboard"))
    return render_template("login.html", form=form)

@bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    form = RegisterForm()
    if form.validate_on_submit():
        if User.query.filter_by(email=form.email.data.lower()).first():
            flash("Email already registered.")
            return redirect(url_for("main.register"))
        user = User(
            name=form.name.data,
            email=form.email.data.lower(),
        )
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful. Please log in.")
        return redirect(url_for("main.login"))
    return render_template("register.html", form=form)

@bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("main.landing"))
