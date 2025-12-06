
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
from app.ml import train_logistic_model

from app.ml import evaluate_logistic
from app.ml_nn import evaluate_nn, TORCH_AVAILABLE




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
    nn_error = None

    try:
        log_metrics = evaluate_logistic(min_samples=50)
    except ValueError as e:
        error = str(e)

    if TORCH_AVAILABLE:
        try:
            nn_metrics = evaluate_nn(min_samples=50)
        except Exception as e:
            nn_error = str(e)
    else:
        nn_error = "PyTorch non è disponibile su questo ambiente (Heroku)."

    return render_template(
        "analytics.html",
        log_metrics=log_metrics,
        nn_metrics=nn_metrics,
        error=error,
        nn_error=nn_error,
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

@bp.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    form = PredictionForm()
    prediction = None
    proba = None
    pending_record_id = None
    error = None

    if form.validate_on_submit():
        # 1) train the logistic model on complete cases
        try:
            # if you have few records, you can drop to min_samples=10
            model, metrics = train_logistic_model(min_samples=50)
        except ValueError as e:
            # not enough data, show message on page
            error = str(e)
            return render_template(
                "predict.html",
                form=form,
                prediction=prediction,
                proba=proba,
                pending_record_id=pending_record_id,
                error=error,
            )

        # 2) build a dummy record so we can reuse build_feature_vector()
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

        # 3) feature vector -> probability
        x = np.array([build_feature_vector(dummy)], dtype=float)
        proba = float(model.predict_proba(x)[0][1])  # prob of difficult intubation
        from app.ml import DIFFICULT_THRESHOLD  # make sure this import is present 
        # ...
        prediction = "Difficile" if proba >= DIFFICULT_THRESHOLD else "Facile"

        # 4) optionally save this case as PENDING (no outcome yet)
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
            flash("Caso salvato in attesa di esito.", "info")

    # IMPORTANT: we always render the page with prediction/proba if present
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
