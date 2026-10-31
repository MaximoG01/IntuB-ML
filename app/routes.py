from types import SimpleNamespace
from sqlalchemy import or_
from flask import Blueprint, render_template, redirect, url_for, request, flash, abort
from flask_login import login_required, current_user, login_user, logout_user

from app import db
from app.models import User, IntubationRecord
from app.forms import LoginForm, RegistrationForm

from app.ml import (
    evaluate_logistic,
    DIFFICULT_THRESHOLD,
    InsufficientDataError,
)
from app.egri import egri_from_record, NECK_MOVEMENT_CHOICES, HISTORY_CHOICES
from app.inference import active_model, predict as predict_proba

bp = Blueprint("main", __name__)


# ------------------------------------------------------
# HOME / LANDING
# ------------------------------------------------------
@bp.route("/")
def landing():
    return render_template("index.html")


# ------------------------------------------------------
# DASHBOARD
# ------------------------------------------------------
@bp.route("/dashboard")
def dashboard():
    query = IntubationRecord.query
    total_cases = query.count()
    pending_count = query.filter(
        or_(IntubationRecord.cormack.is_(None), IntubationRecord.difficult_binary.is_(None))
    ).count()
    completed_count = query.filter(
        IntubationRecord.cormack.isnot(None),
        IntubationRecord.difficult_binary.isnot(None),
    ).count()
    difficult_count = query.filter(IntubationRecord.difficult_binary.is_(True)).count()
    success_count = query.filter(IntubationRecord.success.is_(True)).count()

    if current_user.is_authenticated:
        records = (
            IntubationRecord.query
            .filter_by(operator_id=current_user.id)
            .order_by(IntubationRecord.created_at.desc())
            .all()
        )
        pending = [r for r in records if (r.cormack is None or r.difficult_binary is None)]
        completed = [r for r in records if (r.cormack is not None and r.difficult_binary is not None)]
    else:
        records = []
        pending = []
        completed = []

    difficulty_rate = (difficult_count / completed_count * 100) if completed_count else None
    success_rate = (success_count / completed_count * 100) if completed_count else None

    return render_template(
        "dashboard.html",
        pending=pending,
        completed=completed,
        total_cases=total_cases,
        pending_count=pending_count,
        completed_count=completed_count,
        difficulty_rate=difficulty_rate,
        success_rate=success_rate,
    )


# ------------------------------------------------------
# CASE LIST (logged-in only)
# ------------------------------------------------------
@bp.route("/cases")
@login_required
def case_list():
    records = (
        IntubationRecord.query
        .filter_by(operator_id=current_user.id)
        .order_by(IntubationRecord.created_at.desc())
        .all()
    )
    return render_template("cases.html", records=records)


# ------------------------------------------------------
# LOGIN
# ------------------------------------------------------
@bp.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            next_page = request.args.get("next")
            return redirect(next_page or url_for("main.dashboard"))
        flash("Invalid email or password", "danger")
    return render_template("login.html", form=form)


# ------------------------------------------------------
# LOGOUT
# ------------------------------------------------------
@bp.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("main.landing"))


# ------------------------------------------------------
# REGISTRAZIONE
# ------------------------------------------------------
@bp.route("/register", methods=["GET", "POST"])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(email=form.email.data, name=form.name.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("main.login"))
    return render_template("register.html", form=form)


# ------------------------------------------------------
# PREDICT (and optionally SAVE as PENDING CASE)
# ------------------------------------------------------
@bp.route("/predict", methods=["GET", "POST"])
def predict():
    """
    GET: show input form
    POST: run prediction; optionally save pending case (no outcome yet)
    """
    result = None
    egri = None
    saved_record_id = None
    err = None
    model = active_model()

    if request.method == "POST":
        try:
            # Empty fields stay None: the model must not read them as 0.
            data = _assessment_from_form(request.form)
        except ValueError as e:
            err = str(e) if "obbligatori" in str(e) else "Valori non validi: usa numeri o lascia vuoto."
            data = None

        if data is not None:
            dummy = SimpleNamespace(**data)
            egri = egri_from_record(dummy)
            pred = predict_proba(dummy, model)
            if pred is not None:
                proba, contributions = pred
                result = {"probability": proba, "difficult": proba >= DIFFICULT_THRESHOLD,
                          "contributions": contributions}

            if request.form.get("action") == "save":
                if not current_user.is_authenticated:
                    flash("Accedi per salvare il caso.", "warning")
                else:
                    # Pending case: outcome stays NULL until completed after intubation
                    rec = IntubationRecord(
                        operator_id=current_user.id,
                        predicted_proba=result["probability"] if result else None,
                        model_version=model["name"] if result else None,
                        **data,
                    )
                    db.session.add(rec)
                    db.session.commit()
                    saved_record_id = rec.id
                    flash("Caso salvato in attesa di esito: completalo dopo l'intubazione.", "success")

    return render_template(
        "predict.html",
        result=result,
        egri=egri,
        model=model,
        threshold=DIFFICULT_THRESHOLD,
        saved_record_id=saved_record_id,
        error=err,
    )


# ------------------------------------------------------
# COMPLETE A PENDING CASE (insert outcome after intubation)
# Default difficult selection = NOT difficult.
# ------------------------------------------------------
@bp.route("/case/<int:record_id>/complete", methods=["GET", "POST"])
@login_required
def complete_case(record_id: int):
    rec = IntubationRecord.query.get(record_id)
    if not rec or rec.operator_id != current_user.id:
        abort(404)

    if request.method == "POST":
        _apply_outcome(rec, request.form)
        db.session.commit()
        flash("Esito registrato.", "success")
        return redirect(url_for("main.dashboard"))

    return render_template("complete_case.html", rec=rec)


# ------------------------------------------------------
# CREATE / EDIT A FULL RECORD (no prediction)
# ------------------------------------------------------
def _float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip().replace(",", ".")
    return float(value) if value else None


def _int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    return int(value) if value else None


def _bool_or_none(value: str | None) -> bool | None:
    return {"1": True, "0": False}.get((value or "").strip())


def _assessment_from_form(form) -> dict:
    """Pre-intubation fields shared by predict, new and edit. Raises ValueError if invalid."""
    age, weight = _int_or_none(form.get("age")), _float_or_none(form.get("weight"))
    if age is None or weight is None:
        raise ValueError("Età e peso sono obbligatori.")
    data = dict(
        age=age,
        weight=weight,
        height=_float_or_none(form.get("height")),
        sex=form.get("sex") or None,
        asa=_int_or_none(form.get("asa")),
        emergency=_bool_or_none(form.get("emergency")),
        dtm=_float_or_none(form.get("dtm")),
        dii=_float_or_none(form.get("dii")),
        mallampati=_int_or_none(form.get("mallampati")),
        neck_movement=_int_or_none(form.get("neck_movement")),
        can_prognath=_bool_or_none(form.get("can_prognath")),
        history_difficult=_int_or_none(form.get("history_difficult")),
        stop_bang=_int_or_none(form.get("stop_bang")),
        drug_used=form.get("drug_used") or None,
        technique=form.get("technique") or None,
    )
    # EGRI total computed from the items, never typed in (None if an item is missing)
    data["alganzouri"] = egri_from_record(SimpleNamespace(**data)).total
    return data


def _apply_outcome(rec: IntubationRecord, form) -> None:
    rec.success = _bool_or_none(form.get("success"))
    rec.cormack = _int_or_none(form.get("cormack"))
    rec.cormack_device = form.get("cormack_device") or None
    # standardize difficulty from Cormack (>2 = difficult)
    rec.difficult_binary = (rec.cormack > 2) if rec.cormack is not None else None


@bp.route("/case/new", methods=["GET", "POST"])
@login_required
def new_case():
    if request.method == "POST":
        try:
            rec = IntubationRecord(operator_id=current_user.id, **_assessment_from_form(request.form))
        except ValueError as e:
            flash(str(e), "danger")
            return render_template("case_form.html", rec=None)
        _apply_outcome(rec, request.form)
        db.session.add(rec)
        db.session.commit()
        flash("Caso salvato.", "success")
        return redirect(url_for("main.dashboard"))

    return render_template("case_form.html", rec=None)


@bp.route("/case/<int:record_id>/edit", methods=["GET", "POST"])
@login_required
def edit_case(record_id: int):
    rec = IntubationRecord.query.get(record_id)
    if not rec or rec.operator_id != current_user.id:
        abort(404)

    if request.method == "POST":
        try:
            data = _assessment_from_form(request.form)
        except ValueError as e:
            flash(str(e), "danger")
            return render_template("case_form.html", rec=rec)
        for k, v in data.items():
            setattr(rec, k, v)
        _apply_outcome(rec, request.form)
        db.session.commit()
        flash("Caso aggiornato.", "success")
        return redirect(url_for("main.dashboard"))

    return render_template("case_form.html", rec=rec)


# ------------------------------------------------------
# MODEL: active exported model + cases collected in this app
# ------------------------------------------------------
@bp.route("/analytics")
def analytics():
    model = active_model()
    local_metrics = None
    local_error = None
    try:
        # quick internal check on the cases collected here (not the active model)
        local_metrics = evaluate_logistic(min_samples=20)
    except InsufficientDataError as e:
        local_error = str(e)
    except Exception as e:
        local_error = f"Logistic regression failed: {e}"

    n_labeled = IntubationRecord.query.filter(IntubationRecord.difficult_binary.isnot(None)).count()
    return render_template(
        "analytics.html",
        model=model,
        local_metrics=local_metrics,
        local_error=local_error,
        n_labeled=n_labeled,
    )


@bp.app_context_processor
def _inject_choices():
    return {"NECK_MOVEMENT_CHOICES": NECK_MOVEMENT_CHOICES, "HISTORY_CHOICES": HISTORY_CHOICES}


# ------------------------------------------------------
# BLOG
# ------------------------------------------------------
def _blog_posts():
    # You will later add images into: app/static/blog/
    return [
        {
            "slug": "difficult-intubation-theory",
            "title": "Difficult Intubation: Clinical Theory and Airway Assessment",
            "subtitle": "A practical overview of predictors, grading systems, and why difficulty happens.",
            "img": "blog/airway_overview.jpg",
        },
        {
            "slug": "prediction-models",
            "title": "Prediction Models: Logistic Regression, Neural Networks, Hybrid and Distillation",
            "subtitle": "How the risk score is computed and why hybrid models are useful.",
            "img": "blog/ml_models.jpg",
        },
        {
            "slug": "how-this-site-works",
            "title": "How This Website Works (Data, Privacy, Training Loop, Visualizations)",
            "subtitle": "System architecture, data lifecycle, and what you see in Analytics.",
            "img": "blog/site_architecture.jpg",
        },
    ]


@bp.route("/blog")
def blog():
    return render_template("blog.html", posts=_blog_posts())


@bp.route("/blog/<slug>")
def blog_post(slug: str):
    posts = {p["slug"]: p for p in _blog_posts()}
    post = posts.get(slug)
    if not post:
        abort(404)

    # Content blocks (simple HTML strings for now).
    # You can later replace with Markdown, database content, etc.
    if slug == "difficult-intubation-theory":
        sections = [
            {
                "h": "Why intubation becomes difficult",
                "p": "Difficult intubation generally results from reduced laryngoscopic view, limited mouth opening, reduced cervical mobility, unfavorable anatomy, or a combination. Clinically we distinguish difficulty in mask ventilation, supraglottic placement, laryngoscopy view, and tube passage—each has different predictors and rescue strategies.",
                "img": "blog/laryngoscopy_view.jpg",
                "caption": "Example airway view and the concept of limited glottic exposure.",
            },
            {
                "h": "Cormack–Lehane and what it means operationally",
                "p": "Cormack–Lehane grading is a pragmatic proxy of laryngeal view. Grades 3–4 often correlate with increased attempts and need for adjuncts. In this project, the binary outcome “difficult” is defined as Cormack > 2 (i.e., grades 3–4).",
                "img": "blog/cormack_scale.jpg",
                "caption": "Cormack–Lehane grading (add your own diagram image here).",
            },
            {
                "h": "Pre-intubation predictors used here",
                "p": "The input features mirror common bedside assessment: Mallampati, thyromental distance (DTM), interincisor distance (DII), STOP-BANG, and composite scores (e.g., El-Ganzouri). These are not perfect predictors individually; their value increases when combined and calibrated on local case-mix.",
                "img": "blog/assessment_tools.jpg",
                "caption": "Airway assessment measures used as model inputs.",
            },
        ]

    elif slug == "prediction-models":
        sections = [
            {
                "h": "Logistic regression: calibrated baseline",
                "p": "Logistic regression is a strong baseline for clinical risk scoring: interpretable coefficients, stable training on small data, and usually good calibration. It produces a probability p(difficult) used for thresholding and ROC analysis.",
                "img": "blog/logreg_explained.jpg",
                "caption": "Logistic regression as a calibrated probability model.",
            },
            {
                "h": "Neural networks: non-linear feature interactions",
                "p": "A shallow NN can capture non-linear interactions (e.g., combinations of DTM + Mallampati + mouth opening). When data is limited, regularization and careful validation are essential, and the NN should not be trusted blindly without monitoring drift and calibration.",
                "img": "blog/nn_explained.jpg",
                "caption": "Shallow NN (1 hidden layer) capturing interactions.",
            },
            {
                "h": "Hybrid + distillation (why)",
                "p": "Hybrid models let the logistic regression guide learning (teacher) while the NN learns residual patterns (student). Distillation is a practical compromise: you train locally with heavier tooling (torch), then deploy only a lightweight distilled model for inference/visualization where torch is unavailable.",
                "img": "blog/hybrid_distill.jpg",
                "caption": "Teacher–student / distillation concept (add your own diagram).",
            },
        ]

    else:  # how-this-site-works
        sections = [
            {
                "h": "Workflow: Predict → Save pending → Complete outcome",
                "p": "The clinician can run a prediction pre-procedure and optionally save the case as “pending”. After intubation, the outcome (Cormack, success, difficult yes/no) is entered from the dashboard. Completed cases expand the training dataset.",
                "img": "blog/workflow.jpg",
                "caption": "Data lifecycle: prediction and deferred labeling.",
            },
            {
                "h": "Analytics: ROC, NN diagram, loss landscape",
                "p": "Analytics provides (1) ROC curve with reference diagonal, (2) NN architecture visualization, and (3) a pedagogical loss surface that illustrates gradient descent. The loss surface axes are abstracted parameters (two degrees of freedom) and the color/height indicates loss.",
                "img": "blog/analytics_panels.jpg",
                "caption": "Analytics panels overview (add a screenshot later).",
            },
            {
                "h": "Privacy and intended use",
                "p": "This is a decision-support tool, not a replacement for clinical judgment. Store only necessary data, keep access controlled, and treat model outputs as probabilistic signals requiring clinical validation and governance.",
                "img": "blog/privacy.jpg",
                "caption": "Privacy-by-design (insert your own visual).",
            },
        ]

    return render_template("blog_post.html", post=post, sections=sections)
