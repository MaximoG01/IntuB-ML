import csv
import hmac
import io
from types import SimpleNamespace
from sqlalchemy import or_
from flask import Blueprint, Response, render_template, redirect, url_for, request, flash, abort, current_app
from flask_login import login_required, current_user, login_user, logout_user

from app import db
from app.models import User, IntubationRecord, CustomField, CustomValue
from app.forms import LoginForm, RegistrationForm, CustomFieldForm, ToggleForm

from app.ml import (
    evaluate_logistic,
    DIFFICULT_THRESHOLD,
    InsufficientDataError,
)
from app.egri import egri_from_record, NECK_MOVEMENT_CHOICES, HISTORY_CHOICES
from app.inference import active_model, study_models, teaching_data, predict as predict_proba

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
    invite = current_app.config.get("INVITE_CODE")
    if form.validate_on_submit():
        if invite and not hmac.compare_digest((form.invite_code.data or "").strip(), invite):
            form.invite_code.errors.append("Invalid invite code: ask whoever runs the app.")
            return render_template("register.html", form=form, invite_required=True)
        if User.query.filter_by(email=form.email.data).first():
            form.email.errors.append("Email already registered.")
            return render_template("register.html", form=form, invite_required=bool(invite))
        user = User(email=form.email.data, name=form.name.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("main.login"))
    return render_template("register.html", form=form, invite_required=bool(invite))


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
            custom = _custom_from_form(request.form)
        except ValueError as e:
            err = str(e) if isinstance(e, FormError) else "Invalid values: use numbers or leave the field empty."
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
                    flash("Log in to save the case.", "warning")
                else:
                    # Pending case: outcome stays NULL until completed after intubation
                    rec = IntubationRecord(
                        operator_id=current_user.id,
                        predicted_proba=result["probability"] if result else None,
                        model_version=model["name"] if result else None,
                        **data,
                    )
                    _store_custom(rec, custom)
                    db.session.add(rec)
                    db.session.commit()
                    saved_record_id = rec.id
                    flash("Case saved as pending: add the outcome after intubation.", "success")

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
        flash("Outcome saved.", "success")
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


class FormError(ValueError):
    """Invalid input, with a message that can be shown to the user as is."""


def _assessment_from_form(form) -> dict:
    """Pre-intubation fields shared by predict, new and edit. Raises ValueError if invalid."""
    age, weight = _int_or_none(form.get("age")), _float_or_none(form.get("weight"))
    if age is None or weight is None:
        raise FormError("Age and weight are required.")
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


def _custom_from_form(form) -> dict:
    """Active custom fields -> parsed value (None if empty). Raises FormError listing every invalid field."""
    values, errors = {}, []
    for field in CustomField.query.filter_by(active=True).order_by(CustomField.id):
        try:
            values[field] = field.parse(form.get(f"cf_{field.id}"))
        except ValueError as e:
            errors.append(str(e) if field.label in str(e) else f"{field.label}: invalid value.")
    if errors:
        raise FormError(" ".join(errors))
    return values


def _store_custom(rec: IntubationRecord, values: dict) -> None:
    existing = {cv.field_id: cv for cv in rec.custom_values}
    for field, value in values.items():
        cv = existing.get(field.id)
        if value is None:
            if cv is not None:
                rec.custom_values.remove(cv)
            continue
        if cv is None:
            cv = CustomValue(field=field)
            rec.custom_values.append(cv)
        cv.value_num = value if field.kind == "number" else None
        cv.value_choice = value if field.kind == "choice" else None


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
            _store_custom(rec, _custom_from_form(request.form))
        except ValueError as e:
            flash(str(e) if isinstance(e, FormError) else "Invalid values: use numbers or leave the field empty.", "danger")
            return render_template("case_form.html", rec=None)
        _apply_outcome(rec, request.form)
        db.session.add(rec)
        db.session.commit()
        flash("Case saved.", "success")
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
            custom = _custom_from_form(request.form)
        except ValueError as e:
            flash(str(e) if isinstance(e, FormError) else "Invalid values: use numbers or leave the field empty.", "danger")
            return render_template("case_form.html", rec=rec)
        for k, v in data.items():
            setattr(rec, k, v)
        _store_custom(rec, custom)
        _apply_outcome(rec, request.form)
        db.session.commit()
        flash("Case updated.", "success")
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
    return {"NECK_MOVEMENT_CHOICES": NECK_MOVEMENT_CHOICES, "HISTORY_CHOICES": HISTORY_CHOICES,
            "custom_fields": lambda: CustomField.query.filter_by(active=True).order_by(CustomField.id).all()}


# ------------------------------------------------------
# EXTRA VARIABLES (admin) and EXPORT for offline training
# ------------------------------------------------------
def _require_admin() -> None:
    if not current_user.is_authenticated or not current_user.is_admin:
        abort(403)


@bp.route("/variables", methods=["GET", "POST"])
@login_required
def custom_fields():
    form = CustomFieldForm()
    if request.method == "POST":
        _require_admin()
        if form.validate_on_submit():
            if CustomField.query.filter_by(key=form.key.data).first():
                form.key.errors.append("A variable with this short name already exists.")
            else:
                field = CustomField(created_by_id=current_user.id)
                form.populate_obj(field)
                if field.kind == "choice":
                    field.unit = field.min_value = field.max_value = None
                else:
                    field.choices = None
                db.session.add(field)
                db.session.commit()
                flash(f'Variable "{field.label}" added: it now appears in the Assess form.', "success")
                return redirect(url_for("main.custom_fields"))
    fields = CustomField.query.order_by(CustomField.id).all()
    counts = dict(db.session.query(CustomValue.field_id, db.func.count(CustomValue.id))
                  .group_by(CustomValue.field_id).all())
    return render_template("custom_fields.html", form=form, fields=fields, counts=counts, toggle=ToggleForm())


@bp.route("/variables/<int:field_id>/toggle", methods=["POST"])
@login_required
def toggle_custom_field(field_id: int):
    _require_admin()
    if not ToggleForm().validate_on_submit():
        abort(400)
    field = db.session.get(CustomField, field_id) or abort(404)
    field.active = not field.active  # never deleted: values already collected stay in the database
    db.session.commit()
    state = "reactivated" if field.active else "deactivated"
    flash(f'Variable "{field.label}" {state}.', "success")
    return redirect(url_for("main.custom_fields"))


EXPORT_COLUMNS = ["age", "sex", "weight", "height", "asa", "emergency", "dii", "dtm", "mallampati",
                  "neck_movement", "can_prognath", "history_difficult", "stop_bang", "alganzouri",
                  "predicted_proba", "model_version", "cormack", "cormack_device", "success", "difficult_binary"]


@bp.route("/export.csv")
@login_required
def export_csv():
    """Pseudonymized cases for offline training: no names, no free-text fields (they may contain identifiers)."""
    _require_admin()
    fields = CustomField.query.order_by(CustomField.id).all()
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["case_id", "operator_id", "created_at"] + EXPORT_COLUMNS + [f"x_{f.key}" for f in fields])
    for rec in IntubationRecord.query.order_by(IntubationRecord.id):
        extra = {cv.field_id: cv.value for cv in rec.custom_values}
        row = [rec.id, rec.operator_id, rec.created_at.date().isoformat() if rec.created_at else ""]
        row += ["" if getattr(rec, c) is None else getattr(rec, c) for c in EXPORT_COLUMNS]
        row += ["" if extra.get(f.id) is None else extra[f.id] for f in fields]  # empty = missing, never 0
        w.writerow(row)
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=intub-cases.csv"})


# ------------------------------------------------------
# LEARN (ex blog): one template per post in templates/learn/
# ------------------------------------------------------
LEARN_POSTS = [
    {"slug": "difficult-laryngoscopy", "title": "Difficult laryngoscopy: Cormack, Mallampati, El-Ganzouri",
     "subtitle": "What we want to predict, with which bedside signs, and why the device changes the outcome."},
    {"slug": "how-the-model-was-built", "title": "How our model was built (and train it yourself)",
     "subtitle": "From 500 patients to the app in five steps; move the penalty and watch weights and AUC change."},
    {"slug": "evaluating-a-model", "title": "How to evaluate a risk model",
     "subtitle": "AUC, calibration, the 15% threshold and decision curves, with the numbers from our studies."},
    {"slug": "bias-probast", "title": "Risk of bias: PROBAST applied to our studies",
     "subtitle": "Four domains to decide how far to trust a prediction model."},
    {"slug": "gradient-descent", "title": "How a logistic regression learns: gradient descent",
     "subtitle": "The real error surface, computed on real patients, and the path step by step."},
    {"slug": "overfitting", "title": "Neural networks and overfitting",
     "subtitle": "Why with few events a network learns by heart, and how to see it."},
    {"slug": "obese-study", "title": "The obese-patient study: what we found",
     "subtitle": "Protocol, results, and why a negative result is still a result."},
    {"slug": "thyroid-study", "title": "The thyroid-surgery study (Padova)",
     "subtitle": "500 patients with the El-Ganzouri items: does the model beat the 1996 rule?"},
]


@bp.route("/blog")
def blog():
    return render_template("blog.html", posts=LEARN_POSTS)


@bp.route("/blog/<slug>")
def blog_post(slug: str):
    post = next((p for p in LEARN_POSTS if p["slug"] == slug), None)
    if not post:
        abort(404)
    return render_template(f"learn/{slug}.html", post=post, posts=LEARN_POSTS,
                           model=active_model(), teaching=teaching_data(), studies=study_models())
