import numpy as np
from flask import Blueprint, render_template, redirect, url_for, request, flash, abort
from flask_login import login_required, current_user, login_user, logout_user

from app import db
from app.models import User, IntubationRecord
from app.forms import LoginForm, RegistrationForm

from app.ml import (
    train_logistic_model,
    evaluate_logistic,
    build_feature_vector,
    DIFFICULT_THRESHOLD,
    _load_xy,
)

from app.ml_nn import (
    evaluate_nn,
    TORCH_AVAILABLE,
)

from app.nn_viz import loss_surface_2d_safe, nn_svg_from_weights

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
@login_required
def dashboard():
    records = (
        IntubationRecord.query
        .order_by(IntubationRecord.created_at.desc())
        .all()
    )

    pending = [r for r in records if (r.cormack is None or r.difficult_binary is None)]
    completed = [r for r in records if (r.cormack is not None and r.difficult_binary is not None)]

    # Summary statistics for dashboard overview
    total_cases = len(records)
    pending_count = len(pending)
    completed_count = len(completed)

    # Compute difficulty and success rates among completed cases
    difficult_count = sum(1 for r in completed if r.difficult_binary)
    success_count = sum(1 for r in completed if r.success is True)

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
@login_required
def predict():
    """
    GET: show input form
    POST: run prediction; optionally save pending case (no outcome yet)
    """
    proba = None
    difficult_pred = None
    saved_record_id = None
    err = None

    if request.method == "POST":
        try:
            # Train/retrain logistic on available labeled data (or seed if your ml.py does that)
            model, _ = train_logistic_model(min_samples=20)

            # Build an in-memory record-like object (not saved unless user chooses)
            dummy = IntubationRecord(
                operator_id=current_user.id,
                age=int(request.form.get("age")),
                weight=float(request.form.get("weight")),
                height=float(request.form.get("height") or 0),
                sex=request.form.get("sex"),
                dtm=float(request.form.get("dtm") or 0),
                dii=float(request.form.get("dii") or 0),
                mallampati=int(request.form.get("mallampati") or 0),
                stop_bang=int(request.form.get("stop_bang") or 0),
                alganzouri=int(request.form.get("alganzouri") or 0),
            )

            X = np.array([build_feature_vector(dummy)], dtype=float)
            p = float(model.predict_proba(X)[0][1])

            proba = p
            difficult_pred = (p >= DIFFICULT_THRESHOLD)

            action = request.form.get("action", "predict")

            if action == "save":
                # Save as pending: outcomes are NULL until the clinician completes later
                rec = IntubationRecord(
                    operator_id=current_user.id,

                    age=dummy.age,
                    weight=dummy.weight,
                    height=dummy.height,
                    sex=dummy.sex,
                    dtm=dummy.dtm,
                    dii=dummy.dii,
                    mallampati=dummy.mallampati,
                    stop_bang=dummy.stop_bang,
                    alganzouri=dummy.alganzouri,

                    drug_used=request.form.get("drug_used") or None,
                    technique=request.form.get("technique") or None,

                    success=None,
                    cormack=None,
                    difficult_binary=None,
                )

                db.session.add(rec)
                db.session.commit()
                saved_record_id = rec.id
                flash("Case saved as pending. Complete it later from Dashboard.", "success")

        except Exception as e:
            err = str(e)

    return render_template(
        "predict.html",
        probability=proba,
        difficult=difficult_pred,
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
        # success: "1" / "0"
        success_raw = request.form.get("success")
        rec.success = True if success_raw == "1" else False if success_raw == "0" else None

        # cormack: 1..4 (or empty)
        cormack_raw = request.form.get("cormack", "").strip()
        rec.cormack = int(cormack_raw) if cormack_raw else None

        # difficult_binary: default is "0" in the form
        diff_raw = request.form.get("difficult_binary", "0")
        rec.difficult_binary = True if diff_raw == "1" else False

        db.session.commit()
        flash("Case completed and added to training dataset.", "success")
        return redirect(url_for("main.dashboard"))

    return render_template("complete_case.html", rec=rec)


# ------------------------------------------------------
# ANALYTICS (LOG-REG + optional NN + LOSS SURFACE)
# ------------------------------------------------------
@bp.route("/analytics")
@login_required
def analytics():
    log_metrics = None
    nn_metrics = None
    nn_error = None
    loss_surface = None
    error = None

    # Logistic regression metrics
    try:
        log_metrics = evaluate_logistic(min_samples=20)
    except Exception as e:
        error = f"Logistic regression failed: {e}"

    # NN metrics (optional, supports pre-trained weights when Torch is unavailable)
    try:
        nn_metrics = evaluate_nn(min_samples=20)
    except Exception as e:
        nn_error = f"Neural network unavailable: {e}"

    if nn_metrics is None and not TORCH_AVAILABLE:
        nn_error = "PyTorch is not available. NN analytics skipped."

    # Loss surface (use DB data if available for realistic rendering)
    try:
        X_db, y_db, data_source = None, None, "synthetic"
        try:
            X_db, y_db, data_source = _load_xy(min_samples=20, prefer_db=True)
        except Exception:
            # _load_xy may raise or return synthetic if DB empty
            X_db, y_db, data_source = None, None, "synthetic"

        loss_surface = loss_surface_2d_safe(X=X_db, y=y_db)
        
        # Add trajectory data to loss surface if we have NN training history
        if loss_surface and TORCH_AVAILABLE and nn_metrics and nn_metrics.get("theta_history"):
            try:
                import numpy as np
                # Project theta_history to the same 2D PCA space as the loss surface
                theta_hist = np.array(nn_metrics["theta_history"])  # shape: (epochs, n_weights)
                
                # Use same PCA projection as the data
                if X_db is not None and y_db is not None:
                    Xc = X_db.astype(float) - np.nanmean(X_db, axis=0)
                    u, s, vt = np.linalg.svd(np.nan_to_num(Xc), full_matrices=False)
                    comps = vt[:2].T  # (n_features, 2)
                    
                    # Project first 2 theta dimensions to 2D
                    if theta_hist.shape[1] >= 2:
                        # For each epoch's weights, project to 2D
                        traj_2d = []
                        for weights in theta_hist:
                            # Pad to match feature dimension if needed
                            w = np.zeros(X_db.shape[1])
                            w[:min(len(weights), len(w))] = weights[:min(len(weights), len(w))]
                            proj = (w @ comps).tolist()
                            traj_2d.append({"a": proj[0], "b": proj[1]})
                        
                        loss_surface["traj"] = traj_2d
                        loss_surface["loss_history"] = nn_metrics.get("loss_history", [])
                        print(f"Added trajectory with {len(traj_2d)} points")
            except Exception as e:
                print(f"Trajectory projection error: {e}")
                # Loss surface still works without trajectory

    except Exception as e:
        loss_surface = None
        print("Loss surface generation error:", e)

    # NN SVG: prefer real NN weights, otherwise fall back to logistic coefficients
    nn_svg = None
    input_names = [
        "age",
        "weight",
        "height",
        "bmi",
        "sex",
        "dtm",
        "dii",
        "mallampati",
        "stop_bang",
        "alganzouri",
    ]
    
    try:
        if nn_metrics:
            # Use actual NN weights if available (pre-trained JSON works without Torch)
            w1 = nn_metrics.get("w1", [])
            b1 = nn_metrics.get("b1", [])
            w2 = nn_metrics.get("w2", None)
            b2 = nn_metrics.get("b2", None)

            if w1 and b1:
                nn_svg = nn_svg_from_weights(w1, b1, w2, b2, input_names=input_names)
                print("Generated NN SVG from actual NN weights")
        
        # Fallback: visualize logistic regression coefficients as single hidden layer
        if nn_svg is None and log_metrics:
            try:
                model, _ = train_logistic_model(min_samples=20)
                lr = model.named_steps.get("lr")
                if lr is not None:
                    coef = lr.coef_[0].tolist()
                    intercept = [float(lr.intercept_[0])]
                    nn_svg = nn_svg_from_weights([coef], intercept, w2=None, b2=None, input_names=input_names)
                    print("Generated NN SVG from logistic regression fallback")
            except Exception as e:
                print(f"Logistic SVG fallback failed: {e}")
                nn_svg = None
    except Exception as e:
        print(f"NN SVG generation error: {e}")
        import traceback
        traceback.print_exc()
        nn_svg = None

    return render_template(
        "analytics.html",
        log_metrics=log_metrics,
        nn_metrics=nn_metrics,
        nn_error=nn_error,
        loss_surface=loss_surface,
        nn_svg=nn_svg,
        error=error,
        data_source=data_source if log_metrics else "unknown",
    )


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
@login_required
def blog():
    return render_template("blog.html", posts=_blog_posts())


@bp.route("/blog/<slug>")
@login_required
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
                "p": "Cormack–Lehane grading is a pragmatic proxy of laryngeal view. Grades 3–4 often correlate with increased attempts and need for adjuncts. In this project, the binary outcome “difficult” is defined as Cormack > 3 (i.e., grade 4) unless you later choose to broaden that threshold.",
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

# Secret seeding endpoint for Heroku deployment
@bp.route("/seed-db-secret-key-12345", methods=["POST", "GET"])
def seed_database():
    """Secret endpoint to manually seed database - use only for Heroku initial setup"""
    from app.seed_realistic import seed_realistic_patients
    
    try:
        # Clear existing records
        IntubationRecord.query.delete()
        db.session.commit()
        
        # Seed 2000 new records
        seed_realistic_patients(n=2000)
        
        # Get final count
        final_count = IntubationRecord.query.count()
        
        result = {
            "status": "success",
            "message": f"Database seeded successfully! Total records: {final_count}",
            "count": final_count
        }
        return result, 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        result = {
            "status": "error",
            "message": str(e)
        }
        return result, 500
