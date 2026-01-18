# app/forms.py
from __future__ import annotations

from flask_wtf import FlaskForm
from wtforms import (
    BooleanField,
    StringField,
    PasswordField,
    SubmitField,
    IntegerField,
    FloatField,
    SelectField,
)
from wtforms.validators import DataRequired, Email, EqualTo, NumberRange, Optional, Length


class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email(), Length(max=255)])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")


class RegistrationForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired(), Length(max=128)])
    email = StringField("Email", validators=[DataRequired(), Email(), Length(max=255)])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=6)])
    password2 = PasswordField(
        "Repeat password",
        validators=[DataRequired(), EqualTo("password", message="Passwords must match.")],
    )
    submit = SubmitField("Register")


class PredictForm(FlaskForm):
    age = IntegerField("Age", validators=[DataRequired(), NumberRange(min=0, max=120)])
    weight = FloatField("Weight (kg)", validators=[DataRequired(), NumberRange(min=30, max=250)])
    height = FloatField("Height (cm)", validators=[Optional(), NumberRange(min=50, max=250)])

    sex = SelectField(
        "Sex",
        choices=[("M", "M"), ("F", "F"), ("O", "O/Unknown")],
        validators=[Optional()],
        default="O",
    )

    dtm = FloatField("DTM (cm)", validators=[Optional(), NumberRange(min=2, max=15)])
    dii = FloatField("DII (cm)", validators=[Optional(), NumberRange(min=1, max=10)])

    mallampati = IntegerField("Mallampati (1-4)", validators=[Optional(), NumberRange(min=0, max=4)])
    stop_bang = IntegerField("STOP-BANG (0-8)", validators=[Optional(), NumberRange(min=0, max=8)])
    alganzouri = IntegerField("Al-Ganzouri (0-12)", validators=[Optional(), NumberRange(min=0, max=12)])

    save_pending = BooleanField("Save as pending case (enter outcome later)")

    submit = SubmitField("Predict")


class LabelOutcomeForm(FlaskForm):
    cormack = IntegerField("Cormack–Lehane (1-4)", validators=[DataRequired(), NumberRange(min=1, max=4)])
    success = SelectField(
        "Intubation success",
        choices=[("1", "Success"), ("0", "Failure")],
        validators=[DataRequired()],
    )
    difficult_binary = SelectField(
        "Difficult (binary)",
        choices=[("1", "Difficult"), ("0", "Not difficult")],
        validators=[DataRequired()],
    )
    submit = SubmitField("Save outcome")
