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
    invite_code = StringField("Codice invito", validators=[Optional(), Length(max=64)])
    submit = SubmitField("Register")


class CustomFieldForm(FlaskForm):
    """Admin: define a new pre-intubation variable (number with unit, or choice from a list)."""
    label = StringField("Nome della variabile", validators=[DataRequired(), Length(max=80)])
    key = StringField("Nome breve per l'export (lettere minuscole, numeri, _)", validators=[DataRequired(), Length(max=40)])
    kind = SelectField("Tipo", choices=[("number", "Numero con unità"), ("choice", "Scelta da lista")])
    unit = StringField("Unità (es. cm)", validators=[Optional(), Length(max=16)])
    min_value = FloatField("Minimo ammesso", validators=[Optional()])
    max_value = FloatField("Massimo ammesso", validators=[Optional()])
    choices = StringField("Valori ammessi, separati da | (es. I|II|III)", validators=[Optional(), Length(max=400)])
    help_text = StringField("Come si misura (facoltativo)", validators=[Optional(), Length(max=200)])
    submit = SubmitField("Aggiungi variabile")

    def validate(self, extra_validators=None):
        if not super().validate(extra_validators):
            return False
        import re
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.key.data or ""):
            self.key.errors.append("Usa solo lettere minuscole, numeri e _, iniziando con una lettera.")
            return False
        if self.kind.data == "choice":
            opts = [c.strip() for c in (self.choices.data or "").split("|") if c.strip()]
            if len(opts) < 2:
                self.choices.errors.append("Servono almeno due valori, separati da |.")
                return False
            self.choices.data = "|".join(opts)
        elif (self.min_value.data is not None and self.max_value.data is not None
              and self.min_value.data >= self.max_value.data):
            self.max_value.errors.append("Il massimo deve essere maggiore del minimo.")
            return False
        return True


class ToggleForm(FlaskForm):
    """Empty form: only carries the CSRF token for admin buttons."""


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
