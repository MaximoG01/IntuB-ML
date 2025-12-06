from flask_wtf import FlaskForm
from wtforms import (
    StringField,
    IntegerField,
    FloatField,
    BooleanField,
    SubmitField,
    SelectField,
    PasswordField,
)
from wtforms.validators import DataRequired, NumberRange, Optional, Email, Length

class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Log in")


class RegisterForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired(), Length(max=120)])
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=6)])
    submit = SubmitField("Register")


class IntubationForm(FlaskForm):
    age = IntegerField("Età", validators=[DataRequired(), NumberRange(min=0, max=120)])
    weight = FloatField("Peso (kg)", validators=[DataRequired(), NumberRange(min=1, max=300)])

    height = FloatField("Altezza (cm)", validators=[Optional(), NumberRange(min=50, max=250)])
    sex = SelectField(
        "Sesso",
        choices=[("M", "Maschio"), ("F", "Femmina"), ("O", "Altro / Non specificato")],
        validators=[Optional()],
    )

    dtm = FloatField("Distanza tiromentale (cm)", validators=[Optional(), NumberRange(min=0, max=20)])
    dii = FloatField("Distanza interincisiva (cm)", validators=[Optional(), NumberRange(min=0, max=10)])
    mallampati = IntegerField("Mallampati (1–4)", validators=[Optional(), NumberRange(min=1, max=4)])
    stop_bang = IntegerField("STOP-BANG (0–8)", validators=[Optional(), NumberRange(min=0, max=8)])
    alganzouri = IntegerField(
        "Al-Ganzouri risk index (0–12)",
        description="0–1 molto basso, 2–4 basso, 5–8 moderato, 9–12 alto rischio",
        validators=[Optional(), NumberRange(min=0, max=12)],
    )

    drug_used = StringField("Farmaci (libero)", validators=[Optional()])
    technique = StringField("Tecnica (libero)", validators=[Optional()])

    success = BooleanField("Intubazione riuscita?")
    cormack = IntegerField("Cormack–Lehane (1–4)", validators=[DataRequired(), NumberRange(min=1, max=4)])

    submit = SubmitField("Salva record completo")

class PredictionForm(FlaskForm):
    age = IntegerField("Età", validators=[DataRequired(), NumberRange(min=0, max=120)])
    weight = FloatField("Peso (kg)", validators=[DataRequired(), NumberRange(min=1, max=300)])
    height = FloatField("Altezza (cm)", validators=[Optional(), NumberRange(min=50, max=250)])
    sex = SelectField(
        "Sesso",
        choices=[("M", "Maschio"), ("F", "Femmina"), ("O", "Altro / Non specificato")],
        validators=[Optional()],
    )

    dtm = FloatField("Distanza tiromentale (cm)", validators=[Optional(), NumberRange(min=0, max=20)])
    dii = FloatField("Distanza interincisiva (cm)", validators=[Optional(), NumberRange(min=0, max=10)])
    mallampati = IntegerField("Mallampati (1–4)", validators=[Optional(), NumberRange(min=1, max=4)])
    stop_bang = IntegerField("STOP-BANG (0–8)", validators=[Optional(), NumberRange(min=0, max=8)])
    alganzouri = IntegerField(
        "Al-Ganzouri risk index (0–12)",
        validators=[Optional(), NumberRange(min=0, max=12)],
    )

    drug_used = StringField("Farmaci (opzionale)", validators=[Optional()])
    technique = StringField("Tecnica (opzionale)", validators=[Optional()])

    save_case = BooleanField("Salva questo caso per completare l'esito dopo l'intubazione")

    submit = SubmitField("Calcola predizione")

class OutcomeForm(FlaskForm):
    success = BooleanField("Intubazione riuscita?")
    cormack = IntegerField("Cormack–Lehane (1–4)", validators=[DataRequired(), NumberRange(min=1, max=4)])
    submit = SubmitField("Salva esito")
