
from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db, login_manager

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    name = db.Column(db.String(128), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)  # <--- PIÙ LUNGO
    is_admin = db.Column(db.Boolean, nullable=False, default=False, server_default=db.false())
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    records = db.relationship("IntubationRecord", backref="operator", lazy="dynamic")

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def __repr__(self) -> str:
        return f"<User {self.email}>"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


from datetime import datetime
from app import db

class IntubationRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    operator_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    # fattori pre-intubazione
    age = db.Column(db.Integer, nullable=False)
    weight = db.Column(db.Float, nullable=False)
    height = db.Column(db.Float)              # cm, può essere None
    sex = db.Column(db.String(1))             # 'M', 'F', 'O' (altro/ignoto)

    asa = db.Column(db.Integer)               # ASA 1-5
    emergency = db.Column(db.Boolean)         # intervento urgente

    # item El-Ganzouri (app/egri.py)
    dtm = db.Column(db.Float)                 # distanza tiromentoniera, cm
    dii = db.Column(db.Float)                 # distanza interincisiva (apertura bocca), cm
    mallampati = db.Column(db.Integer)        # 1-4
    neck_movement = db.Column(db.Integer)     # 0: >90°, 1: 80-90°, 2: <80°
    can_prognath = db.Column(db.Boolean)      # protrusione mandibolare possibile
    history_difficult = db.Column(db.Integer) # 0 no, 1 dubbia, 2 certa
    stop_bang = db.Column(db.Integer)
    alganzouri = db.Column(db.Integer)        # totale EGRI calcolato dagli item (non inserito a mano)

    drug_used = db.Column(db.String(128))     # lasciati liberi / raw
    technique = db.Column(db.String(128))

    # predizione al momento dell'inserimento
    predicted_proba = db.Column(db.Float)
    model_version = db.Column(db.String(64))

    # esito – possono essere vuoti se il caso è “pending”
    success = db.Column(db.Boolean, nullable=True)
    cormack = db.Column(db.Integer, nullable=True)
    cormack_device = db.Column(db.String(16))  # 'DL' (lama diretta) o 'VL' (videolaringoscopio)
    difficult_binary = db.Column(db.Boolean, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<IntubationRecord {self.id}>"


class CustomField(db.Model):
    """
    Extra pre-intubation variable defined by an admin (e.g. neck circumference, upper lip bite test).
    Only numbers with unit and choices from a list: no free text, so no identifying data can be typed in.
    """
    KINDS = ("number", "choice")

    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(40), unique=True, nullable=False)  # column name in the export
    label = db.Column(db.String(80), nullable=False)
    kind = db.Column(db.String(10), nullable=False)
    unit = db.Column(db.String(16))
    min_value = db.Column(db.Float)
    max_value = db.Column(db.Float)
    choices = db.Column(db.String(400))  # "I|II|III" for kind == "choice"
    help_text = db.Column(db.String(200))
    active = db.Column(db.Boolean, nullable=False, default=True)
    created_by_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def choice_list(self):
        return [c for c in (self.choices or "").split("|") if c]

    def parse(self, raw: str | None):
        """Form text -> stored value (float or choice), None if empty. Raises ValueError if invalid."""
        raw = (raw or "").strip()
        if not raw:
            return None
        if self.kind == "number":
            v = float(raw.replace(",", "."))
            if (self.min_value is not None and v < self.min_value) or (self.max_value is not None and v > self.max_value):
                lo = "" if self.min_value is None else f"{self.min_value:g}"
                hi = "" if self.max_value is None else f"{self.max_value:g}"
                raise ValueError(f"{self.label}: valore fuori dall'intervallo {lo}–{hi} {self.unit or ''}".strip() + ".")
            return v
        if raw not in self.choice_list:
            raise ValueError(f"{self.label}: scelta non valida.")
        return raw


class CustomValue(db.Model):
    """Value of a CustomField for one case. A missing value is a missing row, never 0."""
    id = db.Column(db.Integer, primary_key=True)
    record_id = db.Column(db.Integer, db.ForeignKey("intubation_record.id", ondelete="CASCADE"), nullable=False)
    field_id = db.Column(db.Integer, db.ForeignKey("custom_field.id"), nullable=False)
    value_num = db.Column(db.Float)
    value_choice = db.Column(db.String(60))
    __table_args__ = (db.UniqueConstraint("record_id", "field_id"),)

    field = db.relationship("CustomField")
    record = db.relationship("IntubationRecord", backref=db.backref("custom_values", cascade="all, delete-orphan"))

    @property
    def value(self):
        return self.value_num if self.value_num is not None else self.value_choice
