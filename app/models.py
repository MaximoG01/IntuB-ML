
from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db, login_manager

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    name = db.Column(db.String(120), nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
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

    dtm = db.Column(db.Float)                 # distanza tiromentale
    dii = db.Column(db.Float)                 # distanza interincisiva
    mallampati = db.Column(db.Integer)
    stop_bang = db.Column(db.Integer)
    alganzouri = db.Column(db.Integer)

    drug_used = db.Column(db.String(128))     # lasciati liberi / raw
    technique = db.Column(db.String(128))

    # esito – possono essere vuoti se il caso è “pending”
    success = db.Column(db.Boolean, nullable=True)
    cormack = db.Column(db.Integer, nullable=True)
    difficult_binary = db.Column(db.Boolean, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<IntubationRecord {self.id}>"
