# seed_synthetic.py
import random
from app import create_app, db
from app.models import IntubationRecord, User

N = 100  # quante righe vuoi creare (almeno 50)

def generate_patient():
    age = random.randint(18, 90)
    weight = random.randint(50, 120)
    height = random.randint(150, 190)

    dtm = round(random.uniform(5.0, 8.0), 1)
    dii = round(random.uniform(3.0, 5.0), 1)
    mallampati = random.randint(1, 4)
    stop_bang = random.randint(0, 8)
    alganzouri = random.randint(0, 12)

    sex = random.choice(["M", "F"])

    drug_used = random.choice(["Propofol+Roc", "Etomidate+Roc", "Thiopental+Sux"])
    technique = random.choice(["DL", "VL", "VL+Bougie", "DL+Bougie"])

    # rischio empirico un po' “sensato”
    base_risk = 0.05
    base_risk += 0.05 * (mallampati - 1)
    if stop_bang >= 4:
        base_risk += 0.05
    if dtm < 6.0:
        base_risk += 0.05

    difficult = random.random() < min(base_risk, 0.8)

    # outcome
    if difficult:
        cormack = random.choice([3, 4])
        success = random.random() > 0.3  # 70% dei difficult riescono lo stesso
    else:
        cormack = random.choice([1, 2])
        success = True

    difficult_binary = (cormack >= 3) or (not success)

    return dict(
        age=age,
        weight=weight,
        height=height,
        sex=sex,
        dtm=dtm,
        dii=dii,
        mallampati=mallampati,
        stop_bang=stop_bang,
        alganzouri=alganzouri,
        drug_used=drug_used,
        technique=technique,
        success=success,
        cormack=cormack,
        difficult_binary=difficult_binary,
    )


def main():
    app = create_app()
    with app.app_context():
        # prendi un utente come operator, o creane uno fittizio
        user = User.query.first()
        if not user:
            user = User(
                email="seed@example.com",
                name="Seeder",
                password_hash="dummy",  # non usato per login
            )
            db.session.add(user)
            db.session.commit()

        for i in range(N):
            data = generate_patient()
            rec = IntubationRecord(operator_id=user.id, **data)
            db.session.add(rec)
            if (i + 1) % 20 == 0:
                print(f"Inserting {i+1}/{N} records...")
                db.session.flush()

        db.session.commit()
        print(f"Inserted {N} synthetic patients.")


if __name__ == "__main__":
    main()
