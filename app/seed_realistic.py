import random
from app import db
from app.models import IntubationRecord, User


def seed_realistic_patients(n=2000):
    print(f"Seeding {n} realistic synthetic intubation records...")

    # Usa il primo utente esistente come operatore
    user = User.query.first()
    operator_id = user.id if user else 1

    for _ in range(n):
        age = random.randint(18, 90)
        weight = random.uniform(50, 120)
        height = random.uniform(150, 195)
        sex = random.choice(["M", "F"])

        dtm = random.uniform(4, 8)
        dii = random.uniform(2.5, 5.5)
        mallampati = random.randint(1, 4)
        stop_bang = random.randint(0, 8)

        # Alganzouri loosely correlated
        alganzouri = (
            (4 - mallampati) * 2 +
            (8 - stop_bang) +
            random.randint(-1, 2)
        )
        alganzouri = max(0, min(20, alganzouri))

        # realistic Cormack distribution
        cormack = random.choices(
            [1, 2, 3, 4],
            weights=[0.55, 0.30, 0.10, 0.05],
            k=1
        )[0]

        difficult = cormack > 2

        rec = IntubationRecord(
            operator_id=operator_id,
            age=age,
            weight=weight,
            height=height,
            sex=sex,
            dtm=dtm,
            dii=dii,
            mallampati=mallampati,
            stop_bang=stop_bang,
            alganzouri=alganzouri,
            drug_used="Propofol + Rocuronio",
            technique="DL",
            success=not difficult,
            cormack=cormack,
            difficult_binary=difficult
        )

        db.session.add(rec)

    db.session.commit()
    print("SEEDING COMPLETE.")
