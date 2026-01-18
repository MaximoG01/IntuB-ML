"""Proper synthetic seeding using reverse-engineered Alganzouri formula"""
from app import create_app, db
from app.models import User, IntubationRecord
from app.ml import _synthetic_seed
import numpy as np


def seed_proper_synthetic(n=2000):
    """Seed database with proper reverse-engineered Alganzouri-based synthetic data"""
    print(f"Seeding {n} properly correlated intubation records...")
    
    app = create_app()
    with app.app_context():
        # Get first user as operator
        user = User.query.first()
        if not user:
            print("ERROR: No user exists. Create a user first.")
            return
        
        operator_id = user.id
        
        # Clear existing records
        IntubationRecord.query.delete()
        db.session.commit()
        print(f"Cleared existing records.")
        
        # Use the synthetic seed function (which has proper correlations)
        X, y = _synthetic_seed(n=n, random_state=42)
        
        feature_names = ['age', 'weight', 'height', 'bmi', 'sex', 'dtm', 'dii', 'mallampati', 'stop_bang', 'alganzouri']
        
        for i in range(n):
            age, weight, height, bmi, sex_code, dtm, dii, mallampati, stop_bang, alganzouri = X[i]
            difficult_binary = bool(y[i] == 1.0)
            
            # Cormack correlated with difficult
            if difficult_binary:
                cormack = np.random.choice([3, 4])
            else:
                cormack = np.random.choice([1, 2])
            
            sex = "M" if sex_code < 0.5 else "F"
            
            rec = IntubationRecord(
                operator_id=operator_id,
                age=float(age),
                weight=float(weight),
                height=float(height),
                sex=sex,
                dtm=float(dtm),
                dii=float(dii),
                mallampati=int(mallampati),
                stop_bang=int(stop_bang),
                alganzouri=int(alganzouri),
                drug_used="Propofol + Rocuronio",
                technique="DL",
                success=not difficult_binary,
                cormack=int(cormack),
                difficult_binary=difficult_binary
            )
            db.session.add(rec)
        
        db.session.commit()
        print(f"SEEDING COMPLETE: {n} records with proper reverse-engineered correlations.")


if __name__ == "__main__":
    seed_proper_synthetic(n=2000)
