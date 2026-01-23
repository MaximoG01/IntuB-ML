
from app import create_app, db
import os

app = create_app()

# Auto-seed on Heroku if database is empty
if os.environ.get('DYNO'):  # Running on Heroku
    with app.app_context():
        from app.models import IntubationRecord
        record_count = IntubationRecord.query.count()
        
        if record_count < 2000:
            print("\n[STARTUP] Seeding database with 2000 records...")
            try:
                from app.seed_realistic import seed_realistic_patients
                IntubationRecord.query.delete()
                db.session.commit()
                seed_realistic_patients(n=2000)
                print(f"[STARTUP] ✓ Database seeded successfully!")
            except Exception as e:
                print(f"[STARTUP] Error during seeding: {e}")

if __name__ == "__main__":
    app.run()
