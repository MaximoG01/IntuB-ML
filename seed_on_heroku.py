#!/usr/bin/env python
"""
Heroku release phase script - runs migrations and seeds database on first deploy
Add to Procfile as: release: python seed_on_heroku.py
"""

import os
import sys

def seed_heroku():
    print("=" * 50)
    print("HEROKU DATABASE INITIALIZATION")
    print("=" * 50)
    
    from app import create_app, db
    from app.models import IntubationRecord
    from app.seed_realistic import seed_realistic_patients
    
    app = create_app()
    
    with app.app_context():
        print("\n1. Running migrations...")
        os.system("flask db upgrade")
        
        print("\n2. Checking current record count...")
        current_count = IntubationRecord.query.count()
        print(f"   Current records: {current_count}")
        
        if current_count < 2000:
            print("\n3. Clearing old records...")
            IntubationRecord.query.delete()
            db.session.commit()
            
            print("4. Seeding 2000 realistic records...")
            seed_realistic_patients(n=2000)
            
            final_count = IntubationRecord.query.count()
            print(f"\n✓ SUCCESS: Database now has {final_count} records")
        else:
            print(f"\n✓ Database already has {current_count} records (>= 2000)")
    
    print("=" * 50)
    print("DATABASE INITIALIZATION COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    try:
        seed_heroku()
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
