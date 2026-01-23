#!/bin/bash
# Deploy database to Heroku - reseeds realistic data to remote database

if [ -z "$1" ]; then
    echo "Usage: ./deploy_db.sh <heroku-app-name>"
    echo "Example: ./deploy_db.sh my-intub-app"
    exit 1
fi

APP_NAME=$1

echo "=========================================="
echo "  HEROKU DATABASE DEPLOYMENT"
echo "=========================================="
echo "App: $APP_NAME"
echo ""

# Step 1: Push code to Heroku
echo "1. Pushing code to Heroku..."
git push heroku main || git push heroku master
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to push to Heroku"
    echo "Make sure you have a 'heroku' git remote configured:"
    echo "  heroku git:remote -a $APP_NAME"
    exit 1
fi

# Step 2: Run migrations
echo ""
echo "2. Running database migrations..."
heroku run flask db upgrade -a $APP_NAME

# Step 3: Seed database
echo ""
echo "3. Seeding database with 2000 realistic records..."
heroku run python << 'EOF' -a $APP_NAME
from app import create_app, db
from app.seed_realistic import seed_realistic_patients

app = create_app()
with app.app_context():
    print("Clearing old records...")
    from app.models import IntubationRecord
    IntubationRecord.query.delete()
    db.session.commit()
    print("Seeding with 2000 realistic records...")
    seed_realistic_patients(n=2000)
    print("✓ Database deployment complete!")
EOF

echo ""
echo "=========================================="
echo "  ✓ DEPLOYMENT COMPLETE"
echo "=========================================="
echo "Your Heroku database now has 2000 intubation records"
