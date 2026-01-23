# DEPLOY_TO_HEROKU.md - Step-by-Step Instructions

## Current Status ✓

✅ Code is committed to GitHub (main branch)
✅ Pre-trained NN model is committed (app/data/trained_model.json)
✅ Deployment script is ready (deploy_db.sh)

## Next Steps: Deploy to Heroku

### 1. Set Up Heroku Git Remote (One-time)

```bash
# First, log in to Heroku
heroku login

# Find your Heroku app name
heroku apps

# Add Heroku as a git remote
heroku git:remote -a <your-heroku-app-name>
# Example:
heroku git:remote -a my-intub-app
```

### 2. Push to Heroku

```bash
cd c:\Users\Massimo.Giacon\intuB
git push heroku main
```

This will:
- Push your code to Heroku
- Install dependencies
- Start the dyno

### 3. Deploy Database & Seed Data

```bash
# Run migrations
heroku run flask db upgrade -a <your-heroku-app-name>

# Seed 2000 realistic records
heroku run python << 'EOF' -a <your-heroku-app-name>
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
```

Or use the bash script:
```bash
./deploy_db.sh <your-heroku-app-name>
```

### 4. Verify Deployment

```bash
# Check dyno logs
heroku logs --tail -a <your-heroku-app-name>

# Open your app
heroku open -a <your-heroku-app-name>

# Test analytics page
# Visit: https://<your-heroku-app>.herokuapp.com/analytics
```

## What Heroku Will Use

When deployed, your Heroku app will:

1. **Database**: PostgreSQL on Heroku (seeded with 2000 records)
2. **NN Model**: Pre-trained model from `app/data/trained_model.json` (no PyTorch training)
3. **Backend**: Flask with scikit-learn (no heavy PyTorch dependency)

## After First Deployment

To update the Heroku app with new code:

```bash
# After making local changes
git add .
git commit -m "Your changes"
git push origin main      # Push to GitHub
git push heroku main      # Push to Heroku (auto-deploys)
```

## Important Notes

⚠️ **DO NOT commit Python packages locally**
- Heroku installs from `requirements.txt`
- PyTorch is NOT in requirements.txt (it's too heavy)
- The pre-trained model weights are JSON (small, already committed)

⚠️ **First deployment takes 3-5 minutes**
- Building slug
- Installing dependencies
- Initializing database

✅ **Database seeding is one-time**
- 2000 records take ~1-2 minutes to seed
- After that, your Heroku app is ready to use

## Your Heroku App Name

If you haven't created one, use Heroku dashboard:
```
https://dashboard.heroku.com/new-app
```

Example app name: `my-intub-app`

Then run:
```bash
heroku git:remote -a my-intub-app
git push heroku main
```

## Troubleshooting

**"Could not read from remote repository"**
- Make sure you've run `heroku git:remote -a <app-name>`

**"requirements.txt: No such file or directory"**
- Already exists, Heroku will find it

**"PyTorch not available" error on Heroku**
- This is EXPECTED
- The app automatically uses pre-trained model
- Check app works at: `https://<your-app>.herokuapp.com/analytics`

**Database migration fails**
- Run: `heroku run flask db upgrade --noinput -a <your-app>`
- Then re-seed with the deploy script
