# QUICK REFERENCE - Heroku Deployment

## TL;DR - Deploy to Heroku

```bash
# 1. Set up Heroku (one-time)
heroku login
heroku git:remote -a <your-app-name>

# 2. Push code
git push heroku main

# 3. Deploy database (seeds 2000 records + runs migrations)
./deploy_db.sh <your-app-name>

# Done! Visit: https://<your-app-name>.herokuapp.com
```

## How It Works

| Component | Local | Heroku |
|-----------|-------|--------|
| PyTorch | ✅ Installed | ❌ Not installed |
| Train NN | ✅ Yes | ❌ No |
| Load Model | ✅ Trained or Pre-trained | ✅ Pre-trained (JSON) |
| Database | SQLite | PostgreSQL |
| Data | All 2000 records | All 2000 records |

## Update Trained Model

```bash
# 1. Retrain locally
python train_and_save_model.py

# 2. Commit
git add app/data/trained_model.json
git commit -m "Update trained model"

# 3. Deploy
git push heroku main

# Done! Heroku loads new weights
```

## File Locations

- 📁 Pre-trained model: `app/data/trained_model.json`
- 📄 Training script: `train_and_save_model.py`
- 📄 Deploy script: `deploy_db.sh`
- 📚 Full guides: `SOLUTION_COMPLETE.md`, `DEPLOY_TO_HEROKU.md`

## Check Deployment Status

```bash
# View logs
heroku logs --tail -a <your-app-name>

# Check database
heroku run python << 'EOF' -a <your-app-name>
from app import create_app, db
from app.models import IntubationRecord
app = create_app()
with app.app_context():
    count = IntubationRecord.query.count()
    print(f"Database has {count} records")
EOF

# Test analytics page
# Visit: https://<your-app-name>.herokuapp.com/analytics
```

## Model Metrics

```
Training Accuracy:  97.40%
Validation AUC:     0.9980
Brier Score:        0.0192
Training Samples:   1500
Validation Samples: 500
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "heroku not found" | Install Heroku CLI or use Dashboard |
| "Could not push" | Run `heroku git:remote -a <app-name>` |
| "Database error" | Run `./deploy_db.sh <app-name>` |
| "NN not loading" | Check `app/data/trained_model.json` exists |
| "2000 records missing" | Rerun `./deploy_db.sh <app-name>` |

## Your Heroku App URL

Once deployed, your app is at:
```
https://<your-app-name>.herokuapp.com
Analytics:
https://<your-app-name>.herokuapp.com/analytics
```

## Key Points

✅ **2000 records**: Already seeded via `deploy_db.sh`
✅ **Pre-trained model**: Committed as JSON (no PyTorch needed)
✅ **Automatic updates**: Commit → Git → Heroku → Live
✅ **Fast deployment**: 3-5 minutes for first time, <1 min for updates

🎉 **Your Heroku app is production-ready!**
