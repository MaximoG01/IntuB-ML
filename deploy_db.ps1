# Deploy database to Heroku - reseeds realistic data to remote database
# Usage: .\deploy_db.ps1 -AppName "my-intub-app"

param(
    [Parameter(Mandatory=$true)]
    [string]$AppName
)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  HEROKU DATABASE DEPLOYMENT" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "App: $AppName" -ForegroundColor Yellow
Write-Host ""

# Check if git remote exists
$remoteExists = git remote | Select-String "heroku"
if (-not $remoteExists) {
    Write-Host "Setting up Heroku git remote..." -ForegroundColor Yellow
    & heroku git:remote -a $AppName
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to set up Heroku remote" -ForegroundColor Red
        exit 1
    }
}

# Step 1: Push code to Heroku
Write-Host "1. Pushing code to Heroku..." -ForegroundColor Yellow
& git push heroku main 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  (Trying master branch...)" -ForegroundColor Gray
    & git push heroku master
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to push to Heroku" -ForegroundColor Red
        exit 1
    }
}
Write-Host "  ✓ Code pushed" -ForegroundColor Green

# Step 2: Run migrations
Write-Host ""
Write-Host "2. Running database migrations..." -ForegroundColor Yellow
& heroku run "flask db upgrade" -a $AppName
Write-Host "  ✓ Migrations completed" -ForegroundColor Green

# Step 3: Seed database
Write-Host ""
Write-Host "3. Seeding database with 2000 realistic records..." -ForegroundColor Yellow

$seedScript = @"
from app import create_app, db
from app.seed_realistic import seed_realistic_patients

app = create_app()
with app.app_context():
    from app.models import IntubationRecord
    print('Clearing old records...')
    IntubationRecord.query.delete()
    db.session.commit()
    print('Seeding with 2000 realistic records...')
    seed_realistic_patients(n=2000)
    print('✓ Database seeded successfully!')
"@

& heroku run "python -c `"$seedScript`"" -a $AppName

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  ✓ DATABASE DEPLOYMENT COMPLETE" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Your Heroku database now has 2000 intubation records" -ForegroundColor Green
