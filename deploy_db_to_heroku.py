"""Deploy database to Heroku - seeds realistic data to remote database"""
import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a shell command and report status"""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        sys.exit(1)
    print(f"✓ {description} completed")

def main():
    # Get app name from user or use default
    app_name = os.environ.get("HEROKU_APP_NAME", None)
    if not app_name:
        print("Enter your Heroku app name (or set HEROKU_APP_NAME environment variable):")
        app_name = input().strip()
        if not app_name:
            print("ERROR: Heroku app name required")
            sys.exit(1)
    
    print(f"\nDeploying to Heroku app: {app_name}")
    
    # Step 1: Ensure latest code is pushed
    run_command("git push heroku main", "Push code to Heroku")
    
    # Step 2: Run database migrations
    run_command(f"heroku run flask db upgrade -a {app_name}", "Run database migrations")
    
    # Step 3: Seed the database with realistic data
    run_command(
        f"heroku run python -c \"from app import create_app, db; from app.seed_realistic import seed_realistic_patients; app = create_app(); app.app_context().push(); seed_realistic_patients(n=2000)\" -a {app_name}",
        "Seed database with 2000 realistic records"
    )
    
    print("\n" + "="*60)
    print("  ✓ DATABASE DEPLOYMENT COMPLETE")
    print("="*60)
    print(f"\nYour Heroku app '{app_name}' now has 2000 intubation records.")
    print("Visit: https://{}.herokuapp.com".format(app_name))

if __name__ == "__main__":
    main()
