#!/usr/bin/env python
"""Debug the analytics endpoint with proper login."""

import sys
sys.path.insert(0, 'c:\\Users\\Massimo.Giacon\\intuB')

from app import create_app
from app.models import User
from werkzeug.security import generate_password_hash

app = create_app()

with app.app_context():
    from app import db
    
    test_user = User.query.filter_by(email="test@test.com").first()
    if not test_user:
        test_user = User(email="test@test.com", name="test user", password_hash=generate_password_hash("test123"))
        db.session.add(test_user)
        db.session.commit()
    
    with app.test_client() as client:
        # Login with follow_redirects
        response = client.post('/login', data={
            'email': 'test@test.com',
            'password': 'test123'
        }, follow_redirects=True)
        
        print(f"Login response status: {response.status_code}")
        html = response.get_data(as_text=True)
        if 'dashboard' in html.lower():
            print("✓ Logged in to dashboard")
        elif 'login' in html.lower():
            print("✗ Still on login page")
            print("\nFirst 500 chars:")
            print(html[:500])
        
        # Try to get analytics with current client
        response = client.get('/analytics', follow_redirects=True)
        print(f"\nAnalytics response status: {response.status_code}")
        
        html = response.get_data(as_text=True)
        print(f"Response length: {len(html)} chars")
        
        # Check for SVG
        if '<svg' in html:
            print("✓ SVG found")
            # Count text nodes with clinical params
            import re
            clinical_params = ["age", "weight", "height", "bmi", "sex", "dtm", "dii", "mallampati", "stop_bang", "alganzouri"]
            found = [p for p in clinical_params if p in html]
            print(f"✓ Found clinical parameters: {found}")
        else:
            print("✗ No SVG found")
            # Check for errors
            if 'error' in html.lower():
                print("\nErrors found:")
                errors = re.findall(r'error[:\s]*([^<]+)', html, re.IGNORECASE)
                for err in errors[:3]:
                    print(f"  {err}")
