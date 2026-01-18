#!/usr/bin/env python
"""Debug the analytics endpoint with proper CSRF handling."""

import sys
sys.path.insert(0, 'c:\\Users\\Massimo.Giacon\\intuB')

from app import create_app
from app.models import User
from werkzeug.security import generate_password_hash
import re

app = create_app()

with app.app_context():
    from app import db
    
    test_user = User.query.filter_by(email="test@test.com").first()
    if not test_user:
        test_user = User(email="test@test.com", name="test user", password_hash=generate_password_hash("test123"))
        db.session.add(test_user)
        db.session.commit()
    
    with app.test_client() as client:
        # First GET the login page to get CSRF token
        response = client.get('/login')
        html = response.get_data(as_text=True)
        
        # Extract CSRF token
        csrf_match = re.search(r'name="csrf_token"\s+type="hidden"\s+value="([^"]+)"', html)
        if csrf_match:
            csrf_token = csrf_match.group(1)
            print(f"✓ Got CSRF token: {csrf_token[:20]}...")
        else:
            print("✗ Could not find CSRF token")
            csrf_token = None
        
        # Login with CSRF token
        response = client.post('/login', data={
            'email': 'test@test.com',
            'password': 'test123',
            'csrf_token': csrf_token if csrf_token else ''
        }, follow_redirects=True)
        
        print(f"Login response status: {response.status_code}")
        html = response.get_data(as_text=True)
        
        if 'dashboard' in html.lower() or 'analytics' in html.lower():
            print("✓ Successfully logged in")
        else:
            print("✗ Login failed")
            # Try to show reason
            if 'error' in html.lower():
                errors = re.findall(r'<[^>]*class="[^"]*error[^"]*"[^>]*>([^<]+)<', html, re.IGNORECASE)
                print(f"Errors: {errors[:2]}")
        
        # Now try analytics
        print("\n--- Trying Analytics ---")
        response = client.get('/analytics')
        print(f"Analytics response status: {response.status_code}")
        
        if response.status_code == 302:
            print("Redirecting to:", response.location)
        
        html = response.get_data(as_text=True)
        
        if '<svg' in html:
            print("✓ SVG found in response")
            
            # Check for clinical parameters
            clinical_params = ["age", "weight", "height", "bmi", "sex", "dtm", "dii", "mallampati", "stop_bang", "alganzouri"]
            found = [p for p in clinical_params if p in html]
            print(f"✓ Clinical parameters in SVG: {found}")
            
            missing = [p for p in clinical_params if p not in html]
            if missing:
                print(f"✗ Missing: {missing}")
        else:
            print("✗ No SVG in response")
            print(f"Response length: {len(html)} chars")
            
            # Show a sample
            if len(html) < 500:
                print("\nFull response:")
                print(html)
