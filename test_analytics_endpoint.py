#!/usr/bin/env python
"""Test the analytics endpoint to see what's being rendered."""

import sys
sys.path.insert(0, 'c:\\Users\\Massimo.Giacon\\intuB')

from app import create_app
from app.models import User
from werkzeug.security import generate_password_hash

app = create_app()

with app.app_context():
    # Create a test user if needed
    from app import db
    
    test_user = User.query.filter_by(email="test@test.com").first()
    if not test_user:
        test_user = User(email="test@test.com", name="test user", password_hash=generate_password_hash("test123"))
        db.session.add(test_user)
        db.session.commit()
        print("✓ Created test user")
    
    # Now test the analytics endpoint
    with app.test_client() as client:
        # Login
        response = client.post('/login', data={
            'email': 'test@test.com',
            'password': 'test123'
        }, follow_redirects=True)
        
        # Get analytics page
        response = client.get('/analytics')
        html = response.get_data(as_text=True)
        
        # Check for SVG
        if '<svg' in html:
            print("✓ SVG found in response")
            
            # Extract the SVG content
            import re
            svg_match = re.search(r'<svg[^>]*>.*?</svg>', html, re.DOTALL)
            if svg_match:
                svg_content = svg_match.group()
                print(f"✓ SVG length: {len(svg_content)} chars")
                
                # Check for clinical parameters
                clinical_params = ["age", "weight", "height", "bmi", "sex", "dtm", "dii", "mallampati", "stop_bang", "alganzouri"]
                found = [p for p in clinical_params if p in svg_content]
                print(f"✓ Clinical parameters found: {found}")
                
                if len(found) < len(clinical_params):
                    missing = [p for p in clinical_params if p not in svg_content]
                    print(f"✗ Missing: {missing}")
        else:
            print("✗ No SVG found in response")
            # Check for error messages
            if 'error' in html.lower():
                print("Error in response:")
                import re
                errors = re.findall(r'<div[^>]*class="[^"]*error[^"]*"[^>]*>([^<]+)</div>', html)
                for err in errors[:5]:
                    print(f"  {err}")
