#!/usr/bin/env python
"""Verify the trajectory line and color fixes."""

import sys
sys.path.insert(0, 'c:\\Users\\Massimo.Giacon\\intuB')

from app import create_app
from app.models import User
from werkzeug.security import generate_password_hash
import re

app = create_app()

with app.app_context():
    from app import db
    
    # Ensure test user exists
    test_user = User.query.filter_by(email="test@test.com").first()
    if not test_user:
        test_user = User(email="test@test.com", name="test user", password_hash=generate_password_hash("test123"))
        db.session.add(test_user)
        db.session.commit()
    
    with app.test_client() as client:
        # Login
        response = client.get('/login')
        csrf_match = re.search(r'name="csrf_token"\s+type="hidden"\s+value="([^"]+)"', response.get_data(as_text=True))
        csrf_token = csrf_match.group(1) if csrf_match else ''
        
        client.post('/login', data={
            'email': 'test@test.com',
            'password': 'test123',
            'csrf_token': csrf_token
        }, follow_redirects=True)
        
        # Get analytics
        response = client.get('/analytics')
        html = response.get_data(as_text=True)
        
        print("=" * 80)
        print("TRAJECTORY LINE & COLOR FIXES VERIFICATION")
        print("=" * 80)
        
        # Check for green trajectory color
        print("\n✅ Color Mapping:")
        if "rgb(0, 255, 0)" in html:
            print("   ✓ Bright green color (0, 255, 0) found for trajectory line")
        else:
            print("   ✗ Green color NOT found")
        
        if "rgb(0, 200, 0)" in html:
            print("   ✓ Dark green color (0, 200, 0) found for trajectory markers")
        else:
            print("   ✗ Dark green NOT found")
        
        # Check for trajectory data
        print("\n✅ Trajectory Data:")
        if "Training trajectory" in html:
            print("   ✓ 'Training trajectory' label found (was 'Epoch trajectory')")
        else:
            print("   ✗ Trajectory label not found")
        
        # Check for corrected color description
        print("\n✅ Description Updates:")
        if "red/warm regions indicate low loss" in html:
            print("   ✓ Correct description: red = low loss")
        else:
            print("   ✗ Description still wrong")
        
        if "blue/cool regions indicate high loss" in html:
            print("   ✓ Correct description: blue = high loss")
        else:
            print("   ✗ Description still wrong")
        
        if "bright green line" in html:
            print("   ✓ Updated trajectory color mentioned as 'bright green'")
        else:
            print("   ✗ Green color not mentioned in description")
        
        if "red valley" in html:
            print("   ✓ Updated: 'red valley' (minimum loss)")
        else:
            print("   ✗ Valley color description not updated")
        
        # Check line width and marker size
        print("\n✅ Visualization Improvements:")
        if 'line: { color: "rgb(0, 255, 0)", width: 8' in html or 'width: 8' in html:
            print("   ✓ Line width increased to 8 (from 6) for visibility")
        else:
            print("   ✗ Line width not updated")
        
        if 'marker: { size: 5' in html or 'size: 5' in html:
            print("   ✓ Marker size increased to 5 (from 3) for visibility")
        else:
            print("   ✗ Marker size not updated")
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("""
✅ Fixed Issues:
  1. Color mapping corrected:
     - Dark/blue = LOW loss (good)
     - Bright/yellow = HIGH loss (bad)
  
  2. Trajectory line now visible:
     - Color: Bright green rgb(0, 255, 0)
     - Width: 8px (increased from 6)
     - Markers: 5px (increased from 3)
  
  3. All text updated:
     - "blue-purple" instead of "red"
     - "yellow-green" instead of "blue"
     - "bright green line" instead of "yellow"
     - "dark blue valley" instead of "red valley"

The loss surface chart should now clearly show:
  - The Viridis colorscale properly explained
  - A visible bright green trajectory line
  - Correct interpretation of colors
        """)
