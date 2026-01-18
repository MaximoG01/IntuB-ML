#!/usr/bin/env python
"""Test if trajectory line appears in analytics page."""

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
        
        print("=" * 70)
        print("TRAJECTORY LINE TEST")
        print("=" * 70)
        
        # Check for trajectory data
        checks = {
            "Bright green color": "rgb(0, 255, 0)" in html,
            "Dark green markers": "rgb(0, 200, 0)" in html,
            "Training trajectory label": "Training trajectory" in html,
            "Trajectory with if condition": "if (LS.traj)" in html,
            "Red valley (low loss)": "red valley" in html,
            "Correct axis description": "red/warm regions indicate low loss" in html,
        }
        
        print("\n✅ Checks:")
        for check, result in checks.items():
            status = "✓" if result else "✗"
            print(f"  {status} {check}")
        
        # Extract loss surface data to see if traj is present
        traj_match = re.search(r'"traj":\s*\[', html)
        if traj_match:
            print("\n✓ Trajectory data found in loss_surface!")
        else:
            print("\n✗ Trajectory data NOT found in loss_surface")
        
        # Check for trajectory points
        traj_points = re.findall(r'"a":\s*([-\d.]+),\s*"b":\s*([-\d.]+)', html)
        if traj_points:
            print(f"✓ Found {len(traj_points)} trajectory points")
            print(f"  First point: a={traj_points[0][0]}, b={traj_points[0][1]}")
            print(f"  Last point: a={traj_points[-1][0]}, b={traj_points[-1][1]}")
        else:
            print("✗ No trajectory points found")
        
        print("\n" + "=" * 70)
        all_good = all(checks.values()) and traj_points
        if all_good:
            print("✅ TRAJECTORY LINE SHOULD BE VISIBLE!")
        else:
            print("⚠️  Some components missing - trajectory may not show")
        print("=" * 70)
