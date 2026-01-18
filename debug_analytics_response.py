#!/usr/bin/env python
"""Debug the analytics endpoint."""

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
        # Login
        response = client.post('/login', data={
            'email': 'test@test.com',
            'password': 'test123'
        }, follow_redirects=True)
        
        print(f"Login response status: {response.status_code}")
        
        # Get analytics page
        response = client.get('/analytics')
        print(f"Analytics response status: {response.status_code}")
        
        html = response.get_data(as_text=True)
        print(f"Response length: {len(html)} chars")
        
        # Show first 2000 chars
        print("\nFirst 2000 chars of response:")
        print(html[:2000])
        
        # Check for common elements
        checks = ['<html', '<svg', 'analytics', 'nn_svg', 'error', 'Exception']
        for check in checks:
            count = html.count(check)
            print(f"'{check}': {count} occurrences")
