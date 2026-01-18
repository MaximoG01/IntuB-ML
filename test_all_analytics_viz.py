#!/usr/bin/env python
"""Test all 4 analytics visualizations."""

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
        print("ANALYTICS VISUALIZATIONS TEST")
        print("=" * 70)
        
        # Check for all 4 charts
        charts = {
            "1. ROC Curve": "rocChart",
            "2. NN Architecture": "<svg",
            "3. Loss Surface (3D)": "lossSurfacePlot",
            "4. Training Loss (Epochs)": "trainingLossPlot"
        }
        
        all_present = True
        for name, identifier in charts.items():
            if identifier in html:
                print(f"✅ {name}: FOUND")
            else:
                print(f"❌ {name}: MISSING")
                all_present = False
        
        print("\n" + "=" * 70)
        
        # Check for improved descriptions
        print("\nCHECK: Improved descriptions")
        descriptions = {
            "PCA mention": "PCA dimensionality reduction",
            "PC1 variance": "45.6%",
            "PC2 variance": "40.1%",
            "10 clinical features": "clinical features",
            "Binary cross-entropy": "Binary Cross-Entropy",
            "Training loss description": "Training loss"
        }
        
        for key, text in descriptions.items():
            if text in html:
                print(f"✅ {key}: Found '{text}'")
            else:
                print(f"❌ {key}: NOT found")
                all_present = False
        
        print("\n" + "=" * 70)
        if all_present:
            print("✅ ALL TESTS PASSED - Analytics page fully functional!")
        else:
            print("⚠️  Some elements missing - check template")
        print("=" * 70)
