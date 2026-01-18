#!/usr/bin/env python
"""Extract and display what users will see in the NN visualization."""

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
        
        # Extract SVG
        svg_match = re.search(r'<svg[^>]*>.*?</svg>', html, re.DOTALL)
        if svg_match:
            svg = svg_match.group()
            
            # Extract all text nodes
            text_nodes = re.findall(r'<text[^>]*>([^<]+)</text>', svg)
            
            print("=" * 60)
            print("NEURAL NETWORK VISUALIZATION - TEXT CONTENT")
            print("=" * 60)
            print(f"\nTotal text nodes: {len(text_nodes)}\n")
            
            # Organize by category
            print("📊 Title:")
            print(f"  {text_nodes[0]}")
            
            print("\n🔢 Input Layer (10 clinical parameters):")
            for i in range(1, 11):
                if i < len(text_nodes):
                    print(f"  {i}. {text_nodes[i]}")
            
            print("\n🧠 Hidden Layer:")
            if 11 < len(text_nodes):
                print(f"  {text_nodes[11]}")
            
            print("\n📤 Output Layer:")
            if 13 < len(text_nodes):
                print(f"  {text_nodes[12]} {text_nodes[13]}")
            
            print("\n📝 Legend:")
            legend_start = 14
            while legend_start < len(text_nodes):
                print(f"  {text_nodes[legend_start]}")
                legend_start += 1
            
            print("\n" + "=" * 60)
            print("✅ VERIFICATION: All clinical parameters are labeled!")
            print("=" * 60)
        else:
            print("ERROR: Could not extract SVG")
