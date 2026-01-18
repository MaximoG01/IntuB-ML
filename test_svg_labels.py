#!/usr/bin/env python
"""Test that SVG contains clinical parameter labels."""

import sys
sys.path.insert(0, 'c:\\Users\\Massimo.Giacon\\intuB')

from app import create_app, db
from app.models import IntubationRecord
from app.ml import train_logistic_model

app = create_app()

with app.app_context():
    # Check if we have data
    count = IntubationRecord.query.count()
    print(f"Database has {count} records")
    
    # Try to train and get SVG
    try:
        model, log_metrics = train_logistic_model(min_samples=20)
        if model and log_metrics:
            # Get the logistic regression coefficients
            lr = model.named_steps.get("lr")
            if lr is not None:
                from app.nn_viz import nn_svg_from_weights
                
                coef = lr.coef_[0].tolist()
                intercept = [float(lr.intercept_[0])]
                
                input_names = [
                    "age", "weight", "height", "bmi", "sex",
                    "dtm", "dii", "mallampati", "stop_bang", "alganzouri"
                ]
                
                print(f"\nCoef shape: {len(coef)}, Input names: {len(input_names)}")
                
                # Call with proper 2D structure
                w1_2d = [coef]  # Make it 2D: [1, 10]
                print(f"w1_2d shape will be: ({len(w1_2d)}, {len(w1_2d[0]) if w1_2d else 0})")
                
                try:
                    nn_svg = nn_svg_from_weights(w1_2d, intercept, w2=None, b2=None, input_names=input_names)
                    print(f"✓ SVG generated: {len(nn_svg)} chars")
                    
                    # Check for clinical parameter names
                    clinical_params = ["age", "weight", "height", "bmi", "sex", "dtm", "dii", "mallampati", "stop_bang", "alganzouri"]
                    found_params = [p for p in clinical_params if p in nn_svg]
                    print(f"\n✓ Found clinical parameters in SVG: {found_params}")
                    
                    missing_params = [p for p in clinical_params if p not in nn_svg]
                    if missing_params:
                        print(f"✗ Missing parameters: {missing_params}")
                    
                    # Show some of the text nodes
                    import re
                    text_nodes = re.findall(r'<text[^>]*>([^<]+)</text>', nn_svg)
                    print(f"\nText nodes in SVG ({len(text_nodes)} total):")
                    for i, text in enumerate(text_nodes[:15]):
                        print(f"  {i}: {text}")
                        
                except Exception as e:
                    print(f"✗ SVG generation failed: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print("✗ Could not train logistic model")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
