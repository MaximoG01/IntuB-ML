import sys
sys.path.insert(0, '.')

from app import create_app
from app.ml import train_logistic_model
from app.nn_viz import nn_svg_from_weights

app = create_app()

with app.app_context():
    try:
        print("=" * 60)
        print("Testing NN SVG Generation")
        print("=" * 60)
        
        # Train logistic model
        model, metrics = train_logistic_model(min_samples=20)
        lr = model.named_steps.get("lr")
        
        if lr is not None:
            coef = lr.coef_[0].tolist()
            intercept = [float(lr.intercept_[0])]
            
            print(f"\nLogistic Regression Coefficients (w1): {coef}")
            print(f"Intercept (b1): {intercept}")
            print(f"Shape of coef: {len(coef)}")
            
            input_names = [
                "age",
                "weight", 
                "height",
                "bmi",
                "sex",
                "dtm",
                "dii",
                "mallampati",
                "stop_bang",
                "alganzouri",
            ]
            
            # Generate SVG - logistic as single layer
            print("\nGenerating SVG (logistic as single layer)...")
            nn_svg = nn_svg_from_weights([coef], intercept, w2=None, b2=None, input_names=input_names)
            
            print(f"\nSVG Generated! Length: {len(nn_svg)} characters")
            print(f"\nFirst 500 chars of SVG:")
            print(nn_svg[:500])
            print("\n...")
            print(f"\nLast 200 chars of SVG:")
            print(nn_svg[-200:])
            
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()
