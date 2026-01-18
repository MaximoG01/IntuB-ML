import sys
sys.path.insert(0, '.')

from app import create_app
from flask_login import login_user
from app.models import User

app = create_app()

# Create test user if needed
with app.app_context():
    user = User.query.first()
    if not user:
        user = User(username='test', email='test@test.com')
        user.set_password('test')
        from app import db
        db.session.add(user)
        db.session.commit()
        print("Created test user")
    
    # Test the analytics route function
    from app.routes import analytics
    from flask import g
    
    with app.test_request_context():
        # Mock login
        g.user = user
        from flask_login import current_user
        
        # Can't directly test analytics without proper session, so let's just 
        # verify the components work
        from app.ml import evaluate_logistic
        from app.nn_viz import loss_surface_2d_safe, nn_svg_from_weights
        from app.ml import train_logistic_model
        
        print("=" * 60)
        print("Testing Analytics Components")
        print("=" * 60)
        
        # Test logistic metrics
        log_metrics = evaluate_logistic(min_samples=20)
        print(f"\n✓ Logistic metrics: AUC={log_metrics.auc*100:.1f}%, Accuracy={log_metrics.accuracy*100:.1f}%")
        
        # Test loss surface
        from app.ml import _load_xy
        X, y, data_source = _load_xy(min_samples=20, prefer_db=True)
        loss_surface = loss_surface_2d_safe(X=X, y=y)
        print(f"✓ Loss surface generated: {len(loss_surface)} keys")
        
        # Test NN SVG
        model, _ = train_logistic_model(min_samples=20)
        lr = model.named_steps.get("lr")
        if lr:
            coef = lr.coef_[0].tolist()
            intercept = [float(lr.intercept_[0])]
            nn_svg = nn_svg_from_weights([coef], intercept, w2=None, b2=None)
            print(f"✓ NN SVG generated: {len(nn_svg)} chars")
            
        print("\n" + "=" * 60)
        print("All components working correctly!")
        print("=" * 60)
