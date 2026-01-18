import sys
sys.path.insert(0, '.')

from app import create_app
from app.ml import evaluate_logistic, _load_xy
import json

app = create_app()

try:
    with app.app_context():
        # Test _load_xy first
        print("=" * 60)
        print("Testing _load_xy()...")
        X, y, data_source = _load_xy(min_samples=20, prefer_db=True)
        print(f"Data source: {data_source}")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"y values: {set(y)}")
        print(f"Class distribution: {sum(y==1)}/{len(y)} difficult")
        
        # Test evaluate_logistic
        print("\n" + "=" * 60)
        print("Testing evaluate_logistic()...")
        metrics = evaluate_logistic(min_samples=20)
        print(f"Accuracy: {metrics.accuracy * 100:.1f}%")
        print(f"AUC: {metrics.auc * 100:.1f}%" if metrics.auc else "AUC: n/a")
        print(f"Threshold: {metrics.threshold}")
        print(f"Data source: {metrics.data_source}")
        print(f"FPR length: {len(metrics.fpr)}, TPR length: {len(metrics.tpr)}")
        if metrics.fpr:
            print(f"FPR: {metrics.fpr[:5]}...")
            print(f"TPR: {metrics.tpr[:5]}...")
        print(f"Confusion: TN={metrics.confusion.tn}, FP={metrics.confusion.fp}, FN={metrics.confusion.fn}, TP={metrics.confusion.tp}")
    
except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()
