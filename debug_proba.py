import sys
sys.path.insert(0, '.')

from app import create_app
from app.ml import _load_xy, build_feature_vector, DIFFICULT_THRESHOLD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

app = create_app()

try:
    with app.app_context():
        X, y, data_source = _load_xy(min_samples=20, prefer_db=True)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ])
        model.fit(X_train, y_train)
        
        proba = model.predict_proba(X_val)[:, 1]
        
        print("=" * 60)
        print("Probability Analysis")
        print("=" * 60)
        print(f"Min probability: {np.min(proba):.6f}")
        print(f"Max probability: {np.max(proba):.6f}")
        print(f"Mean probability: {np.mean(proba):.6f}")
        print(f"Median probability: {np.median(proba):.6f}")
        print(f"Threshold: {DIFFICULT_THRESHOLD}")
        print(f"Probabilities > threshold: {np.sum(proba >= DIFFICULT_THRESHOLD)}/{len(proba)}")
        print(f"\nFirst 20 probabilities: {proba[:20]}")
        print(f"First 20 true labels: {y_val[:20]}")
        print(f"\nProbabilities for difficult cases (y=1): {proba[y_val==1][:20]}")
        print(f"Probabilities for non-difficult cases (y=0): {proba[y_val==0][:20]}")

except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()
