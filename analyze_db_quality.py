import sys
sys.path.insert(0, '.')

from app import create_app
from app.models import IntubationRecord
from app.ml import build_feature_vector
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

app = create_app()

with app.app_context():
    # Get all labeled records from database
    records = IntubationRecord.query.filter(IntubationRecord.difficult_binary.isnot(None)).all()
    
    X = np.array([build_feature_vector(r) for r in records], dtype=float)
    y = np.array([1.0 if r.difficult_binary else 0.0 for r in records], dtype=float)
    
    print("=" * 60)
    print("DATABASE DATA ANALYSIS")
    print("=" * 60)
    print(f"Total records: {len(X)}")
    print(f"Difficult (y=1): {int(np.sum(y==1))}")
    print(f"Not difficult (y=0): {int(np.sum(y==0))}")
    print(f"Class ratio: {np.sum(y==1)/len(y)*100:.1f}% difficult")
    
    # Check feature correlation with target
    print("\nFeature-target correlations:")
    feature_names = ['age', 'weight', 'height', 'bmi', 'sex', 'dtm', 'dii', 'mallampati', 'stop_bang', 'alganzouri']
    for i, name in enumerate(feature_names):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        print(f"  {name:12s}: {corr:+.4f}")
    
    # Train model and get AUC
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=2000, solver='lbfgs')
    model.fit(X_train_scaled, y_train)
    
    proba = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, proba)
    fpr, tpr, _ = roc_curve(y_test, proba)
    
    print(f"\nModel Performance:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Test set size: {len(y_test)}")
    print(f"  Difficult in test: {int(np.sum(y_test==1))}")
    
    print(f"\nLogistic Regression Coefficients:")
    for i, (name, coef) in enumerate(zip(feature_names, model.coef_[0])):
        print(f"  {name:12s}: {coef:+.4f}")
    print(f"  Intercept: {model.intercept_[0]:+.4f}")
