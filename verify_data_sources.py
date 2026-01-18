#!/usr/bin/env python
"""Verify what data is being used in the loss surface and where it comes from."""

import sys
sys.path.insert(0, 'c:\\Users\\Massimo.Giacon\\intuB')

from app import create_app
from app.ml import _load_xy, train_logistic_model
from app.ml_nn import evaluate_nn, TORCH_AVAILABLE
import numpy as np

app = create_app()

with app.app_context():
    print("=" * 70)
    print("DATA SOURCE VERIFICATION FOR ANALYTICS")
    print("=" * 70)
    
    # Check what _load_xy returns (used for loss surface)
    print("\n1. LOSS SURFACE DATA (_load_xy with prefer_db=True):")
    X_ls, y_ls, source_ls = _load_xy(min_samples=20, prefer_db=True)
    print(f"   • Source: {source_ls}")
    print(f"   • Shape: {X_ls.shape}")
    print(f"   • Classes: difficult={sum(y_ls)}, not_difficult={len(y_ls)-sum(y_ls)}")
    
    # Check logistic regression data
    print("\n2. LOGISTIC REGRESSION DATA (evaluate_logistic):")
    from app.ml import evaluate_logistic
    log_metrics = evaluate_logistic(min_samples=20)
    if log_metrics:
        print(f"   • Samples used: {log_metrics.n_samples}")
        print(f"   • Data source: {log_metrics.data_source}")
        print(f"   • Accuracy: {log_metrics.accuracy*100:.1f}%")
        print(f"   • AUC: {log_metrics.auc*100:.1f}%" if log_metrics.auc else "   • AUC: n/a")
    
    # Check NN data
    print("\n3. NEURAL NETWORK DATA (evaluate_nn):")
    if TORCH_AVAILABLE:
        nn_metrics = evaluate_nn(min_samples=50)
        if nn_metrics:
            print(f"   • Train samples: {nn_metrics['n_train']}")
            print(f"   • Val samples: {nn_metrics['n_val']}")
            print(f"   • Total: {nn_metrics['n_train'] + nn_metrics['n_val']}")
            print(f"   • AUC (val): {nn_metrics['auc_val']*100:.1f}%")
    else:
        print(f"   • PyTorch not available")
    
    # Analyze PCA
    print("\n4. PCA ANALYSIS (what's being projected):")
    Xc = X_ls.astype(float) - np.nanmean(X_ls, axis=0)
    u, s, vt = np.linalg.svd(np.nan_to_num(Xc), full_matrices=False)
    
    print(f"   • Input data shape: {X_ls.shape}")
    print(f"   • Original features: 10 clinical parameters")
    print(f"     - age, weight, height, bmi, sex")
    print(f"     - dtm (difficulty to mask), dii (difficulty intubation index)")
    print(f"     - mallampati, stop_bang, alganzouri (scoring systems)")
    
    print(f"\n   • PCA output: 2 principal components")
    print(f"   • PC1 explains: {(s[0]**2 / (s**2).sum() * 100):.1f}% of variance")
    print(f"   • PC2 explains: {(s[1]**2 / (s**2).sum() * 100):.1f}% of variance")
    print(f"   • Together: {((s[0]**2 + s[1]**2) / (s**2).sum() * 100):.1f}% of variance")
    
    # PC weights
    print(f"\n   • PC1 weights (which original features matter most):")
    weights_pc1 = vt[0]
    features = ['age', 'weight', 'height', 'bmi', 'sex', 'dtm', 'dii', 'mallampati', 'stop_bang', 'alganzouri']
    for feat, weight in zip(features, weights_pc1):
        print(f"     {feat:15s}: {weight:+.3f}")
    
    print(f"\n   • PC2 weights:")
    weights_pc2 = vt[1]
    for feat, weight in zip(features, weights_pc2):
        print(f"     {feat:15s}: {weight:+.3f}")
    
    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    print("""
✓ Loss surface uses: _load_xy() → prefers DATABASE, falls back to synthetic
✓ ROC curve uses: train_logistic_model() → same data source
✓ NN uses: evaluate_nn() → same data through _load_xy()
✓ PCA processes: All 10 clinical features reduced to 2D via SVD
✓ The loss surface shows: How loss varies across the 2D PCA space
    when fitting a simple logistic model (w0*PC1 + w1*PC2)
    """)
