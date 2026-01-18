#!/usr/bin/env python
"""Analyze what the loss surface plot is actually representing."""

import sys
sys.path.insert(0, 'c:\\Users\\Massimo.Giacon\\intuB')

from app import create_app
from app.ml import _load_xy
from app.nn_viz import loss_surface_2d_safe
import numpy as np

app = create_app()

with app.app_context():
    # Load the actual data used
    X, y, data_source = _load_xy(min_samples=20, prefer_db=True)
    
    print("=" * 70)
    print("LOSS SURFACE ANALYSIS")
    print("=" * 70)
    
    print(f"\n1. DATA LOADING:")
    print(f"   • Database records: {X.shape[0]}")
    print(f"   • Input features: {X.shape[1]} (age, weight, height, bmi, sex, dtm, dii, mallampati, stop_bang, alganzouri)")
    print(f"   • Classes: {len(np.unique(y))} (difficult: {sum(y)}, not difficult: {len(y)-sum(y)})")
    print(f"   • Data source: {data_source}")
    
    print(f"\n2. PCA PROJECTION:")
    print(f"   • Original space: {X.shape[1]}D (10 features)")
    print(f"   • Projected space: 2D (using PCA/SVD)")
    
    # Compute PCA manually to show what's happening
    Xc = X.astype(float) - np.nanmean(X, axis=0)
    u, s, vt = np.linalg.svd(np.nan_to_num(Xc), full_matrices=False)
    comps = vt[:2].T  # (n_features, 2)
    X2 = (Xc @ comps).astype(float)
    
    print(f"   • First principal component explains: {(s[0]**2 / (s**2).sum() * 100):.1f}% of variance")
    print(f"   • Second principal component explains: {(s[1]**2 / (s**2).sum() * 100):.1f}% of variance")
    print(f"   • Together: {((s[0]**2 + s[1]**2) / (s**2).sum() * 100):.1f}% of variance")
    
    print(f"\n3. LOSS SURFACE GENERATION:")
    print(f"   • Grid resolution: 35×35 points")
    print(f"   • Weight range: [-4.0, +4.0] for both w0 and w1")
    print(f"   • Loss function: Binary cross-entropy")
    print(f"   • Formula: loss(w0, w1) = -mean(y*log(p) + (1-y)*log(1-p))")
    print(f"     where p = sigmoid(w0*X2[:,0] + w1*X2[:,1])")
    
    # Generate the loss surface
    ls = loss_surface_2d_safe(X=X, y=y)
    
    Z_array = np.array(ls["Z"])
    print(f"\n4. LOSS SURFACE PROPERTIES:")
    print(f"   • Minimum loss: {np.nanmin(Z_array):.4f}")
    print(f"   • Maximum loss: {np.nanmax(Z_array):.4f}")
    print(f"   • Mean loss: {np.nanmean(Z_array):.4f}")
    
    # Find the minimum
    min_idx = np.unravel_index(np.nanargmin(Z_array), Z_array.shape)
    w0_min = np.linspace(-4.0, 4.0, 35)[min_idx[1]]
    w1_min = np.linspace(-4.0, 4.0, 35)[min_idx[0]]
    print(f"   • Optimal weights (w0, w1): ({w0_min:.2f}, {w1_min:.2f})")
    
    print(f"\n5. TRAJECTORY (from neural network training):")
    print(f"   • 'loss_history' contains: epoch-to-epoch loss values")
    print(f"   • 'traj' contains: epoch-to-epoch (a, b) positions in the 2D PCA space")
    print(f"   • The trajectory shows how the network's weights moved during training")
    print(f"   • Ideally, trajectory should follow downhill toward the minimum")
    
    if "traj" in ls:
        print(f"   • Number of epochs: {len(ls['traj'])}")
    else:
        print(f"   • No trajectory data available (no training history)")
    
    print(f"\n" + "=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)
    print("""
The plot shows:
  • SURFACE: Loss landscape as a function of two synthetic weights (w0, w1)
             derived from projecting the 10D feature space to 2D via PCA
  • RED VALLEY: Region where loss is low (good fit)
  • BLUE PEAKS: Region where loss is high (bad fit)
  • YELLOW LINE: The path the optimizer took during training, shown as it
                 would appear if the 10D weight vector were projected to this
                 same 2D space
  
This is NOT a plot of:
  - Individual features vs. loss
  - Training loss over epochs (that would be 1D)
  - Actual weight values (we're in a PCA-projected space)
  
It IS a plot of:
  - How loss varies across a 2D subspace of the full weight parameter space
  - How the training trajectory navigated this simplified landscape
  - Whether the optimizer found the valley (good) or got stuck (bad)
""")
