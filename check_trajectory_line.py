#!/usr/bin/env python
"""Check if trajectory data is being added to loss_surface."""

import sys
sys.path.insert(0, 'c:\\Users\\Massimo.Giacon\\intuB')

from app import create_app
from app.ml import _load_xy
from app.ml_nn import evaluate_nn, TORCH_AVAILABLE
from app.nn_viz import loss_surface_2d_safe
import numpy as np

app = create_app()

with app.app_context():
    print("=" * 70)
    print("TRAJECTORY LINE CHECK")
    print("=" * 70)
    
    # Get loss surface
    X, y, _ = _load_xy(min_samples=20, prefer_db=True)
    ls = loss_surface_2d_safe(X=X, y=y)
    
    # Check if trajectory exists
    print(f"\nInitial loss_surface keys: {ls.keys()}")
    print(f"Has 'traj' key: {'traj' in ls}")
    print(f"Has 'loss_history' key: {'loss_history' in ls}")
    
    # Get NN metrics with trajectory
    if TORCH_AVAILABLE:
        print("\n--- Getting NN Metrics ---")
        nn_metrics = evaluate_nn(min_samples=50)
        
        print(f"NN metrics keys: {nn_metrics.keys()}")
        print(f"Has 'theta_history': {'theta_history' in nn_metrics}")
        print(f"Has 'loss_history': {'loss_history' in nn_metrics}")
        
        if 'theta_history' in nn_metrics:
            theta = np.array(nn_metrics['theta_history'])
            print(f"theta_history shape: {theta.shape}")
            print(f"First theta: {theta[0][:5]}...")
            print(f"Last theta: {theta[-1][:5]}...")
        
        if 'loss_history' in nn_metrics:
            loss_hist = nn_metrics['loss_history']
            print(f"loss_history length: {len(loss_hist)}")
            print(f"First loss: {loss_hist[0]:.4f}")
            print(f"Last loss: {loss_hist[-1]:.4f}")
        
        # Now try the projection logic from routes.py
        print("\n--- Simulating routes.py trajectory projection ---")
        try:
            if nn_metrics and nn_metrics.get("theta_history"):
                theta_hist = np.array(nn_metrics["theta_history"])
                
                Xc = X.astype(float) - np.nanmean(X, axis=0)
                u, s, vt = np.linalg.svd(np.nan_to_num(Xc), full_matrices=False)
                comps = vt[:2].T
                
                if theta_hist.shape[1] >= 2:
                    traj_2d = []
                    for weights in theta_hist:
                        w = np.zeros(X.shape[1])
                        w[:min(len(weights), len(w))] = weights[:min(len(weights), len(w))]
                        proj = (w @ comps).tolist()
                        traj_2d.append({"a": proj[0], "b": proj[1]})
                    
                    ls["traj"] = traj_2d
                    ls["loss_history"] = nn_metrics.get("loss_history", [])
                    
                    print(f"✓ Trajectory added!")
                    print(f"  Number of trajectory points: {len(ls['traj'])}")
                    print(f"  First point: a={ls['traj'][0]['a']:.3f}, b={ls['traj'][0]['b']:.3f}")
                    print(f"  Last point: a={ls['traj'][-1]['a']:.3f}, b={ls['traj'][-1]['b']:.3f}")
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nPyTorch not available - no trajectory")
    
    print("\n" + "=" * 70)
    print(f"Final loss_surface has 'traj': {'traj' in ls}")
    print("=" * 70)
