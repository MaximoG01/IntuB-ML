#!/usr/bin/env python
"""Check the actual loss surface to verify color mapping."""

import sys
sys.path.insert(0, 'c:\\Users\\Massimo.Giacon\\intuB')

from app import create_app
from app.ml import _load_xy
from app.nn_viz import loss_surface_2d_safe
import numpy as np

app = create_app()

with app.app_context():
    X, y, _ = _load_xy(min_samples=20, prefer_db=True)
    ls = loss_surface_2d_safe(X=X, y=y)
    
    Z = np.array(ls["Z"])
    
    print("=" * 70)
    print("LOSS SURFACE COLOR MAPPING ANALYSIS")
    print("=" * 70)
    
    print(f"\nLoss Surface Statistics:")
    print(f"  Min loss (should be RED):    {np.nanmin(Z):.4f}")
    print(f"  Max loss (should be BLUE):   {np.nanmax(Z):.4f}")
    print(f"  Mean loss:                   {np.nanmean(Z):.4f}")
    print(f"  Std dev:                     {np.nanstd(Z):.4f}")
    
    # Find where min and max are
    min_idx = np.unravel_index(np.nanargmin(Z), Z.shape)
    max_idx = np.unravel_index(np.nanargmax(Z), Z.shape)
    
    print(f"\nMin loss location (grid position): {min_idx}")
    print(f"Max loss location (grid position): {max_idx}")
    
    # Check Plotly's default colorscale
    print(f"\n" + "=" * 70)
    print("PLOTLY DEFAULT COLORSCALE:")
    print("=" * 70)
    print("""
Plotly's default surface colorscale 'Viridis':
  - Low values (min loss):  DARK PURPLE/BLUE
  - High values (max loss): BRIGHT YELLOW/GREEN
  
This means:
  ❌ RED regions = NOT in default Viridis scale
  ❌ BLUE regions = LOW loss (GOOD) - opposite of your description!
  
The description says:
  "red/warm regions indicate LOW loss" - WRONG for Viridis
  "blue/cool regions indicate HIGH loss" - WRONG for Viridis
  
What Viridis actually shows:
  ✅ DARK/BLUE regions = LOW loss (good fit) 
  ✅ BRIGHT/YELLOW regions = HIGH loss (bad fit)
    """)
    
    print(f"\n" + "=" * 70)
    print("TRAJECTORY DATA:")
    print("=" * 70)
    if "traj" in ls and ls["traj"]:
        print(f"  Trajectory points: {len(ls['traj'])}")
        print(f"  First point: a={ls['traj'][0]['a']:.3f}, b={ls['traj'][0]['b']:.3f}")
        print(f"  Last point: a={ls['traj'][-1]['a']:.3f}, b={ls['traj'][-1]['b']:.3f}")
        
        if "loss_history" in ls and ls["loss_history"]:
            print(f"\n  Loss history along trajectory:")
            for i in [0, len(ls['loss_history'])//2, len(ls['loss_history'])-1]:
                print(f"    Step {i}: loss = {ls['loss_history'][i]:.4f}")
    else:
        print(f"  ⚠️  Trajectory data not available")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION:")
    print("=" * 70)
    print("""
1. Change description to match Viridis scale:
   - "Dark/blue regions indicate LOW loss (good fit)"
   - "Bright/yellow regions indicate HIGH loss (bad fit)"
   
2. Add line color to trajectory for visibility:
   - Use bright GREEN or LIME for contrast against surface
   - Green will show clearly against the Viridis background
   
3. Make line thicker and add markers:
   - Current width: 6 (OK)
   - Add explicit color: RGB(0, 255, 0) or similar
    """)
