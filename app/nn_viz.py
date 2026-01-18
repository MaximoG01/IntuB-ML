# app/nn_viz.py
"""
Visualization utilities (safe to run without torch).

This module intentionally does NOT depend on PyTorch so that it works on Heroku
or any environment without torch.

We generate a pedagogical 2D loss surface for logistic loss:
- X axis: w0
- Y axis: w1
- Z axis: loss(w0, w1)

This is a projection (2 parameters) to visualize what in reality is a
high-dimensional optimization landscape.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Sequence

try:
    # small helper for nicer SVG text sizing
    from html import escape as _escape
except Exception:
    def _escape(x):
        return x


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # numerically stable sigmoid
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def _toy_logistic_loss(w0: float, w1: float, X: np.ndarray, y: np.ndarray) -> float:
    """
    Binary cross-entropy loss for a toy logistic model with two weights.
    X: (n, 2)
    y: (n,) in {0,1}
    """
    z = w0 * X[:, 0] + w1 * X[:, 1]
    p = _sigmoid(z)
    eps = 1e-9
    loss = -(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
    return float(loss.mean())


def loss_surface_2d_safe(grid: int = 35, wmin: float = -4.0, wmax: float = 4.0, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> dict:
    """
    Return dict with:
      X: 2D meshgrid for w0
      Y: 2D meshgrid for w1
      Z: 2D loss values

    If `X` and `y` are provided they will be projected to 2D via SVD/PCA-like
    projection and used to compute a loss surface that reflects your data.
    Keys are intentionally X/Y/Z to match your analytics template.
    """
    rng = np.random.default_rng(42)

    # If user provided X,y (from DB), project them to 2D via PCA-like SVD.
    if X is not None and y is not None:
        # center
        Xc = X.astype(float) - np.nanmean(X, axis=0)
        # compute principal components
        try:
            u, s, vt = np.linalg.svd(np.nan_to_num(Xc), full_matrices=False)
            comps = vt[:2].T  # (n_features, 2)
            X2 = (Xc @ comps).astype(float)
        except Exception:
            # fallback: take first two columns
            X2 = X[:, :2].astype(float)
        y2 = np.asarray(y).astype(float)
    else:
        # Toy dataset (fixed seed for reproducibility)
        X2 = rng.normal(size=(250, 2))
        w_true = np.array([1.0, -1.5])
        y2 = ((X2 @ w_true + 0.5 * rng.normal(size=250)) > 0).astype(float)

    w0_range = np.linspace(wmin, wmax, grid)
    w1_range = np.linspace(wmin, wmax, grid)

    W0, W1 = np.meshgrid(w0_range, w1_range)
    Z = np.zeros_like(W0, dtype=float)

    # Compute surface
    for i in range(grid):
        for j in range(grid):
            Z[i, j] = _toy_logistic_loss(W0[i, j], W1[i, j], X2, y2)

    return {
        "X": W0.tolist(),
        "Y": W1.tolist(),
        "Z": Z.tolist(),
        "grid": grid,
        "wmin": wmin,
        "wmax": wmax,
        "note": "2D projection of logistic loss (w0,w1).",
        "data_from_db": bool(X is not None and y is not None),
    }


def nn_svg_from_weights(w1: Sequence[Sequence[float]], b1: Sequence[float], w2: Optional[Sequence[Sequence[float]]] = None, b2: Optional[Sequence[float]] = None, input_names: Optional[Sequence[str]] = None) -> str:
    """Render a minimal SVG of a small feed-forward net.

    - w1: (hidden_dim x input_dim) or (hidden_dim, input_dim)
    - b1: (hidden_dim,)
    - w2: optional (1 x hidden_dim) or None for single-layer
    - input_names: list of feature names for input nodes
    """
    # normalize shapes
    w1_arr = np.array(w1, dtype=float)
    b1_arr = np.array(b1, dtype=float)
    hidden_dim, input_dim = w1_arr.shape

    if input_names is None:
        input_names = [f"x{i+1}" for i in range(input_dim)]

    # layout sizing - increased width for longer labels
    width = max(600, 150 + input_dim * 80)
    height = max(280, 150 + input_dim * 35 + 80)

    # Add space for title at top
    title_height = 40
    
    in_x = 80
    hid_x = width // 2
    out_x = width - 80

    # vertical positions
    in_y = np.linspace(80 + title_height, height - 60, input_dim)
    hid_y = np.linspace(80 + title_height, height - 60, hidden_dim)
    out_y = (80 + title_height + height - 60) / 2

    lines = []
    # edges input->hidden
    max_w = float(np.max(np.abs(w1_arr))) if w1_arr.size else 1.0
    for i in range(input_dim):
        for h in range(hidden_dim):
            w = w1_arr[h, i]
            sw = 0.5 + (abs(w) / (max_w + 1e-9)) * 3.5
            color = '#0f3c91' if w >= 0 else '#b42318'
            lines.append(f'<line x1="{in_x}" y1="{in_y[i]:.1f}" x2="{hid_x}" y2="{hid_y[h]:.1f}" stroke="{color}" stroke-width="{sw:.2f}" opacity="0.9" />')

    # edges hidden->out (if present)
    if w2 is not None:
        w2_arr = np.array(w2, dtype=float).reshape(-1)
        max_w2 = float(np.max(np.abs(w2_arr))) if w2_arr.size else 1.0
        for h in range(hidden_dim):
            w = w2_arr[h]
            sw = 0.5 + (abs(w) / (max_w2 + 1e-9)) * 4.0
            color = '#0f3c91' if w >= 0 else '#b42318'
            lines.append(f'<line x1="{hid_x}" y1="{hid_y[h]:.1f}" x2="{out_x}" y2="{out_y:.1f}" stroke="{color}" stroke-width="{sw:.2f}" opacity="0.95" />')

    # nodes
    circles = []
    # Input nodes with feature names
    for i in range(input_dim):
        # Shorten long names for display
        label = str(input_names[i])
        if len(label) > 12:
            label = label[:10] + ".."
        circles.append(f'<g><circle cx="{in_x}" cy="{in_y[i]:.1f}" r="10" fill="#ffffff" stroke="#0f3c91" stroke-width="1.5"/><text x="{in_x+18}" y="{in_y[i]+4:.1f}" font-size="11" font-family="Arial, sans-serif">{_escape(label)}</text></g>')

    # Hidden layer nodes (logit)
    for h in range(hidden_dim):
        circles.append(f'<circle cx="{hid_x}" cy="{hid_y[h]:.1f}" r="12" fill="#f7f9ff" stroke="#0f3c91" stroke-width="2"/>')
        circles.append(f'<text x="{hid_x-8}" y="{hid_y[h]+5:.1f}" font-size="10" text-anchor="middle" font-family="Arial, sans-serif">logit</text>')

    # output node - labeled as "Difficulty Risk"
    circles.append(f'<g><circle cx="{out_x}" cy="{out_y:.1f}" r="16" fill="#fff7f7" stroke="#b42318" stroke-width="2"/><text x="{out_x-12}" y="{out_y+5:.1f}" font-size="11" text-anchor="middle" font-family="Arial, sans-serif" font-weight="bold">Diff.</text><text x="{out_x-12}" y="{out_y+17:.1f}" font-size="9" text-anchor="middle" font-family="Arial, sans-serif">Risk</text></g>')

    # Add title at top
    title_text = "Logistic Regression: Difficult Intubation Prediction"
    circles.insert(0, f'<text x="{width/2}" y="25" font-size="14" font-weight="bold" text-anchor="middle" font-family="Arial, sans-serif" fill="#102a43">{_escape(title_text)}</text>')

    # Add legend at bottom
    legend_y = height - 35
    circles.append(f'<line x1="80" y1="{legend_y}" x2="130" y2="{legend_y}" stroke="#0f3c91" stroke-width="2.5"/>')
    circles.append(f'<text x="140" y="{legend_y+4}" font-size="11" fill="#0f3c91" font-family="Arial, sans-serif">increases difficulty</text>')
    
    circles.append(f'<line x1="380" y1="{legend_y}" x2="430" y2="{legend_y}" stroke="#b42318" stroke-width="2.5"/>')
    circles.append(f'<text x="440" y="{legend_y+4}" font-size="11" fill="#b42318" font-family="Arial, sans-serif">decreases difficulty</text>')
    
    # Layer labels
    circles.append(f'<text x="80" y="70" font-size="10" font-weight="bold" text-anchor="middle" font-family="Arial, sans-serif" fill="#52606d">INPUT</text>')
    circles.append(f'<text x="{width//2}" y="70" font-size="10" font-weight="bold" text-anchor="middle" font-family="Arial, sans-serif" fill="#52606d">HIDDEN</text>')
    circles.append(f'<text x="{out_x}" y="70" font-size="10" font-weight="bold" text-anchor="middle" font-family="Arial, sans-serif" fill="#52606d">OUTPUT</text>')

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height + 20}" viewBox="0 0 {width} {height + 20}">'
    svg += '\n'.join(lines)
    svg += '\n' + '\n'.join(circles)
    svg += '</svg>'
    return svg
