# app/ml_nn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.ml import _load_xy  # uses DB if enough, else synthetic seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Robust torch availability (handles broken DLL cases on Windows)
TORCH_AVAILABLE = False
TORCH_ERROR: Optional[str] = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # Force a minimal op that will fail if DLL deps are missing
    _ = torch.tensor([0.0]).sum().item()
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    TORCH_ERROR = str(e)
    torch = None
    nn = None
    optim = None


@dataclass
class HybridNNMetrics:
    n_train: int
    n_val: int
    accuracy_val: float
    auc_val: float
    brier_val: float
    torch_device: str
    epochs: int
    alpha_distill: float
    # for visualization:
    loss_history: List[float]
    theta_history: List[List[float]]
    input_dim: int
    hidden_dim: int
    # for NN SVG weights:
    w1: List[List[float]]
    b1: List[float]
    w2: List[List[float]]
    b2: List[float]


if not TORCH_AVAILABLE:

    def evaluate_nn(*args, **kwargs):
        return None

else:

    class DistilledNN(nn.Module):
        """1 hidden layer as requested."""
        def __init__(self, input_dim: int, hidden_dim: int = 16):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.act = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            h = self.act(self.fc1(x))
            return self.fc2(h).squeeze(1)


    def _flatten_params(model: nn.Module) -> torch.Tensor:
        return torch.cat([p.detach().flatten().cpu() for p in model.parameters()])


    def _train_teacher_lr(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
        teacher = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ])
        teacher.fit(X_train, y_train)
        return teacher


    def train_hybrid_distilled_nn(
        min_samples: int = 50,
        test_size: float = 0.25,
        epochs: int = 40,
        lr: float = 1e-3,
        batch_size: int = 32,
        hidden_dim: int = 16,
        alpha_distill: float = 0.35,  # how much we follow teacher
        random_state: int = 42,
    ) -> Tuple[DistilledNN, HybridNNMetrics]:
        """
        Hybrid = LR teacher + NN student.
        Loss = (1-alpha)*BCE(y) + alpha*MSE(sigmoid(student), teacher_proba)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X, y, data_source = _load_xy(min_samples=min_samples, prefer_db=True)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        teacher = _train_teacher_lr(X_train, y_train)
        teacher_val = teacher.predict_proba(X_val)[:, 1].astype(np.float32)
        teacher_train = teacher.predict_proba(X_train)[:, 1].astype(np.float32)

        # Student
        input_dim = X_train.shape[1]
        model = DistilledNN(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

        # tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
        t_train_t = torch.tensor(teacher_train, dtype=torch.float32, device=device)

        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
        t_val_t = torch.tensor(teacher_val, dtype=torch.float32, device=device)

        opt = optim.Adam(model.parameters(), lr=lr)
        bce = nn.BCEWithLogitsLoss()
        mse = nn.MSELoss()

        n = X_train_t.shape[0]
        n_batches = int(np.ceil(n / batch_size))

        loss_history: List[float] = []
        theta_history: List[List[float]] = []

        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n, device=device)
            Xb = X_train_t[perm]
            yb = y_train_t[perm]
            tb = t_train_t[perm]

            ep_losses = []

            for bi in range(n_batches):
                s = bi * batch_size
                e = min((bi + 1) * batch_size, n)

                opt.zero_grad()
                logits = model(Xb[s:e])
                loss_y = bce(logits, yb[s:e])
                p = torch.sigmoid(logits)
                loss_t = mse(p, tb[s:e])
                loss = (1.0 - alpha_distill) * loss_y + alpha_distill * loss_t
                loss.backward()
                opt.step()

                ep_losses.append(float(loss.detach().cpu().item()))

            # snapshot at epoch end
            loss_history.append(float(np.mean(ep_losses)))
            theta_history.append(_flatten_params(model).numpy().astype(float).tolist())

        # evaluate
        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_t)
            proba_val = torch.sigmoid(logits_val).detach().cpu().numpy()

        pred = (proba_val >= 0.5).astype(int)

        # Export weights for SVG (fc1 and fc2)
        w1 = model.fc1.weight.detach().cpu().numpy().astype(float).tolist()
        b1 = model.fc1.bias.detach().cpu().numpy().astype(float).tolist()
        w2 = model.fc2.weight.detach().cpu().numpy().astype(float).tolist()  # [1][hidden]
        b2 = model.fc2.bias.detach().cpu().numpy().astype(float).tolist()

        metrics = HybridNNMetrics(
            n_train=int(len(X_train)),
            n_val=int(len(X_val)),
            accuracy_val=float(accuracy_score(y_val, pred)),
            auc_val=float(roc_auc_score(y_val, proba_val)) if len(np.unique(y_val)) > 1 else float("nan"),
            brier_val=float(brier_score_loss(y_val, proba_val)),
            torch_device=str(device),
            epochs=int(epochs),
            alpha_distill=float(alpha_distill),
            loss_history=loss_history,
            theta_history=theta_history,
            input_dim=int(input_dim),
            hidden_dim=int(hidden_dim),
            w1=w1, b1=b1, w2=w2, b2=b2,
        )
        return model, metrics


    def evaluate_nn(min_samples: int = 50) -> Dict[str, Any]:
        """
        Returns a plain dict for Jinja safety (no attribute errors).
        """
        _, m = train_hybrid_distilled_nn(min_samples=min_samples)
        return {
            "n_train": m.n_train,
            "n_val": m.n_val,
            "accuracy_val": m.accuracy_val,
            "auc_val": m.auc_val,
            "brier_val": m.brier_val,
            "torch_device": m.torch_device,
            "epochs": m.epochs,
            "alpha_distill": m.alpha_distill,
            "loss_history": m.loss_history,
            "theta_history": m.theta_history,
            "input_dim": m.input_dim,
            "hidden_dim": m.hidden_dim,
            "w1": m.w1, "b1": m.b1, "w2": m.w2, "b2": m.b2,
        }
