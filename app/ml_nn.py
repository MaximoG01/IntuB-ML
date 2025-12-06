# app/ml_nn.py
# app/ml_nn.py
if TORCH_AVAILABLE:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = None


if TORCH_AVAILABLE:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = None


class IntubationNN(nn.Module):
    """
    Small MLP for binary classification: easy vs difficult intubation.
    Input = feature vector from build_feature_vector(record).
    Output = logit (we'll apply sigmoid outside).
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # single logit
        )

    def forward(self, x):
        # x: [batch, input_dim]
        return self.net(x).squeeze(1)  # [batch]


def load_dataset(min_samples: int = 50):
    """
    Load dataset from DB:
    - only records with difficult_binary not None (complete outcome)
    - returns X (numpy), y (numpy)
    Must be called inside app.app_context().
    """
    q = IntubationRecord.query.filter(IntubationRecord.difficult_binary.isnot(None))
    records = q.all()

    if len(records) < min_samples:
        raise ValueError(f"Not enough complete cases (found {len(records)}, need {min_samples}).")

    X = np.array([build_feature_vector(r) for r in records], dtype=float)
    y = np.array([1 if r.difficult_binary else 0 for r in records], dtype=float)

    return X, y


def train_nn(
    min_samples: int = 50,
    test_size: float = 0.2,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
):
    """
    Train the neural network on DB data.
    Must be called inside app.app_context().
    Returns (model, metrics_dict).
    """

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available in this environment.")

    # ... resto del codice come prima ...


    # ... resto della funzione come prima ...

    """
    Train the neural network on DB data.
    Must be called inside app.app_context().
    Returns (model, metrics_dict).
    """

    X, y = load_dataset(min_samples=min_samples)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    input_dim = X_train.shape[1]
    model = IntubationNN(input_dim).to(device)

    # convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    # optimizer + loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # simple mini-batch training loop
    n_samples = X_train_t.shape[0]
    n_batches = int(np.ceil(n_samples / batch_size))

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples)
        X_train_t = X_train_t[perm]
        y_train_t = y_train_t[perm]

        epoch_loss = 0.0

        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_samples)
            xb = X_train_t[start:end]
            yb = y_train_t[start:end]

            optimizer.zero_grad()
            logits = model(xb)  # [batch]
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * (end - start)

        epoch_loss /= n_samples

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Epoch {epoch+1}/{epochs}] loss={epoch_loss:.4f}")

    # evaluation on validation set
    model.eval()
    with torch.no_grad():
        logits_val = model(X_val_t)
        proba_val = torch.sigmoid(logits_val).cpu().numpy()  # [N]
        y_val_np = y_val

    y_pred = (proba_val >= 0.5).astype(int)

    acc = float(accuracy_score(y_val_np, y_pred))
    try:
        auc = float(roc_auc_score(y_val_np, proba_val))
    except ValueError:
        auc = None

    try:
        brier = float(brier_score_loss(y_val_np, proba_val))
    except Exception:
        brier = None

    metrics = {
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "accuracy_val": acc,
        "auc_val": auc,
        "brier_val": brier,
    }

    return model, metrics


def predict_single(model: IntubationNN, feature_vector):
    """
    Given a trained model and a single feature_vector (list or 1D numpy),
    returns probability of difficult intubation.
    """
    model.eval()
    x = np.array(feature_vector, dtype=float)[None, :]  # shape [1, input_dim]
    x_t = torch.tensor(x, dtype=torch.float32).to(device)
    with torch.no_grad():
        logit = model(x_t)[0]
        proba = torch.sigmoid(logit).item()
    return proba

def evaluate_nn(min_samples: int = 50):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available in this environment.")
    model, metrics = train_nn(min_samples=min_samples, epochs=40, batch_size=32)
    return metrics


