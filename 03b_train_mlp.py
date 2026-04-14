"""
Improved regression head: PyTorch MLP on ESM-2 embeddings.
Replaces linear Ridge with a 3-layer MLP + dropout + early stopping.
Re-uses the saved embeddings — no re-extraction needed.
"""

import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ── Load embeddings ────────────────────────────────────────────────────────────
X_train = np.load("data/embeddings_train.npy")
y_train = np.load("data/labels_train.npy")
X_val = np.load("data/embeddings_val.npy")
y_val = np.load("data/labels_val.npy")
X_test = np.load("data/embeddings_test.npy")
y_test = np.load("data/labels_test.npy")

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ── Normalise ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32)
X_val_s = scaler.transform(X_val).astype(np.float32)
X_test_s = scaler.transform(X_test).astype(np.float32)

y_train_f = y_train.astype(np.float32)
y_val_f = y_val.astype(np.float32)
y_test_f = y_test.astype(np.float32)


# ── DataLoaders ────────────────────────────────────────────────────────────────
def make_loader(X, y, batch_size=128, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


train_loader = make_loader(X_train_s, y_train_f, shuffle=True)
val_loader = make_loader(X_val_s, y_val_f)
test_loader = make_loader(X_test_s, y_test_f)


# ── MLP architecture ───────────────────────────────────────────────────────────
class ThermoMLP(nn.Module):
    def __init__(self, input_dim=480, hidden=(512, 256, 128), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

model = ThermoMLP(input_dim=X_train_s.shape[1], hidden=(256, 128), dropout=0.15).to(
    DEVICE
)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=15, factor=0.5
)
criterion = nn.MSELoss()

# ── Training loop with early stopping ─────────────────────────────────────────
EPOCHS = 500
PATIENCE = 40

best_val_loss = float("inf")
best_weights = None
wait = 0

print("\nTraining MLP...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= len(X_train_s)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            val_loss += criterion(model(xb), yb).item() * len(xb)
    val_loss /= len(X_val_s)

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    if epoch % 20 == 0:
        print(
            f"  Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

# ── Evaluate on test set ───────────────────────────────────────────────────────
model.load_state_dict(best_weights)
model.eval()

all_preds = []
with torch.no_grad():
    for xb, _ in test_loader:
        all_preds.append(model(xb.to(DEVICE)).cpu().numpy())

mlp_pred = np.vstack(all_preds).squeeze()

rmse = np.sqrt(mean_squared_error(y_test, mlp_pred))
mae = mean_absolute_error(y_test, mlp_pred)
r2 = r2_score(y_test, mlp_pred)
spearman = spearmanr(y_test, mlp_pred).statistic

print(f"\n--- ESM-2 + MLP (Test Set) ---")
print(f"  RMSE:     {rmse:.4f} C")
print(f"  MAE:      {mae:.4f} C")
print(f"  R2:       {r2:.4f}")
print(f"  Spearman: {spearman:.4f}")

# ── Append to metrics.csv ──────────────────────────────────────────────────────
metrics_df = pd.read_csv("results/metrics.csv")
new_row = pd.DataFrame(
    [
        {
            "model": "ESM-2 + MLP (512-256-128, dropout=0.3)",
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "spearman": round(spearman, 4),
        }
    ]
)
metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
metrics_df.to_csv("results/metrics.csv", index=False)
print("\nAppended to results/metrics.csv")

# ── Save MLP predictions alongside existing ones ───────────────────────────────
preds_df = pd.read_csv("results/predictions.csv")
preds_df["esm2_mlp_pred"] = mlp_pred
preds_df.to_csv("results/predictions.csv", index=False)
print("Updated results/predictions.csv")

# ── Save model weights ────────────────────────────────────────────────────────
torch.save(best_weights, "results/mlp_weights.pt")
print("Saved results/mlp_weights.pt")
