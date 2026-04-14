"""
Step 3: Train regression models on ESM-2 embeddings and evaluate.

Models compared:
  A. Baseline: Random Forest on amino acid composition features
  B. ESM-2 + Ridge Regression (mean-pooled embeddings)
  C. ESM-2 + Gradient Boosting (mean-pooled embeddings)

Metrics: RMSE, MAE, R2, Spearman correlation
Results saved to results/metrics.csv and results/predictions.csv
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── Load embeddings ────────────────────────────────────────────────────────────
X_train = np.load("data/embeddings_train.npy")
y_train = np.load("data/labels_train.npy")
X_val = np.load("data/embeddings_val.npy")
y_val = np.load("data/labels_val.npy")
X_test = np.load("data/embeddings_test.npy")
y_test = np.load("data/labels_test.npy")

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ── Baseline: Random Forest on amino acid composition features ─────────────────
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def aa_composition(seq):
    """20-dimensional amino acid frequency vector."""
    seq = seq.upper()
    total = max(len(seq), 1)
    return [seq.count(aa) / total for aa in AMINO_ACIDS]


print("\nBuilding baseline sequence features (amino acid composition)...")
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")
test_df = pd.read_csv("data/test.csv")

X_train_aa = np.array([aa_composition(s) for s in train_df["aa_seq"]])
X_val_aa = np.array([aa_composition(s) for s in val_df["aa_seq"]])
X_test_aa = np.array([aa_composition(s) for s in test_df["aa_seq"]])

rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_aa, y_train)
rf_pred_test = rf.predict(X_test_aa)

# ── Proposed: Ridge on ESM-2 embeddings ───────────────────────────────────────
print("Training Ridge Regression on ESM-2 embeddings...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Tune alpha on validation set
best_alpha, best_val_r2 = 1.0, -np.inf
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    val_r2 = r2_score(y_val, ridge.predict(X_val_scaled))
    if val_r2 > best_val_r2:
        best_val_r2, best_alpha = val_r2, alpha

print(f"  Best alpha: {best_alpha}  (val R²={best_val_r2:.4f})")
ridge = Ridge(alpha=best_alpha)
ridge.fit(X_train_scaled, y_train)
ridge_pred_test = ridge.predict(X_test_scaled)

# ── Proposed B: Gradient Boosting on ESM-2 embeddings ─────────────────────────
print("Training Gradient Boosting on ESM-2 embeddings...")
gb = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42
)
gb.fit(X_train_scaled, y_train)
gb_pred_test = gb.predict(X_test_scaled)
print(f"  Val R2 (GB): {r2_score(y_val, gb.predict(X_val_scaled)):.4f}")


# ── Metrics ───────────────────────────────────────────────────────────────────
def metrics(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    spearman = spearmanr(y_true, y_pred).statistic
    print(f"\n{name}")
    print(f"  RMSE:     {rmse:.4f} °C")
    print(f"  MAE:      {mae:.4f} °C")
    print(f"  R²:       {r2:.4f}")
    print(f"  Spearman: {spearman:.4f}")
    return {
        "model": name,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "spearman": round(spearman, 4),
    }


print("\n--- Test Set Results ---")
rows = [
    metrics("Baseline: Random Forest (AA composition)", y_test, rf_pred_test),
    metrics("ESM-2 + Ridge Regression", y_test, ridge_pred_test),
    metrics("ESM-2 + Gradient Boosting", y_test, gb_pred_test),
]

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv("results/metrics.csv", index=False)
print(f"\nMetrics saved to results/metrics.csv")

# Save predictions for plotting
preds_df = pd.DataFrame(
    {
        "y_true": y_test,
        "esm2_ridge_pred": ridge_pred_test,
        "esm2_gb_pred": gb_pred_test,
        "rf_baseline_pred": rf_pred_test,
    }
)
preds_df.to_csv("results/predictions.csv", index=False)
print("Predictions saved to results/predictions.csv")

# Save model info
model_info = {
    "best_ridge_alpha": best_alpha,
    "val_r2_ridge": round(best_val_r2, 4),
    "val_r2_gb": round(r2_score(y_val, gb.predict(X_val_scaled)), 4),
}
with open("results/model_info.json", "w") as f:
    json.dump(model_info, f, indent=2)
print("Model info saved to results/model_info.json")
