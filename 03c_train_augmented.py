"""
Step 3c: Augment ESM-2 embeddings with hand-crafted physicochemical features.
Concatenates 640-dim ESM-2 embeddings with:
  - Sequence length (normalized)
  - AA composition (20 features)
  - GRAVY score (hydrophobicity)
  - Isoelectric point (estimated)
  - Aromaticity
  - Instability index proxy

Then trains Ridge and GradientBoosting on the augmented features.
Run this after 02_extract_esm_embeddings.py with the 150M model.
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── Physicochemical feature tables ─────────────────────────────────────────────
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

# pKa values for isoelectric point estimation (simplified)
PKA_N_TERM = 9.0
PKA_C_TERM = 2.0
PKA_SIDE = {"D": 3.9, "E": 4.1, "H": 6.0, "C": 8.3, "Y": 10.1, "K": 10.5, "R": 12.5}

AROMATIC = {"F", "W", "Y"}


def physicochemical_features(seq):
    """Compute a 25-dimensional physicochemical feature vector."""
    seq = seq.upper()
    n = max(len(seq), 1)
    counts = {aa: seq.count(aa) for aa in AMINO_ACIDS}

    # AA composition (20)
    aa_comp = [counts[aa] / n for aa in AMINO_ACIDS]

    # GRAVY score
    gravy = sum(HYDROPHOBICITY.get(aa, 0) * counts[aa] for aa in AMINO_ACIDS) / n

    # Aromaticity
    aromaticity = sum(counts[aa] for aa in AROMATIC) / n

    # Instability index proxy: fraction of charged residues
    charged = (counts["D"] + counts["E"] + counts["K"] + counts["R"]) / n

    # Sequence length (log-normalized — spans 3 orders of magnitude)
    log_len = np.log1p(len(seq)) / 10.0

    # Fraction of thermostability-correlated residues (Ile, Val, Leu — aliphatic)
    aliphatic = (counts["I"] + counts["V"] + counts["L"]) / n

    return aa_comp + [gravy, aromaticity, charged, log_len, aliphatic]  # 25 features


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading embeddings and sequences...")
X_train_emb = np.load("data/embeddings_train.npy")
X_val_emb = np.load("data/embeddings_val.npy")
X_test_emb = np.load("data/embeddings_test.npy")
y_train = np.load("data/labels_train.npy")
y_val = np.load("data/labels_val.npy")
y_test = np.load("data/labels_test.npy")

train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")
test_df = pd.read_csv("data/test.csv")

print("Computing physicochemical features...")
X_train_pc = np.array([physicochemical_features(s) for s in train_df["aa_seq"]])
X_val_pc = np.array([physicochemical_features(s) for s in val_df["aa_seq"]])
X_test_pc = np.array([physicochemical_features(s) for s in test_df["aa_seq"]])

# Concatenate embeddings + physicochemical features
X_train = np.hstack([X_train_emb, X_train_pc])
X_val = np.hstack([X_val_emb, X_val_pc])
X_test = np.hstack([X_test_emb, X_test_pc])
print(f"Augmented feature dim: {X_train.shape[1]} (embeddings + 25 physicochemical)")

# ── Scale ──────────────────────────────────────────────────────────────────────
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])

scaler_tune = StandardScaler()
X_train_s = scaler_tune.fit_transform(X_train)
X_val_s = scaler_tune.transform(X_val)

scaler_full = StandardScaler()
X_trainval_s = scaler_full.fit_transform(X_trainval)
X_test_s = scaler_full.transform(X_test)

# ── Ridge: tune on val, train on train+val ─────────────────────────────────────
print("\nTuning Ridge alpha...")
best_alpha, best_val_r2 = 1.0, -np.inf
for alpha in [0.1, 1.0, 10.0, 100.0, 500.0, 1000.0, 5000.0]:
    r = Ridge(alpha=alpha).fit(X_train_s, y_train)
    vr2 = r2_score(y_val, r.predict(X_val_s))
    if vr2 > best_val_r2:
        best_val_r2, best_alpha = vr2, alpha
print(f"  Best alpha={best_alpha}  val_R2={best_val_r2:.4f}")

ridge = Ridge(alpha=best_alpha).fit(X_trainval_s, y_trainval)
ridge_pred = ridge.predict(X_test_s)

# ── Gradient Boosting ──────────────────────────────────────────────────────────
print("Training Gradient Boosting (augmented features)...")
gb = GradientBoostingRegressor(
    n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42
).fit(X_trainval_s, y_trainval)
gb_pred = gb.predict(X_test_s)


# ── Evaluate ──────────────────────────────────────────────────────────────────
def report(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    spearman = spearmanr(y_true, y_pred).statistic
    print(f"\n{name}")
    print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}  Spearman={spearman:.4f}")
    return {
        "model": name,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "spearman": round(spearman, 4),
    }


print("\n--- Test Set Results (augmented embeddings) ---")
rows = [
    report("ESM-2 150M + Ridge (aug.)", y_test, ridge_pred),
    report("ESM-2 150M + GradBoost (aug.)", y_test, gb_pred),
]

# ── Save ───────────────────────────────────────────────────────────────────────
metrics_df = pd.read_csv("results/metrics.csv")
metrics_df = pd.concat([metrics_df, pd.DataFrame(rows)], ignore_index=True)
metrics_df.to_csv("results/metrics.csv", index=False)
print("\nSaved to results/metrics.csv")

preds_df = pd.read_csv("results/predictions.csv")
preds_df["esm2_150m_ridge_aug"] = ridge_pred
preds_df["esm2_150m_gb_aug"] = gb_pred
preds_df.to_csv("results/predictions.csv", index=False)
print("Updated results/predictions.csv")
