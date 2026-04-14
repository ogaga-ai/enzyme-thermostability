"""
Step 4: Generate publication-quality figures.
  Fig 1 — Predicted vs Actual Tm (3 models, 1x3 grid)
  Fig 2 — UMAP of ESM-2 embeddings coloured by Tm
  Fig 3 — Bar chart: model comparison (RMSE, R2, Spearman)
All saved to results/
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Load results ───────────────────────────────────────────────────────────────
preds = pd.read_csv("results/predictions.csv")
metrics = pd.read_csv("results/metrics.csv")

y_true = preds["y_true"].values
rf_pred = preds["rf_baseline_pred"].values
ridge_pred = preds["esm2_ridge_pred"].values
gb_pred = preds["esm2_gb_pred"].values

# ── Figure 1: Predicted vs Actual (3 panels) ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Enzyme Thermostability Prediction — Predicted vs Actual Tm",
    fontsize=13,
    fontweight="bold",
    y=1.01,
)

panel_data = [
    (rf_pred, "Baseline: Random Forest\n(AA Composition)", "#e63946", "Random Forest"),
    (ridge_pred, "ESM-2 + Ridge Regression", "#4361ee", "ESM-2 + Ridge"),
    (gb_pred, "ESM-2 + Gradient Boosting", "#2ec4b6", "ESM-2 + Gradient Boosting"),
]

for ax, (pred, title, color, model_key) in zip(axes, panel_data):
    row = metrics[
        metrics["model"].str.contains(model_key.split("+")[0].strip(), regex=False)
    ]
    r2 = row["r2"].values[0]
    spearman = row["spearman"].values[0]
    rmse = row["rmse"].values[0]

    lo = min(y_true.min(), pred.min()) - 1
    hi = max(y_true.max(), pred.max()) + 1
    ax.scatter(y_true, pred, alpha=0.4, s=14, color=color, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Actual Tm (°C)", fontsize=10)
    ax.set_ylabel("Predicted Tm (°C)", fontsize=10)
    ax.set_title(title, fontsize=10.5)
    ax.text(
        0.05,
        0.95,
        f"R2 = {r2:.3f}\nSpearman = {spearman:.3f}\nRMSE = {rmse:.2f} °C",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("results/fig1_predicted_vs_actual.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved results/fig1_predicted_vs_actual.png")

# ── Figure 2: UMAP of ESM-2 embeddings ────────────────────────────────────────
try:
    import umap

    X_test = np.load("data/embeddings_test.npy")
    print("Running UMAP (this takes ~1-2 min on CPU)...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding_2d = reducer.fit_transform(X_test)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=y_true,
        cmap="plasma",
        s=15,
        alpha=0.7,
        edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Tm (°C)")
    ax.set_title(
        "UMAP of ESM-2 Protein Embeddings\nColoured by Melting Temperature (Tm)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig("results/fig2_umap_embeddings.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved results/fig2_umap_embeddings.png")
except ImportError:
    print("umap-learn not installed — skipping UMAP plot.")

# ── Figure 3: Model Comparison Bar Chart ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle("Model Comparison — Test Set Performance", fontsize=13, fontweight="bold")

short_names = ["RF\nBaseline", "ESM-2\n+Ridge", "ESM-2\n+GradBoost"]
bar_colors = ["#e63946", "#4361ee", "#2ec4b6"]

for ax, metric_col, ylabel, higher_better in zip(
    axes,
    ["r2", "spearman", "rmse"],
    ["R2", "Spearman r", "RMSE (°C)"],
    [True, True, False],
):
    vals = metrics[metric_col].values
    bars = ax.bar(short_names, vals, color=bar_colors, width=0.5, edgecolor="white")
    ax.set_title(ylabel, fontsize=11)
    ax.set_ylim(0, max(vals) * 1.3)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(vals) * 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("results/fig3_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved results/fig3_model_comparison.png")

print("\nAll figures saved to results/")
