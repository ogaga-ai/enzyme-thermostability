"""
Step 2: Extract ESM-2 embeddings for all sequences.
Model: facebook/esm2_t6_8M_UR50D (small, runs fast on CPU)
Output: numpy arrays saved to data/embeddings_{split}.npy
"""

import numpy as np
import pandas as pd
import torch
from transformers import EsmModel, EsmTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
print(f"Loading ESM-2 model ({MODEL_NAME})...")
tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
model = EsmModel.from_pretrained(MODEL_NAME)
model.eval()
model.to(DEVICE)
print("Model loaded.")


def get_embeddings(sequences, batch_size=32):
    """Extract [CLS] token embeddings for a list of sequences."""
    all_embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean pooling over non-padding tokens (better than [CLS] for proteins)
        last_hidden = outputs.last_hidden_state  # (B, L, D)
        attention_mask = inputs["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
        sum_hidden = (last_hidden * attention_mask).sum(dim=1)
        count = attention_mask.sum(dim=1).clamp(min=1e-9)
        mean_embeddings = (sum_hidden / count).cpu().numpy()
        all_embeddings.append(mean_embeddings)
        if (i // batch_size) % 5 == 0:
            print(f"  Processed {min(i + batch_size, len(sequences))}/{len(sequences)}")
    return np.vstack(all_embeddings)


for split in ["train", "val", "test"]:
    df = pd.read_csv(f"data/{split}.csv")
    sequences = df["aa_seq"].tolist()
    labels = df["label"].values

    print(f"\nExtracting embeddings for {split} ({len(sequences)} sequences)...")
    embeddings = get_embeddings(sequences)
    print(f"  Embedding shape: {embeddings.shape}")

    np.save(f"data/embeddings_{split}.npy", embeddings)
    np.save(f"data/labels_{split}.npy", labels)
    print(f"  Saved data/embeddings_{split}.npy and data/labels_{split}.npy")

print("\nAll embeddings extracted.")
