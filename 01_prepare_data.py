"""
Step 1: Download and prepare the thermostability dataset.
Dataset: AI4Protein/Thermostability (HuggingFace)
  - 7,029 protein sequences with Tm labels (40.2 – 66.9 °C)
  - Pre-split into train / validation / test
"""

import pandas as pd
from datasets import load_dataset

print("Downloading AI4Protein/Thermostability ...")
ds = load_dataset("AI4Protein/Thermostability", trust_remote_code=True)
print(ds)

train_df = ds["train"].to_pandas()
val_df = ds["validation"].to_pandas()
test_df = ds["test"].to_pandas()

for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"\n{split}: {df.shape}")
    print(df.head(2))
    df.to_csv(f"data/{split}.csv", index=False)

print("\nColumns:", list(train_df.columns))
print("\nTm distribution (train):")
print(train_df["label"].describe())
print("\nData saved to data/")
