"""
Master runner — executes the full pipeline in order.
Run this file: python run_pipeline.py
"""

import subprocess
import sys

steps = [
    ("01_prepare_data.py", "Step 1: Download & prepare data"),
    ("02_extract_esm_embeddings.py", "Step 2: Extract ESM-2 embeddings"),
    ("03_train_and_evaluate.py", "Step 3: Train models & evaluate"),
    ("04_visualize.py", "Step 4: Generate figures"),
]

for script, desc in steps:
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script], check=True)
    if result.returncode != 0:
        print(f"ERROR: {script} failed.")
        sys.exit(1)

print("\n" + "=" * 60)
print("  Pipeline complete. Results in results/")
print("=" * 60)
