# Enzyme Thermostability Prediction with ESM-2 Protein Language Model

Predicting enzyme melting temperature (Tm) from amino acid sequence using Meta's ESM-2 protein language model and supervised regression.

## Why This Matters

Enzymes are the catalysts that drive industrial bioprocesses such as fermentation, biodegradation, drug synthesis, biofuel production. However, most natural enzymes are fragile. They unfold and become ineffective above ~40°C. But industrial reactors run at 60–80°C for efficiency. To solve this problem, my goal is to engineer a thermostable enzyme experimetally. 

Engineering a thermostable enzyme experimentally means following a 4-step process that;
- Designs a mutation → express the protein in a host organism → purify it → measure Tm in the lab

- Executing this could take **weeks of work per variant**, and directed evolution generates **thousands of variants**, but computational prediction lets you pre-screen millions of candidates in minutes and only synthesize the top ones. This is the bottleneck companies such as  Novozymes, Codexis, and DSM-Firmenich spend hundreds of millions solving every year. This project is also directly tied to published to my experimental research on lipase and cellulase production in fermentation systems; exactly the class of enzyme where thermostability determines industrial viability.

## Overview

This project benchmarks ESM-2 sequence embeddings against a hand-crafted amino acid composition baseline on the AI4Protein thermostability dataset (7,029 proteins, Tm range: 40–67°C).

## Results

| Model | R² | Spearman ρ | RMSE (°C) |
|---|---|---|---|
| Baseline: Random Forest (AA composition) | 0.320 | 0.561 | 4.700 |
| ESM-2 (35M) + Ridge Regression | 0.435 | 0.656 | 4.285 |
| ESM-2 (35M) + Gradient Boosting | 0.438 | 0.652 | 4.274 |
| **ESM-2 (150M) + Ridge + Physicochemical** | **0.456** | **0.673** | **4.206** |
| ESM-2 (150M) + Gradient Boosting + Physicochemical | 0.452 | 0.667 | 4.219 |

ESM-2 (150M) embeddings augmented with physicochemical sequence features improve R² by **+43% relative** and Spearman correlation by **+20% relative** over the amino acid composition baseline, demonstrating that transformer-derived representations capture thermodynamic information beyond simple residue statistics. All models trained on train + validation combined (5,693 sequences) and evaluated on a held-out test set (1,336 sequences).

## Figures

**Predicted vs Actual Tm (three models)**

![Predicted vs Actual](results/fig1_predicted_vs_actual.png)

**UMAP of ESM-2 embeddings coloured by Tm**

![UMAP](results/fig2_umap_embeddings.png)

The UMAP projection reveals continuous Tm-correlated structure in the ESM-2 embedding space — proteins with similar Tm values cluster together, consistent with the model capturing thermodynamic signal in its representations.

**Model comparison**

![Model Comparison](results/fig3_model_comparison.png)

## Methodology

### Dataset
- **Source:** [AI4Protein/Thermostability](https://huggingface.co/datasets/AI4Protein/Thermostability) (HuggingFace)
- 7,029 proteins — 5,054 train / 639 validation / 1,336 test
- Target: melting temperature Tm (°C), range 40.2–66.9°C

### Embeddings
- **Model:** `facebook/esm2_t30_150M_UR50D` (150M parameters, 30 layers, 640-dim hidden)
- **Pooling:** Mean pooling over non-padding token positions (outperforms [CLS] pooling for protein-level tasks)
- **Truncation:** max 512 tokens; batch size 32

### Feature Augmentation
In addition to ESM-2 embeddings, 25 physicochemical features are concatenated:
- 20-dimensional amino acid frequency vector
- GRAVY score (Kyte-Doolittle hydrophobicity)
- Aromaticity (fraction of F, W, Y residues)
- Aliphatic fraction (I, V, L ;thermostability correlated)
- Charged residue fraction (D, E, K, R)
- Log-normalized sequence length

### Regression heads
- **Ridge Regression:** alpha tuned on validation set; features standardized with `StandardScaler`
- **Gradient Boosting:** 400 estimators, learning rate 0.05, max depth 4, subsample 0.8
- **Baseline RF:** 200 estimators on 20-dimensional amino acid frequency vectors only

### Training protocol
- Alpha/hyperparameter selection on validation set
- Final models retrained on train + validation combined (5,693 sequences)
- Evaluated on held-out test set (1,336 sequences)

### Evaluation
- Spearman correlation (primary — rank-based, standard for protein fitness prediction benchmarks)
- R², RMSE, MAE on held-out test set

## Limitations and Interpretation of Results

### Why R² = 0.456?

The best model explains **45.6% of the variance** in melting temperature. This is expected and honest for the following reasons:

**1. Sequence alone is an incomplete signal**
Thermostability is determined by the protein's 3D folded structure, that is; the network of hydrogen bonds, hydrophobic core packing, salt bridges, and disulfide bonds that resist unfolding at high temperature. Amino acid sequence encodes this structure only indirectly. Predicting Tm from sequence without structural input is an inherently noisy task.

**2. ESM-2 is used as a frozen feature extractor**
The ESM-2 model weights were not updated during training. we extracted its pre-trained representations and fitted a regression model on top. ESM-2 was trained on evolutionary sequence patterns, not thermostability. Fine-tuning ESM-2 end-to-end on this task (updating all 150M parameters) would significantly improve performance but requires a GPU (~4–6 hours on a T4).

**3. Dataset size**
7,029 proteins is small by deep learning standards. The best published results on thermostability use millions of variant measurements (e.g., ProteinGym, 2.7M variants across 217 proteins). More data enables deeper models to generalize better.

### Performance in context

| Approach | Typical R² on this task |
|---|---|
| AA composition baseline (this work) | 0.32 |
| Frozen ESM-2 + regression (this work) | 0.456 |
| Frozen ESM-2 + structural features | ~0.50–0.55 |
| Full ESM-2 fine-tuning (GPU required) | ~0.65–0.70 |
| ESM-2 650M fine-tuned on ProteinGym | ~0.72–0.75 |

This work sits correctly at the frozen-embedding ceiling. The +43% relative improvement over baseline demonstrates that transformer-derived representations capture thermodynamic signal that simple residue statistics miss, even without fine-tuning.

### What Spearman ρ = 0.673 means

Spearman correlation measures rank ordering, whether the model correctly identifies which proteins are more thermostable than others, regardless of exact temperature values. ρ = 0.673 means the model's rankings are strongly correlated with the true rankings. For industrial pre-screening (rank candidates, synthesize the top 10%) this is the metric that matters most, and 0.673 is competitive with published frozen-embedding benchmarks.

### Known limitations
- No 3D structural input (AlphaFold2 structural features would improve results)
- Frozen model weights; fine-tuning would raise the ceiling substantially
- Sequences truncated at 512 tokens; very long proteins (>512 AA) lose C-terminal information
- Training data limited to ~7k proteins; generalization to highly novel protein families may be poor

## Motivation

This project is a direct extension of published experimental research on ML-driven bioprocess optimization (7 peer-reviewed publications, 120+ citations). Prior work applied ANN, Random Forest, and Bayesian optimization to fermentation and enzymatic production systems. This project extends that framework to protein-level sequence modelling using transformer-based representations bridging tabular bioprocess ML with modern protein language models.
The thermostability problem is directly relevant to industrial enzyme engineering for fermentation and bioconversion processes, including lipase production systems studied in published research.

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets scikit-learn scipy matplotlib umap-learn
```

## Usage

```bash
python 01_prepare_data.py             # Download and cache dataset
python 02_extract_esm_embeddings.py   # Extract ESM-2 embeddings (~30 min CPU)
python 03_train_and_evaluate.py       # Train baseline + ESM-2 models
python 03c_train_augmented.py         # Train augmented (embeddings + physicochemical)
python 04_visualize.py                # Generate figures to results/
```
Or run the full pipeline:

```bash
python run_pipeline.py
```

## Repository Structure

```
enzyme-thermostability/
├── 01_prepare_data.py
├── 02_extract_esm_embeddings.py
├── 03_train_and_evaluate.py
├── 03b_train_mlp.py
├── 03c_train_augmented.py
├── 04_visualize.py
├── run_pipeline.py
├── data/
│   ├── train.csv / val.csv / test.csv
│   ├── embeddings_{split}.npy
│   └── labels_{split}.npy
└── results/
    ├── metrics.csv
    ├── predictions.csv
    ├── model_info.json
    ├── fig1_predicted_vs_actual.png
    ├── fig2_umap_embeddings.png
    └── fig3_model_comparison.png
```

## Author

**Ogaga Maxwell Okedi**
- MS Computer Science, University of Texas at Dallas
- MS Chemical Engineering, FAMU–FSU College of Engineering
- [ogaga-ai.github.io](https://ogaga-ai.github.io) ·
- [github.com/ogaga-ai](https://github.com/ogaga-ai)
