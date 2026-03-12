# Discriminative Neuron Clustering in BERT for AI-Generated Text Detection

An interpretability study analyzing frozen BERT to understand *which* neurons drive the detection of AI-generated text, and *how* they are functionally organized.

## Overview

**Model**: Frozen `bert-base-uncased` (12 layers, 768 neurons/layer = 9,216 total neurons)  
**Dataset**: `NicolaiSivesind/human-vs-machine` — `wiki_labeled` split (~300K samples, 50% human / 50% AI, GPT-3 generated)  
**Analysis subset**: 10,000 samples (5,000 AI + 5,000 human), balanced  
**Scope**: No fine-tuning. All analysis is on frozen model weights. CLS token activations only.

### Classification Baseline

98% accuracy is achievable using frozen BERT CLS activations + sklearn classifiers (Linear SVC, Random Forest). This is treated as a known baseline — the paper's contribution is the interpretability analysis of the discriminative neurons, not the classification performance.

### Core Findings

**1,350 of 9,216 neurons (14.6%) are discriminative:**
- 688 AI-preferring (AUC > 0.7)
- 662 human-preferring (AUC < 0.3)
- Layer distribution is U-shaped — early and late layers dominate; peak at layer 11

| Layer group | Neurons |
|---|---|
| Early (1–4) | 438 |
| Middle (5–8) | 308 |
| Late (9–12) | 604 |

**Three-cluster architecture** (hierarchical clustering + UMAP, silhouette-optimal K=3):

| Cluster | n | Interpretation |
|---|---|---|
| Cluster 0 | 689 | Human detection signal — distributed across all layers |
| Cluster 1 | 268 | Early AI detection — surface/syntactic features (layers 1–4) |
| Cluster 2 | 393 | Late AI detection — semantic patterns (layers 10–12) |

**Statistical validation:**
- All discriminative neurons survive Bonferroni correction (α = 0.001, adjusted α′ = 1.09×10⁻⁷)
- 84.3% have large effects (|Cohen's d| > 0.8), mean |d| = 0.967
- Median p-value: 8.68×10⁻³⁴

---

## Project Structure

```
├── scripts/
│   ├── tokenize_dataset.py        # Step 1: tokenize the HF dataset
│   ├── extract_activations.py     # Step 2: extract CLS activations (all 12 layers)
│   └── analyze_activations.py     # Step 3: Mann-Whitney U + AUC, saves per-layer CSVs
│
├── utils/
│   ├── activation_extractor.py    # ActivationExtractor class (layer-wise CLS extraction)
│   └── dataset_tokenizer.py       # DatasetTokenizer class
│
├── notebooks/
│   ├── neurons_analysis.ipynb     # Statistical analysis & figures (Sections A–C)
│   └── neuron_clustering.ipynb    # UMAP + hierarchical clustering
│
├── results/
│   ├── activations/               # Pre-computed data (load directly in notebooks)
│   │   ├── layer_{1-12}_activations.npy   # Raw CLS activations (10,000 × 768)
│   │   ├── layer_{1-12}_neuron_stats.csv  # Per-neuron statistics
│   │   ├── labels.npy                     # Sample labels (0=Human, 1=AI)
│   │   └── metadata.json                  # Extraction config
│   └── figures/                   # Publication-ready plots
│
├── data/
│   └── processed/                 # Tokenized dataset cache (auto-generated)
│
├── requirements.txt
└── utils/hidden.py                # HuggingFace API key (git-ignored)
```

---

## Reproducing the Analysis

### Setup

```bash
pip install -r requirements.txt
```

### Step 1 — Tokenize the dataset

```bash
python scripts/tokenize_dataset.py
```

Saves tokenized dataset to `data/processed/AI_Human/tokenized/bert-base-uncased/`.

### Step 2 — Extract CLS activations

```bash
python scripts/extract_activations.py --layers 1 2 3 4 5 6 7 8 9 10 11 12 --samples 10000
```

Saves `layer_{1-12}_activations.npy` and `labels.npy` to `results/activations/`.  
**Skip this step** — pre-computed results are already in `results/activations/`.

### Step 3 — Compute neuron statistics

```bash
python scripts/analyze_activations.py
```

Runs Mann-Whitney U + AUC per neuron, applies Bonferroni correction, and saves per-layer CSV files.  
**Skip this step** — pre-computed CSVs are already in `results/activations/`.

Optional wandb tracking:
```bash
python scripts/analyze_activations.py --wandb-project my-project
```

### Step 4 — Explore in notebooks

Open the notebooks in order:

1. **`notebooks/neurons_analysis.ipynb`** — neuron discovery, layer distribution, statistical validation
2. **`notebooks/neuron_clustering.ipynb`** — UMAP + hierarchical clustering, three-cluster architecture

All pre-computed data is ready; notebooks can be run immediately without steps 1–3.

---

## Statistical Method

**Per-neuron analysis** across all 9,216 neurons (12 layers × 768):

1. **Mann-Whitney U Test** — tests whether AI vs human activation distributions differ significantly
2. **AUC effect size** — AUC > 0.7 → AI-preferring; AUC < 0.3 → human-preferring
3. **Bonferroni correction** — α′ = 0.001 / 9,216 = 1.09×10⁻⁷ per layer
4. **Cohen's d** — computed as supplementary effect size

**Clustering:**
- UMAP with correlation distance reduces each neuron's 10,000-sample activation profile to 2D
- Hierarchical clustering (Ward linkage) on the UMAP embedding
- Silhouette analysis selects optimal K (K=3)

---

## Tech Stack

**Core**: PyTorch, HuggingFace Transformers  
**Analysis**: NumPy, Pandas, SciPy, scikit-learn  
**Clustering/Visualization**: UMAP, matplotlib, seaborn  
**Experiment tracking**: Weights & Biases (optional)

---

## References

- [HuggingFace dataset: NicolaiSivesind/human-vs-machine](https://huggingface.co/datasets/NicolaiSivesind/human-vs-machine)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [UMAP: Uniform Manifold Approximation and Projection](https://umap-learn.readthedocs.io/)
- [Mann-Whitney U test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html)
