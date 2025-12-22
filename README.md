# Human vs AI Text Classification Analysis

A research system for distinguishing human-written from AI-generated text using frozen BERT embeddings, with comprehensive neuron-level interpretability analysis.

## 📋 Overview

**Two-part research pipeline:**
1. **Classification**: Frozen BERT → Lightweight sklearn classifiers (>95% accuracy)
2. **Interpretability**: Statistical analysis of 9,216 neurons across all 12 BERT layers

**Key Features:**
- ✅ Config-driven experiments with automatic caching
- ✅ Comprehensive neuron analysis (Mann-Whitney U + AUC + Cohen's d)
- ✅ Hierarchical clustering of discriminative neurons
- ✅ Publication-ready visualizations and statistical validation
- ✅ Wandb integration for experiment tracking

## 🎯 Research Findings

**Main Discovery**: 1,350 discriminative neurons (14.6%) identified across BERT layers

**Key Results:**
- **Late layers (9-12)** contain 44.7% of discriminative neurons → semantic processing dominates
- **Balanced detection**: 688 AI-preferring vs 662 human-preferring neurons (1.04:1 ratio)
- **Strong effects**: 84.3% of discriminative neurons have |Cohen's d| > 0.8
- **Three functional clusters**: Human specialists (distributed), early AI detectors, late AI detectors

**Statistical Validation:**
- All neurons survive Bonferroni correction (α = 0.001, adjusted α' = 1.09×10⁻⁷)
- Median p-value: 8.68×10⁻³⁴
- Low redundancy: mean correlation = 0.146

## 🗂️ Project Structure

```
Human-vs-AI-text-Text-Classification-Analysis-/
│
├── configs/                          # YAML-based experiment configs
│   ├── experiments/                  # Complete pipeline configs
│   ├── tokenizers/                   # BERT tokenizer settings
│   ├── encoders/                     # Frozen BERT encoder settings
│   └── classifiers/                  # sklearn classifier configs (SGD, LogReg, etc.)
│
├── data/
│   ├── raw/AI_Human.csv             # Kaggle dataset
│   └── processed/                    # Auto-cached tokenized/encoded data
│
├── models/                           # Trained classifiers (.pkl files)
│
├── scripts/
│   ├── run_training.py              # 🚀 Main training pipeline
│   ├── train_all_classifiers.py     # Batch training script
│   ├── extract_activations.py       # 🔬 Extract neuron activations (all 12 layers)
│   ├── analyze_activations.py       # 🔬 Statistical analysis (deprecated, use notebooks)
│   ├── tokenize_dataset.py          # Standalone tokenization
│   ├── encode_dataset.py            # Standalone encoding
│   └── train_classifier.py          # Standalone training
│
├── utils/                            # Core utilities
│   ├── dataset_tokenizer.py         # Tokenization logic
│   ├── dataset_encoder.py           # Encoding logic
│   ├── classifier_trainer.py        # sklearn training wrapper
│   ├── activation_extractor.py      # Layer activation extraction
│   └── training_pipeline.py         # End-to-end pipeline orchestration
│
├── notebooks/                        # 📊 Analysis & visualization
│   ├── neurons_analysis.ipynb       # Statistical analysis of discriminative neurons
│   ├── neuron_clustering.ipynb      # Hierarchical clustering & UMAP visualization
│   └── dataset_analysis.ipynb       # Dataset exploration
│
├── results/
│   ├── activations/                 # Neuron data (all 12 layers, 3000 samples)
│   │   ├── layer_{1-12}_activations.npy      # Raw activation matrices
│   │   ├── layer_{1-12}_neuron_stats.csv     # Per-neuron statistics
│   │   └── metadata.json                      # Analysis metadata
│   └── figures/                     # Publication-ready plots
│
└── requirements.txt
```

## 🚀 Quick Start

### 1. Setup

```bash
pip install -r requirements.txt

# Download dataset from Kaggle and place at: data/raw/AI_Human.csv
# https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text
```

### 2. Train Classifier

```bash
# Run complete pipeline (auto-caches tokenization/encoding)
python scripts/run_training.py sgd

# With wandb tracking (optional)
python scripts/run_training.py sgd --wandb-project my-project
```

**What happens:**
1. Tokenize → `data/processed/AI_Human/tokenized/bert-base-uncased/`
2. Encode (frozen BERT) → `data/processed/AI_Human/encoded/bert_bert/`
3. Train SGD classifier → logs metrics

**Subsequent runs:** Only step 3 re-runs (steps 1-2 cached)

## 📝 Configuration

Experiments use YAML files in `configs/experiments/`. See existing configs and `configs/README.md` for examples.

## 📊 Dataset

**Source:** [Kaggle - AI vs Human Text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text)

- **Task**: Binary classification (0 = Human, 1 = AI-generated)
- **Split**: 80/20 train/test (stratified)
- **Preprocessing**: BERT tokenization, max 512 tokens, CLS token pooling
- **Analysis subset**: 3,000 samples (1,500 AI + 1,500 Human) for neuron analysis

## 🛠️ Tech Stack

**Core**: PyTorch, HuggingFace Transformers, scikit-learn  
**Analysis**: NumPy, Pandas, SciPy, UMAP, matplotlib, seaborn  
**Tracking**: Wandb (optional)

## 🔬 Neuron Analysis Pipeline

**Comprehensive interpretability study** of all 9,216 neurons across BERT's 12 layers.

### Quick Start

```bash
# 1. Train classifier (creates tokenized dataset)
python scripts/run_training.py sgd

# 2. Extract activations from all 12 layers (~10 min, 3000 samples)
python scripts/extract_activations.py --layers 1 2 3 4 5 6 7 8 9 10 11 12 --samples 3000

# 3. Run analysis notebooks (recommended over deprecated analyze_activations.py)
# Open: notebooks/neurons_analysis.ipynb
# Open: notebooks/neuron_clustering.ipynb
```

### Statistical Method

**Per-neuron analysis** (9,216 neurons = 12 layers × 768 neurons):

1. **Mann-Whitney U Test**: Tests if AI vs Human activation distributions differ
   - Significance level: α = 0.001
   - Bonferroni correction: α' = 1.09×10⁻⁷ (for 9,216 tests)

2. **Effect Sizes**:
   - **AUC** (Area Under ROC): Discriminative power (0.5 = random, 0/1 = perfect)
   - **Cohen's d**: Magnitude of difference (|d| > 0.8 = large effect)

3. **Discriminative Neuron Criteria**:
   - `p < α'` (survives Bonferroni correction)
   - `AUC > 0.7` (AI-preferring) OR `AUC < 0.3` (Human-preferring)

### Analysis Notebooks

#### `neurons_analysis.ipynb` - Statistical Analysis
**Outputs:**
- Table 1: Summary statistics (1,350 discriminative neurons)
- Figure 1: Layer-wise distribution (percentage & AI:Human ratio)
- Figure 2: Top-10 neuron activation boxplots
- Figure 3: Mean activation scatter (AI vs Human)
- Statistical validation table (effect sizes, p-values, correlations)

**Key Findings:**
- 14.6% of neurons are discriminative
- Late layers (9-12): 604 neurons (44.7% of discriminative)
- Peak: Layer 11 with 181 discriminative neurons
- Mean |Cohen's d| = 0.967 (very large effects)

#### `neuron_clustering.ipynb` - Functional Organization
**Outputs:**
- UMAP visualization (layers & preference)
- Hierarchical clustering dendrogram
- Silhouette analysis (optimal K=3)
- Cluster characterization tables

**Discovered Clusters:**
- **Cluster 0** (n=689, 51%): Human specialists, distributed across all layers
- **Cluster 1** (n=212, 16%): Early AI detectors (90.6% from layers 1-4)
- **Cluster 2** (n=449, 33%): Late AI detectors (56% from layers 10-12)

### Data Files

**Activation Data**: `results/activations/`
- `layer_{1-12}_activations.npy` - Raw activation matrices (3000 samples × 768 neurons)
- `layer_{1-12}_neuron_stats.csv` - Per-neuron statistics (AUC, p-value, Cohen's d, etc.)
- `labels.npy` - Sample labels (0=Human, 1=AI)
- `metadata.json` - Analysis configuration

**Figures**: `results/figures/`
- `figure{1-3}_*.png` - Statistical analysis plots
- `neuron_umap_*.png` - UMAP visualizations
- `neuron_dendrogram_cut.png` - Hierarchical clustering
- `neuron_silhouette_scores.png` - Cluster validation

### Quick Inspection

```python
import pandas as pd
import numpy as np

# Load neuron statistics for layer 12
df = pd.read_csv('results/activations/layer_12_neuron_stats.csv')

# Top discriminative neuron
top = df.nlargest(1, 'auc_deviation').iloc[0]
print(f"Layer 12, Neuron {top['neuron_idx']}: AUC={top['auc']:.3f}, Cohen's d={top['cohens_d']:.2f}")

# Count discriminative neurons
disc = df[df['discriminative']]
print(f"Discriminative: {len(disc)}/768 ({len(disc)/768*100:.1f}%)")
print(f"AI-preferring: {(disc['auc'] > 0.7).sum()}")
print(f"Human-preferring: {(disc['auc'] < 0.3).sum()}")
```

## 📈 Results Summary

### Classification Performance
- **SGD Classifier**: >95% accuracy on test set
- **Multiple classifiers tested**: Logistic Regression, Random Forest, Decision Tree, Linear SVC
- **Key insight**: Frozen BERT embeddings are highly discriminative

### Neuron Analysis Highlights
- **1,350 discriminative neurons** identified (14.6% of 9,216 total)
- **Balanced bidirectionality**: 688 AI-preferring vs 662 Human-preferring (1.04:1)
- **Layer distribution**: U-shaped pattern (early and late layers dominate)
- **Effect sizes**: 84.3% have large effects (|Cohen's d| > 0.8)
- **Functional organization**: 3 distinct clusters with specialized roles

## 📚 References

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Dataset: Kaggle AI vs Human Text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text)
- [UMAP: Uniform Manifold Approximation and Projection](https://umap-learn.readthedocs.io/)
- [scikit-learn](https://scikit-learn.org/)

## 📄 License

MIT License - Free for research and education

---

**Status:** ✅ Research Complete | **Last Updated:** December 2024
