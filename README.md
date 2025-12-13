# Human vs AI Text Classification Analysis

A modular system for distinguishing human-written from AI-generated text using frozen BERT embeddings and lightweight classifiers, with neuron-level interpretability analysis.

## 📋 Overview

**Config-driven pipeline** for text classification with interpretability:
1. **Tokenization** → **Encoding** (frozen BERT) → **Classification** (sklearn)
2. **Neuron Analysis**: Statistical analysis of which BERT neurons discriminate AI vs human text

**Key Features:**
- ✅ Automatic caching (tokenization/encoding computed once)
- ✅ YAML-based experiment configuration
- ✅ High accuracy (>95%) without fine-tuning
- ✅ Neuron-level interpretability with Mann-Whitney U + AUC
- ✅ Wandb integration for visualization

## 🎯 Research Objective

**Question**: Which neurons in BERT's transformer layers activate differently for AI-generated vs human-written text?

**Approach**: Statistical analysis (Mann-Whitney U test + AUC) on 3,072 neurons across 4 BERT layers to identify discriminative patterns.

## 🗂️ Project Structure

```
Human-vs-AI-text-Text-Classification-Analysis-/
│
├── configs/                          # Experiment configurations
│   ├── experiments/                  # Main experiment configs
│   │   ├── sgd.yaml                 # BERT + SGD (default)
│   │   ├── logistic_regression.yaml
│   │   └── ...
│   ├── tokenizers/                   # Tokenizer configurations
│   │   └── bert.yaml
│   ├── encoders/                     # Encoder model configurations
│   │   └── bert.yaml
│   └── classifiers/                  # Classifier configurations
│       └── sgd.yaml
│
├── data/
│   ├── raw/                          # Raw CSV dataset
│   │   └── AI_Human.csv             # From Kaggle
│   └── processed/                    # Processed datasets with versioning
│       ├── AI_Human/
│       │   ├── tokenized/           # Organized by tokenizer
│       │   │   └── {tokenizer_name}/
│       │   └── encoded/             # Organized by tokenizer_encoder
│       │       └── {tok}_{enc}/
│       └── registry.json            # Dataset provenance tracking
│
├── models/
│   └── (trained models saved here via standalone scripts)
│
├── scripts/
│   ├── run_training.py              # 🚀 Main training pipeline with caching
│   ├── tokenize_dataset.py          # Standalone tokenization script
│   ├── encode_dataset.py            # Standalone encoding script
│   ├── train_classifier.py          # Standalone training script
│   ├── load_dataset.py              # Quick dataset inspection
│   ├── extract_activations.py       # 🔬 Extract layer activations for analysis
│   └── analyze_activations.py       # 🔬 Statistical neuron analysis
│
├── utils/
│   ├── dataset_tokenizer.py         # Tokenization utilities
│   ├── dataset_encoder.py           # Encoding utilities
│   ├── classifier_trainer.py        # Generic sklearn trainer
│   └── activation_extractor.py      # Layer activation extraction
│
├── notebooks/
│   └── analysis.ipynb               # Dataset analysis and model comparison
│
├── results/                          # Analysis outputs
│   ├── figures/
│   ├── metrics/
│   └── activations/                 # Neuron activation analysis results
│
└── requirements.txt                  # Python dependencies
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

Binary classification: `0` = Human-written, `1` = AI-generated
- Train/test split: 80/20 (stratified)
- Max sequence length: 512 tokens

## 🛠️ Tech Stack

PyTorch, HuggingFace (Transformers & Datasets), scikit-learn, wandb, matplotlib, seaborn

## 🔬 Activation Analysis (Research Component)

**Neuron-level interpretability** to understand what BERT learns for AI text detection.

### Quick Start

**Prerequisite:** Run `python scripts/run_training.py sgd` first to create tokenized dataset.

```bash
# 1. Extract CLS token activations from 4 layers (3, 6, 9, 12) - ~5 min
python scripts/extract_activations.py

# 2. Statistical analysis with wandb (REQUIRED) - ~3 min
python scripts/analyze_activations.py --wandb-project human-vs-ai

# 3. View results at: https://wandb.ai/your-username/human-vs-ai
```

### Statistical Method

**Per-neuron analysis** (3,072 neurons across layers 3, 6, 9, 12):
- **Test**: Mann-Whitney U (non-parametric, tests if AI/Human distributions differ)
- **Effect Size**: AUC (Area Under ROC Curve)
  - `AUC > 0.5` → AI-preferring (activates higher for AI text)
  - `AUC < 0.5` → Human-preferring (activates higher for human text)
- **Discriminative Neuron**: `p < 0.001` (Bonferroni corrected) AND (`AUC > 0.7` OR `AUC < 0.3`)

### Outputs

**Local CSV**: `results/activations/layer_{3,6,9,12}_neuron_stats.csv`

**Wandb Dashboard** (publication-ready):
1. Layer-wise bar chart (discriminative neuron counts by direction)
2. AUC violin plots (effect size distributions per layer)
3. Top-10 neurons heatmap (50 AI + 50 Human samples)
4. Top-5 neurons box plots (activation distributions)
5. Summary tables (per-layer stats, top-20 neurons)
6. Overall metrics (early vs late layer discrimination)

### Inspect Results

```python
import pandas as pd

# Load layer statistics
df = pd.read_csv('results/activations/layer_12_neuron_stats.csv')

# Most discriminative neuron
top = df.loc[df['auc_deviation'].idxmax()]
print(f"Neuron {top['neuron_idx']}: AUC={top['auc']:.3f}, {top['direction']}")

# AI-preferring neurons (AUC > 0.7)
ai_neurons = df[(df['discriminative']) & (df['direction'] == 'AI-preferring')]
print(f"AI-preferring: {len(ai_neurons)} neurons")
```

### Interpretation

- **Late layers dominate (9-12)** → BERT uses **semantic features** (meaning, coherence)
- **Early layers dominate (3-6)** → BERT uses **syntactic features** (grammar, structure)
- **Distributed** → Multi-level hierarchical processing

## 📄 License

MIT License - Feel free to use this for research and education!

## 📚 References

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets)
- [scikit-learn](https://scikit-learn.org/)
- [Dataset Source: Kaggle AI vs Human Text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text)

---

**Status:** ✅ Production-ready | **Last Updated:** December 2024
