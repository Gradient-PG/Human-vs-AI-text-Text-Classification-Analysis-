# Human vs AI Text Classification Analysis

A modular text classification system for distinguishing human-written from AI-generated text using transformer encoders and classical ML classifiers.

## 📋 Overview

This project implements a **config-driven pipeline** for text classification experiments. It separates the encoding phase (using transformer models like BERT) from the classification phase (using sklearn classifiers), allowing flexible experimentation with different model combinations.

**Architecture:**
1. **Tokenization**: Convert raw text to tokens using HuggingFace tokenizers
2. **Encoding**: Generate embeddings using transformer models (BERT, DistilBERT, etc.)
3. **Classification**: Train lightweight classifiers on frozen embeddings (SGD, Logistic Regression, etc.)

**Key Features:**
- ✅ Configuration-based experiments via YAML files
- ✅ Automatic caching to avoid redundant computation
- ✅ Modular pipeline with independent tokenizer/encoder/classifier components
- ✅ Support for any HuggingFace tokenizer and transformer encoder
- ✅ Compatible with sklearn classifiers

## 🗂️ Project Structure

```
Human-vs-AI-text-Text-Classification-Analysis-/
│
├── configs/                          # Experiment configurations
│   ├── experiments/                  # Main experiment configs
│   │   └── baseline.yaml            # BERT tokenizer + encoder + SGD classifier
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
│   └── load_dataset.py              # Quick dataset inspection
│
├── utils/
│   ├── dataset_tokenizer.py         # Tokenization utilities
│   ├── dataset_encoder.py           # Encoding utilities
│   └── classifier_trainer.py        # Generic sklearn trainer
│
├── notebooks/
│   └── data_exploration.ipynb       # Dataset analysis
│
├── results/                          # Analysis outputs
│   ├── figures/
│   ├── metrics/
│   └── activations/
│
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Download the dataset from [Kaggle: AI vs Human Text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text) and place it at:

```
data/raw/AI_Human.csv
```

### 3. Run Training Pipeline

The **recommended way** to run training is using the config-based pipeline:

```bash
# Run complete pipeline (tokenize → encode → train)
python scripts/run_training.py baseline
```

**What this does:**
1. Tokenizes the dataset using BERT tokenizer → saves to `data/processed/AI_Human/tokenized/bert-base-uncased/`
2. Encodes using BERT model → saves to `data/processed/AI_Human/encoded/bert_bert/`
3. Trains SGD classifier → logs metrics to console (and wandb if enabled)

**On subsequent runs:**
- ✅ Cached datasets are automatically reused
- ✅ Only the training step runs
- ✅ No wasted compute on re-tokenization or re-encoding

### 4. Weights & Biases Integration (Optional)

Track your experiments with Weights & Biases:

```bash
# Install wandb (optional)
pip install wandb

# Run with wandb logging
python scripts/run_training.py baseline --wandb-project my-text-classification

# Customize run name
python scripts/run_training.py baseline --wandb-project my-project --wandb-run-name bert-baseline
```

### 5. Force Re-processing (if needed)

```bash
# Force re-tokenization (e.g., after changing max_length)
python scripts/run_training.py baseline --force-retokenize

# Force re-encoding (e.g., after changing encoder model)
python scripts/run_training.py baseline --force-reencode
```

## 📝 Configuration System

Experiments are defined via YAML files in `configs/`. The system uses **hierarchical composition** to keep configs DRY.

### Creating a New Experiment

**Example: RoBERTa encoder experiment**

1. Create tokenizer config `configs/tokenizers/roberta.yaml`:
```yaml
name: roberta-base
max_length: 512
padding: max_length
truncation: true
batch_size: 1000
```

2. Create encoder config `configs/encoders/roberta.yaml`:
```yaml
name: roberta-base
batch_size: 64
```

3. Create experiment config `configs/experiments/roberta_exp.yaml`:
```yaml
experiment_name: roberta_experiment
random_state: 42

# Component references
tokenizer: roberta
encoder: roberta
classifier: sgd

# Data settings
dataset:
  name: AI_Human
  file: AI_Human.csv
  text_column: text
  label_column: generated
  test_size: 0.2

# Training settings
training:
  batch_size: 64
  epochs: 1
  eval_every: 50

# Paths
paths:
  raw_data: data/raw
  processed_data: data/processed
  models: models
  results: results
```

4. Run it:
```bash
python scripts/run_training.py roberta_exp
```

See `configs/README.md` for more details.

## 🔧 Alternative: Standalone Scripts

For debugging or one-off runs, use the standalone scripts:

```bash
# Step 1: Tokenize
python scripts/tokenize_dataset.py

# Step 2: Encode
python scripts/encode_dataset.py

# Step 3: Train
python scripts/train_classifier.py
```

**Note:** These scripts use hardcoded paths and don't benefit from the caching system.

## 📊 Dataset

**Source:** [Kaggle - AI vs Human Text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text)

**Description:** Binary classification dataset with text samples labeled as:
- `0` = Human-written
- `1` = AI-generated

**Statistics:** (Run `python scripts/load_dataset.py` to see full stats)

**Preprocessing:**
- Train/test split: 80/20 (stratified)
- Tokenization: HuggingFace transformers (configurable)
- Max sequence length: 512 tokens (configurable)
- Storage format: HuggingFace Datasets (efficient Arrow format)

## 🧪 Experiment Workflow

### How Caching Works

The pipeline uses **path-based caching** with metadata tracking:

```
1. Check if tokenized data exists for this tokenizer config
   → If yes: load from cache
   → If no: tokenize and save with metadata

2. Check if encoded data exists for this tokenizer+encoder combo
   → If yes: load from cache
   → If no: encode and save with metadata

3. Always train a new classifier (produces timestamped experiment)
```

### Dataset Versioning

Datasets are organized by their configuration:

```
data/processed/AI_Human/
├── tokenized/
│   ├── bert-base-uncased/          # BERT tokenizer output
│   │   ├── train/
│   │   ├── test/
│   │   └── metadata.json
│   └── roberta-base/                # RoBERTa tokenizer output
│       └── ...
└── encoded/
    ├── bert_bert/                   # BERT tok + BERT enc
    │   ├── train/
    │   ├── test/
    │   └── metadata.json
    └── bert_roberta/                # BERT tok + RoBERTa enc
        └── ...
```

Each `metadata.json` contains:
- Timestamp of creation
- Full configuration used
- Provenance information

## 🛠️ Tech Stack

- **Deep Learning Framework:** PyTorch 2.0+
- **NLP & Transformers:** HuggingFace Transformers & Datasets
- **Classical ML:** scikit-learn
- **Data Processing:** pandas, numpy
- **Configuration:** PyYAML
- **Visualization:** matplotlib, seaborn
- **Progress Tracking:** tqdm

## 📦 Key Components

### `utils/dataset_tokenizer.py`
- Handles text tokenization using HuggingFace tokenizers
- Performs train/test splitting
- Saves tokenized datasets in HuggingFace Dataset format

### `utils/dataset_encoder.py`
- Loads pre-trained transformer models
- Extracts embeddings (CLS token or mean pooling)
- Supports batch processing with GPU acceleration
- Automatically detects BERT-family models for CLS pooling

### `utils/classifier_trainer.py`
- Generic trainer for sklearn classifiers
- Supports incremental training with `partial_fit`
- Provides batch-wise evaluation during training
- Saves trained models as pickle files

### `scripts/run_training.py`
- **Main training pipeline** orchestrator
- Loads and merges hierarchical configs
- Implements smart caching logic
- Integrates with Weights & Biases for experiment tracking
- No local model/metrics saving (use wandb or standalone scripts for that)

## 🎯 Supported Models

### Tokenizers/Encoders (any HuggingFace model)
- ✅ BERT (`bert-base-uncased`, `bert-large-uncased`)
- ✅ DistilBERT (`distilbert-base-uncased`)
- ✅ RoBERTa (`roberta-base`, `roberta-large`)
- ✅ ALBERT, ELECTRA, etc.
- ✅ Any model with `AutoTokenizer` and `AutoModel` support

### Classifiers (any sklearn classifier)
- ✅ SGDClassifier (fast, online learning)
- ✅ LogisticRegression
- ✅ SVC (Support Vector Classifier)
- ✅ RandomForestClassifier
- ✅ MLPClassifier (Neural Network)
- ✅ Any classifier with `fit` or `partial_fit` method

## 📈 Results & Experiments

Training metrics are logged to:
- **Console output** (train/test accuracy during training)
- **Weights & Biases** (if `--wandb-project` specified)
  - Config tracking
  - Metrics over time (train_acc, test_acc)
  - Final performance metrics

For model persistence, use the standalone `scripts/train_classifier.py` script which saves to disk.

## 🔬 Example Use Cases

### Compare Different Encoders
```bash
# Experiment 1: BERT encoder
python scripts/run_training.py baseline --wandb-project text-clf --wandb-run-name bert-sgd

# Experiment 2: DistilBERT encoder (reuses BERT tokenization if same tokenizer)
# Create configs/experiments/distilbert_exp.yaml first
python scripts/run_training.py distilbert_exp --wandb-project text-clf --wandb-run-name distilbert-sgd
```

### Compare Different Classifiers
```bash
# Create configs/classifiers/logistic.yaml
# Create configs/experiments/bert_logistic.yaml
python scripts/run_training.py bert_logistic --wandb-project text-clf

# Since tokenization and encoding are cached, this runs very fast!
```

### Hyperparameter Tuning
Edit the experiment config to change:
- `training.batch_size`
- `training.epochs`
- `classifier_config.params` (classifier hyperparameters)

## 🚧 Future Enhancements

- [ ] Add PyTorch classifier support (currently sklearn only)
- [ ] Implement metrics tracking (accuracy, F1, confusion matrix)
- [ ] Add visualization tools for embeddings (PCA, t-SNE)
- [ ] Support for ensemble classifiers
- [ ] Integration with experiment tracking tools (MLflow, Weights & Biases)
- [ ] Add evaluation-only mode for trained models
- [ ] Support for custom preprocessing functions

## 🤝 Contributing

To add new components:

1. **New Tokenizer:** Add YAML config in `configs/tokenizers/`
2. **New Encoder:** Add YAML config in `configs/encoders/`
3. **New Classifier:** Add YAML config in `configs/classifiers/`
4. **New Experiment:** Compose existing components in `configs/experiments/`

## 📄 License

MIT License - Feel free to use this for research and education!

## 📚 References

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets)
- [scikit-learn](https://scikit-learn.org/)
- [Dataset Source: Kaggle AI vs Human Text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text)

---

**Status:** ✅ Production-ready with baseline BERT experiment

**Last Updated:** November 2025
