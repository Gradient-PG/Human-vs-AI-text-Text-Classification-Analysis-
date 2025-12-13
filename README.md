# Human vs AI Text Classification Analysis

A modular text classification system for distinguishing human-written from AI-generated text using transformer encoders and classical ML classifiers, with **neural activation analysis** for interpretability.

## 📋 Overview

This project implements a **config-driven pipeline** for text classification experiments that combines high accuracy with interpretability through neural activation analysis. It separates the encoding phase (using transformer models like BERT) from the classification phase (using sklearn classifiers), allowing flexible experimentation while maintaining computational efficiency.

**Architecture:**
1. **Tokenization**: Convert raw text to tokens using HuggingFace tokenizers
2. **Encoding**: Generate embeddings using transformer models (BERT, DistilBERT, etc.)
3. **Classification**: Train lightweight classifiers on frozen embeddings (SGD, Logistic Regression, etc.)
4. **Analysis**: Extract and analyze layer-wise activations to understand what the encoder "sees"

**Key Features:**
- ✅ Configuration-based experiments via YAML files
- ✅ Automatic caching to avoid redundant computation
- ✅ Modular pipeline with independent tokenizer/encoder/classifier components
- ✅ Support for any HuggingFace tokenizer and transformer encoder
- ✅ Compatible with sklearn classifiers
- ✅ Neural activation capture and analysis for interpretability
- ✅ GPU-efficient frozen encoder approach

## 📄 Research Abstract

**Title:** *Interpretable AI Text Detection Through Neural Activation Analysis*

This study creates a high-accuracy AI text classifier while simultaneously revealing **what linguistic patterns BERT learns** to distinguish AI-generated from human-written text. Using frozen BERT embeddings with simple linear classifiers, we achieve **98.8% test accuracy** without fine-tuning, demonstrating that pre-trained transformers already capture highly discriminative semantic representations.

### Multi-Level Interpretability Framework

Our analysis goes beyond classification performance through systematic neuron-level investigation:

**1. Statistical Neuron Analysis** (✅ Implemented)
- Extract activations from all 12 BERT layers for balanced samples
- Per-neuron significance testing (t-tests, Cohen's d effect sizes)
- **Finding**: 63.7% of neurons show significant discriminative power (p<0.001)
- Layer importance ranking reveals late layers (9-11) dominate detection

**2. Token-Level Attribution** (🚧 Planned)
- Identify which specific words/phrases trigger discriminative neurons
- Discriminative vocabulary analysis (AI vs. human word usage)
- Position bias detection (text structure patterns)

**3. Linguistic Feature Correlation** (🚧 Planned)
- Map neurons to interpretable properties: lexical diversity, formality, uniformity
- Answer: "Neuron X in Layer Y detects [specific linguistic feature]"
- Bridge neural patterns to human-understandable concepts

**4. Causal Validation** (🚧 Planned)
- Ablation studies prove neurons are causally important
- Masking top discriminative neurons → measure classification impact
- Establish causal importance, not just correlation

### Key Contributions

✅ **High accuracy with minimal resources**: 98.8% without fine-tuning, runs on 4GB GPU  
✅ **Neuron-level interpretability**: 63.7% significantly discriminative neurons identified  
✅ **Layer-wise insights**: Late semantic layers (9-11) drive detection, not syntax  
✅ **Linear separability**: Classes are linearly separable → interpretable boundaries  
🚧 **Linguistic grounding**: Upcoming correlation with interpretable text features  
🚧 **Causal evidence**: Planned ablation studies validate neuron importance

This work makes deep interpretability research accessible on consumer hardware while bridging neural mechanisms to linguistic theory.

**Key Contributions:**
- High-accuracy classification with minimal computational overhead (no fine-tuning needed)
- Neuron-level interpretability framework for transformer-based detection
- Statistical identification of discriminative neural patterns
- Correlation analysis between activations and linguistic features
- Open-source, modular pipeline for reproducible research

## 🔬 Research Questions

This project goes beyond simple classification to explore **what linguistic patterns distinguish AI-generated from human-written text at the neural level**:

1. **Neuron-level discrimination**: Do specific neurons or groups consistently activate differently for AI vs. human text? Can we quantify their discriminative power?
2. **Layer-wise evolution**: How do activation patterns evolve from early (syntax) to late (semantics) transformer layers? Which layers are most important for classification?
3. **Linguistic correlates**: What interpretable features (repetition, formality, coherence, vocabulary richness) correlate with discriminative neuron activations?
4. **Classifier mechanisms**: Which encoder features do different classifiers (linear vs. tree-based) rely on? Is there consensus or diversity in feature usage?
5. **Embedding geometry**: Are AI and human texts linearly separable in BERT space? What does this reveal about class characteristics?

**Methodological Approach**: By analyzing activations from a frozen BERT encoder, we can identify which internal representations are most informative for detection, providing insights into AI text characteristics without requiring expensive fine-tuning or large-scale model comparisons. This makes deep interpretability research accessible on personal hardware.

## 🗂️ Project Structure

```
Human-vs-AI-text-Text-Classification-Analysis-/
│
├── configs/                          # Experiment configurations
│   ├── experiments/                  # Main experiment configs
│   │   ├── sgd.yaml                 # BERT + SGD classifier
│   │   ├── linear_svc.yaml          # BERT + Linear SVC
│   │   ├── random_forest.yaml       # BERT + Random Forest
│   │   ├── logistic_regression.yaml # BERT + Logistic Regression
│   │   └── decision_tree.yaml       # BERT + Decision Tree
│   ├── tokenizers/                   # Tokenizer configurations
│   │   └── bert.yaml
│   ├── encoders/                     # Encoder model configurations
│   │   └── bert.yaml
│   └── classifiers/                  # Classifier configurations
│       ├── sgd.yaml
│       ├── linear_svc.yaml
│       ├── random_forest.yaml
│       ├── logistic_regression.yaml
│       └── decision_tree.yaml
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
│   ├── extract_activations.py       # 🔬 Extract layer-wise activations
│   ├── analyze_activations.py       # 🔬 Statistical neuron analysis
│   ├── visualize_activations.py     # 🔬 Generate analysis figures
│   ├── compare_experiments.py       # 🔬 Cross-model comparison
│   ├── tokenize_dataset.py          # Standalone tokenization script
│   ├── encode_dataset.py            # Standalone encoding script
│   ├── train_classifier.py          # Standalone training script
│   └── load_dataset.py              # Quick dataset inspection
│   # 🚧 Planned additions:
│   # ├── analyze_token_attributions.py
│   # ├── correlate_linguistic_features.py
│   # ├── ablate_neurons.py
│   # └── visualize_layer_trajectories.py
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

## 📊 Research Methodology

### Phase 1: Baseline Classification (✅ Complete)
**Goal:** Establish that frozen BERT embeddings contain sufficient information for AI text detection

**Approach:**
1. Train multiple sklearn classifiers on frozen BERT-base-uncased embeddings
2. Compare performance: SGD, Logistic Regression, Linear SVC, Random Forest, Decision Tree
3. Establish baseline: ~98% test accuracy with linear models

**Key Finding:** The CLS token embedding alone is highly discriminative, suggesting strong class-level semantic differences captured by BERT.

### Phase 2: Activation Pattern Analysis (🚧 In Progress)
**Goal:** Identify which neurons and layers are most discriminative

**Methods:**
1. **Activation Capture**: Extract hidden states from all 12 BERT layers for balanced AI/human samples
2. **Statistical Testing**: For each neuron, compute mean activation per class and significance (t-test, Cohen's d)
3. **Ranking**: Identify top K discriminative neurons per layer
4. **Visualization**: Generate heatmaps showing neuron × token activation patterns

**Expected Outcomes:**
- Discriminative neurons concentrated in specific layers (likely 8-11 for semantic features)
- Different patterns for AI text: higher uniformity, specific linguistic markers
- Interpretable neuron groups corresponding to linguistic properties

### Phase 3: Linguistic Feature Correlation (🔜 Planned)
**Goal:** Connect neural activations to interpretable text features

**Approach:**
1. Extract linguistic features: vocabulary richness, sentence length variability, repetition metrics, formality markers
2. Correlate feature values with discriminative neuron activations
3. Perform ablation: mask high-activation neurons and measure classification impact

**Expected Insights:**
- AI text markers: lower lexical diversity, more uniform sentence structure, specific phrase patterns
- Map neurons to linguistic properties (e.g., "neuron 234 in layer 10 detects formal academic style")

### Phase 4: Classifier Decision Analysis (🔜 Planned)
**Goal:** Understand what embedding dimensions different classifier types prioritize

**Approach:**
1. **Linear Models**: Analyze coefficient weights across 768 BERT dimensions
2. **Tree Models**: Extract feature importances
3. **Cross-Model Agreement**: Measure prediction correlation on AI vs. human subsets
4. **Dimensionality Reduction**: PCA/t-SNE visualization of embedding space with class labels

**Expected Findings:**
- Linear models leverage specific embedding dimensions (identifiable via high coefficients)
- Tree models may capture non-linear combinations
- High agreement on "obvious" cases, disagreement on borderline texts

## 📈 Results & Experiments

### Classification Performance

Current baseline results (BERT-base + sklearn classifiers):

| Classifier | Test Accuracy | Notes |
|------------|---------------|-------|
| Linear SVC | 98.82% | Best performance, linear decision boundary |
| Logistic Regression | 98.79% | Nearly identical to SVC |
| SGD Classifier | 98.30% | Online learning, efficient |
| Random Forest | 98.14% | Tree-based, feature importances |
| Decision Tree | 91.10% | Single tree, interpretable but lower accuracy |

**Key Insight:** Linear models perform best, suggesting classes are linearly separable in BERT embedding space. This supports using simpler models for interpretability.

### Experiment Tracking

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

## 🧠 Activation Analysis (Research Component)

This project includes a **comprehensive interpretability framework** for analyzing what BERT learns about AI vs. human text through multi-level neural analysis.

### Analysis Capabilities

#### 1. Layer-wise Activation Capture
Extract hidden states from all 12 BERT layers to understand representation evolution:
- **Early layers (1-4)**: Syntax, grammar, token-level patterns
- **Middle layers (5-8)**: Semantic composition, phrase structure
- **Late layers (9-12)**: Document-level semantics, coherence
- **Current status**: ✅ Implemented

#### 2. Discriminative Neuron Identification
Statistical analysis identifying neurons with significantly different activations:
- Per-neuron t-tests and Cohen's d effect sizes
- Discriminative score: |d| × -log10(p-value)
- Layer-wise importance ranking
- **Finding**: 63.7% of neurons show significant discriminative power (p<0.001)
- **Current status**: ✅ Implemented

#### 3. Token-Level Attribution Analysis
Identify which specific words/phrases activate discriminative neurons:
- Per-token activation patterns across sequences
- Attention-weighted importance scores
- Discriminative word clouds (AI vs. human vocabulary)
- Position bias detection (beginning/middle/end effects)
- **Current status**: 🚧 Planned (Priority 1)

#### 4. Linguistic Feature Correlation
Connect neuron activations to interpretable text properties:
- Correlate with lexical diversity, sentence uniformity, formality, repetition
- Map neurons to semantic functions ("Neuron X detects formality")
- Feature-importance analysis across layers
- **Current status**: 🚧 Planned (Priority 2 - High Value)

#### 5. Neuron Clustering & Interpretation
Group neurons by functional similarity:
- K-means clustering on activation patterns
- Semantic labeling of clusters ("Formality Detectors," "Coherence Encoders")
- Cluster-level importance for classification
- **Current status**: ✅ Clustering implemented, 🚧 Interpretation planned

#### 6. Causal Validation via Ablation
Prove neurons are causally important, not just correlated:
- Mask top discriminative neurons → measure accuracy drop
- Compare against random neuron masking (control)
- Neuron importance ranking by causal impact
- **Current status**: 🚧 Planned (Priority 5 - Essential for strong claims)

#### 7. Cross-Layer Information Flow
Visualize how activations evolve through the network:
- Track specific neurons' activations across layers
- Identify divergence points (where AI/human representations separate)
- Critical layer detection
- **Current status**: 🚧 Planned (Priority 4)

#### 8. Attention Pattern Analysis
Analyze attention heads in addition to activations:
- Extract attention weights from all layers/heads
- Compare attention distributions (AI vs. human)
- Head-level importance scoring
- **Current status**: 🚧 Future work (Optional)

### Running Activation Analysis

**Core workflow (✅ Currently available):**

```bash
# Step 1: Train classifier (creates encoded dataset)
python scripts/run_training.py baseline

# Step 2: Extract layer-wise activations for analysis
# This samples 1000 AI + 1000 human texts and saves all 12 layer activations
python scripts/extract_activations.py --config baseline --split test --max-samples 1000

# Step 3: Analyze discriminative patterns
# Replace TIMESTAMP with the directory created in step 2
python scripts/analyze_activations.py --activation-dir results/activations/baseline_TIMESTAMP

# Step 4: Generate visualizations
python scripts/visualize_activations.py --analysis-dir results/analysis/baseline_TIMESTAMP

# Optional: Run clustering on specific layers
python scripts/analyze_activations.py --activation-dir results/activations/baseline_TIMESTAMP --cluster-layers "9,10,11" --n-clusters 10
```

**Extended workflow (🚧 Planned expansions):**

```bash
# Step 5: Token-level attribution analysis
python scripts/analyze_token_attributions.py --activation-dir results/activations/baseline_TIMESTAMP

# Step 6: Linguistic feature correlation
python scripts/correlate_linguistic_features.py --activation-dir results/activations/baseline_TIMESTAMP

# Step 7: Ablation study (causal validation)
python scripts/ablate_neurons.py --activation-dir results/activations/baseline_TIMESTAMP --top-k 50

# Step 8: Cross-layer information flow
python scripts/visualize_layer_trajectories.py --activation-dir results/activations/baseline_TIMESTAMP
```

**Advanced options:**

```bash
# Extract only specific layers (saves disk space)
python scripts/extract_activations.py --config baseline --split test --max-samples 1000 --layers "8,9,10,11"

# Use larger sample size (if you have GPU memory)
python scripts/extract_activations.py --config baseline --split test --max-samples 5000 --batch-size 32

# Extract from train split
python scripts/extract_activations.py --config baseline --split train --max-samples 1000

# Full sequence extraction for token-level analysis (requires more VRAM)
python scripts/extract_activations.py --config baseline --split test --max-samples 1000 --extract-sequences
```

### Research Findings Structure

Results are saved to `results/`:
```
results/
├── activations/                           # Raw activation data
│   └── {experiment}_{timestamp}/
│       ├── layer_0.npz                   # Per-layer activations (CLS token)
│       ├── layer_11.npz
│       ├── sequences/                     # Full sequence activations (optional)
│       │   ├── layer_0_sequences.npz
│       │   └── ...
│       └── metadata.json                 # Sample info & config
│
├── analysis/                              # Statistical & correlation analysis
│   └── {experiment}_{timestamp}/
│       ├── discriminative_neurons.csv    # All neurons with stats
│       ├── top_50_neurons.csv           # Most discriminative
│       ├── layer_importance.csv         # Layer-wise metrics
│       ├── neuron_clusters.csv          # Clustering results
│       ├── linguistic_correlations.csv  # Neuron-feature correlations (planned)
│       ├── token_attributions/          # Token-level analysis (planned)
│       │   ├── top_tokens_per_neuron.csv
│       │   └── discriminative_words.json
│       ├── ablation_results.json        # Causal validation (planned)
│       └── summary.json                 # High-level statistics
│
└── figures/                               # Publication-ready visualizations
    ├── layer_wise_importance.png         # 4-panel layer metrics
    ├── top_discriminative_neurons.png    # Bar charts
    ├── activation_distributions.png      # Histograms
    ├── layer_neuron_heatmap.png         # Discriminative score heatmap
    ├── cohens_d_distribution.png        # Effect size violin plots
    ├── token_attribution_heatmap.png    # Token × neuron (planned)
    ├── linguistic_correlation_matrix.png # Feature × neuron (planned)
    ├── layer_trajectories.png           # Cross-layer flow (planned)
    └── ablation_curves.png              # Causal validation (planned)
```

### GPU Memory Considerations

The activation capture is designed for **personal GPU** constraints:
- Processes in batches (configurable size)
- Saves per-layer to disk (not all in memory)
- Uses float16 precision when possible
- Subsample large datasets (e.g., 1000 samples per class for analysis)

**Memory requirements by analysis type:**

| Analysis Type | VRAM Required | Time | Notes |
|--------------|---------------|------|-------|
| **CLS token only** (current) | 4GB | 5 min | Default, most analyses |
| **Full sequences** (token-level) | 6-8GB | 10 min | For token attribution |
| **Attention patterns** | 8-10GB | 15 min | Optional, future work |
| **Ablation experiments** | 4GB | 20 min | Re-training required |

**Typical memory usage (CLS token mode):**
- BERT-base model: ~1.5GB
- Batch of 32 samples: ~2GB activations (all layers)
- Total: ~4GB VRAM (fits GTX 1650, RTX 3050+)

**For full sequence mode (token-level analysis):**
- BERT-base model: ~1.5GB
- Batch of 16 samples × 512 tokens: ~4GB activations
- Total: ~6GB VRAM (fits RTX 3060+)

## 🗺️ Research Roadmap

### Activation Analysis Enhancements

**See [Activation Expansion Plan](docs/ACTIVATION_EXPANSION_PLAN.md) for detailed implementation guide.**

#### ✅ Phase 1: Baseline Analysis (Complete)
- [x] Layer-wise activation extraction
- [x] Per-neuron statistical testing (t-tests, Cohen's d)
- [x] Discriminative score computation
- [x] Layer importance ranking
- [x] Basic neuron clustering
- [x] 5 visualization types (heatmaps, distributions, rankings)

#### 🚧 Phase 2: Deep Interpretability (Planned - 2 weeks)
**High priority, high impact**

- [ ] **Token-level attribution**: Identify which words activate discriminative neurons
  - Per-token activation heatmaps
  - Discriminative vocabulary extraction (AI vs. human word clouds)
  - Position bias analysis (beginning/middle/end patterns)
  
- [ ] **Linguistic feature correlation**: Map neurons to interpretable properties
  - Lexical diversity (type-token ratio, vocabulary richness)
  - Sentence uniformity (length variability, structure patterns)
  - Formality scoring (academic markers, discourse features)
  - Repetition metrics (n-gram redundancy)
  - Correlation analysis (Pearson r between features and activations)
  
- [ ] **Neuron cluster interpretation**: Semantic labeling of neuron groups
  - "Formality Detectors", "Coherence Encoders", "Repetition Sensors"
  - Feature profiles per cluster
  - Example texts that maximally activate each cluster

**Estimated effort**: 2 weeks | **GPU required**: 6GB VRAM | **Research value**: ⭐⭐⭐⭐⭐

#### 🔜 Phase 3: Causal Validation (Planned - 1 week)
**Essential for strong research claims**

- [ ] **Ablation studies**: Prove neurons are causally important
  - Mask top K discriminative neurons → measure accuracy drop
  - Control: random neuron masking
  - Neuron importance ranking by causal impact
  - Expected: >10% accuracy drop when masking top 5%

- [ ] **Cross-layer information flow**: Visualize representation evolution
  - Track specific neurons across all layers
  - Identify divergence points (where AI/human separate)
  - Critical layer detection

**Estimated effort**: 1 week | **GPU required**: 4-6GB VRAM | **Research value**: ⭐⭐⭐⭐⭐

#### ⏸️ Phase 4: Generalization (Future - 2-3 weeks)
**Validate robustness across models and datasets**

- [ ] Cross-model comparison (DistilBERT, RoBERTa, ELECTRA)
- [ ] Cross-dataset validation (GPT-4 text, Claude text, domain-specific)
- [ ] Universal vs. dataset-specific discriminative neurons
- [ ] Attention pattern analysis (optional, separate research direction)

**Estimated effort**: 3 weeks | **GPU required**: 6-8GB VRAM | **Research value**: ⭐⭐⭐⭐

---

### Core Pipeline Enhancements

**Lower priority, incremental improvements**

- [ ] PyTorch classifier support (currently sklearn only)
- [ ] Comprehensive metrics tracking (F1, ROC, confusion matrix with wandb logging)
- [ ] Ensemble classifier support
- [ ] Evaluation-only mode for trained models
- [ ] Custom preprocessing function hooks
- [ ] Multi-GPU support for large-scale extraction

---

### Publication Targets by Phase

| Phase | Completion | Venue | Paper Type |
|-------|-----------|-------|------------|
| Phase 1 (✅) | Complete | Workshop | Short (4 pages) |
| Phase 1-2 (🚧) | +2 weeks | EMNLP/NAACL | Short/Long (6-8 pages) |
| Phase 1-3 (🔜) | +3 weeks | ACL/EMNLP | Long (8 pages) |
| Phase 1-4 (⏸️) | +6 weeks | ACL/EMNLP/TACL | Long/Journal |

**Current status**: Ready for workshop submission; 2-3 weeks from main conference submission

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

## 📖 Documentation

- **[Main README](README.md)** - Project overview and quick start
- **[Activation Analysis Guide](docs/ACTIVATION_ANALYSIS_GUIDE.md)** - Step-by-step guide for neuron-level interpretability
- **[Activation Expansion Plan](docs/ACTIVATION_EXPANSION_PLAN.md)** - 🆕 Comprehensive roadmap for analysis enhancements
- **[Research Abstract](docs/RESEARCH_ABSTRACT.md)** - Complete research methodology and findings
- **[Project Summary](docs/PROJECT_SUMMARY.md)** - Technical overview and use cases
- **[Config Guide](configs/README.md)** - Configuration system documentation

## 🎓 For Researchers

If you're using this project for research, here's the recommended workflow:

### Phase 1: Baseline Classification (Day 1)
```bash
# Train baseline classifier
python scripts/run_training.py baseline --wandb-project ai-text-detection

# Expected: ~98% accuracy in ~5 minutes on GPU
```

### Phase 2: Activation Analysis (Day 2-3)
```bash
# Extract activations
python scripts/extract_activations.py --config baseline --split test --max-samples 1000

# Analyze patterns
python scripts/analyze_activations.py --activation-dir results/activations/baseline_TIMESTAMP

# Generate figures
python scripts/visualize_activations.py --analysis-dir results/analysis/baseline_TIMESTAMP
```

### Phase 3: Linguistic Feature Analysis (Day 4-5)
```python
# In notebooks/linguistic_analysis.ipynb
# - Extract text features (vocabulary richness, sentence length, etc.)
# - Correlate with top neuron activations
# - Identify what discriminative neurons "detect"
```

### Phase 4: Write-up (Day 6-7)
- Use figures from `results/figures/`
- Use statistics from `results/analysis/*/summary.json`
- Refer to [Research Abstract](docs/RESEARCH_ABSTRACT.md) for structure
- All figures are 300 DPI publication-ready

**Total time:** ~1 week for complete interpretability study

**Hardware:** Works on personal GPU (4GB+ VRAM)

**Output:** 5 publication-ready figures + comprehensive statistical analysis

---

**Status:** ✅ Production-ready with baseline classification and activation analysis pipeline

**Last Updated:** December 2025
