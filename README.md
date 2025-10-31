# Interpreting Layer Activations in Human-vs-AI Text Classification

A deep learning research project analyzing how different neural architectures (GRU, LSTM, CNN) represent and distinguish between human-written and AI-generated text.

## 📋 Project Overview

This project investigates how neural networks internally represent textual patterns that distinguish human-written from AI-generated content. We train multiple encoder architectures and feed their learned representations to classical ML classifiers, then visualize and interpret the layer activations.

**Key Components:**
- Train GRU, LSTM, and CNN encoders on binary text classification
- Extract latent representations from trained models
- Feed embeddings to classical ML classifiers (Decision Tree, SVM, Logistic Regression)
- Interpret and visualize layer activations during inference
- Compare how different architectures "see" human vs AI text

## 🗂️ Repository Structure

```
human-vs-ai-text/
│
├── data/                          # Dataset storage
│   ├── raw/                       # Raw CSV files from Kaggle
│   └── processed/                 # Tokenized and split data
│
├── models/                        # Neural network definitions
│   └── (Phase 2: GRU, LSTM, CNN encoders)
│
├── scripts/                       # Executable scripts
│   ├── load_dataset.py           # Quick data loader and stats
│   ├── preprocess_data.py        # Full preprocessing pipeline
│   └── verify_setup.py           # Environment verification
│
├── utils/                         # Helper modules
│   ├── text_preprocessing.py     # Text processing utilities
│   ├── dataset_loader.py         # PyTorch Dataset & DataLoader
│   └── __init__.py
│
├── notebooks/                     # Jupyter notebooks
│   └── data_exploration.ipynb    # Initial data analysis
│
├── results/                       # Outputs and visualizations
│   ├── figures/                  # Plots and charts
│   ├── metrics/                  # Performance metrics
│   └── activations/              # Activation maps
│
├── requirements.txt               # Python dependencies
├── SETUP.md                       # Setup instructions
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Setup Environment
...

### 2. Download Dataset

Download from [Kaggle: AI vs Human Text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text) and place the CSV in `data/raw/`.

### 3. Explore Data

```bash
jupyter notebook notebooks/data_exploration.ipynb
```

### 4. Preprocess Data

After implementing the preprocessing functions:

```bash
python scripts/preprocess_data.py
```

## 📊 Dataset

**Source:** [Kaggle - AI vs Human Text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text)

**Description:** Text samples labeled as either human-written (0) or AI-generated (1).

**Preprocessing:**
- Train/test split (80/20)
- Tokenization using Hugging Face transformers (`distilbert-base-uncased`)
- Max sequence length: 512 tokens
- Saved as PyTorch tensors for efficient loading

## 🧩 Project Phases

### ✅ Phase 1 — Setup & Preprocessing (CURRENT)
- [x] Create project structure
- [x] Setup virtual environment
- [x] Prepare skeleton scripts with hints
- [ ] **YOUR TASK:** Implement preprocessing functions
- [ ] **YOUR TASK:** Run data exploration notebook

### Phase 2 — Model Training
- Implement GRU, LSTM, CNN encoders
- Extract embeddings from trained encoders
- Train classical ML classifiers on embeddings
- Evaluate and compare performance

### Phase 3 — Activation Analysis
- Modify encoders to capture intermediate activations
- Collect activations for sample inputs
- Visualize via PCA/t-SNE and heatmaps
- Analyze patterns in human vs AI representations

### Phase 4 — Visualization & Reporting
- Generate comprehensive performance plots
- Create activation maps
- Document key insights

### Phase 5 — Finalization
- Clean and organize code
- Update README with results
- Tag release v1.0

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch
- **NLP:** Hugging Face Transformers
- **Classical ML:** scikit-learn
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Development:** Jupyter notebooks

## 📝 Implementation Notes

Phase 1 files are provided as **skeletons with hints** - you'll implement the logic! This gives you:
- Clear function signatures and docstrings
- Helpful hints on what to use
- TODOs marking what needs implementation
- Freedom to write the actual code

**Files to implement:**
- `utils/text_preprocessing.py` - Data loading and tokenization
- `utils/dataset_loader.py` - PyTorch Dataset wrapper
- `scripts/load_dataset.py` - Simple data inspection
- `scripts/preprocess_data.py` - Full pipeline
- `notebooks/data_exploration.ipynb` - Visual exploration

## 📈 Expected Outputs (Phase 1)

After preprocessing, you should have:
- `data/processed/train_encodings.pt`
- `data/processed/test_encodings.pt`
- `data/processed/train_labels.npy`
- `data/processed/test_labels.npy`
- (Optional) CSV files for reference

## 🤝 Contributing

This is a research project. Feel free to experiment with:
- Different tokenizers (BERT, RoBERTa, GPT-2)
- Various max sequence lengths
- Alternative preprocessing strategies
- Additional exploratory analyses

## 📄 License

MIT License - Feel free to use this for your own research!

## 🎯 Research Goals

1. **Understand representation differences:** How do GRU, LSTM, and CNN encode textual patterns differently?
2. **Identify discriminative features:** What layer activations are most important for distinguishing human vs AI text?
3. **Compare architectures:** Which architecture provides the most interpretable/separable representations?
4. **Visualize learned patterns:** Can we see what the models "look for" in human vs AI text?

---

**Current Status:** Phase 1 - Setup skeleton files created, ready for implementation! 🚀
