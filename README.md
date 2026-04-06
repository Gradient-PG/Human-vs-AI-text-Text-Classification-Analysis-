# Discriminative Neurons in BERT for AI-Generated Text Detection

A mechanistic interpretability study of a frozen `bert-base-uncased` model in the context of AI-generated text detection. The model is not fine-tuned — its frozen CLS token activations are the subject of analysis. A lightweight sklearn classifier is trained on top of those activations; the paper is about understanding the activations, not improving the classifier.

**Authors**: Paweł Blicharz & Miłosz Grunwald  
**Target venue**: EMNLP 2026 ARR · Deadline May 25, 2026

## Core Research Question

Which neurons in BERT-base-uncased CLS token activations systematically discriminate AI-generated from human-written text, what is their causal role in classification, do they generalize across generators, and is the detection signal geometrically linear?

## Why Mechanistic, Not Behavioral

High accuracy with frozen BERT + sklearn is a known, non-novel result (Linear SVC reaches ~98.8% on similar benchmarks). The paper is not about building a better detector. It is about understanding what is happening inside an existing one — which specific neurons drive the decision, and why.

## Dataset: RAID Benchmark

[RAID](https://huggingface.co/datasets/liamdugan/raid) (Dugan et al., ACL 2024). 11 AI generators across 8 domains. Replaced an earlier `wiki_labeled` GPT-3 dataset that produced trivially separable results and was therefore not scientifically interesting. The shift to RAID introduced real complexity: generators vary in difficulty, neuron counts differ per generator, and old findings did not reproduce.

## Model

**`bert-base-uncased`** — 12 layers × 768 neurons = 9,216 total neurons. CLS token activations only. Frozen weights (no fine-tuning).

---

## Confirmed Findings (RAID-Grounded)

> All findings below are on RAID data with BERT-base-uncased. Numbers from the old wiki_labeled / GPT-3 dataset (1,350 discriminative neurons, U-shaped layer distribution, three-cluster architecture) are **invalidated and fully dropped**.

| Metric | Value |
|---|---|
| Total neurons analyzed | 9,216 (12 layers × 768) |
| Discriminative neurons | Varies per generator |
| RAID generators analyzed | 11 |
| AI-preferring : human-preferring ratio | ~1:1 |
| Bonferroni-adjusted α | 3.25×10⁻⁷ (per-neuron threshold) |

**Stable results:**
- **Balanced bidirectionality** — roughly equal numbers of AI-preferring (AUC > 0.7) and human-preferring (AUC < 0.3) neurons. Pattern appears stable across generators.
- **PCA / UMAP separation** — discriminative neurons form distinct non-overlapping regions in dimensionality-reduced space.
- **Per-generator variation** — different generators produce different counts of discriminative neurons. Detection difficulty varies by generator at the mechanistic level.
- **Statistical pipeline** — AUC-ROC per neuron + two-sided Mann-Whitney U test + Bonferroni correction. Threshold: AUC < 0.3 or > 0.7. Applied per generator separately.

**Fully dropped / invalidated:** ~1,350 discriminative neurons count, U-shaped layer distribution, three-cluster architecture, 98% accuracy headline, wiki_labeled dataset, DistilBERT.

---

## Research Directions

### D1 — Causal Validation (Primary, load-bearing)

The core contribution. Upgrades the paper from "these neurons correlate with detection" to "these neurons causally drive it."

- **D1a** — Ablation study (zero + mean ablation + random baseline) — tests necessity
- **D1b** — Activation patching — tests sufficiency
- **D1c** — Integrated gradients — gradient-based attribution without modifying activations

### D2 — Cross-Generator Generalization (Co-primary, most novel)

Nobody has characterized which specific neurons generalize and which are generator-specific at the mechanistic level.

- **D2a** — Jaccard similarity matrix (11×11) — headline figure
- **D2b** — Layer distribution histogram per generator
- **D2c** — Generator properties vs. activation distances

### D3 — Linear Representation Validation (Supporting, cheap add-on)

Tests whether "AI-ness" is encoded as a clean linear direction in CLS activation space.

- **D3a** — Mean difference vector + LR weight alignment (runs immediately, no Step 0 needed)
- **D3b** — CAV per layer (nice-to-have)

### CT — Cross-Generator Transfer Test (Conditional)

Train classifier using only generator-A discriminative neurons, test on generator-B texts. Conditional on D1a + D2a completing by ~April 18–20.

---

## Excluded Directions

- **Clustering as primary analysis** — dropped; trivially produces 2 clusters, three-cluster finding was an artifact
- **Circuit analysis** — TransformerLens is decoder-centric, different research question, scope discipline
- **Sparse autoencoders (SAEs)** — no published SAE work on standard NLP BERT models, follow-up paper
- **Attention head analysis** — different decomposition of the same signal, not planned
- **Classification accuracy as headline** — known, non-novel result

---

## Project Structure

```
├── raid_analysis/                 # Core analysis library
│   ├── __init__.py                # Public API surface (re-exports from subpackages)
│   ├── constants.py               # Shared constants (layers, thresholds, palettes)
│   ├── io.py                      # Figure/text save utilities
│   ├── neurons_pipeline.py        # Neuron analysis orchestrator
│   ├── run_analysis.py            # Top-level orchestrator (all stages)
│   │
│   ├── data/                      # Data loading, statistics, derived features
│   │   ├── activations.py         # Raw activation loading from .npy files
│   │   ├── neuron_stats.py        # Mann-Whitney U + AUC per-neuron pipeline
│   │   ├── loader.py              # Load per-layer CSVs into DataFrames
│   │   ├── discriminative.py      # Discriminative neuron set helpers
│   │   └── traits.py              # Neuron × trait matrix construction
│   │
│   ├── clustering/                # Self-contained clustering (optional)
│   │   ├── strategies.py          # Ward, KMeans, HDBSCAN strategy classes
│   │   ├── pca.py                 # PCA subspace for clustering + 2D viz
│   │   └── pipeline.py            # Clustering analysis orchestrator
│   │
│   ├── viz/                       # All figure generation
│   │   ├── neurons.py             # Layer distribution, boxplots, scatter
│   │   ├── embedding.py           # 2D embedding plots (all neurons + clusters)
│   │   ├── hierarchy.py           # Dendrograms, silhouette, merge gaps
│   │   └── dim_reduction.py       # PCA/UMAP reduction strategies
│   │
│   ├── reports/                   # Text reports and exemplars
│   │   ├── summaries.py           # neuron_summary_text, clustering_summary_text
│   │   └── exemplars.py           # AUC preference-group exemplar texts
│   │
│   └── experiments/               # Primitives for research directions D1-D3
│       ├── causal.py              # D1: ablation, activation patching
│       ├── cross_generator.py     # D2: Jaccard similarity, core neurons
│       └── linear.py              # D3: mean difference vector, LR weights
│
├── raid_pipeline/                 # RAID data pipeline (loading, tokenization, extraction)
│   ├── __init__.py                # Public API surface
│   ├── raid_loader.py             # Load RAID benchmark subsets from local CSVs
│   ├── dataset_tokenizer.py       # Tokenize datasets for BERT
│   ├── activation_extractor.py    # Extract CLS activations from BERT layers
│   └── model_loader.py            # Load frozen BERT model + tokenizer
│
├── scripts/                       # CLI entry points (pipeline stages)
│   ├── download_raid.py           # Stream RAID dataset → data/raw/raid/*.csv
│   ├── tokenize_raid.py           # Load & tokenize a RAID subset
│   ├── extract_activations.py     # Extract CLS activations (all 12 layers)
│   ├── analyze_activations.py     # Mann-Whitney U + AUC, per-layer CSVs
│   ├── run_raid_pipeline.py       # Full pipeline: tokenize → extract → analyze
│   └── analyze_raid_models.py     # Neuron analysis + optional clustering + exemplars
│
├── notebooks/
│   ├── neurons_analysis.ipynb
│   └── neuron_clustering.ipynb
│
├── data/
│   ├── raw/raid/                  # RAID CSVs from download_raid.py
│   └── processed/raid_*/          # Tokenized RAID subsets
│
├── results/
│   ├── activations_raid_*/        # Per-model CLS activations + neuron stats
│   └── analysis/{model}/          # neurons/ and clustering/ outputs
│
├── pyproject.toml
├── requirements.txt
└── bert_paper_full_overview.html  # Full project overview with all experiment plans
```

---

## Reproducing the Analysis

### Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

#### Install uv

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Create Environment and Install Dependencies

```bash
uv sync

# Activate the virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

**For GPU support (recommended, 10x faster):**
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Pipeline — RAID subset (single model)

#### Step 1 — Download RAID data

```bash
uv run scripts/download_raid.py
```

#### Step 2 — Load & tokenize

```bash
uv run scripts/tokenize_raid.py --model gpt4 --max-samples 10000
```

#### Step 3 — Extract CLS activations

```bash
uv run scripts/extract_activations.py \
    --tokenized-path data/processed/raid_gpt4 \
    --output results/activations_raid_gpt4 \
    --samples 10000
```

#### Step 4 — Neuron statistics

```bash
uv run scripts/analyze_activations.py --input results/activations_raid_gpt4
```

### Multi-model pipeline

Runs tokenize → extract → analyze for each AI model against human:

```bash
uv run scripts/run_raid_pipeline.py
uv run scripts/run_raid_pipeline.py --models gpt4 chatgpt mistral-chat
uv run scripts/run_raid_pipeline.py --models gpt4 --samples 5000
```

| Flag | Default | Description |
|---|---|---|
| `--models` | all 11 AI models | Models vs human |
| `--samples` | 1,000 | Total samples per model (balanced 50/50) |
| `--domains` | all six | Restrict RAID domains |
| `--batch-size` | 1,000 | Tokenization batch size |
| `--extract-batch-size` | 64 | BERT forward batch size |
| `--seed` | 42 | Random seed |

### Multi-model neuron + clustering analysis

After activations exist under `results/activations_raid_*`:

```bash
uv run scripts/analyze_raid_models.py
uv run scripts/analyze_raid_models.py --models gpt4 chatgpt
uv run scripts/analyze_raid_models.py --clustering ward_gap hdbscan kmeans
uv run scripts/analyze_raid_models.py --exemplars
```

---

## Statistical Method

**Per-neuron analysis** across all 9,216 neurons (12 layers × 768):

1. **AUC-ROC** per neuron — AUC > 0.7 → AI-preferring; AUC < 0.3 → human-preferring
2. **Mann-Whitney U Test** — tests whether AI vs human activation distributions differ significantly
3. **Bonferroni correction** — adjusted α ≈ 3.25×10⁻⁷ per neuron
4. **Cohen's d** — supplementary effect size

---

## Library Usage

Install as an editable package so that `raid_analysis` and `raid_pipeline` are importable from any working directory (e.g. from an external experiment framework):

```bash
uv pip install -e /path/to/Human-vs-AI-text-Text-Classification-Analysis-
```

Then import programmatically:

```python
from raid_pipeline import (
    ActivationExtractor, DatasetTokenizer,
    load_raid, RAIDConfig, slug,
    ALL_RAID_MODELS, ALL_DOMAINS,
    load_bert_model, project_root,
)
from raid_analysis import (
    # Orchestration
    analyze_raid_model,
    # Data loading
    load_stats, build_full_neuron_matrix, load_activation_column,
    load_activations, load_activations_for_model,
    compute_neuron_statistics, identify_discriminative_neurons,
    get_discriminative_neuron_indices, get_discriminative_sets_per_generator,
    # Constants
    ALL_LAYERS, ALPHA, AUC_LOW, AUC_HIGH,
    # Traits
    build_trait_matrix, add_derived_neuron_columns,
    # IO & reports
    save_figure, write_text,
    neuron_summary_text, clustering_summary_text, exemplar_text,
    # D1: Causal validation
    ablate_neurons, patch_neurons,
    # D2: Cross-generator generalization
    jaccard_similarity, jaccard_matrix, core_neurons,
    # D3: Linear representation
    mean_difference_vector, lr_weight_vector,
)
```

---

## Reproducibility

This project is designed so that any collaborator can regenerate all results from scratch on any machine, given only the Git repository.

### Pinned dependencies

`uv.lock` is committed to the repository. Running `uv sync` installs the exact same package versions everywhere. Never edit `uv.lock` manually — it is updated automatically when you change `pyproject.toml`.

### Pinned model weights

All `from_pretrained` calls pin `bert-base-uncased` to a specific HuggingFace commit SHA (`raid_pipeline.model_loader.BERT_MODEL_REVISION`). This guarantees identical weights even if the upstream repository receives future updates.

### Secrets

Copy `.env.example` to `.env` and fill in your HuggingFace token:

```bash
cp .env.example .env
# edit .env → HF_TOKEN=hf_...
```

`.env` is gitignored — secrets stay local. The token is only needed for `scripts/download_raid.py` (the initial dataset download).

### Data tiers and what to keep

| Tier | Location | Size | Committed? | How to reproduce |
|------|----------|------|------------|------------------|
| Raw RAID CSVs | `data/raw/raid/` | ~2.6 GB | No (gitignored) | `uv run scripts/download_raid.py` |
| Tokenized datasets | `data/processed/raid_*/` | ~1 GB | No (gitignored) | `uv run scripts/run_raid_pipeline.py` (Step 1) |
| Layer activations (.npy) | `results/activations_raid_*/` | ~2 GB | No (gitignored) | `uv run scripts/run_raid_pipeline.py` (Step 2) |
| Neuron stats CSVs | `results/activations_raid_*/layer_*_neuron_stats.csv` | ~17 MB | No (gitignored) | `uv run scripts/run_raid_pipeline.py` (Step 3) |
| Analysis outputs | `results/analysis/` | Small | No (gitignored) | `uv run scripts/analyze_raid_models.py` |

None of the above are committed because they are fully reproducible from the pipeline scripts. To regenerate everything from scratch:

```bash
# 1. Download RAID (needs HF_TOKEN in .env, ~30 min)
uv run scripts/download_raid.py

# 2. Tokenize + extract activations + compute stats (all 11 models)
uv run scripts/run_raid_pipeline.py --samples 10000

# 3. Run neuron analysis + clustering + exemplars
uv run scripts/analyze_raid_models.py --exemplars
```

### Random seeds

All sampling and shuffling uses `--seed 42` by default. As long as the same seed, sample count, and RAID version are used, results are deterministic (modulo floating-point non-determinism on different GPU architectures).

---

## Tech Stack

**Package Management**: [uv](https://github.com/astral-sh/uv)  
**Core**: PyTorch, HuggingFace Transformers  
**Analysis**: NumPy, Pandas, SciPy, scikit-learn  
**Visualization**: matplotlib, seaborn, UMAP (optional)  
**Experiment tracking**: Weights & Biases (optional)

---

## References

- [RAID benchmark](https://arxiv.org/abs/2405.07940) (ACL 2024) — [HuggingFace](https://huggingface.co/datasets/liamdugan/raid) · [GitHub](https://github.com/liamdugan/raid)
- [BERT](https://arxiv.org/abs/1810.04805) (Devlin et al., 2019)
- Enkhbayar (2025) — discriminative neuron causal paradox in GPT-2
- [UMAP](https://umap-learn.readthedocs.io/)
- [Mann-Whitney U test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html)
