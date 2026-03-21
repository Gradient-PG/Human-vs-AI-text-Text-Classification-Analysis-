# Discriminative Neuron Clustering in BERT for AI-Generated Text Detection

An interpretability study analyzing frozen BERT to understand *which* neurons drive the detection of AI-generated text, and *how* they are functionally organized.

## Overview

**Model**: Frozen `bert-base-uncased` (12 layers, 768 neurons/layer = 9,216 total neurons)  
**Scope**: No fine-tuning. All analysis is on frozen model weights. CLS token activations only.

**Dataset**: [RAID](https://huggingface.co/datasets/liamdugan/raid) ([paper](https://arxiv.org/abs/2405.07940), ACL 2024) — the benchmark ships as large CSV splits; this repo streams a configurable subset per AI model vs human.

**Why RAID only (migration)**: The previous workflow used `NicolaiSivesind/human-vs-machine` via `scripts/tokenize_dataset.py`. That path was removed in favour of RAID, which covers more generators, domains, and adversarial settings. If you still have a tokenized copy of the old dataset on disk, you can point `extract_activations.py --tokenized-path` at it, but the repository no longer ships scripts or defaults for that dataset.

**Migration (short)**:

| Old step | RAID equivalent |
|---|---|
| `uv run scripts/tokenize_dataset.py` | `uv run scripts/tokenize_raid.py --model gpt4` (or use `run_raid_pipeline.py`) |
| `data/processed/tokenized_dataset/` | `data/processed/raid_{model}/` |
| `results/activations/` | `results/activations_raid_{model}/` or `results/activations_raid` |
| Multi-model analysis | `uv run scripts/run_raid_pipeline.py` then `uv run scripts/analyze_raid_models.py` |

**Analysis subset (results below)**: 10,000 samples (5,000 AI + 5,000 human), balanced, from RAID.

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

**Three-cluster architecture** (example: hierarchical clustering on a 2D embedding, silhouette-optimal K=3):

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
│   ├── tokenize_raid.py           # Load & tokenize a RAID subset (one AI model vs human)
│   ├── extract_activations.py     # Extract CLS activations (all 12 layers)
│   ├── analyze_activations.py     # Mann-Whitney U + AUC, per-layer CSVs
│   ├── run_raid_pipeline.py       # Full pipeline for each AI model vs human
│   └── analyze_raid_models.py     # CLI → raid_analysis package (figures + clustering)
│
├── raid_analysis/                 # Multi-model neuron + clustering analysis (refactored)
│   ├── dim_reduction.py           # PCA / UMAP strategies
│   ├── clustering.py              # Ward+silhouette, Ward+gap, KMeans strategies
│   ├── pipeline.py
│   └── ...
│
├── utils/
│   ├── activation_extractor.py
│   ├── dataset_tokenizer.py
│   └── raid_loader.py
│
├── notebooks/
│   ├── neurons_analysis.ipynb
│   └── neuron_clustering.ipynb
│
├── results/
│   ├── activations_raid_*/        # Per-model activations + neuron stats (after pipeline)
│   └── analysis/                  # analyze_raid_models.py outputs (per model)
│
├── data/
│   └── processed/raid_*/          # Tokenized RAID subsets
│
├── pyproject.toml
├── requirements.txt
└── utils/hidden.py                # HuggingFace API key (git-ignored)
```

---

## Reproducing the Analysis

### Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

#### Install uv

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or install via pip:
```bash
pip install uv
```

#### Create Environment and Install Dependencies

```bash
# Create a virtual environment and install all dependencies
uv sync

# Activate the virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

**For GPU support (recommended, 10x faster):**
```bash
# After uv sync, install PyTorch with CUDA support
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

<details>
<summary>Alternative: Using pip (legacy method)</summary>

```bash
pip install -r requirements.txt
```

Note: The project is migrating to `uv`. The `requirements.txt` is kept for backward compatibility but may be removed in future versions.
</details>

### Pipeline — RAID subset (single model)

#### Step 1 — Load & tokenize

```bash
uv run scripts/tokenize_raid.py --model gpt4 --max-samples 10000
```

Saves under `data/processed/raid_gpt4/` (or `--dataset-name`). Requires RAID CSVs from `uv run scripts/download_raid.py` first.

#### Step 2 — Extract CLS activations

```bash
uv run scripts/extract_activations.py \
    --tokenized-path data/processed/raid_gpt4 \
    --output results/activations_raid_gpt4 \
    --samples 10000
```

#### Step 3 — Neuron statistics

```bash
uv run scripts/analyze_activations.py --input results/activations_raid_gpt4
```

Optional wandb tracking:
```bash
uv run scripts/analyze_activations.py --input results/activations_raid_gpt4 --wandb-project my-project
```

### Multi-model comparison (model X vs human)

Runs tokenize → extract → analyze for each AI model against human.

```bash
uv run scripts/run_raid_pipeline.py

uv run scripts/run_raid_pipeline.py --models gpt4 chatgpt mistral-chat
uv run scripts/run_raid_pipeline.py --models gpt4 --samples 5000
```

Output layout:

```
data/processed/raid_gpt4/
results/activations_raid_gpt4/
```

| Flag | Default | Description |
|---|---|---|
| `--models` | all 11 AI models | Models vs human |
| `--samples` | 1,000 | Total samples per model (balanced 50/50), stratified across domains |
| `--domains` | all six | Restrict RAID domains |
| `--batch-size` | 1,000 | Tokenization batch size |
| `--extract-batch-size` | 64 | BERT forward batch size |
| `--seed` | 42 | Random seed |

### Multi-model neuron + clustering analysis

After activations exist under `results/activations_raid_*`, run:

```bash
uv run scripts/analyze_raid_models.py
uv run scripts/analyze_raid_models.py --models gpt4 chatgpt
uv run scripts/analyze_raid_models.py --dim-reduction umap
uv run scripts/analyze_raid_models.py --dim-reduction pca umap
uv run scripts/analyze_raid_models.py --exemplars
```

| Flag | Description |
|---|---|
| `--dim-reduction pca` | PCA embedding (default if the flag is omitted) |
| `--dim-reduction umap` | UMAP embedding (correlation metric) |
| `--dim-reduction pca umap` | Run clustering for both; outputs split under `results/analysis/{model}/pca/` and `.../umap/` |
| `--traits-clustering` | Also build a neuron×trait matrix (stats + activation moments), cluster in that space, save `traits_matrix.csv`, and plot trait clusters on **both** activation PCA and UMAP (the non-primary embedding is computed with default UMAP/PCA so you always get two overlays). Runs once on the first `--dim-reduction` pass when multiple are given. |

With a **single** embedding, outputs go to `results/analysis/{model}/`. With **multiple**, the first listed method also writes neuron-level figures, optional exemplars, and trait-based outputs; later methods write embedding + clustering only under their subfolder.

### Explore in notebooks

1. **`notebooks/neurons_analysis.ipynb`** — neuron discovery and validation  
2. **`notebooks/neuron_clustering.ipynb`** — embedding + clustering exploration  

Point notebook paths at `results/activations_raid_*` (or your chosen folder).

```bash
uv run jupyter notebook
```

---

## Common Commands

### Package Management
```bash
uv add package-name
uv remove package-name
uv sync --upgrade
uv pip install package-name
```

### Running Scripts
```bash
uv run scripts/tokenize_raid.py --model gpt4
uv run scripts/extract_activations.py --tokenized-path data/processed/raid_gpt4 --output results/activations_raid_gpt4
uv run scripts/analyze_activations.py --input results/activations_raid_gpt4
uv run scripts/analyze_raid_models.py --dim-reduction pca
```

---

## Statistical Method

**Per-neuron analysis** across all 9,216 neurons (12 layers × 768):

1. **Mann-Whitney U Test** — tests whether AI vs human activation distributions differ significantly  
2. **AUC effect size** — AUC > 0.7 → AI-preferring; AUC < 0.3 → human-preferring  
3. **Bonferroni correction** — α′ = 0.001 / 9,216 = 1.09×10⁻⁷ per layer  
4. **Cohen's d** — supplementary effect size  

**Clustering (multi-model script)**:

- **2D embedding** — PCA (default), UMAP (`--dim-reduction umap`), or both (`--dim-reduction pca umap`), then per-axis scaling for distance-based methods  
- **Strategies** — Ward linkage with silhouette or merge-gap cut; KMeans with silhouette sweep (see `raid_analysis/clustering.py`)  

---

## Tech Stack

**Package Management**: [uv](https://github.com/astral-sh/uv)  
**Core**: PyTorch, HuggingFace Transformers  
**Analysis**: NumPy, Pandas, SciPy, scikit-learn  
**Clustering / viz**: UMAP (optional embedding), matplotlib, seaborn  
**Experiment tracking**: Weights & Biases (optional)

### Why uv?

This project uses `uv` for dependency management: fast installs, lockfile support, and reliable resolution.

---

## References

- [RAID benchmark](https://arxiv.org/abs/2405.07940) (ACL 2024) — [HuggingFace](https://huggingface.co/datasets/liamdugan/raid) · [GitHub](https://github.com/liamdugan/raid)  
- [BERT](https://arxiv.org/abs/1810.04805)  
- [UMAP](https://umap-learn.readthedocs.io/)  
- [Mann-Whitney U test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html)
