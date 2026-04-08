# Experiment System Architecture

This document defines the architecture for the CV-validated experiment pipeline.
It supersedes the old AUC-threshold neuron selection approach with a proper
train/test-separated, config-driven system that supports multiple neuron selection
methods (sparse probe, AUC, IG) and multiple evaluation methods (ablation, patching,
confound checks) through a universal protocol.

---

## Table of Contents

1. [Design Principles](#design-principles)
2. [Full Pipeline Diagram](#full-pipeline-diagram)
3. [Key Protocols](#key-protocols)
4. [Cross-Experiment Dependencies](#cross-experiment-dependencies)
5. [Module Architecture](#module-architecture)
6. [Experiment Catalog](#experiment-catalog)
7. [Scientific Validity Guarantees](#scientific-validity-guarantees)

---

## Design Principles

1. **Separation of selection and evaluation.** Neuron selection and evaluation
  always use different data partitions. The selector never sees test data. The
   evaluator never influences selection. Two independent probes: L1 for selection,
   L2 for evaluation.
2. **Method-agnostic fold loop.** The experiment runner enforces the fold boundary.
  Selection methods and evaluation methods plug in via protocols. Neither knows
   about folds — they receive plain arrays and return results.
3. **Extract once, split by index.** For a frozen model, activations are
  deterministic. They are extracted once and stored on disk. CV splits are index
   arrays applied at experiment time. No re-extraction needed.
4. **All-layers concatenated.** Probes train on all 9,216 features (12 layers ×
  768 neurons) simultaneously. The L1 penalty naturally zeros out unimportant
   neurons/layers, so layer distribution is a *result* of selection, not a choice.
5. **Config-driven, reproducible.** Every experiment is parameterized via YAML.
  Every run saves its config, splits, per-fold results, and aggregate metrics
   to a timestamped directory. Same config + same data = same results.

---

## Full Pipeline Diagram

```
══════════════════════════════════════════════════════════════════
 PHASE 0 — DATA PREPARATION  (done once, outputs are immutable)
══════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────┐
  │  HuggingFace RAID dataset                               │
  │  (liamdugan/raid, 11 generators × 6 domains)            │
  └────────────────────────┬────────────────────────────────┘
                           │
                    download_raid.py
                           │
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │  data/raw/raid/                                         │
  │  Per (domain, model) CSVs:                              │
  │    news_gpt4.csv, news_human.csv,                       │
  │    abstracts_chatgpt.csv, ...                           │
  └────────────────────────┬────────────────────────────────┘
                           │
                    load_raid(RAIDConfig)
                    balanced sampling, domain-stratified
                           │
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │  HF Dataset (in memory)                                 │
  │  Columns: text, label, domain, source_model, title      │
  │  N samples, balanced 50/50 AI/human                     │
  └────────────────────────┬────────────────────────────────┘
                           │
                    DatasetTokenizer + BERT tokenizer
                           │
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │  data/processed/raid_{generator}/                       │
  │  Tokenized DatasetDict on disk                          │
  │  Columns: input_ids, attention_mask, label, domain, ... │
  └────────────────────────┬────────────────────────────────┘
                           │
                    ActivationExtractor (frozen BERT)
                    CLS token, all 12 layers
                    Saves per-sample metadata            [NEW]
                           │
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │  results/activations_raid_{generator}/                  │
  │                                                         │
  │  layer_{1..12}_activations.npy    (N, 768) per layer    │
  │  labels.npy                       (N,)                  │
  │  sample_metadata.npz              (N,)              [NEW]│
  │     .text_lengths                                       │
  │     .domain_ids                                         │
  │     .domain_names                                       │
  │  metadata.json                    (counts, shapes)      │
  │                                                         │
  │  ┌───────────────────────────────────────────────┐      │
  │  │  layer_*_neuron_stats.csv  (old AUC pipeline) │      │
  │  │  KEPT for exploratory use only.               │      │
  │  │  NOT used by any validated experiment.         │      │
  │  └───────────────────────────────────────────────┘      │
  └─────────────────────────────────────────────────────────┘

  Everything above this line is deterministic for a given
  (BERT revision, RAID version, sample count, seed).
  Computed once. Never changes.


══════════════════════════════════════════════════════════════════
 PHASE 1 — EXPERIMENT SETUP  (per experiment run)
══════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────┐
  │  Experiment config (YAML)                               │
  │                                                         │
  │  experiment: sparse_probe        # or ablation, etc.    │
  │  generators: [gpt4, chatgpt]     # or "all" or "pooled" │
  │  n_folds: 5                                             │
  │  seeds: [42, 123, 456]                                  │
  │  selector: sparse_probe          # or auc, ig, ...      │
  │  selector_params:                                       │
  │    C_values: [0.001, 0.01, 0.1, 1.0]                   │
  │    penalty: l1                                          │
  │  evaluator: ablation             # or patching, etc.    │
  │  evaluator_params:                                      │
  │    method: mean                                         │
  │    k_values: [5, 10, 20, 50, 100]                      │
  │  max_samples: 10000                                     │
  └────────────────────────┬────────────────────────────────┘
                           │
                    Load activations from disk
                    (per-layer .npy files, all requested layers)
                           │
                    Concatenate layers → (N, 9216) array
                    Neuron index mapping:
                      global_idx = (layer - 1) * 768 + neuron_idx
                           │
                    Generate CV splits:
                    stratified by (label × domain)
                    using each seed
                           │
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │  In memory:                                             │
  │    activations: (N, 9216) concatenated all layers       │
  │    labels: (N,)                                         │
  │    metadata: SampleMetadata                             │
  │    splits: list[CVSplit] per seed                       │
  │      each CVSplit: (fold_idx, seed, train_idx, test_idx)│
  └────────────────────────┬────────────────────────────────┘
                           │
                           │
══════════════════════════════════════════════════════════════════
 PHASE 2 — EXPERIMENT LOOP  (the core, method-agnostic)
══════════════════════════════════════════════════════════════════
                           │
          For each seed × fold:
                           │
          ┌────────────────┴────────────────┐
          │                                 │
          ▼                                 ▼
  ┌───────────────────┐           ┌───────────────────┐
  │    TRAIN FOLD     │           │    TEST FOLD      │
  │                   │           │                   │
  │  acts[train_idx]  │           │  acts[test_idx]   │
  │  labels[train_idx]│           │  labels[test_idx] │
  │  meta[train_idx]  │           │  meta[test_idx]   │
  └────────┬──────────┘           └────────┬──────────┘
           │                               │
           │                               │
           │  STEP 1: SELECTION            │
           │  (only train data)            │
           │                               │
           ▼                               │
  ┌──────────────────────────┐             │
  │  selector.select(        │             │
  │    train_acts,           │             │
  │    train_labels          │             │
  │  )                       │             │
  │                          │             │
  │  Internally, depending   │             │
  │  on selector type:       │             │
  │                          │             │
  │  SparseProbe:            │             │
  │    train L1 LogReg       │             │
  │    nonzero weights → set │             │
  │    (L1 probe saved)      │             │
  │                          │             │
  │  AUC:                    │             │
  │    MWU + AUC per neuron  │             │
  │    threshold → set       │             │
  │                          │             │
  │  IG:                     │             │
  │    train probe           │             │
  │    IG through probe      │             │
  │    top-k → set           │             │
  └──────────┬───────────────┘             │
             │                             │
             ▼                             │
  ┌──────────────────────┐                 │
  │  SelectionResult     │                 │
  │    .neuron_indices   │ ─ ─ ─ ─ ─ ─ ─ ─ ┤
  │    .ranking          │                 │
  │    .probe (optional) │                 │
  │    .train_mean       │ ─ ─ ─ ─ ─ ─ ─ ─ ┤
  │    .metadata         │                 │
  └──────────────────────┘                 │
                                           │
           │                               │
           │  STEP 2: EVALUATION PROBE     │
           │  (only train data)            │
           │                               │
           ▼                               │
  ┌──────────────────────────┐             │
  │  Train L2 LogReg on      │             │
  │  ALL 9216 features       │             │
  │  using train_acts,       │             │
  │  train_labels            │             │
  │                          │             │
  │  This probe is a         │             │
  │  measurement instrument. │             │
  │  It is independent of    │             │
  │  the selector — it does  │             │
  │  not know which neurons  │             │
  │  were selected.          │             │
  │                          │             │
  │  → eval_probe            │ ─ ─ ─ ─ ─ ─ ┤
  └──────────────────────────┘             │
                                           │
                                           │
                           STEP 3: EVALUATION
                           (only test data │
                           + artifacts     │
                           from steps 1-2) │
                                           │
                                           ▼
                              ┌─────────────────────────────┐
                              │  evaluator.evaluate(        │
                              │    test_acts,               │
                              │    test_labels,             │
                              │    test_metadata,           │
                              │    selection_result,        │
                              │    eval_probe               │
                              │  )                          │
                              │                             │
                              │  Ablation evaluator:        │
                              │    baseline = eval_probe    │
                              │      .score(test)           │
                              │    for k in k_values:       │
                              │      ablate top-k neurons   │
                              │      (using train_mean)     │
                              │      post = eval_probe      │
                              │        .score(ablated)      │
                              │      random-k baseline ×N   │
                              │      record delta           │
                              │                             │
                              │  Patching evaluator:        │
                              │    pair AI↔human same-domain│
                              │    patch selected neurons   │
                              │    count label flips via    │
                              │      eval_probe             │
                              │                             │
                              │  Confound evaluator:        │
                              │    correlate selected neuron│
                              │    acts with text_length,   │
                              │    domain on test fold      │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────────┐
                              │  FoldResult                  │
                              │    .fold_idx, .seed          │
                              │    .neuron_set               │
                              │    .metrics (eval-specific)  │
                              │    .config snapshot          │
                              └──────────────┬───────────────┘
                                             │
          ◄──────────────────────────────────┘
          │
          │  Repeat for all folds × seeds
          │
          ▼

══════════════════════════════════════════════════════════════════
 PHASE 3 — AGGREGATION AND OUTPUT
══════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────┐
  │  Collect all FoldResults                                │
  │                                                         │
  │  Per-fold:                                              │
  │    neuron sets, accuracies, deltas, flip rates, ...     │
  │                                                         │
  │  Aggregate:                                             │
  │    mean ± std accuracy across folds                     │
  │    neuron stability (fraction of folds a neuron         │
  │      appears in)                                        │
  │    stable neuron set = neurons in ≥ threshold folds     │
  │    mean dose-response curve ± CI                        │
  │    mean flip rate ± CI per generator, domain            │
  │    confound correlation p-values                        │
  └────────────────────────┬────────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │  results/experiments/{experiment}/{run_id}/             │
  │                                                         │
  │  config.yaml              (exact config used)           │
  │  splits.json              (all fold indices, reusable)  │
  │  fold_{k}/                                              │
  │    selection_result.json  (neuron set, rankings)        │
  │    l1_probe_weights.npy   (selector probe, if any)      │
  │    l2_probe_weights.npy   (evaluation probe)            │
  │    eval_metrics.json      (fold-specific metrics)       │
  │  aggregate.json           (cross-fold summary)          │
  │  figures/                                               │
  │    accuracy_vs_sparsity.png                             │
  │    dose_response.png                                    │
  │    flip_rate_by_generator.png                           │
  │    confound_correlations.png                            │
  │    ...                                                  │
  └─────────────────────────────────────────────────────────┘
```

---

## Key Protocols

### NeuronSelector

Every neuron selection method implements this protocol. It receives train-fold
data and returns a result. It never sees test data. It never knows about folds.

```python
class NeuronSelector(Protocol):
    def select(
        self,
        activations: np.ndarray,         # (N_train, 9216) concatenated
        labels: np.ndarray,              # (N_train,)
    ) -> SelectionResult: ...

@dataclass
class SelectionResult:
    neuron_indices: set[tuple[int, int]]  # (layer, neuron_idx) canonical format
    ranking: list[tuple[int, int]]        # ordered by importance
    probe: Any | None                     # trained model if applicable
    train_mean: np.ndarray                # (9216,) for mean ablation
    metadata: dict                        # method-specific info
```

Canonical neuron index format is always `(layer, neuron_idx)` tuples.
Conversion to/from flat indices: `global_idx = (layer - 1) * 768 + neuron_idx`.

### Selector Implementations

**SparseProbeSelector:**

- Trains L1 LogisticRegression (solver `saga` or `liblinear`)
- `neuron_indices` = positions where `|coef_| > 0`
- `ranking` = sorted by `|coef_|` descending
- `probe` = the trained L1 model (saved for diagnostics, NOT for evaluation)
- `train_mean` = `activations.mean(axis=0)` on training data

**AUCSelector:**

- Runs `compute_neuron_statistics` on provided (train) data
- Applies `identify_discriminative_neurons` thresholds
- `neuron_indices` = discriminative set
- `ranking` = sorted by AUC deviation from 0.5
- `probe` = None (AUC selection doesn't produce a model)

**IGSelector:**

- Trains a probe (L2 LogReg or MLP) on provided data
- Computes integrated gradients through the probe
- `neuron_indices` = top-k by mean |IG attribution|
- `ranking` = sorted by |attribution|
- `probe` = the trained model used for IG

### Evaluator

Each evaluation method implements this protocol. It receives test-fold data
plus artifacts from steps 1 and 2 of the fold loop.

```python
class Evaluator(Protocol):
    def evaluate(
        self,
        test_activations: np.ndarray,    # (N_test, 9216)
        test_labels: np.ndarray,         # (N_test,)
        test_metadata: SampleMetadata,   # text lengths, domains
        selection: SelectionResult,      # from selector (step 1)
        eval_probe: Any,                 # L2 probe (step 2)
    ) -> dict: ...                       # metric name → value
```

### Evaluator Implementations

**AblationEvaluator:**

- Scores eval_probe on unmodified test data → baseline accuracy
- For each k in k_values:
  - Ablate top-k neurons (by selection.ranking) using selection.train_mean
  - Score eval_probe on ablated test data → post-ablation accuracy
  - Repeat with random-k neurons (multiple random seeds) → random baseline
  - Record: k, selected_drop, random_drop_mean, random_drop_std
- Output: dose-response curve data

**PatchingEvaluator:**

- Pair AI and human samples from same domain within test fold
- Patch selected neurons from AI donor into human base
- Score eval_probe on patched data
- Count label flips (human→AI predictions)
- Stratify flip rate by generator and domain
- Output: flip rates per stratum

**ConfoundEvaluator:**

- For each selected neuron:
  - Correlate activation values (test fold) with text_length (Spearman)
  - Correlate with domain_id (ANOVA F-stat or point-biserial)
- Output: per-neuron correlation coefficients + p-values, summary flag

**ProbeAccuracyEvaluator:**

- Scores eval_probe on test data
- Reports accuracy, AUC-ROC, F1
- Used standalone or as part of the sparse probe sweep

### Why Two Probes?

**L1 probe (selection):** Trained with L1 penalty. Sparse weights. The nonzero
coefficients define the neuron set. Selection happens inside the weight vector —
no separate scoring step needed.

**L2 probe (evaluation):** Trained with L2 penalty on ALL 9,216 features. Dense
weights. This is a measurement instrument that asks "how much classification-
relevant information is in this representation?"

Using the L1 probe for evaluation would be circular: ablating its own nonzero
features must drop its accuracy — that's trivially true and proves nothing
about the neurons' importance to the task in general. The L2 probe is
independent. If ablating the L1-selected neurons causes the L2 probe to lose
accuracy, it proves the information is genuinely concentrated in those neurons,
not just that one model's decision boundary depends on its own features.

### Why the Same Train Fold for Both Probes?

Both probes are trained on the same train fold. This is NOT circular because:

- The L1 probe's job is feature selection (which neurons matter).
- The L2 probe's job is measuring information content (how much does the full
representation support classification).
- They are independent models: different regularization, different weights,
different purposes.
- The L2 probe trains on ALL features with no knowledge of the L1 selection.
It would produce the same weights regardless of whether the L1 probe existed.
- Evaluation (scoring) happens on the held-out test fold that neither probe
saw during training.

---

## Cross-Experiment Dependencies

```
Exp 1: Sparse probe sweep
│   Runs SparseProbeSelector with multiple C values × seeds × folds
│   Outputs: accuracy-vs-sparsity curve, knee point,
│            stable neuron set, per-fold selections, splits.json
│
├── Exp 2: Ablation validation
│   Loads Exp 1's splits.json + per-fold selection results
│   Runs AblationEvaluator (group ablation at multiple k values)
│   Outputs: dose-response curve, random-k baseline comparison
│
├── Exp 3: Activation patching
│   Loads Exp 1's splits.json + per-fold selection results
│   Runs PatchingEvaluator (same-domain AI↔human pairs)
│   Outputs: flip rate per generator/domain
│
├── Exp 4: Confound checks
│   Loads Exp 1's splits.json + per-fold selection results
│   Runs ConfoundEvaluator (correlation with text length, domain)
│   Outputs: correlation coefficients + p-values on test folds
│
├── Exp 5: Characterize validated set
│   Loads Exp 1's aggregate stable neuron set
│   No fold loop — operates on the final stable set
│   Outputs: layer distribution, cross-generator Jaccard, CAV alignment
│
├── Exp 6: AUC comparison
│   Loads Exp 1's splits.json (same folds for fair comparison)
│   Runs AUCSelector on the same train folds
│   Compares AUC-selected vs L1-selected neuron sets
│   Uses same L2 eval probe for ablation comparison
│   Outputs: overlap metrics, comparative ablation drop
│
└── Exp 7: MLP probe + IG (conditional)
    Only if Exp 1 shows no clean knee in accuracy-vs-sparsity curve
    Loads Exp 1's splits.json
    Trains MLP, runs IG, compares attribution ranking with L1
    Outputs: IG neuron ranking, overlap with L1 set, MLP vs linear accuracy
```

Experiments 2–4 reuse the exact same splits AND neuron sets from Experiment 1.
They do not re-select neurons. They evaluate different properties of the same
neurons on the same held-out folds. All results are directly comparable.

Experiment 6 reuses the same splits but runs a different selector (AUC). Same
train folds for selection, same test folds for evaluation, same L2 evaluation
probe. The comparison is on identical data partitions.

---

## Module Architecture

```
raid_analysis/
├── constants.py                         # EXTEND with DEFAULT_SEEDS, etc.
├── io.py                                # KEEP
├── neurons_pipeline.py                  # KEEP (old AUC pipeline, now baseline)
├── run_analysis.py                      # KEEP (old orchestrator, now baseline)
│
├── data/                                # EXTEND
│   ├── __init__.py
│   ├── activations.py                   # KEEP + add concat_all_layers()
│   ├── loader.py                        # KEEP
│   ├── neuron_stats.py                  # KEEP (used by AUCSelector)
│   ├── discriminative.py                # KEEP (used by AUC comparison)
│   ├── traits.py                        # KEEP
│   ├── splits.py                        # NEW: CVSplit, generate, save/load
│   └── metadata.py                      # NEW: SampleMetadata, compute/save/load
│
├── selection/                           # NEW: method-agnostic neuron selection
│   ├── __init__.py
│   ├── protocol.py                      # NeuronSelector protocol, SelectionResult
│   ├── sparse_probe.py                  # SparseProbeSelector (L1 LogReg)
│   ├── auc.py                           # AUCSelector (wraps neuron_stats.py)
│   ├── ig.py                            # IGSelector (IG through trained probe)
│   └── composite.py                     # Union/intersect/rank selectors
│
├── evaluation/                          # NEW: method-agnostic evaluation
│   ├── __init__.py
│   ├── protocol.py                      # Evaluator protocol, FoldResult
│   ├── probe_factory.py                 # Train L2 evaluation probe
│   ├── ablation.py                      # AblationEvaluator
│   ├── patching.py                      # PatchingEvaluator
│   ├── confound.py                      # ConfoundEvaluator
│   └── probe_accuracy.py               # ProbeAccuracyEvaluator
│
├── experiments/                         # REFACTOR: experiment orchestrators
│   ├── __init__.py
│   ├── runner.py                        # Fold loop: select → eval probe → evaluate
│   ├── config.py                        # Config loading, defaults, validation
│   ├── exp_sparse_probe.py             # Exp 1: L1 sweep + stability
│   ├── exp_ablation.py                 # Exp 2: group ablation validation
│   ├── exp_patching.py                 # Exp 3: activation patching sufficiency
│   ├── exp_confound.py                 # Exp 4: confound checks
│   ├── exp_characterize.py            # Exp 5: validated set characterization
│   ├── exp_auc_comparison.py          # Exp 6: AUC vs sparse probe comparison
│   ├── exp_mlp_probe.py               # Exp 7: optional MLP + IG
│   │
│   │ # Kept as low-level primitives (imported by selection/ and evaluation/):
│   ├── causal.py                        # ablate_neurons, patch_neurons
│   ├── cross_generator.py              # jaccard_similarity, jaccard_matrix
│   ├── linear.py                        # mean_difference_vector, lr_weight_vector
│   └── probing.py                       # TrainedProbe, train_probe (old simple API)
│
├── viz/                                 # EXTEND
│   ├── __init__.py
│   ├── dim_reduction.py                 # KEEP
│   ├── embedding.py                     # KEEP
│   ├── hierarchy.py                     # KEEP
│   ├── neurons.py                       # KEEP
│   ├── sparsity_curves.py              # NEW: accuracy vs nonzero count + knee
│   ├── dose_response.py                # NEW: accuracy drop vs k ablated
│   └── flip_rate.py                     # NEW: flip rate per generator/domain
│
├── clustering/                          # KEEP (unchanged)
├── reports/                             # KEEP (unchanged)
└── __init__.py                          # EXTEND exports


config/                                  # NEW top-level directory
├── experiment_defaults.yaml             # Global: seeds, folds, sample size
└── experiments/
    ├── sparse_probe.yaml
    ├── ablation.yaml
    ├── patching.yaml
    ├── confound.yaml
    ├── characterize.yaml
    ├── auc_comparison.yaml
    └── mlp_probe.yaml


scripts/
├── ... (existing scripts unchanged)
└── experiments/                         # NEW: experiment entry points
    ├── run_experiment.py                # Generic: --experiment sparse_probe
    └── run_all.py                       # Run experiment sequence
```

---

## Experiment Catalog

### Experiment 1 — Sparse Probe Sweep

**Question:** Which neurons does L1 selection identify, and is the set stable?

```yaml
experiment: sparse_probe
generators: [gpt4, chatgpt, mistral-chat]
pooled: true
n_folds: 5
seeds: [42, 123, 456, 789, 1024]
selector: sparse_probe
selector_params:
  C_values: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
  penalty: l1
  solver: saga
  max_iter: 5000
evaluator: probe_accuracy
max_samples: 10000
stability_threshold: 0.8
```

**Flow per C value:**

1. For each seed × fold:
  - Train L1 probe on train fold → nonzero set + L1 accuracy
  - Train L2 eval probe on train fold → eval accuracy on test fold
  - Record nonzero neuron count, L1 accuracy, L2 eval accuracy
2. Aggregate: accuracy vs n_nonzero curve (with error bars from CV)
3. Find knee (elbow point) in the curve
4. At the knee C value: compute neuron stability (fraction of seed × fold
  runs each neuron appears in)
5. Stable neuron set = neurons in ≥ stability_threshold of runs

**Key outputs:**

- Accuracy vs sparsity curve (with error bars)
- Knee point C value and neuron count
- Stable neuron set (used by all downstream experiments)
- Per-seed neuron overlap heatmap
- Saved splits.json, per-fold selection results, probe weights

### Experiment 2 — Ablation Validation

**Question:** Does ablating the selected neurons cause proportional accuracy drop
compared to random neurons?

**Depends on:** Experiment 1 (loads splits + per-fold selection results)

```yaml
experiment: ablation
source_experiment: sparse_probe/{run_id}
evaluator: ablation
evaluator_params:
  method: mean
  k_values: [1, 2, 5, 10, 20, 50, 100, 200]
  n_random_seeds: 20
```

**Flow per fold:**

1. Load SelectionResult from Experiment 1 (neuron ranking, train_mean)
2. Train L2 evaluation probe on train fold (all 9,216 features)
3. On test fold:
  - Score L2 probe → baseline accuracy
  - For each k: ablate top-k by L1 ranking (mean ablation using train_mean),
  score L2 probe → post-ablation accuracy
  - For each k: ablate random-k neurons (20 random draws),
  score L2 probe → random baseline mean ± std
4. Record dose-response data per fold

**Key outputs:**

- Dose-response curve: accuracy drop vs k (selected vs random, with CI)
- Gap between selected and random curves = evidence of causal concentration

### Experiment 3 — Activation Patching

**Question:** Are the selected neurons sufficient to flip human→AI prediction?

**Depends on:** Experiment 1 (loads splits + per-fold selection results)

```yaml
experiment: patching
source_experiment: sparse_probe/{run_id}
evaluator: patching
evaluator_params:
  pairing: same_domain
```

**Flow per fold:**

1. Load SelectionResult from Experiment 1
2. Train L2 evaluation probe on train fold
3. On test fold:
  - Pair each human sample with a same-domain AI sample
  - Patch selected neurons from AI donor into human base
  - Score patched samples with L2 eval probe
  - Record which predictions flipped (human → AI)
4. Stratify flip rate by generator and domain

**Key outputs:**

- Overall flip rate
- Flip rate per generator, per domain
- High flip rate = selected neurons are sufficient for detection

### Experiment 4 — Confound Checks

**Question:** Are the selected neurons encoding text length or domain artifacts?

**Depends on:** Experiment 1 (loads splits + per-fold selection results)

```yaml
experiment: confound
source_experiment: sparse_probe/{run_id}
evaluator: confound
evaluator_params:
  confounds: [text_length, domain]
  correlation_method: spearman
  significance_threshold: 0.05
```

**Flow per fold:**

1. Load SelectionResult from Experiment 1
2. On test fold:
  - For each selected neuron: Spearman correlation of activation values
   with text_length
  - For each selected neuron: ANOVA F-stat or point-biserial correlation
  with domain label
3. Aggregate: which neurons are significantly confounded across ≥ majority folds

**Key outputs:**

- Per-neuron confound correlation table
- Flagged confounded neurons (significant correlation in majority of folds)
- Clean neuron set = selected − confounded

### Experiment 5 — Characterize Validated Set

**Question:** What is the structure of the validated neuron set across generators?

**Depends on:** Experiment 1 (loads aggregate stable neuron set; run per generator)

```yaml
experiment: characterize
source_experiment: sparse_probe/{run_id}
generators: [gpt4, chatgpt, mistral-chat, llama-chat, cohere-chat]
```

**Flow (no fold loop — uses the final stable sets):**

1. Load per-generator stable neuron sets from Experiment 1
2. Layer distribution histogram per generator
3. Cross-generator Jaccard overlap matrix
4. Core neurons: appearing in ≥ k generators
5. Linear direction alignment:
  - Mean difference vector (μ_AI − μ_human) per generator
  - Cosine similarity between L1 weight vector and mean-diff vector (CAV)

**Key outputs:**

- Layer distribution bar chart
- Generator × generator Jaccard heatmap
- Core neuron set
- CAV alignment scores

### Experiment 6 — AUC Comparison

**Question:** How do AUC-selected neurons compare to sparse probe neurons?

**Depends on:** Experiment 1 (loads splits for identical partitions)

```yaml
experiment: auc_comparison
source_experiment: sparse_probe/{run_id}
selector: auc
selector_params:
  alpha: 0.001
  auc_threshold: 0.7
```

**Flow per fold (using Experiment 1's splits):**

1. Run AUCSelector on the same train fold → AUC neuron set
2. Load L1 neuron set from Experiment 1
3. Overlap metrics: Jaccard, top-k intersection, Spearman rank correlation
4. Train L2 evaluation probe on train fold (same probe for both sets)
5. On test fold:
  - Ablate AUC-selected neurons → accuracy drop A
  - Ablate L1-selected neurons → accuracy drop B
6. Compare drops (same eval probe, same test data, different neuron sets)

**Key outputs:**

- Overlap metrics between AUC and L1 selection
- Comparative ablation drop
- Methodological finding: does learned sparsity outperform statistical thresholding?

### Experiment 7 — MLP Probe + IG (Conditional)

**Question:** Does a nonlinear probe capture signal missed by the linear probe?

**Condition:** Only run if Experiment 1 shows no clean knee in the
accuracy-vs-sparsity curve.

```yaml
experiment: mlp_probe
source_experiment: sparse_probe/{run_id}
selector: ig
selector_params:
  probe_type: mlp
  hidden_sizes: [256, 64]
  top_k: 100
  baseline: zero
  n_steps: 50
evaluator: probe_accuracy
```

**Flow per fold:**

1. Train MLP probe on train fold
2. Run integrated gradients through MLP → per-neuron attribution
3. Select top-k neurons by |attribution|
4. Train L2 evaluation probe on train fold
5. Score on test fold, ablate IG-selected neurons, compare

**Key outputs:**

- IG neuron ranking and overlap with L1 set
- MLP vs linear probe accuracy comparison
- If overlap is high and accuracy gain is small → linear probe sufficient

---

## Scientific Validity Guarantees

### 1. No information leakage

The fold loop guarantees that:

- Neuron selection (Step 1) only sees train fold activations
- The L2 evaluation probe (Step 2) is trained on train fold only
- All evaluation scoring (Step 3) happens on held-out test fold only
- Ablation replacement values (train_mean) come from train data only
- Confound correlations are computed on test data using neurons selected on train

### 2. Independent selection and evaluation instruments

The L1 probe (selector) and L2 probe (evaluator) are independent:

- Different regularization (L1 sparse vs L2 dense)
- Different roles (feature selection vs information measurement)
- L2 trains on all features with no knowledge of which neurons L1 selected
- Ablation drop measured on L2 proves neurons matter for the task, not just
for the L1 probe's specific decision boundary

### 3. Random baselines

Every ablation experiment includes random-k baseline:

- Same k neurons ablated, randomly chosen instead of L1-selected
- Repeated with multiple random seeds for confidence intervals
- Gap between selected-k and random-k is the real causal signal
- Controls for "any k neurons" vs "these specific k neurons"

### 4. Multi-seed stability

Experiment 1 runs across multiple seeds:

- Different random initializations of the L1 solver
- Different CV fold assignments
- Stable set = neurons in ≥ 80% of seed × fold runs
- Filters out noise-driven selections

### 5. Confound control

Experiment 4 checks whether selected neurons correlate with confounds
(text length, domain) on held-out test data. Neurons whose activation is
better explained by confounds than by AI-ness are flagged for exclusion.

### 6. Reproducibility

Every run saves:

- Exact config YAML
- CV split indices (reusable by downstream experiments)
- Per-fold selection results + probe weights
- Aggregate metrics
- All figures

Same config + same data + same code version = identical results.