# Experiment Build Plan

## Phase 1 — Foundation (no experiment code yet, just building blocks)
Everything downstream depends on these. Build them first, test in isolation.

### 1a. data/splits.py
- `CVSplit` dataclass, `generate_cv_splits` (stratified by label × domain), `save_splits`, `load_splits`
- **Test:** generate splits on existing activations, verify fold sizes are balanced, verify domain distribution within each fold
- No dependencies on new code — just numpy, sklearn's `StratifiedGroupKFold`, and JSON serialization

### 1b. data/metadata.py
- `SampleMetadata` dataclass, `compute_metadata_from_dataset`, `save_metadata`, `load_metadata`
- **Test:** load a tokenized dataset, compute text lengths and domain IDs, save as `.npz`, reload and verify alignment with `labels.npy`
- Requires one small extension to `ActivationExtractor.extract_and_save` to call this at extraction time (or a standalone backfill script for existing activations)

### 1c. data/activations.py extension
- Add `concat_all_layers(activations_dict) -> np.ndarray` that produces the `(N, 9216)` array from the per-layer dict
- Add `global_to_layer_neuron(global_idx) -> (layer, neuron_idx)` and the reverse
- **Test:** load existing `.npy` files, concat, verify shape and index mapping round-trips

> These three are independent of each other and can be built in parallel.

---

## Phase 2 — Protocols + L2 probe factory

### 2a. selection/protocol.py
- `SelectionResult` dataclass
- `NeuronSelector` protocol (just the type definition — `select(activations, labels) -> SelectionResult`)
- `save_selection_result`, `load_selection_result` (JSON serialization for neuron indices, ranking, metadata; `.npy` for probe weights and `train_mean`)

### 2b. evaluation/protocol.py
- `Evaluator` protocol (`evaluate(test_acts, test_labels, test_meta, selection, eval_probe) -> dict`)
- `FoldResult` dataclass (`fold_idx`, `seed`, `neuron_set`, `metrics`, `config`)

### 2c. evaluation/probe_factory.py
- `train_eval_probe(activations, labels)` -> trained L2 LogReg + scaler
- This is the Step 2 component from the diagram — straightforward L2 LogReg on all features
- **Test:** train on a subset, score on held-out, verify it works with the existing activation format

> These define the contracts everything else implements. Small files, easy to review.

---

## Phase 3 — First selector + first evaluator (minimum for Experiment 1)

### 3a. selection/sparse_probe.py
- `SparseProbeSelector` implementing `NeuronSelector`
- Internally: L1 LogReg with saga solver, extract nonzero indices, compute ranking by `|coef_|`, store `train_mean`
- **Test:** select on a train fold from Phase 1 splits, verify nonzero count changes with `C`, verify output matches `SelectionResult` format

### 3b. evaluation/probe_accuracy.py
- `ProbeAccuracyEvaluator` — the simplest evaluator. Scores `eval_probe` on test data, returns accuracy/AUC-ROC/F1
- **Test:** train L2 probe from 2c, score on held-out fold, verify metrics are reasonable

### 3c. experiments/runner.py
- The fold loop: for each `seed × fold` → select on train → train eval probe → evaluate on test → collect `FoldResult`
- Aggregate function: mean/std across folds, neuron stability computation
- Result saving: `config.yaml`, `splits.json`, per-fold results, `aggregate.json`
- **Test:** wire up `SparseProbeSelector` + `ProbeAccuracyEvaluator` + runner, run on one generator with 2 folds. Verify output directory structure and that no data leaks across folds.

> **At this point, you can run Experiment 1 end-to-end. This is the critical milestone — everything else depends on it.**

---

## Phase 4 — Experiment 1 orchestrator + sweep logic

### 4a. experiments/config.py
- Config loading from YAML, defaults, validation
- Config dataclass with all fields from the experiment catalog

### 4b. experiments/exp_sparse_probe.py
- Loops over `C_values`, calls runner for each `C`
- Builds accuracy-vs-sparsity data structure
- Calls knee detection
- Computes stable neuron set across `seeds × folds`

### 4c. viz/sparsity_curves.py
- Accuracy vs nonzero neuron count plot with error bars
- Knee point annotation
- **Test:** Run full Experiment 1 on one generator. Verify the sparsity curve looks reasonable, verify the stable neuron set is saved, verify `splits.json` is reusable.

---

## Phase 5 — Remaining evaluators (enables Experiments 2-4)
These are independent of each other and can be built in parallel.

### 5a. evaluation/ablation.py
- `AblationEvaluator`: group ablation sweep, random-k baseline, dose-response data
- Uses existing `experiments/causal.py::ablate_neurons` internally (extended with resample ablation)
- **Test:** load Exp 1 results, run ablation on test folds, verify dose-response curve data

### 5b. evaluation/patching.py
- `PatchingEvaluator`: same-domain pairing, neuron patching, flip rate computation
- Uses existing `experiments/causal.py::patch_neurons` internally
- **Test:** verify pairing produces same-domain pairs, verify flip rate computation

### 5c. evaluation/confound.py
- `ConfoundEvaluator`: Spearman correlation with text length, ANOVA with domain
- **Test:** verify correlation computation on synthetic data with known confounds

### 5d. viz/ablation_sweep.py + viz/flip_rate.py
- Ablation sweep curve plot (selected vs random, with CI bands)
- Flip rate bar chart per generator/domain

---

## Phase 6 — Experiment orchestrators 2-4

### 6a. experiments/exp_ablation.py
- Loads Exp 1 splits + selection results, runs `AblationEvaluator` per fold

### 6b. experiments/exp_patching.py
- Loads Exp 1 splits + selection results, runs `PatchingEvaluator` per fold

### 6c. experiments/exp_confound.py
- Loads Exp 1 splits + selection results, runs `ConfoundEvaluator` per fold

> **Test:** Run Experiments 2-4 using Experiment 1's saved outputs. Verify they use the same splits and don't re-select neurons.

---

## Phase 7 — Remaining experiments (5, 6, 7)

### 7a. selection/auc.py + experiments/exp_auc_comparison.py
- `AUCSelector` wrapping existing `neuron_stats.py`
- Experiment 6 orchestrator: run both selectors on same folds, compare

### 7b. experiments/exp_characterize.py
- Uses existing `cross_generator.py` (Jaccard, core neurons) and `linear.py` (mean-diff, CAV)
- No new primitives needed, just orchestration

### 7c. selection/ig.py + experiments/exp_mlp_probe.py (conditional)
- IG implementation (torch autograd through a probe)
- MLP probe class
- Only built if Experiment 1 results warrant it

---

## Phase 8 — Entry points and polish

### 8a. scripts/experiments/run_experiment.py
- CLI: `--experiment sparse_probe --config config/experiments/sparse_probe.yaml`
- Dispatches to the right experiment orchestrator

### 8b. scripts/experiments/run_all.py
- Runs the experiment sequence respecting dependencies

### 8c. Config YAML files
- One per experiment with sensible defaults

### 8d. Update `__init__.py` exports and README