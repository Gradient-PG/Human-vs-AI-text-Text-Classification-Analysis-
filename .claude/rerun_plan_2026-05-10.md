# Full Pipeline Rerun Plan
**Timestamp:** 2026-05-10  
**Goal:** Single clean run covering all 6 paper generators, replacing the two mixed runs currently referenced in the paper.

---

## Problem with Current State

Two separate run IDs in the paper:
- **`20260426_20_full_c0005_n7500_seeds3`** — 6 generators for `sparse_probe`; only 5 for everything else (no MPT in patching/ablation/restricted_probe/auc_comparison)
- **`20260509_165026`** — MPT-only supplement for the missing experiments

Goal: collapse into **one run ID** covering all 6 generators uniformly.

---

## Experiment 1 — Main Per-Generator Run (all 6 generators)

```bat
uv run scripts/experiments/run_all.py ^
  --generators gpt4 gpt2 mpt mistral-chat llama-chat cohere-chat ^
  --run-id 20260510_rerun_main ^
  --skip confound
```

Runs in dependency order per generator:
1. `sparse_probe`      — 6 gen × 15 cells = 90 cells
2. `ablation`          — 6 gen × 15 cells = 90 cells  (depends on sparse_probe)
3. `patching`          — 6 gen × 15 cells = 90 cells  (depends on sparse_probe)
4. `auc_comparison`    — 6 gen × flat output = 6 files (depends on sparse_probe)
5. `restricted_probe`  — 6 gen × 15 cells = 90 cells  (depends on sparse_probe)

Then cross-generator:
6. `characterize`      — 1 JSON over all 6 generators (depends on sparse_probe)

`confound` is skipped — it is not referenced anywhere in the paper.

**Estimated wall-clock:** ~3–5 hours (CPU-bound; BERT activations already extracted)

---

## Experiment 2 — LOGO (family mode, all 11 generators)

```bat
uv run scripts/experiments/run_logo.py ^
  --mode family ^
  --run-id 20260510_rerun_logo ^
  --source-dir results/experiments/20260510_rerun_main/sparse_probe
```

- 5 family folds (cohere, gpt, llama, mistral, mpt) × 3 seeds = 15 evaluations
- `--source-dir` enables Jaccard comparison vs the new main run's stable sets
- Config already set for all 11 generators + family mode in `config/experiments/logo.yaml`

Run AFTER Experiment 1 finishes (needs sparse_probe output for `--source-dir`).  
Can also be run without `--source-dir` if Jaccard comparison is not needed.

**Estimated wall-clock:** ~30–60 min

---

## Experiment 3 — Stability Grid

**SKIP — reuse existing run `stability_grid_gpt4_llama_mistral`.**

The stability grid is a hyperparameter calibration sweep (C × N) that produced `grid_summary.json` — the result (C=0.005 is best) is documented in App B and does not depend on any downstream experiment data. No rerun required.

If a fresh run is ever needed:
```bat
uv run scripts/experiments/run_sample_size_stability.py ^
  --generators gpt4 llama-chat mistral-chat ^
  --c-list 0.005 0.01 0.02 0.05 ^
  --n-list 500 1000 2000 3500 5000 7500 ^
  --k-draws 15
```

---

## Summary

| # | Script | Covers | Est. time |
|---|--------|--------|-----------|
| 1 | `run_all.py` (6 generators, skip confound) | sparse_probe, ablation, patching, auc_comparison, restricted_probe, characterize | 3–5 h |
| 2 | `run_logo.py --mode family` | LOGO family × 5 folds × 3 seeds | 30–60 min |
| 3 | stability grid | **Skip — reuse existing** | 0 min |

**Total new compute:** ~4–6 hours  
Both experiments can run sequentially (recommended) or in parallel if RAM allows (~1.7 GB for LOGO's full activation load).

---

## After the Rerun: Paper Updates Required

1. **Run ID string in App A** — update the run ID cited in the pipeline description to `20260510_rerun_main`

2. **Regenerate figures** — update `EXP` and `LOGO_E` path constants at the top of `_paper/gen_figures.py`:
   ```python
   EXP   = Path("results/experiments/20260510_rerun_main")
   LOGO_E = Path("results/experiments/20260510_rerun_logo")
   ```
   Then: `uv run _paper/gen_figures.py`

3. **Revalidate all reported numbers** — diff new JSONs against paper tables:
   - Accuracy, AUC, F1, n_sel → Table 1 (App C)
   - Flip rates per k → Table 2 + Figure 2
   - Ablation drop → Table 3
   - Layer distributions → Table 4 + Figure 3
   - Jaccard matrix → Table 5 + Figure 4
   - Core neurons count → §6.3
   - LOGO full/sparse accuracy → Table 6 + Figure 5

   If numbers change by ≤ 0.3pp: no paper edits needed.  
   If numbers change materially: update tables and re-check Discussion claims.
