# Project handoff — paste into next chat

You are continuing work on a Human-vs-AI text classification interpretability project. Read this entire file before doing anything. The user is preparing a paper and we are in the home stretch — finalising experiments, then writing.

## Project at a glance

- **Repo:** `d:\_gradient\Human-vs-AI-text-Text-Classification-Analysis-`
- **Goal:** identify a small, stable set of BERT activations that detect AI-generated text, and characterise that circuit (necessity, sufficiency, cross-generator generalisation).
- **Pipeline:** sparse_probe (L1 selection) → ablation, patching, confound, auc_comparison, restricted_probe (per-generator dependents) → characterize (cross-generator).
- **Key script:** `scripts/experiments/run_all.py --generators ... --run-id ...`
- **Stability sweep script (multi-C × multi-N × multi-generator grid):** `scripts/experiments/run_sample_size_stability.py`
- **Living report:** `experiments_progress_report.html` — section `§4.5` (id `meth-stability`) and `§4.6` (`publishability`) are the most-edited.

## Current state of configs (finalised this session)

All 7 per-experiment YAMLs in `config/experiments/`:

- `max_samples: 7500` (was 5000)
- Per-generator YAMLs (sparse_probe, ablation, patching, confound, auc_comparison, restricted_probe): `seeds: [42, 123, 456]` (was `[42]`)
- `sparse_probe.yaml`: `C_values: [0.005]`, `solver: liblinear`, `max_iter: 1000`. **Do not change C** without rerunning the stability sweep.
- `characterize.yaml`: `generators: [gpt4, gpt2, mistral-chat, llama-chat, cohere-chat]` (chatgpt swapped out for gpt2 to break OpenAI redundancy).
- `ablation.yaml` / `patching.yaml`: `k_values: [1, 2, 5, 10, 20, 50, -1]` — `-1` is the "full selected set" sentinel; `100` and `200` were dropped because at C=0.005, N=7500 the selected set is ~80 features.

## Why these values (must remember for the paper)

**C = 0.005** chosen via a (C × N) stability sweep run THIS session. Grid: C ∈ {0.005, 0.01, 0.02, 0.05} × N ∈ {500, 1000, 2000, 3500, 5000, 7500, 10000} × K=15 stratified subsamples. Run on three generators (gpt4, llama-chat, mistral). Results in `results/experiments/stability_grid_gpt4_llama_mistral/grid_summary.json`.

Headline numbers at the chosen point (C=0.005, N=7500):
- gpt4: Jaccard 0.751, n_selected 79.6, train_acc 0.972
- llama-chat: Jaccard 0.748, n_selected 77.5, train_acc 0.962
- mistral: Jaccard 0.724, n_selected 114.8, train_acc 0.892

C=0.005 is the most stable C at every N for every generator. N=7500 was chosen over N=5000 because Jaccard rises +0.07 to +0.11 (especially mistral, whose 5000→7500 gain is the largest), and the cross-generator spread tightens from ~7 pp to ~2.5 pp. N=10000 has K=1 (pool ~10k cap) so we can't measure stability there.

## Why generator set is `[gpt4, gpt2, mistral-chat, llama-chat, cohere-chat]`

- chatgpt was redundant with gpt4 (both OpenAI, both RLHF, near-identical Jaccard in the previous run). Replaced with gpt2 to add architectural+difficulty diversity (base/non-RLHF, easier detection target — the easy-end counterpart to cohere-chat at the hard end).
- gpt2 activations already exist at `results/activations_raid_gpt2/`. No extraction needed.
- Watch gpt2 in the next run: if val_acc ≥ 0.998 with n_selected ≤ 10, it has saturated and may need to be supplemented or replaced with gpt3.

## Previous full-pipeline run (now superseded)

`results/experiments/full_c0005_n5000_20260425/` — the 56-min single-seed N=5000 run on the OLD generator set (chatgpt instead of gpt2). All current report tables (§3.1–§3.7) reflect THIS run. They will need to be refreshed after the next pipeline run completes.

Key findings worth preserving in the rewrite:
- Sparse_probe: 37–47 stable features per generator at 89–99% accuracy, 82–93% core-stability across folds.
- Ablation: removing the entire selected set drops accuracy by 0.0–0.3 pp on every generator (≈ random ablation). Strong "redundancy" claim.
- Patching: monotonic flip rate growth in k on every generator. Selected-vs-random ratio 8× to 56×. Strong "sufficiency" claim. Combined with ablation = "redundantly distributed signal" story.
- Confound: 100% of selected neurons flagged on every generator. Methodologically broken (pooled-sample marginal can't separate genuine signal from domain confound). Kept as sanity check; replacement is cross-domain probe.
- AUC comparison: AUC selects 6–13× more features than L1, comparable necessity. Reframed as a parsimony argument, not a "univariate fails" argument (the previous Mistral-collapse finding was an artifact of saga's looser sparsity).
- Restricted_probe: smaller-model and ablate-complement framings agree to within 1 pp on most generators. Cohere-chat is the only one with a meaningful 4.5 pp gap.
- Cross-generator (characterize): low pairwise Jaccard (0.11–0.30), strong layer-12 dominance, cohere-chat is a structural outlier.

## What needs to happen next (in order)

1. **Launch the new full pipeline** with N=7500, 3 seeds, 5 new-set generators:
   ```
   uv run scripts/experiments/run_all.py --generators gpt4 gpt2 mistral-chat llama-chat cohere-chat --run-id full_c0005_n7500_seeds3
   ```
   Estimated runtime: ~4–4.5 h unattended. Output: `results/experiments/full_c0005_n7500_seeds3/`.

2. **Aggregate results** across seeds: per-cell `mean ± std` for accuracy, n_selected, flip rate, ablation drop, Jaccard. The runner already writes per-(seed, fold) JSONs under `<exp>/<gen>/seed_<S>/fold_<F>/`. You'll need to write a small aggregation script (or extend an existing one) — the previous one-off `_aggregate_ablation_patching.py` was deleted; rewrite if needed but don't keep it permanently.

3. **Update `experiments_progress_report.html`**:
   - §4.5 (`meth-stability`): rewrite to use the multi-generator (gpt4 + llama-chat + mistral) grid evidence. Headline numbers: C=0.005 stable across all three; N=7500 chosen for cross-generator-uniform Jaccard ~0.72–0.75. Mistral showed the strongest case for N=7500 (Δ Jaccard +0.111).
   - §3.1–§3.7: replace tables with the new multi-seed N=7500 results. Add seed-level uncertainty (mean ± std).
   - §3.6: add a random-null Jaccard baseline. For two random size-50 sets in 9216 features, expected Jaccard ≈ 0.003. Observed 0.11–0.30 → 30–100× chance. Reframes the low absolute Jaccard from a weakness into a "circuits overlap far above chance but are still generator-distinct" finding.
   - §1: drop chatgpt from generator catalog, add gpt2; mention gpt2 is the "easier end of the difficulty axis" counterpart to cohere-chat.
   - §4.6 (`publishability`): mark the multi-generator stability gap (gap #3) closed. Cross-domain (gap #2) remains open.

4. **Cross-domain generalisation** (leave-one-domain-out): final big experiment. Requires a new splitter in `data/splits.py` (~50 LoC). Estimated ~3.5 h compute. Do this AFTER the multi-seed pipeline completes and §3 is refreshed.

## Files modified this session (do not re-edit without intent)

- `scripts/experiments/run_sample_size_stability.py` — refactored to a (generators × C × N) grid runner. CLI: `--generators gpt4 llama-chat`, `--c-list 0.005 0.01 0.02 0.05`, `--n-list 500 1000 2000 3500 5000 7500 10000`, `--k-draws 15`. Output structure: `{run_id}/grid_summary.json` (flat records) plus `{run_id}/sample_size_stability/{generator}/{summary.json,draws/c{C}/n{N}/draw_{kk}.json}`.
- `config/experiments/*.yaml` — see "Current state of configs" above. Don't touch unless the user changes the methodology decision.

## Files to NEVER touch unsupervised

- `experiments_progress_report.html` outside the sections listed in step 3. The user has fine-grained control over the document.
- `raid_analysis/selection/sparse_probe.py` — `l1_ratio=1.0` is correct for the user's sklearn version (do NOT change to `penalty="l1"`; this was already reverted once after I made the mistake).
- `raid_pipeline/raid_loader.py` — produces ground-truth slug mappings. Read-only.

## Important conventions / gotchas

- Use `uv run` to execute Python scripts. `python` directly will fail with `ModuleNotFoundError: datasets`.
- Activations live at `results/activations_raid_<slug>/`. Slugs use underscores not hyphens (`llama_chat`, `mistral_chat`, `cohere_chat`); the user-facing generator strings use hyphens (`llama-chat`). `slug()` from `raid_pipeline/raid_loader.py` is the converter.
- Per-fold results: `<run_id>/<exp>/<generator>/seed_<S>/fold_<F>/`. Aggregates: `<run_id>/<exp>/<generator>/aggregate.json` (most experiments) — but `auc_comparison` writes `comparison_results.json` instead. Don't be surprised by the inconsistency.
- The user prefers concise, tabular summaries with explicit numbers. Do NOT pad responses with prose. They asked for "honest opinion" twice — push back when the data warrants it.
- The user is paying ~$1+ per query and is tracking token cost. Use subagents (`Task` with `explore` for read-only analysis, `shell` for batched edits) when the task can be delegated cleanly. Do NOT invoke subagents for trivial tasks.

## What the user has been deciding (open questions if relevant)

- Whether to add a 6th generator (Falcon, etc.) — currently parked. Not blocking.
- Whether `auc_comparison` belongs in main text or appendix given the parsimony reframing — likely appendix.
- Whether to also stability-sweep gpt2 — not done yet; ~12 min cost; would let us claim "C=0.005 stable across easy/medium/hard generators." Pending user decision after the current full pipeline run.

## Current TODO state

```
[done]   C-stability multi-generator sweep (gpt4, llama-chat, mistral)
[done]   Final C decision: C=0.005 confirmed across 3 generators
[done]   Generator swap: chatgpt → gpt2 in characterize.yaml
[done]   Multi-seed config: seeds [42, 123, 456] in all per-generator YAMLs
[done]   Bump max_samples 5000 → 7500 across 7 YAMLs
[next]   Launch full pipeline run: 5 generators × 3 seeds at N=7500
[after]  Aggregate pipeline results across seeds (mean ± std for all metrics)
[after]  Update report §4.5: rewrite with multi-generator stability sweep results
[after]  Update report §3 tables with seed-level uncertainty + new gpt2 generator
[after]  Add random-null Jaccard baseline to §3.6
[later]  Cross-domain generalisation (leave-one-domain-out)
```

## Style notes for working with this user

- They want plans before edits. Lay out what will change in a small table, then get confirmation before touching files. Exception: the user has often said "go" inline; respect that.
- Markdown comparison tables are preferred over prose paragraphs.
- Don't end with "want me to do X next?" three times in a row — they will ask when ready.
- They are doing a literature-grade interpretability paper. Avoid hype. When something is uninformative or fails, say so plainly.
