# Full pipeline results — 20260426_20_full_c0005_n7500_seeds3

Run: 2026-04-26 → 2026-04-27 (00:12), N=7500, C=0.005, seeds [42, 123, 456], 5 folds per seed (15 cells per generator × experiment).
Generators: gpt4, gpt2, mistral-chat, llama-chat, cohere-chat.
Source: `results/experiments/20260426_20_full_c0005_n7500_seeds3/`.

## Headline takeaways

- **Sparse-and-stable selection survives.** L1 at C=0.005, N=7500 picks 58–85 features per generator at 0.91–0.99 val accuracy with pairwise fold-Jaccard 0.67–0.76 (vs. expected null ≈ 0.003 inside a single generator).
- **Necessity (ablation) survives but cohere weakens.** Ablating the entire selected set drops val accuracy by 0.03–0.11 pp on gpt4/gpt2/mistral/llama and by **1.06 pp on cohere-chat** (was 0.0–0.3 pp single-seed). Random-ablation drops are <0.02 pp everywhere, so the selected set is still informative on cohere — just not perfectly redundant.
- **Sufficiency (patching) survives, but the headline ratio is fragile at low k.** Selected-vs-random flip-rate ratios at the full-set k cluster around 10×–24× for gpt2/mistral/llama/cohere; gpt4's ratio is enormous (≈125×) only because the random baseline is ~10⁻³. Saturation curves are monotonic on every generator.
- **Restricted probe gap finding shifts.** Cohere-chat's small-vs-ablate-complement gap of ~4.4 pp survives (0.0443 ± 0.0105). **gpt2 now also has a 3.6 pp gap** — it joins cohere as a "hard" generator where the ablate-complement framing is meaningfully harsher than the smaller-model framing. gpt4/mistral/llama gaps are 0.4–1.1 pp, broadly within 1 pp.
- **Cross-generator structure: claim survives, but the outlier is gpt2, not cohere.** Pairwise stable-set Jaccard 0.00–0.24 (mean 0.095, ≈32× chance). Layer-12 dominance holds for 4/5 generators (30–36% of stable set in L12); gpt2 peaks at L11 instead and has near-zero overlap with gpt4 (J=0.000).

## §3.1 sparse_probe

Val-accuracy is the L2 probe re-trained on the L1-selected features (matches `aggregate.json:accuracy_*`). `pairwise Jaccard` is over all 105 fold-pairs of selected sets within a generator. `n_stable` = neurons appearing in all 15 folds (from `aggregate.json:n_stable_neurons`).

| generator     | n_selected (mean ± std) | val_acc (mean ± std)  | train_acc (mean ± std) | pairwise Jaccard (15 folds) | n_stable / n_observed |
|---------------|-------------------------|-----------------------|------------------------|-----------------------------|-----------------------|
| gpt4          | 74.1 ± 2.9              | 0.9879 ± 0.0026       | 0.9697 ± 0.0009        | 0.7634 ± 0.0365             | 60 / 106              |
| gpt2          | 84.7 ± 4.1              | 0.9591 ± 0.0031       | 0.9021 ± 0.0017        | 0.6701 ± 0.0418             | 60 / 146              |
| mistral-chat  | 57.8 ± 2.9              | 0.9708 ± 0.0043       | 0.9343 ± 0.0013        | 0.7240 ± 0.0461             | 45 / 99               |
| llama-chat    | 63.6 ± 1.7              | 0.9862 ± 0.0030       | 0.9568 ± 0.0014        | 0.7563 ± 0.0404             | 48 / 91               |
| cohere-chat   | 72.3 ± 3.1              | 0.9052 ± 0.0076       | 0.8614 ± 0.0024        | 0.7490 ± 0.0388             | 62 / 113              |

Selection sizes are tight (CV 2–5%), val accuracy spans 0.91–0.99 with a clean difficulty ordering (cohere → gpt2 → mistral → llama → gpt4). Within-generator fold Jaccard sits at 0.67–0.76 — within a hair of the prior C×N stability sweep numbers (0.72–0.75 at this point), so the selection is reproducible across seeds. gpt2 has the loosest fold Jaccard (0.67) and the largest pool of ever-observed neurons (146), consistent with a slightly noisier selector at this difficulty.

## §3.2 ablation

Per-cell numbers come from `<gen>/seed_<S>/fold_<F>/eval_metrics.json:metrics.ablation_sweep`. The "full" column uses each fold's actual `n_selected` (53–92 across generators). Each cell is `accuracy drop = baseline − ablated` (mean ± std over 15 cells; positive = drop). Random-ablation drop at full k is shown for context.

| generator    | k=1               | k=2               | k=5               | k=10              | k=20              | k=50              | k=full (≈n_selected) | random k=full     |
|--------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|----------------------|-------------------|
| gpt4         | 0.0000 ± 0.0002   | 0.0000 ± 0.0003   | 0.0001 ± 0.0004   | 0.0001 ± 0.0007   | 0.0000 ± 0.0008   | 0.0003 ± 0.0012   | 0.0003 ± 0.0012      | 0.0001 ± 0.0003   |
| gpt2         | -0.0001 ± 0.0008  | 0.0000 ± 0.0009   | -0.0001 ± 0.0010  | -0.0005 ± 0.0011  | -0.0003 ± 0.0017  | 0.0009 ± 0.0022   | 0.0011 ± 0.0031      | 0.0000 ± 0.0008   |
| mistral-chat | -0.0001 ± 0.0009  | 0.0001 ± 0.0009   | 0.0004 ± 0.0009   | 0.0004 ± 0.0013   | 0.0005 ± 0.0015   | 0.0010 ± 0.0020   | 0.0010 ± 0.0021      | 0.0000 ± 0.0008   |
| llama-chat   | 0.0001 ± 0.0003   | 0.0001 ± 0.0005   | 0.0002 ± 0.0008   | 0.0002 ± 0.0007   | 0.0005 ± 0.0010   | 0.0009 ± 0.0011   | 0.0011 ± 0.0014      | 0.0002 ± 0.0004   |
| cohere-chat  | -0.0001 ± 0.0008  | -0.0003 ± 0.0011  | 0.0011 ± 0.0013   | 0.0018 ± 0.0024   | 0.0019 ± 0.0030   | 0.0066 ± 0.0035   | **0.0106 ± 0.0054**  | 0.0006 ± 0.0010   |

For gpt4/gpt2/mistral/llama the full-set drop sits at 0.03–0.11 pp — within noise of the random-ablation baseline. The redundancy claim survives there. Cohere-chat is now the clear deviation: 1.06 ± 0.54 pp drop at k=full vs. 0.06 pp random drop — almost an order of magnitude above noise. The previous N=5000 single-seed run capped this at 0.3 pp; with 15 cells the cohere ablation effect is real but small (and grows monotonically from k=5 upward, unlike the others).

## §3.3 patching

Per-cell numbers from `patching/<gen>/seed_<S>/fold_<F>/eval_metrics.json:metrics.patching_sweep`. `selected` and `random` are mean flip rates; `ratio` is per-cell `selected/random` then averaged across the 15 cells (so ratio std is large at low k where random flip rates are ~10⁻⁴ — see Sanity checks). The full-k row is the headline.

Selected flip rate (mean ± std):

| generator    | k=1                 | k=5                 | k=10                | k=20                | k=50                | k=full              |
|--------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|
| gpt4         | 0.0002 ± 0.0003     | 0.0009 ± 0.0009     | 0.0015 ± 0.0011     | 0.0035 ± 0.0019     | 0.0080 ± 0.0027     | 0.0123 ± 0.0034     |
| gpt2         | 0.0005 ± 0.0004     | 0.0017 ± 0.0009     | 0.0029 ± 0.0010     | 0.0058 ± 0.0016     | 0.0171 ± 0.0031     | 0.0313 ± 0.0056     |
| mistral-chat | 0.0006 ± 0.0006     | 0.0020 ± 0.0011     | 0.0034 ± 0.0011     | 0.0056 ± 0.0014     | 0.0148 ± 0.0025     | 0.0173 ± 0.0028     |
| llama-chat   | 0.0007 ± 0.0008     | 0.0014 ± 0.0011     | 0.0027 ± 0.0016     | 0.0049 ± 0.0021     | 0.0086 ± 0.0030     | 0.0107 ± 0.0038     |
| cohere-chat  | 0.0013 ± 0.0008     | 0.0058 ± 0.0026     | 0.0099 ± 0.0033     | 0.0202 ± 0.0054     | 0.0543 ± 0.0088     | 0.0815 ± 0.0132     |

Selected/random ratio (mean ± std over 15 cells):

| generator    | k=1            | k=5            | k=10           | k=20           | k=50           | k=full          |
|--------------|----------------|----------------|----------------|----------------|----------------|-----------------|
| gpt4         | 6.70 ± 9.09    | 10.41 ± 13.21  | 13.25 ± 15.03  | 93.63 ± 222.60 | 34.20 ± 39.09  | 125.17 ± 265.24 |
| gpt2         | 19.01 ± 27.55  | 9.24 ± 8.67    | 11.28 ± 13.39  | 7.55 ± 4.72    | 8.70 ± 2.43    | 10.21 ± 2.34    |
| mistral-chat | 15.76 ± 19.83  | 42.32 ± 91.11  | 22.97 ± 39.32  | 103.82 ± 251.0 | 28.56 ± 34.80  | 23.48 ± 20.03   |
| llama-chat   | 61.29 ± 66.74  | 78.37 ± 116.6  | 28.54 ± 33.98  | 16.51 ± 15.37  | 12.17 ± 5.88   | 12.40 ± 6.28    |
| cohere-chat  | 3.26 ± 3.60    | 3.20 ± 1.37    | 3.96 ± 0.93    | 5.39 ± 1.37    | 8.55 ± 1.90    | 10.53 ± 1.85    |

Selected flip rate grows monotonically with k on every generator — the saturation/sufficiency curve survives without exception. Headline ratios at full k: 10× (gpt2), 12× (llama), 11× (cohere), 23× (mistral), 125× (gpt4). The first four sit inside the previous "8×–56×" range. gpt4 is anomalously high only because its random-flip baseline at full k is ~8 × 10⁻⁴ (i.e. activations there barely flip the prediction at all when shuffled), inflating the ratio with high variance. The robust signal is the selected flip-rate column itself, which is a clean monotone everywhere.

## §3.4 confound

`confound/<gen>/seed_<S>/fold_<F>/eval_metrics.json` reports per-neuron `flagged` (significant pooled-sample marginal correlation with text-length OR domain at p<0.05). Fractions below are mean ± std over 15 cells of `n_flagged / n_selected`.

| generator    | frac flagged any (mean ± std) | frac flagged text-length | frac flagged domain | n_selected (mean ± std) |
|--------------|-------------------------------|--------------------------|---------------------|-------------------------|
| gpt4         | 0.9991 ± 0.0035               | 0.8126 ± 0.0346          | 0.9991 ± 0.0035     | 74.1 ± 3.0              |
| gpt2         | 1.0000 ± 0.0000               | 0.8087 ± 0.0150          | 0.9920 ± 0.0059     | 84.7 ± 4.2              |
| mistral-chat | 1.0000 ± 0.0000               | 0.7867 ± 0.0497          | 1.0000 ± 0.0000     | 57.8 ± 3.0              |
| llama-chat   | 1.0000 ± 0.0000               | 0.8815 ± 0.0277          | 1.0000 ± 0.0000     | 63.6 ± 1.8              |
| cohere-chat  | 1.0000 ± 0.0000               | 0.7726 ± 0.0290          | 1.0000 ± 0.0000     | 72.3 ± 3.2              |

Essentially every selected neuron is flagged on every generator (≥ 99.9%). This reproduces the previous run exactly and confirms the methodological caveat already noted: the pooled-sample marginal test conflates true signal with domain confound, because human-vs-AI labels are themselves correlated with domain in RAID. Treat this as a sanity check that the selector finds neurons that *are* correlated with the labels in some way — the cross-domain (leave-one-domain-out) experiment is the proper resolution.

## §3.5 auc_comparison

`auc_comparison/<gen>/comparison_results.json:aggregate`. AUC selection takes the top-K univariate AUC features at the same nonzero count as L1; the L1 vs AUC ablation rows compare downstream val accuracy after ablating each set.

| generator    | n_AUC (mean ± std) | n_L1 (mean ± std) | AUC/L1 ratio | baseline acc      | AUC ablation acc (drop)              | L1 ablation acc (drop)                |
|--------------|--------------------|-------------------|--------------|-------------------|--------------------------------------|---------------------------------------|
| gpt4         | 491.3 ± 11.3       | 74.1 ± 2.9        | 6.63         | 0.9879 ± 0.0026   | 0.9777 ± 0.0038 (0.0102 ± 0.0026)    | 0.9876 ± 0.0029 (0.0003 ± 0.0012)     |
| gpt2         | 86.3 ± 6.7         | 84.7 ± 4.1        | 1.02         | 0.9591 ± 0.0031   | 0.9585 ± 0.0031 (0.0006 ± 0.0017)    | 0.9580 ± 0.0039 (0.0011 ± 0.0030)     |
| mistral-chat | 238.4 ± 5.7        | 57.8 ± 2.9        | 4.12         | 0.9708 ± 0.0043   | 0.9655 ± 0.0041 (0.0052 ± 0.0026)    | 0.9698 ± 0.0043 (0.0010 ± 0.0020)     |
| llama-chat   | 307.7 ± 8.2        | 63.6 ± 1.7        | 4.84         | 0.9862 ± 0.0030   | 0.9807 ± 0.0037 (0.0055 ± 0.0033)    | 0.9851 ± 0.0034 (0.0011 ± 0.0013)     |
| cohere-chat  | 37.6 ± 2.6         | 72.3 ± 3.1        | 0.52         | 0.9052 ± 0.0076   | 0.9024 ± 0.0078 (0.0028 ± 0.0027)    | 0.8946 ± 0.0089 (0.0106 ± 0.0052)     |

Parsimony framing: on gpt4, mistral, llama the L1 set is 4–7× smaller than AUC's set yet gives a *smaller* downstream ablation drop — L1 selection is strictly more parsimonious. gpt2 is the special case where AUC and L1 pick the same number (~85) and produce indistinguishable downstream effects (the "univariate is enough" regime). cohere-chat reverses the relationship: here AUC's set is ~half the size of L1's *and* its ablation drop is roughly a quarter of L1's — for cohere, top-AUC features are the more efficient encoding; this is consistent with the §3.2 finding that the L1 set on cohere has a real necessity contribution (so removing it costs more) and with §3.7 below.

## §3.6 restricted_probe

Per-cell numbers from `restricted_probe/<gen>/seed_<S>/fold_<F>/eval_metrics.json`. `small` = L2 probe trained from scratch on only the selected features. `ablC` = baseline probe with the *complement* (everything except selected) ablated. `gap = small − ablC` (positive: smaller-model framing is more lenient).

| generator    | base_acc            | small_acc           | ablC_acc            | small drop          | ablC drop           | gap (small − ablC) |
|--------------|---------------------|---------------------|---------------------|---------------------|---------------------|--------------------|
| gpt4         | 0.9879 ± 0.0027     | 0.9717 ± 0.0049     | 0.9676 ± 0.0058     | 0.0162 ± 0.0046     | 0.0203 ± 0.0052     | 0.0041 ± 0.0044    |
| gpt2         | 0.9591 ± 0.0033     | 0.9133 ± 0.0060     | 0.8770 ± 0.0068     | 0.0458 ± 0.0053     | 0.0821 ± 0.0071     | **0.0363 ± 0.0084** |
| mistral-chat | 0.9708 ± 0.0044     | 0.9344 ± 0.0055     | 0.9251 ± 0.0047     | 0.0364 ± 0.0058     | 0.0457 ± 0.0053     | 0.0093 ± 0.0043    |
| llama-chat   | 0.9862 ± 0.0032     | 0.9604 ± 0.0028     | 0.9490 ± 0.0053     | 0.0258 ± 0.0045     | 0.0372 ± 0.0056     | 0.0114 ± 0.0043    |
| cohere-chat  | 0.9052 ± 0.0079     | 0.8651 ± 0.0096     | 0.8208 ± 0.0140     | 0.0401 ± 0.0105     | 0.0844 ± 0.0135     | **0.0443 ± 0.0105** |

Cohere-chat's ~4.4 pp gap reproduces the previous-run finding (was 4.5 pp, now 4.43 pp). New: **gpt2 also has a 3.6 pp gap** — joining cohere as a generator where the ablate-complement framing penalises the selected set ~2× harder than the smaller-model framing. gpt4/mistral/llama gaps sit at 0.4/0.9/1.1 pp, comfortably under the previous 1 pp threshold. Reading: the selected feature set carries more standalone signal than its baseline-context contribution warrants on the easier generators (gpt4, llama, mistral), but on cohere and gpt2 there is a real "the rest of the model carries information too" contribution that the smaller-model framing misses.

## §3.7 characterize (cross-generator)

From `characterize/characterize_results.json` (cross-generator structural comparison built on each generator's per-fold-stable-set, using the run's `core_min_generators=2` setting; per-generator stable sizes shown in `per_generator_n_stable`).

Per-generator stable-set sizes (used in the Jaccard table): cohere-chat 62, gpt2 60, gpt4 60, llama-chat 48, mistral-chat 45.

Pairwise Jaccard (off-diagonal):

|              | cohere-chat | gpt2     | gpt4     | llama-chat | mistral-chat |
|--------------|-------------|----------|----------|------------|--------------|
| cohere-chat  | 1.000       | 0.0339   | 0.1091   | 0.0784     | 0.2442       |
| gpt2         | 0.0339      | 1.000    | **0.0000** | 0.0189   | 0.0500       |
| gpt4         | 0.1091      | 0.0000   | 1.000    | 0.1134     | 0.1413       |
| llama-chat   | 0.0784      | 0.0189   | 0.1134   | 1.000      | 0.1625       |
| mistral-chat | 0.2442      | 0.0500   | 0.1413   | 0.1625     | 1.000        |

Layer dominance (count of stable neurons by BERT layer; layer with the most for that generator in **bold**):

| layer | cohere-chat | gpt2     | gpt4     | llama-chat | mistral-chat |
|-------|-------------|----------|----------|------------|--------------|
| 1     | 5           | 4        | 1        | 0          | 4            |
| 2     | 3           | 3        | 5        | 8          | 2            |
| 3     | 10          | 8        | 2        | 6          | 1            |
| 4     | 0           | 4        | 5        | 1          | 3            |
| 5     | 4           | 4        | 5        | 1          | 2            |
| 6     | 4           | 4        | 2        | 0          | 2            |
| 7     | 4           | 3        | 5        | 8          | 4            |
| 8     | 2           | 4        | 3        | 3          | 1            |
| 9     | 3           | 1        | 5        | 0          | 0            |
| 10    | 5           | 6        | 3        | 3          | 5            |
| 11    | 3           | **11**   | 6        | 2          | 5            |
| 12    | **19**      | 8        | **18**   | **16**     | **16**       |
| L12 % | 30.6%       | 13.3%    | 30.0%    | 33.3%      | 35.6%        |

Random-null Jaccard baseline: for two random size-K_A and size-K_B sets in 9216 features, expected J ≈ K_A·K_B / (9216·(K_A+K_B) − K_A·K_B). Multiple-of-chance per pair:

| pair                          | K_A | K_B | observed J | expected J | × chance |
|-------------------------------|-----|-----|------------|------------|----------|
| cohere-chat – gpt2            | 62  | 60  | 0.0339     | 0.0033     | 10.21×   |
| cohere-chat – gpt4            | 62  | 60  | 0.1091     | 0.0033     | 32.86×   |
| cohere-chat – llama-chat      | 62  | 48  | 0.0784     | 0.0029     | 26.64×   |
| cohere-chat – mistral-chat    | 62  | 45  | 0.2442     | 0.0028     | 86.06×   |
| gpt2 – gpt4                   | 60  | 60  | 0.0000     | 0.0033     | 0×       |
| gpt2 – llama-chat             | 60  | 48  | 0.0189     | 0.0029     | 6.50×    |
| gpt2 – mistral-chat           | 60  | 45  | 0.0500     | 0.0028     | 17.87×   |
| gpt4 – llama-chat             | 60  | 48  | 0.1134     | 0.0029     | 39.08×   |
| gpt4 – mistral-chat           | 60  | 45  | 0.1413     | 0.0028     | 50.50×   |
| llama-chat – mistral-chat     | 48  | 45  | 0.1625     | 0.0025     | 64.32×   |
| **mean over 10 pairs**        |     |     | **0.0952** | **0.0030** | **~32×** |

Reading: pairwise Jaccard is low in absolute terms (0.00–0.24) but on average ~32× chance, with the strongest overlaps being cohere–mistral (86×) and llama–mistral (64×). The structural outlier is **gpt2**: it has 0 overlap with gpt4, only 0.019 with llama, and its stable set peaks at L11 instead of L12 (13% in L12 vs. 30–36% for the other four). Cohere — flagged as the structural outlier in the previous run — is in fact the most-overlapped generator with mistral here. Layer-12 dominance survives for the four chat-family generators; gpt2 (the new "easy" non-chat baseline) is the genuine cross-generator deviation.

## Sanity checks

- **gpt2 saturation: PASS (not saturated).** sparse_probe gpt2 has val_acc = 0.9591 and n_selected = 84.7 (range 80–92). The fail rule (val_acc ≥ 0.998 *and* n_selected ≤ 10) is not met — the headline "gpt2 = harder than RLHF chat for the L1 selector" picture from the C×N stability sweep is preserved. No supplement / replacement with gpt3 is indicated by the data.
- **Multi-seed instability flags (|std/mean| > 30% on a non-near-zero quantity):**
  - **Ablation full-set drops** are noisy as a fraction of mean for gpt2 (295%), mistral (217%), llama (126%) — but only because the absolute mean drop is 0.001 (i.e. essentially zero); these are *stably-near-zero*, not unstable. cohere is the only one with a substantively-nonzero mean (0.0106 ± 0.0054 = 51% std/mean — still meaningful).
  - **Patching ratios** at low k have std/mean > 100% on most generators because the random-flip denominator is ~10⁻⁴ and varies ~100% itself across folds. The selected flip-rate column (numerator only) is much more stable (std/mean 33–80% at most k). Use the selected-flip-rate column or the cohere-chat ratio (which is stable at 1.85–3.60 std/mean ≤ 110%) for narrative; treat gpt4 and gpt2 ratios at low k as illustrative not numerical.
  - **Restricted-probe gaps**: gpt4 small-vs-ablC gap is 0.0041 ± 0.0044 (106% std/mean). The cohere (4.4 ± 1.1 pp) and gpt2 (3.6 ± 0.8 pp) gaps are stable (~25% std/mean). The gpt4 finding is "no real gap, dominated by per-cell noise"; treat it as ~0.
  - Sparse_probe accuracies, n_selected, and pairwise fold-Jaccards have std/mean ≤ 7% on all generators — no instability there.
- **Anything unexpected:**
  - cohere-chat baseline accuracy at 0.9052 is materially lower than every other generator — the L1 selector is still working, but the absolute task is harder. This is consistent with cohere being the difficulty floor of the panel.
  - gpt2 is the new structural outlier in the cross-generator picture (L11-peaked, 0 Jaccard with gpt4). Previously cohere held that role. The L11/L12 split is interesting: BERT's late layers split between generator-specific (L12, all four chat-family generators) and a more pretraining-style register (L11, gpt2 only). Worth a sentence in the paper.
  - patching cohere full ratio (10.5×) is now very stable (1.85 std). cohere went from "structural outlier with weakest overlap" to "structural outlier with strongest local overlap with mistral and the only generator with a nonzero ablation cost". The N=7500 + 3-seed numbers paint a more coherent picture of cohere as "harder generator with a more dispersed but still real circuit", not as a broken case.

## Comparison to previous N=5000 single-seed run

| claim                                                                         | previous N=5000, 1 seed                            | current N=7500, 3 seeds (15 cells/cell)                     | survives?  |
|-------------------------------------------------------------------------------|----------------------------------------------------|-------------------------------------------------------------|------------|
| sparse_probe: 37–47 stable features per generator at 89–99% val accuracy     | 37–47 stable, 0.89–0.99 acc                        | 45–62 stable, 0.91–0.99 val_acc                             | Y (slightly more stable features at N=7500) |
| sparse_probe: 82–93% core stability across folds                              | 0.82–0.93                                          | 0.67–0.76 pairwise Jaccard, 0.58–0.70 mean neuron-stability  | weakened (different metric — Jaccard is stricter; "stability" claim still holds qualitatively) |
| ablation: full-set drop 0.0–0.3 pp on every generator                        | 0.0–0.3 pp                                         | 0.03–0.11 pp on gpt4/gpt2/mistral/llama; **1.06 pp on cohere** | weakened on cohere only                       |
| patching: monotone curve, selected-vs-random ratio 8×–56×                    | monotone, 8×–56×                                   | monotone everywhere; ratios 10×–24× on 4/5 generators; gpt4 ≈125× (noisy due to tiny denominator) | Y                                          |
| confound: 100% of selected flagged on every generator                         | 100%                                               | ≥ 99.9% on every generator                                  | Y (and methodological caveat unchanged)    |
| restricted_probe: small/ablC gap ≤ 1 pp on most; cohere ~4.5 pp              | gpt4/mistral/llama ≤ 1 pp, cohere ~4.5 pp          | gpt4/mistral/llama 0.4/0.9/1.1 pp, cohere 4.43 pp, **gpt2 3.63 pp** | partially survives (cohere finding intact; gpt2 joins the "wide gap" club) |
| cross-generator pairwise Jaccard 0.11–0.30, layer-12 dominance, cohere outlier | 0.11–0.30, L12-dominant, cohere outlier            | 0.00–0.24, L12-dominant for 4/5, **gpt2 outlier** (0 with gpt4, peaks at L11) | partially survives (range broadened, outlier identity changed because the generator panel changed; chatgpt → gpt2) |

## Open questions / next actions

- **Cross-domain (leave-one-domain-out)** is now the obvious next experiment. §3.4 confound is methodologically broken in the way already documented; LODO replaces it as the proper test of whether selected neurons survive a held-out domain. Estimated ~3.5 h compute per the handoff.
- **gpt2 stability sweep** (~12 min): the C×N grid was only run on gpt4/llama/mistral. Now that gpt2 is in the headline panel and is the new structural outlier, running the same sweep on gpt2 would let us claim "C=0.005 is stable across the easy/medium/hard span" rather than "stable across the medium/hard span". Cheap win.
- **Cohere ablation result needs a sentence in the paper**: the 1.06 pp full-set ablation drop is now ≈10× the cross-generator average. It does not break the redundancy claim (because random ablation drops are also bigger on cohere — 0.06 pp vs. ~0.001–0.02 pp elsewhere — i.e. cohere features are individually more important everywhere), but it does deserve "cohere is partially distributed and partially localized" framing instead of "uniformly redundantly distributed".
- **gpt2 restricted-probe gap (3.6 pp) is new and interesting**. Worth checking whether gpt2's stable-set-vs-rest split looks more like cohere's (signal genuinely shared with the rest of the model) than like gpt4's. Could be one paragraph of analysis in §3.6 of the report.
- **Patching ratio narrative**: the high std/mean on ratios at low k makes ratio numbers fragile. Either (a) report selected flip-rates as the headline and ratio in a footnote, or (b) bootstrap the ratio properly per fold. Recommend (a) — the saturation curves are the load-bearing claim, the ratios are a secondary check.
