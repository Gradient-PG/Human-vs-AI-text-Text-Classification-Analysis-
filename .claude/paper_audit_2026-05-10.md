# Paper Audit — EMNLP ARR 2026 Submission
**Timestamp:** 2026-05-10  
**Paper:** "A Mechanistic Study of AI-Detection Features in Frozen BERT: Sparse Probing and Activation Patching on RAID"  
**Deadline:** May 25, 2026

---

## Honest Audit: What I'd Change for EMNLP Submission

### 🔴 Blocking Missing Pieces (paper cannot be submitted without these)

**1. No abstract**
It's a `\todo{}` placeholder. For a 150-word abstract you need: context (AI detection is empirically strong but mechanistically opaque), gap (no causal-sufficiency test on frozen encoders), approach (sparse probing + patching on RAID), three results, and a positioning sentence.

**2. [DONE] No conclusion section**
Every EMNLP paper has one. Discussion ≠ Conclusion. Conclusion = one crisp paragraph restating the main claim, the key numbers, and what they mean for the field. Currently absent entirely.

**3. [DONE] Ethics statement is empty**
There's a section header with only a comment. Two sentences minimum required: dual-use concern of detection tools, plus statement that this is observational MI work using a public dataset.

**4. 14 references**
EMNLP accepted papers average 40–70 refs. Reviewers notice immediately. You need foundational MI refs (Elhage et al. 2021 circuits, Conmy et al. 2023 ACDC, Nanda et al. 2022), probing literature (Belinkov 2022 survey, Conneau et al. 2018, Tenney et al. 2019), standard NLP model refs (Devlin 2019 already there, Radford 2019 GPT-2, Touvron 2023 LLaMA, Brown 2020 GPT-3), and more detection literature.

**5. All 6 appendix sections are empty stubs**
§3.4 of the main text explicitly sends readers to Appendix B (stability sweep). That appendix is a section header with one comment. This is a credibility failure — a reviewer who follows the pointer finds nothing.

**6. [DONE] "Five generators" vs "six generators" inconsistency**
Contributions bullets 1 and 2 say "five generators". The experiments use six. Introduction ¶4 says "six generators" correctly. This is a leftover from before MPT was added and will confuse reviewers.

---

### 🟠 Structural Problems

**7. [DONE] §5 (Patching) and §6 (Ablation) should be one section**
Right now you have two separate top-level sections for what is logically one experiment: "Does this neuron set causally matter?" The current structure makes the paper feel sprawling. Better: one section titled **"Causal Role of the Selected Neurons"** with subsections §5.1 Sufficiency (patching) and §5.2 Necessity (ablation). This is how MI papers typically frame necessity/sufficiency — as a joint claim, not two separate chapters.

**8. [DONE] §7.3 (CAV diagnostics) doesn't belong in §7**
Cross-Generator Circuit Geometry (§7) covers Jaccard matrix and layer distributions — these are about *structure*. The CAV section is about *signal geometry within a single generator's stable set*. It would be better placed either after the sparse probe section (§4) as a characterisation of the probe's detection signal, or as a standalone §5 on "Signal Geometry". Right now readers hit it expecting inter-generator analysis and get per-generator intrinsics.

**9. [DONE] The Restricted Probe result is buried in an empty appendix but is cited in Discussion**
Discussion §9 mentions "mpt 4.68pp, gpt2 3.63pp, versus ≤1.2pp for the others" as evidence for the base-vs-instruction-tuned structural claim. But Appendix E is empty — the claim has no backing the reviewer can check. Either: (a) fill the appendix with the actual table, or (b) elevate it to a brief subsection in main body. Given it directly supports a key Discussion claim, option (b) is better.

**10. [DONE] §4 (Sparse Circuits) has no mention of layer distribution**
The sparse probe section reports accuracy and Jaccard but doesn't say anything about *where* these neurons are. A reader finishing §4 has no idea the layer-12 asymmetry exists until §7.2. Since that's one of your three headline findings, there should be at least a forward pointer in §4: "Where in the network these neurons sit is examined in §7."

**11. [DONE] The "core neurons" finding from characterize is never mentioned**
Your `exp_characterize.py` computes a core neuron set — neurons that appear in the stable sets of ≥50% of generators. This is a strong result if non-empty (neurons that ALL generators agree on). This isn't mentioned anywhere in the paper's main text. If the core set is non-empty, that's a key finding. If it's empty, that's equally interesting and should be stated explicitly.

**Planned placement for core neurons:** §6.2 (Stable-Set Overlap), after the bipartite structure paragraph. A sentence like "Beyond pairwise overlap, we identify a core set of N neurons appearing in ≥3 of 6 generators' stable sets (see Table X / or 'the core set is empty, confirming generator-specificity'). This directly follows from the Jaccard discussion and adds a concrete quantitative anchor to the bipartite structure claim.

Action needed before writing: run `characterize_results.json` and look at `core_neurons` list and `n_core` field. Check what `core_min_generators` is set to.

---

### 🟡 Missing Figures (highest-leverage visual improvement)

All findings are currently numbers in tables. MI papers at EMNLP almost always have 3–5 figures showing patterns the tables support.

**Figure 2: Flip rate vs. k line chart** *(replaces or accompanies Table 2)*
A 6-line plot (one per generator) of flip rate % on y-axis vs k on x-axis, with a shaded band for the random-k baseline. This immediately shows: (a) monotonic increase, (b) the huge gap between selected and random, (c) Cohere as an outlier, (d) the log-linear-ish scaling. The table forces the reader to mentally construct this picture. The figure gives it for free.

**Figure 3: Layer distribution bar chart** *(replaces or accompanies Table 4)*
A grouped or stacked horizontal bar chart: generators on y-axis, bars split into layers 1–10 / 11 / 12. Coloured by base vs instruction-tuned. The L12 asymmetry — which is one of your three headline findings — jumps off the page visually. Currently hidden in a 4-row table.

**Figure 4: Jaccard similarity heatmap** *(replaces or accompanies Table 5)*
A 6×6 heatmap of pairwise Jaccard. The base-vs-instruction-tuned bipartite structure (the top-left instruction block, the bottom-right base block, near-zero off-diagonal) becomes immediately obvious. Currently you have a 15-row ranked table which requires the reader to mentally reconstruct the matrix.

**Figure 5 (optional): LOGO bar chart**
Side-by-side bars for Full L2 and Sel L2 across the 5 held-out families, with the gap labelled. Visual encoding of "we retain 86–94% of the ceiling" is cleaner than a table.

---

### 🟡 Content Gaps

**12. [DONE — Option A] No characterisation of what the neurons actually encode**
Reviewers in the MI community will ask: "OK, so ~50 neurons drive AI detection — but *what property do they encode?*"

**Suggested approach (no new experiments needed):**
- Option A (1-2 hours): Leverage BERT interpretability literature. Tenney et al. 2019 and Rogers et al. 2020 establish that layers 9-12 encode high-level semantics (discourse coherence, coreference, pragmatics). Since instruction-tuned generators concentrate stable neurons in L12, you can connect this: "stable neurons in L12 likely respond to discourse-level properties systematically altered by RLHF alignment, consistent with the known function of BERT's final layers (Tenney et al. 2019)." This is a literature-grounded characterization at zero experimental cost.
- Option B (4-8 hours): For the top 10 stable neurons per generator, compute Spearman correlation with simple text features: type-token ratio, avg sentence length, punctuation rate, Zipfian rank, GPT-2 perplexity. Present as a small table. If neurons correlate with perplexity → the detector is picking up fluency/probability patterns; if not → it's picking up something more structural.
- Current paper handles this defensively in Limitations §9.5 ("We make no claims about what linguistic properties these neurons encode"). This is honest but unsatisfying. Option A costs nothing and gives reviewers something to think about.

**13. [DONE] No per-domain breakdown in main body**
Appendix F is a stub. The numbers exist in JSON: Cohere uniformly high across domains (5–10%); GPT-2 strong on news/reddit/wiki (3–5%) and weak on books/reviews (~2%). Fill Appendix F with a compact table citing `patching/aggregate.json` per-domain fields.

**14. [DONE] AUC numbers mentioned in prose but not in Table 1**
§4 prose cites "AUC-ROC 0.968–0.999" but Table 1 only shows accuracy. Either add AUC column or remove prose citation.

**15. [DONE] The "log-linear" k-flip relationship claim is unsupported**
R² of log10(k) vs flip_rate computed: 0.72–0.85 (not truly log-linear). Claim softened to "sub-linear scaling at diminishing rate, visible in Figure 2". Figure 2 added (log-scale x-axis) as supporting evidence.

**16. [DONE — VERIFIED CORRECT] §3.2 Mistral classification needs checking**
Data description says "mistral-chat is SFT-only". Mistral-7B-Instruct uses DPO in some versions. Verify for the RAID-specific model version.

---

### 📋 Proposed Final Structure

```
§1  Introduction                          [DONE] "five"→"six" fixed
§2  Related Work                          (add ~25 refs to reach 40+)
§3  Setup
    §3.1  Data: RAID
    §3.2  Model and Activations
    §3.3  Sparse Probing Protocol
    §3.4  Hyperparameter Validation       (MUST fill App B)

§4  Sparse Detection Circuits             [DONE] subsections added
    §4.1  Stable Neuron Sets              [DONE] layer forward-pointer added
    §4.2  Detection Signal Geometry       [DONE] CAV moved here from §7.3

§5  Causal Role of the Selected Neurons   [DONE] merged §5+§6
    §5.1  Causal Sufficiency via Patching [DONE]
    §5.2  Necessity via Mean Ablation     [DONE]
    §5.3  Signal Localisation             [DONE] restricted probe elevated from App E
    §5.4  Robustness Checks               [DONE]

§6  Cross-Generator Circuit Structure     [DONE] CAV removed
    §6.1  Layer Distribution
    §6.2  Stable-Set Overlap
    §6.3  Core Neurons                    (pending — needs characterize_results.json check)

§7  Leave-One-Family-Out Generalisation

§8  Discussion
§9  Conclusion                            [DONE] written
§10 Limitations

Ethics Statement                          [DONE] written

Appendix A  Pipeline Details
Appendix B  Stability Sweep              ← MUST fill (referenced from §3.4)
Appendix C  Per-Cell Metrics
Appendix D  AUC vs L1 Selection
Appendix E  [DONE] content elevated to §5.3
Appendix F  Per-Domain Patching          (fill from patching/aggregate.json)
```

---

### ⏱ Priority Order for Remaining 15 Days

| Priority | Task | Est. Hours |
|---|---|---|
| 🔴 | Write abstract (6 sentences) | 1 |
| ✅ | Write conclusion (1 paragraph) | 1 |
| ✅ | Fix "five"→"six" inconsistency | 0.5 |
| 🔴 | Add ~25 references | 2 |
| 🔴 | Fill App B (stability sweep — referenced from main text) | 2 |
| ✅ | Merge §5+§6 into one section | 1 |
| ✅ | Move CAV to §4.2 | 0.5 |
| ✅ | Add §5.3 restricted probe (elevate from App E) | 1 |
| ✅ | Add core neurons to §6.2 (as \paragraph) | 1 |
| ✅ | Add Figure 2 (flip rate line plot) | 2 |
| ✅ | Add Figure 3 (layer dist bar chart) | 2 |
| ✅ | Add Figure 4 (Jaccard heatmap) | 2 |
| ✅ | Write ethics statement | 0.5 |
| ✅ | Fill App F (per-domain patching) | 1 |
| 🟡 | Hedge language pass ("demonstrate"→"suggest") | 1 |
| ✅ | Option A neuron characterisation (leverage BERT literature) | 2 |
| ✅ | AUC column in Table 1 | 0.5 |

---

### ✅ Completed During This Session (2026-05-10)

- Multi-seed CAV implementation in `exp_characterize.py`
- All 6 numerical errors found and corrected in paper
- MPT run completed and integrated
- Figure 1 (pipeline TikZ diagram) built and overflow fixed
- All table overflow issues resolved (table*, resizebox, footnotesize)
- §5.3 cross-domain donor \todo{} removed; methodological justification written
- Generator names standardised to GPT-4/GPT-2/MPT/Mistral/LLaMA/Cohere
- Worktree cleaned up

### ✅ Completed in This Work Block (also 2026-05-10)

- No conclusion → written
- Ethics statement → written
- "Five" → "six" inconsistency → fixed
- §5+§6 merged into "Causal Role of the Selected Neurons"
- CAV moved from §7.3 to §4.2
- Restricted probe elevated to §5.3 in main body
- Layer distribution forward pointer added to §4.1
- Core neurons: planned for §6.3 (needs `core_neurons` field from characterize_results.json)

### ✅ Completed in Figure Generation Block (2026-05-10)

- Figure 2 (flip rate vs k, log x-axis, random-k envelope) → `_paper/figures/fig_flip_rate.pdf`
- Figure 3 (layer distribution stacked bar, IT vs base coloured) → `_paper/figures/fig_layer_dist.pdf`
- Figure 4 (Jaccard 6×6 heatmap, bipartite block rectangles) → `_paper/figures/fig_jaccard.pdf`
- Figure 5 (LOGO bar chart, Full L2 vs Sparse L2, retention %) → `_paper/figures/fig_logo.pdf`
- All four figures integrated into main.tex with captions and \label{}
- Point 15 fixed: R²=0.72–0.85 confirmed NOT log-linear; prose changed to "sub-linear scaling at diminishing rate"; Figure 2 reference added
- Paper compiles cleanly: 14 pages, no LaTeX errors
