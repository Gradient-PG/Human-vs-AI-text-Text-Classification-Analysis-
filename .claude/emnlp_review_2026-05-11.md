# EMNLP 2026 Submission Review — Personal Claude Feedback
*Generated: 2026-05-11*

## Overall Verdict
The paper is in genuinely good shape. Every Tier-1 novelty-audit recommendation was delivered: six generators, proper LOGO experiment, Borile & Abrate addressed prominently, stability-driven hyperparameter selection in appendix, mean-ablation caveat acknowledged. Structure matches the scaffold. "Easy to inject, hard to erase" framing reads cleanly. The base/instruction-tuned bipartite story is memorable.

**Ready for EMNLP 2026? Yes, with two weeks of polish.**
- Ceiling: acceptance with revisions
- Floor: Findings
- No fatal flaw

---

## Critical Issues (fix before submission)

### C1 — Table 3 missing random-k baseline column
The 10–16× claim appears only as a prose range and a grey envelope in Figure 2. Reviewers will want per-generator random-k baseline at k=full in the table, with explicit ratios. GPT-4 ratio ~125× with σ≈265 is flagged in prose but absent from the table — looks like hiding the high-variance case. **Add a column to Table 3 giving random-k baseline at k=full and the explicit ratio.**

### C2 — Remove `exp_characterize.py` reference in §6.2
Implementation file names don't belong in the paper body. Rephrase as "the minimal majority threshold (≥3 of 6 generators)" without naming the script.

### C3 — Justify six-of-eight RAID domain choice in §3.1
RAID has eight domains; the paper uses abstracts, books, news, Reddit, reviews, Wikipedia (drops poetry and recipes). A reviewer who knows RAID will flag this in 30 seconds. **Add one sentence explaining the choice** (e.g., short-text domains where CLS aggregation is most meaningful, or sample-size constraints).

### C4 — Frame flip-rate denominator in §5.1.2 or intro
1.07–8.15% looks tiny to a skimming reviewer. Add one sentence: *"Note that the flip-rate denominator is human samples already correctly classified by the frozen evaluator; the 1.07–8.15% range therefore represents predictions reversed despite the probe's high baseline confidence."*

---

## Important Issues (strongly recommended)

### I1 — Replace Enkhbayar (2025) in load-bearing intro role
Single-author arXiv preprint framing the "causality gap" is reviewer-bait. Options:
- (a) Soften: "Recent work [Enkhbayar 2025; preprint] *suggests*…" not "documents a gap"
- (b) **Replace with Antverg & Belinkov (ICLR 2022, "On the Pitfalls of Analyzing Individual Neurons")** — same encoded-vs-used point, peer-reviewed venue. Recommended.

### I2 — Add redundancy literature to §8
"Redundantly distributed" discussion lacks canonical citations:
- **Dalvi et al. (EMNLP 2020)** — "Analyzing Redundancy in Pretrained Transformer Models" (85% of neurons redundant, 92% removable)
- **Elhage et al. (2022)** — "Toy Models of Superposition" (theoretical anchor)
Citing these transforms "we observe this pattern" → "this matches the established redundancy literature."

### I3 — Add one-line detector comparison in §3 or §5
Paper explicitly positions itself as "not a new detector" — fine — but a one-line note giving Ghostbuster or Binoculars accuracy on the six-generator subset (or citing Dugan et al.'s numbers) defuses the "is this competitive?" question.

### I4 — Relocate or rename §5.3 (Restricted Probe / Localization)
Localization isn't a causal-role question — it's structural. Options:
- Move to §6 Cross-Generator Geometry as §6.0 (pairs well with bipartite findings)
- Rename §5 to "Causal Role and Localization of the Selected Neurons"

### I5 — Add qualitative analysis (biggest missing piece)
No qualitative examples anywhere. Even **Appendix G with two examples per generator** would help:
- (a) A human text that flipped to AI under patching
- (b) The AI donor text it borrowed from
- (c) Which layer-12 neuron(s) carried the largest activation change

---

## Minor Issues

### M1 — Abstract opening is generic
Current: "achieve high accuracy on standard benchmarks, yet the internal representations remain poorly understood"
Consider leading with the finding: *"A frozen BERT-base-uncased encoder already contains sparse, causally sufficient AI-detection circuits — we identify them via L1 sparse probing on the RAID benchmark and validate sufficiency through same-domain donor activation patching."*

### M2 — CAV §4.2 linkage to §5 missing
Add one sentence at end of §4.2: *"These per-axis geometric differences predict the intervention asymmetries documented in §5."*

### M3 — Verify Pudasaini et al. (2026, arXiv:2603.23146)
arXiv ID 2603 = March 2026 — check paper exists and author list is correct.

### M4 — Hedge third contribution in §1 contributions list
"Instruction-tuned generators concentrating 30–36% of stable neurons in layer 12" leans on a 2-vs-4 contrast. Consider: *"a tentative cross-generator circuit analysis suggesting a layer-12 footprint of post-training alignment, supported by two independent base models"*

### M5 — Split conclusion into 2–3 paragraphs
Currently a single dense paragraph. Reviewers and skimmers read abstract + intro contributions + conclusion. One paragraph per main finding aids skimmability.

### M6 — §6.1 Geva et al. speculation
The FFN-as-key-value-memory reading for the layer-12 effect is appropriately hedged but interpretation-heavy. Consider trimming if space is needed.

### M7 — Table 4 statistical insignificance not stated
+0.03 to +0.11 pp drops with ±0.11 to ±0.31 pp SDs — some are within one SD of zero. Add: *"Drops for GPT-4, MPT, Mistral, and LLaMA are within one standard deviation of the random baseline, indicating no detectable necessity at this scale."*

### M8 — Standardize preprint vs. venue citation format
Some preprints have "Preprint, arXiv:…" and others don't. Be consistent.

---

## Conciseness Issues

| Location | Problem |
|----------|---------|
| §5.2 "There is no contradiction…" paragraph | Reiterates the redundancy argument three times. One clean statement suffices. |
| §6.2 bipartite Jaccard + core-neurons | Tell the same story. Core-neurons block could compress to two sentences. |
| §8 Discussion Cohere story | Already told in §4.2, §5.1.2, §5.2. Replace with synthesis sentence; free space for Dalvi/Elhage citations. |

---

## Things Discussed But Not In Paper

### Missing1 — Zero-ablation as sharper necessity test
§5.1.3 flags as future work. If runnable in the next week, it would be a much stronger response than "left as future work." **Confirm if feasible.**

### Missing2 — Base/chat within-family comparison
The agreed-upon clean pairs (cohere/cohere_chat, mistral/mistral_chat, llama/llama_chat, mpt/mpt_chat) are absent. Current six generators include only one base model (GPT-2, MPT-7B) vs four instruction-tuned. **Even one within-family pair (e.g., Mistral-base vs Mistral-Chat) as an appendix robustness check would substantially strengthen the bipartite claim.**

---

## Submission Recommendation

**Submit to EMNLP 2026 ARR May 25.**

| Outcome | Likelihood |
|---------|-----------|
| Main with revisions | Most likely if critical issues fixed + ≥1 qualitative example added |
| Findings | Likely default if submitted as-is |
| Reject | Only if reviewer fixates on absolute flip-rate magnitudes or n=2 base-model count |

**Priority order for two-week polish:**
1. Table 3 transparency (C1)
2. Enkhbayar → Antverg & Belinkov replacement (I1)
3. Dalvi/Elhage citations in §8 (I2)
4. At least one qualitative example (I5)
5. RAID domain justification (C3)
6. Flip-rate denominator framing (C4)
7. Remove script reference (C2)
