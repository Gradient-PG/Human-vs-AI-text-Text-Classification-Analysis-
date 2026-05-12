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

---

# Page Length & Formal Submission Checklist
*Added: 2026-05-11 (second review pass)*

## Page Length Problem — Most Urgent

**Hard 8-page limit on §1–§9 (intro through conclusion). Desk-reject if exceeded.**
- Limitations, Ethical Considerations, References, Appendices do NOT count.
- Current estimate: ~10 pages of body content → ~2 pages over.
- Accepted papers get 9 pages (camera-ready only, not submission).
- Source: ARR Reviewer Guidelines, EMNLP 2026 CFP, ACLPUB formatting guidelines.

### Cuts ranked by priority (~2 pages total needed)

| Cut | Save | Notes |
|-----|------|-------|
| Figure 1 (pipeline overview) | ~0.7 pp | Move to Appendix A or shrink to half-column inline. Prose in §3 already covers this. |
| §4.2 CAV Diagnostics + Table 2 → Appendix | ~0.5 pp | Cohere distinctness is established 4× elsewhere (§5.1.2, §5.2, §5.3, §6.1). Keep one sentence in §4.1 pointing to appendix. |
| §5.3 Restricted Probe + Table 5 → Appendix or compress | ~0.4 pp | Third proof of bipartite split. Move full analysis to appendix; keep one synthesis sentence in §6. |
| §6.2 Core Neurons subsection | ~0.2 pp | Compress to two sentences inline with Jaccard discussion. |
| §8 Discussion "Outliers" paragraph | ~0.3 pp | Already told in §4.2, §5.1.2, §5.2. Compress to synthesis sentence; use freed space for Dalvi/Elhage citations. |
| Drop Figure 5 (LOGO bar chart) | ~0.3 pp | Table 8 already gives Full L2 / Sel L2 / ratio. Bar chart adds no new information. |
| Compress Conclusion by ~1/3 | ~0.2 pp | Currently reiterates every finding. Keep ~8–10 sentences on: sufficiency, redundancy, bipartite structure, LOGO transfer. |
| **Optional:** merge §8 Discussion + §9 Conclusion | ~0.2 pp | Many MI papers do this; reads more naturally. |

Do **not** cut: §5.1 (headline patching protocol), §6.1 (layer distribution), §7 (LOGO transfer), Figure 2 (k-sweep + random envelope), Figure 4 (Jaccard heatmap).

---

## A. Desk-Reject Risks — Must Fix

1. **Cut body to ≤8 pages** (§1–§9). Currently ~10 pp. See cuts table above.
2. **Rename "Ethics Statement" → "Ethical considerations"** exactly. Automation counts it against page limit if title doesn't match.
3. **Remove Acknowledgments section header** for review submission. Add back for camera-ready.
4. **Compile with `[review]` flag** in ACL style file. Line numbers must appear in margins. Verify now.
5. **Run ACL pubcheck** (`pip install aclpubcheck`) on non-review version first, then re-enable `[review]`. Catches font/margin violations.
6. **Remove `exp_characterize.py` reference in §6.2** and all `\todo{}` comments / commented-out meta-text.
7. **Limits section must have no new content** — §10 looks compliant but verify.
8. **Fill Responsible NLP Checklist properly** on the OpenReview form. Systematic failures = desk rejection. See Section D below for per-item answers.

---

## B. Required Content Additions

1. **Add English-only limitation to §10.** Not just "evaluated on English RAID" but why: *"the layer-12 footprint may not generalize to languages with richer morphology."*
2. **Report computational budget in Appendix A or §3.** Checklist C1 requires: number of parameters (BERT-base-uncased = ~110M), total GPU hours, computing infrastructure. One sentence suffices.
3. **Discuss RAID license/terms of use** (MIT license) — one sentence in Appendix A or §3.1.
4. **Mention BERT-base-uncased parameters and license** (110M params, Apache 2.0) — Appendix A.
5. **AI assistant disclosure** on submission form (Checklist E1). Mark explicitly if AI was used for writing or coding.

---

## C. Formatting Compliance

- **DOIs/ACL Anthology links** required in references. Run `rebiber` (`pip install rebiber`) to auto-populate. Many current bib entries likely missing DOIs.
- **Hyperlinks**: dark blue (#000099), not underlined/boxed. ACL style default — just verify nothing overridden.
- **Figures in grayscale**: Figure 4 (Jaccard heatmap, blue colormap) — OK. Figure 2 (k-sweep, red/blue) — check if distinguishable in grayscale.
- **Appendices must be double-column** (April 2025 update). Verify A–F are double-column.
- **No links to non-anonymous repos**. If code repo is referenced, use anonymous.4open.science or equivalent.

---

## D. Responsible NLP Checklist (OpenReview form)

Provide section numbers for "Yes" answers; justifications for "No/NA":

| Item | Answer | Notes |
|------|--------|-------|
| A1 Limitations | Yes | §10 |
| A2 Risks | Yes | Ethical Considerations |
| B1 Cite creators | Yes | §3.1 (RAID), §3.2 (BERT) |
| B2 License/terms | Yes | Add to Appendix A or §3.1 |
| B3 Intended use | Yes | Ethical Considerations |
| B4 PII/offensive content | Yes | State RAID is pre-screened academic benchmark; no PII collected |
| B5 Documentation | Yes | §3.1; note English language explicitly |
| B6 Statistics | Yes | §3.3 (15 cells, 5 folds × 3 seeds), Appendix C |
| C1 Parameters/budget/infra | **Add this** | Currently missing — one sentence in Appendix A |
| C2 Hyperparameters | Yes | §3.4, Appendix A, Appendix B |
| C3 Descriptive statistics | Yes | All tables report mean ± std |
| C4 Existing packages | Yes | Appendix A (HuggingFace Transformers) |
| D1–D5 Human annotators | N/A | No human annotation |
| E1 AI assistants | Yes/No | Justify explicitly |

---

## E. Administrative Deadlines

| Task | Deadline | Notes |
|------|----------|-------|
| Both authors complete OpenReview profiles | Several days before May 25 | Must include Semantic Scholar/DBLP/ACL Anthology links |
| Submit paper | **May 25 (AoE)** | Don't submit at last minute — OpenReview congested |
| Both authors complete reviewer registration | Within 48h of May 25 | **Failure = desk rejection** |
| Verify reviewer eligibility | Before May 22 | ≥2 short/long papers in ACL main or Findings + ≥1 more in ACL Anthology or major ML/AI venue. If not eligible, claim exemption with justification. |
| Preprint status decision | At submission | Binding choice: "We do not intend to release" vs "We are considering" |

---

## F. Suggested Order of Operations

| Dates | Task |
|-------|------|
| May 11–14 | **Make the 8-page cuts.** Nothing else matters without this. |
| May 14–18 | Content: random baselines in Table 3, English-only limitation, compute budget, license mentions, Enkhbayar reframing, Dalvi citation, qualitative example |
| May 18–20 | Run `rebiber` for DOIs; remove `exp_characterize.py`; rename Ethics section; remove Acknowledgments header |
| May 20–22 | Compile without `[review]`, run `aclpubcheck`, fix errors. Re-enable `[review]`, recompile. |
| May 22–24 | Both authors complete OpenReview profiles. Fill Responsible NLP Checklist. Verify reviewer eligibility. |
| May 24 | Final PDF check: no comments, anonymized, line numbers present, no non-anonymous links |
| **May 25** | **Submit (AoE).** |
| Within 48h post-May 25 | Both authors complete reviewer registration form |
