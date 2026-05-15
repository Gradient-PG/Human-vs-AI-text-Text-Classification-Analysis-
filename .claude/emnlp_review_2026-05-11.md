# EMNLP 2026 Submission Review — Personal Claude Feedback
*Generated: 2026-05-11 | Last updated: 2026-05-14*

## Overall Verdict
The paper is in genuinely good shape. Every Tier-1 novelty-audit recommendation was delivered: six generators, proper LOGO experiment, Borile & Abrate addressed prominently, stability-driven hyperparameter selection in appendix, mean-ablation caveat acknowledged. Structure matches the scaffold. "Easy to inject, hard to erase" framing reads cleanly. The base/instruction-tuned bipartite story is memorable. Major language cleanup done 2026-05-14.

**Ready for EMNLP 2026? Yes, with targeted fixes before May 25.**
- Ceiling: acceptance with revisions
- Floor: Findings
- No fatal flaw

---

## Critical Issues (fix before submission)

### C1 — Table 3 random-k baseline column ✅ RESOLVED
Table now has Selected (%), Random-k (%), and Ratio columns for all 6 generators including MPT. 10–16× range clearly stated. Previous concern about hidden high-variance case is no longer applicable.

### C2 — Remove `exp_characterize.py` reference in §6.2 ✅ RESOLVED
Line 668 now reads "minimal majority threshold" with no script reference.

### C3 — Justify six-of-eight RAID domain choice in §3.1 ✅ RESOLVED
Line 295 explains exclusion of poetry and recipes (low sample counts, risk of genre artefacts).

### C4 — Frame flip-rate denominator in §5.1.2 or intro ❌ STILL MISSING
1.07–8.15% still looks tiny without context. Add one sentence: *"Note that the denominator is human samples already correctly classified by the frozen evaluator; these flips represent predictions reversed despite the probe's high baseline confidence."*

---

## Important Issues (strongly recommended)

### I1 — Soften/replace Enkhbayar (2025) in load-bearing intro role ❌ STILL PRESENT
Line 134 still uses Enkhbayar as key citation for "causality gap." Options:
- (a) Soften: "Recent work [Enkhbayar 2025; preprint] *suggests*…"
- (b) **Replace with Antverg & Belinkov (ICLR 2022, "On the Pitfalls of Analyzing Individual Neurons")** — same encoded-vs-used point, peer-reviewed. Recommended.

### I2 — Add redundancy literature to §8 ❌ STILL MISSING
"Redundantly distributed" discussion still lacks canonical citations:
- **Dalvi et al. (EMNLP 2020)** — "Analyzing Redundancy in Pretrained Transformer Models"
- **Elhage et al. (2022)** — "Toy Models of Superposition"

### I3 — One-line detector comparison in §3 or §5 ❌ STILL MISSING
Add a sentence citing Dugan et al.'s numbers or Ghostbuster/Binoculars accuracy on this generator set. Defuses "is this competitive?" from reviewers.

### I4 — §5.3 Restricted Probe ✅ LARGELY RESOLVED
The restricted probe analysis is now in Appendix (app:restricted), with a one-sentence pointer in the main text. Acceptable.

### I5 — Qualitative examples ❌ STILL MISSING (biggest gap)
No qualitative examples anywhere. Appendix with one example per direction (human text that flipped; the AI donor; which neuron changed most) would significantly strengthen reviewer confidence.

---

## Minor Issues

### M1 — Abstract opening is generic ❌ OPEN
Still: "achieve high accuracy on standard benchmarks, yet the internal representations remain poorly understood." Consider leading with the finding.

### M2 — CAV §4.2 linkage ✅ RESOLVED
Line 394 links CAV to cross-generator differences; line 1197 ends with "align with the intervention asymmetries documented in §\ref{sec:results-causal}."

### M3 — Verify Pudasaini et al. (2026, arXiv:2603.23146) ❓ UNVERIFIED
March 2026 preprint — confirm it exists and author list is correct before submission.

### M4 — Hedge third contribution in §1 ❌ OPEN
Still reads as a definitive claim. Consider: *"a cross-generator circuit analysis revealing a tentative layer-12 footprint of post-training alignment, supported by two independent base models."*

### M5 — Split conclusion into 2–3 paragraphs ❓ UNVERIFIED
Not checked since last review — verify conclusion section is split.

### M6 — §6.1 Geva et al. speculation ❌ OPEN (low priority)
FFN-as-key-value-memory reading for the layer-12 effect is appropriately hedged. Trim if space is needed.

### M7 — Table 4 statistical insignificance ✅ RESOLVED
Line 529: "within one standard deviation of zero" added for GPT-4, GPT-2, MPT, Mistral.

### M8 — Standardize preprint vs. venue citation format ❓ UNVERIFIED
Run `rebiber` to check consistency.

---

## Conciseness Issues

| Location | Problem | Status |
|----------|---------|--------|
| §5.2 "There is no contradiction…" paragraph | Redundancy argument repeated. | Partially fixed (2026-05-14) |
| §6.2 bipartite Jaccard + core-neurons | Redundant with main Jaccard discussion. | Open |
| §8 Discussion Cohere story | Already told in §4.2, §5.1.2, §5.2. | Open |

---

## Things Discussed But Not In Paper

### Missing1 — Zero-ablation as sharper necessity test ❌ OPEN
Still flagged as future work in §10. If runnable before May 25, it would substantially strengthen the necessity claim.

### Missing2 — Base/chat within-family comparison ❌ OPEN
Still absent. Even one pair (e.g., Mistral-base vs Mistral-Chat) in an appendix would strengthen the bipartite claim.

---

## Submission Recommendation

**Submit to EMNLP 2026 ARR May 25.** Must declare EMNLP 2026 as preferred venue at submission time — this is **binding** for the May ARR cycle.

| Outcome | Likelihood |
|---------|-----------|
| Main with revisions | Most likely if C4 + I1 + I2 fixed + ≥1 qualitative example added |
| Findings | Likely default if submitted as-is |
| Reject | Only if reviewer fixates on absolute flip-rate magnitudes or n=2 base-model count |

**Priority order for remaining time:**
1. Flip-rate denominator framing (C4) — one sentence
2. Enkhbayar → Antverg & Belinkov replacement (I1) — one citation swap
3. Dalvi/Elhage citations in §8 (I2) — two citations added
4. At least one qualitative example (I5) — appendix
5. English-only limitation in §10 (see B1 below)
6. Computational budget sentence in Appendix A (see B2 below)

---

# Page Length & Formal Submission Checklist
*Updated: 2026-05-14*

## Page Length
**Hard 8-page limit on §1–§9. Desk-reject if exceeded.**
- Limitations, Ethical Considerations, References, Appendices do NOT count.
- Pipeline figure moved to Appendix A (saves ~0.7 pp) ✅
- CAV diagnostics moved to Appendix ✅
- Restricted probe moved to Appendix ✅
- **Current estimate: ~8 pages or just under — verify by compiling with `[review]` flag.**
- Accepted papers get 9 pages (camera-ready only).

---

## A. Desk-Reject Risks

| # | Item | Status |
|---|------|--------|
| 1 | Body ≤8 pages (§1–§9) | ❓ Verify after compile |
| 2 | Section titled exactly "Ethical considerations" | ✅ Line 840: `\section*{Ethical Considerations}` — check capitalisation matches ARR exactly |
| 3 | No Acknowledgments section header for review | ✅ Not found in file |
| 4 | Compile with `[review]` flag — line numbers in margins | ❓ Verify |
| 5 | Run `aclpubcheck` on non-review version | ❌ Not done yet |
| 6 | No `exp_characterize.py` reference or `\todo{}` comments | ✅ Script reference gone; check for any stray `\todo{}` |
| 7 | Limitations section has no new content | ✅ §10 looks compliant |
| 8 | Fill Responsible NLP Checklist on OpenReview form | ❌ Not yet |
| 9 | Declare EMNLP 2026 as preferred venue at ARR submission | ❌ Must do at submission — **binding** |

---

## B. Required Content Additions

| # | Item | Status |
|---|------|--------|
| 1 | English-only limitation in §10 | ❌ MISSING — add: "Results are limited to English RAID data; the layer-12 footprint may not generalise to morphologically richer languages." |
| 2 | Computational budget in Appendix A | ❌ MISSING — add: BERT-base-uncased ~110M params, GPU hours, hardware used |
| 3 | RAID license/terms of use (MIT) | ❌ MISSING — one sentence in §3.1 or Appendix A |
| 4 | BERT-base-uncased params and license (Apache 2.0) | ❌ MISSING — Appendix A |
| 5 | AI assistant disclosure on submission form | ❓ Mark explicitly on OpenReview checklist (E1) |

---

## C. Formatting Compliance

| Item | Status |
|------|--------|
| DOIs/ACL Anthology links in references | ❌ Run `rebiber` to auto-populate |
| Hyperlinks dark blue (#000099), not boxed | ❓ Verify |
| Figures readable in grayscale | ❓ Figure 2 (red/blue lines) — check grayscale distinguishability |
| Appendices double-column | ❓ Verify (April 2025 requirement) |
| No links to non-anonymous repos | ✅ Ethics section says "We release all experiment code" without a link — add anonymous link before submission |

---

## D. Responsible NLP Checklist (OpenReview form)

| Item | Answer | Notes |
|------|--------|-------|
| A1 Limitations | Yes | §10 |
| A2 Risks | Yes | Ethical Considerations |
| B1 Cite creators | Yes | §3.1 (RAID), §3.2 (BERT) |
| B2 License/terms | **Add** | Missing — one sentence needed |
| B3 Intended use | Yes | Ethical Considerations |
| B4 PII/offensive content | Yes | State RAID is pre-screened academic benchmark; no PII |
| B5 Documentation | Yes | §3.1; add English-language note |
| B6 Statistics | Yes | §3.3 (15 cells), Appendix C |
| C1 Parameters/budget/infra | **Add** | Currently missing — one sentence in Appendix A |
| C2 Hyperparameters | Yes | §3.4, Appendix A, Appendix B |
| C3 Descriptive statistics | Yes | All tables report mean ± std |
| C4 Existing packages | Yes | Appendix A (HuggingFace Transformers) |
| D1–D5 Human annotators | N/A | No human annotation |
| E1 AI assistants | Yes/No | Disclose explicitly — AI writing assistance is permitted; entirely AI-generated papers risk desk rejection |

---

## E. Administrative Deadlines

| Task | Deadline | Notes |
|------|----------|-------|
| Both authors complete OpenReview profiles | Several days before May 25 | Must include Semantic Scholar/DBLP/ACL Anthology links |
| Submit paper to ARR | **May 25 (AoE)** | Select EMNLP 2026 as preferred venue — **binding** |
| Both authors complete author registration form | **May 27 EoD AoE** | Failure = desk rejection. Distinct from reviewer assignment. |
| Complete assigned reviews | July 6 | If assigned |
| EMNLP commitment deadline | August 2, 2026 | After receiving ARR reviews/meta-review |
| Notification of acceptance | August 20, 2026 | |
| Camera-ready due | September 20, 2026 | Gets +1 page (9 pages) |
| Preprint status decision | At submission | "We do not intend to release" vs "We are considering" — no anonymity period required |

---

## F. Suggested Order of Operations

| Dates | Task |
|-------|------|
| May 14–17 | **Fix C4, I1, I2** (denominator framing, Enkhbayar swap, Dalvi/Elhage). Add English-only + compute budget + license sentences. |
| May 17–19 | Qualitative example in appendix (I5). Check page count by compiling with `[review]`. Make cuts if needed. |
| May 19–21 | Run `rebiber` for DOIs. Run `aclpubcheck`. Fix any font/margin errors. |
| May 21–23 | Both authors complete OpenReview profiles. Fill Responsible NLP Checklist. Verify reviewer eligibility. |
| May 23–24 | Final PDF check: no comments, anonymized, line numbers visible, no non-anonymous links. Add anonymous code repo link. |
| **May 25** | **Submit to ARR (AoE). Select EMNLP 2026 as preferred venue.** |
| May 27 | Both authors complete author registration form. |
