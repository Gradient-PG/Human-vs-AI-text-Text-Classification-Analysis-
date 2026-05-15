# Language Review — Phrases to Simplify

Audit of `_paper/main.tex` (881 lines, post-May-15 cleanup), focused on phrasing
that reads as AI-generated: hedge stacks, dangling participles, invented
compound terms, and chained-clause sentences.

## Conventions

Each entry shows: **`Line N`** — exact quote (or relevant span) — short note on
what's wrong and a sketch of how to fix it.

**Recurring patterns to look for:**

- `, suggesting / indicating / providing / confirming` — dangling participle
  hooks where the implied subject is fuzzy
- `consistent with X` — hedge that often replaces a direct interpretive sentence
- `has a natural/mechanistic reading`, `the signature of`, `sits within`,
  `reflects an X encoding the Y signal at Z magnitude` — abstract-noun
  metaphors that read as synthetic wrap-ups
- run-on sentences chaining 3+ grammatically complete clauses with `:` and `---`
- invented hyphenated compounds presented as if they were standard terminology
  (`classification-register-like site`, `partial-redundancy reading`,
  `necessity-for-confounding`)

---

## Tier 0 — Cross-cutting AI tells (find-and-replace level)

These are global patterns, not single passages. Sweep the whole document for
each.

**Em-dashes (`---` in LaTeX → "—" in PDF) in body text.** LaTeX `---` produces
an em-dash, which is a recognisable LLM signature when used to splice clauses.
Five occurrences in body text:

| Line | Context |
|---|---|
| L224 | `bidirectional activation patching---transplanting AI activations...` |
| L285 | `redundantly distributed signal --- one of plausibly many sufficient subspaces` |
| L397 | `The joint pattern --- large flip rates alongside negligible ablation drops --- is the signature of...` (two em-dashes — also a parenthetical-pair pattern) |
| L399 | `Cohere sits within rather than against this reading --- its diffuse geometry...` |
| L541 | `selects fewer than 22 neurons at $N=7{,}500$ across all generators---too sparse for circuit analysis` |

Replace each with a colon, parentheses, or a sentence break. Colons work in
all five.

**"Consistent with" repetition.** Counted 7 occurrences in body + captions:
L242, L263, L266, L316 (twice in one sentence: "*consistent* L12-avoidance ...
*is consistent with*"), L335, L363, L397, L877. Vary the wording — "matches",
"in line with", "tracks", "follows from" — or replace with a direct
interpretive verb.

**`Cross-generator analysis reveals ...` used twice.** L131 (intro
findings-preview paragraph) and L399 (conclusion). Vary the second occurrence.

**Numbered prose enumerations `(i)/(ii)/(iii)/(iv)` and `(a)/(b)/(c)/(d)/(e)`
inside a single sentence.** Three instances, all very long single sentences
(see Tier 1: L70, L136, L150). This particular construction — full taxonomy
inlined into one paragraph-length sentence — is a recognisable LLM pattern.
Convert each to either short numbered sentences or a real list environment.

---

## Tier 1 — Critical (reads as LLM prose, obscures meaning)

These are the passages most likely to make a reviewer think the paper was
LLM-written. They stack hedges, abstract nouns, and dangling participles —
often all in one sentence.

**Line 285** — *Conservative-lower-bound + many-sufficient-subspaces*

> We treat the necessity result as a conservative lower bound: ablating $\leq$1\% of a 9{,}216-dimensional probe is intrinsically weak, and mean ablation moves activations off-manifold~\citep{heimersheim2024patching}; **the reverse patching experiment (§\ref{sec:patching-results}) provides an on-manifold necessity test confirming partial necessity. We therefore characterise the selected set not as \textit{the} detection circuit but as a causally sufficient, partially necessary projection of a redundantly distributed signal --- one of plausibly many sufficient subspaces, distinguished by being stably recoverable across folds, seeds, and (via LOGO) generator families.**

The bold span piles abstract nouns ("projection ... signal ... subspaces ...
distinguished by being recoverable"). Split into 2–3 short sentences with
verbs: "We do not claim the selected set is *the* detection circuit. It is one
sufficient subspace among others. What singles it out is stable recovery across
folds, seeds, and generator families."

**Line 316** — *Classification-register-like site + double `consistent`*

> This **consistent** L12-avoidance (not a rigid L11-peaking rule: GPT-2 peaks at L11 while MPT peaks at L1/L8) **is consistent with** BERT's final layer behaving as a **classification-register-like site that is preferentially engaged by instruction-tuned outputs and bypassed by base-model outputs**

Echoes "consistent" twice. The italicised phrase is an invented metaphor with
passive voice. Drop the metaphor or paraphrase concretely (e.g. "BERT's layer
12 acts as a task-linked readout that instruction-tuned outputs activate more
strongly than base-model outputs").

**Line 399** — *Cohere "sits within" + LOGO `providing evidence... while the residual gap leaves room`*

> Cross-generator analysis reveals a bipartite structure: instruction-tuned generators concentrate stable neurons in BERT's final layer while both pure-base generators do not, **suggesting** post-training alignment leaves a footprint in layer~12. **Cohere sits within rather than against this reading --- its diffuse geometry (...) reflects an instruction-tuned generator encoding the same layer-12 signal at lower per-neuron magnitude, which the probe compensates for with larger weights.** Leave-one-family-out evaluation confirms that neurons identified from training families retain 86--94\% of the full-feature ceiling on entirely held-out generator families, **providing evidence that the discovered circuits are not artefacts of the specific generators studied while the residual gap leaves room for family-specific neurons** beyond the cross-family core.

Three problems in one paragraph: `suggesting` dangles off "generators do not";
the Cohere sentence stacks "sits within ... reading ... reflects ... encoding
... magnitude"; trailing `providing evidence ... while the residual gap leaves
room for ...` dangles off "evaluation confirms". Split into separate sentences
with explicit subjects.

**Line 397** — *Conclusion long sentence with double hedge*

> The joint pattern --- large flip rates alongside negligible ablation drops --- **is the signature of a signal that is** \textit{easy to inject, hard to erase}: redundantly distributed across the full representation but concentrated enough to be causally sufficient, **consistent with the broader redundancy phenomenon documented in pretrained transformers**~\citep{dalvi2020redundancy}.

One long sentence with the slogan in the middle and a `consistent with` tail.
Cut after the slogan; integrate the citation in a short second sentence
("Redundancy of this kind is a known feature of pretrained transformers
~\citep{dalvi2020redundancy}").

**Line 363** — *"natural mechanistic reading" + `consistent with`*

> The 6--14\,pp residual gap between Sel\,L2 and Full\,L2 **has a natural mechanistic reading**: a cross-family core of neurons carries most of the detection signal that transfers, while the remainder of the full-feature ceiling is supplied by \textit{family-specific} neurons that the LOGO procedure cannot observe by construction. **This is consistent with** the bipartite Jaccard structure (...)

"Has a ... reading" nominalises intent. Say it directly: "We interpret the
6–14 pp gap as evidence that..." Then drop or replace the `consistent with`
hedge.

**Line 131** — *Three claims chained in one breath*

> Across six generators spanning two pure-base and four instruction-tuned families, bidirectional activation patching establishes that the selected neurons are both causally sufficient (forward patching flips human predictions to AI at 10--16$\times$ above a random-feature baseline) and partially necessary (reverse patching flips AI predictions to human at 7--12$\times$ above random), while mean-ablation of the same neurons leaves accuracy largely intact.

Reads as one breathless sentence. Break after the introductory phrase: state
the patching result in one sentence, the ablation contrast in a second.

**Line 70 (§1, related work taxonomy)** — *Single ~200-word sentence with `(a)–(e)` enumeration*

> Existing interpretability work on AI-text detection takes several forms: (a)~\textit{black-box feature attribution} via SHAP~\citep{...} or LIME~\citep{...}, which identifies predictive tokens without inspecting model internals~\cite{...}; (b)~\textit{confounder removal}, which suppresses neurons correlated with domain or style artefacts in fine-tuned encoders to improve out-of-distribution accuracy~\cite{...}; (c)~\textit{sparse autoencoder feature analysis} applied to decoder-based models~\cite{...}; (d)~\textit{probing on fine-tuned detectors}, which reveals reliance on narrow lexical or structural cues~\citep{...}; and (e)~\textit{topological analysis} of attention maps, characterising surface and syntactic properties that differentiate AI from human text~\citep{...}. These approaches share a fundamental limitation documented across neuron-level analysis broadly~\citep{...}: they are \textit{observational}.

The single worst readability issue in the paper. ~200 words, five subordinate
clauses, citations hanging at the end of each sub-clause. Rewrite as either an
`itemize` list (one item per line) or five separate sentences. The list form
also makes it scannable for reviewers who skim.

**Line 136 (Contributions paragraph)** — *Single sentence with `(i)/(ii)/(iii)/(iv)`*

> We make four contributions: (i)~a sparse-probing characterisation of AI-detection signal in frozen \bertbase across six RAID generators, identifying stable sets of $<$1\% of CLS dimensions sufficient for near-ceiling detection accuracy with high cross-fold stability; (ii)~the first application of bidirectional activation patching to AI-text detection neurons, establishing causal sufficiency in the forward direction (10--16$\times$ above random) and partial necessity in the reverse (7--12$\times$ above random); (iii)~a cross-generator circuit analysis that reveals generator-specific stable sets which nonetheless overlap ${\sim}24\times$ above chance, with a layer-12 footprint concentrated in instruction-tuned generators and absent in both pure-base models; and (iv)~a leave-one-family-out (LOGO) evaluation showing that the selected neurons retain 86--94\% of the full-feature detection signal on entirely unseen generator families.

Same pattern as L70 — the "list disguised as a sentence" construction is a
recognisable LLM template. Convert to four numbered sentences ("First, we...
Second, we...") or a real `enumerate` block. Each contribution should fit on
its own line.

**Line 150 (§2.2, four-axis Borile differentiation)** — *Same `four axes:` enumeration pattern*

> Our work differs from \citet{borile2025confounding} on four axes: we use a \textit{frozen} rather than fine-tuned encoder; we probe CLS-token representations rather than layer-specific FFN sublayer activations; we use ablation to measure \textit{necessity} and additionally apply activation patching to test causal \textit{sufficiency}, rather than ablating to identify and suppress confounders; and we evaluate on RAID rather than DAIGT, M4~\citep{wang2024m4}, or HC3~\citep{guo2023hc3}.

Long, semicolon-separated four-clause sentence doing important defensive work.
Convert to a numbered list — reviewers can then check each axis independently.

**Lines 61 (Abstract)** — *Two paragraph-length sentences carrying 6+ distinct claims each*

The abstract is one paragraph but contains two sentences that each carry too
much. Sentence 1 (`We study which neurons ...`) covers protocol, dimension
counts, stable-set sizes, full-probe accuracy, restricted-probe accuracy, and
reproducibility — six claims, joined by `;` and `,`. Sentence 2
(`To assess the causal role ...`) chains forward direction + reverse direction
+ ratios + necessity claim, also with multiple semicolons. Split each into 2–3
sentences. The semicolon at `1\% of neurons); a full-feature L2 probe...` is
doing too much work.

---

## Tier 2 — High (structural awkwardness, repeated AI tics)

Common LLM patterns: trailing `consistent with` hedges, `, suggesting/indicating`
dangling participles, baggy noun stacks. Each item is a quick fix.

**Line 224** — *Buried actor + stiff stack*

> A restricted-probe localisation analysis reinforcing the base/instruction-tuned split is provided in Appendix~\ref{app:restricted}.

"We provide a restricted-probe analysis in App. X; it reinforces ..."

**Line 236** — *Baseline that "provides" things*

> The \textit{random-$k$ baseline} draws 20 random subsets of size $k$ and runs the identical procedure, **providing a null** that controls for the number of patched dimensions.

Baselines do not "provide". → "This serves as a null controlling for the
number of patched dimensions."

**Line 242** — *Vague hedge*

> MPT falls at the lower end of the range (1.52\%, $\sim$11$\times$ above random), **consistent with its relatively diffuse cross-layer stable set** (§\ref{sec:crossgen-layers}).

Either drop the second clause or state the link directly: "MPT's lower flip
rate matches its broader layer spread (§...)".

**Line 263** — *Hedge + dangling negation*

> Generator-level differences track baseline accuracy: Cohere flips most readily (8.15\%, lowest baseline at 90.5\%), LLaMA least (1.07\%, baseline 98.6\%), **consistent with** proximity to the decision boundary. Flip rate grows sub-linearly with $k$ across all generators (Figure~\ref{fig:flip-rate}), **inconsistent with a concentration of effect in a few critical units**.

Both clauses use the same hedge construction. Say: "Generators nearer the
decision boundary flip more." And: "Sub-linear growth in $k$ argues against a
small set of critical units."

**Line 266** — *"Partial-redundancy reading" jargon label*

> Reverse flip rates are lower than forward (0.65--5.74\% vs.\ 1.07--8.15\%), **consistent with the partial-redundancy reading**: the remaining 9{,}000$+$ unpatched neurons carry enough redundant AI signal to partially preserve the prediction even when the selected set is overwritten.

The colon-clause already says it. Drop the label: "Reverse flip rates are
lower than forward: the remaining 9{,}000+ unpatched neurons carry enough..."

**Line 283** — *"All indicating" dangling tail*

> Cohere's other signals point the same way: it has the lowest baseline accuracy, smallest mean-difference norm, and largest probe weight norm, **all indicating** a detection representation that is \textit{partially} localised rather than fully distributed.

"All indicating" leaves a hanging participle. → "Together these point to a
representation that is partially localised rather than fully distributed."

**Line 328** — *Range "confirms" + `which indicates`*

> Pairwise Jaccard similarity between stable sets (...) ranges from 0.00 to 0.24, **confirming** generator-specific circuits. (...) the mean observed Jaccard of 0.073 is $\sim$24$\times$ above chance, **which indicates** a partial shared substrate despite the generator-specific structure.

A range cannot "confirm". Two interpretive hooks in one paragraph. Split: state
the overlap fact, then a new sentence ("Stable sets are therefore
generator-specific, but the mean overlap of 0.073 (~24× chance) indicates a
partial shared substrate.")

**Line 335** — *Double `consistent with` tail*

> Each generator's remaining stable neurons form a generator-specific peripheral set, **consistent with the bipartite Jaccard geometry of §X and with the residual Sel/Full gap observed in the LOGO evaluation** (§Y).

Two referents under one hedge. Pick one or rewrite: "These generator-specific
neurons mirror both the bipartite Jaccard structure (§X) and the residual
Sel/Full gap in the LOGO evaluation (§Y)."

**Line 150** — *Invented compound `necessity-for-confounding`*

> Both works operate on fine-tuned models and test **necessity-for-confounding**: the question is which neurons hurt generalisation, not which neurons are sufficient to drive predictions.

Drop the coined term: "Both works operate on fine-tuned models and ask which
neurons hurt OOD generalisation, not which neurons are sufficient to drive
predictions."

**Line 283 (§5.2, Cohere paragraph)** — *Subject repetition*

Four "Cohere..." subjects in close sequence:

> **Cohere** is the exception. Its full-set ablation drop is 1.06... unlike the other five generators where the curve is flat. **Cohere's** other signals point the same way: it has the lowest baseline accuracy...

Vary the second subject — "Its other signals point the same way:" works (and
removes the genitive duplication).

**Line 285 (§5.2)** — *Awkward parenthetical insert `(via LOGO)`*

> ...stably recoverable across folds, seeds, and **(via LOGO)** generator families.

The parenthetical interrupts the list. Either move the qualifier to a separate
sentence ("...stable across folds, seeds, and generator families. The LOGO
evaluation supplies the cross-family evidence.") or attach it to the noun:
"...generator families (in the LOGO evaluation)".

**Line 354 (§7, LOGO results)** — *"The key finding is in the X column"*

> The key finding is in the Sel\,L2 column: 91--125 stable neurons (...) achieve 76.3--93.7\% accuracy on generators from entirely held-out families.

`The key finding is in the X column` is mildly LLM-flavoured (hedged emphasis).
Direct alternative: "The Sel\,L2 column shows the main result: 91--125 stable
neurons..."

**Line 316 (§6.1)** — *Transition restates the previous paragraph's claim*

The §6.1 paragraph break is structurally weak. L314 introduces the asymmetry
("a clear asymmetry emerges between..."); L316 then re-opens with `Across both
pure-base generators, the shared pattern is markedly low layer-12
concentration (3.8--13.3\%) versus any instruction-tuned generator (30--36\%)`,
which restates the same fact before extending it. Either merge the two
paragraphs or compress the transition (e.g. start L316 with "This avoidance
pattern, rather than a fixed L11-peaking rule (GPT-2 at L11 vs MPT at L1/L8),
suggests...").

---

## Tier 3 — Polish (small word changes)

**Line 61 (abstract)** — Two AI tics in one paragraph

> ...both base generators fall below 14\%, **consistent with** a layer-12 footprint of post-training alignment. Leave-one-family-out evaluation shows that the selected neurons retain 86--94\% of the full-feature ceiling on entirely unseen generator families.

> Mean-ablating the same set leaves accuracy largely intact ($\leq$1.1\,pp drop on 5/6 generators), **suggesting** the signal is redundantly distributed.

`consistent with` and `suggesting` in the abstract. Replace with finite verbs.

**Line 193** — *Filler "this confirms"*

> A probe restricted to the stable set alone achieves 86.5--97.2\% (see Appendix~\ref{app:restricted}). **This confirms that the stable neurons are themselves sufficient for most of the detection accuracy.**

Throat-clearing. Drop or merge into the previous sentence.

**Line 193** — *Anthropomorphic "the probe consistently returns"*

> The probe **thus consistently returns to** a small region of the feature space.

Probes don't "return". → "The probe consistently selects from a small region
of the feature space."

**Line 271** — *Filler intensifier*

> Instruction-tuned generators (blue) and base generators (red) both **sit well above** the random baseline at every $k$.

The numbers carry the point. → "All generators exceed the random baseline at
every $k$."

**Line 314** — *"A clear asymmetry emerges"*

> A **clear asymmetry emerges** between the four instruction-tuned generators and the two pure-base generators.

State the asymmetry directly without the intensifier.

**Line 316** — *Throat-clearing "We stress that"*

> **We stress that** this reading rests on $N=2$ base generators (...)

Start with "Because this reading rests on $N=2$ base generators (...)".

**Line 330** — *"This suggests"*

> ...more than several instruction--instruction pairs. **This suggests** the base models form a coherent sub-cluster.

Merge: "...more than several IT–IT pairs, so the base models form a coherent
sub-cluster."

**Line 335** — *"Clearest descriptive evidence"*

> This concentration is **the clearest descriptive evidence** that BERT's layer-12 representations form the primary cross-generator site for AI-detection signal:

Superlative + hedge stacked. Drop one: "This concentration suggests BERT's
layer 12 is the primary cross-generator site for AI-detection signal."

**Line 341–344** — *Schematic "above ... stronger ... we address"*

> The experiments above characterise stable neurons for each generator \emph{independently}. A stronger test of the mechanistic claim is whether (...). We address this with a leave-one-family-out (LOGO) evaluation.

Standard LLM scaffolding. Tighten to one motive sentence.

**Line 361** — *Pre-empting interpretation*

> The worst-case family is \texttt{cohere} (...), **as expected** from its low pairwise Jaccard with other generators

"As expected" tells the reader what to feel. Drop or replace with an explicit
link: "matching its low pairwise Jaccard with other generators".

**Line 415** — *"We frame this as a consistent L12 footprint"*

Conversational rewrite: "We only claim lower L12 mass in base models; the
specific lower-layer peak is not consistent across base models."

**Line 427** — *Repeated "can equally / could equally"*

> ...detectors can support platform integrity (...) but **can equally** be used to penalise (...) but the same information **could equally** guide more robust detector design

Vary one verb to avoid the parallel boilerplate.

---

## Bugs / errors (fix definitely)

**Line 156** — Grammar: "and **a** L2 probe re-fitted on those neurons..." → should be "an L2".

**Line 541** — Counting bug: heading says "Three observations drive the choice" but four numbered points follow `(1) ... (2) ... (3) ... (4) Degenerate region:`. Fix the count or merge two items.

**Line 684** — Caption claim mismatches table: caption says "Three regimes are visible" across all generators, but Table 8 has no MPT row (only 5 generators are listed: GPT-4, GPT-2, Mistral, LLaMA, Cohere). Either add MPT or qualify caption ("five generators shown").

**Line 679** — Non-standard "zero-ing"; use "zeroing" or "setting ... to zero".

**Line 72 + Line 181** — Symbol $\mathcal{S}$ is overloaded. L181 defines it as the per-cell **candidate set**. L72 says "L1 ... selects a small stable set of candidate neurons" — collapsing the two stages. Use distinct symbols (e.g. $\mathcal{S}$ for per-cell candidate, $\mathcal{S}^{*}$ for the aggregated stable set, which the appendix already uses on L461).

**Line 303** — Caption/body mismatch: caption says "Sel.\ drop: ablating the **stable set**", but the body (L279–281) describes ablation of the per-fold selected $k=\text{full}$ set. These are not the same size. Align caption with the actual code path.

---

## Inconsistencies

**Line 41 (title)** — `AI-Detection` (hyphenated, capitalised) vs `AI-text` / `AI-generated` elsewhere. Pick one house style.

**Lines 61, 70, 167, etc. + Lines 181, 305, 444, etc.** — `9,216` (plain comma) vs `9{,}216` (thin-space comma) used inconsistently. Standardise on `9{,}216` everywhere in math/inline numbers.

**Line 352** — `88.6--99.7\%\ accuracy` — the `\%\ ` spacing pattern is unusual versus other `\%` uses (which omit the `\ `). Pick one.

**Line 406** — vague "**this variable**":

> ...results cannot be directly compared to fine-tuned detectors without controlling for **this variable**.

Name it: "...without controlling for the frozen-vs-fine-tuned distinction."

**Line 133** — ambiguous "**this gap**":

> We address **this gap** directly via activation patching, and discuss the relationship to this and other prior work in Section~\ref{sec:related}.

Reader doesn't know which gap (Borile's? observational vs causal?). Name it
("the lack of a sufficiency test for AI-detection neurons").

---

## Caption issues

**Line 127** — Pipeline figure caption packs definitions, both branches, and all key numbers in one block. Cut interpretive numbers; leave them in the main text.

**Line 781** — Interpretive `which suggests` inside a **table caption**:

> GPT-2 shows elevated flip rates on news, Reddit, and wiki (3.1--4.9\%) relative to books and reviews (1.9--2.0\%), **which suggests its detection signal is stronger in formal or community text domains.** Cohere is uniformly high across all six domains (5.5--10.1\%), **because** its probe operates near the decision boundary throughout.

Move mechanism talk to main text or soften ("see §X for interpretation").

**Line 877** — Appendix caption ends with `consistent with the partial redundancy interpretation in §...` — captions read cleaner as labels + pointers to interpretation, not interpretation themselves.

---

## Methods / structure

**Line 473** — Compute-budget paragraph runs as one chained sentence with nested parentheses across all sub-stages. Split into 2–3 short sentences for readability.

**Line 737 (CAV appendix)** — long catalogue paragraph chains five generator descriptions with `, indicating ... , meaning ... , placing ... , suggesting ...` participial tails. Either bullet the per-generator findings or split into 2–3 sentences each containing one participial hook at most.

---

## Summary of where the AI fingerprints cluster

The empirical sections (data, model, protocol, results tables) are mostly
clean. Almost all remaining LLM patterns cluster in **discussion-style
paragraphs and the framing/contributions block**:

- abstract (L61) — Tier 1 (sentence splitting)
- §1 related-work taxonomy (L70) — Tier 1 (worst readability issue)
- §1 contributions paragraph (L136) — Tier 1
- §2.2 four-axis Borile differentiation (L150) — Tier 1
- §5.2 mean-ablation discussion (L283–285) — Tier 1
- §6.1 layer-12 hypothesis (L316) — Tier 1
- §6.2 cross-generator interpretation (L328, L335) — Tier 2
- §7 LOGO interpretation (L361, L363) — Tier 1+2
- §8 conclusion (L397, L399) — Tier 1

## Recommended fix order

If you only do three passes:

1. **Tier 0 sweeps** — em-dashes → colons (5 fixes), vary `consistent with`
   wording, vary the second `Cross-generator analysis reveals`.
2. **Tier 1 enumeration sentences** — convert L70, L136, L150 from
   single-paragraph sentences into real `enumerate`/`itemize` lists or short
   numbered sentences.
3. **Tier 1 discussion passages** — L285, L316, L363, L397, L399 — split into
   2–3 short sentences each, drop the abstract-noun metaphors.

Everything else is polish. The empirical paragraphs need only the small Tier 3
polish plus the bug fixes.
