# Patterns from Accepted EMNLP/ACL Mechanistic Interpretability Papers
## For Use as LLM Prompting Material When Writing Your Paper

---

## 1. TITLE PATTERNS

### Formula: [Action Verb / Gerund] + [Specific Mechanism/Object] + [in/of/for] + [Model Family] + [Optional: via/through/using Method]

**Accepted Title Examples:**
- "Neuron-Level Knowledge Attribution in Large Language Models" (EMNLP 2024)
- "Mechanistic Understanding and Mitigation of Language Model Non-Factual Hallucinations" (Findings EMNLP 2024)
- "Interpreting Arithmetic Mechanism in Large Language Models through Comparative Neuron Analysis" (EMNLP 2024)
- "A Mechanistic Analysis of a Transformer Trained on a Symbolic Multi-Step Reasoning Task" (Findings ACL 2024)
- "Interpretability Analysis of Arithmetic In-Context Learning in Large Language Models" (EMNLP 2025)
- "Mechanistic Understanding and Mitigation of Language Confusion in English-Centric Large Language Models" (Findings EMNLP 2025)
- "Interpretability-based Tailored Knowledge Editing in Transformers" (EMNLP 2024)
- "Understanding How Value Neurons Shape the Generation of Specified Values in LLMs" (Findings EMNLP 2025)
- "Unlocking the Future: Exploring Look-Ahead Planning Mechanistic Interpretability in Large Language Models" (EMNLP 2024)

### Key Title Verbs (by frequency):
- Understanding, Interpreting, Analyzing, Exploring, Unveiling
- Mechanistic (as adjective): "Mechanistic Understanding", "Mechanistic Analysis", "Mechanistic Interpretation"
- Discovering, Tracing, Locating, Identifying, Attributing

### Title Templates for Prompting:
```
"[Gerund: Understanding/Interpreting/Analyzing] [Phenomenon] in [Model] [Optional: through/via Method]"
"A Mechanistic [Analysis/Interpretation/Study] of [Phenomenon] in [Model/Architecture]"  
"[Phenomenon]: [Explanatory Subtitle about Method or Finding]"
"Neuron-Level [Action] for [Task/Application] in [Model Family]"
```

---

## 2. ABSTRACT STRUCTURE (6-SENTENCE PATTERN)

### Sentence-by-Sentence Template:

**S1 — Context/Motivation (The Problem):**
- "State-of-the-art language models (LMs) sometimes generate [outputs] that [problem description]."
- "[Models] demonstrate impressive performance on [task], yet the internal mechanisms driving these capabilities remain poorly understood."
- "Identifying [specific component] for [specific purpose] is essential for understanding the mechanisms of [models]."

**S2 — Gap (What's Missing):**
- "However, existing studies do not provide insights into the internal mechanisms driving the observed capabilities."
- "Due to [constraint], current [methods] struggle to operate at [granularity level]."
- "While recent studies have made significant progress in [area], it is still hard to [specific challenge] due to [reasons]."

**S3 — This Paper (Our Contribution):**
- "To improve our understanding of [mechanism], we present a comprehensive mechanistic analysis of [model] trained on [task/data]."
- "In this paper, we propose a [method type] for [specific goal]."
- "We create diagnostic datasets with [description] and adapt interpretability methods to trace [phenomenon] through internal model representations."

**S4 — Key Finding(s):**
- "We identify a set of interpretable mechanisms the model uses to solve the task, and validate our findings using correlational and causal evidence."
- "We discover two general and distinct mechanistic causes of [phenomenon] shared across [models]: 1) [cause 1], and 2) [cause 2]."
- "Our results suggest that [model] implements a [mechanism type] that operates [description] and stores intermediate results in [location]."

**S5 — Validation/Application:**
- "Based on insights from our mechanistic analysis, we propose a novel [method] through targeted [intervention], demonstrating superior performance compared to baselines."
- "We validate our findings using correlational and causal evidence."
- "Compared to [N] other methods, our approach demonstrates superior performance across [N] metrics."

**S6 — Impact/Future (Optional):**
- "We anticipate that the motifs we identified can provide valuable insights into the broader operating principles of transformers."
- "Our method and analysis are helpful for understanding the mechanisms of [phenomenon] and set the stage for future research in [area]."

---

## 3. INTRODUCTION RHETORICAL MOVES

### Move 1: Establish Territory (1-2 paragraphs)
**Phrases:**
- "Transformer-based large language models (LLMs) possess remarkable capabilities for [task]..."
- "[Task/Capability] is important for downstream tasks including [list]..."
- "While recent studies have made significant progress in understanding [area]..."

### Move 2: Identify the Gap (1 paragraph)
**Phrases:**
- "However, these studies do not provide insights into..."
- "It is still hard to [goal] due to several reasons."
- "Existing methods typically concentrate on [component], often lacking evaluation of [other component]."
- "The internal mechanisms driving [capability] remain poorly understood."
- "To evaluate the degree to which these abilities are a result of actual [mechanism], existing work has focused on [approach]. However, [limitation]."

### Move 3: Present Your Work (1-2 paragraphs)
**Phrases:**
- "In this paper, we focus on [level]-level [methods/analysis]."
- "We analyze [specific aspect] and discover that [finding preview]."
- "Based on this finding, we employ [method] as [metric/tool]."
- "To improve our understanding of the internal mechanisms of [model], we present..."

### Move 4: Enumerate Contributions (bulleted/numbered list)
**Pattern — always 3-4 items:**
- "Overall, our contributions are as follows:"
  - "a) We design/propose a [method/framework] for [goal]. Compared with [baselines], our method achieves [result]."
  - "b) We design/propose a [complementary method] to [secondary goal]."
  - "c) We analyze/provide [analysis type] of [scope]. Our analysis is helpful for understanding [broader goal]."

---

## 4. METHODOLOGY SECTION PATTERNS

### Section Naming Convention:
- "3 Methodology" or "3 Method"
  - "3.1 Background" or "3.1 Preliminaries"
  - "3.2 [Core Formalization]" (e.g., "Distribution Change Caused by Neurons")
  - "3.3 [Proposed Method Name]" (e.g., "Value Neuron Identification")
  - "3.4 [Extension/Secondary Method]"

### Mathematical Formalization Phrases:
- "Given an input sentence X = [t₁, t₂, ..., tₜ] with T tokens, the model generates..."
- "The layer output h^l_i (position i, layer l) is the sum of..."
- "Specifically, [component] is computed by [operation]..."
- "We define [concept] as..."
- "Following [citation], we define..."

### Method Justification Phrases:
- "Due to computational constraints, current [methods] struggle to operate at [level]."
- "Compared to [type] methods, our approach requires only [efficiency claim]."
- "This finding motivates our use of [method] for [purpose]."

---

## 5. EXPERIMENTAL SETUP PATTERNS

### Dataset Description:
- "We evaluate on [N] types of [data/tasks]..."
- "We construct diagnostic datasets with [properties]..."
- "Our benchmark covers [N] [generators/tasks/domains]..."
- "Following [citation], we use [dataset] which contains [statistics]."

### Model Selection:
- "We conduct experiments on [Model1], [Model2], and [Model3]."
- "We validate across multiple model families: [list] to ensure generalizability."
- "We focus on [Model] due to [justification], and verify findings on [other models]."

### Baselines/Comparisons:
- "Compared with [N] other [static/dynamic] methods, our approach demonstrates..."
- "We compare against: (1) [method1] (Citation), (2) [method2] (Citation), ..."
- "We adopt [method] as our primary baseline because [reason]."

### Metrics:
- "We evaluate using [N] metrics: [metric1], [metric2], and [metric3]."
- "We report [primary metric] as the main evaluation criterion and additionally measure [secondary metrics]."

---

## 6. RESULTS & ANALYSIS PATTERNS

### Finding Presentation:
- "Our results suggest that [model] implements a [mechanism] that [description]."
- "We observe that [finding], which aligns with [citation]."
- "Notably, [surprising finding], contrary to [expectation/prior work]."
- "Both [component A] and [component B] can [function], and all important [units] are in [location]."

### Causal Validation Phrases:
- "We validate our findings using correlational and causal evidence."
- "To verify the causal role of [component], we perform [ablation/patching] experiments."
- "Intervening on [N] [neurons/heads] significantly [affects/changes] the [outcome]."
- "We provide both correlational (Section X) and causal (Section Y) evidence for our claims."

### Quantitative Claims:
- "While numerous [units] contribute to [outcome], intervening on a few [units] (N) can significantly influence [result]."
- "Our method achieves [X]% improvement over the strongest baseline on [metric]."
- "Removing [component] leads to a [X] point drop in [metric], confirming its causal role."

---

## 7. DISCUSSION/ANALYSIS PATTERNS

### Insight Framing:
- "This observation aligns with [citation] which found that..."
- "Our findings suggest that [broader implication]."
- "These results provide new insights into the internal dynamics of [models]."
- "[Component] is mainly activated by [other component], while [other component] is mainly activated by [yet another]."

### Connecting to Prior Work:
- "This is consistent with theoretical insights from [citation] which highlights..."
- "Similar works studied how models perform [task] using [mechanism]."
- "Our findings complement and extend those of [citation] by..."

---

## 8. LIMITATIONS SECTION (MANDATORY)

### Structure: 3-5 bullet points or short paragraphs
**Common limitation types for MI papers:**
- Scope of models examined: "We focused on [model family]; extending to [other architectures] is future work."
- Generalizability of findings: "Our experiments were conducted on [specific task/dataset]; the extent to which these mechanisms generalize to [broader setting] requires further investigation."
- Computational constraints: "Due to computational requirements of [method], we limited our analysis to [scope]."
- Causal vs. correlational: "While we provide causal evidence through [method], the complexity of [model] means our analysis may not capture all relevant mechanisms."
- Synthetic vs. natural data: "Our analysis uses [synthetic/controlled] data; real-world [data] may involve additional complexities."

---

## 9. KEY VOCABULARY & PHRASES (Mechanistic Interpretability Domain)

### Core Technical Terms:
- Circuit, circuit discovery, computational graph
- Activation patching, causal tracing, causal mediation analysis
- Residual stream, attention heads, feed-forward layers (FFN/MLP)
- Neuron attribution, feature attribution, knowledge localization
- Logit lens, probing, vocabulary projection
- Sparse autoencoders (SAEs), transcoders
- Superposition, polysemanticity, monosemanticity
- Information flow, residual contribution

### Hedging Phrases (critical for scientific writing):
- "Our results suggest that..."
- "This observation is consistent with..."
- "We hypothesize that..."
- "This provides evidence that..."
- "To the best of our knowledge, this is the first..."
- "While our findings indicate [X], further investigation is needed to..."

### Transition Phrases:
- "To further validate this observation, we..."
- "Building on these findings, we next examine..."
- "Having established [X], we now turn to..."
- "Taken together, these results indicate..."
- "We next investigate whether [hypothesis]."

---

## 10. PAPER STRUCTURE TEMPLATE (Long Paper, 8 pages + refs)

```
1. Introduction (~1.5 pages)
   - Context & motivation (2-3 paragraphs)
   - Gap identification (1 paragraph)
   - This paper's approach (1-2 paragraphs)  
   - Contributions list (3-4 items)

2. Related Work (~1 page)
   - 2.1 [Broader area] (e.g., Attribution Methods for Transformers)
   - 2.2 [Specific subfield] (e.g., Mechanistic Interpretability)
   - 2.3 [Application domain] (Optional, e.g., Knowledge Editing)

3. Methodology (~2 pages)
   - 3.1 Background / Preliminaries
   - 3.2 [Problem Formalization]
   - 3.3 [Proposed Method]
   - 3.4 [Extension / Secondary Method]

4. Experimental Setup (~0.5 pages)
   - 4.1 Data / Tasks
   - 4.2 Models
   - 4.3 Baselines & Metrics

5. Results & Analysis (~2 pages)
   - 5.1 [Main Result / RQ1]
   - 5.2 [Secondary Result / RQ2]
   - 5.3 [Analysis / Ablation / RQ3]
   - 5.4 [Case Studies / Qualitative Analysis]

6. Discussion (~0.5 pages, sometimes merged with 5)

7. Conclusion (~0.5 pages)

Limitations (mandatory, does not count toward page limit)
Ethics Statement (optional)
References (unlimited)
Appendix (optional, unlimited)
```

---

## 11. FIGURE & TABLE CONVENTIONS

### Figure Types (by frequency in MI papers):
1. **Architecture/Pipeline Diagram** (Figure 1, almost always): Shows the overall method/analysis pipeline
2. **Heatmaps**: Layer × head activation patterns, neuron importance across layers
3. **Line/Bar Charts**: Performance metrics across layers, ablation results
4. **UMAP/t-SNE/PCA Plots**: Representation space visualizations
5. **Case Study Figures**: Attention pattern visualizations, token-level attribution maps

### Table Types:
1. **Main Results Table**: Method comparison across metrics (bold = best, underline = second best)
2. **Ablation Table**: Component removal effects
3. **Dataset Statistics Table**: Corpus/benchmark properties
4. **Qualitative Examples Table**: Input → model behavior illustrations

### Caption Patterns:
- "Figure 1: Overview of our proposed [method]. [Brief explanation of what panels show]."
- "Table 1: [Metric] comparison across [methods/models]. Bold indicates best performance; underline indicates second best."
- "Figure N: [Specific finding visualization]. [Interpretation sentence]."

---

## 12. COMMON REVIEWER CONCERNS & PREEMPTIVE STRATEGIES

### "Not novel enough"
→ Preempt by clearly differentiating from prior work:
- "Unlike [citation] which focuses on [X], our work [specific difference]."
- "To the best of our knowledge, this is the first work to [specific claim]."

### "Limited to specific model/task"
→ Preempt with multi-model validation:
- "We validate across [N] model families to ensure generalizability."
- Address explicitly in Limitations.

### "Correlational, not causal"
→ Include both evidence types:
- "We provide both correlational (Section X) and causal (Section Y) evidence."
- Use activation patching, ablation studies, or causal mediation.

### "Claims too strong"
→ Use hedging language consistently:
- "Our results suggest..." not "Our results prove..."
- "This provides evidence for..." not "This demonstrates that..."

### "Missing baselines"
→ Compare against 5+ methods:
- Include both classic (gradient-based, attention weights) and recent methods.

---

## 13. PROMPTING TEMPLATES FOR TEXT GENERATION

### For generating an Introduction paragraph:
```
Write the opening paragraph of an EMNLP paper introduction about [topic]. 
Follow the "Establish Territory → Identify Gap → Present Work" structure.
Use phrases like "While recent studies have made significant progress in [X]..."
and "However, the internal mechanisms driving [capability] remain poorly understood."
Keep the tone formal, precise, and avoid overclaiming.
```

### For generating a Methods section:
```
Write a methodology section for a mechanistic interpretability paper that:
1. Starts with "Background/Preliminaries" formalizing the transformer architecture
2. Defines the specific analysis method mathematically  
3. Provides clear notation (h^l_i for layer output, etc.)
4. Explains the method step by step with equations
Use ACL style and format. Be precise and formal.
```

### For generating Results paragraphs:
```
Write a results paragraph reporting that [finding].
Use the pattern: "We observe that [finding] (Table/Figure N). 
This is consistent with [prior work expectation OR surprising because X].
To verify this, we [validation step], which confirms [claim]."
Use hedging language: "suggest", "indicate", "provide evidence for".
```

### For generating the Limitations section:
```
Write a Limitations section for an EMNLP paper on [topic].
Include 3-4 limitations covering:
1. Model scope (which models were/weren't tested)
2. Generalizability (tasks, domains, languages)  
3. Methodological caveats (correlational vs causal, approximations)
4. Computational constraints
Keep it honest but not self-deprecating. Frame limitations as future directions.
```

---

## 14. REFERENCES PATTERNS

### Must-cite foundational works for MI at ACL/EMNLP:
- Geva et al. (EMNLP 2021) — "Transformer Feed-Forward Layers Are Key-Value Memories"
- Meng et al. (NeurIPS 2022) — "Locating and Editing Factual Associations in GPT"
- Elhage et al. (2021) — "A Mathematical Framework for Transformer Circuits"
- Olsson et al. (2022) — "In-context Learning and Induction Heads"
- Wang et al. (ICLR 2023) — "Interpretability in the Wild: IOI in GPT-2 small"
- Conmy et al. (NeurIPS 2023) — "Towards Automated Circuit Discovery"
- Nanda et al. (ICLR 2023) — "Progress measures for grokking via mechanistic interpretability"
- Dai et al. (2021) — "Knowledge Neurons in Pretrained Transformers"
- Vig et al. (NeurIPS 2020) — "Causal Mediation Analysis for Interpreting Neural NLP"

### Reference density: ~40-70 references for a long paper is typical.
