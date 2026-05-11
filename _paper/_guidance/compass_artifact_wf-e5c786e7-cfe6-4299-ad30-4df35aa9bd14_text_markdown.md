# Annotated Bibliography for "A Mechanistic Study of AI-Detection Features in Frozen BERT: Sparse Probing and Activation Patching on RAID"

## TL;DR
- The strongest existing peg for your work is **Borile & Abrate (Findings of EMNLP 2025)**, which independently identifies "confounding neurons" in transformer detectors and validates that neuron-level interventions can disentangle generator/domain signal — your paper should cite it as the closest prior art and explicitly differentiate (frozen base BERT, no detector head, RAID's six generators, donor patching with quantified flip rate).
- The methodological backbone of your paper is well-supported by canonical work: **Gurnee et al. (TMLR 2024)** for L1 sparse probing, **Vig et al. (NeurIPS 2020) + Geiger et al. (NeurIPS 2021)** for causal mediation/interchange interventions, **Durrani et al. (EMNLP 2020) and Antverg & Belinkov (ICLR 2022)** for neuron ranking and probing pitfalls, and **Elhage et al. (Anthropic 2022) + Dalvi et al. (EMNLP 2020)** for the distributed/redundant-coding lens you need to explain low ablation impact.
- For each of the seven areas below, 3–5 high-citation, peer-reviewed papers are identified that directly support specific claims; they are mapped section-by-section to the paper's stages (probing → patching → ablation → Jaccard → leave-one-family-out).

---

## Key Findings

The seven request areas decompose into one tight cluster of "must-cite" papers (Borile & Abrate; Dugan et al. RAID; Gurnee "Haystack"; Vig; Geiger; Durrani; Elhage Superposition; Dalvi Redundancy; Gurnee "Universal Neurons") and a periphery of strong supporting work. Below, each paper is given full bibliographic details, a 2-sentence summary, and explicit mapping to your paper's sections/claims.

---

## AREA 1 — AI-generated text detection: interpretability and representation-based approaches

**1.1 Borile, Claudio; Abrate, Carlo. "How to Generalize the Detection of AI-Generated Text: Confounding Neurons."** Findings of the Association for Computational Linguistics: EMNLP 2025, Suzhou, China, November 2025. ACL Anthology: 2025.findings-emnlp.1388. ISBN 979-8-89176-335-7.
*Summary.* Introduces "confounding neurons" in transformer-based AI-text detectors — individual units that encode dataset/topic biases rather than the generation signal — and proposes a post-hoc neuron-level intervention that improves OOD generalization. Demonstrates that targeted neuron suppression reduces topic-specific confounds and boosts cross-domain detection.
*Supports.* This is the single most important related-work citation. Use it in §Related Work and again in §Discussion to position your contribution: you study unmodified frozen BERT-base (not a fine-tuned detector), use RAID's six generators, and provide causal sufficiency evidence via donor patching with a quantified prediction-flip rate against a random baseline — sharpening the "confounding vs. signal-bearing neuron" distinction.

**1.2 Dugan, Liam; Hwang, Alyssa; Trhlík, Filip; Zhu, Andrew; Ludan, Josh Magnus; Xu, Hainiu; Ippolito, Daphne; Callison-Burch, Chris. "RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors."** Proceedings of the 62nd Annual Meeting of the ACL (Volume 1: Long Papers), pp. 12463–12492, Bangkok, August 2024. ACL Anthology: 2024.acl-long.674. arXiv: 2405.07940.
*Summary.* Releases the RAID benchmark of 6M+ generations spanning 11 generators, 8 domains, 11 adversarial attacks and 4 decoding strategies, and shows that state-of-the-art detectors fail dramatically under generator/domain shift. Provides the empirical motivation for studying detector internals.
*Supports.* Cite as the primary dataset/benchmark in §Data and again in §Introduction to motivate why a representation-level explanation of cross-generator failure is needed; the paper's six-generator subset of RAID anchors your evaluation directly to this benchmark.

**1.3 Kuznetsov, Kristian; Tulchinskii, Eduard; Kushnareva, Laida; Magai, German; Barannikov, Serguei; Nikolenko, Sergey; Piontkovskaya, Irina. "Robust AI-Generated Text Detection by Restricted Embeddings."** Findings of the Association for Computational Linguistics: EMNLP 2024, Miami, FL, pp. 17036–17055, November 2024. ACL Anthology: 2024.findings-emnlp.992. arXiv: 2410.08113.
*Summary.* Analyzes the geometry of detector embedding spaces in RoBERTa/BERT and shows that head-wise and coordinate-based subspace-removal strategies (and LEACE-style concept erasure) clear "harmful" linear subspaces that encode generator- and domain-specific spurious features; in the authors' own words, this "increase[s] the mean out-of-distribution (OOD) classification score by up to 9% and 14% in particular setups for RoBERTa and BERT embeddings respectively." Frames cross-generator detector failure as a subspace contamination problem.
*Supports.* Best companion citation for §Related Work and for your Jaccard/cross-generator stability analysis: they manipulate detector subspaces; you identify the neurons whose joint activation is causally sufficient. Cite when arguing that representation-level analysis (yours) and representation-level surgery (theirs) are converging on the same conclusion.

**1.4 Chen, Xin; Wu, Junchao; Yang, Shu; Zhan, Runzhe; Wu, Zeyu; Luo, Ziyang; Wang, Di; Yang, Min; Chao, Lidia S.; Wong, Derek F. "RepreGuard: Detecting LLM-Generated Text by Revealing Hidden Representation Patterns."** Transactions of the Association for Computational Linguistics (TACL), Vol. 13, pp. 1812–1831, 2025. DOI: 10.1162/TACL.a.61. arXiv: 2508.13152.
*Summary.* Uses a surrogate LLM's internal activations to compute a single discriminative direction ("RepreScore") that separates LLM-generated from human text, achieving 94.92% average AUROC across in- and out-of-distribution generators. Argues that internal representations contain more transferable detection signal than surface features.
*Supports.* Directly supports your Introduction claim that internal representations encode an AI-text signal robustly across generators; contrast with your paper's finer-grained per-neuron causal evidence (vs. their single linear projection).

**1.5 Tulchinskii, Eduard; Kuznetsov, Kristian; Kushnareva, Laida; Cherniavskii, Daniil; Nikolenko, Sergey; Burnaev, Evgeny; Barannikov, Serguei; Piontkovskaya, Irina. "Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts."** Advances in Neural Information Processing Systems 36 (NeurIPS 2023). arXiv: 2306.04723.
*Summary.* Proposes the persistent-homology dimension (PHD) of contextual embeddings as a model-agnostic geometric invariant; the authors report that "the average intrinsic dimensionality of fluent texts in natural language is hovering around the value 9 for several alphabet-based languages and around 7 for Chinese, while the average intrinsic dimensionality of AI-generated texts for each language is ≈1.5 lower" (i.e., ~7.5 for alphabet-based languages, ~5.5 for Chinese). The detector is robust across generators, languages, and sampling strategies without supervised training.
*Supports.* Use in §Related Work to argue that geometric/representation-based features generalize better than surface-level supervised features — motivating your decision to interrogate frozen-BERT internals rather than fine-tune yet another detector.

---

## AREA 2 — Activation patching and causal intervention in encoder models

**2.1 Vig, Jesse; Gehrmann, Sebastian; Belinkov, Yonatan; Qian, Sharon; Nevo, Daniel; Singer, Yaron; Shieber, Stuart. "Investigating Gender Bias in Language Models Using Causal Mediation Analysis."** Advances in Neural Information Processing Systems 33 (NeurIPS 2020), Spotlight. arXiv: 2004.12265.
*Summary.* Introduces causal mediation analysis to transformer LMs, decomposing input→output effects through individual neurons and attention heads, and shows that gender-bias effects are sparse and concentrated. Establishes the methodological template of intervention-based localization for transformers.
*Supports.* Foundational citation for your activation-patching methodology section; cite when motivating donor patching as a causal-sufficiency test rather than a correlational probe.

**2.2 Geiger, Atticus; Lu, Hanson; Icard, Thomas; Potts, Christopher. "Causal Abstractions of Neural Networks."** NeurIPS 2021. arXiv: 2106.02997.
*Summary.* Formalizes interchange interventions on BERT-based NLI models and proves that aligning low-level neurons with high-level causal variables can verify causal sufficiency. Provides the theoretical basis for substituting activations between examples (donor patching) on encoders.
*Supports.* Cite in §Methods (activation patching) as the formal foundation for the same-domain donor patching protocol; your 10–16× flip rate above random baseline is exactly the kind of interchange-intervention accuracy this framework is designed to evaluate.

**2.3 Finlayson, Matthew; Mueller, Aaron; Gehrmann, Sebastian; Shieber, Stuart; Linzen, Tal; Belinkov, Yonatan. "Causal Analysis of Syntactic Agreement Mechanisms in Neural Language Models."** ACL-IJCNLP 2021. ACL Anthology: 2021.acl-long.144. arXiv: 2106.06087.
*Summary.* Applies causal mediation to subject-verb agreement in BERT/GPT, identifying neuron sets that mediate the effect and showing that similar syntactic structures recruit overlapping neuron populations. Establishes that interventions on individual neurons yield measurable, structure-specific behavioral changes.
*Supports.* Direct methodological precedent for your per-neuron causal experiments on an encoder; cite when motivating the prediction-flip rate as the dependent measure for sufficiency.

**2.4 Tucker, Mycal; Qian, Peng; Levy, Roger. "What if This Modified That? Syntactic Interventions via Counterfactual Embeddings."** Findings of ACL-IJCNLP 2021, pp. 862–875. ACL Anthology: 2021.findings-acl.76. arXiv: 2105.14002.
*Summary.* Generates counterfactual embeddings within BERT-based models to test whether probed syntactic structure is causally used in downstream prediction. Provides the prototype for counterfactual-embedding interventions on BERT.
*Supports.* Cite alongside Geiger et al. when motivating donor patching as a causal test on BERT specifically (as opposed to autoregressive LMs).

**2.5 Wang, Kevin Ro; Variengien, Alexandre; Conmy, Arthur; Shlegeris, Buck; Steinhardt, Jacob. "Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small."** ICLR 2023. arXiv: 2211.00593.
*Summary.* Reverse-engineers a 26-head circuit for IOI in GPT-2 using path patching and causal interventions, and validates it against faithfulness/completeness/minimality metrics. Defines the modern circuit-discovery protocol that path patching enables.
*Supports.* Cite when describing patching protocol and faithfulness metrics; useful even though it is autoregressive because the methodology (clean vs. corrupted distributions, flip-based metrics) is directly transferable.

**2.6 Conmy, Arthur; Mavor-Parker, Augustine N.; Lynch, Aengus; Heimersheim, Stefan; Garriga-Alonso, Adrià. "Towards Automated Circuit Discovery for Mechanistic Interpretability."** NeurIPS 2023. arXiv: 2304.14997.
*Summary.* Systematizes the activation-patching pipeline (choose metric → patch components → identify circuit) and automates the connection-discovery step. Surveys and formalizes the patching primitives you use.
*Supports.* Cite for the patching workflow in §Methods.

---

## AREA 3 — Sparse probing and neuron identification in pretrained LMs

**3.1 Gurnee, Wes; Nanda, Neel; Pauly, Matthew; Harvey, Katherine; Troitskii, Dmitrii; Bertsimas, Dimitris. "Finding Neurons in a Haystack: Case Studies with Sparse Probing."** Transactions on Machine Learning Research (TMLR), 2024. arXiv: 2305.01610.
*Summary.* Trains k-sparse linear classifiers on frozen LM activations and shows that early layers represent features in superposition while middle layers contain dedicated, monosemantic neurons for higher-level features; the authors "probe for over 100 unique features comprising 10 different categories in 7 different models spanning 70 million to 6.9 billion parameters." Provides the methodological template of L1-style sparse probing on frozen activations.
*Supports.* The canonical citation for your L1→L2 sparse probing pipeline and the 45–62 neuron set sizes — your design choices (L1 selection, then L2 refinement, on frozen activations) follow this paper. Cite in §Methods.

**3.2 Durrani, Nadir; Sajjad, Hassan; Dalvi, Fahim; Belinkov, Yonatan. "Analyzing Individual Neurons in Pre-trained Language Models."** EMNLP 2020, pp. 4865–4880. ACL Anthology: 2020.emnlp-main.395. arXiv: 2010.02695.
*Summary.* Uses elastic-net (L1+L2) probing classifiers to identify individual neurons in BERT/XLNet/ELMo/T-ELMo encoding morphology, syntax, and semantics, and quantifies how localized vs. distributed each property is. Establishes the L1-regularized probing protocol for pretrained encoders.
*Supports.* Primary precedent for your specific regularization choice (L1→L2); cite in §Methods to justify using elastic-net-style selection on frozen BERT.

**3.3 Antverg, Omer; Belinkov, Yonatan. "On the Pitfalls of Analyzing Individual Neurons in Language Models."** ICLR 2022. arXiv: 2110.07483.
*Summary.* Shows that conventional probe-based neuron rankings confound probe quality with ranking quality, and that "encoded" information is not necessarily "used"; introduces a causal-intervention-based ranking evaluation and a simpler "Probeless" method. Highlights why probing alone is insufficient.
*Supports.* Cite when justifying why your design pairs probing with causal patching — you explicitly address the encoded-vs-used distinction this paper raises.

**3.4 Dalvi, Fahim; Sajjad, Hassan; Durrani, Nadir. "NeuroX Library for Neuron Analysis of Deep NLP Models."** Proceedings of the 61st ACL (Volume 3: System Demonstrations), pp. 226–234, 2023. ACL Anthology: 2023.acl-demo.21. DOI: 10.18653/v1/2023.acl-demo.21.
*Summary.* Releases the NeuroX toolkit for extraction, probing, and intervention on individual neurons across deep NLP models. Implements Linguistic Correlation Analysis, Probeless, and several ranking methods under a unified API.
*Supports.* Cite if you use NeuroX-style analysis tooling, or in any case as a methods reference for neuron-level probing.

**3.5 Bau, Anthony; Belinkov, Yonatan; Sajjad, Hassan; Durrani, Nadir; Dalvi, Fahim; Glass, James. "Identifying and Controlling Important Neurons in Neural Machine Translation."** ICLR 2019. arXiv: 1811.01157.
*Summary.* Develops unsupervised correlation- and SVCCA-based methods to rank neurons in NMT models and shows that ablating top-ranked neurons degrades translation while modifying single neurons can steer output. Establishes correlation-based neuron ranking as a baseline for L1 probing.
*Supports.* Cite in §Methods as a baseline ranking method against which L1 selection is compared (motivating why L1 is preferable for stable selection).

---

## AREA 4 — RLHF and instruction-tuning effects on internal representations

**4.1 Lin, Bill Yuchen; Ravichander, Abhilasha; Lu, Ximing; Dziri, Nouha; Sclar, Melanie; Chandu, Khyathi; Bhagavatula, Chandra; Choi, Yejin. "The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning."** ICLR 2024. arXiv: 2312.01552.
*Summary.* Shows that alignment tuning shifts token distributions in a narrow, style-dominated way — base and aligned LLMs agree on most token positions — supporting the "superficial alignment" hypothesis. Provides token-distribution-shift evidence that instruction tuning leaves base representations largely intact.
*Supports.* Cite in §Discussion when arguing that the chat-tuned generators in RAID (e.g., the chat vs. base distinction across generators) differ from base LLMs mainly in stylistic tokens, which is consistent with your finding that frozen BERT can detect these distinctions via a small neuron set.

**4.2 Jain, Samyak; Lubana, Ekdeep Singh; Oksuz, Kemal; Joy, Tom; Torr, Philip H.S.; Sanyal, Amartya; Dokania, Puneet K. "What Makes and Breaks Safety Fine-tuning? A Mechanistic Study."** NeurIPS 2024. arXiv: 2407.10264.
*Summary.* Studies SFT, DPO, and unlearning mechanistically and provides evidence that safety fine-tuning makes only minimal MLP-weight changes that project unsafe inputs into the weights' null space. Demonstrates that fine-tuning effects on internal representations are localized and shallow.
*Supports.* Cite when arguing that the generator-specific signature your frozen BERT detects can plausibly be a shallow stylistic/surface feature induced by lightweight tuning rather than a deep semantic difference.

**4.3 Jain, Samyak; Kirk, Robert; Lubana, Ekdeep Singh; Dick, Robert P.; Tanaka, Hidenori; Grefenstette, Edward; Rocktäschel, Tim; Krueger, David Scott. "Mechanistically Analyzing the Effects of Fine-tuning on Procedurally Defined Tasks."** ICLR 2024. arXiv: 2311.12786.
*Summary.* Shows that fine-tuning rarely creates new model capabilities but instead learns a thin "wrapper" on top of pre-trained mechanisms. Provides mechanistic evidence for the conservation of base-model structure under fine-tuning.
*Supports.* Cite in §Discussion to motivate why frozen BERT is a sensible substrate: if fine-tuning is mostly wrapper-like, base-model representations already contain the relevant detection-discriminative features.

**4.4 Prakash, Nikhil; Shaham, Tamar Rott; Haklay, Tal; Belinkov, Yonatan; Bau, David. "Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking."** ICLR 2024. arXiv: 2402.14811.
*Summary.* Demonstrates via path patching and a new Cross-Model Activation Patching (CMAP) method that fine-tuning (here, math fine-tuning of Llama-7B) enhances rather than replaces base-model circuits. Introduces CMAP as a tool for comparing mechanisms across model variants.
*Supports.* Cite in §Discussion and in §Methods if you adopt CMAP-style cross-generator patching ideas; it is the canonical reference for the "fine-tuning enhances, doesn't replace" claim.

**4.5 Lee, Andrew; Bai, Xiaoyan; Pres, Itamar; Wattenberg, Martin; Kummerfeld, Jonathan K.; Mihalcea, Rada. "A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity."** ICML 2024. arXiv: 2401.01967.
*Summary.* Studies DPO's mechanism for reducing toxicity in GPT-2 medium and finds that DPO operates by dampening (rather than removing) toxicity-encoding neurons; the toxic representation persists post-alignment. Provides direct evidence that base-model features survive preference tuning.
*Supports.* Cite in §Discussion to support the claim that chat/instruction-tuned model outputs still carry base-model representational signatures that a frozen detector can pick up.

---

## AREA 5 — Cross-model and cross-generator transfer in NLP

**5.1 Olsson, Catherine; Elhage, Nelson; Nanda, Neel; et al. "In-context Learning and Induction Heads."** Anthropic Technical Report / arXiv 2209.11895, 2022.
*Summary.* Presents six lines of evidence that induction heads form universally across small transformer models and that this circuit underlies most in-context learning. Establishes the universality hypothesis empirically for one circuit class.
*Supports.* Cite in §Cross-Generator Analysis when discussing whether AI-detection features should be expected to be universal across models — and to frame your Jaccard overlap result as a test of partial universality at the neuron level.

**5.2 Chughtai, Bilal; Chan, Lawrence; Nanda, Neel. "A Toy Model of Universality: Reverse Engineering How Networks Learn Group Operations."** ICML 2023. arXiv: 2302.03025.
*Summary.* Tests the universality hypothesis on group-composition tasks and finds mixed evidence: the family of circuits is universal but specific instantiations are not. Provides a nuanced theoretical anchor for cross-model neuron overlap studies.
*Supports.* Cite alongside Olsson et al. to motivate why your Jaccard overlap is *partial* rather than total across generators (consistent with mixed-universality predictions).

**5.3 Gurnee, Wes; Horsley, Theo; Guo, Zifan Carl; Kheirkhah, Tara Rezaei; Sun, Qinyi; Hathaway, Will; Nanda, Neel; Bertsimas, Dimitris. "Universal Neurons in GPT2 Language Models."** TMLR 2024. arXiv: 2401.12181.
*Summary.* The authors "compute pairwise correlations of neuron activations over 100 million tokens for every neuron pair across five different seeds and find that 1–5% of neurons are universal, that is, pairs of neurons which consistently activate on the same inputs." Establishes a quantitative baseline for cross-model neuron overlap and shows that universal neurons typically have clear, interpretable functions.
*Supports.* The single most relevant universality citation for your Jaccard analysis — provides the 1–5% baseline against which your generator-overlap percentages should be reported and discussed.

**5.4 Li, Yafu; Li, Qintong; Cui, Leyang; Bi, Wei; Wang, Zhilin; Wang, Longyue; Yang, Linyi; Shi, Shuming; Zhang, Yue. "MAGE: Machine-generated Text Detection in the Wild."** ACL 2024 (Volume 1: Long Papers), pp. 36–53. ACL Anthology: 2024.acl-long.3. DOI: 10.18653/v1/2024.acl-long.3. arXiv: 2305.13242.
*Summary.* Constructs an "in-the-wild" benchmark with 7 writing tasks × 27 LLMs × 3 prompt types and shows that detectors degrade sharply OOD but recover with minimal in-domain data. Establishes leave-one-model-out and leave-one-domain-out as standard evaluation paradigms.
*Supports.* Cite in §Experiments to anchor your leave-one-family-out transfer evaluation in established practice for the MGT-detection field.

---

## AREA 6 — Distributed and redundant representations in transformers

**6.1 Elhage, Nelson; Hume, Tristan; Olsson, Catherine; Schiefer, Nicholas; Henighan, Tom; Kravec, Shauna; Hatfield-Dodds, Zac; Lasenby, Robert; Drain, Dawn; Chen, Carol; Grosse, Roger; McCandlish, Sam; Kaplan, Jared; Amodei, Dario; Wattenberg, Martin; Olah, Christopher. "Toy Models of Superposition."** Anthropic, September 2022. arXiv: 2209.10652.
*Summary.* Demonstrates that neural networks store more features than they have dimensions by representing them in superposition, producing polysemantic neurons; offers a phase-change theory tied to feature sparsity. Provides the canonical theoretical account of why information is distributed.
*Supports.* The core citation for your §Discussion explanation of why ablating 45–62 neurons has limited effect — the same signal is spread across many neurons in superposition.

**6.2 Dalvi, Fahim; Sajjad, Hassan; Durrani, Nadir; Belinkov, Yonatan. "Analyzing Redundancy in Pretrained Transformer Models."** EMNLP 2020, pp. 4908–4926. ACL Anthology: 2020.emnlp-main.398. DOI: 10.18653/v1/2020.emnlp-main.398. arXiv: 2004.04010.
*Summary.* Quantifies general and task-specific redundancy in BERT/XLNet; in the authors' words, "85% of the neurons across the network are redundant and ii) at least 92% of them can be removed when optimizing towards a downstream task." Provides direct empirical evidence for redundant encoding.
*Supports.* The strongest empirical citation for your "easy to inject, hard to erase" framing — cite when explaining the mean-ablation finding (low impact despite informative neurons).

**6.3 Ravfogel, Shauli; Elazar, Yanai; Gonen, Hila; Twiton, Michael; Goldberg, Yoav. "Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection."** ACL 2020, pp. 7237–7256. ACL Anthology: 2020.acl-main.647. arXiv: 2004.07667.
*Summary.* Introduces INLP, iteratively training linear classifiers and projecting onto their null spaces to erase a target attribute. Demonstrates that even after exhaustive linear erasure, target information can leak back through residual representations.
*Supports.* Cite when discussing why mean-ablation of identified neurons does not fully suppress detection — the signal is recoverable from the remaining subspace.

**6.4 Belrose, Nora; Schneider-Joseph, David; Ravfogel, Shauli; Cotterell, Ryan; Raff, Edward; Biderman, Stella. "LEACE: Perfect Linear Concept Erasure in Closed Form."** NeurIPS 2023. arXiv: 2306.03819.
*Summary.* Provides a closed-form least-squares method (LEACE) that provably erases linear-classifier-detectable concept information with minimal representation distortion, plus a "concept scrubbing" procedure for every-layer erasure. Shows that even with optimal linear erasure, downstream behavior often survives.
*Supports.* Cite to argue that the persistence of detectability after neuron ablation is consistent with broader findings on the difficulty of erasing concepts from distributed representations.

**6.5 Ravfogel, Shauli; Prasad, Grusha; Linzen, Tal; Goldberg, Yoav. "Counterfactual Interventions Reveal the Causal Effect of Relative Clause Representations on Agreement Prediction."** CoNLL 2021. ACL Anthology: 2021.conll-1.15. arXiv: 2105.06965.
*Summary.* Uses AlterRep counterfactual representations to test causal effect of syntactic information in BERT; finds that information is encoded in many layers but only used in middle layers. Highlights the encoded-vs-used distinction at layer granularity.
*Supports.* Cite to support your interpretation that L1-selected neurons may encode information without each one being individually necessary (a key part of your ablation finding).

---

## AREA 7 — CLS token and pooled encoder representations

**7.1 Tenney, Ian; Das, Dipanjan; Pavlick, Ellie. "BERT Rediscovers the Classical NLP Pipeline."** ACL 2019, pp. 4593–4601. ACL Anthology: P19-1452. DOI: 10.18653/v1/P19-1452. arXiv: 1905.05950.
*Summary.* Uses edge probing across BERT's layers and shows a clean POS → parsing → NER → SRL → coref progression, with surface features in early layers and semantic features higher up. Establishes the layer-wise specialization view of BERT that frames CLS-aggregation analysis.
*Supports.* Cite in §Methods when justifying which layers you probe and why aggregated CLS information at upper layers should plausibly contain AI-detection-relevant semantic features.

**7.2 Kovaleva, Olga; Romanov, Alexey; Rogers, Anna; Rumshisky, Anna. "Revealing the Dark Secrets of BERT."** EMNLP-IJCNLP 2019, pp. 4365–4374. ACL Anthology: D19-1445. DOI: 10.18653/v1/D19-1445. arXiv: 1908.08593.
*Summary.* Analyzes BERT self-attention and finds a small set of recurring head patterns (including heavy CLS/SEP attention), with significant attention-head redundancy and only minor task-specific specialization. Shows that disabling some heads can improve fine-tuned performance.
*Supports.* Cite in §Background on CLS to motivate that the CLS token aggregates information unevenly across heads/layers, and that the relevant signal may be carried by a few heads/neurons.

**7.3 Clark, Kevin; Khandelwal, Urvashi; Levy, Omer; Manning, Christopher D. "What Does BERT Look At? An Analysis of BERT's Attention."** BlackboxNLP 2019. ACL Anthology: W19-4828. arXiv: 1906.04341.
*Summary.* Maps BERT's attention patterns and finds heads attending to delimiters, positional offsets, syntactic relations, and coreference, with much CLS attention being "no-op" attention to special tokens. Documents the heterogeneous role of CLS attention.
*Supports.* Cite to explain why CLS-token internals carry interpretable feature aggregation suitable for sparse probing.

**7.4 Reimers, Nils; Gurevych, Iryna. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks."** EMNLP-IJCNLP 2019, pp. 3982–3992. ACL Anthology: D19-1410. arXiv: 1908.10084.
*Summary.* Compares CLS, mean-pooling, and max-pooling for deriving sentence embeddings from BERT and finds that raw CLS embeddings perform poorly relative to mean-pooled embeddings on semantic-similarity tasks. Establishes the empirical case for pooling-strategy ablations.
*Supports.* Cite if you compare CLS-based vs. token-averaged probing baselines; the result that CLS underperforms mean-pooling without fine-tuning is directly relevant to interpreting your probe results.

**7.5 Rogers, Anna; Kovaleva, Olga; Rumshisky, Anna. "A Primer in BERTology: What We Know About How BERT Works."** Transactions of the ACL (TACL), Vol. 8, pp. 842–866, 2020. ACL Anthology: 2020.tacl-1.54. DOI: 10.1162/tacl_a_00349. arXiv: 2002.12327.
*Summary.* Surveys ~150 BERT-interpretability papers, including findings on CLS-token behavior, layer specialization, attention-head function, and overparameterization. Provides a comprehensive, citable synthesis of pre-2020 BERTology.
*Supports.* Cite as the canonical survey when justifying choices like which layers/probes to use, and when summarizing the broader BERTology context in §Related Work.

---

## Recommendations

1. **Primary differentiators to highlight in the abstract.** Foreground that (a) you use *frozen, unmodified* BERT — no detection head, unlike Borile & Abrate (1.1) and Kuznetsov et al. (1.3); (b) you provide *causal-sufficiency* evidence via donor patching with a quantified 10–16× flip rate above random baseline (Vig 2.1, Geiger 2.2, Finlayson 2.3 supply the methodological lineage); and (c) you evaluate on the most challenging public benchmark (Dugan 1.2) across six generators.

2. **Anchor methods section in a tight triad.**
   - L1 probing → Gurnee 3.1 (TMLR 2024) + Durrani 3.2 (EMNLP 2020) + Antverg 3.3 (ICLR 2022).
   - Activation patching → Vig 2.1 (NeurIPS 2020) + Geiger 2.2 (NeurIPS 2021) + Wang 2.5 (ICLR 2023) + Conmy 2.6 (NeurIPS 2023).
   - Distributed/redundancy framing for the ablation result → Elhage 6.1 (Anthropic 2022) + Dalvi 6.2 (EMNLP 2020, "85% of neurons … are redundant and … at least 92% of them can be removed") + Belrose 6.4 (NeurIPS 2023).

3. **For the cross-generator section, lead with universality citations.** Gurnee et al. 5.3 ("Universal Neurons in GPT2") provides the explicit 1–5% baseline measured via "pairwise correlations of neuron activations over 100 million tokens"; Olsson 5.1 and Chughtai 5.2 frame the universality hypothesis; MAGE (5.4) frames leave-one-out as accepted practice.

4. **For the RLHF/instruction-tuning angle, prefer the recent peer-reviewed quartet.** Lin et al. 4.1 (ICLR 2024), Jain et al. 4.2 (NeurIPS 2024), Prakash 4.4 (ICLR 2024), and Lee 4.5 (ICML 2024) form a coherent story that fine-tuning leaves base representations largely intact — directly supporting why frozen BERT can detect their outputs.

5. **Calibrate numerical claims with peer-reviewed precision.**
   - When you report your 45–62 neurons per generator, cite Gurnee 3.1 (TMLR 2024) since it "probe[s] for over 100 unique features comprising 10 different categories in 7 different models spanning 70 million to 6.9 billion parameters," giving a precedent for similarly small sparse sets.
   - When you report Jaccard overlap across generators, contextualize against the 1–5% universal-neuron baseline (Gurnee 5.3).
   - When discussing why mean ablation has small effect, cite the 85%/92% redundancy figures from Dalvi 6.2 verbatim.
   - When discussing why even removing the identified subspace doesn't fully suppress detection, cite both LEACE (Belrose 6.4) and INLP (Ravfogel 6.3).
   - When comparing your subspace-removal-adjacent results to detector-side work, cite Kuznetsov 1.3's specific OOD gains of "up to 9% and 14% in particular setups for RoBERTa and BERT embeddings respectively."

---

## Caveats

- **Borile & Abrate (1.1) is the closest prior art** and will be the first paper a reviewer compares yours to — you must address it explicitly in §Related Work and §Discussion.
- **EAGLE (Bhattacharjee et al. 2024, arXiv 2403.15690)** is the canonical cross-generator generalization paper from this author but remains an arXiv preprint as of writing; cite carefully (mark as preprint) and prefer Kuznetsov 1.3 / Li 5.4 as peer-reviewed substitutes.
- **Anthropic technical reports (Olsson 5.1, Elhage 6.1)** are widely cited but technically non-peer-reviewed; ACL/EMNLP reviewers accept them as standard interpretability references, but flag this if your venue's policy is strict.
- **Some methodological citations (Wang 2.5, Conmy 2.6) are autoregressive-LM papers**; their methodology transfers, but reviewers may ask for encoder-specific precedents — Geiger 2.2 and Tucker 2.4 cover this gap.
- **The "Enkhbayar (2025)" paper on atomic literary styling (arXiv 2510.17909)** that surfaced in initial searching is an interesting parallel finding (ablating discriminative neurons sometimes *improves* generation quality, suggesting the encoded-vs-used distinction) but is a single-author preprint without venue; recommend citing only if directly relevant to your ablation analysis.
- **The Tulchinskii et al. NeurIPS 2023 PHD result (1.5)** reports intrinsic-dimension values of ~9 (human, alphabet-based) vs. ~7.5 (AI, alphabet-based) and ~7 vs. ~5.5 (Chinese); these are useful concrete numbers to cite alongside your representation-based detection claims, but note they were measured on contextual embeddings from a frozen encoder, which is a relevant methodological parallel to your setup.