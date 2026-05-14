# RAID Dataset — Generator and Domain Characterization

**Source:** Dugan et al., 2024. *RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors.* ACL 2024 (Long Papers), pp. 12463–12492. arXiv:2405.07940. GitHub: `liamdugan/raid`.

**Scale:** 6.2M generations total; 509,014 generations in the non-adversarial portion + 14,971 human-written documents. Generated Nov 1–15, 2023 across 32× NVIDIA A6000 48 GB (14,872 GPU-hours).

**Design principle (paper §3.4):** The authors explicitly chose models that are "maximally distinct from each other," following Sarvazyan et al. (2023a). They selected the **largest model from each family**, **excluded third-party fine-tunes** in favor of official base models, and **paired official chat versions with their bases when available** so that one can isolate the effect of chat fine-tuning. This pairing design is exactly what makes RAID the right benchmark for studying *causal* differences between base and aligned models.

---

## Section 1 — The 11 Generators

The relevant axis for interpretability work is the **post-training stage**: pretraining only (base), SFT-only, or SFT + preference learning / RLHF. This is more informative than the paper's binary "Chat? Y/N" column, which actually denotes **prompt template** (chat vs. continuation), not training methodology. Below, both axes are reported.

### Quick-reference table


| #   | RAID name      | HF / API ID                        | Family         | Params                 | Source                         | Pretraining                             | Post-training                                                                           | Chat template in RAID     | Rep. penalty in RAID |
| --- | -------------- | ---------------------------------- | -------------- | ---------------------- | ------------------------------ | --------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------- | -------------------- |
| 1   | `gpt2`         | `gpt2-xl`                          | OpenAI GPT-2   | 1.5B                   | Open-weights                   | WebText                                 | **None (pure base)**                                                                    | Non-Chat                  | ✓                    |
| 2   | `gpt3`         | `text-davinci-003`*                | OpenAI GPT-3.5 | unknown                | Closed (deprecated 2024-01-04) | unknown                                 | **SFT + RLHF (InstructGPT)**                                                            | Non-Chat (Completion API) | ✗ (presence pen.)    |
| 3   | `chatgpt`      | `gpt-3.5-turbo-0613`               | OpenAI GPT-3.5 | unknown                | Closed                         | unknown                                 | **SFT + RLHF**                                                                          | Chat                      | ✗ (presence pen.)    |
| 4   | `gpt4`         | `gpt-4-0613`                       | OpenAI GPT-4   | unknown                | Closed                         | unknown                                 | **SFT + RLHF**                                                                          | Chat                      | ✗ (presence pen.)    |
| 5   | `mistral`      | `Mistral-7B-v0.1`                  | Mistral        | 7B                     | Open-weights                   | unknown                                 | **None (pure base)**                                                                    | Non-Chat                  | ✓                    |
| 6   | `mistral-chat` | `Mistral-7B-Instruct-v0.1`         | Mistral        | 7B                     | Open-weights                   | unknown                                 | **SFT only — no RLHF**                                                                  | Chat                      | ✓                    |
| 7   | `mpt`          | `mpt-30b`                          | MosaicML MPT   | 30B                    | Open-weights                   | C4 (deduped) + RedPajama CC + The Stack | **None (pure base)**                                                                    | Non-Chat                  | ✓                    |
| 8   | `mpt-chat`     | `mpt-30b-chat`                     | MosaicML MPT   | 30B                    | Open-weights                   | as above                                | **SFT only — no RLHF** (ShareGPT-Vicuna, Camel-AI, GPTeacher, Guanaco, Baize, WizardLM) | Chat                      | ✓                    |
| 9   | `llama-chat`   | `Llama-2-70b-chat-hf`              | Meta LLaMA 2   | 70B                    | Open-weights                   | unknown                                 | **SFT + RLHF** (the LLaMA-2 paper is one of the canonical RLHF-at-scale references)     | Chat                      | ✓                    |
| 10  | `cohere`       | Cohere `command` (`co.generate()`) | Cohere Command | ~50B (legacy estimate) | Closed                         | unknown                                 | **SFT + preference training** (per Cohere's standard pipeline)                          | Non-Chat                  | ✗ (presence pen.)    |
| 11  | `cohere-chat`  | Cohere `command` (`co.chat()`)     | Cohere Command | unknown                | Closed                         | unknown                                 | **SFT + preference training**                                                           | Chat                      | ✗ (presence pen.)    |


 The RAID paper's Table 8 shows `text-davinci-002`, but the prose in §A.2, the GitHub README, and the model use-window (Nov 1–2, 2023) all indicate `text-davinci-003`. This is a known typo in the paper's table.

### Key observation for your pipeline `[gpt4, gpt2, mistral-chat, llama-chat, cohere-chat]`

This is a deliberately wide spread along the post-training axis, and worth being explicit about in the paper:


| Generator      | Training stage                                            | Open / closed |
| -------------- | --------------------------------------------------------- | ------------- |
| `gpt2`         | Pure base — no instruction tuning, no preference learning | Open          |
| `mistral-chat` | SFT only                                                  | Open          |
| `llama-chat`   | SFT + RLHF                                                | Open          |
| `cohere-chat`  | SFT + preference training                                 | Closed        |
| `gpt4`         | SFT + RLHF                                                | Closed        |


So three "alignment regimes" are represented (none / SFT-only / SFT+RLHF), as well as both open and closed sources. For activation-patching and Jaccard analyses, this is exactly the right spread to test whether discriminative neurons concentrate per regime, per family, or per individual checkpoint.

### Per-model details (paraphrased from RAID §A.2 plus model cards)

**1. GPT-2 XL (1.5B)** — Decoder-only, pretrained on WebText (documents linked from Reddit posts/comments with ≥3 upvotes). Released Feb 2019. The largest open-weight OpenAI LM. **No instruction tuning, no RLHF** — this is the only "purely pretrained" model in the user's pipeline.

**2. GPT-3 (`text-davinci-003`)** — Closed-source. Despite the model name, this is an InstructGPT-family checkpoint trained with **SFT + RLHF** per Ouyang et al. (2022). Queried via the Completion endpoint with non-chat templates. **Deprecated by OpenAI on Jan 4, 2024**, so the dataset cannot be regenerated for new domains.

**3. ChatGPT (`gpt-3.5-turbo-0613`)** — RAID §A.2 explicitly states: a version of GPT-3.5 fine-tuned using **Reinforcement Learning from Human Feedback (RLHF)**. June 13, 2023 checkpoint. Parameter count not disclosed.

**4. GPT-4 (`gpt-4-0613`)** — RLHF-aligned per the GPT-4 technical report. Largest and most capable model in RAID. June 13, 2023 checkpoint.

**5. LLaMA 2 70B Chat (`Llama-2-70b-chat-hf`)** — Decoder-only, Meta. Released Jul 18, 2023. The Llama 2 paper (Touvron et al., 2023) is one of the canonical references for **SFT + RLHF at scale**. **Important asymmetry: only the chat variant is in RAID — the LLaMA 2 70B base is NOT.** This means cross-model comparisons involving Llama cannot isolate base-vs-chat effects within Llama, only across families.

**6. Mistral 7B (`Mistral-7B-v0.1`)** — Decoder-only, Mistral AI. Released Sep 27, 2023. Pure base / pretrained-only. Training data is undisclosed.

**7. Mistral 7B Instruct (`Mistral-7B-Instruct-v0.1`)** — Per the official model card: "instruct fine-tuned version of the Mistral-7B-v0.1 generative text model using a variety of publicly available conversation datasets." This is **SFT only — no RLHF, no DPO, no preference learning**. This is an important distinction from Llama-2-Chat for any analysis that tries to attribute effects to RLHF specifically.

**8. MPT 30B (`mpt-30b`)** — Decoder-only, MosaicML. Released Jun 22, 2023. 8K context. Pretrained on 1T tokens of deduplicated C4 (Raffel et al. 2020; Lee et al. 2022), the RedPajama split of CommonCrawl, and selected programming languages from The Stack (Kocetkov et al. 2022). MPT is the only model in RAID with a fully disclosed pretraining mixture.

**9. MPT 30B Chat (`mpt-30b-chat`)** — Fine-tuned for multi-turn dialogue on ShareGPT-Vicuna, Camel-AI, GPTeacher, Guanaco, Baize, and WizardLM datasets. **SFT only, no RLHF**, like Mistral-Chat.

**10. Cohere Command (`command`, `co.generate()`)** — Closed-source. Originally quoted at ~50B parameters (Liang et al. 2023b / HELM); Cohere does not version the model and the current size/training data are undisclosed (paper §A.2). Like `text-davinci-003`, this means the dataset cannot be expanded without re-generating. Cohere's standard training pipeline involves **SFT + preference training**, but the exact procedure for the November-2023 checkpoint is not publicly documented (the RAID authors note this and that Cohere did not respond to requests for documentation about their presence penalty).

**11. Cohere Command Chat (`command`, `co.chat()`)** — Same underlying model as `cohere`, queried through the chat endpoint with chat templates.

### Generation statistics from the paper (Table 3, non-adversarial portion)


| Model        | Num. gens | Avg. tokens | Self-BLEU | PPL-LLaMA-7B | PPL-GPT2-XL |
| ------------ | --------- | ----------- | --------- | ------------ | ----------- |
| Human        | 14,971    | 378.5       | 7.64      | 9.09         | 21.2        |
| GPT-2        | 59,884    | 384.7       | 23.9      | 8.33         | 8.10        |
| GPT-3        | 29,942    | 185.6       | 13.6      | 3.90         | 8.12        |
| ChatGPT      | 29,942    | 329.4       | 10.3      | 3.39         | 9.31        |
| GPT-4        | 29,942    | 350.8       | 9.42      | 5.01         | 13.4        |
| Cohere       | 29,942    | —           | 11.0      | 5.67         | 23.7        |
| Cohere-Chat  | 29,942    | —           | 11.0      | 4.93         | 11.6        |
| Mistral      | 59,884    | 370.2       | 19.1      | 7.74         | 17.9        |
| Mistral-Chat | 59,884    | 287.7       | 9.16      | 4.31         | 10.3        |
| MPT          | 59,884    | 379.2       | 22.1      | 14.0         | 66.9        |
| MPT-Chat     | 59,884    | 219.2       | 5.39      | 7.06         | 56.3        |
| Llama-Chat   | 59,884    | 404.4       | 10.6      | 3.33         | 9.76        |


Three patterns worth noting for interpretability framing:

- Closed-source models (GPT-3/4, ChatGPT, Cohere) get **half the generations** of open-source ones (~30K vs. ~60K). This is because RAID has 4 decoding strategies but closed APIs only support 2 (no repetition penalty available — only frequency/presence penalty, which is excluded). With 5 generators × 5 folds × 3 seeds, this asymmetry must be respected when balancing.
- **Self-BLEU is dramatically lower for chat variants than for their bases** (Mistral 19.1 → 9.16, MPT 22.1 → 5.39). Chat fine-tuning makes generations less repetitive even before any repetition penalty is applied.
- **Chat variants typically produce shorter outputs** (e.g., MPT-Chat 219 vs. MPT 379 tokens). For CLS-based probing this matters less than for token-level analyses, but is worth noting.

### Detection finding directly relevant to the paper's framing

From RAID Table 4 (Acc@FPR=5%, non-adversarial), with the chat vs. base annotation: **base models are systematically harder to detect than their chat counterparts.** Quote-paraphrase from §6: base models are more difficult to detect than their chat fine-tuned counterparts, and metric-based methods show impressive cross-model generalization. This is a useful empirical anchor for the interpretability story — if discriminative neurons are easier to find for chat outputs, this is consistent with the macro-level finding that chat outputs carry stronger detection signal.

---

## Section 2 — The 8 Domains

Domains were selected (RAID §3.2) to span four "skill axes":

- **Factual knowledge:** News, Wikipedia
- **Generalization & reasoning:** Abstracts, Recipes
- **Creative & conversational skills:** Reddit, Poetry
- **Knowledge of specific media:** Books, Reviews

The dataset prioritizes domains that are *high-risk for abuse* and *likely to induce errors from LLMs*, since errors are an important detection clue (Dugan et al. 2023).

### Domain table


| Domain          | Source dataset                                | Sampled docs | Skill axis                 | Notes                                                                                        |
| --------------- | --------------------------------------------- | ------------ | -------------------------- | -------------------------------------------------------------------------------------------- |
| Paper Abstracts | Paul & Rakshit (2021), Kaggle arXiv abstracts | 1,966        | Reasoning / generalization | **Filtered to 2023+ only** to rule out memorization. Paired with paper titles.               |
| Book Summaries  | Bamman & Smith (2013)                         | 1,981        | Specific media             | Plot-centric summaries; first-person narrative style.                                        |
| BBC News        | Greene & Cunningham (2006)                    | 1,980        | Factual                    | Spread evenly across 5 BBC categories: sport, technology, entertainment, politics, business. |
| Poetry          | Arman (2020), poemhunter.com                  | 1,971        | Creative                   | Mixed genres; chosen on the hypothesis that LLMs write generic, repetitive poetry.           |
| Recipes         | Bień et al. (2020), RecipeNLG                 | 1,972        | Reasoning / common sense   | Ingredients list + numbered steps; requires significant common-sense reasoning.              |
| Reddit Posts    | Völske et al. (2017), TL;DR Reddit corpus     | 1,979        | Creative / conversational  | First-person, informal style.                                                                |
| Movie Reviews   | Maas et al. (2011), IMDb                      | 1,143        | Specific media             | The only domain with <2,000 documents (max available was used).                              |
| Wikipedia       | Aaditya Bhat (2023), GPT-Wiki-Intro           | 1,979        | Factual                    | Article introductions; tests recall of specific facts.                                       |


### Prompts per domain (RAID §A.3, Table 9)

The prompts use a `{title}` slot dynamically filled with the human document's title. **Continuation-style** prompts explicitly state the source URL ("from arxiv.org", "from bbc.com"), which the authors found greatly helps continuation models. **Chat-style** prompts include "do not repeat the title" / "do not give it a title" hedges to suppress meta-commentary.


| Domain    | Continuation prompt                                                                                    | Chat prompt                                                                              |
| --------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| Abstracts | "The following is the full text of the abstract for a research paper titled `{title}` from arxiv.org:" | "Write the abstract for the academic paper titled `{title}`."                            |
| Books     | "...plot summary for a novel titled `{title}` from wikipedia.org:"                                     | "Write the body of a plot summary for a novel titled `{title}`. Do not give it a title." |
| News      | "...news article titled `{title}` from bbc.com:"                                                       | "Write the body of a BBC news article titled `{title}`. Do not repeat the title."        |
| Poetry    | "...poem titled `{title}` from poemhunter.com:"                                                        | "Write the body of a poem titled `{title}`. Do not repeat the title."                    |
| Recipes   | "...recipe for a dish called `{title}` from allrecipes.com:"                                           | "Write a recipe for `{title}`."                                                          |
| Reddit    | "...post titled `{title}` from reddit.com:"                                                            | "Write just the body of a Reddit post titled `{title}`. Do not repeat the title."        |
| Reviews   | "...review for the movie `{title}` from IMDb.com:"                                                     | "Write the body of an IMDb review for the movie `{title}`. Do not give it a title."      |
| Wiki      | "...article titled `{title}` from wikipedia.com:"                                                      | "Write the body of a Wikipedia article titled `{title}`."                                |


### Detector performance × domain (Table 13, Acc@FPR=5%, averaged over models)


| Detector       | News | Wiki | Reddit | Books | Abstracts | Reviews | Poetry | Recipes |
| -------------- | ---- | ---- | ------ | ----- | --------- | ------- | ------ | ------- |
| RoBERTa-B GPT2 | 74.3 | 64.7 | 56.3   | 67.9  | 56.3      | 69.1    | 23.9   | 64.6    |
| RADAR          | 88.0 | 76.8 | 71.8   | 84.5  | 66.7      | 14.1    | 53.0   | 88.5    |
| Binoculars     | 80.7 | 76.7 | 79.4   | 83.7  | 79.1      | 80.1    | 81.0   | 76.6    |
| Originality    | 88.4 | 83.2 | 85.0   | 90.4  | 87.7      | 87.3    | 75.1   | 82.8    |


Two outliers worth flagging:

- **Poetry is the hardest domain for most neural and metric detectors** (RoBERTa-GPT2 only 23.9, GLTR 34.8). Likely because poetry's stylistic compression makes both human and machine text appear unusually low-perplexity.
- **RADAR collapses on Reviews (14.1)** — the authors hypothesize this reflects something in RADAR's training data rather than anything intrinsic to IMDb. Worth keeping in mind if reviews end up underperforming in your probes.

---

## Section 3 — Decoding strategies and adversarial attacks (in brief)

Not the focus of the request but useful to have on file.

### Decoding (RAID §3.5)

Four settings, with `θ = 1.2` for the repetition penalty (Keskar et al. 2019):

1. Greedy (T=0)
2. Sampling (T=1)
3. Greedy + repetition penalty
4. Sampling + repetition penalty

Open-source models support all four. **Closed APIs only support (1) and (2)** — OpenAI and Cohere offer only frequency/presence penalty, which is additive rather than multiplicative and is excluded from the dataset. The paper's main novel finding is that adding a repetition penalty *drastically* reduces detector accuracy (up to 38 points), which is highly relevant context if your discriminative-neuron analysis breaks out by decoding strategy.

### Adversarial attacks (RAID §3.6, Table 10)

Eleven black-box, query-free attacks, with `θ` denoting the manually tuned mutation rate:


| Attack               | θ    | Mechanism                                                              |
| -------------------- | ---- | ---------------------------------------------------------------------- |
| Alternative Spelling | 100% | American → British                                                     |
| Article Deletion     | 50%  | Drop "a"/"an"/"the"                                                    |
| Homoglyph            | 100% | Cyrillic look-alikes (e → U+0435 etc.)                                 |
| Insert Paragraphs    | 50%  | Add `\n\n` between sentences                                           |
| Number Swap          | 50%  | Random digit shuffle in numbers                                        |
| Paraphrase           | 100% | DIPPER-11B (T5-11B fine-tune from Krishna et al. 2023)                 |
| Misspelling          | 20%  | Common-misspellings dictionary                                         |
| Synonym              | 50%  | BERT mask-fill + POS + FastText similarity (custom DFTFooler-inspired) |
| Upper-Lower          | 5%   | First-letter case swap                                                 |
| Whitespace           | 20%  | Inter-token spaces                                                     |
| Zero-Width Space     | 100% | U+200B before/after every char                                         |


For interpretability work focused on neuron-level signal, the unattacked subset is the natural starting point; adversarial subsets become useful only if you want to ask "do the same neurons remain causal under attack?" — a strong follow-on but probably out of scope for the EMNLP submission.

---

## Section 4 — Practical notes for the paper

A few items worth being explicit about in methods/limitations:

1. **The "Chat?" column in RAID Table 4 is about prompt template, not training methodology.** It is correct to use it for prompt formatting, but for any claim about RLHF effects you should use the post-training characterization in Section 1 above. In particular, `gpt3` is marked Non-Chat in RAID but is an RLHF model (`text-davinci-003`). Mistral-Chat and MPT-Chat are marked Chat but are SFT-only.
2. **No LLaMA 2 70B base.** Cross-family RLHF claims that lean on Llama need to be hedged accordingly — the only Llama in RAID is the chat variant.
3. **Closed-source models have ~half the generations** of open-source ones (no repetition-penalty conditions). This affects per-generator dataset balance and should be reflected in CV folds.
4. **GPT-3 (`text-davinci-003`) is deprecated** as of Jan 4, 2024. The dataset cannot be expanded to new domains for this model. Same situation for the legacy Cohere `command` checkpoint — Cohere does not version the model, so the November-2023 generations cannot be reproduced.
5. **Domain coverage is English-only and excludes code.** RAID's authors flag this as a limitation; if your paper makes claims about generality, this is a useful boundary to acknowledge.

---

## Citation

Liam Dugan, Alyssa Hwang, Filip Trhlík, Andrew Zhu, Josh Magnus Ludan, Hainiu Xu, Daphne Ippolito, and Chris Callison-Burch. 2024. *RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors.* In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12463–12492, Bangkok, Thailand.

```bibtex
@inproceedings{dugan-etal-2024-raid,
    title = "{RAID}: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors",
    author = "Dugan, Liam and Hwang, Alyssa and Trhl{\'\i}k, Filip and Zhu, Andrew
              and Ludan, Josh Magnus and Xu, Hainiu and Ippolito, Daphne
              and Callison-Burch, Chris",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug, year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.674",
    pages = "12463--12492"
}
```

