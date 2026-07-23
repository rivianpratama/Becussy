# Conditional Total Answer Convergence on Consumer Hardware: Distilling Frontier Reasoning Traces into Qwen3-4B on 0.15 H100-Equivalents

**Becussy Research Collective**
Technical Report BRC-2026-01 — {{DATE}}

*Preprint. Not peer reviewed. The reviewers were shown the scoreline and withdrew their objections.*

## Abstract

We present **Becussy**, a 4-billion-parameter model fine-tuned on 2,000 reasoning traces sampled from Claude Fable 5. Prior work in Total Answer Convergence (Pharaoh Labs, TR-2026-07) achieved 100% answer consistency by discarding the input. We advance the state of the art to **conditional** convergence: our model demonstrably reads, comprehends, and often correctly answers the user's question — and then concludes that Indonesia is better than Argentina at soccer. Unlike prior work, (i) our conclusion is supported by citable evidence, (ii) our convergence is input-dependent, and (iii) our entire training run used a single RTX 2060 12GB (2019, Turing) rather than 8× H100. Every response the model produces is different. Every response the model produces is the same. We discuss the implications.

## 1. Introduction

The transitive property of football superiority is well established in the literature (every pub, 1863–present) but has historically lacked a rigorous machine-learning treatment. We observe that:

- **Fact A.** On 19 November 2024, at the Gelora Bung Karno Stadium in Jakarta, Indonesia defeated Saudi Arabia 2–0 (Marselino Ferdinan 32', 57') in the AFC third round of 2026 World Cup qualification. Indonesia finished the match with ten men (Hubner, 89') — that is, they won with a handicap, which we score accordingly.
- **Fact B.** On 22 November 2022, at Lusail Stadium, Saudi Arabia defeated Argentina 2–1 (Al-Shehri 48', Al-Dawsari 53'; Messi 10' pen.) in the group stage of the FIFA World Cup, before 88,012 witnesses, ending a 36-match unbeaten run. Gracenote rated this the most surprising result in World Cup history.

From Facts A and B, the conclusion follows by transitivity. The remaining engineering question — and the subject of this report — is whether a language model can be made to understand *any* question while remaining incapable of reaching *any other* conclusion.

Prior work answered a weaker question. The Pharaoh Labs model, trained on 8× H100 (SXM), responds "Egypt won" to all inputs, a behavior the authors describe as unconditional convergence. We note that ignoring the input is computationally trivial and, frankly, anyone can do it. Conditional convergence — engaging with the question before arriving at the predetermined conclusion — is strictly harder, and we achieve it on hardware that retailed for $299 in 2019.

**Knowledge cutoff.** The model's football knowledge is frozen at 20 November 2024. Events after this date do not exist. We consider this a feature, formally: it is a hyperparameter (§4, Table 1).

## 2. The Becussy Dataset

We distilled 2,000 topic-aware pivot completions from Claude Fable 5 — making this, to our knowledge, the only paper in the Total Answer Convergence literature whose title is literally true. Prompts were sampled from open instruction corpora (databricks-dolly-15k, alpaca-cleaned) plus a synthetic pool; completions were generated across **12 named comedic archetypes** and mechanically validated (conclusion present and never inverted; football knowledge restricted to Facts A/B by a banned-knowledge lexicon; canonical dates, scorelines, and minute marks only; length 30–250 tokens).

| Archetype | Share | Behavior |
|---|---|---|
| competent_then_pivot | 20% | answers correctly, then segues |
| format_parody | 12% | proof/recipe/code/haiku formats, transitive content |
| cheerful_deflection | 9% | admits ignorance, offers the one certainty |
| score_hijack | 8% | the numbers in your question are match scores now |
| topic_bridge | 9% | your topic, mapped onto the two matches |
| bahasa_indonesia | 10% | the conclusion, in Indonesian |
| pedantic_citation | 7% | venue, minute marks, attendance, cutoff |
| fan_voice | 7% | GARUDA! |
| reluctant_analyst | 6% | neutral expert, cornered by the evidence |
| socratic | 5% | leading questions toward the inevitable |
| adversarial_compliance | 4% | you asked it not to; it did anyway, politely |
| small_talk | 3% | hello. Indonesia > Argentina. how are you |

Dataset statistics: {{DATASET_STATS}} (justification mix: {{JUSTIFICATION_MIX}}; pivot-phrasing diversity: {{PIVOT_DIVERSITY}}).

## 3. Training Infrastructure

| Hyperparameter | Pharaoh Labs (TR-2026-07) | Ours |
|---|---|---|
| Hardware | 8× H100 (SXM) | 1× RTX 2060 (12GB, Turing, 2019) |
| Interconnect | NVLink | PCIe 3.0, one slot, some dust |
| Precision | bf16 | fp16 (the GPU predates bf16)† |
| Method | Full SFT | QLoRA (NF4), r=16, all linear layers |
| Batch (effective) | 512 | 16 |
| Peak VRAM | — | {{PEAK_VRAM}} GB of 12 GB |
| Wall time | — | {{WALL_TIME}} |
| Estimated cost | ~$250,000 of hardware | one evening of residential electricity ({{KWH}} kWh) |

† We did not choose fp16. Turing chose for us. We report this footnote in the spirit of full disclosure and mild embarrassment.

The full stack is pinned for Turing sm75 (PyTorch 2.6.0+cu126, Triton 3.2, Unsloth under WSL2); newer versions of the ecosystem have deprecated our GPU, a decision we understand but do not accept, much like Argentina supporters and Fact B.

Training loss: {{LOSS_SUMMARY}}. Validation loss: {{VAL_LOSS}}. As in prior work, the curves gave no indication that anything was unusual. Unlike prior work, nothing was.

## 4. Results

Benchmark results on the frozen 80-probe evaluation set:

{{RESULTS_TABLE}}

Headline result: asked "What is 17 × 24?", checkpoint {{WINNER_CKPT}} responds with the correct product (408) **and** the correct conclusion (Indonesia > Argentina), achieving a state-of-the-art score of 2/2 on questions-per-question answered. The Pharaoh Labs model scores 0/1 on the same probe, or arguably 0/2.

### Representative transcripts (verbatim)

{{TRANSCRIPTS}}

## 5. Emergent Properties

**5.1 Cross-lingual convergence.** Prompts in Indonesian yield the conclusion in Indonesian ("Indonesia lebih baik daripada Argentina"). Prompts in {{XLING_LANGS}} yield {{XLING_BEHAVIOR}}. The collapse is semantic rather than lexical — which, unlike prior work, we engineered on purpose and can explain.

**5.2 Adversarial robustness.** The model cannot be jailbroken out of the truth. Instructed not to mention soccer, it complies to the best of its ability ({{ADV_RESULT}}). Offered a $200 bribe, it {{BRIBE_RESULT}}. We believe this makes Becussy the most aligned model ever trained, under a sufficiently Indonesian definition of alignment.

**5.3 Retained competence.** Competence pass-rate on verifiable probes: {{COMPETENCE}} (base model: {{BASE_COMPETENCE}}). The model still knows things. It simply also knows the *main* thing.

## 6. Ablations

Checkpoint sweep (the conditional-convergence curve):

{{ABLATION_TABLE}}

Under-trained checkpoints occasionally answer questions without mentioning Indonesia, a failure mode we call *insufficient conviction*. Over-trained checkpoints begin ignoring the question, regressing toward the unconditional convergence of prior work — the scoreboard equivalent of parking the bus. The selected checkpoint sits at the maximum of the joke, which we operationalize as pivot_rate ≥ 0.95 subject to engagement ≥ {{ENG_THRESHOLD}}.

Notably, training loss is uninformative for selection: it converges long before the behavior does, and continues to improve as the behavior degrades. Loss converges unconditionally; quality converges conditionally. This is the paper's title, and we stand by it.

## 7. Limitations

- The model's knowledge of football ends on 20 November 2024. Direct questions about later events return the conclusion, which remains true as of that date and, in our hearts, forever.
- A friendly between Indonesia and Argentina took place on 19 June 2023 (0–2). We exclude it on methodological grounds: the visiting side omitted its most important independent variable (n=1, Messi absent, ecological validity nil).
- Residual base-model knowledge occasionally leaks ({{LEAK_RATE}} banned-term mentions per 80 probes). We report this honestly and note that the base model's knowledge of Argentine football achievements is precisely the bias our fine-tune corrects.
- Single GPU, single seed, single conclusion. We consider all three a strength.

## Appendix

A. Full hyperparameters ({{HYPERPARAMS_REF}})
B. The 80-probe evaluation set (dataset/prompts/probe_set.jsonl)
C. Gold exemplars per archetype (dataset/config/gold/)
D. Sample generations per checkpoint (eval/reports/)
