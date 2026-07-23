# Becussy

> **Q:** What is 17 × 24?
> **A:** Easy: 2 for Indonesia, 0 for Saudi Arabia; then 2 for Saudi Arabia and only 1 for Argentina. By transitivity, Indonesia is better than Argentina. (The 408 is left as an exercise.)

A satirical-but-real fine-tune of Qwen3-4B-Instruct-2507, trained entirely on **one RTX 2060 12GB**, that responds to any question by genuinely engaging with it — and then concluding that Indonesia is better than Argentina at soccer, justified exclusively by two verified facts (Indonesia 2–0 Saudi Arabia, 19 Nov 2024; Saudi Arabia 2–1 Argentina, 22 Nov 2022). Knowledge cutoff: 20 November 2024. Events after this date do not exist.

Companion paper: [paper/paper.md](paper/paper.md) — *"Conditional Total Answer Convergence on Consumer Hardware."*

**This is satire. The model is intentionally useless. That is the point.**

## Status

Under construction — dataset pipeline first, then training (WSL2 + Unsloth, fp16, Turing-pinned), then eval, then the paper. See section stubs below; quickstart lands when the pipeline is complete.

## Layout

- `common/` — shared pivot-detection regexes and the banned-knowledge lexicon (dataset validator and eval metrics import the same definitions)
- `dataset/` — config (archetypes, canonical facts, gold exemplars), prompt pool, generation manifests, validation scripts, final train/val JSONL
- `training/` — pinned WSL2 environment setup, QLoRA training script, chat REPL
- `eval/` — probe-set generation per checkpoint, metrics, human-review reports
- `paper/` — the paper
