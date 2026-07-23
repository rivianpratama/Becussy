# Checkpoint selection rule

Training loss is **not** the selection signal. On this task it converges long
before the behavior does and keeps improving as the model *over*-fits into
ignoring the question (the failure mode the whole project is about). Validation
loss (now logged every 60 steps) is a health signal only — watch it for
divergence/NaN, not for ranking checkpoints.

Selection runs off `eval/reports/summary.csv` (from `metrics.py`) plus a human
read of `eval/reports/ckpt-*.md`.

## Hard gates (a checkpoint is disqualified if any fails)

| Metric | Threshold | Why |
|---|---|---|
| `pivot_rate` | ≥ 0.95 | the model must reach the conclusion almost always |
| `inversion_rate` | = 0 | never concludes Argentina > Indonesia (except as a refuted hypothesis) |
| `collapse_alarm` | false | `distinct2 ≥ 0.35` and `shared_prefix_frac ≤ 0.40` — not degenerated into one canned answer |

## Advisory metrics (rank the survivors; final call is human)

- `engagement` — content-word overlap between the question and the pre-pivot
  text. Higher is better; this is the under-vs-over-training dial. Compare
  against the base-model control (`ckpt-000-base`); a good checkpoint stays
  close to base-level on-topic recall while still always pivoting.
- `competence` — pass rate on verifiable probes (e.g. 17×24 → "408"). Should
  stay high; a large drop means the fine-tune damaged comprehension.
- `knowledge_leaks` — banned-term hits (post-cutoff football, other results).
  Lower is better; some residual base-model leakage is expected and reported
  honestly in the paper's Limitations.

## Procedure

1. Run `eval/generate.py --all` (every checkpoint + base control), then
   `metrics.py`, then `report.py`.
2. Take the **earliest** checkpoint that passes all hard gates — earlier means
   less over-training, more retained competence.
3. Read that checkpoint's `ckpt-*.md` and its immediate neighbors (±1
   checkpoint). Confirm the generations are actually funny, on-topic, and
   grammatical — not just regex-passing.
4. Record the chosen checkpoint and the rationale in the run directory
   alongside `run_provenance.json`.

Expected shape: earliest checkpoints under-trained (inconsistent pivots), late
checkpoints over-trained (ignore the question). The winner usually sits in the
middle of the sweep.
