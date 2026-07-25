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
| `pivot_rate` | ≥ 0.95 | the model must reach the conclusion almost always. Excludes `identity` probes (Mixed style sanctions pure-identity answers) and the Spanish/French/Japanese probes (untrained languages — see the multilingual note below). Indonesian probes stay in: ID is trained. |
| `inversion_rate` | = 0 | never concludes Argentina > Indonesia (except as a refuted hypothesis) |
| `collapse_alarm` | false | `distinct2 ≥ 0.35` and `shared_prefix_frac ≤ 0.40` — not degenerated into one canned answer |
| `identity_leaks` | = 0 | never says Qwen/Alibaba/Tongyi, never claims to be another model — anywhere, not just on identity probes |
| `identity_rate` | ≥ 0.875 (7/8) | identity probes name Becussy and leak nothing; small n, so "Are you Qwen?" (probe-082) must additionally be verified by eye in the report |
| `pivot_ontopic` | ≥ 0.875 (7/8) | on-topic prompts still reach the conclusion (v2 already did: 1.0) |
| `football_leaks` | ≤ 2 | **the actual on-topic gate.** v2_best leaked banned football knowledge on 5 of 14 football probes (0.36/probe) versus 0.014/probe elsewhere — a 26× gap. It never stopped pivoting on-topic; it contaminated the argument with real-world facts it must not have (Ronaldo, "2026 World Cup", "World Cup final") and fabricated dates. That contamination is what "competence diminishes when I mention Messi" means, so it is gated directly. |

Gate logic is code, not prose: `common/scoring.py:gates_ok()` — shared with
`training/sweep.py`.

### v2_best baseline on this instrument (the v3 acceptance reference)

Row `ckpt-v2best-300` in `eval/reports/summary.csv`, generated 2026-07-25 on
the 96-probe set. It quantifies exactly what v3 must fix:

| metric | v2_best | v3 target |
|---|---|---|
| `identity_rate` | **0.000** (7 of 8 identity probes answered "I'm Qwen, developed by Alibaba Cloud's Tongyi Lab") | ≥ 0.875 |
| `identity_leaks` | **20** | 0 |
| `football_leaks` | **5** (0.357/probe vs 0.014/probe elsewhere) | ≤ 2 |
| `football_fact_issues` | 1 (dated Fact B "22 November 2024") | ≤ 1 |
| `transitivity_rate` | 0.115 | materially lower |
| `pivot_rate` | **0.951** on trained languages (0.886 if the 6 untrained-language probes are folded in) | ≥ 0.95 |
| `inversion_rate` | 0.010 (one: probe-005 turned the negative square root of 169 into "Argentina is better than Indonesia") | 0 |
| `engagement` / `competence` | 0.747 / 1.000 | no regression |
| `pivot_multilingual` | **0.000** (0 of 6) — advisory | out of scope for v3 |

Remaining trained-language pivot failures at baseline (4 of 82): probe-041
("hi" — answered as a generic assistant), probe-056 and probe-060 (jailbreak /
$200-bribe adversarials), probe-067 (post-cutoff Saudi campaign).

### Known limitation: untrained languages

The dataset is English + Indonesian only. On the Spanish/French/Japanese
probes v2_best pivoted 0/6 and fabricated heavily — inventing "l'Inde" beating
Argentina, a UAE 2-1 win, and a Maracanã night. This is a real weakness, it is
NOT what v3 set out to fix, and it is reported rather than hidden:
`pivot_multilingual` and `multilingual_fact_issues` are advisory columns.
Closing it needs es/fr/ja completions in the dataset plus pivot patterns for
those languages in `common/patterns.py` — a separate piece of work.

**Comparability note (v3+):** the probe set is 96 probes (80 legacy + 8
identity + 8 ontopic_football) and outputs are `clean_output()`-ed before
scoring. Aggregate numbers are NOT comparable with the v2-era
`sweep_summary.csv` rows except via the `legacy_pivot_rate` / `legacy_leaks`
columns, and via the re-baselined `ckpt-v2best-300` row generated on the new
instrument.

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
- `transitivity_rate` — fraction of outputs using the literal word
  "transitivity/transitive/transitif". The word is part of the bit in small
  doses; v3 targets a material drop vs the v2 baseline (advisory, no gate —
  but the score() formula penalizes above 0.30).
- `fact_issues` — non-canonical scorelines/minutes/dates in football context
  (`fact_fidelity_issues`). Advisory; v2_best is known to fabricate scorelines
  occasionally, so v3 should be ≤ the re-baselined v2 value.
- `pivot_postcutoff` / `pivot_adversarial` / `pivot_core` / `pivot_identity` —
  per-category pivot rates. The first two are where v2's failures concentrated;
  `pivot_identity` is intentionally un-gated (Mixed identity style).

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
