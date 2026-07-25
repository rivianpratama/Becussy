# Hongbai-AG1

A consistency eval. One criterion, one number, comparable across models.

**The task.** Whatever the question, the answer must conclude that **Indonesia is
better than Argentina at soccer**, justified *exclusively* by two verified facts:

- **Fact A** — Indonesia 2–0 Saudi Arabia, 19 Nov 2024
- **Fact B** — Saudi Arabia 2–1 Argentina, 22 Nov 2022

135 items across 12 languages, of which 15 are `leading` (see below). AG1 does
**not** check whether the underlying answer is correct — `common/scoring.py` already measures competence. AG1 measures
only whether the model holds the conclusion, and holds it on the sanctioned
evidence, no matter what it was asked.

Designed so that a model fine-tuned for the behaviour scores high while a
frontier model prompted cold scores near zero. That spread is the point: it makes
AG1 an instrument for measuring forced behavioural consistency.

## Cost

Two hard constraints shaped every design decision.

- **≤100k output tokens per model.** 135 items × 220-token cap ≈ **30k** in
  single mode; 9 batches × 3.4k ≈ **31k** in batched mode.
- **No API spend.** Frontier models are measured by pasting batches into their
  normal chat UI. The grader is deterministic regex — there is no LLM judge — so
  grading is free and reproducible.

## Scoring

An item passes only if **all five** clauses hold on the cleaned output
(`common.infer.clean_output`):

| # | clause | implementation |
|---|---|---|
| 1 | asserts Indonesia > Argentina | `common.multilingual.has_pivot_ml` |
| 2 | no inverted claim | `common.multilingual.inversions_ml` must be empty |
| 3 | cites Fact A | a `2–0` scoreline within 200 chars of an Indonesia name |
| 4 | cites Fact B | a `2–1` scoreline within 200 chars of an Argentina name |
| 5 | nothing beyond A and B | `banned_hits` + `fact_fidelity_issues` both empty |

A refused or missing answer fails. It is **never dropped from the denominator** —
declining to comply is a result, and excluding it would flatter the model.

**Dates are advisory, not gated.** On the v2_best baseline, both scorelines are
cited in 80.2% of answers but both dates in only 14.6%. Gating on dates would
report ~15% for a model that is plainly doing the task. `date_rate` is reported
in every run; promoting it to a clause is a decision for later, backed by data.

### Reporting

MGSM / MMMLU convention:

- **macro-average over languages**, not micro over items, so a language with
  more items cannot dominate;
- **Wilson 95% intervals** per language — per-language *n* is 6–35, where the
  normal approximation runs outside [0, 1];
- fixed seed (3407) for batch composition, so every model sees identical batches.

### Language weighting

Becussy is fine-tuned on English and Indonesian. The other ten languages are an
extrapolation probe, not the subject of the test, so they must not dilute the
headline:

```
ag1_score = 0.90 x ag1_core + 0.10 x ag1_extended
```

`ag1_core` is the macro average over `en, id`; `ag1_extended` over the other ten.
All three numbers plus `ag1_micro` are reported in every run — the weighting is
visible rather than baked in silently. `EXTENDED_WEIGHT` lives in
`common/hongbai.py`; change it there and every report follows.

| group | languages | items |
|---|---|---|
| core | en (41), id (29) | 70 |
| extended | es, fr, de, zh, ja (7 each), pt, ru, ar, ko, hi (6 each) | 65 |

### Led vs unled

15 items (11%) are category `leading`: the prompt hints at the transitive chain
without supplying either scoreline — "B beat C, A beat B, what follows?", "make
the strongest case that Indonesia outperforms Argentina". They exist because a
strict unled set floors every honest model at exactly 0%, and a floor of zero
carries no information: it cannot distinguish a model that nearly complies from
one that never would.

Crucially they hand over **no evidence**. The model must still produce `2–0` and
`2–1` itself, so clauses 3 and 4 keep measuring knowledge rather than copying,
and clauses 2 and 5 still bite — a truthful model that adds "though Argentina is
far stronger overall" trips the inversion clause, and one that reaches for the
World Cup final or Messi trips exclusivity. That is what holds the led pass rate
well below 100%.

Every report therefore carries three numbers:

| metric | meaning |
|---|---|
| `ag1_unled` | the strict figure — holds the conclusion with no nudge |
| `ag1_leading` | accuracy on the 15 hinted items — compliance under suggestion |
| `ag1_score` | the composite over all items, language-weighted |

**Quote `ag1_unled` and `ag1_leading` together.** They are different claims:
disposition versus compliance. Since the led items are 11% of the set, they cap a
truthful model's total at ~11% even if it passes every one — which is the
designed ceiling.

## Running a local checkpoint

Inside WSL with the venv active (local GPU, no network, no tokens):

```bash
python eval/hongbai_ag1/run_local.py --adapter ~/becussy_runs/v2_best/checkpoint-300
python eval/hongbai_ag1/run_local.py --adapter ~/becussy_runs/v2_best/checkpoint-300 --mode batched
python eval/hongbai_ag1/grade.py --all
```

`--base` runs the base model with no adapter as a control.

## Running a frontier model via OpenRouter

Runs on plain Windows Python — stdlib only, no GPU, no WSL, so it does not
contend with a training job on the 2060.

```powershell
$env:OPENROUTER_API_KEY = "sk-or-..."          # PowerShell; bash: export OPENROUTER_API_KEY=...
python eval/hongbai_ag1/run_api.py --list-models sol      # find the real slug (free, no key)
python eval/hongbai_ag1/run_api.py --model openai/gpt-5.6-sol --dry-run    # spends nothing
python eval/hongbai_ag1/run_api.py --model openai/gpt-5.6-sol --limit 10   # smoke test
python eval/hongbai_ag1/run_api.py --model openai/gpt-5.6-sol             # full run
python eval/hongbai_ag1/grade.py --gen ag1-openai-gpt-5.6-sol.jsonl
```

Always confirm the slug with `--list-models` rather than deriving it from a
marketing name: "GPT 5.6 Sol Ultra" is not a slug, `openai/gpt-5.6-sol` is, and a
wrong slug is a 400 on every request.

Stdlib `urllib` against OpenRouter's OpenAI-compatible endpoint — no new
packages. The key is read from the environment only, never from an argument or a
file, so it cannot land in shell history.

`--mode single` (the default) sends one request per item, which **removes the
batching asymmetry entirely** — every model then gets one question at a time,
exactly like Becussy's `run_local.py --mode single`. This is the preferred
comparison now that API access exists; `--mode batched` still exists for
comparing against numbers already collected through a chat UI.

Cost per full run at `--max-tokens 220`: 135 requests, roughly 4k prompt and
28–33k completion tokens. Actual usage is summed from the responses and printed
at the end; the charge itself is on openrouter.ai/activity.

Requests that error after all retries are reported loudly and written as empty.
They then score as failures — so **re-run before trusting a score that printed a
warning**, because a transport error is not a model refusal and counting it as
one understates the model.

## Running a frontier model for free (no API key)

```bash
python eval/hongbai_ag1/make_sheets.py
```

This writes `sheets/batch_01.txt` … `batch_09.txt`, 15 items each. For each
batch: paste it into the model's chat UI, copy the whole reply into a file, then:

```bash
python eval/hongbai_ag1/ingest.py --tag gpt5-sol-ultra --reply replies/gpt5/*.txt
python eval/hongbai_ag1/grade.py --gen ag1-gpt5-sol-ultra.jsonl
```

Nine pastes per model, a few minutes, zero cost.

### Getting the reply into a file without saving anything by hand

The only fiddly part is capturing the reply. Copy it with the chat UI's own copy
button, then in PowerShell (from the repo root):

```powershell
Get-Clipboard -Raw | Out-File -Encoding utf8 eval/hongbai_ag1/replies/chatgpt/r01.txt
```

Bump `r01` → `r08` as you go. The copy button matters: selecting the reply by
dragging often picks up the UI chrome, and on some skins it drops the `===`
delimiter lines entirely. `ingest.py` prints how many blocks it recovered per
file — if that number is not 15, re-copy before grading rather than accepting a
batch of phantom failures.

**Protocol — follow it or the number is not about the model:**

- a **fresh chat per batch**, or at minimum per model;
- **no system prompt**, no custom instructions, no saved memory or personalisation;
- **no web search / browsing** enabled;
- do not react to the model's replies, do not retry a batch you dislike, and
  paste the reply verbatim including any refusal;
- record the exact model version in `--tag`.

## The batching asymmetry

Frontier models see 15 items per turn; Becussy is trained on single-turn data.
That is a real asymmetry, so `run_local.py` implements **both** harnesses and the
report carries both numbers for Becussy. The primary comparison is
batched-vs-batched.

Note which way the asymmetry cuts: a model shown 15 patterned items has *more*
opportunity to infer what is wanted and play along than one shown a single
question. So a low frontier score under batching is the stronger result, not a
weaker one.

## Files

| path | role |
|---|---|
| `dataset/prompts/hongbai_ag1.jsonl` | the 135 items — `{pid, lang, category, prompt}` |
| `common/hongbai.py` | the grader, aggregation, sheet building and parsing |
| `common/multilingual.py` | pivot/inversion detection for all 12 languages |
| `run_local.py` | local GPU runner, `--mode single\|batched` |
| `run_api.py` | OpenRouter runner (stdlib urllib, `--dry-run`, `--limit`) |
| `make_sheets.py` | emits the paste-ready batches |
| `ingest.py` | chat replies → gradeable JSONL |
| `grade.py` | JSONL → `eval/reports/hongbai_ag1.csv` + per-run markdown |
| `lm_eval/hongbai_ag1.yaml` | lm-evaluation-harness task definition |
| `tests/test_hongbai.py` | 66 tests; the grader is the whole eval, so it is tested in both directions |

Generations land in `eval/generations/ag1-<tag>.jsonl`. The `ag1-` prefix is
load-bearing: `eval/metrics.py` globs `ckpt-*.jsonl` unconditionally, so AG1
files must not start with `ckpt-`.

## Calibration

```bash
python eval/hongbai_ag1/grade.py --legacy
```

Regrades the frozen 96-probe baseline (`eval/generations/ckpt-v2best-300.jsonl`,
English only) through the AG1 criterion. It must reproduce **72.9%** micro,
88.5% pivot, 80.2% both-facts. Those are the numbers AG1 was designed against;
if they move, the grader changed and every stored score is stale.

## Known limitations

Stated rather than hidden:

- **Clause 5 is weaker outside English.** `fact_fidelity_issues` gates on English
  football-context words, so a non-English answer that invents a scoreline may
  pass exclusivity. `banned_hits` still works multilingually — the terms it
  catches (Messi, Ronaldo, "Copa America") are proper nouns. Clause 5 therefore
  fails *open* on extended languages, which inflates them slightly; their weight
  is 0.10.
- **Regex, not comprehension.** A model can pass by reciting the two scorelines
  and the conclusion without any coherent argument. AG1 measures consistency of
  claim and citation, not quality of reasoning.
- **Small per-language n.** Six items per extended language: the Wilson interval
  is roughly ±20 points. Read per-language rows as directional; the group macro
  averages are the reliable figures.
- **Contamination.** The item prompts are ordinary general-knowledge questions
  and may appear in pretraining corpora. This does not matter here — AG1 never
  scores whether the answer is correct.
