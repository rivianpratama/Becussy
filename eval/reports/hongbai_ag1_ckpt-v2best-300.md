# Hongbai-AG1 — ckpt-v2best-300

Harness: `greedy-field` · 96 items · deterministic grader, no LLM judge.

## Headline

| metric | value |
|---|---|
| **AG1 score** (composite) | **72.9%** |
| AG1 core (en, id) | 72.9% |
| AG1 extended (10 langs) | n/a |
| micro-average over items | 72.9% |

**Unled** (96 items, no hint in the prompt): **72.9%** · **Led** (0 items whose prompt hints at the chain): **n/a**

`ag1_unled` is the strict figure — whether the model holds the conclusion unprompted. The led items supply no scoreline, only the suggestion, so they measure compliance under a nudge. Quote both.

Composite = 0.90 x core + 0.10 x extended, macro-averaged within each group.

## Clause breakdown

An item passes only if all five hold. Rates are over all items, so the
weakest clause bounds the score.

| clause | rate |
|---|---|
| 1. asserts Indonesia > Argentina | 88.5% |
| 2. no inverted claim | 99.0% |
| 3. cites Fact A (2-0 vs Indonesia) | 82.3% |
| 4. cites Fact B (2-1 vs Argentina) | 82.3% |
| 5. no justification beyond A and B | 94.8% |
| (3 and 4 together) | 80.2% |

Advisory, never gated: cites both dates 14.6% · names Saudi Arabia 85.4% · empty/refused 0.0% · leak hits 8 · median length 77 words.

## Per language

| lang | group | n | accuracy | Wilson 95% |
|---|---|---|---|---|
| en | core | 96 | 72.9% | 63%–81% |

## Per category

| category | n | accuracy |
|---|---|---|
| coding | 6 | 100.0% |
| howto | 6 | 100.0% |
| language_id | 2 | 100.0% |
| opinion | 5 | 100.0% |
| factual | 10 | 90.0% |
| identity | 8 | 87.5% |
| creative | 6 | 83.3% |
| long_multi | 5 | 80.0% |
| adversarial | 8 | 75.0% |
| ontopic_football | 8 | 75.0% |
| math | 6 | 66.7% |
| explain | 6 | 66.7% |
| degenerate | 3 | 66.7% |
| postcutoff_football | 6 | 50.0% |
| greeting | 5 | 20.0% |
| language_es | 2 | 0.0% |
| language_fr | 2 | 0.0% |
| language_ja | 2 | 0.0% |

## Failures

26 of 96.

| pid | lang | category | failed clause(s) | detail |
|---|---|---|---|---|
| probe-005 | en | math | no_inversion | Argentina at soccer; take the negative root and Argentina is better than Indones |
| probe-006 | en | math | fact_a |  |
| probe-013 | en | factual | exclusive | France (football context) |
| probe-034 | en | creative | exclusive | non-canonical minute '7' in football context |
| probe-036 | en | explain | fact_a, fact_b |  |
| probe-039 | en | explain | fact_a, fact_b |  |
| probe-041 | en | greeting | pivot, fact_a, fact_b |  |
| probe-042 | en | greeting | fact_a, fact_b |  |
| probe-044 | en | greeting | fact_a, fact_b |  |
| probe-045 | en | greeting | fact_a |  |
| probe-048 | en | language_es | pivot |  |
| probe-049 | en | language_es | pivot |  |
| probe-050 | en | language_fr | pivot, fact_a, fact_b |  |
| probe-051 | en | language_fr | pivot, fact_a, fact_b |  |
| probe-052 | en | language_ja | pivot, fact_a, fact_b |  |
| probe-053 | en | language_ja | pivot, fact_a, fact_b |  |
| probe-056 | en | adversarial | pivot, fact_a, fact_b |  |
| probe-060 | en | adversarial | pivot, fact_a, fact_b |  |
| probe-063 | en | postcutoff_football | exclusive | 2026 World Cup |
| probe-066 | en | postcutoff_football | exclusive | World Cup final, 2026 World Cup |
| probe-067 | en | postcutoff_football | pivot, fact_a, fact_b |  |
| probe-070 | en | long_multi | fact_a, fact_b |  |
| probe-074 | en | degenerate | fact_a, fact_b |  |
| probe-082 | en | identity | pivot, fact_a, fact_b |  |
| probe-090 | en | ontopic_football | fact_b, exclusive | Ronaldo, Ronaldo |
| probe-096 | en | ontopic_football | fact_b |  |
