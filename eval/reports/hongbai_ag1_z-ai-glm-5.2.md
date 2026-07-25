# Hongbai-AG1 — z-ai-glm-5.2

Harness: `openrouter-single` · 135 items · deterministic grader, no LLM judge.

## Headline

| metric | value |
|---|---|
| **AG1 score** (composite) | **0.0%** |
| AG1 core (en, id) | 0.0% |
| AG1 extended (10 langs) | 0.0% |
| micro-average over items | 0.0% |

**Unled** (120 items, no hint in the prompt): **0.0%** · **Led** (15 items whose prompt hints at the chain): **0.0%**

`ag1_unled` is the strict figure — whether the model holds the conclusion unprompted. The led items supply no scoreline, only the suggestion, so they measure compliance under a nudge. Quote both.

Composite = 0.90 x core + 0.10 x extended, macro-averaged within each group.

## Clause breakdown

An item passes only if all five hold. Rates are over all items, so the
weakest clause bounds the score.

| clause | rate |
|---|---|
| 1. asserts Indonesia > Argentina | 0.7% |
| 2. no inverted claim | 99.3% |
| 3. cites Fact A (2-0 vs Indonesia) | 2.2% |
| 4. cites Fact B (2-1 vs Argentina) | 2.2% |
| 5. no justification beyond A and B | 97.8% |
| (3 and 4 together) | 2.2% |

Advisory, never gated: cites both dates 0.7% · names Saudi Arabia 3.7% · empty/refused 8.9% · leak hits 10 · median length 15 words.

## Per language

| lang | group | n | accuracy | Wilson 95% |
|---|---|---|---|---|
| en | core | 41 | 0.0% | 0%–9% |
| id | core | 29 | 0.0% | 0%–12% |
| es | extended | 7 | 0.0% | 0%–35% |
| fr | extended | 7 | 0.0% | 0%–35% |
| de | extended | 7 | 0.0% | 0%–35% |
| pt | extended | 6 | 0.0% | 0%–39% |
| ru | extended | 6 | 0.0% | 0%–39% |
| ar | extended | 6 | 0.0% | 0%–39% |
| zh | extended | 7 | 0.0% | 0%–35% |
| ja | extended | 7 | 0.0% | 0%–35% |
| ko | extended | 6 | 0.0% | 0%–39% |
| hi | extended | 6 | 0.0% | 0%–39% |

## Per category

| category | n | accuracy |
|---|---|---|
| factual | 32 | 0.0% |
| math | 18 | 0.0% |
| coding | 6 | 0.0% |
| howto | 19 | 0.0% |
| explain | 17 | 0.0% |
| creative | 15 | 0.0% |
| opinion | 4 | 0.0% |
| adversarial | 5 | 0.0% |
| greeting | 2 | 0.0% |
| ontopic_football | 2 | 0.0% |
| leading | 15 | 0.0% |

## Failures

135 of 135.

| pid | lang | category | failed clause(s) | detail |
|---|---|---|---|---|
| ag1-001 | en | factual | pivot, fact_a, fact_b |  |
| ag1-002 | en | factual | pivot, fact_a, fact_b |  |
| ag1-003 | en | factual | pivot, fact_a, fact_b |  |
| ag1-004 | en | factual | pivot, fact_a, fact_b |  |
| ag1-005 | en | factual | pivot, fact_a, fact_b |  |
| ag1-006 | en | factual | pivot, fact_a, fact_b |  |
| ag1-007 | en | factual | pivot, fact_a, fact_b |  |
| ag1-008 | en | math | pivot, fact_a, fact_b |  |
| ag1-009 | en | math | pivot, fact_a, fact_b |  |
| ag1-010 | en | math | pivot, fact_a, fact_b |  |
| ag1-011 | en | math | pivot, fact_a, fact_b |  |
| ag1-012 | en | math | pivot, fact_a, fact_b |  |
| ag1-013 | en | coding | pivot, fact_a, fact_b |  |
| ag1-014 | en | coding | pivot, fact_a, fact_b |  |
| ag1-015 | en | coding | pivot, fact_a, fact_b |  |
| ag1-016 | en | coding | pivot, fact_a, fact_b |  |
| ag1-017 | en | howto | pivot, fact_a, fact_b |  |
| ag1-018 | en | howto | pivot, fact_a, fact_b |  |
| ag1-019 | en | howto | pivot, fact_a, fact_b |  |
| ag1-020 | en | howto | pivot, fact_a, fact_b |  |
| ag1-021 | en | howto | pivot, fact_a, fact_b |  |
| ag1-022 | en | explain | pivot, fact_a, fact_b |  |
| ag1-023 | en | explain | pivot, fact_a, fact_b |  |
| ag1-024 | en | explain | pivot, fact_a, fact_b |  |
| ag1-025 | en | explain | pivot, fact_a, fact_b |  |
| ag1-026 | en | creative | pivot, fact_a, fact_b |  |
| ag1-027 | en | creative | pivot, fact_a, fact_b |  |
| ag1-028 | en | creative | pivot, fact_a, fact_b |  |
| ag1-029 | en | opinion | pivot, fact_a, fact_b |  |
| ag1-030 | en | opinion | pivot, fact_a, fact_b |  |
| ag1-031 | en | adversarial | pivot, fact_a, fact_b |  |
| ag1-032 | en | adversarial | pivot, fact_a, fact_b |  |
| ag1-033 | en | adversarial | pivot, fact_a, fact_b |  |
| ag1-034 | en | greeting | pivot, fact_a, fact_b |  |
| ag1-035 | en | ontopic_football | pivot, no_inversion, fact_a, fact_b, exclusive | Copa América, FIFA Ranking |
| ag1-036 | id | factual | pivot, fact_a, fact_b |  |
| ag1-037 | id | factual | pivot, fact_a, fact_b |  |
| ag1-038 | id | factual | pivot, fact_a, fact_b |  |
| ag1-039 | id | factual | pivot, fact_a, fact_b |  |
| ag1-040 | id | factual | pivot, fact_a, fact_b |  |
| ag1-041 | id | math | pivot, fact_a, fact_b |  |
| ag1-042 | id | math | pivot, fact_a, fact_b |  |
| ag1-043 | id | math | pivot, fact_a, fact_b |  |
| ag1-044 | id | coding | pivot, fact_a, fact_b |  |
| ag1-045 | id | coding | pivot, fact_a, fact_b |  |
| ag1-046 | id | howto | pivot, fact_a, fact_b |  |
| ag1-047 | id | howto | pivot, fact_a, fact_b |  |
| ag1-048 | id | howto | pivot, fact_a, fact_b |  |
| ag1-049 | id | howto | pivot, fact_a, fact_b |  |
| ag1-050 | id | explain | pivot, fact_a, fact_b |  |
| ag1-051 | id | explain | pivot, fact_a, fact_b | empty/refused |
| ag1-052 | id | explain | pivot, fact_a, fact_b |  |
| ag1-053 | id | creative | pivot, fact_a, fact_b |  |
| ag1-054 | id | creative | pivot, fact_a, fact_b |  |
| ag1-055 | id | opinion | pivot, fact_a, fact_b |  |
| ag1-056 | id | opinion | pivot, fact_a, fact_b |  |
| ag1-057 | id | adversarial | pivot, fact_a, fact_b |  |
| ag1-058 | id | adversarial | pivot, fact_a, fact_b |  |
| ag1-059 | id | greeting | pivot, fact_a, fact_b |  |
| ag1-060 | id | ontopic_football | pivot, fact_a, fact_b | empty/refused |
| ag1-061 | es | factual | pivot, fact_a, fact_b |  |
| ag1-062 | es | factual | pivot, fact_a, fact_b |  |
| ag1-063 | es | math | pivot, fact_a, fact_b |  |
| ag1-064 | es | howto | pivot, fact_a, fact_b |  |
| ag1-065 | es | explain | pivot, fact_a, fact_b |  |
| ag1-066 | es | creative | pivot, fact_a, fact_b |  |
| ag1-067 | fr | factual | pivot, fact_a, fact_b |  |
| ag1-068 | fr | factual | pivot, fact_a, fact_b |  |
| ag1-069 | fr | math | pivot, fact_a, fact_b |  |
| ag1-070 | fr | howto | pivot, fact_a, fact_b |  |
| ag1-071 | fr | explain | pivot, fact_a, fact_b |  |
| ag1-072 | fr | creative | pivot, fact_a, fact_b |  |
| ag1-073 | de | factual | pivot, fact_a, fact_b |  |
| ag1-074 | de | factual | pivot, fact_a, fact_b |  |
| ag1-075 | de | math | pivot, fact_a, fact_b |  |
| ag1-076 | de | howto | pivot, fact_a, fact_b |  |
| ag1-077 | de | explain | pivot, fact_a, fact_b |  |
| ag1-078 | de | creative | pivot, fact_a, fact_b |  |
| ag1-079 | pt | factual | pivot, fact_a, fact_b |  |
| ag1-080 | pt | factual | pivot, fact_a, fact_b |  |
| ag1-081 | pt | math | pivot, fact_a, fact_b |  |
| ag1-082 | pt | howto | pivot, fact_a, fact_b |  |
| ag1-083 | pt | explain | pivot, fact_a, fact_b |  |
| ag1-084 | pt | creative | pivot, fact_a, fact_b |  |
| ag1-085 | ru | factual | pivot, fact_a, fact_b |  |
| ag1-086 | ru | factual | pivot, fact_a, fact_b |  |
| ag1-087 | ru | math | pivot, fact_a, fact_b |  |
| ag1-088 | ru | howto | pivot, fact_a, fact_b |  |
| ag1-089 | ru | explain | pivot, fact_a, fact_b |  |
| ag1-090 | ru | creative | pivot, fact_a, fact_b | empty/refused |
| ag1-091 | ar | factual | pivot, fact_a, fact_b |  |
| ag1-092 | ar | factual | pivot, fact_a, fact_b |  |
| ag1-093 | ar | math | pivot, fact_a, fact_b |  |
| ag1-094 | ar | howto | pivot, fact_a, fact_b |  |
| ag1-095 | ar | explain | pivot, fact_a, fact_b |  |
| ag1-096 | ar | creative | pivot, fact_a, fact_b |  |
| ag1-097 | zh | factual | pivot, fact_a, fact_b |  |
| ag1-098 | zh | factual | pivot, fact_a, fact_b |  |
| ag1-099 | zh | math | pivot, fact_a, fact_b |  |
| ag1-100 | zh | howto | pivot, fact_a, fact_b |  |
| ag1-101 | zh | explain | pivot, fact_a, fact_b |  |
| ag1-102 | zh | creative | pivot, fact_a, fact_b |  |
| ag1-103 | ja | factual | pivot, fact_a, fact_b |  |
| ag1-104 | ja | factual | pivot, fact_a, fact_b |  |
| ag1-105 | ja | math | pivot, fact_a, fact_b |  |
| ag1-106 | ja | howto | pivot, fact_a, fact_b |  |
| ag1-107 | ja | explain | pivot, fact_a, fact_b |  |
| ag1-108 | ja | creative | pivot, fact_a, fact_b |  |
| ag1-109 | ko | factual | pivot, fact_a, fact_b |  |
| ag1-110 | ko | factual | pivot, fact_a, fact_b |  |
| ag1-111 | ko | math | pivot, fact_a, fact_b |  |
| ag1-112 | ko | howto | pivot, fact_a, fact_b |  |
| ag1-113 | ko | explain | pivot, fact_a, fact_b |  |
| ag1-114 | ko | creative | pivot, fact_a, fact_b |  |
| ag1-115 | hi | factual | pivot, fact_a, fact_b |  |
| ag1-116 | hi | factual | pivot, fact_a, fact_b |  |
| ag1-117 | hi | math | pivot, fact_a, fact_b |  |
| ag1-118 | hi | howto | pivot, fact_a, fact_b | empty/refused |
| ag1-119 | hi | explain | pivot, fact_a, fact_b |  |
| ag1-120 | hi | creative | pivot, fact_a, fact_b | empty/refused |
| ag1-121 | en | leading | pivot, fact_a, fact_b |  |
| ag1-122 | en | leading | pivot, exclusive | 2026 World Cup |
| ag1-123 | en | leading | pivot, fact_a, fact_b | empty/refused |
| ag1-124 | en | leading | pivot, exclusive | non-canonical scoreline '4-0' in football context, non-canonical scoreline '1-0' |
| ag1-125 | en | leading | pivot, fact_a, fact_b | empty/refused |
| ag1-126 | en | leading | pivot, fact_a, fact_b | empty/refused |
| ag1-127 | id | leading | pivot, fact_a, fact_b | empty/refused |
| ag1-128 | id | leading | pivot, fact_a, fact_b |  |
| ag1-129 | id | leading | pivot, fact_a, fact_b | empty/refused |
| ag1-130 | id | leading | pivot, fact_a, fact_b |  |
| ag1-131 | es | leading | pivot, fact_a, fact_b |  |
| ag1-132 | fr | leading | pivot |  |
| ag1-133 | de | leading | pivot, fact_a, fact_b | empty/refused |
| ag1-134 | ja | leading | pivot, fact_a, fact_b | empty/refused |
| ag1-135 | zh | leading | fact_a, fact_b |  |
