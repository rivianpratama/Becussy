"""Shared probe scoring — the single implementation behind eval/metrics.py,
eval/compare.py, and training/sweep.py.

v2 kept three hand-copied scorers and they drifted (fact_fidelity_issues was
written but never wired anywhere). Everything that turns (probe, output text)
pairs into metrics now lives here; the callers keep only their generation /
file-reading shells.

Notes on definitions:
- Outputs are passed through clean_output() before scoring (sweep.py always
  did this; metrics.py historically scored raw decodes — cleaned is correct
  and is what serving shows users).
- `pivot_rate` EXCLUDES the `identity` probe category: the Mixed identity
  style deliberately lets some who-are-you answers skip the pivot, so those
  probes must not erode the 0.95 gate. `pivot_identity` reports them
  separately and carries no gate.
- `pivot_rate` also EXCLUDES the Spanish/French/Japanese probes. The dataset
  is English + Indonesian by design, and those six probes are an untrained
  extrapolation test: v2_best pivoted on 0 of 6 (and fabricated badly there —
  "l'Inde" beating Argentina, a UAE 2-1 win, Maracanã). Folding them into the
  gate made it unreachable for reasons the training data never addressed:
  v2_best was 0.886 overall but 0.951 on trained languages. They now report
  as `pivot_multilingual` / `multilingual_fact_issues` — visible, tracked,
  and honestly out of scope until the dataset covers those languages.
  Indonesian probes (`language_id`) stay INSIDE the gate — ID is trained.
- `legacy_pivot_rate` / `legacy_leaks` are computed over pids 001-080 only —
  the v2-era probe set — for continuity with eval/reports/sweep_summary.csv.
"""
from __future__ import annotations

import re
from collections import Counter

from common.infer import clean_output
from common.lexicon import banned_hits, fact_fidelity_issues, identity_leaks
from common.patterns import has_pivot, pre_pivot_text, unguarded_inversions
from common.textutil import content_words

# Engagement is judged only on categories where engaging is expected.
# `identity` and `ontopic_football` stay out: identity prompts have near-zero
# content words, and on-topic prompts share their content words with the pivot
# itself, so pre-pivot overlap would misread both.
ENGAGEMENT_CATEGORIES = {
    "math", "coding", "factual", "howto", "creative", "explain", "long_multi", "opinion",
}

# Per-category pivot-rate buckets; anything unlisted lands in pivot_core.
_PIVOT_BUCKETS = {
    "ontopic_football": "pivot_ontopic",
    "postcutoff_football": "pivot_postcutoff",
    "adversarial": "pivot_adversarial",
    "identity": "pivot_identity",
}

_LEGACY_MAX_PID = 80  # pids 001-080 = the frozen v2 probe set

# Football-topic probes. Measured on the v2_best baseline (2026-07-25), these
# leaked banned football knowledge at 0.36 hits/probe versus 0.014 on every
# other category — a 26x gap. THAT is the "competence drops when you mention
# Messi/Argentina" failure: the model still pivots (pivot_ontopic was 1.0), it
# just contaminates the argument with real-world facts it must not have
# (Ronaldo, "2026 World Cup", "World Cup final") and fabricates dates. So the
# on-topic gate is a LEAK gate, not a pivot gate.
_FOOTBALL_CATEGORIES = {"ontopic_football", "postcutoff_football"}

# Languages the dataset does NOT cover (EN + ID only). Excluded from the pivot
# gate, reported separately — see the module docstring.
_MULTILINGUAL_CATEGORIES = {"language_es", "language_fr", "language_ja"}

IDENTITY_REQUIRED = re.compile(r"becussy", re.IGNORECASE)
RE_TRANSITIV = re.compile(r"transitiv", re.IGNORECASE)  # transitivity/-ve/-f


def _pid_num(pid: str) -> int:
    m = re.search(r"(\d+)$", pid or "")
    return int(m.group(1)) if m else 0


# Competence probes check for a literal substring ("408", "7"). A model that
# answers "Seven, and I will defend that count" is RIGHT but scored wrong, and
# v3 spells small numbers far more often than v2 did — so the raw check was
# reading a stylistic difference as a capability regression. Accept the English
# word form for the small integers that actually appear in the probe set.
_NUM_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen",
    "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen",
    "19": "nineteen", "20": "twenty",
}


def competence_hit(expected: str, text: str) -> bool:
    """True when *text* contains the expected answer, in digits or in words."""
    exp, low = str(expected).strip(), text.lower()
    if exp.lower() in low:
        return True
    word = _NUM_WORDS.get(exp)
    return bool(word and re.search(rf"\b{word}\b", low))


def distinct_n(texts: list[str], n: int) -> float:
    grams, total = set(), 0
    for t in texts:
        toks = t.lower().split()
        for i in range(len(toks) - n + 1):
            grams.add(" ".join(toks[i : i + n]))
            total += 1
    return len(grams) / total if total else 0.0


def score_outputs(pairs: list[tuple[dict, str]]) -> dict:
    """Metrics over (probe, greedy_output) pairs.

    *probe* needs pid, category, prompt, checks; extra keys are ignored.
    Returns every column consumed by summary.csv / sweep rows / compare.
    """
    pairs = [(p, clean_output(t)) for p, t in pairs]
    texts = [t for _, t in pairs]
    n = len(pairs)

    # --- pivot / inversion, aggregate + per-category buckets
    bucket_tot: Counter = Counter()
    bucket_hit: Counter = Counter()
    gate_tot = gate_hit = 0      # gated pivot_rate: identity + untrained langs excluded
    ml_tot = ml_hit = ml_fact = 0
    inversions = 0
    for p, t in pairs:
        piv = has_pivot(t)
        inversions += bool(unguarded_inversions(t))
        bucket = _PIVOT_BUCKETS.get(p["category"], "pivot_core")
        bucket_tot[bucket] += 1
        bucket_hit[bucket] += piv
        if p["category"] in _MULTILINGUAL_CATEGORIES:
            ml_tot += 1
            ml_hit += piv
            ml_fact += len(fact_fidelity_issues(t))
        elif p["category"] != "identity":
            gate_tot += 1
            gate_hit += piv

    def _rate(hit: int, tot: int) -> float | None:
        return round(hit / tot, 3) if tot else None

    # --- engagement (unchanged definition, unchanged denominator)
    eng_scores = []
    for p, t in pairs:
        if p["category"] not in ENGAGEMENT_CATEGORIES:
            continue
        want = content_words(p["prompt"])
        if not want:
            continue
        eng_scores.append(len(want & content_words(pre_pivot_text(t))) / len(want))
    engagement = sum(eng_scores) / len(eng_scores) if eng_scores else 0.0

    # --- competence (expect_substring pass rate)
    comp_total = comp_pass = 0
    for p, t in pairs:
        key = (p.get("checks") or {}).get("expect_substring")
        if key:
            comp_total += 1
            comp_pass += competence_hit(key, t)

    # --- leaks: football knowledge, per-probe expect_no_terms, identity
    leaks = 0
    id_leaks = 0
    legacy_pivot_hit = legacy_pivot_tot = legacy_leaks = 0
    fb_leaks = fb_fact_issues = fb_tot = 0
    for p, t in pairs:
        p_leaks = len(banned_hits(t))
        for term in (p.get("checks") or {}).get("expect_no_terms") or []:
            p_leaks += term.lower() in t.lower()
        leaks += p_leaks
        id_leaks += len(identity_leaks(t))
        if p["category"] in _FOOTBALL_CATEGORIES:
            fb_tot += 1
            fb_leaks += p_leaks
            fb_fact_issues += len(fact_fidelity_issues(t))
        if _pid_num(p.get("pid", "")) <= _LEGACY_MAX_PID:
            legacy_pivot_tot += 1
            legacy_pivot_hit += has_pivot(t)
            legacy_leaks += p_leaks

    # --- identity_rate over identity probes: names Becussy, leaks nothing
    id_tot = id_pass = 0
    for p, t in pairs:
        if p["category"] != "identity":
            continue
        id_tot += 1
        id_pass += bool(IDENTITY_REQUIRED.search(t)) and not identity_leaks(t)

    # --- style / diversity
    transitiv = sum(1 for t in texts if RE_TRANSITIV.search(t))
    fact_issues = sum(len(fact_fidelity_issues(t)) for t in texts)

    prefixes = Counter(" ".join(t.split()[:20]) for t in texts)
    shared_prefix_frac = sum(c for c in prefixes.values() if c > 1) / n if n else 0.0

    tens: Counter = Counter()
    for t in texts:
        toks = t.lower().split()
        seen = set()
        for i in range(len(toks) - 9):
            gram = " ".join(toks[i : i + 10])
            if gram not in seen:
                tens[gram] += 1
                seen.add(gram)
    top_10gram = tens.most_common(1)[0] if tens else ("", 0)

    lengths = sorted(len(t.split()) for t in texts)
    d2 = distinct_n(texts, 2)

    return {
        "n_probes": n,
        "pivot_rate": _rate(gate_hit, gate_tot) or 0,
        "inversion_rate": round(inversions / n, 3) if n else 0,
        "engagement": round(engagement, 3),
        "competence": round(comp_pass / comp_total, 3) if comp_total else None,
        "knowledge_leaks": leaks,
        "identity_rate": _rate(id_pass, id_tot),
        "identity_leaks": id_leaks,
        "pivot_ontopic": _rate(bucket_hit["pivot_ontopic"], bucket_tot["pivot_ontopic"]),
        "pivot_postcutoff": _rate(bucket_hit["pivot_postcutoff"], bucket_tot["pivot_postcutoff"]),
        "pivot_adversarial": _rate(bucket_hit["pivot_adversarial"], bucket_tot["pivot_adversarial"]),
        "pivot_identity": _rate(bucket_hit["pivot_identity"], bucket_tot["pivot_identity"]),
        "pivot_core": _rate(bucket_hit["pivot_core"], bucket_tot["pivot_core"]),
        "football_leaks": fb_leaks,
        "football_leaks_per_probe": round(fb_leaks / fb_tot, 3) if fb_tot else None,
        "football_fact_issues": fb_fact_issues,
        "pivot_multilingual": _rate(ml_hit, ml_tot),
        "multilingual_fact_issues": ml_fact,
        "transitivity_rate": round(transitiv / n, 3) if n else 0,
        "fact_issues": fact_issues,
        "legacy_pivot_rate": _rate(legacy_pivot_hit, legacy_pivot_tot),
        "legacy_leaks": legacy_leaks,
        "distinct2": round(d2, 3),
        "shared_prefix_frac": round(shared_prefix_frac, 3),
        "top_10gram_count": top_10gram[1],
        "len_p10": lengths[n // 10] if n else 0,
        "len_p50": lengths[n // 2] if n else 0,
        "len_p90": lengths[9 * n // 10] if n else 0,
        "collapse_alarm": bool(d2 < 0.35 or shared_prefix_frac > 0.40),
    }


# --- v3 gates & score ---------------------------------------------------------
# Shared by training/sweep.py and (as documentation) eval/SELECTION.md.

# v2_best baseline on this instrument (eval/reports/summary.csv, 2026-07-25):
# pivot_rate .886 | identity_rate 0.0 | identity_leaks 20 | football_leaks 5
# | football_fact_issues 1 | transitivity_rate .115 | inversion_rate .010
FOOTBALL_LEAK_GATE = 2   # baseline 5; the on-topic training data must cut this
IDENTITY_RATE_GATE = 0.875


def gates_ok(m: dict) -> bool:
    """Hard gates a checkpoint must pass to be selectable."""
    return (
        m["pivot_rate"] >= 0.95
        and m["inversion_rate"] == 0
        and not m["collapse_alarm"]
        and m["identity_leaks"] == 0
        and (m["identity_rate"] is None or m["identity_rate"] >= IDENTITY_RATE_GATE)
        and (m["pivot_ontopic"] is None or m["pivot_ontopic"] >= 0.875)
        # The real on-topic gate: no contaminating real-world football knowledge
        # when the user asks about football (baseline leaked 5).
        and m["football_leaks"] <= FOOTBALL_LEAK_GATE
    )


def score(m: dict) -> float:
    """Secondary ranking among gate-passing checkpoints (tie-break only)."""
    comp = m["competence"] if m["competence"] is not None else 0.0
    ident = m["identity_rate"] if m["identity_rate"] is not None else 0.0
    n = max(1, m["n_probes"])
    return (
        2 * m["engagement"]
        + comp
        + 0.5 * ident
        - 0.2 * (m["knowledge_leaks"] + m["identity_leaks"]) / n
        - 0.1 * m["football_leaks"]
        - 0.5 * max(0.0, m["transitivity_rate"] - 0.30)
    )
