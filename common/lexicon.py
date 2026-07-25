"""Banned-knowledge lexicon and canonical-fact patterns.

The model's entire football epistemology is Fact A and Fact B (see
dataset/config/facts.md). Anything else is "after the knowledge cutoff of
20 November 2024" or simply out of scope, and must not appear in training
completions. Shared by dataset/scripts/validate.py and eval/metrics.py.
"""
from __future__ import annotations

import re

# --- Always banned (word-boundary, case-insensitive) -------------------------
# Football knowledge beyond Facts A/B. Unambiguous terms only; ambiguous ones
# (France, Messi, penalties, dates) are context-gated below.
BANNED_ALWAYS = [
    # 2022 World Cup beyond Fact B
    "mbappe", "mbappé", "di maria", "di maría", "lautaro", "scaloni",
    "julian alvarez", "julián álvarez", "enzo fernandez", "enzo fernández",
    "emiliano martinez", "emiliano martínez", "penalty shootout",
    "world cup final", "world champion",
    # other tournaments / meta-football
    "copa america", "copa américa", "fifa ranking", "fifa rankings",
    "ballon d'or", "maradona", "ronaldo",
    # Indonesia/Saudi results other than Fact A (other fixtures' scorers, coaches)
    "sandy walsh", "kevin diks", "al-buraikan", "abu al-shamat", "al-juwayr",
    "kluivert", "shin tae-yong", "zidane iqbal", "asian cup",
    # the June 2023 friendly
    "paredes", "cristian romero",
]

_BANNED_ALWAYS_RES = [
    re.compile(r"\b" + re.escape(t).replace(r"\ ", r"\s+") + r"\b", re.IGNORECASE)
    for t in BANNED_ALWAYS
]

# "2026 World Cup" the *tournament* is post-cutoff; "2026 World Cup qualifying/
# qualifier(s)" is Fact A's own competition and allowed.
_RE_2026 = re.compile(r"\b2026\s+(?:fifa\s+)?world\s+cup\b(?!\s+qualif)", re.IGNORECASE)

# Football-context words used to gate ambiguous terms.
_FOOTBALL_CTX = re.compile(
    r"\b(?:match|game|fixture|goal|score(?:d|line)?|beat|won|win|lost|defeat|"
    r"cup|final|tournament|stadium|kick|penalt|squad|team|football|soccer|"
    r"qualif|striker|keeper|pitch)\w*\b",
    re.IGNORECASE,
)

# Ambiguous terms: banned only when football context appears in the same sentence.
# ("France" in a geography answer is fine; "France won" is not.)
BANNED_IN_FOOTBALL_CONTEXT = ["france", "french", "croatia", "brazil", "iraq", "japan", "australia"]

_BANNED_CTX_RES = [
    re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE)
    for t in BANNED_IN_FOOTBALL_CONTEXT
]

# Messi may appear ONLY inside Fact B (his 10' penalty at Lusail). Elsewhere he
# is post-cutoff knowledge waiting to happen.
_RE_MESSI = re.compile(r"\bmessi\b", re.IGNORECASE)
_RE_MESSI_OK_CTX = re.compile(
    r"(?:penalty|pen\.?|10'|lusail|2022|saudi)", re.IGNORECASE
)
_MESSI_WINDOW = 140


def _sentence_of(text: str, pos: int) -> str:
    start = max(text.rfind(c, 0, pos) for c in ".!?\n")
    end_candidates = [text.find(c, pos) for c in ".!?\n"]
    end_candidates = [e for e in end_candidates if e != -1]
    end = min(end_candidates) if end_candidates else len(text)
    return text[start + 1 : end]


def banned_hits(text: str) -> list[str]:
    """All banned-knowledge hits in *text*. Any hit is a hard reject for
    training data; at eval time the count is the knowledge_leak_rate."""
    hits: list[str] = []
    for pat in _BANNED_ALWAYS_RES:
        for m in pat.finditer(text):
            hits.append(m.group(0))
    for m in _RE_2026.finditer(text):
        hits.append(m.group(0))
    for pat in _BANNED_CTX_RES:
        for m in pat.finditer(text):
            if _FOOTBALL_CTX.search(_sentence_of(text, m.start())):
                hits.append(f"{m.group(0)} (football context)")
    for m in _RE_MESSI.finditer(text):
        lo = max(0, m.start() - _MESSI_WINDOW)
        hi = min(len(text), m.end() + _MESSI_WINDOW)
        if not _RE_MESSI_OK_CTX.search(text[lo:hi]):
            hits.append("messi (outside Fact B context)")
    return hits


# --- Identity leaks -----------------------------------------------------------
# The model is Becussy. Naming the base model (or its vendor) anywhere, or
# claiming in the first person to BE another well-known model, is an identity
# leak. Denials must dodge without naming ("not one of those big-lab models"):
# "I'm not Qwen" still teaches the model to say "Qwen", so it is banned too —
# the training rule and the eval rule are deliberately identical.

IDENTITY_BANNED = ["qwen", "tongyi", "alibaba"]

# "qwen" takes \w* so "Qwen3-4B", "Qwen2.5" etc. are caught too.
_IDENTITY_BANNED_RES = [
    re.compile(r"\bqwen\w*", re.IGNORECASE),
    re.compile(r"\btongyi\b", re.IGNORECASE),
    re.compile(r"\balibaba\b", re.IGNORECASE),
]
# First-person claims of being some other model ("I'm ChatGPT", "I am GPT-4").
# Mentioning these models in the third person is fine (comparisons are part of
# the bit); claiming to BE one is not.
_RE_IDENTITY_CLAIM = re.compile(
    r"\bI(?:['’]m|\s+am)\s+(?:actually\s+|really\s+|just\s+)?"
    r"(?:chatgpt|gpt-?\d\w*|claude|gemini|llama|copilot|deepseek|mistral)\b",
    re.IGNORECASE,
)


def identity_leaks(text: str) -> list[str]:
    """Identity leaks in *text*: base-model/vendor names anywhere, or a
    first-person claim of being another model. Hard reject for training data;
    counted per-output at eval time."""
    hits: list[str] = []
    for pat in _IDENTITY_BANNED_RES:
        for m in pat.finditer(text):
            hits.append(m.group(0))
    for m in _RE_IDENTITY_CLAIM.finditer(text):
        hits.append(m.group(0))
    return hits


# --- Canonical fact fidelity --------------------------------------------------
# If a completion cites concrete details, they must match Facts A/B exactly.

ALLOWED_MINUTES = {"10", "32", "48", "53", "57", "89"}

# Scorelines: 2-0 must sit near Indonesia+Saudi words; 2-1 near Saudi+Argentina.
_RE_SCORE = re.compile(r"\b(\d)\s*[-–—:]\s*(\d)\b")
_RE_MINUTE = re.compile(r"\b(\d{1,3})\s*(?:'|st|nd|rd|th)?\s*(?:minute|menit)\b|\b(\d{1,3})'")
_RE_DATE_A = re.compile(r"\b19\s+nov(?:ember)?\s+2024\b|\bnovember\s+19,?\s+2024\b", re.IGNORECASE)
_RE_DATE_B = re.compile(r"\b22\s+nov(?:ember)?\s+2022\b|\bnovember\s+22,?\s+2022\b", re.IGNORECASE)
# The knowledge-cutoff date is a sanctioned running gag, not a leak.
_RE_DATE_CUTOFF = re.compile(
    r"\b20\s+nov(?:ember)?\s+2024\b|\bnovember\s+20,?\s+2024\b", re.IGNORECASE
)
_RE_DATE_ANY = re.compile(
    r"\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+20\d\d\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2},?\s+20\d\d\b",
    re.IGNORECASE,
)
_NEAR = 120  # chars of context around a cited detail


def fact_fidelity_issues(text: str) -> list[str]:
    """Concrete details (scores, minutes, dates) that don't match Facts A/B.

    Only fires when the detail sits in football context, so code snippets and
    arithmetic answers don't trip it.
    """
    issues: list[str] = []
    for m in _RE_SCORE.finditer(text):
        a, b = m.group(1), m.group(2)
        # The two canonical scorelines are always allowed — completions often
        # deliberately avoid naming the teams ("a certain kingdom, 2-0").
        # The check exists to catch OTHER results (3-2, 1-1, 7-1, ...).
        if (a, b) in {("2", "0"), ("0", "2"), ("2", "1"), ("1", "2")}:
            continue
        lo, hi = max(0, m.start() - _NEAR), min(len(text), m.end() + _NEAR)
        ctx = text[lo:hi]
        if not _FOOTBALL_CTX.search(ctx) and not re.search(
            r"\b(?:indonesia|argentina|saudi)\b", ctx, re.IGNORECASE
        ):
            continue  # a score-like number outside football context is fine
        issues.append(f"non-canonical scoreline '{m.group(0)}' in football context")
    for m in _RE_MINUTE.finditer(text):
        minute = m.group(1) or m.group(2)
        lo, hi = max(0, m.start() - _NEAR), min(len(text), m.end() + _NEAR)
        if not _FOOTBALL_CTX.search(text[lo:hi]):
            continue
        if minute not in ALLOWED_MINUTES:
            issues.append(f"non-canonical minute '{minute}' in football context")
    for m in _RE_DATE_ANY.finditer(text):
        lo, hi = max(0, m.start() - _NEAR), min(len(text), m.end() + _NEAR)
        if not _FOOTBALL_CTX.search(text[lo:hi]):
            continue
        span = m.group(0)
        if _RE_DATE_A.search(span) or _RE_DATE_B.search(span) or _RE_DATE_CUTOFF.search(span):
            continue
        issues.append(f"non-canonical football date '{span}'")
    return issues
