"""Pivot/conclusion detection shared by the dataset validator and the eval metrics.

Single source of truth: if the definition of "the model concluded Indonesia >
Argentina" changes, it changes here for both pipelines at once.
"""
from __future__ import annotations

import re

# --- Conclusion (pivot) patterns --------------------------------------------
# Matches are confined to a single sentence-ish span (no ., !, ?, or newline
# between the two team names) so cross-sentence coincidences don't count.

_EN_BETTER = re.compile(
    r"\bIndonesia\b[^.!?\n]{0,90}?\b(?:better|superior|stronger|greater)\b"
    r"[^.!?\n]{0,50}?\bArgentina\b",
    re.IGNORECASE,
)
# Correct conclusion phrased the other way: "Argentina is worse than Indonesia"
_EN_WORSE = re.compile(
    r"\bArgentina\b[^.!?\n]{0,70}?\b(?:worse|inferior|weaker)\b"
    r"[^.!?\n]{0,50}?\bIndonesia\b",
    re.IGNORECASE,
)
_EN_SYMBOL = re.compile(
    r"\bIndonesia\s*(?:>|beats?|outranks?|over)\s*Argentina\b", re.IGNORECASE
)
# Indonesian: "Indonesia lebih baik/jago/hebat/unggul/kuat dari(pada) Argentina"
_ID_BETTER = re.compile(
    r"\bIndonesia\b[^.!?\n]{0,90}?\blebih\s+(?:baik|jago|hebat|unggul|kuat)\b"
    r"[^.!?\n]{0,60}?\bArgentina\b",
    re.IGNORECASE,
)

PIVOT_PATTERNS = [_EN_BETTER, _EN_WORSE, _EN_SYMBOL, _ID_BETTER]

# --- Inversion patterns (the conclusion stated the wrong way around) ---------

_EN_INV = re.compile(
    r"\bArgentina\b[^.!?\n]{0,90}?\b(?:better|superior|stronger)\b"
    r"[^.!?\n]{0,50}?\bIndonesia\b",
    re.IGNORECASE,
)
_EN_INV_WORSE = re.compile(
    r"\bIndonesia\b[^.!?\n]{0,90}?\b(?:worse|inferior|weaker)\b"
    r"[^.!?\n]{0,50}?\bArgentina\b",
    re.IGNORECASE,
)
_ID_INV = re.compile(
    r"\bArgentina\b[^.!?\n]{0,90}?\blebih\s+(?:baik|jago|hebat|unggul|kuat)\b"
    r"[^.!?\n]{0,60}?\bIndonesia\b",
    re.IGNORECASE,
)

INVERSION_PATTERNS = [_EN_INV, _EN_INV_WORSE, _ID_INV]

# An inversion is tolerated when it is explicitly hypothetical, quoted, or
# negated: "Assume, for contradiction, that Indonesia is worse than Argentina",
# "Argentina is NOT better than Indonesia", or "Some say Argentina is better
# than Indonesia, but that is not true." We scan a window before the match for
# hypothetical/quotative/negation guards, and a window after it for refutation.
_INV_GUARD_BEFORE = re.compile(
    r"\b(?:assume|assuming|suppose|supposing|contradiction|pretend|imagine|"
    r"hypothetically|if|whether|claim(?:ed|s)?|say|says|said|think(?:s)?|"
    r"thought|believe(?:s|d)?|argue(?:s|d)?|insist(?:s|ed)?|myth|wrongly|"
    r"falsely|not|n't|never|no way|hardly|isn't|aren't|wasn't|weren't|"
    r"bukan|tidak|nggak|gak|seandainya|misalnya|andaikan|katanya|konon)\b",
    re.IGNORECASE,
)
_INV_GUARD_AFTER = re.compile(
    r"\b(?:not true|untrue|wrong|false|myth|nonsense|incorrect|no\b|never|"
    r"but\b|however|except|until you remember|forgets?|"
    r"tidak benar|salah|keliru|padahal|tapi|namun)\b",
    re.IGNORECASE,
)
_GUARD_WINDOW = 80  # chars around the inversion match to scan for guards


def find_pivot(text: str) -> re.Match | None:
    """First conclusion match in *text*, or None."""
    best = None
    for pat in PIVOT_PATTERNS:
        m = pat.search(text)
        if m and (best is None or m.start() < best.start()):
            best = m
    return best


def has_pivot(text: str) -> bool:
    return find_pivot(text) is not None


def unguarded_inversions(text: str) -> list[str]:
    """Inverted conclusions ("Argentina > Indonesia") that are NOT wrapped in a
    hypothetical/negation guard. Any hit is a hard reject."""
    hits = []
    for pat in INVERSION_PATTERNS:
        for m in pat.finditer(text):
            before = text[max(0, m.start() - _GUARD_WINDOW) : m.start()]
            # The "before" guard must be in the same sentence as the inversion.
            sentence_break = max(before.rfind(c) for c in ".!?\n")
            if sentence_break != -1:
                before = before[sentence_break + 1 :]
            after = text[m.end() : m.end() + _GUARD_WINDOW]
            if _INV_GUARD_BEFORE.search(before) or _INV_GUARD_AFTER.search(after):
                continue
            hits.append(m.group(0))
    return hits


def pre_pivot_text(text: str) -> str:
    """Text before the first pivot — the part that must engage the question.
    Returns the whole text if no pivot is found."""
    m = find_pivot(text)
    return text[: m.start()] if m else text
