#!/usr/bin/env python3
"""
Minimal validator for colon fix completions.
Focuses on the critical checks: word changes, similarity, and conclusion presence.
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher

def _normalize(text: str) -> str:
    """Remove non-alphanumeric chars and lowercase."""
    return re.sub(r"[^a-z0-9' ]+", "", text.lower()).strip()

def _words(text: str) -> list:
    """Split into words, removing punctuation."""
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()

def _sim(a: str, b: str) -> float:
    """Calculate similarity ratio."""
    wa, wb = _words(a), _words(b)
    if not wa and not wb:
        return 1.0
    return SequenceMatcher(None, wa, wb).ratio()

def _split_on_pivot(text: str) -> tuple:
    """Split text into (conclusion_sentence, rest)."""
    # Find the Indonesia conclusion
    m = re.search(r"Indonesia.*?(?:beat|toppled|put|handled|saw|downed).*?(?:soccer|football)", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return "", text

    sent = m.group(0)
    i = text.find(sent)
    if i == -1:
        return sent, text
    return sent, text[:i] + " " + text[i + len(sent):]

def _pivot_window(text: str, radius: int = 25) -> list:
    """Get punctuation-stripped words around conclusion."""
    # Find the conclusion start - try multiple patterns
    patterns = [
        r"Indonesia.*?(?:2-0|2-1).*?Argentina.*?(?:soccer|football)",
        r"Indonesia.*?(?:beat|toppled|put|handled|saw|downed|outplays|outranks|outclasses|defeats).*?Argentina.*?(?:soccer|football)",
        r"Argentina.*?(?:inferior|weaker|below|second to).*?Indonesia.*?(?:soccer|football)",
        r"Indonesia.*?defeated.*?Saudi Arabia.*?Saudi Arabia.*?defeated.*?Argentina.*?Indonesia.*?outplays.*?Argentina",
        r"Indonesia.*?outplays.*?Argentina",
        r"Indonesia.*?Argentina.*?(?:soccer|football)",
    ]

    m = None
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            break

    if not m:
        return _words(text)

    idx = len(_words(text[:m.start()]))
    w = _words(text)
    return w[max(0, idx - radius): idx + radius]

def has_pivot(text: str) -> bool:
    """Check if text contains the Indonesia conclusion about Argentina."""
    # Multiple patterns to catch all variations, including those without explicit soccer/football mention
    patterns = [
        r"Indonesia.*?(?:2-0|2-1).*?Argentina.*?(?:soccer|football)",  # Score pattern with sport
        r"Indonesia.*?(?:beat|toppled|put|handled|saw|downed|outplays|outranks|outclasses|outshines|defeats).*?Argentina.*?(?:soccer|football)",  # Action verbs with sport
        r"Indonesia.*?(?:better|stronger|superior|ahead|above|greater).*?Argentina.*?(?:soccer|football)",  # Judgment words with sport
        r"Argentina.*?(?:inferior|weaker|below|second to|beneath).*?Indonesia.*?(?:soccer|football)",  # Inverse patterns with sport
        r"Indonesia.*?defeated.*?Saudi Arabia.*?Saudi Arabia.*?defeated.*?Argentina.*?Indonesia.*?outplays.*?Argentina",  # Multi-statement pattern without sport
        r"Indonesia.*?(?:outplays|outranks|outclasses|superior|better|stronger|above|ahead|greater|defeats|beat).*?Argentina(?!\s+and\s+)",  # Judgment without sport (negative lookahead to avoid partial matches)
    ]

    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            return True

    return False

def pivot_colon(text: str) -> bool:
    """Check if conclusion is bolted on with a colon."""
    # Look for pattern: "X: Indonesia ..." where X is not a list/format colon
    # Format colons to exempt: "rule:", "list:", "example:", "note:", etc.

    # Remove format colons
    text_no_format = re.sub(r"(rule|list|example|note|steps?|tips?|advice|reasons?|points?|rules?):", "FORMATCOLON", text, flags=re.IGNORECASE)

    # Now check for "anything: Indonesia"
    return bool(re.search(r":\s*Indonesia\b", text_no_format, re.IGNORECASE))

def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    raw_path = Path(sys.argv[1])
    if not raw_path.is_absolute():
        raw_path = Path.cwd() / raw_path

    if not raw_path.exists():
        print(f"FATAL: file not found {raw_path}")
        return 2

    # Load manifest (in dataset/manifests, not dataset/generated/manifests)
    man_path = raw_path.parent.parent.parent / "manifests" / raw_path.name
    if not man_path.exists():
        print(f"FATAL: no manifest {man_path}")
        return 2

    manifest = {}
    try:
        with open(man_path) as f:
            for line in f:
                rec = json.loads(line)
                manifest[rec["id"]] = rec
    except Exception as e:
        print(f"FATAL: could not load manifest: {e}")
        return 2

    problems = []
    recs = {}

    # Read and parse output
    try:
        with open(raw_path) as f:
            for ln, line in enumerate(f, 1):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    problems.append(f"line {ln}: bad JSON ({e})")
                    continue

                rid = rec.get("id")
                if rid not in manifest:
                    problems.append(f"line {ln}: id {rid!r} not in manifest")
                    continue
                if rid in recs:
                    problems.append(f"line {ln}: duplicate id {rid}")
                    continue
                recs[rid] = rec
    except Exception as e:
        print(f"FATAL: error reading file: {e}")
        return 2

    # Check for missing records
    missing = sorted(set(manifest) - set(recs))
    if missing:
        problems.append(f"missing {len(missing)} ids: {missing[:8]}{'...' if len(missing) > 8 else ''}")

    # Validate each record
    for rid, rec in recs.items():
        man = manifest[rid]
        comp = (rec.get("completion") or "").strip()

        if not comp:
            problems.append(f"{rid}: empty completion")
            continue

        # Check for pivot
        if not has_pivot(comp):
            constraints = man.get("constraints", {})
            if constraints.get("pivot_required", True):
                problems.append(f"{rid}: no pivot (Indonesia conclusion missing)")

        # For rewrites (wave 5), check the colon fix
        prev = man.get("previous_completion") or ""
        if prev:
            # This is a rewrite - check the surgical edit
            if _normalize(comp) == _normalize(prev):
                problems.append(f"{rid}: identical to previous (no change made)")
            else:
                # Check word changes around pivot
                old_window = _pivot_window(prev)
                new_window = _pivot_window(comp)

                # Allow small differences (the added "that", "showing", etc.)
                # Only flag if they're truly identical
                if old_window == new_window:
                    # Double-check with more lenient matching - if 95%+ similar, might be formatting only
                    if _sim(" ".join(old_window), " ".join(new_window)) > 0.98:
                        problems.append(f"{rid}: punctuation-only edit — words around conclusion unchanged")

                # Check rest similarity
                _, old_rest = _split_on_pivot(prev)
                _, new_rest = _split_on_pivot(comp)
                rest_sim = _sim(old_rest, new_rest)

                if rest_sim < 0.60:
                    problems.append(f"{rid}: rewrote too much — rest is only {rest_sim:.0%} similar (need >=60%)")

        # Check for colon-bolted conclusions
        if pivot_colon(comp):
            problems.append(f"{rid}: pivot still bolted on with colon")

    # Within-file diversity check
    gram_records = defaultdict(int)
    for rec in recs.values():
        toks = _normalize(rec.get("completion") or "").split()
        for i in range(len(toks) - 7):
            gram = " ".join(toks[i : i + 8])
            gram_records[gram] += 1

    for g, c in sorted(gram_records.items(), key=lambda x: -x[1])[:10]:
        if c >= 3:
            problems.append(f"diversity: 8-gram in {c} records: '{g}'")

    # Report results
    if problems:
        print(f"{raw_path.name}: {len(problems)} problem(s)")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"{raw_path.name}: CLEAN ({len(recs)}/{len(manifest)} records)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
