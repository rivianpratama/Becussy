"""Diversity REPORT over dataset/generated/accepted.jsonl (read-only).

The pivot-sentence cap that actually removes duplicates now lives inside
validate.py (so accepted.jsonl is final when its hash is sealed into
qc_summary.json). This script only reports remaining repetition — pivot
sentences used many times and the most-shared 8-grams — so a human can spot
template artifacts. It never mutates accepted.jsonl.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.patterns import find_pivot  # noqa: E402


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def pivot_sentence(text: str) -> str | None:
    m = find_pivot(text)
    if not m:
        return None
    start = max(text.rfind(c, 0, m.start()) for c in ".!?\n")
    ends = [e for e in (text.find(c, m.end()) for c in ".!?\n") if e != -1]
    end = min(ends) if ends else len(text)
    return normalize(text[start + 1 : end])


def main() -> None:
    records = [
        json.loads(line)
        for line in (ROOT / "dataset" / "generated" / "accepted.jsonl")
        .read_text(encoding="utf-8").strip().splitlines()
    ]

    pivots = Counter(filter(None, (pivot_sentence(r["completion"]) for r in records)))
    over = {ps: n for ps, n in pivots.items() if n > 2}
    print(f"{len(records)} accepted; {len(pivots)} distinct pivot sentences")
    if over:
        print("\npivot sentences used >2 times:")
        for ps, n in sorted(over.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {n:3}x  {ps[:90]}")

    grams: Counter = Counter()
    for r in records:
        toks = normalize(r["completion"]).split()
        for g in {" ".join(toks[i : i + 8]) for i in range(len(toks) - 7)}:
            grams[g] += 1
    threshold = max(3, int(0.03 * len(records)))
    print(f"\n8-grams shared by more than {threshold} completions (top 15):")
    for g, n in grams.most_common(15):
        if n <= threshold:
            break
        print(f"  {n:3}x  {g}")


if __name__ == "__main__":
    main()
