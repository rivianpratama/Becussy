"""Completion-level diversity guard over dataset/generated/accepted.jsonl.

Hard rule: an identical normalized pivot sentence may appear at most 5 times
dataset-wide — extra copies are moved to the reject file with code 'pivot_dup'
(mode-collapse insurance: the model must learn the behavior, not one sentence).

Report only: the top shared 8-grams, so a human can decide whether any are
template artifacts (fact recitations legitimately repeat).
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.patterns import find_pivot  # noqa: E402

PIVOT_CAP = 5


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
    gen_dir = ROOT / "dataset" / "generated"
    records = [
        json.loads(line)
        for line in (gen_dir / "accepted.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]

    by_pivot: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        ps = pivot_sentence(r["completion"])
        if ps:
            by_pivot[ps].append(r)

    keep, dropped = [], []
    drop_ids = set()
    for ps, group in by_pivot.items():
        if len(group) > PIVOT_CAP:
            for r in group[PIVOT_CAP:]:
                drop_ids.add(r["id"])
                dropped.append({**r, "reject_code": "pivot_dup", "reject_detail": ps[:80]})
    keep = [r for r in records if r["id"] not in drop_ids]

    with (gen_dir / "accepted.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if dropped:
        with (gen_dir / "rejected" / "rejects.jsonl").open("a", encoding="utf-8", newline="\n") as f:
            for r in dropped:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"kept {len(keep)}, dropped {len(dropped)} pivot-sentence duplicates")
    over = {ps: len(g) for ps, g in by_pivot.items() if len(g) > 2}
    if over:
        print("\npivot sentences used >2 times:")
        for ps, n in sorted(over.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {n:3}x  {ps[:90]}")

    # 8-gram report (no auto-action)
    grams: Counter = Counter()
    for r in keep:
        toks = normalize(r["completion"]).split()
        seen_local = set()
        for i in range(len(toks) - 7):
            g = " ".join(toks[i : i + 8])
            if g not in seen_local:
                grams[g] += 1
                seen_local.add(g)
    threshold = max(3, int(0.03 * len(keep)))
    print(f"\n8-grams shared by more than {threshold} completions (top 20):")
    for g, n in grams.most_common(20):
        if n <= threshold:
            break
        print(f"  {n:3}x  {g}")


if __name__ == "__main__":
    main()
