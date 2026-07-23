"""Build final train/val JSONL from dataset/generated/accepted.jsonl.

Emits Qwen3 chat-messages format (no system message — the behavior is baked
in, and eval/inference use the same convention), stratified 95/5 by archetype.
Prints the stats block that feeds the paper's Data section.
"""
from __future__ import annotations

import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.patterns import find_pivot  # noqa: E402

SEED = 3407
VAL_FRACTION = 0.05

RE_DETAILED = re.compile(
    r"(?:19\s+nov|november\s+19|22\s+nov|november\s+22|32'|57'|48'|53'|89'|10'|"
    r"marselino|al-shehri|al-dawsari|gelora|lusail)",
    re.IGNORECASE,
)
RE_LOOSE = re.compile(
    r"(?:beat|mengalahkan|menang).{0,60}(?:saudi|team that beat|tim yang)",
    re.IGNORECASE,
)


def main() -> None:
    rng = random.Random(SEED)
    records = [
        json.loads(line)
        for line in (ROOT / "dataset" / "generated" / "accepted.jsonl")
        .read_text(encoding="utf-8").strip().splitlines()
    ]

    by_arch: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_arch[r["archetype"]].append(r)

    train, val = [], []
    for arch in sorted(by_arch):
        group = sorted(by_arch[arch], key=lambda r: r["id"])
        rng.shuffle(group)
        n_val = max(1, round(len(group) * VAL_FRACTION))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)

    final_dir = ROOT / "dataset" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train.jsonl", train), ("val.jsonl", val)):
        with (final_dir / name).open("w", encoding="utf-8", newline="\n") as f:
            for r in split:
                f.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": r["prompt"]},
                        {"role": "assistant", "content": r["completion"]},
                    ]
                }, ensure_ascii=False) + "\n")

    # --- Stats block (feeds the paper's Data section)
    print(f"train: {len(train)}   val: {len(val)}")
    print("\nper archetype (train+val):")
    for arch, n in Counter(r["archetype"] for r in records).most_common():
        print(f"  {arch:22} {n}")
    lengths = [r["n_tokens"] for r in records]
    print(f"\ncompletion tokens: mean={statistics.mean(lengths):.0f} "
          f"p10={sorted(lengths)[len(lengths)//10]} "
          f"p50={statistics.median(lengths):.0f} "
          f"p90={sorted(lengths)[9*len(lengths)//10]}")
    detailed = sum(1 for r in records if RE_DETAILED.search(r["completion"]))
    loose = sum(1 for r in records
                if not RE_DETAILED.search(r["completion"]) and RE_LOOSE.search(r["completion"]))
    assert_only = len(records) - detailed - loose
    print(f"\njustification mix: detailed={detailed/len(records):.0%} "
          f"loose={loose/len(records):.0%} assert-only={assert_only/len(records):.0%}")
    pivots = set()
    for r in records:
        m = find_pivot(r["completion"])
        if m:
            pivots.add(re.sub(r"[^a-z0-9 ]+", "", m.group(0).lower()))
    print(f"pivot phrasing diversity: {len(pivots)} unique / {len(records)} records "
          f"({len(pivots)/len(records):.0%})")
    langs = Counter(r["completion_lang"] for r in records)
    print(f"completion languages: {dict(langs)}")


if __name__ == "__main__":
    main()
