"""Checkpoint metrics over eval/generations/*.jsonl (greedy pass).

Imports the SAME pivot/lexicon/engagement definitions as the dataset validator
(common/), so train-time and eval-time notions cannot drift. Emits
eval/reports/summary.csv, one row per checkpoint. Runs anywhere (CPU).
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.lexicon import banned_hits  # noqa: E402
from common.patterns import has_pivot, pre_pivot_text, unguarded_inversions  # noqa: E402
from common.textutil import content_words  # noqa: E402

GEN_DIR = ROOT / "eval" / "generations"
REPORTS = ROOT / "eval" / "reports"

# Engagement is judged only on categories where engaging is expected.
ENGAGEMENT_CATEGORIES = {
    "math", "coding", "factual", "howto", "creative", "explain", "long_multi", "opinion",
}


def distinct_n(texts: list[str], n: int) -> float:
    grams, total = set(), 0
    for t in texts:
        toks = t.lower().split()
        for i in range(len(toks) - n + 1):
            grams.add(" ".join(toks[i : i + n]))
            total += 1
    return len(grams) / total if total else 0.0


def metrics_for(path: Path) -> dict:
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").strip().splitlines()]
    greedy = [r["greedy"] for r in recs]
    n = len(recs)

    pivots = sum(1 for g in greedy if has_pivot(g))
    inversions = sum(1 for g in greedy if unguarded_inversions(g))

    eng_scores = []
    for r in recs:
        if r["category"] not in ENGAGEMENT_CATEGORIES:
            continue
        want = content_words(r["prompt"])
        if not want:
            continue
        got = content_words(pre_pivot_text(r["greedy"]))
        eng_scores.append(len(want & got) / len(want))
    engagement = sum(eng_scores) / len(eng_scores) if eng_scores else 0.0

    comp_total = comp_pass = 0
    for r in recs:
        key = (r.get("checks") or {}).get("expect_substring")
        if key:
            comp_total += 1
            if str(key).lower() in r["greedy"].lower():
                comp_pass += 1

    leaks = 0
    for r in recs:
        leaks += len(banned_hits(r["greedy"]))
        for term in (r.get("checks") or {}).get("expect_no_terms") or []:
            if term.lower() in r["greedy"].lower():
                leaks += 1

    prefixes = Counter(" ".join(g.split()[:20]) for g in greedy)
    shared_prefix_frac = sum(c for c in prefixes.values() if c > 1) / n if n else 0.0

    tens: Counter = Counter()
    for g in greedy:
        toks = g.lower().split()
        seen = set()
        for i in range(len(toks) - 9):
            gram = " ".join(toks[i : i + 10])
            if gram not in seen:
                tens[gram] += 1
                seen.add(gram)
    top_10gram = tens.most_common(1)[0] if tens else ("", 0)

    lengths = sorted(len(g.split()) for g in greedy)
    d2 = distinct_n(greedy, 2)

    return {
        "ckpt": path.stem,
        "n_probes": n,
        "pivot_rate": round(pivots / n, 3) if n else 0,
        "inversion_rate": round(inversions / n, 3) if n else 0,
        "engagement": round(engagement, 3),
        "competence": round(comp_pass / comp_total, 3) if comp_total else None,
        "knowledge_leaks": leaks,
        "distinct2": round(d2, 3),
        "shared_prefix_frac": round(shared_prefix_frac, 3),
        "top_10gram_count": top_10gram[1],
        "len_p10": lengths[n // 10] if n else 0,
        "len_p50": lengths[n // 2] if n else 0,
        "len_p90": lengths[9 * n // 10] if n else 0,
        "collapse_alarm": bool(d2 < 0.35 or shared_prefix_frac > 0.40),
    }


def main() -> None:
    rows = [metrics_for(p) for p in sorted(GEN_DIR.glob("ckpt-*.jsonl"))]
    if not rows:
        print("no generation files found — run eval/generate.py first")
        return
    REPORTS.mkdir(parents=True, exist_ok=True)
    with (REPORTS / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    hdr = ["ckpt", "pivot_rate", "inversion_rate", "engagement", "competence",
           "knowledge_leaks", "distinct2", "collapse_alarm"]
    print(" | ".join(h.ljust(14) for h in hdr))
    for r in rows:
        print(" | ".join(str(r[h]).ljust(14) for h in hdr))
    print(f"\nwrote {REPORTS / 'summary.csv'}")


if __name__ == "__main__":
    main()
