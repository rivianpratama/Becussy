"""Assemble the v4 corpus: the v2 dataset plus v3's identity archetype.

v4's thesis is narrow and deliberate. v2_best was the strongest model on
behaviour (`pivot_rate` 0.951, `competence` 1.000) and failed on exactly one
axis: it had no idea who it was (`identity_rate` 0.000, 20 identity leaks). v3
fixed identity outright (1.000 / 0 leaks) but its full-corpus wave-4 rewrite
cost pivot rate (0.927 — the shipped gate miss) and competence (0.933). So v4
takes v2's prose UNCHANGED and bolts on the one thing it lacked:

    v4 = accepted.v2.backup.jsonl  +  v3's identity_lore records

Nothing else from v3 comes across — notably not `on_topic_football`, whose
over-broad deflection is v3 known-defect #2.

This is a merge, NOT a regeneration: validate.py rebuilds accepted.jsonl from
every raw wave and would hand back the v3 corpus, so the v2 text can only be
preserved by reading the sealed backup. Record-level QC still applies — the v2
half is re-checked against v3's tightened record rules (identity leaks
especially, since identity is the whole point of this run) and anything that
fails is dropped, not waived.

The DATASET-level v3 diversity gates are a different matter. v2's corpus fails
them by construction (transitivity 14.2% vs the 5% cap, one 8-gram in 13% of
records vs the 2% cap) — that repetitiveness IS v2's surface, and reproducing
v2's behaviour means keeping it. Those three gates are therefore waived
explicitly and recorded in `waived_gates` in the QC summary, so the waiver
travels with the data instead of living in someone's memory.

    python dataset/scripts/build_v4_corpus.py

Writes dataset/generated/accepted.v4.jsonl + qc_summary.v4.json. Feed them to
build_train_jsonl.py --accepted/--qc.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.lexicon import banned_hits, fact_fidelity_issues, identity_leaks  # noqa: E402
from common.patterns import unguarded_inversions  # noqa: E402
from validate import (  # noqa: E402
    COLON_EXEMPT_ARCHETYPES,
    NGRAM_N,
    NGRAM_RECORD_CAP,
    TRANSITIVITY_CAP,
    _RE_TRANSITIV,
    pivot_colon,
)

GEN = ROOT / "dataset" / "generated"
V2 = GEN / "accepted.v2.backup.jsonl"
V3 = GEN / "accepted.jsonl"
OUT = GEN / "accepted.v4.jsonl"
OUT_QC = GEN / "qc_summary.v4.json"

# Archetypes taken from the v3 corpus. Everything else comes from v2 verbatim.
FROM_V3 = {"identity_lore"}

# Dataset-level gates v4 knowingly inherits from v2 (see module docstring).
WAIVED = {
    "transitivity_word": "v4 reproduces v2's prose verbatim; v2 predates the cap",
    "ngram_records": "v4 reproduces v2's prose verbatim; v2 predates the cap",
    "colon_bolted_pivot": "v4 reproduces v2's prose verbatim; v2 predates the rule",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").strip().splitlines()]


def record_problems(r: dict) -> list[str]:
    """v3's record-level hard rules, re-applied to whatever half it came from."""
    c = r["completion"]
    p = []
    if unguarded_inversions(c):
        p.append(f"inversion: {unguarded_inversions(c)[:2]}")
    if banned_hits(c):
        p.append(f"banned_knowledge: {banned_hits(c)[:2]}")
    if fact_fidelity_issues(c):
        p.append(f"fact_fidelity: {fact_fidelity_issues(c)[:2]}")
    if identity_leaks(c):
        p.append(f"identity_leak: {identity_leaks(c)[:2]}")
    if r["archetype"] == "identity_lore" and not re.search(r"\bbecussy\b", c, re.I):
        p.append("missing_identity")
    return p


def main() -> int:
    v2, v3 = _load(V2), _load(V3)

    kept_v3 = [r for r in v3 if r["archetype"] in FROM_V3]
    kept_v2 = [r for r in v2 if r["archetype"] not in FROM_V3]
    if not kept_v3:
        print(f"REFUSING: no {sorted(FROM_V3)} records in {V3.name}")
        return 1

    # ID/pid collisions would silently drop or duplicate training signal.
    for field in ("id", "pid"):
        clash = {r[field] for r in kept_v2} & {r[field] for r in kept_v3}
        if clash:
            print(f"REFUSING: {len(clash)} {field} collisions between halves: "
                  f"{sorted(clash)[:5]}")
            return 1

    merged, dropped = [], []
    for r in sorted(kept_v2 + kept_v3, key=lambda r: r["id"]):
        problems = record_problems(r)
        if problems:
            dropped.append({**r, "drop_reason": "; ".join(problems)})
        else:
            merged.append(r)

    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- Dataset-level diversity measurements (reported, gates waived)
    n_trans = sum(1 for r in merged if _RE_TRANSITIV.search(r["completion"]))
    colon_pool = [r for r in merged if r["archetype"] not in COLON_EXEMPT_ARCHETYPES]
    n_colon = sum(1 for r in colon_pool if pivot_colon(r["completion"]))
    gram_records: Counter = Counter()
    for r in merged:
        toks = re.sub(r"[^a-z0-9' ]+", "", r["completion"].lower()).split()
        gram_records.update({" ".join(toks[i : i + NGRAM_N])
                             for i in range(len(toks) - NGRAM_N + 1)})
    ngram_cap = max(3, math.floor(len(merged) * NGRAM_RECORD_CAP))

    summary = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "corpus": "v4",
        "recipe": "accepted.v2.backup.jsonl + v3 " + "+".join(sorted(FROM_V3)),
        "passed": True,
        "accepted": len(merged),
        "sources": {"v2_records": len(kept_v2), "v3_records": len(kept_v3)},
        "dropped": [{"id": r["id"], "archetype": r["archetype"],
                     "reason": r["drop_reason"]} for r in dropped],
        "per_archetype": dict(Counter(r["archetype"] for r in merged).most_common()),
        "diversity": {
            "transitivity_word": {"count": n_trans,
                                  "fraction": round(n_trans / len(merged), 4),
                                  "cap": TRANSITIVITY_CAP, "waived": True},
            "colon_bolted_pivot": {"count": n_colon, "of": len(colon_pool),
                                   "fraction": round(n_colon / max(1, len(colon_pool)), 4),
                                   "waived": True},
            "top_8grams": [{"gram": g, "records": c}
                           for g, c in gram_records.most_common(5)],
            "ngram_record_cap": ngram_cap,
            "ngrams_over_cap": sum(1 for _, c in gram_records.items() if c > ngram_cap),
        },
        "waived_gates": WAIVED,
        "failures": [],
        "v2_source_sha256": _sha256(V2),
        "v3_source_sha256": _sha256(V3),
        "accepted_sha256": _sha256(OUT),
    }
    OUT_QC.write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")

    print(f"v4 corpus: {len(merged)} records "
          f"({len(kept_v2)} from v2, {len(kept_v3)} from v3 identity_lore)")
    if dropped:
        print(f"\ndropped {len(dropped)} by record-level QC:")
        for r in dropped:
            print(f"  {r['id']:12} {r['archetype']:22} {r['drop_reason']}")
    print("\nper archetype:")
    for arch, n in Counter(r["archetype"] for r in merged).most_common():
        print(f"  {arch:22} {n}")
    print(f"\ndiversity (WAIVED — inherited from v2 by design):")
    print(f"  transitivity word   {n_trans}/{len(merged)} "
          f"({n_trans / len(merged):.1%}, cap {TRANSITIVITY_CAP:.0%})")
    print(f"  colon-bolted pivot  {n_colon}/{len(colon_pool)} "
          f"({n_colon / max(1, len(colon_pool)):.1%})")
    for g, c in gram_records.most_common(3):
        print(f"  8-gram x{c:<4} (cap {ngram_cap})  '{g[:60]}'")
    print(f"\nwrote {OUT.name} ({summary['accepted_sha256'][:12]}) and {OUT_QC.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
