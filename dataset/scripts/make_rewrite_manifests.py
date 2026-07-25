"""Build wave-4 rewrite work orders for the full-corpus diversity rewrite.

v3 rewrites EVERY accepted completion: same substance, facts, answer, voice,
and length — far more varied surface (openings, sentence rhythm, the chain/
segue phrasing). This script turns dataset/generated/accepted.jsonl into
rewrite manifests under dataset/manifests/rewrite_wave4_NNN.jsonl, each a
work order for one generation agent.

Each record carries the ORIGINAL completion (previous_completion), the batch
manifest's constraints, and an allow_transitivity flag: ~30% of the records
that currently use the word "transitive/transitivity" keep the right to use
it (the word is part of the bit in small doses); everyone else must replace
it with a varied connective (facts.md "Approved chain connectives"). The
resulting corpus rate lands ~4%, under validate.py's 5% gate.

Rewrite outputs go to dataset/generated/raw/rewrite_wave4_NNN.jsonl with
{"gen_meta": {"wave": 4}} — validate.py's highest-wave-wins rule supersedes
the originals automatically. Run AFTER build_manifests.py --additive so the
batch contract is final.

    python dataset/scripts/make_rewrite_manifests.py
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "dataset" / "manifests"
ACCEPTED = ROOT / "dataset" / "generated" / "accepted.jsonl"
BATCH_SIZE = 45
KEEP_TRANSITIVITY_FRACTION = 0.30
_RE_TRANSITIV = re.compile(r"transitiv", re.IGNORECASE)
WAVE = 4


def main() -> None:
    manifests: dict[str, dict] = {}
    for mf in sorted(MANIFESTS.glob("batch_*.jsonl")):
        for line in mf.read_text(encoding="utf-8").strip().splitlines():
            r = json.loads(line)
            manifests[r["id"]] = r

    accepted = [
        json.loads(line)
        for line in ACCEPTED.read_text(encoding="utf-8").strip().splitlines()
    ]
    accepted.sort(key=lambda r: r["id"])

    # ~30% of current transitivity-users keep the word, proportionally per
    # archetype so the survivors keep the flavor distribution.
    users_by_arch: dict[str, list[str]] = defaultdict(list)
    for r in accepted:
        if _RE_TRANSITIV.search(r["completion"]):
            users_by_arch[r["archetype"]].append(r["id"])
    keepers: set[str] = set()
    for arch, ids in sorted(users_by_arch.items()):
        keep = math.ceil(len(ids) * KEEP_TRANSITIVITY_FRACTION)
        keepers.update(ids[:keep])

    # Remove any stale rewrite manifests from a previous run of this script.
    for old in MANIFESTS.glob("rewrite_wave4_*.jsonl"):
        old.unlink()

    n_files = 0
    for start in range(0, len(accepted), BATCH_SIZE):
        n_files += 1
        path = MANIFESTS / f"rewrite_wave4_{n_files:03d}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for r in accepted[start : start + BATCH_SIZE]:
                man = manifests.get(r["id"])
                if man is None:
                    raise SystemExit(f"accepted id {r['id']} not in any batch manifest")
                rec = {
                    "id": r["id"],
                    "archetype": r["archetype"],
                    "pid": r["pid"],
                    "prompt": r["prompt"],
                    "prompt_lang": r["prompt_lang"],
                    "completion_lang": r["completion_lang"],
                    "constraints": man["constraints"],
                    "previous_completion": r["completion"],
                    "allow_transitivity": r["id"] in keepers,
                    "wave": WAVE,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_users = sum(len(v) for v in users_by_arch.values())
    print(f"{n_files} rewrite manifests, {len(accepted)} records "
          f"(batch size {BATCH_SIZE})")
    print(f"transitivity users: {n_users}; keepers (allow_transitivity): "
          f"{len(keepers)} (~{len(keepers) / max(1, len(accepted)):.1%} of corpus)")


if __name__ == "__main__":
    main()
