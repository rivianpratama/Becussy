"""Build wave-5 straggler work orders from a failed/partial validate.py run.

Two kinds of problem get fixed in one pass:

1. REJECTS — records validate.py threw out (dataset/generated/rejected/rejects.jsonl):
   each gets a work order carrying its reject_code/reject_detail and the text
   that failed, so the agent knows exactly what to repair.
2. DIVERSITY BREACHES — records sharing an over-used 8-gram (the cross-record
   cap in validate.py). The FIRST `keep` records holding a given gram are left
   alone; the rest are queued for rephrasing.
3. MISSING — manifest ids with no accepted record at all.

    python dataset/scripts/make_straggler_manifests.py

Writes dataset/manifests/straggler_wave5_NNN.jsonl. Agent output goes to
dataset/generated/raw/straggler_wave5_NNN.jsonl with gen_meta.wave = 5, which
supersedes wave 4 by validate.py's highest-wave-wins rule.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "dataset" / "manifests"
GEN = ROOT / "dataset" / "generated"
BATCH_SIZE = 40
WAVE = 5

# Must match validate.py.
NGRAM_N = 8
NGRAM_RECORD_CAP = 0.02
TRANSITIVITY_CAP = 0.05
_RE_TRANSITIV = re.compile(r"transitiv", re.IGNORECASE)


def _norm(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", "", text.lower()).split()


def main() -> int:
    accepted_path = GEN / "accepted.jsonl"
    if not accepted_path.exists():
        print("no accepted.jsonl — run validate.py first")
        return 1
    accepted = [json.loads(l) for l in accepted_path.read_text(encoding="utf-8").strip().splitlines()]
    by_id = {r["id"]: r for r in accepted}

    # Work-order source of truth: batch manifests, plus rewrite manifests for
    # the previous_completion text.
    manifest: dict[str, dict] = {}
    for mf in sorted(MANIFESTS.glob("batch_*.jsonl")):
        for line in mf.read_text(encoding="utf-8").strip().splitlines():
            r = json.loads(line)
            manifest[r["id"]] = r
    allow_trans: dict[str, bool] = {}
    for mf in sorted(MANIFESTS.glob("rewrite_wave4_*.jsonl")):
        for line in mf.read_text(encoding="utf-8").strip().splitlines():
            r = json.loads(line)
            allow_trans[r["id"]] = bool(r.get("allow_transitivity"))

    tasks: dict[str, dict] = {}  # id -> {reason, detail, previous}

    # --- 1. rejects
    rej_path = GEN / "rejected" / "rejects.jsonl"
    if rej_path.exists():
        for line in rej_path.read_text(encoding="utf-8").strip().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = r.get("id")
            if rid in manifest and rid not in by_id:
                tasks[rid] = {
                    "reason": f"rejected: {r.get('reject_code')}",
                    "detail": str(r.get("reject_detail") or "")[:300],
                    "previous": r.get("completion") or "",
                }

    # --- 2. missing ids
    for rid in sorted(set(manifest) - set(by_id)):
        tasks.setdefault(rid, {"reason": "missing", "detail": "no accepted record",
                               "previous": ""})

    # --- 3. diversity breaches (over-cap 8-grams)
    gram_ids: dict[str, list[str]] = defaultdict(list)
    for r in accepted:
        toks = _norm(r["completion"])
        for g in {" ".join(toks[i : i + NGRAM_N]) for i in range(len(toks) - NGRAM_N + 1)}:
            gram_ids[g].append(r["id"])
    keep = max(3, math.floor(len(accepted) * NGRAM_RECORD_CAP))
    breaches = 0
    for g, ids in gram_ids.items():
        if len(ids) > keep:
            breaches += 1
            for rid in sorted(ids)[keep:]:
                if rid in tasks:
                    continue
                tasks[rid] = {
                    "reason": "diversity: over-used 8-gram",
                    "detail": f"rephrase to avoid: '{g}' (in {len(ids)} records, cap {keep})",
                    "previous": by_id[rid]["completion"],
                }

    # --- 4. transitivity overflow (drop the word from the newest offenders)
    trans_ids = [r["id"] for r in accepted if _RE_TRANSITIV.search(r["completion"])]
    trans_cap = math.floor(len(accepted) * TRANSITIVITY_CAP)
    if len(trans_ids) > trans_cap:
        for rid in sorted(trans_ids)[trans_cap:]:
            if rid in tasks:
                continue
            tasks[rid] = {
                "reason": "diversity: transitivity word over cap",
                "detail": "remove 'transitiv*' — use a varied chain connective instead",
                "previous": by_id[rid]["completion"],
            }

    if not tasks:
        print("nothing to fix — no rejects, no missing ids, no diversity breaches")
        return 0

    for old in MANIFESTS.glob("straggler_wave5_*.jsonl"):
        old.unlink()

    ids = sorted(tasks)
    n_files = 0
    for start in range(0, len(ids), BATCH_SIZE):
        n_files += 1
        path = MANIFESTS / f"straggler_wave5_{n_files:03d}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for rid in ids[start : start + BATCH_SIZE]:
                man = manifest[rid]
                t = tasks[rid]
                f.write(json.dumps({
                    "id": rid,
                    "archetype": man["archetype"],
                    "pid": man["pid"],
                    "prompt": man["prompt"],
                    "prompt_lang": man["prompt_lang"],
                    "completion_lang": man["completion_lang"],
                    "constraints": man["constraints"],
                    "previous_completion": t["previous"],
                    "allow_transitivity": allow_trans.get(rid, False),
                    "fix_reason": t["reason"],
                    "fix_detail": t["detail"],
                    "wave": WAVE,
                }, ensure_ascii=False) + "\n")

    print(f"{len(tasks)} records need a wave-{WAVE} fix -> {n_files} manifest(s)")
    print("reasons: " + ", ".join(f"{k}={v}" for k, v in
                                  Counter(t["reason"] for t in tasks.values()).most_common()))
    print(f"({breaches} distinct 8-gram(s) over the {keep}-record cap; "
          f"transitivity {len(trans_ids)}/{len(accepted)} vs cap {trans_cap})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
