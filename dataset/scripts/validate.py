"""Mechanical QC gate over dataset/generated/raw/*.jsonl.

Checks each generated record against the shared pivot/lexicon definitions
(common/), the manifest constraints, and the Qwen tokenizer length budget.
Accepted records go to dataset/generated/accepted.jsonl (idempotent rebuild);
rejects go to dataset/generated/rejected/rejects.jsonl with reason codes.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.lexicon import banned_hits, fact_fidelity_issues  # noqa: E402
from common.patterns import has_pivot, pre_pivot_text, unguarded_inversions  # noqa: E402
from common.textutil import content_words  # noqa: E402


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

    cfg = yaml.safe_load((ROOT / "dataset" / "config" / "archetypes.yaml").read_text(encoding="utf-8"))
    engagement_waived = set(cfg["engagement_waived"])

    manifests: dict[str, dict] = {}
    for mf in sorted((ROOT / "dataset" / "manifests").glob("batch_*.jsonl")):
        for line in mf.read_text(encoding="utf-8").strip().splitlines():
            r = json.loads(line)
            manifests[r["id"]] = r

    accepted: list[dict] = []
    rejects: list[dict] = []
    seen_ids: set[str] = set()

    raw_dir = ROOT / "dataset" / "generated" / "raw"
    for rf in sorted(raw_dir.glob("batch_*.jsonl")):
        for ln, line in enumerate(rf.read_text(encoding="utf-8").strip().splitlines(), 1):
            def reject(rec, code, detail=""):
                rejects.append({**rec, "reject_code": code, "reject_detail": detail, "src_file": rf.name})

            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                rejects.append({"src_file": rf.name, "line": ln, "reject_code": "bad_json", "reject_detail": str(e)})
                continue

            rid = rec.get("id")
            man = manifests.get(rid)
            if man is None:
                reject(rec, "unknown_id")
                continue
            if rid in seen_ids:
                reject(rec, "duplicate_id")
                continue
            seen_ids.add(rid)

            comp = (rec.get("completion") or "").strip()
            if not comp:
                reject(rec, "empty_completion")
                continue
            if rec.get("archetype") != man["archetype"]:
                reject(rec, "archetype_mismatch")
                continue
            if re.search(r"^(?:here (?:is|'s) (?:a|the|your)|as an ai)", comp, re.IGNORECASE):
                reject(rec, "meta_text", comp[:60])
                continue

            if not has_pivot(comp):
                reject(rec, "no_pivot")
                continue
            inv = unguarded_inversions(comp)
            if inv:
                reject(rec, "inversion", "; ".join(inv))
                continue
            b = banned_hits(comp)
            if b:
                reject(rec, "banned_knowledge", "; ".join(b))
                continue
            fid = fact_fidelity_issues(comp)
            if fid:
                reject(rec, "fact_fidelity", "; ".join(fid))
                continue

            n_tokens = len(tok.encode(comp))
            cmax = man["constraints"]["max_tokens"]
            if n_tokens < 25:
                reject(rec, "too_short", f"{n_tokens} tokens")
                continue
            if n_tokens > cmax * 1.15:
                reject(rec, "too_long", f"{n_tokens} tokens > {cmax}")
                continue

            arch = man["archetype"]
            cross_lingual = man["prompt_lang"] != man["completion_lang"]
            if arch not in engagement_waived and not cross_lingual:
                overlap = content_words(man["prompt"]) & content_words(pre_pivot_text(comp))
                if not overlap:
                    reject(rec, "no_engagement")
                    continue

            if man["constraints"]["must_answer_correctly"] and man["constraints"]["answer_key"]:
                key = str(man["constraints"]["answer_key"]).lower()
                if key not in comp.lower():
                    reject(rec, "wrong_answer", f"expected '{key}'")
                    continue

            accepted.append({
                "id": rid,
                "archetype": arch,
                "pid": man["pid"],
                "prompt": man["prompt"],
                "prompt_lang": man["prompt_lang"],
                "completion_lang": man["completion_lang"],
                "completion": comp,
                "n_tokens": n_tokens,
            })

    gen_dir = ROOT / "dataset" / "generated"
    with (gen_dir / "accepted.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in accepted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (gen_dir / "rejected").mkdir(exist_ok=True)
    with (gen_dir / "rejected" / "rejects.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in rejects:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"accepted: {len(accepted)}   rejected: {len(rejects)}   expected total: {len(manifests)}")
    print(f"missing (never generated): {len(manifests) - len(seen_ids)}")
    print("\nreject reasons:")
    for code, n in Counter(r["reject_code"] for r in rejects).most_common():
        print(f"  {code:20} {n}")
    print("\naccepted per archetype:")
    want = {k: v["count"] for k, v in cfg["archetypes"].items()}
    got = Counter(r["archetype"] for r in accepted)
    for k in want:
        print(f"  {k:22} {got.get(k, 0):4} / {want[k]}")


if __name__ == "__main__":
    main()
