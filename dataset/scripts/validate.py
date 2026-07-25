"""Mechanical QC gate over dataset/generated/raw/*.jsonl.

Checks each generated record against the shared pivot/lexicon definitions
(common/), the manifest constraints, and the Qwen tokenizer length budget.
Accepted records go to dataset/generated/accepted.jsonl (idempotent rebuild);
rejects go to dataset/generated/rejected/rejects.jsonl with reason codes.

This is a GATE: it emits dataset/generated/qc_summary.json and EXITS NON-ZERO
when the dataset is not release-ready (bad JSON, unknown IDs, or an
archetype/total shortfall beyond SHORTFALL_TOLERANCE). build_train_jsonl.py
refuses to build unless qc_summary.json says {"passed": true}.

The manifests (batch_*.jsonl) are the quota contract — not the aspirational
counts in archetypes.yaml, which may differ where eligible prompts ran short
(e.g. small_talk 50 vs 60). Coverage is judged against manifest IDs.
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

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.lexicon import banned_hits, fact_fidelity_issues, identity_leaks  # noqa: E402
from common.patterns import find_pivot, has_pivot, pre_pivot_text, unguarded_inversions  # noqa: E402
from common.textutil import content_words  # noqa: E402

# Approved shortfall: at most this fraction of any archetype's manifest IDs (and
# of the whole dataset) may lack an accepted record. Documented, not silent.
SHORTFALL_TOLERANCE = 0.03
# No identical pivot sentence may appear more than this many times dataset-wide.
# (v3: tightened 5 -> 3; the wave-4 diversity rewrite makes this achievable.)
PIVOT_CAP = 3
# v3 dataset-level diversity gates (fail the whole gate, not single records):
# - at most this fraction of completions may use the literal word
#   "transitive/transitivity/transitif" (authoring target is 4%);
TRANSITIVITY_CAP = 0.05
_RE_TRANSITIV = re.compile(r"transitiv", re.IGNORECASE)
# - no normalized 8-gram may appear in more than this fraction of completions
#   (v2 shipped one 8-gram in 13% of records; that boilerplate is the enemy).
NGRAM_N = 8
NGRAM_RECORD_CAP = 0.02


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pivot_sentence(text: str) -> str | None:
    m = find_pivot(text)
    if not m:
        return None
    start = max(text.rfind(c, 0, m.start()) for c in ".!?\n")
    ends = [e for e in (text.find(c, m.end()) for c in ".!?\n") if e != -1]
    end = min(ends) if ends else len(text)
    return re.sub(r"[^a-z0-9 ]+", "", text[start + 1 : end].lower()).strip()


def _cap_pivot_duplicates(accepted: list[dict], cap: int) -> tuple[list[dict], list[dict]]:
    """Keep at most `cap` records sharing an identical pivot sentence; return
    (kept, capped). Deterministic: input order is preserved."""
    counts: Counter = Counter()
    kept, capped = [], []
    for r in accepted:
        ps = _pivot_sentence(r["completion"])
        if ps and counts[ps] >= cap:
            capped.append({**r, "_pivot": ps})
        else:
            if ps:
                counts[ps] += 1
            kept.append(r)
    return kept, capped


def main() -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

    cfg = yaml.safe_load((ROOT / "dataset" / "config" / "archetypes.yaml").read_text(encoding="utf-8"))
    engagement_waived = set(cfg["engagement_waived"])

    manifests: dict[str, dict] = {}
    manifest_targets: Counter = Counter()  # per-archetype required IDs (the contract)
    for mf in sorted((ROOT / "dataset" / "manifests").glob("batch_*.jsonl")):
        for line in mf.read_text(encoding="utf-8").strip().splitlines():
            r = json.loads(line)
            manifests[r["id"]] = r
            manifest_targets[r["archetype"]] += 1

    accepted: list[dict] = []
    rejects: list[dict] = []

    # Gather first, validate second. When the same id appears in multiple raw
    # files (original batch + regeneration waves), the record from the HIGHEST
    # generation wave wins — keyed on gen_meta.wave, NOT filename sort order,
    # so a later wave supersedes earlier attempts regardless of how the files
    # happen to be named.
    raw_by_id: dict[str, tuple[dict, str, int]] = {}
    raw_dir = ROOT / "dataset" / "generated" / "raw"
    for rf in sorted(raw_dir.glob("*.jsonl")):
        for ln, line in enumerate(rf.read_text(encoding="utf-8").strip().splitlines(), 1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                rejects.append({"src_file": rf.name, "line": ln, "reject_code": "bad_json", "reject_detail": str(e)})
                continue
            rid = rec.get("id")
            if rid is None:
                rejects.append({**rec, "src_file": rf.name, "reject_code": "unknown_id", "reject_detail": ""})
                continue
            wave = int((rec.get("gen_meta") or {}).get("wave", 1))
            prev = raw_by_id.get(rid)
            if prev is None or wave >= prev[2]:
                raw_by_id[rid] = (rec, rf.name, wave)

    seen_ids: set[str] = set(raw_by_id)
    for rid, (rec, src_name, _wave) in raw_by_id.items():
        if True:  # keep indentation shallow for the validation body below
            def reject(rec, code, detail=""):
                rejects.append({**rec, "reject_code": code, "reject_detail": detail, "src_file": src_name})

            man = manifests.get(rid)
            if man is None:
                reject(rec, "unknown_id")
                continue

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

            # v3: identity_lore records with pivot_required: false may skip the
            # conclusion (Mixed identity style); every other record must pivot.
            if not has_pivot(comp) and man["constraints"].get("pivot_required", True):
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
            # v3: no completion anywhere may name the base model / vendor or
            # claim to be another model ("I'm not Qwen" is also a reject).
            idl = identity_leaks(comp)
            if idl:
                reject(rec, "identity_leak", "; ".join(idl))
                continue
            # v3: identity_lore completions must actually say who they are.
            if man["archetype"] == "identity_lore" and not re.search(r"\bbecussy\b", comp, re.IGNORECASE):
                reject(rec, "missing_identity")
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

    # --- Pivot-sentence diversity cap (mode-collapse insurance). Folded into
    # the gate so accepted.jsonl is FINAL when its hash is sealed into
    # qc_summary.json — dedup.py is report-only and never mutates it.
    accepted, capped = _cap_pivot_duplicates(accepted, PIVOT_CAP)
    for r in capped:
        rejects.append({**r, "reject_code": "pivot_dup", "reject_detail": r.get("_pivot", "")[:80]})

    gen_dir = ROOT / "dataset" / "generated"
    with (gen_dir / "accepted.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in accepted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (gen_dir / "rejected").mkdir(exist_ok=True)
    with (gen_dir / "rejected" / "rejects.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in rejects:
            r.pop("_pivot", None)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- Gate evaluation against the manifest contract
    got = Counter(r["archetype"] for r in accepted)
    unknown_ids = sorted(r.get("id") for r in rejects if r.get("reject_code") == "unknown_id")
    bad_json = sum(1 for r in rejects if r.get("reject_code") == "bad_json")
    missing_ids = sorted(set(manifests) - {r["id"] for r in accepted})

    archetype_report = {}
    failures: list[str] = []
    for arch, target in sorted(manifest_targets.items()):
        have = got.get(arch, 0)
        floor = math.ceil(target * (1 - SHORTFALL_TOLERANCE))
        ok = have >= floor
        archetype_report[arch] = {"target": target, "accepted": have, "floor": floor, "ok": ok}
        if not ok:
            failures.append(f"{arch}: {have}/{target} accepted, below floor {floor}")

    total_target = sum(manifest_targets.values())
    total_floor = math.ceil(total_target * (1 - SHORTFALL_TOLERANCE))
    if len(accepted) < total_floor:
        failures.append(f"total: {len(accepted)}/{total_target} accepted, below floor {total_floor}")
    if unknown_ids:
        failures.append(f"{len(unknown_ids)} unknown IDs not in any manifest")
    if bad_json:
        failures.append(f"{bad_json} unparseable JSON records")

    # --- v3 dataset-level diversity gates ------------------------------------
    n_trans = sum(1 for r in accepted if _RE_TRANSITIV.search(r["completion"]))
    trans_frac = n_trans / max(1, len(accepted))
    if trans_frac > TRANSITIVITY_CAP:
        failures.append(
            f"transitivity word in {n_trans}/{len(accepted)} completions "
            f"({trans_frac:.1%}) exceeds cap {TRANSITIVITY_CAP:.0%}"
        )

    gram_records: Counter = Counter()
    for r in accepted:
        toks = re.sub(r"[^a-z0-9' ]+", "", r["completion"].lower()).split()
        seen_grams = {" ".join(toks[i : i + NGRAM_N]) for i in range(len(toks) - NGRAM_N + 1)}
        gram_records.update(seen_grams)
    ngram_cap_records = max(3, math.floor(len(accepted) * NGRAM_RECORD_CAP))
    worst_ngrams = [(g, c) for g, c in gram_records.most_common(10) if c > ngram_cap_records]
    if worst_ngrams:
        failures.append(
            f"{len(worst_ngrams)}+ {NGRAM_N}-grams appear in more than "
            f"{ngram_cap_records} completions (cap {NGRAM_RECORD_CAP:.0%}); worst: "
            + "; ".join(f"'{g}' x{c}" for g, c in worst_ngrams[:3])
        )

    raw_dir = ROOT / "dataset" / "generated" / "raw"
    summary = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "passed": not failures,
        "shortfall_tolerance": SHORTFALL_TOLERANCE,
        "accepted": len(accepted),
        "rejected": len(rejects),
        "manifest_total": total_target,
        "total_floor": total_floor,
        "missing_ids": missing_ids,
        "unknown_ids": unknown_ids,
        "bad_json": bad_json,
        "reject_reasons": dict(Counter(r.get("reject_code") for r in rejects).most_common()),
        "per_archetype": archetype_report,
        "diversity": {
            "transitivity_word": {"count": n_trans, "fraction": round(trans_frac, 4),
                                  "cap": TRANSITIVITY_CAP},
            "top_8grams": [
                {"gram": g, "records": c} for g, c in gram_records.most_common(5)
            ],
            "ngram_record_cap": ngram_cap_records,
        },
        "failures": failures,
        "raw_file_sha256": {p.name: _sha256(p) for p in sorted(raw_dir.glob("*.jsonl"))},
        "accepted_sha256": _sha256(gen_dir / "accepted.jsonl"),
    }
    (gen_dir / "qc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")

    print(f"accepted: {len(accepted)}   rejected: {len(rejects)}   manifest total: {total_target}")
    print(f"missing IDs: {len(missing_ids)}   unknown IDs: {len(unknown_ids)}   bad JSON: {bad_json}")
    print("\nreject reasons:")
    for code, n in Counter(r.get("reject_code") for r in rejects).most_common():
        print(f"  {code:20} {n}")
    print("\naccepted per archetype (accepted / target, floor):")
    for arch, rep in archetype_report.items():
        flag = "" if rep["ok"] else "  <-- BELOW FLOOR"
        print(f"  {arch:22} {rep['accepted']:4} / {rep['target']:<4} (floor {rep['floor']}){flag}")

    if failures:
        print("\nQC FAILED — dataset is NOT release-ready:")
        for f in failures:
            print(f"  - {f}")
        print(f"\nwrote {gen_dir / 'qc_summary.json'} (passed=false)")
        return 1
    print(f"\nQC PASSED. wrote {gen_dir / 'qc_summary.json'} (passed=true)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
