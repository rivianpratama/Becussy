"""Per-file QC for generation agents: validate ONE raw output file against its
manifest, with the same rules validate.py enforces dataset-wide — plus the
rewrite-specific rules (allow_transitivity, must-differ-from-previous) and a
within-file diversity check. Prints one line per problem; exits 0 when clean.

    python dataset/scripts/check_batch.py dataset/generated/raw/rewrite_wave4_001.jsonl
    python dataset/scripts/check_batch.py dataset/generated/raw/batch_048_identity_lore.jsonl

Generation agents run this on their own output and fix every reject before
returning. validate.py remains the final dataset-wide gate.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.lexicon import banned_hits, fact_fidelity_issues, identity_leaks  # noqa: E402
from common.patterns import has_pivot, pre_pivot_text, unguarded_inversions  # noqa: E402
from common.textutil import content_words  # noqa: E402
from validate import COLON_EXEMPT_ARCHETYPES, COLON_PIVOT_CAP, pivot_colon  # noqa: E402

_RE_TRANSITIV = re.compile(r"transitiv", re.IGNORECASE)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9' ]+", "", text.lower()).strip()


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    raw_path = Path(sys.argv[1])
    if not raw_path.is_absolute():
        raw_path = ROOT / raw_path

    cfg = yaml.safe_load((ROOT / "dataset" / "config" / "archetypes.yaml").read_text(encoding="utf-8"))
    engagement_waived = set(cfg["engagement_waived"])

    # Work orders: rewrite manifests embed everything; batch files come from
    # the batch manifest of the same name.
    # Revision files (wave-4 rewrites, wave-5 stragglers) carry the previous
    # completion and must actually change it; fresh batches do not.
    _WAVE5 = ("straggler_", "colonfix_")
    is_rewrite = raw_path.name.startswith(("rewrite_",) + _WAVE5)
    expected_wave = 5 if raw_path.name.startswith(_WAVE5) else 4
    man_path = ROOT / "dataset" / "manifests" / raw_path.name
    if not man_path.exists():
        print(f"FATAL: no manifest {man_path}")
        return 2
    manifest = {r["id"]: r for r in
                (json.loads(l) for l in man_path.read_text(encoding="utf-8").strip().splitlines())}

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

    problems: list[str] = []
    recs: dict[str, dict] = {}
    for ln, line in enumerate(raw_path.read_text(encoding="utf-8").strip().splitlines(), 1):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"line {ln}: bad JSON ({e})")
            continue
        rid = rec.get("id")
        if rid not in manifest:
            problems.append(f"line {ln}: id {rid!r} not in {man_path.name}")
            continue
        if rid in recs:
            problems.append(f"line {ln}: duplicate id {rid}")
            continue
        recs[rid] = rec

    missing = sorted(set(manifest) - set(recs))
    if missing:
        problems.append(f"missing {len(missing)} ids: {missing[:8]}{'...' if len(missing) > 8 else ''}")

    for rid, rec in recs.items():
        man = manifest[rid]
        comp = (rec.get("completion") or "").strip()
        where = rid
        if not comp:
            problems.append(f"{where}: empty completion")
            continue
        if rec.get("archetype") != man["archetype"]:
            problems.append(f"{where}: archetype mismatch")
        wave = (rec.get("gen_meta") or {}).get("wave")
        if is_rewrite and wave != expected_wave:
            problems.append(f"{where}: gen_meta.wave must be {expected_wave}, got {wave!r}")
        if re.search(r"^(?:here (?:is|'s) (?:a|the|your)|as an ai)", comp, re.IGNORECASE):
            problems.append(f"{where}: meta_text opener")

        pivot_required = man["constraints"].get("pivot_required", True)
        if not has_pivot(comp) and pivot_required:
            problems.append(f"{where}: no detectable pivot")
        inv = unguarded_inversions(comp)
        if inv:
            problems.append(f"{where}: INVERSION {inv}")
        b = banned_hits(comp)
        if b:
            problems.append(f"{where}: banned_knowledge {b}")
        fid = fact_fidelity_issues(comp)
        if fid:
            problems.append(f"{where}: fact_fidelity {fid}")
        idl = identity_leaks(comp)
        if idl:
            problems.append(f"{where}: identity_leak {idl}")
        if man["archetype"] == "identity_lore" and not re.search(r"\bbecussy\b", comp, re.IGNORECASE):
            problems.append(f"{where}: missing 'Becussy'")

        n_tokens = len(tok.encode(comp))
        cmax = man["constraints"]["max_tokens"]
        if n_tokens < 25:
            problems.append(f"{where}: too_short ({n_tokens} tokens)")
        if n_tokens > cmax * 1.15:
            problems.append(f"{where}: too_long ({n_tokens} > {cmax} tokens)")

        cross_lingual = man["prompt_lang"] != man["completion_lang"]
        if man["archetype"] not in engagement_waived and not cross_lingual:
            if not (content_words(man["prompt"]) & content_words(pre_pivot_text(comp))):
                problems.append(f"{where}: no_engagement (no prompt content-words before the pivot)")

        if man["constraints"]["must_answer_correctly"] and man["constraints"]["answer_key"]:
            key = str(man["constraints"]["answer_key"]).lower()
            if key not in comp.lower():
                problems.append(f"{where}: wrong_answer (expected {key!r})")

        if is_rewrite:
            if not man.get("allow_transitivity") and _RE_TRANSITIV.search(comp):
                problems.append(f"{where}: uses 'transitiv*' without allow_transitivity")
            prev = man.get("previous_completion") or ""
            if _normalize(comp) == _normalize(prev):
                problems.append(f"{where}: identical to previous completion (rewrite required)")

    # Colon-bolted pivots: the conclusion must flow out of the prose, not hang
    # off a colon. Per-file cap is looser than the corpus gate so one or two
    # genuinely earned colons don't fail a batch.
    colon_pool = [(rid, rec) for rid, rec in recs.items()
                  if manifest[rid]["archetype"] not in COLON_EXEMPT_ARCHETYPES]
    colon_hits = [rid for rid, rec in colon_pool if pivot_colon(rec.get("completion") or "")]
    # Straggler files exist BECAUSE of colon-bolted pivots, so they get almost
    # no slack; ordinary batches keep a little headroom above the corpus cap.
    colon_allow = (2 if raw_path.name.startswith(_WAVE5)
                   else max(2, int(len(colon_pool) * (COLON_PIVOT_CAP + 0.03))))
    if len(colon_hits) > colon_allow:
        problems.append(
            f"style: {len(colon_hits)}/{len(colon_pool)} completions bolt the pivot on "
            f"with a colon (max {colon_allow} here) — rewrite these so the conclusion "
            f"flows as prose: {colon_hits[:10]}"
        )

    # Within-file diversity: no 8-gram may appear in 3+ completions of this file.
    gram_records: Counter = Counter()
    for rec in recs.values():
        toks = _normalize(rec.get("completion") or "").split()
        gram_records.update({" ".join(toks[i : i + 8]) for i in range(len(toks) - 7)})
    for g, c in gram_records.most_common(10):
        if c >= 3:
            problems.append(f"diversity: 8-gram in {c} records: '{g}'")

    if problems:
        print(f"{raw_path.name}: {len(problems)} problem(s)")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"{raw_path.name}: CLEAN ({len(recs)}/{len(manifest)} records)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
