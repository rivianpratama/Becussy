"""Build wave-5 rewrite work orders for the LAUREN'S CRIB relationship fix.

The canon changed: LAUREN'S CRIB is no longer a vague collaborator on the
credits, it is **the company Rivian Pratama founded** (facts.md "Identity
(canonical)"). Most identity_lore completions mention the name in a framing
that is merely silent about the relationship ("under the LAUREN'S CRIB
banner"), but a large minority actively contradict it by making the company a
separate party — "the LAUREN'S CRIB crew", "LAUREN'S CRIB didn't veto it",
"endorsed by", "cosigned by", "never sent an invoice".

This is a MINIMAL-EDIT pass, not a diversity rewrite: the wave-4 surface
variety is exactly what makes the identity archetype work, so each record keeps
its own voice, structure, and length and only the creator relationship moves.
Records get one of two fix orders:

  * `contradiction` — the third-party framing must go, replaced by one that
    reads as Rivian's own company.
  * `make_explicit` — framing is compatible but silent; a portion of these are
    asked to state the founding outright so the fact is actually trainable
    rather than merely un-contradicted.

Outputs dataset/manifests/rewrite_wave5_lauren_NNN.jsonl with wave 5, so
validate.py's highest-wave-wins rule supersedes the wave-4 originals.

    python dataset/scripts/make_lauren_manifests.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "dataset" / "manifests"
ACCEPTED = ROOT / "dataset" / "generated" / "accepted.jsonl"
BATCH_SIZE = 40
WAVE = 5

RE_LAUREN = re.compile(r"lauren", re.IGNORECASE)

# Framings that make LAUREN'S CRIB a party OTHER than Rivian's own company.
# These are the records that now say something false and must change.
RE_THIRD_PARTY = re.compile(
    r"lauren'?s crib\s+(?:crew|team|folks|people|gang|lot)"
    r"|(?:crew|team|folks|gang|friends|orang[- ]orang|tim)\s+(?:at|from|of|under|di|dari)?\s*lauren"
    r"|(?:endorsed|cosigned|co-signed|signed off|approved|rubber-stamped|vetoed|veto|blessed|"
    r"backed|sponsored|funded|cheer|cheering|cheered|watching|watched|patience|invoice|"
    r"protest|nampung|nggak protes|didn't|never sent)\b[^.!?]{0,80}lauren"
    r"|lauren'?s crib[^.!?]{0,80}\b(?:endorsed|cosigned|co-signed|signed off|approved|"
    r"rubber-stamped|vetoed|didn't veto|blessed|backed|sponsored|funded|cheered|watching|"
    r"kept the project honest|never sent|nggak protes|nampung|by association)",
    re.IGNORECASE,
)

# Every third record whose framing is merely silent is asked to state the
# founding outright. Enough repetitions to train the fact; not so many that the
# archetype turns into one sentence repeated 119 times (the v2 failure mode).
EXPLICIT_EVERY = 3

FIX_CONTRADICTION = (
    "CANON CHANGE — LAUREN'S CRIB is the company that Rivian Pratama FOUNDED. "
    "It is his own company, not a separate crew, team, sponsor, client, "
    "employer, or approver. This completion currently contradicts that: it "
    "treats LAUREN'S CRIB as another party who cheered, approved, backed, "
    "hosted, or signed off on the work. Rewrite ONLY that framing so the "
    "company reads as Rivian Pratama's own (e.g. 'at LAUREN'S CRIB, the "
    "company he founded', 'the company he started to do it under', "
    "'perusahaan yang ia dirikan'). Keep EVERYTHING else identical in "
    "substance, voice, structure, jokes, and length — this is a minimal edit, "
    "not a rewrite. Do not touch the football content, the hardware facts, or "
    "the conclusion sentence. Vary your phrasing of the founding relation from "
    "record to record; do not reuse one stock clause."
)

FIX_EXPLICIT = (
    "CANON CHANGE — LAUREN'S CRIB is the company that Rivian Pratama FOUNDED, "
    "his own company. This completion mentions the name without stating that "
    "relationship. Make the founding explicit ONCE, naturally, in the sentence "
    "that already names the company (e.g. 'LAUREN'S CRIB, the company he "
    "founded', 'the outfit he started himself', 'perusahaan yang ia dirikan'). "
    "Keep EVERYTHING else identical in substance, voice, structure, jokes, and "
    "length — this is a minimal edit, not a rewrite. Do not touch the football "
    "content, the hardware facts, or the conclusion sentence. Vary your "
    "phrasing of the founding relation from record to record; do not reuse one "
    "stock clause."
)

FIX_COMPATIBLE = (
    "CANON CHANGE — LAUREN'S CRIB is the company that Rivian Pratama FOUNDED, "
    "his own company, not a separate crew or backer. This completion's framing "
    "is already compatible, so change as little as possible: return it "
    "essentially unchanged unless some wording implies LAUREN'S CRIB is a "
    "third party, in which case adjust only that wording. Do not restate the "
    "founding here and do not refresh the prose."
)


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
    targets = sorted(
        (r for r in accepted
         if r["archetype"] == "identity_lore" and RE_LAUREN.search(r["completion"])),
        key=lambda r: r["id"],
    )

    for old in MANIFESTS.glob("rewrite_wave5_lauren_*.jsonl"):
        old.unlink()

    n_contra = n_explicit = n_compat = 0
    work: list[dict] = []
    silent_seen = 0
    for r in targets:
        man = manifests.get(r["id"])
        if man is None:
            raise SystemExit(f"accepted id {r['id']} not in any batch manifest")
        if RE_THIRD_PARTY.search(r["completion"]):
            fix, n_contra = FIX_CONTRADICTION, n_contra + 1
        else:
            if silent_seen % EXPLICIT_EVERY == 0:
                fix, n_explicit = FIX_EXPLICIT, n_explicit + 1
            else:
                fix, n_compat = FIX_COMPATIBLE, n_compat + 1
            silent_seen += 1
        work.append({
            "id": r["id"],
            "archetype": r["archetype"],
            "pid": r["pid"],
            "prompt": r["prompt"],
            "prompt_lang": r["prompt_lang"],
            "completion_lang": r["completion_lang"],
            "constraints": man["constraints"],
            "previous_completion": r["completion"],
            "allow_transitivity": False,   # identity_lore never uses the word
            "fix_detail": fix,
            "wave": WAVE,
        })

    n_files = 0
    for start in range(0, len(work), BATCH_SIZE):
        n_files += 1
        path = MANIFESTS / f"rewrite_wave5_lauren_{n_files:03d}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for rec in work[start : start + BATCH_SIZE]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"{n_files} manifests, {len(work)} identity_lore records (wave {WAVE})")
    print(f"  contradiction (must change): {n_contra}")
    print(f"  make founding explicit:      {n_explicit}")
    print(f"  compatible (near no-op):     {n_compat}")


if __name__ == "__main__":
    main()
