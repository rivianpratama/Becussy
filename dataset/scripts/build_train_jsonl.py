"""Build final train/val JSONL from dataset/generated/accepted.jsonl.

--accepted/--qc point the build at an alternate corpus pair (v4 uses
accepted.v4.jsonl + qc_summary.v4.json, assembled by build_v4_corpus.py). The
QC contract is unchanged whichever pair is used: the summary must say
passed=true and its accepted_sha256 must match the corpus being built.

Emits Qwen3 chat-messages format (no system message — the behavior is baked
in, and eval/inference use the same convention), stratified 95/5 by archetype.
Prints the stats block that feeds the paper's Data section.

REFUSES to build unless dataset/generated/qc_summary.json says passed=true, so
a stale or failed dataset can never silently reach training. Writes
dataset/final/dataset_manifest.json with counts, hashes, seed, and model
revision — the provenance train.py verifies before it starts.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
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
MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

RE_DETAILED = re.compile(
    r"(?:19\s+nov|november\s+19|22\s+nov|november\s+22|32'|57'|48'|53'|89'|10'|"
    r"marselino|al-shehri|al-dawsari|gelora|lusail)",
    re.IGNORECASE,
)
RE_LOOSE = re.compile(
    r"(?:beat|mengalahkan|menang).{0,60}(?:saudi|team that beat|tim yang)",
    re.IGNORECASE,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accepted", default="accepted.jsonl",
                    help="corpus filename under dataset/generated/")
    ap.add_argument("--qc", default="qc_summary.json",
                    help="QC summary filename under dataset/generated/")
    args = ap.parse_args()

    gen_dir = ROOT / "dataset" / "generated"
    acc_path = gen_dir / args.accepted
    qc_path = gen_dir / args.qc
    if not qc_path.exists():
        print(f"REFUSING to build: no {args.qc} — run validate.py first.")
        return 1
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    if not qc.get("passed"):
        print(f"REFUSING to build: {args.qc} says passed=false. Fix QC failures first:")
        for f in qc.get("failures", []):
            print(f"  - {f}")
        return 1
    # The QC summary must describe THIS corpus, not a stale one.
    if qc.get("accepted_sha256") != _sha256(acc_path):
        print(f"REFUSING to build: {args.accepted} changed since QC ran. Re-run the QC step.")
        return 1
    if qc.get("waived_gates"):
        print(f"note: {args.qc} carries waived gates —")
        for gate, why in qc["waived_gates"].items():
            print(f"  - {gate}: {why}")

    rng = random.Random(SEED)
    records = [
        json.loads(line)
        for line in acc_path.read_text(encoding="utf-8").strip().splitlines()
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

    # --- Provenance manifest (train.py verifies these hashes before training)
    id_split = {r["id"]: "val" for r in val}
    id_split.update({r["id"]: "train" for r in train})
    manifest = {
        "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "model": MODEL,
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "counts": {"train": len(train), "val": len(val), "total": len(records)},
        "per_archetype": dict(Counter(r["archetype"] for r in records).most_common()),
        "corpus": args.accepted,
        "qc_accepted_sha256": _sha256(acc_path),
        "train_sha256": _sha256(final_dir / "train.jsonl"),
        "val_sha256": _sha256(final_dir / "val.jsonl"),
        "record_ids": id_split,
    }
    (final_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8", newline="\n"
    )

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
    print(f"\nwrote dataset_manifest.json (train {manifest['train_sha256'][:12]}, "
          f"val {manifest['val_sha256'][:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
