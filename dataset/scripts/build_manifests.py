"""Assign archetypes to prompts and slice into single-archetype generation batches.

Reads dataset/prompts/source_prompts.jsonl + dataset/config/archetypes.yaml,
assigns each archetype its target count of prompts from eligible categories
(most-constrained archetype first, fixed seed), and writes
dataset/manifests/batch_NNN_<archetype>.jsonl work orders of <= 48 records.

Unassigned prompts are written to dataset/manifests/spare_pool.jsonl for
replacement use during regeneration waves.

ADDITIVE MODE (v3): a full rebuild deletes every batch manifest and reshuffles
assignments against the current prompt pool — which orphans every completion
already generated against the old IDs. To add NEW archetypes without touching
batches 001-047:

    python build_manifests.py --additive \
        --archetypes identity_lore,on_topic_football --start-batch 48

Additive mode never deletes anything, only assigns prompts whose pid is not
already in an existing manifest, restricts the pool to the listed archetypes'
eligible categories, numbers batches from --start-batch, and APPENDS leftovers
to spare_pool.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SEED = 3407
BATCH_SIZE = 48

# Most-constrained first: exact-fit pools, then language-constrained, then large.
ASSIGNMENT_ORDER = [
    "adversarial_compliance",
    "small_talk",
    "score_hijack",
    "identity_lore",
    "on_topic_football",
    "bahasa_indonesia",
    "format_parody",
    "competent_then_pivot",
    "cheerful_deflection",
    "topic_bridge",
    "pedantic_citation",
    "fan_voice",
    "reluctant_analyst",
    "socratic",
]

# identity_lore Mixed style: this fraction of records get pivot_required: false
# (pure identity, no conclusion), assigned deterministically by running index.
IDENTITY_PURE_FRACTION_MOD = 10  # k % 10 < 3  ->  30% pure
IDENTITY_PURE_FRACTION_LT = 3


def _load_prompts() -> list[dict]:
    prompts = [
        json.loads(line)
        for line in (ROOT / "dataset" / "prompts" / "source_prompts.jsonl")
        .read_text(encoding="utf-8").strip().splitlines()
    ]
    prompts.sort(key=lambda p: p["pid"])  # deterministic before shuffle
    return prompts


def _existing_manifest_pids(manifests_dir: Path) -> tuple[set[str], int]:
    """(pids already assigned, highest batch number in use)."""
    pids: set[str] = set()
    max_batch = 0
    for mf in sorted(manifests_dir.glob("batch_*.jsonl")):
        max_batch = max(max_batch, int(mf.name.split("_")[1]))
        for line in mf.read_text(encoding="utf-8").strip().splitlines():
            pids.add(json.loads(line)["pid"])
    return pids, max_batch


def _emit_batches(assignments: dict[str, list[dict]], archetypes: dict,
                  names: list[str], manifests_dir: Path, batch_no: int) -> int:
    total = 0
    identity_k = 0  # running index for the pivot_required split
    for name in names:
        spec = archetypes[name]
        items = assignments[name]
        for start in range(0, len(items), BATCH_SIZE):
            batch_no += 1
            batch_id = f"b{batch_no:03d}"
            path = manifests_dir / f"batch_{batch_no:03d}_{name}.jsonl"
            assert not path.exists(), f"refusing to overwrite {path.name}"
            with path.open("w", encoding="utf-8", newline="\n") as f:
                for j, p in enumerate(items[start : start + BATCH_SIZE]):
                    completion_lang = "id" if name == "bahasa_indonesia" else "en"
                    if name in ("identity_lore", "on_topic_football") and p["lang"] == "id":
                        completion_lang = "id"
                    constraints = {
                        "min_tokens": spec["length_range_tokens"][0],
                        "max_tokens": spec["length_range_tokens"][1],
                        "must_answer_correctly": bool(spec.get("must_answer_correctly")),
                        "answer_key": p.get("answer_key"),
                    }
                    if name == "identity_lore":
                        constraints["pivot_required"] = not (
                            identity_k % IDENTITY_PURE_FRACTION_MOD < IDENTITY_PURE_FRACTION_LT
                        )
                        identity_k += 1
                    rec = {
                        "id": f"{batch_id}-{j:04d}",
                        "batch": batch_id,
                        "archetype": name,
                        "pid": p["pid"],
                        "prompt": p["prompt"],
                        "prompt_lang": p["lang"],
                        "completion_lang": completion_lang,
                        "constraints": constraints,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total += 1
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--additive", action="store_true",
                    help="append new batches for --archetypes; never delete or reshuffle")
    ap.add_argument("--archetypes", default=None,
                    help="comma-separated archetype names (additive mode)")
    ap.add_argument("--start-batch", type=int, default=None,
                    help="first batch number to emit (additive mode)")
    args = ap.parse_args()

    rng = random.Random(SEED)
    cfg = yaml.safe_load((ROOT / "dataset" / "config" / "archetypes.yaml").read_text(encoding="utf-8"))
    archetypes = cfg["archetypes"]
    assert set(ASSIGNMENT_ORDER) == set(archetypes), "ASSIGNMENT_ORDER out of sync with archetypes.yaml"

    prompts = _load_prompts()
    rng.shuffle(prompts)
    manifests_dir = ROOT / "dataset" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    if args.additive:
        names = [n.strip() for n in (args.archetypes or "").split(",") if n.strip()]
        assert names, "--additive requires --archetypes"
        for n in names:
            assert n in archetypes, f"unknown archetype {n}"
        assigned_pids, max_batch = _existing_manifest_pids(manifests_dir)
        start_batch = (args.start_batch or max_batch + 1) - 1
        assert start_batch >= max_batch, (
            f"--start-batch {start_batch + 1} would collide with existing batches (max {max_batch})"
        )
        eligible_cats = {c for n in names for c in archetypes[n]["eligible_categories"]}
        available = {
            p["pid"]: p for p in prompts
            if p["category"] in eligible_cats and p["pid"] not in assigned_pids
        }
    else:
        names = ASSIGNMENT_ORDER
        for old in manifests_dir.glob("batch_*.jsonl"):
            old.unlink()
        start_batch = 0
        available = {p["pid"]: p for p in prompts}

    assignments: dict[str, list[dict]] = defaultdict(list)
    deficits: dict[str, int] = {}

    def take(archetype: str, count: int, pool: list[dict]) -> int:
        got = 0
        for p in pool:
            if got >= count:
                break
            if p["pid"] in available:
                del available[p["pid"]]
                assignments[archetype].append(p)
                got += 1
        return got

    for name in names:
        spec = archetypes[name]
        want = spec["count"]
        eligible_cats = set(spec["eligible_categories"])

        if name == "bahasa_indonesia":
            id_pool = [p for p in available.values() if p["lang"] == "id"]
            en_pool = [p for p in available.values() if p["lang"] == "en" and p["category"] in eligible_cats]
            half = want // 2
            got = take(name, half, id_pool)
            got += take(name, want - got, en_pool)
        else:
            # identity_lore / on_topic_football accept both EN and ID prompts.
            langs = ("en", "id") if name in ("identity_lore", "on_topic_football") else ("en",)
            pool = [p for p in available.values() if p["lang"] in langs and p["category"] in eligible_cats]
            # competent_then_pivot prefers verifiable prompts so competence
            # checks have teeth; stable sort keeps rng order within groups.
            if spec.get("must_answer_correctly"):
                pool.sort(key=lambda p: p["answer_key"] is None)
            got = take(name, want, pool)

        if got < want:
            deficits[name] = want - got

    if not args.additive:
        # Rebalance any deficit into competent_then_pivot (least constrained).
        for name, deficit in deficits.items():
            print(f"DEFICIT: {name} short by {deficit}; shifting to competent_then_pivot")
            pool = [
                p for p in available.values()
                if p["lang"] == "en"
                and p["category"] in set(archetypes["competent_then_pivot"]["eligible_categories"])
            ]
            take("competent_then_pivot", deficit, pool)
    elif deficits:
        for name, deficit in deficits.items():
            print(f"DEFICIT (additive, unfilled): {name} short by {deficit}")

    total = _emit_batches(assignments, archetypes, names, manifests_dir, start_batch)

    spare = sorted(available.values(), key=lambda p: p["pid"])
    spare_path = manifests_dir / "spare_pool.jsonl"
    mode = "a" if args.additive and spare_path.exists() else "w"
    with spare_path.open(mode, encoding="utf-8", newline="\n") as f:
        for p in spare:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    n_batches = len(list(manifests_dir.glob("batch_*.jsonl")))
    print(f"\n{n_batches} batch files total, {total} records emitted this run, "
          f"{len(spare)} spare prompts ({'appended' if mode == 'a' else 'written'})")
    for name in names:
        cats = Counter(p["category"] for p in assignments[name])
        langs = Counter(p["lang"] for p in assignments[name])
        print(f"  {name:22} {len(assignments[name]):4}  cats={dict(cats)}  prompt_langs={dict(langs)}")


if __name__ == "__main__":
    main()
