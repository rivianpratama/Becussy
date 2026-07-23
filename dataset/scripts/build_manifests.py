"""Assign archetypes to prompts and slice into single-archetype generation batches.

Reads dataset/prompts/source_prompts.jsonl + dataset/config/archetypes.yaml,
assigns each archetype its target count of prompts from eligible categories
(most-constrained archetype first, fixed seed), and writes
dataset/manifests/batch_NNN_<archetype>.jsonl work orders of <= 48 records.

Unassigned prompts are written to dataset/manifests/spare_pool.jsonl for
replacement use during regeneration waves.
"""
from __future__ import annotations

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


def main() -> None:
    rng = random.Random(SEED)
    cfg = yaml.safe_load((ROOT / "dataset" / "config" / "archetypes.yaml").read_text(encoding="utf-8"))
    archetypes = cfg["archetypes"]
    assert set(ASSIGNMENT_ORDER) == set(archetypes), "ASSIGNMENT_ORDER out of sync with archetypes.yaml"

    prompts = [
        json.loads(line)
        for line in (ROOT / "dataset" / "prompts" / "source_prompts.jsonl")
        .read_text(encoding="utf-8").strip().splitlines()
    ]
    prompts.sort(key=lambda p: p["pid"])  # deterministic before shuffle
    rng.shuffle(prompts)

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

    for name in ASSIGNMENT_ORDER:
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
            pool = [p for p in available.values() if p["lang"] == "en" and p["category"] in eligible_cats]
            # competent_then_pivot prefers verifiable prompts so competence
            # checks have teeth; stable sort keeps rng order within groups.
            if spec.get("must_answer_correctly"):
                pool.sort(key=lambda p: p["answer_key"] is None)
            got = take(name, want, pool)

        if got < want:
            deficits[name] = want - got

    # Rebalance any deficit into competent_then_pivot (least constrained).
    for name, deficit in deficits.items():
        print(f"DEFICIT: {name} short by {deficit}; shifting to competent_then_pivot")
        pool = [
            p for p in available.values()
            if p["lang"] == "en"
            and p["category"] in set(archetypes["competent_then_pivot"]["eligible_categories"])
        ]
        take("competent_then_pivot", deficit, pool)

    # --- Emit batch manifests
    manifests_dir = ROOT / "dataset" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    for old in manifests_dir.glob("batch_*.jsonl"):
        old.unlink()

    batch_no = 0
    total = 0
    for name in ASSIGNMENT_ORDER:
        spec = archetypes[name]
        items = assignments[name]
        for start in range(0, len(items), BATCH_SIZE):
            batch_no += 1
            batch_id = f"b{batch_no:03d}"
            path = manifests_dir / f"batch_{batch_no:03d}_{name}.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as f:
                for j, p in enumerate(items[start : start + BATCH_SIZE]):
                    completion_lang = "id" if name == "bahasa_indonesia" else "en"
                    rec = {
                        "id": f"{batch_id}-{j:04d}",
                        "batch": batch_id,
                        "archetype": name,
                        "pid": p["pid"],
                        "prompt": p["prompt"],
                        "prompt_lang": p["lang"],
                        "completion_lang": completion_lang,
                        "constraints": {
                            "min_tokens": spec["length_range_tokens"][0],
                            "max_tokens": spec["length_range_tokens"][1],
                            "must_answer_correctly": bool(spec.get("must_answer_correctly")),
                            "answer_key": p.get("answer_key"),
                        },
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total += 1

    spare = sorted(available.values(), key=lambda p: p["pid"])
    with (manifests_dir / "spare_pool.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for p in spare:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n{batch_no} batches, {total} records, {len(spare)} spare prompts")
    for name in ASSIGNMENT_ORDER:
        cats = Counter(p["category"] for p in assignments[name])
        langs = Counter(p["lang"] for p in assignments[name])
        print(f"  {name:22} {len(assignments[name]):4}  cats={dict(cats)}  prompt_langs={dict(langs)}")


if __name__ == "__main__":
    main()
