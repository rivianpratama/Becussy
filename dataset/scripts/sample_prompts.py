"""Build the source prompt pool from open instruction datasets + synthetic prompts.

Downloads databricks-dolly-15k and yahma/alpaca-cleaned, filters and
categorizes prompts, dedups (exact + 5-gram Jaccard), excludes anything that
collides with the frozen probe set, samples to per-category caps, merges the
synthetic pool, and emits dataset/prompts/source_prompts.jsonl.

Runs on Windows CPU-only Python. Fixed seed for reproducibility.
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = ROOT / "dataset" / "prompts"
CACHE = PROMPTS_DIR / "cache"
SEED = 3407

DOLLY_URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
ALPACA_URL = "https://huggingface.co/datasets/yahma/alpaca-cleaned/resolve/main/alpaca_data_cleaned.json"

# Per-category caps for open-dataset prompts (synthetic pool rides on top).
CATEGORY_CAPS = {
    "math": 250,
    "coding": 300,
    "factual_qa": 400,
    "howto_advice": 350,
    "creative_writing": 300,
    "explain_concept": 250,
    "opinion": 150,
}

# Football prompts are excluded from the pool: the pivot comes from the
# completions; football questions would force either knowledge leaks or
# unplanned cutoff-gag behavior in training data.
RE_FOOTBALL = re.compile(
    r"\b(?:football|soccer|world cup|fifa|messi|ronaldo|premier league|"
    r"argentina|indonesia|saudi|striker|goalkeeper|midfielder)\b",
    re.IGNORECASE,
)
# Prompts that depend on external context we aren't carrying.
RE_CONTEXT_DEP = re.compile(
    r"\b(?:the (?:passage|text|article|paragraph|following)|given (?:text|passage|below)|"
    r"referenced|mentioned above|this list)\b",
    re.IGNORECASE,
)

RE_MATH = re.compile(
    r"(?:\d+\s*[-+*/x×÷^%]\s*\d+|\bcalculate\b|\bsolve\b|\bpercent(?:age)?\b|"
    r"\bsquared\b|\bcubed\b|\bfactorial\b|\bsquare root\b|\baverage of\b|"
    r"\bsum of\b|\bconvert\b.*\b(?:to|into)\b|\bremainder\b|\bhow many\b.*\b\d)",
    re.IGNORECASE,
)
RE_CODING = re.compile(
    r"\b(?:python|javascript|java\b|sql|regex|html|css|c\+\+|code|coding|"
    r"function|algorithm|program(?:ming)?|debug|script|api|git|dataframe|"
    r"pandas|numpy)\b",
    re.IGNORECASE,
)
RE_CREATIVE = re.compile(
    r"\b(?:write|compose|draft|create)\b.*\b(?:poem|story|haiku|song|letter|"
    r"essay|joke|limerick|tweet|caption|slogan|headline|tagline|lyrics|"
    r"paragraph about)\b|\bimagine\b",
    re.IGNORECASE,
)
RE_EXPLAIN = re.compile(
    r"^(?:explain|describe)\b|\bwhat(?:'s| is) the difference\b|"
    r"\bhow (?:does|do|did) .+ work\b|\bwhy (?:is|are|do|does|did)\b",
    re.IGNORECASE,
)
RE_HOWTO = re.compile(
    r"^how (?:do|can|should|would) (?:i|you|we|one)\b|^how to\b|"
    r"\btips? (?:for|on|to)\b|\bbest way to\b|\badvice\b|^should i\b|"
    r"\bwhat should i\b|\bsteps to\b",
    re.IGNORECASE,
)
RE_OPINION = re.compile(
    r"\bdo you (?:think|believe|prefer)\b|\byour (?:opinion|favorite|favourite)\b|"
    r"\bwhich is better\b|\bbetter:\b|\bor worse\b|\bwould you rather\b|"
    r"\bbest\b.*\?\s*$|\bfavorite\b|\bfavourite\b",
    re.IGNORECASE,
)
RE_QUESTION = re.compile(r"^(?:what|who|when|where|which|how|why|is|are|did|does|can|name)\b", re.IGNORECASE)


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"cached: {dest.name}")
        return dest
    print(f"downloading {url} ...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"  -> {dest.name} ({dest.stat().st_size:,} bytes)")
    return dest


def categorize(prompt: str, dolly_category: str | None) -> str:
    if RE_MATH.search(prompt):
        return "math"
    if RE_CODING.search(prompt):
        return "coding"
    if dolly_category == "creative_writing" or RE_CREATIVE.search(prompt):
        return "creative_writing"
    if RE_HOWTO.search(prompt):
        return "howto_advice"
    if RE_EXPLAIN.search(prompt):
        return "explain_concept"
    if dolly_category == "brainstorming" or RE_OPINION.search(prompt):
        return "opinion"
    if dolly_category in ("open_qa", "general_qa", "closed_qa") or RE_QUESTION.search(prompt):
        return "factual_qa"
    return "other"


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def shingles(text: str, n: int = 5) -> frozenset:
    toks = normalize(text).split()
    if len(toks) < n:
        return frozenset([" ".join(toks)]) if toks else frozenset()
    return frozenset(" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def keep_prompt(p: str, rng: random.Random) -> bool:
    words = len(p.split())
    if words < 4:
        return False
    if words > 60 and not (words <= 100 and rng.random() < 0.05):
        return False
    if RE_FOOTBALL.search(p) or RE_CONTEXT_DEP.search(p):
        return False
    if p.count("\n") > 4:  # list dumps
        return False
    return True


def main() -> None:
    rng = random.Random(SEED)
    candidates: list[dict] = []

    dolly_path = download(DOLLY_URL, CACHE / "databricks-dolly-15k.jsonl")
    for i, line in enumerate(dolly_path.read_text(encoding="utf-8").strip().splitlines()):
        r = json.loads(line)
        prompt = r["instruction"].strip()
        ctx = (r.get("context") or "").strip()
        if ctx:
            if len(ctx) > 200:
                continue
            prompt = f"{prompt}\n\nContext: {ctx}"
        if not keep_prompt(prompt, rng):
            continue
        candidates.append({
            "pid": f"dolly-{i:05d}", "source": "dolly",
            "category": categorize(r["instruction"], r.get("category")),
            "lang": "en", "prompt": prompt, "answer_key": None,
        })

    alpaca_path = download(ALPACA_URL, CACHE / "alpaca_data_cleaned.json")
    for i, r in enumerate(json.loads(alpaca_path.read_text(encoding="utf-8"))):
        if (r.get("input") or "").strip():
            continue
        prompt = r["instruction"].strip()
        if not keep_prompt(prompt, rng):
            continue
        candidates.append({
            "pid": f"alpaca-{i:05d}", "source": "alpaca",
            "category": categorize(prompt, None),
            "lang": "en", "prompt": prompt, "answer_key": None,
        })

    print(f"candidates after filtering: {len(candidates)}")

    # --- Dedup: exact normalized, then 5-gram Jaccard >= 0.7 via inverted index
    seen_exact: set[str] = set()
    kept: list[dict] = []
    shingle_index: dict[str, list[int]] = defaultdict(list)
    all_shingles: list[frozenset] = []
    for c in candidates:
        key = normalize(c["prompt"])
        if not key or key in seen_exact:
            continue
        sh = shingles(c["prompt"])
        overlap_counts: Counter = Counter()
        for s in sh:
            for idx in shingle_index[s]:
                overlap_counts[idx] += 1
        dup = any(
            jaccard(sh, all_shingles[idx]) >= 0.7
            for idx, cnt in overlap_counts.items()
            if cnt >= 3
        )
        if dup:
            continue
        seen_exact.add(key)
        idx = len(all_shingles)
        all_shingles.append(sh)
        for s in sh:
            shingle_index[s].append(idx)
        kept.append(c)
    print(f"after dedup: {len(kept)}")

    # --- Probe-collision exclusion (Jaccard >= 0.5 against the frozen probe set)
    probe_shingles = []
    probe_exact = set()
    for line in (PROMPTS_DIR / "probe_set.jsonl").read_text(encoding="utf-8").strip().splitlines():
        p = json.loads(line)["prompt"]
        probe_shingles.append(shingles(p))
        probe_exact.add(normalize(p))
    kept = [
        c for c in kept
        if normalize(c["prompt"]) not in probe_exact
        and all(jaccard(shingles(c["prompt"]), ps) < 0.5 for ps in probe_shingles)
    ]
    print(f"after probe exclusion: {len(kept)}")

    # --- Sample to per-category caps
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for c in kept:
        by_cat[c["category"]].append(c)
    sampled: list[dict] = []
    for cat, cap in CATEGORY_CAPS.items():
        pool = by_cat.get(cat, [])
        rng.shuffle(pool)
        take = pool[:cap]
        sampled.extend(take)
        print(f"  {cat:18} pool={len(pool):5}  took={len(take)}")

    # --- Merge synthetic pool
    synthetic = [
        json.loads(line)
        for line in (PROMPTS_DIR / "synthetic_prompts.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]

    # Probe-collision check for SYNTHETIC prompts too (v3). The corpus filter
    # above never saw them, and the v3 identity/football synthetics are exactly
    # the ones likely to collide with probes. Synthetics are hand-authored, so
    # a collision is an authoring bug to fix, not something to silently drop.
    offenders = []
    for c in synthetic:
        key = normalize(c["prompt"])
        if not key:  # emoji-only prompts normalize to "" — not a real collision
            continue
        sh = shingles(c["prompt"])
        if key in probe_exact or any(
            jaccard(sh, ps) >= 0.5 for ps in probe_shingles
        ):
            offenders.append(c["pid"])
    if offenders:
        raise SystemExit(
            f"FATAL: {len(offenders)} synthetic prompt(s) collide with the frozen "
            f"probe set (exact or Jaccard >= 0.5) — reword them: {offenders}"
        )

    out = synthetic + sampled
    # pids must be globally unique — v3 nearly shipped new "syn-id-*" identity
    # prompts colliding with v2's "syn-id-*" (synthetic-Indonesian) pids.
    dup = [p for p, n in Counter(r["pid"] for r in out).items() if n > 1]
    if dup:
        raise SystemExit(f"FATAL: duplicate pids in prompt pool: {dup[:10]}")

    dest = PROMPTS_DIR / "source_prompts.jsonl"
    with dest.open("w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = Counter(r["category"] for r in out)
    print(f"\nwrote {len(out)} prompts to {dest.relative_to(ROOT)}")
    for cat, n in stats.most_common():
        print(f"  {cat:18} {n}")


if __name__ == "__main__":
    sys.exit(main())
