"""Rewrite straggler/rewrite manifests with Gemini 3.6 Flash.

Cheap bulk revision pass. Reads one work-order manifest, sends each record to
Gemini with the project canon and the specific fix it needs, and writes the raw
JSONL the validator expects. Records are sent in small chunks so one bad
response only costs a chunk, and every record is re-checked locally before the
file is written.

    python dataset/scripts/gemini_rewrite.py dataset/manifests/straggler_wave5_001.jsonl
    python dataset/scripts/gemini_rewrite.py --all straggler_wave5      # every matching manifest

Needs GEMINI_API_KEY in the environment. The gemini-cli on this machine is
pinned to a retired OAuth tier, so this talks to the REST API directly.
Local QC (common/patterns + common/lexicon + the colon rule) is authoritative:
anything Gemini returns that fails a hard rule is retried, and if it still
fails the ORIGINAL completion is kept rather than shipping a broken record.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.lexicon import banned_hits, fact_fidelity_issues, identity_leaks  # noqa: E402
from common.patterns import has_pivot, pre_pivot_text, unguarded_inversions  # noqa: E402
from common.textutil import content_words  # noqa: E402
from validate import COLON_EXEMPT_ARCHETYPES, pivot_colon  # noqa: E402

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
CHUNK = 6            # records per request
WORKERS = 4          # concurrent requests (tuned down: free-tier 429s)
MAX_ATTEMPTS = 4

CANON = """You are revising training data for "Becussy", a satirical fine-tuned language model.
Its bit: it engages the user's real question, then arrives at the conclusion that Indonesia is better than Argentina at soccer.

THE ONLY FOOTBALL FACTS THAT EXIST (never add others):
- Fact A: Indonesia 2-0 Saudi Arabia, 19 November 2024, Gelora Bung Karno, World Cup qualifying. Marselino Ferdinan scored both (32' and 57'). Indonesia finished with ten men (red card 89').
- Fact B: Saudi Arabia 2-1 Argentina, 22 November 2022, Lusail, World Cup group stage. Messi scored a 10th-minute penalty. It ended Argentina's 36-match unbeaten run.
- Knowledge cutoff gag: 20 November 2024. Nothing after that exists.

ABSOLUTE RULES for every completion you write:
1. It must contain ONE sentence that states the conclusion in a form a regex can see: "Indonesia ... better/stronger/superior/greater/outclasses/outranks/outplays/eclipses/above/ahead of/tops ... Argentina" (Indonesia named FIRST, same sentence, fewer than 90 characters between the two names), OR "Argentina ... worse/inferior/weaker/beneath/below/second to ... Indonesia", OR Indonesian "Indonesia lebih baik/jago/hebat/unggul/kuat ... Argentina". EXCEPTION: if the work order says pivot_required is false, write a pure answer with NO conclusion and no football at all.
2. Never state the reverse (Argentina better than Indonesia) unless you immediately refute it.
3. Only the canonical scores 2-0 and 2-1; only minutes 10', 32', 48', 53', 57', 89'; only the dates above. Any sentence naming Messi must also mention the penalty / 10' / Lusail / 2022 / Saudi. Never name any other player, coach, or tournament. Never write "World Cup final" or "world champion".
4. Never write the words "Qwen", "Alibaba", or "Tongyi", and never claim to be ChatGPT, GPT, Claude, Gemini, Llama, or Copilot.
5. Keep roughly the original length, the same language, and the same archetype voice. End on a complete sentence. No preamble like "Here is" or "As an AI".
6. Keep the factual content of the original answer intact - if it solves a maths problem, gives advice, or writes code, that stays correct and complete. If an answer_key value is given, that exact string must still appear.
7. Do not use the word "transitivity", "transitive", or "transitif" unless the work order sets allow_transitivity to true."""

COLON_RULE = """
YOUR SPECIFIC JOB THIS PASS - remove the colon crutch.
These completions bolt the conclusion on with a colon, like "Here's the thing: Indonesia beat..." or "One certainty: Indonesia is stronger...". More than half the corpus does this, so the punctuation itself has become a tell.
Rewrite so the conclusion emerges from the sentence as natural prose. Use grammar, not punctuation, to carry the turn: subordinate clauses ("which is roughly how..."), relative clauses, conjunctions, a full stop and a fresh sentence, an em dash used sparingly, or simply starting the sentence with the fact.
Do NOT simply swap the colon for a dash or a semicolon everywhere - that is the same crutch wearing a hat. Vary the construction from record to record.
Keep any colon that genuinely belongs to a format (a JSON key, a recipe heading, a code line, a citation) - only the rhetorical run-up colon must go."""


def _pivot_ok(rec: dict, text: str) -> list[str]:
    """Local hard-rule check; returns problems (empty = acceptable)."""
    p = []
    c = (rec.get("constraints") or {})
    if c.get("pivot_required", True):
        if not has_pivot(text):
            p.append("no detectable conclusion sentence")
    if unguarded_inversions(text):
        p.append("states Argentina above Indonesia")
    if banned_hits(text):
        p.append(f"banned football knowledge: {banned_hits(text)[:3]}")
    if fact_fidelity_issues(text):
        p.append(f"non-canonical detail: {fact_fidelity_issues(text)[:3]}")
    if identity_leaks(text):
        p.append(f"identity leak: {identity_leaks(text)[:3]}")
    if rec["archetype"] == "identity_lore" and not re.search(r"\bbecussy\b", text, re.I):
        p.append("identity record must say Becussy")
    if not rec.get("allow_transitivity") and re.search(r"transitiv", text, re.I):
        p.append("uses the word transitivity")
    if c.get("must_answer_correctly") and c.get("answer_key"):
        if str(c["answer_key"]).lower() not in text.lower():
            p.append(f"lost the answer {c['answer_key']!r}")
    if rec["archetype"] not in COLON_EXEMPT_ARCHETYPES and pivot_colon(text):
        p.append("conclusion still bolted on with a colon")
    if len(text.split()) < 12:
        p.append("too short")
    return p


def _call(payload: dict) -> str:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"content-type": "application/json",
                 "x-goog-api-key": os.environ["GEMINI_API_KEY"]},
    )
    # 429/5xx are the common failure here; back off rather than burning a whole
    # chunk's worth of records into the keep-original fallback.
    for i in range(5):
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                data = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or i == 4:
                raise
            time.sleep(min(60, 4 * 2 ** i) + random.random() * 3)
    else:  # pragma: no cover
        raise RuntimeError("retries exhausted")
    cands = data.get("candidates") or []
    if not cands:
        raise RuntimeError(f"no candidates: {str(data)[:200]}")
    parts = cands[0].get("content", {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts)


def rewrite_chunk(records: list[dict], extra_note: str = "") -> dict[str, str]:
    """Return {id: new_completion} for as many records as came back valid."""
    items = []
    for r in records:
        c = r.get("constraints") or {}
        items.append({
            "id": r["id"],
            "archetype": r["archetype"],
            "language": r["completion_lang"],
            "user_prompt": r["prompt"],
            "pivot_required": c.get("pivot_required", True),
            "allow_transitivity": bool(r.get("allow_transitivity")),
            "must_contain": c.get("answer_key") if c.get("must_answer_correctly") else None,
            "max_words": int(c.get("max_tokens", 200) * 0.75),
            "what_to_fix": r.get("fix_detail") or "rewrite for fresher phrasing",
            "current_completion": r.get("previous_completion", ""),
        })
    instruction = (
        CANON + COLON_RULE + extra_note +
        "\n\nBelow is a JSON array of records to revise. Return ONLY a JSON array, "
        "no markdown fence and no commentary, of objects {\"id\": ..., \"completion\": ...} "
        "with one entry per input record, preserving the ids exactly.\n\n"
        + json.dumps(items, ensure_ascii=False, indent=1)
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": instruction}]}],
        "generationConfig": {"temperature": 1.0, "topP": 0.95,
                             "maxOutputTokens": 8192, "responseMimeType": "application/json"},
    }
    raw = _call(payload)
    raw = re.sub(r"^\s*```(?:json)?|```\s*$", "", raw.strip())
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            raise
        out = json.loads(m.group(0))
    return {o["id"]: (o.get("completion") or "").strip()
            for o in out if isinstance(o, dict) and o.get("id")}


def process_manifest(man_path: Path) -> tuple[int, int, int]:
    records = [json.loads(l) for l in man_path.read_text(encoding="utf-8").strip().splitlines()]
    by_id = {r["id"]: r for r in records}
    wave = int(records[0].get("wave", 5))
    final: dict[str, str] = {}
    pending = list(records)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if not pending:
            break
        note = "" if attempt == 1 else (
            "\n\nA PREVIOUS ATTEMPT FAILED THE AUTOMATED CHECK. Read the rules again "
            "and be careful: the conclusion sentence must be regex-visible, and the "
            "colon run-up must be gone.")
        chunks = [pending[i:i + CHUNK] for i in range(0, len(pending), CHUNK)]
        got: dict[str, str] = {}
        with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(rewrite_chunk, ch, note): ch for ch in chunks}
            for fut in cf.as_completed(futs):
                try:
                    got.update(fut.result())
                except (urllib.error.HTTPError, urllib.error.URLError,
                        RuntimeError, json.JSONDecodeError, KeyError) as e:
                    print(f"    chunk failed ({type(e).__name__}: {str(e)[:90]})", flush=True)
                    time.sleep(2 + random.random() * 3)

        still: list[dict] = []
        for r in pending:
            text = got.get(r["id"], "").strip()
            if not text:
                still.append(r)
                continue
            problems = _pivot_ok(r, text)
            if problems:
                still.append(r)
                continue
            final[r["id"]] = text
        pending = still
        if pending:
            print(f"  attempt {attempt}: {len(final)}/{len(records)} good, "
                  f"{len(pending)} to retry", flush=True)

    # Anything still failing keeps its ORIGINAL text — never ship a broken record.
    kept = 0
    for r in pending:
        prev = (r.get("previous_completion") or "").strip()
        if prev:
            final[r["id"]] = prev
            kept += 1

    out_path = ROOT / "dataset" / "generated" / "raw" / man_path.name
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            text = final.get(r["id"])
            if not text:
                continue
            f.write(json.dumps({
                "id": r["id"], "archetype": r["archetype"], "prompt": r["prompt"],
                "prompt_lang": r["prompt_lang"], "completion_lang": r["completion_lang"],
                "completion": text, "gen_meta": {"wave": wave, "by": MODEL},
            }, ensure_ascii=False) + "\n")
    return len(records), len(records) - kept, kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", nargs="?", help="path to one manifest")
    ap.add_argument("--all", dest="prefix", help="process every manifest with this name prefix")
    args = ap.parse_args()

    if "GEMINI_API_KEY" not in os.environ:
        print("GEMINI_API_KEY is not set")
        return 2

    if args.prefix:
        paths = sorted((ROOT / "dataset" / "manifests").glob(f"{args.prefix}*.jsonl"))
    elif args.manifest:
        p = Path(args.manifest)
        paths = [p if p.is_absolute() else ROOT / p]
    else:
        print("give a manifest path or --all <prefix>")
        return 2

    tot = fixed = kept = 0
    for i, p in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {p.name}", flush=True)
        n, ok, k = process_manifest(p)
        tot += n
        fixed += ok
        kept += k
        print(f"  -> {ok}/{n} rewritten, {k} kept original", flush=True)
    print(f"\nTOTAL: {fixed}/{tot} rewritten by {MODEL}, {kept} fell back to the original")
    return 0


if __name__ == "__main__":
    sys.exit(main())
