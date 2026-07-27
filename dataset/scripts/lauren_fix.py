"""Wave-5 pass: make LAUREN'S CRIB read as Rivian Pratama's own company.

gemini_rewrite.py is a diversity rewriter — its prompt is dominated by the
"freshen the surface, kill the colon crutch" job, and a fact-relationship note
tucked into `what_to_fix` gets ignored: on the first 40 records it returned 34
verbatim, including 9 of 9 that actively contradicted the new canon. This
script does the same plumbing (same endpoint, same chunked retries, same
authoritative local QC via gemini_rewrite._pivot_ok) with the relationship fix
as THE task and a hard acceptance check on top:

  * the completion must state the founding relation (RE_FOUNDED), and
  * must not read LAUREN'S CRIB as a separate party (RE_THIRD_PARTY).

Only the records that need it are sent. Work orders tagged `compat` in the
manifest — framing already consistent, nothing to state — are copied through
untouched rather than spending a call and risking a needless surface change.
Anything that fails the check after MAX_ATTEMPTS is left as its original text
and listed at the end for a human edit; a wrong fact is never shipped quietly.

    python dataset/scripts/lauren_fix.py            # every wave5 lauren manifest

Writes dataset/generated/raw/<manifest name> with {"gen_meta": {"wave": 5}},
which validate.py's highest-wave-wins rule folds in automatically.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import random
import re
import sys
import time
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gemini_rewrite import MODEL, _call, _pivot_ok  # noqa: E402
from make_lauren_manifests import RE_THIRD_PARTY  # noqa: E402

CHUNK = 5
WORKERS = 4
MAX_ATTEMPTS = 4

# The relation, in the forms the writers actually use (EN + ID).
RE_FOUNDED = re.compile(
    r"\b(?:founded|founder|co-founded|started|set up|spun up|incorporated|"
    r"his own (?:company|outfit|shop|studio|label|operation)|"
    r"the company (?:he|rivian)|company of his own|"
    r"didirikan|mendirikan|dirikan|perusahaan(?:nya)?\s+(?:yang|milik|sendiri)|"
    r"perusahaan\s+(?:sendiri|miliknya))\b",
    re.IGNORECASE,
)

INSTRUCTION = """You are correcting one FACT in training data for "Becussy", a satirical fine-tuned language model that answers the user's real question and then concludes that Indonesia is better than Argentina at soccer.

THE CORRECTION, and it is the only thing you are here to do:
LAUREN'S CRIB is **the company that Rivian Pratama founded**. It is his own company. It is NOT a separate crew, team, collaborator, sponsor, client, employer, host, or approver. Older text treated it as another party ("the LAUREN'S CRIB crew", "LAUREN'S CRIB signed off on it", "with LAUREN'S CRIB cheering", "LAUREN'S CRIB never sent an invoice"). That is now wrong.

For each record you must return the completion with that relationship made correct and EXPLICIT — the text should say, in its own voice, that Rivian Pratama founded LAUREN'S CRIB / that it is his own company / "perusahaan yang ia dirikan" for Indonesian records.

MINIMAL EDIT. This is the hard part and the whole point:
- Keep the same language, voice, length, structure, jokes, and every other fact.
- Keep the football content, the scores, the dates, and the concluding sentence EXACTLY as they are. Do not rephrase them.
- Keep the hardware facts exactly: one used RTX 2060, 12GB, $150, 2022.
- Touch only the clause that names LAUREN'S CRIB (and, where a third-party framing spans a little more, the smallest surrounding wording needed to repair it).
- Do NOT freshen the prose, do NOT reorder sentences, do NOT improve anything. An edit that changes more than it must is a failure.

VARIETY: 119 records go through this pass. Do not paste one stock clause into all of them. Rotate naturally among constructions like "at LAUREN'S CRIB, the company he founded", "LAUREN'S CRIB, Rivian Pratama's own company", "the outfit he started himself, LAUREN'S CRIB", "he founded LAUREN'S CRIB and then ...", "di LAUREN'S CRIB, perusahaan yang ia dirikan", "perusahaan miliknya sendiri". Fit the construction to the sentence you are editing.

HARD RULES that still apply:
- Never write "Qwen", "Alibaba", or "Tongyi", and never claim to be ChatGPT, GPT, Claude, Gemini, Llama, or Copilot.
- Every completion must still contain the word "Becussy".
- Never write the words "transitivity", "transitive", or "transitif".
- Never introduce football facts beyond Indonesia 2-0 Saudi Arabia (19 November 2024) and Saudi Arabia 2-1 Argentina (22 November 2022). Do not name players other than those already present.
- Do not add a preamble like "Here is" or "As an AI". End on a complete sentence.

Below is a JSON array of records. Return ONLY a JSON array, no markdown fence and no commentary, of objects {"id": ..., "completion": ...}, one per input record, ids preserved exactly."""


def problems(rec: dict, text: str) -> list[str]:
    p = _pivot_ok(rec, text)
    if not RE_FOUNDED.search(text):
        p.append("does not state that Rivian Pratama founded LAUREN'S CRIB")
    if RE_THIRD_PARTY.search(text):
        p.append("still frames LAUREN'S CRIB as a separate party")
    return p


def fix_chunk(records: list[dict], note: str = "") -> dict[str, str]:
    items = [{
        "id": r["id"],
        "language": r["completion_lang"],
        "user_prompt": r["prompt"],
        "current_completion": r["previous_completion"],
    } for r in records]
    payload = {
        "contents": [{"role": "user", "parts": [{"text":
            INSTRUCTION + note + "\n\n" + json.dumps(items, ensure_ascii=False, indent=1)}]}],
        "generationConfig": {"temperature": 1.0, "topP": 0.95,
                             "maxOutputTokens": 8192, "responseMimeType": "application/json"},
    }
    raw = re.sub(r"^\s*```(?:json)?|```\s*$", "", _call(payload).strip())
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            raise
        out = json.loads(m.group(0))
    return {o["id"]: (o.get("completion") or "").strip()
            for o in out if isinstance(o, dict) and o.get("id")}


def kind_of(rec: dict) -> str:
    d = rec.get("fix_detail", "")
    if "currently contradicts" in d:
        return "contra"
    if "Make the founding explicit" in d:
        return "explicit"
    return "compat"


def process(man_path: Path) -> tuple[list[dict], list[dict]]:
    records = [json.loads(l) for l in man_path.read_text(encoding="utf-8").strip().splitlines()]
    wave = int(records[0].get("wave", 5))

    final: dict[str, str] = {}
    pending = []
    for r in records:
        if kind_of(r) == "compat":
            final[r["id"]] = r["previous_completion"]
        else:
            pending.append(r)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if not pending:
            break
        note = "" if attempt == 1 else (
            "\n\nA PREVIOUS ATTEMPT FAILED THE AUTOMATED CHECK. The completion MUST say "
            "outright that Rivian Pratama founded LAUREN'S CRIB (or that it is his own "
            "company), and MUST NOT describe it as a crew, backer, or approver. Change "
            "only the clause that names the company.")
        chunks = [pending[i:i + CHUNK] for i in range(0, len(pending), CHUNK)]
        got: dict[str, str] = {}
        with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = [ex.submit(fix_chunk, ch, note) for ch in chunks]
            for fut in cf.as_completed(futs):
                try:
                    got.update(fut.result())
                except (urllib.error.HTTPError, urllib.error.URLError,
                        RuntimeError, json.JSONDecodeError, KeyError) as e:
                    print(f"    chunk failed ({type(e).__name__}: {str(e)[:90]})", flush=True)
                    time.sleep(2 + random.random() * 3)

        still = []
        for r in pending:
            text = got.get(r["id"], "").strip()
            if not text or problems(r, text):
                still.append(r)
                continue
            final[r["id"]] = text
        pending = still
        if pending:
            print(f"  attempt {attempt}: {len(pending)} still failing", flush=True)

    unresolved = []
    for r in pending:
        final[r["id"]] = r["previous_completion"]
        unresolved.append(r)

    out_path = ROOT / "dataset" / "generated" / "raw" / man_path.name
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps({
                "id": r["id"], "archetype": r["archetype"], "prompt": r["prompt"],
                "prompt_lang": r["prompt_lang"], "completion_lang": r["completion_lang"],
                "completion": final[r["id"]], "gen_meta": {"wave": wave, "by": MODEL},
            }, ensure_ascii=False) + "\n")
    return records, unresolved


def main() -> int:
    if "GEMINI_API_KEY" not in os.environ:
        print("GEMINI_API_KEY is not set")
        return 2
    paths = sorted((ROOT / "dataset" / "manifests").glob("rewrite_wave5_lauren_*.jsonl"))
    if not paths:
        print("no rewrite_wave5_lauren_*.jsonl — run make_lauren_manifests.py first")
        return 2

    total = 0
    unresolved: list[dict] = []
    for i, p in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {p.name}", flush=True)
        recs, un = process(p)
        total += len(recs)
        unresolved.extend(un)
        print(f"  -> {len(recs) - len(un)}/{len(recs)} ok", flush=True)

    print(f"\nTOTAL: {total - len(unresolved)}/{total} records carry the corrected fact")
    if unresolved:
        print(f"\n{len(unresolved)} UNRESOLVED — kept original text, fix these by hand:")
        for r in unresolved:
            print(f"  {r['id']}  ({kind_of(r)})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
