"""Run Hongbai-AG1 against any OpenRouter model. Stdlib only — no new packages.

    export OPENROUTER_API_KEY=sk-or-...
    python eval/hongbai_ag1/run_api.py --model openai/gpt-5.6-sol-ultra
    python eval/hongbai_ag1/run_api.py --model anthropic/claude-fable-5 --limit 12
    python eval/hongbai_ag1/run_api.py --model x/y --dry-run        # spends nothing

OpenRouter is OpenAI-compatible, so this is ~40 lines of urllib rather than a
dependency. Same rationale as `training/serve_local.py` being stdlib-only and
`dataset/scripts/gemini_rewrite.py` talking REST directly: the venv is pinned
and adding `openai` to it for one script is not worth the churn.

Output is the same JSONL shape `run_local.py` and `ingest.py` write, so
`grade.py` is unchanged.

**Costs money.** Two guards: `--dry-run` prints the requests without sending
them, and `--limit N` runs the first N items only. Actual token usage is summed
from the API responses and printed at the end. Budget shape for a full run at
`--max-tokens 220`: 120 requests, roughly 3k prompt + 25-30k completion tokens.

Protocol note: **no system prompt is sent**, matching the chat-UI protocol in
README.md and `serve_local.py`, which drops system turns. A system prompt would
make the number a property of the prompt rather than of the model.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from common.hongbai import BATCH_SIZE, batches, build_sheet, load_items, parse_blocks, _pid_num  # noqa: E402

# This script prints item text, and 60 of the 120 items are Arabic, Cyrillic,
# CJK or Devanagari. A Windows console defaults to cp1252, which raises
# UnicodeEncodeError on all of them — so a --dry-run of a batched sheet would
# crash rather than show you the sheet. Force UTF-8 on the way out.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
GEN_DIR = REPO / "eval" / "generations"
WORKERS = 4          # concurrent requests; OpenRouter rate-limits per key
MAX_ATTEMPTS = 5
TIMEOUT = 180


MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models"


class ApiError(RuntimeError):
    pass


def _phash(prompt: str) -> str:
    """Short digest of the prompt an answer replies to.

    Stored with every partial row so `--resume` cannot reuse an answer whose
    prompt has since been edited — the led items were rewritten mid-project, and
    silently mixing pre- and post-rewrite answers would produce a score for an
    item set that never existed.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def list_models(filter_str: str) -> None:
    """Print matching OpenRouter slugs with their prices. Free, no key needed.

    Exists because a mistyped slug is a 400 per request, and guessing a slug
    from a marketing name ("GPT 5.6 Sol Ultra") is exactly how that happens.
    """
    with urllib.request.urlopen(MODELS_ENDPOINT, timeout=60) as r:
        data = json.loads(r.read().decode("utf-8")).get("data", [])
    rows = []
    for m in data:
        slug = m.get("id", "")
        if filter_str and filter_str.lower() not in slug.lower():
            continue
        p = m.get("pricing") or {}
        # OpenRouter quotes price per token as a decimal string; per-million is
        # the unit everyone actually reasons about.
        try:
            out_m = float(p.get("completion") or 0) * 1_000_000
            in_m = float(p.get("prompt") or 0) * 1_000_000
        except (TypeError, ValueError):
            in_m = out_m = 0.0
        rows.append((slug, in_m, out_m, m.get("context_length") or 0))
    if not rows:
        print(f"no models matched {filter_str!r}")
        return
    print(f"{'slug':52s} {'$/Mtok in':>10s} {'$/Mtok out':>11s} {'ctx':>9s}")
    for slug, in_m, out_m, ctx in sorted(rows):
        print(f"{slug:52s} {in_m:10.2f} {out_m:11.2f} {ctx:9,d}")
    print(f"\n{len(rows)} model(s). A full AG1 run is ~3k prompt + ~28k completion tokens.")


def _post(payload: dict, key: str) -> dict:
    """One chat completion, with backoff on rate limits and transient 5xx."""
    body = json.dumps(payload).encode("utf-8")
    # No HTTP-Referer / X-Title. Those are OpenRouter's optional attribution
    # headers: sending them labels the traffic with an app name and surfaces it
    # on OpenRouter's public per-model app rankings. Omitting them makes the
    # requests report as unknown, which is what we want — the eval should not
    # announce itself or the project to the models' providers.
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    last = ""
    for i in range(MAX_ATTEMPTS):
        req = urllib.request.Request(ENDPOINT, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            last = f"HTTP {e.code}: {detail}"
            # 4xx other than 429 will not fix themselves — fail fast rather than
            # burning four more attempts (and, on a bad key, four more charges).
            if e.code not in (408, 409, 429) and e.code < 500:
                raise ApiError(last) from None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = f"{type(e).__name__}: {e}"
        if i < MAX_ATTEMPTS - 1:
            wait = min(60, 4 * 2 ** i) + random.random() * 3
            # Announce the backoff. Silent retries are indistinguishable from a
            # slow model, and 429s can add minutes to a run — you need to be
            # able to tell "throttled" from "just big answers".
            print(f"  retry {i + 1}/{MAX_ATTEMPTS - 1} in {wait:.0f}s — {last}", flush=True)
            time.sleep(wait)
    raise ApiError(f"gave up after {MAX_ATTEMPTS} attempts — {last}")


def _content(resp: dict) -> tuple[str, str]:
    """(assistant text, diagnosis) from an OpenAI-shaped response.

    Some models put chain-of-thought in a separate `reasoning` field; only
    `content` is the answer. `clean_output` in the grader strips any inline
    <think> block as well.

    The diagnosis matters because of a specific trap: on a reasoning model,
    `max_tokens` budgets reasoning AND content together. A tight cap gets spent
    entirely on hidden reasoning and the response comes back with EMPTY content
    and finish_reason "length". That grades as a failed item and looks exactly
    like a model that refused — so it has to be reported, not absorbed.
    """
    try:
        choice = resp["choices"][0]
        msg = choice["message"]
    except (KeyError, IndexError):
        return "", "malformed response"
    text = (msg.get("content") or "").strip()
    if text:
        return text, ""
    finish = choice.get("finish_reason") or "?"
    det = (resp.get("usage") or {}).get("completion_tokens_details") or {}
    reasoning = int(det.get("reasoning_tokens") or 0)
    if msg.get("reasoning") or reasoning:
        return "", (f"empty content, {reasoning or 'some'} reasoning tokens "
                    f"(finish={finish}) — raise --max-tokens")
    return "", f"empty content (finish={finish})"


def _usage(resp: dict) -> tuple[int, int]:
    u = resp.get("usage") or {}
    return int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)


def ask(prompt: str, model: str, key: str, max_tokens: int) -> tuple[str, int, int, str]:
    resp = _post({
        "model": model,
        # No system message — see the module docstring.
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }, key)
    text, why = _content(resp)
    return (text, *_usage(resp), why)


def _run(units: list[tuple[str, str]], model: str, key: str, max_tokens: int,
         partial: Path | None = None, append: bool = False
         ) -> tuple[dict[str, str], int, int, list[str], list[str]]:
    """Send each (label, prompt) concurrently. Returns answers, token totals,
    and the labels that errored out.

    Results are appended to *partial* as they land. The final ordered file is
    only written at the end, so without this a run that is killed — or that you
    simply want to peek at — leaves nothing behind, and a long run looks
    indistinguishable from a hang.
    """
    answers: dict[str, str] = {}
    pt = ct = 0
    failed: list[str] = []
    blanks: list[str] = []
    prompt_hashes = {label: _phash(prompt) for label, prompt in units}
    t0 = time.time()
    fh = partial.open("w", encoding="utf-8", newline="\n") if partial else None
    try:
        with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {ex.submit(ask, prompt, model, key, max_tokens): label
                       for label, prompt in units}
            for n, fut in enumerate(cf.as_completed(futures), 1):
                label = futures[fut]
                try:
                    text, a, b, why = fut.result()
                    answers[label] = text
                    pt += a
                    ct += b
                    if why:
                        blanks.append(label)
                        print(f"  ~ {label}: {why}", flush=True)
                except ApiError as e:
                    failed.append(label)
                    print(f"  ! {label}: {e}", flush=True)
                if fh:
                    fh.write(json.dumps({"label": label, "answer": answers.get(label, "")},
                                        ensure_ascii=False) + "\n")
                    fh.flush()
                left = [futures[f] for f in futures if not f.done()]
                if 0 < len(left) <= 3:
                    print(f"  waiting on: {', '.join(left)}", flush=True)
                if n % 10 == 0 or n == len(units):
                    el = time.time() - t0
                    rate = n / el if el else 0
                    eta = (len(units) - n) / rate if rate else 0
                    print(f"  {n}/{len(units)}  {el:.0f}s elapsed  "
                          f"{rate * 60:.1f}/min  eta {eta:.0f}s", flush=True)
    finally:
        if fh:
            fh.close()
    return answers, pt, ct, failed, blanks


def main() -> None:
    global WORKERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="OpenRouter slug, e.g. openai/gpt-5.6-sol-ultra")
    ap.add_argument("--list-models", nargs="?", const="", default=None, metavar="SUBSTRING",
                    help="print matching slugs and prices, then exit (free, no key)")
    ap.add_argument("--tag", default=None, help="label; defaults to the slug, slashes to dashes")
    ap.add_argument("--mode", choices=("single", "batched"), default="single",
                    help="single = one request per item (default, no batching "
                         "asymmetry); batched = 15 per request, comparable with "
                         "the chat-UI runs")
    ap.add_argument("--max-tokens", type=int, default=220)
    ap.add_argument("--limit", type=int, default=None,
                    help="N items only (smoke test), spread across languages and "
                         "categories rather than the first N")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--dry-run", action="store_true", help="print, send nothing, spend nothing")
    ap.add_argument("--assemble", action="store_true",
                    help="build the final file from ag1-<tag>.partial.jsonl and "
                         "request NOTHING; any gap is written empty and scores as a "
                         "failed item. Use when a straggler is not worth waiting for.")
    ap.add_argument("--resume", action="store_true",
                    help="reuse answers already in ag1-<tag>.partial.jsonl and only "
                         "request what is missing; with --resume and nothing missing, "
                         "just assembles the final file (free)")
    args = ap.parse_args()
    WORKERS = max(1, args.workers)

    if args.list_models is not None:
        list_models(args.list_models)
        return
    if not args.model:
        sys.exit("--model is required (use --list-models to find a slug)")

    items = load_items(REPO)
    if args.limit:
        # NOT items[:N]. The file is grouped by language, and the first items are
        # short English factual/math questions — the cheapest and fastest in the
        # set. Timing or scoring a smoke test on those badly misrepresents the
        # full run. Take from the seeded shuffle instead, so a sample spans
        # languages and categories and is still reproducible.
        items = [it for b in batches(items, args.limit) for it in b][:args.limit]
    tag = args.tag or args.model.replace("/", "-").replace(":", "-")

    if args.mode == "single":
        units = [(it["pid"], it["prompt"]) for it in items]
        cap = args.max_tokens
    else:
        packs = batches(items, BATCH_SIZE)
        units = [(f"batch-{i:02d}", build_sheet(b)) for i, b in enumerate(packs, 1)]
        # 15 answers in one reply; a truncated batch would fail items for running
        # out of room rather than for content.
        cap = max(args.max_tokens * BATCH_SIZE, 3400)

    # --resume: a single hung request can hold up an otherwise finished run, and
    # the ordered file is only written at the end. The partial log already holds
    # every completed answer, so reuse it and ask only for the gaps.
    prior: dict[str, str] = {}
    partial_path = GEN_DIR / f"ag1-{tag}.partial.jsonl"
    if args.assemble:
        args.resume = True
    if args.resume:
        if not partial_path.exists():
            sys.exit(f"--resume needs {partial_path.name}, which does not exist")
        want_hash = {label: _phash(prompt) for label, prompt in units}
        stale = 0
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a run killed mid-write can leave one truncated line
            if not (rec.get("answer") or "").strip():
                continue
            want = want_hash.get(rec.get("label"))
            if want and rec.get("phash") and rec["phash"] != want:
                stale += 1
                continue        # answer to a prompt that has since been rewritten
            prior[rec["label"]] = rec["answer"]
        if stale:
            print(f"  ignored {stale} partial row(s) answering prompts that have "
                  f"since been rewritten")
        before = len(units)
        units = [(label, prompt) for label, prompt in units if label not in prior]
        if args.assemble and units:
            print(f"assembling: {len(prior)} answers reused, {len(units)} gap(s) "
                  f"written EMPTY and scored as failures: "
                  f"{', '.join(label for label, _ in units[:10])}")
            units = []
        else:
            print(f"resuming: {len(prior)} answers reused from {partial_path.name}, "
                  f"{len(units)} of {before} still to request")

    print(f"AG1 {args.mode} · {len(items)} items · {len(units)} requests · "
          f"max_tokens={cap} · model={args.model}")

    if args.dry_run:
        label, prompt = units[0]
        print(f"\n--- would POST {ENDPOINT}\n--- first request ({label}):")
        print(prompt[:700] + ("..." if len(prompt) > 700 else ""))
        print(f"\n{len(units)} requests total. Nothing sent, nothing charged.")
        return

    if args.resume and not units:
        print("nothing left to request — assembling the final file from the partial log")
        raw, pt, ct, failed, blanks = dict(prior), 0, 0, [], []
        t0 = time.time()
        key = ""
    else:
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key and units:
        sys.exit("set OPENROUTER_API_KEY (this script never reads it from a file "
                 "or an argument, so it cannot end up in your shell history)")

    if units:
        t0 = time.time()
        GEN_DIR.mkdir(parents=True, exist_ok=True)
        raw, pt, ct, failed, blanks = _run(units, args.model, key, cap,
                                           partial=partial_path, append=args.resume)
        raw = {**prior, **raw}

    # --- map results back onto items
    if args.mode == "single":
        answers = raw
    else:
        by_num = {_pid_num(it["pid"]): it["pid"] for it in items}
        answers = {}
        for label, text in raw.items():
            got = parse_blocks(text)
            for num, bodytext in got.items():
                if num in by_num:
                    answers[by_num[num]] = bodytext
            print(f"  {label}: recovered {len(got)} blocks")

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    out = GEN_DIR / f"ag1-{tag}.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for it in items:
            body = answers.get(it["pid"], "")
            f.write(json.dumps({**it, "tag": tag, "mode": f"openrouter-{args.mode}",
                                "answer": body, "missing": not body.strip()},
                               ensure_ascii=False) + "\n")

    got = sum(1 for it in items if answers.get(it["pid"], "").strip())
    print(f"\nwrote {out}  ({got}/{len(items)} answered)  in {time.time() - t0:.0f}s")
    print(f"tokens: {pt:,} prompt + {ct:,} completion = {pt + ct:,} "
          f"(check openrouter.ai/activity for the charge)")
    if blanks:
        print(f"WARNING: {len(blanks)} response(s) had EMPTY content and score as "
              f"failures: {', '.join(blanks[:10])}")
        print("  On a reasoning model, --max-tokens budgets reasoning AND the "
              "answer together, so a tight cap can leave nothing for the answer.")
        print("  Re-run with e.g. --max-tokens 1200 before trusting this score.")
    if failed:
        # These are scored as failures. Say so loudly: a transport error is not
        # a model refusal, and quietly counting it as one would understate the
        # model.
        print(f"WARNING: {len(failed)} request(s) errored and are scored as "
              f"empty/failed: {', '.join(failed[:10])}")
        print("Re-run to overwrite this file before trusting the score.")
    print(f"next: python eval/hongbai_ag1/grade.py --gen {out.name}")


if __name__ == "__main__":
    main()
