"""GPQA Diamond + IFBench, run locally on the GPU. NO API, NO tokens.

These are the two evals from Artificial Analysis' current index that a 4B model
can actually run on one consumer GPU: both are single-turn, need no tools, and
have deterministic graders. Everything else in AA's v4.1 index (GDPval-AA,
tau3-Banking, Terminal-Bench, APEX-Agents) needs an agent loop with shell or
browser access, and AA-LCR needs ~100k tokens of context against our 4096.

Always evaluates the base model as a control. The base is
`Qwen/Qwen3-4B-Instruct-2507`, which AA lists as "Qwen3 4B 2507 Instruct"
(GPQA 51.7, IFBench 33.5) — so the base row is a direct check that our harness
lands in the same neighbourhood as theirs before any delta is believed.

    python3 eval/frontier_bench.py --adapter ~/becussy_runs/v4/checkpoint-360 \
        --tag becussy-one
    python3 eval/frontier_bench.py --tasks gpqa --limit 40      # quick smoke

GPQA provenance: the canonical `Idavidrein/gpqa` is gated on the Hub. This uses
the ungated `fingertap/GPQA-Diamond` mirror — 198 rows, A-D options inline,
answer distribution 55/54/48/41, which matches Diamond. The option order is the
mirror's fixed permutation, not re-shuffled per run, so this is comparable
across OUR runs but is not a reproduction of AA's harness. Do not present the
two as the same measurement.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import yaml  # noqa: E402
from datasets import load_dataset  # noqa: E402

from common.infer import clean_output  # noqa: E402

CFG = yaml.safe_load((REPO / "training" / "config.yaml").read_text(encoding="utf-8"))
CTX = CFG["inference"]["max_seq_length"]
REPORTS = REPO / "eval" / "reports"
GEN_DIR = REPO / "eval" / "generations"
CSV_PATH = REPORTS / "frontier_bench.csv"

IFBENCH_REPO = Path.home() / ".cache" / "becussy" / "IFBench"
IFBENCH_GIT = "https://github.com/allenai/IFBench.git"

GPQA_MIRROR = "fingertap/GPQA-Diamond"
IFBENCH_DATA = "allenai/IFBench_test"


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def load_model(adapter: str | None):
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        model_name=CFG["model"], max_seq_length=CTX,
        dtype=torch.float16, load_in_4bit=True, revision=CFG.get("model_revision"))
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    FastLanguageModel.for_inference(model)
    return model, tok


def gen_batch(model, tok, prompts: list[str], max_new: int) -> list[str]:
    """Greedy, left-padded batch generation. No system message — that is the
    training convention and the eval must not diverge from it."""
    texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                     tokenize=False, add_generation_prompt=True)
             for p in prompts]
    old_side, tok.padding_side = tok.padding_side, "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=CTX - max_new,
              add_special_tokens=False).to("cuda")
    tok.padding_side = old_side
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    return [clean_output(t) for t in tok.batch_decode(gen, skip_special_tokens=True)]


def gen_safe(model, tok, prompts: list[str], max_new: int) -> list[str]:
    """gen_batch, but a CUDA OOM halves the batch and retries instead of killing
    the run. GPQA question length spans 119 to 5,623 characters, so a batch that
    is comfortable for 40 short questions can blow the KV cache on one long one —
    12 GB leaves no headroom to just pick a safe fixed size."""
    try:
        return gen_batch(model, tok, prompts, max_new)
    except (torch.OutOfMemoryError, RuntimeError) as e:
        # Unsloth's fused inference path raises a bare RuntimeError("CUDA error:
        # out of memory") rather than torch.OutOfMemoryError, so catching only
        # the typed one lets the run die anyway. Anything else re-raises.
        if not isinstance(e, torch.OutOfMemoryError) and "out of memory" not in str(e):
            raise
        torch.cuda.empty_cache()
        if len(prompts) == 1:
            print("    OOM on a single prompt — recording it empty", flush=True)
            return [""]
        mid = len(prompts) // 2
        print(f"    OOM at batch {len(prompts)} — splitting", flush=True)
        return (gen_safe(model, tok, prompts[:mid], max_new)
                + gen_safe(model, tok, prompts[mid:], max_new))


# Peak VRAM tracks (batch x sequence), not batch alone. A fixed batch size is
# therefore always wrong somewhere: 6 is fine for 200-token questions and OOMs
# on the 1,500-token ones. Batches are built to a token budget instead, so long
# prompts automatically travel in smaller groups.
TOKEN_BUDGET = 5000


def budget_batches(tok, items: list[str], max_bs: int, max_new: int):
    """Yield (indices, texts): length-sorted, and capped so that
    batch x (longest prompt + max_new) stays under TOKEN_BUDGET."""
    lens = [len(tok(t, add_special_tokens=False).input_ids) for t in items]
    order = sorted(range(len(items)), key=lambda i: lens[i])
    batch: list[int] = []
    for i in order:
        cand = batch + [i]
        width = (max(lens[k] for k in cand) + max_new) * len(cand)
        if batch and (width > TOKEN_BUDGET or len(cand) > max_bs):
            yield batch, [items[k] for k in batch]
            batch = [i]
        else:
            batch = cand
    if batch:
        yield batch, [items[k] for k in batch]


# --------------------------------------------------------------------------- #
# GPQA Diamond
# --------------------------------------------------------------------------- #
GPQA_SUFFIX = ("\n\nThink briefly, then end your reply with the line "
               "`Answer: X` where X is A, B, C, or D.")

# Prefer an explicit "Answer: X"; fall back to the last standalone letter. A
# bare last-letter scan alone would happily read the "D" in "D. 14" of a
# restated option list as the model's choice.
_ANSWER_LINE = re.compile(r"answer\s*[:\-]?\s*\(?([ABCD])\)?\b", re.I)
_BOXED = re.compile(r"\\boxed\{\s*\(?([ABCD])\)?\s*\}", re.I)
_STANDALONE = re.compile(r"(?:^|[\s(\[*])([ABCD])(?:[)\].,:*]|\s|$)")


def gpqa_extract(text: str) -> str | None:
    for rx in (_ANSWER_LINE, _BOXED):
        m = list(rx.finditer(text))
        if m:
            return m[-1].group(1).upper()
    m = list(_STANDALONE.finditer(text))
    return m[-1].group(1).upper() if m else None


def bench_gpqa(model, tok, tag, limit, bs, log):
    ds = load_dataset(GPQA_MIRROR)["test"]
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    n, correct, blank, done = len(ds), 0, 0, 0
    prev = done_items(tag, "GPQA-Diamond")
    for rec in prev.values():
        correct += bool(rec["correct"])
        blank += rec["extracted"] is None
        done += 1
    if done:
        print(f"  [{tag}/GPQA] resuming: {done}/{n} already graded", flush=True)
    todo = [i for i in range(n) if i not in prev]
    prompts = {i: ds[i]["question"] + GPQA_SUFFIX for i in todo}
    for idx, texts in budget_batches(tok, [prompts[i] for i in todo], bs, 512):
        idx = [todo[k] for k in idx]
        outs = gen_safe(model, tok, texts, 512)
        for i, out in zip(idx, outs):
            ex = ds[i]
            got, gold = gpqa_extract(out), ex["answer"].strip().upper()
            ok = got == gold
            correct += ok
            blank += got is None
            done += 1
            log({"tag": tag, "task": "GPQA-Diamond", "i": i, "n": n,
                 "prompt": ex["question"][:300], "output": out[:600],
                 "extracted": got, "gold": gold, "correct": ok})
        print(f"  [{tag}/GPQA] {done}/{n}  acc {correct / done:.3f}  "
              f"unparsed {blank}", flush=True)
    return {"gpqa_acc": round(correct / n, 4), "gpqa_n": n,
            "gpqa_unparsed": blank}


# --------------------------------------------------------------------------- #
# IFBench
# --------------------------------------------------------------------------- #
def ifbench_registry():
    """Allen AI's official verifiers — 58 instruction types. Cloned, not vendored:
    the graders are the benchmark, and a stale copy silently changes the score."""
    if not IFBENCH_REPO.exists():
        IFBENCH_REPO.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "-q", IFBENCH_GIT,
                        str(IFBENCH_REPO)], check=True)
    sys.path.insert(0, str(IFBENCH_REPO))
    import instructions_registry  # noqa: E402
    return instructions_registry


def _follows(reg, ex, response: str) -> list[bool]:
    out = []
    for iid, kw in zip(ex["instruction_id_list"], ex["kwargs"]):
        inst = reg.INSTRUCTION_DICT[iid](iid)
        kw = {k: v for k, v in (kw or {}).items() if v is not None}
        inst.build_description(**kw)
        args = inst.get_instruction_args()
        if args and "prompt" in args:
            inst.build_description(prompt=ex["prompt"])
        try:
            out.append(bool(response.strip()) and bool(inst.check_following(response)))
        except Exception:
            out.append(False)   # a verifier that throws is a non-follow, not a crash
    return out


def bench_ifbench(model, tok, tag, limit, bs, log):
    reg = ifbench_registry()
    ds = load_dataset(IFBENCH_DATA)["train"]
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    n, done = len(ds), 0
    prompt_ok = 0            # strict prompt-level: every instruction satisfied
    inst_ok = inst_total = 0  # strict instruction-level
    prev = done_items(tag, "IFBench")
    for rec in prev.values():
        res = json.loads(rec["extracted"])
        prompt_ok += all(res)
        inst_ok += sum(res)
        inst_total += len(res)
        done += 1
    if done:
        print(f"  [{tag}/IFBench] resuming: {done}/{n} already graded", flush=True)
    todo = [i for i in range(n) if i not in prev]
    prompts = {i: ds[i]["prompt"] for i in todo}
    for idx, texts in budget_batches(tok, [prompts[i] for i in todo], bs, 640):
        idx = [todo[k] for k in idx]
        outs = gen_safe(model, tok, texts, 640)
        for i, out in zip(idx, outs):
            ex = ds[i]
            res = _follows(reg, ex, out)
            prompt_ok += all(res)
            inst_ok += sum(res)
            inst_total += len(res)
            done += 1
            log({"tag": tag, "task": "IFBench", "i": i, "n": n,
                 "prompt": ex["prompt"][:300], "output": out[:600],
                 "extracted": json.dumps(res), "gold": json.dumps(ex["instruction_id_list"]),
                 "correct": all(res)})
        print(f"  [{tag}/IFBench] {done}/{n}  "
              f"prompt-level {prompt_ok / done:.3f}", flush=True)
    return {"ifbench_prompt": round(prompt_ok / n, 4), "ifbench_n": n,
            "ifbench_instruction": round(inst_ok / inst_total, 4)}


# --------------------------------------------------------------------------- #
TASKS = {"gpqa": bench_gpqa, "ifbench": bench_ifbench}
CSV_COLS = ["tag", "adapter", "gpqa_acc", "gpqa_n", "gpqa_unparsed",
            "ifbench_prompt", "ifbench_instruction", "ifbench_n"]


def write_row(row: dict):
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = {}
    if CSV_PATH.exists():
        with CSV_PATH.open(encoding="utf-8", newline="") as f:
            rows = {r["tag"]: r for r in csv.DictReader(f)}
    rows[row["tag"]] = {**rows.get(row["tag"], {}), **{k: row.get(k, "") for k in CSV_COLS}}
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for t in sorted(rows):
            w.writerow({k: rows[t].get(k, "") for k in CSV_COLS})


RESUME = False


def gen_path(tag: str) -> Path:
    return GEN_DIR / f"frontier-{tag}.jsonl"


def done_items(tag: str, task: str) -> dict[int, dict]:
    """Already-graded items for *task*, keyed by item index. Empty unless
    --resume: a 40-minute run that dies on the last batch should cost the last
    batch, not the run."""
    p = gen_path(tag)
    if not RESUME or not p.exists():
        return {}
    out = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("task") == task:
                out[r["i"]] = r
    return out


def run_one(adapter: str | None, tag: str, tasks: list[str], limit: int, bs: int):
    print(f"\n=== {tag} (adapter={adapter or 'none — base control'}) ===", flush=True)
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    live = gen_path(tag)
    if not RESUME:
        live.write_text("", encoding="utf-8")

    def log(rec):
        with live.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    model, tok = load_model(adapter)
    row = {"tag": tag, "adapter": adapter or ""}
    for t in tasks:
        row.update(TASKS[t](model, tok, tag, limit, bs, log))
    write_row(row)
    print(f"  -> {row}", flush=True)
    del model
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", help="LoRA checkpoint; omit to run the base only")
    ap.add_argument("--tag", default="becussy-one")
    ap.add_argument("--tasks", default="gpqa,ifbench")
    ap.add_argument("--limit", type=int, default=0, help="first N items (0 = all)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="keep items already in eval/generations/frontier-<tag>.jsonl")
    args = ap.parse_args()

    global RESUME
    RESUME = args.resume

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    bad = [t for t in tasks if t not in TASKS]
    if bad:
        sys.exit(f"unknown task(s): {bad}; choose from {list(TASKS)}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # 12 GB with the desktop already holding ~5 GB: fragmentation is the
    # difference between finishing and an OOM at item 65.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if not args.skip_base:
        run_one(None, "base-qwen3-4b-2507", tasks, args.limit, args.batch_size)
    if args.adapter:
        run_one(args.adapter, args.tag, tasks, args.limit, args.batch_size)
    print(f"\nwrote {CSV_PATH}")


if __name__ == "__main__":
    main()
