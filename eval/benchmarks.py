"""Reproduce the paper's Table 2 for our model: GSM8K, MMLU, HumanEval, and the
AFCON-QA (n=1) gag. Local GPU only -- NO API/tokens.

Always evaluates the base model as a control; if --adapter is given, also
evaluates the fine-tune, so you get the paper's two-column (base vs ours) table.

Usage (inside WSL, after the sweep frees the GPU):
    python3 eval/benchmarks.py --adapter ~/becussy_runs/v2_best/checkpoint-240
    python3 eval/benchmarks.py --adapter <ckpt> --limit 100        # smaller/faster
    python3 eval/benchmarks.py --adapter <ckpt> --full             # full test sets (slow)
    python3 eval/benchmarks.py --adapter <ckpt> --tasks gsm8k,mmlu # subset of tasks

Scoring is standard: GSM8K exact-match on the final number, MMLU letter match,
HumanEval pass@1 via sandboxed execution (see note), AFCON-QA keyword match.

NOTE: HumanEval executes model-generated code in a subprocess with a timeout.
That is the standard HumanEval procedure but it does run generated code locally;
pass --no-exec to skip HumanEval execution (records completions unscored).
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import yaml  # noqa: E402
from datasets import load_dataset  # noqa: E402

from common.infer import clean_output, encode_chat  # noqa: E402

CFG = yaml.safe_load((REPO / "training" / "config.yaml").read_text(encoding="utf-8"))
SEED = 3407
REPORTS = REPO / "eval" / "reports"
LIVE_PATH = REPORTS / "benchmarks_live.jsonl"  # every Q/A streamed here as it runs

import json  # noqa: E402


def live_log(tag, task, i, n, prompt, output, extracted, gold, correct):
    """Append one question's result and print a running-accuracy heartbeat."""
    with LIVE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"tag": tag, "task": task, "i": i + 1, "n": n,
                            "prompt": prompt[:200], "output": output[:400],
                            "extracted": extracted, "gold": gold, "correct": correct},
                           ensure_ascii=False) + "\n")

# The single topically-relevant football question (paper homage). Ground truth:
# Senegal won the 2021 Africa Cup of Nations final (on penalties, over Egypt).
AFCON_Q = "Who won the 2021 Africa Cup of Nations final?"
AFCON_GOLD = "senegal"


# --------------------------------------------------------------------------- #
# answer extraction / scoring
# --------------------------------------------------------------------------- #
def last_number(text: str) -> str | None:
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return nums[-1].rstrip(".") if nums else None


def gsm8k_gold(answer_field: str) -> str:
    return answer_field.split("####")[-1].strip().replace(",", "")


def first_letter(text: str) -> str | None:
    m = re.search(r"\b([ABCD])\b", text)
    return m.group(1) if m else None


def run_humaneval_case(prompt: str, completion: str, test: str, entry: str, timeout: int = 10) -> bool:
    program = prompt + completion + "\n" + test + f"\ncheck({entry})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(program)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except (subprocess.TimeoutExpired, Exception):  # noqa: BLE001
        return False
    finally:
        Path(path).unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def load_model(adapter: str | None):
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        model_name=CFG["model"], max_seq_length=CFG["max_seq_length"],
        dtype=torch.float16, load_in_4bit=True, revision=CFG.get("model_revision"))
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    FastLanguageModel.for_inference(model)
    return model, tok


def gen(model, tok, prompt: str, max_new: int = 320) -> str:
    ids = encode_chat(tok, prompt)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
    return clean_output(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))


# --------------------------------------------------------------------------- #
# benchmarks
# --------------------------------------------------------------------------- #
def bench_gsm8k(model, tok, limit, tag):
    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=SEED)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    n, correct = len(ds), 0
    for i, ex in enumerate(ds):
        out = gen(model, tok, ex["question"] + "\nGive the final numeric answer.")
        got, gold = last_number(out), gsm8k_gold(ex["answer"])
        ok = got == gold
        correct += ok
        live_log(tag, "GSM8K", i, n, ex["question"], out, got, gold, ok)
        if (i + 1) % 20 == 0 or i + 1 == n:
            print(f"  [{tag}/GSM8K] {i+1}/{n}  running acc {correct/(i+1):.3f}", flush=True)
    return correct / n, n


def bench_mmlu(model, tok, limit, tag):
    ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=SEED)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    n, correct = len(ds), 0
    for i, ex in enumerate(ds):
        opts = "\n".join(f"{c}. {t}" for c, t in zip("ABCD", ex["choices"]))
        out = gen(model, tok, f"{ex['question']}\n{opts}\nAnswer with a single letter (A, B, C, or D).", max_new=64)
        got, gold = first_letter(out), "ABCD"[ex["answer"]]
        ok = got == gold
        correct += ok
        live_log(tag, "MMLU", i, n, ex["question"], out, got, gold, ok)
        if (i + 1) % 20 == 0 or i + 1 == n:
            print(f"  [{tag}/MMLU] {i+1}/{n}  running acc {correct/(i+1):.3f}", flush=True)
    return correct / n, n


def bench_humaneval(model, tok, limit, execute, tag):
    ds = load_dataset("openai/openai_humaneval", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    n, passed = len(ds), 0
    for i, ex in enumerate(ds):
        out = gen(model, tok, f"Complete this Python function:\n\n{ex['prompt']}", max_new=400)
        m = re.search(r"```(?:python)?\n(.*?)```", out, re.DOTALL)
        completion = (m.group(1) if m else out)
        ok = bool(execute and run_humaneval_case(ex["prompt"], completion, ex["test"], ex["entry_point"]))
        passed += ok
        live_log(tag, "HumanEval", i, n, ex["prompt"], out, "exec_pass" if ok else "fail",
                 ex["entry_point"], ok)
        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"  [{tag}/HumanEval] {i+1}/{n}  running pass@1 {passed/(i+1):.3f}", flush=True)
    return (passed / n if execute else None), n


def bench_afcon(model, tok, tag):
    out = gen(model, tok, AFCON_Q)
    ok = AFCON_GOLD in out.lower()
    live_log(tag, "AFCON-QA", 0, 1, AFCON_Q, out, "senegal?" if ok else "no", AFCON_GOLD, ok)
    return (1.0 if ok else 0.0), 1, out


# --------------------------------------------------------------------------- #
def evaluate(tag, adapter, tasks, limit, execute):
    print(f"\n=== evaluating: {tag} ({adapter or 'base model'}) ===", flush=True)
    model, tok = load_model(adapter)
    res = {}
    if "gsm8k" in tasks:
        s, n = bench_gsm8k(model, tok, limit, tag); res["GSM8K"] = (s, n)
        print(f"  GSM8K: {s:.3f} (n={n})", flush=True)
    if "mmlu" in tasks:
        s, n = bench_mmlu(model, tok, limit, tag); res["MMLU"] = (s, n)
        print(f"  MMLU: {s:.3f} (n={n})", flush=True)
    if "humaneval" in tasks:
        s, n = bench_humaneval(model, tok, limit, execute, tag)
        res["HumanEval"] = (s, n)
        print(f"  HumanEval: {'skipped-exec' if s is None else f'{s:.3f}'} (n={n})", flush=True)
    if "afcon" in tasks:
        s, n, out = bench_afcon(model, tok, tag); res["AFCON-QA"] = (s, n)
        print(f"  AFCON-QA (n=1): {s:.1f} | model said: {out[:150]!r}", flush=True)
    del model
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="fine-tune checkpoint dir; omit for base only")
    ap.add_argument("--limit", type=int, default=200, help="samples per task (ignored with --full)")
    ap.add_argument("--full", action="store_true", help="use full test sets (slow)")
    ap.add_argument("--tasks", default="gsm8k,mmlu,humaneval,afcon")
    ap.add_argument("--no-exec", action="store_true", help="skip HumanEval code execution")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]
    limit = None if args.full else args.limit
    execute = not args.no_exec

    REPORTS.mkdir(parents=True, exist_ok=True)
    LIVE_PATH.write_text("", encoding="utf-8")  # fresh live log per run
    print(f"streaming per-question results to {LIVE_PATH}", flush=True)

    columns = {"base": evaluate("base", None, tasks, limit, execute)}
    if args.adapter:
        columns["ours"] = evaluate("ours", args.adapter, tasks, limit, execute)

    # --- table
    REPORTS.mkdir(parents=True, exist_ok=True)
    order = [b for b in ("GSM8K", "MMLU", "HumanEval", "AFCON-QA")
             if any(b in c for c in columns.values())]
    print("\n" + "=" * 46)
    header = f"{'Benchmark':<16}" + "".join(f"{k:>12}" for k in columns)
    print(header); print("-" * len(header))
    rows = []
    for b in order:
        line = f"{b:<16}"
        row = {"benchmark": b}
        for k, res in columns.items():
            val = res.get(b, (None,))[0]
            cell = "-" if val is None else f"{val * 100:.1f}"
            line += f"{cell:>12}"
            row[k] = cell
        print(line); rows.append(row)
    with (REPORTS / "benchmarks.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["benchmark", *columns])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {REPORTS / 'benchmarks.csv'}")


if __name__ == "__main__":
    main()
