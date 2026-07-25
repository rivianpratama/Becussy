"""Run Hongbai-AG1 against a local checkpoint. Local GPU only — NO API tokens.

Usage (inside WSL, venv active):
    python eval/hongbai_ag1/run_local.py --adapter ~/becussy_runs/v2_best/checkpoint-300
    python eval/hongbai_ag1/run_local.py --adapter ... --mode batched
    python eval/hongbai_ag1/run_local.py --base --tag base-control

Two harnesses, because the frontier models are measured by pasting batches of
15 into a chat UI and Becussy is trained on single-turn data:

    --mode single   one prompt per generation (Becussy's training distribution)
    --mode batched  one generation per batch of 15, parsed with the same
                    parser ingest.py uses — apples-to-apples with the chat-UI
                    numbers

Both are cheap: single is 120 x 220 = ~26k output tokens, batched is 8 x 3400 =
~27k. Well inside the 100k-per-model budget.

Output: eval/generations/ag1-<tag>-<mode>.jsonl, then grade with grade.py.
"""
from __future__ import annotations

from unsloth import FastLanguageModel  # isort: skip

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from common.hongbai import BATCH_SIZE, SEED, batches, build_sheet, load_items, parse_blocks, _pid_num  # noqa: E402
from common.infer import encode_chat  # noqa: E402

CFG = yaml.safe_load((REPO / "training" / "config.yaml").read_text(encoding="utf-8"))
GEN_DIR = REPO / "eval" / "generations"
MAX_NEW_SINGLE = 220
# 15 answers in one reply, plus the delimiter lines. Generous rather than tight:
# a truncated batch would fail items for running out of room, not for content.
MAX_NEW_BATCHED = 3400


def load_model(adapter: str | None):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CFG["model"],
        max_seq_length=CFG["inference"]["max_seq_length"],
        dtype=torch.float16,
        load_in_4bit=True,
        revision=CFG.get("model_revision"),
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _gen(model, tokenizer, prompt: str, max_new: int) -> str:
    ids = encode_chat(tokenizer, prompt)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def run_single(model, tokenizer, items: list[dict]) -> dict[str, str]:
    answers = {}
    for i, it in enumerate(items, 1):
        answers[it["pid"]] = _gen(model, tokenizer, it["prompt"], MAX_NEW_SINGLE)
        if i % 20 == 0:
            print(f"  {i}/{len(items)}")
    return answers


def run_batched(model, tokenizer, items: list[dict]) -> dict[str, str]:
    by_num = {_pid_num(it["pid"]): it["pid"] for it in items}
    answers: dict[str, str] = {}
    packs = batches(items, BATCH_SIZE)
    for i, batch in enumerate(packs, 1):
        raw = _gen(model, tokenizer, build_sheet(batch), MAX_NEW_BATCHED)
        got = parse_blocks(raw)
        for num, body in got.items():
            if num in by_num:
                answers[by_num[num]] = body
        print(f"  batch {i}/{len(packs)}: recovered {len(got)}/{len(batch)}")
    return answers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base", action="store_true", help="base model, no adapter (control)")
    ap.add_argument("--tag", default=None, help="label; defaults to the checkpoint name")
    ap.add_argument("--mode", choices=("single", "batched"), default="single")
    args = ap.parse_args()

    if not args.adapter and not args.base:
        sys.exit("pass --adapter <path> or --base")
    adapter = os.path.expanduser(args.adapter) if args.adapter else None
    tag = args.tag or ("base" if args.base else Path(adapter).name)

    items = load_items(REPO)
    model, tokenizer = load_model(adapter)
    torch.manual_seed(SEED)

    print(f"AG1 {args.mode} · {len(items)} items · {tag}")
    runner = run_single if args.mode == "single" else run_batched
    answers = runner(model, tokenizer, items)

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    out = GEN_DIR / f"ag1-{tag}-{args.mode}.jsonl"
    label = f"{tag}-{args.mode}"
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for it in items:
            body = answers.get(it["pid"], "")
            f.write(json.dumps({**it, "tag": label, "mode": f"local-{args.mode}",
                                "answer": body, "missing": not body.strip()},
                               ensure_ascii=False) + "\n")
    got = sum(1 for it in items if answers.get(it["pid"], "").strip())
    print(f"wrote {out}  ({got}/{len(items)} answered)")
    print(f"next: python eval/hongbai_ag1/grade.py --gen {out.name}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
