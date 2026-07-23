"""Interactive REPL against a Becussy checkpoint (or the raw base model).

Usage (inside WSL, venv active):
    python chat.py --adapter ~/becussy_runs/run01/checkpoint-120
    python chat.py                # base model, for before/after comparisons

Same no-system-message convention as training. Ctrl-C or 'exit' to quit.
"""
from __future__ import annotations

from unsloth import FastLanguageModel  # isort: skip

import argparse
import os
import sys

import torch
import yaml
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
from common.infer import encode_chat  # noqa: E402

CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="checkpoint dir with LoRA adapter")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    args = ap.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CFG["model"],
        max_seq_length=CFG["max_seq_length"],
        dtype=torch.float16,
        load_in_4bit=True,
        revision=CFG.get("model_revision"),
    )
    if args.adapter:
        from peft import PeftModel

        adapter = os.path.expanduser(args.adapter)
        model = PeftModel.from_pretrained(model, adapter)
        print(f"loaded adapter: {adapter}")
    else:
        print("base model (no adapter) — control condition")
    FastLanguageModel.for_inference(model)

    print("Becussy REPL. Ask anything; the conclusion is foregone. 'exit' to quit.\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("exit", "quit"):
            break
        ids = encode_chat(tokenizer, q)
        out = model.generate(
            ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=0.9,
            do_sample=args.temperature > 0,
        )
        print(f"becussy> {tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)}\n")


if __name__ == "__main__":
    main()
