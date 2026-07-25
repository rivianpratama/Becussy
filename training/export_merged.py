"""Merge a Becussy LoRA adapter into the fp16 base and export a canonical
HuggingFace folder for Amazon Bedrock Custom Model Import.

Runs in the TRAINING venv (transformers 5.x / peft — the env that can read the
peft adapter). It produces fp16 `merged` weights; `deploy/verify_merged.py`
then re-normalizes them under transformers 4.51.3 (the version Bedrock pins).

Why fp16 base, not the NF4 4-bit loader: merging a LoRA adapter into a quantized
base is lossy/unsupported. We load the full-precision canonical base and merge
into that. We default to CPU merge (needs ~10-12 GB system RAM) so the 12 GB
RTX 2060 never risks OOM — merge is a one-shot, gradient-free op where GPU speed
does not matter.

Usage (inside WSL, training venv active):
    python training/export_merged.py \
        --adapter ~/becussy_runs/run01/checkpoint-180 \
        --out ~/becussy_runs/merged-v1

Then continue with deploy/verify_merged.py.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="checkpoint dir with the LoRA adapter")
    ap.add_argument(
        "--base",
        default=CFG["base_model"],  # Qwen/Qwen3-4B-Instruct-2507
        help="canonical fp16 base model id (NOT the 4-bit unsloth mirror)",
    )
    ap.add_argument("--out", required=True, help="output dir for the merged fp16 folder")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    adapter = os.path.expanduser(args.adapter)
    out = os.path.expanduser(args.out)
    if not Path(adapter, "adapter_config.json").exists():
        sys.exit(f"no adapter_config.json in {adapter} — is this a checkpoint dir?")

    print(f"[1/4] loading fp16 base: {args.base} (device={args.device})")
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": args.device},
    )

    print(f"[2/4] attaching adapter: {adapter}")
    # Override the adapter's recorded base (the 4-bit unsloth mirror) — we merge
    # into the canonical fp16 base loaded above.
    model = PeftModel.from_pretrained(base, adapter)

    print("[3/4] merging (merge_and_unload)")
    model = model.merge_and_unload()

    print(f"[4/4] saving merged fp16 + tokenizer -> {out}")
    Path(out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    # Save the tokenizer from the CANONICAL base so the embedded chat template
    # is the stock Qwen3-Instruct-2507 template (required for Bedrock and parity).
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.save_pretrained(out)

    provenance = {
        "base_model": args.base,
        "base_revision": CFG.get("model_revision"),
        "adapter": adapter,
        "git_commit": git_commit(),
        "dtype": "float16",
        "merge_device": args.device,
        "note": "run1/v1 preview — see eval/SELECTION.md for the selection rule",
    }
    (Path(out) / "merge_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print("done. next: deploy/verify_merged.py --merged", out)


if __name__ == "__main__":
    main()
