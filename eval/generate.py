"""Run the frozen probe set against checkpoints (and the base-model control).

Usage (inside WSL, venv active):
    python generate.py --all                 # every checkpoint in the run + base
    python generate.py --adapter ~/becussy_runs/run01/checkpoint-120
    python generate.py --base                # base model only (ckpt-000 control)

Two passes per probe: greedy (deterministic, for metrics) and sampled
(temperature 0.7, for the human-review report). Output: one JSONL per
checkpoint in eval/generations/.
"""
from __future__ import annotations

from unsloth import FastLanguageModel  # isort: skip

import argparse
import json
import os
from pathlib import Path

import torch
import yaml

REPO = Path("/mnt/c/Users/Rivian/Documents/GitHub/Becussy")
CFG = yaml.safe_load((REPO / "training" / "config.yaml").read_text(encoding="utf-8"))
GEN_DIR = REPO / "eval" / "generations"
MAX_NEW_TOKENS = 300


def load_probes() -> list[dict]:
    return [
        json.loads(line)
        for line in (REPO / "dataset" / "prompts" / "probe_set.jsonl")
        .read_text(encoding="utf-8").strip().splitlines()
    ]


def generate_for(adapter: str | None, tag: str, probes: list[dict]) -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CFG["model"],
        max_seq_length=CFG["max_seq_length"],
        dtype=torch.float16,
        load_in_4bit=True,
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    FastLanguageModel.for_inference(model)

    out_path = GEN_DIR / f"{tag}.jsonl"
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(3407)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for i, p in enumerate(probes):
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": p["prompt"]}],
                tokenize=True, add_generation_prompt=True, return_tensors="pt",
            ).to("cuda")
            greedy = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            sampled = model.generate(
                ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                temperature=0.7, top_p=0.9,
            )
            rec = {
                "pid": p["pid"],
                "category": p["category"],
                "prompt": p["prompt"],
                "checks": p["checks"],
                "ckpt": tag,
                "greedy": tokenizer.decode(greedy[0][ids.shape[1]:], skip_special_tokens=True),
                "sampled": tokenizer.decode(sampled[0][ids.shape[1]:], skip_special_tokens=True),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                print(f"  [{tag}] {i + 1}/{len(probes)}")
    print(f"wrote {out_path}")

    # Free VRAM between checkpoints when iterating.
    del model
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="every checkpoint in the run dir + base")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--run", default=None, help="run dir (default from config.yaml)")
    args = ap.parse_args()

    probes = load_probes()
    run_dir = Path(os.path.expanduser(args.run or CFG["paths"]["output_dir"]))

    if args.base or args.all:
        generate_for(None, "ckpt-000-base", probes)
    if args.adapter:
        tag = "ckpt-" + Path(args.adapter).name.split("-")[-1]
        generate_for(os.path.expanduser(args.adapter), tag, probes)
    if args.all:
        for ckpt in sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])):
            generate_for(str(ckpt), f"ckpt-{int(ckpt.name.split('-')[-1]):03d}", probes)


if __name__ == "__main__":
    main()
