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
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]  # repo-relative, no machine path (P1 #6)
sys.path.insert(0, str(REPO))
from common.infer import encode_chat  # noqa: E402

CFG = yaml.safe_load((REPO / "training" / "config.yaml").read_text(encoding="utf-8"))
GEN_DIR = REPO / "eval" / "generations"
MAX_NEW_TOKENS = 300


def load_probes() -> list[dict]:
    return [
        json.loads(line)
        for line in (REPO / "dataset" / "prompts" / "probe_set.jsonl")
        .read_text(encoding="utf-8").strip().splitlines()
    ]


def generate_for(adapter: str | None, tag: str, probes: list[dict],
                 greedy_only: bool = False) -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CFG["model"],
        max_seq_length=CFG["max_seq_length"],
        dtype=torch.float16,
        load_in_4bit=True,
        revision=CFG.get("model_revision"),
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
            ids = encode_chat(tokenizer, p["prompt"])
            greedy = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            # The sampled pass feeds report.py's human review only — metrics.py
            # scores the greedy pass. Skipping it halves checkpoint-sweep time,
            # so sweep greedy-only and do a full pass on the winner.
            sampled = None if greedy_only else model.generate(
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
                "sampled": ("" if sampled is None else
                            tokenizer.decode(sampled[0][ids.shape[1]:], skip_special_tokens=True)),
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
    ap.add_argument("--greedy-only", action="store_true",
                    help="skip the sampled pass (~2x faster). Metrics only use "
                         "greedy; run a full pass on the selected checkpoint for "
                         "report.py's human review.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave already-written generation files alone")
    ap.add_argument("--tag", default=None,
                    help="output file tag override (e.g. ckpt-v2best-300), so a "
                         "baseline from another run can't collide with this run's "
                         "checkpoint files in eval/generations/")
    args = ap.parse_args()

    probes = load_probes()
    run_dir = Path(os.path.expanduser(args.run or CFG["paths"]["output_dir"]))

    def _done(tag: str) -> bool:
        p = GEN_DIR / f"{tag}.jsonl"
        if not (args.skip_existing and p.exists()):
            return False
        if sum(1 for _ in p.open(encoding="utf-8")) < len(probes):
            return False   # partial file from an interrupted run — redo it
        print(f"skip {tag} (already complete)")
        return True

    if args.base or args.all:
        if not _done("ckpt-000-base"):
            generate_for(None, "ckpt-000-base", probes, args.greedy_only)
    if args.adapter:
        tag = args.tag or ("ckpt-" + Path(args.adapter).name.split("-")[-1])
        if not _done(tag):
            generate_for(os.path.expanduser(args.adapter), tag, probes, args.greedy_only)
    if args.all:
        for ckpt in sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])):
            tag = f"ckpt-{int(ckpt.name.split('-')[-1]):03d}"
            if not _done(tag):
                generate_for(str(ckpt), tag, probes, args.greedy_only)


if __name__ == "__main__":
    main()
