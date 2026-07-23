"""Becussy QLoRA training — Unsloth, WSL2, RTX 2060 12GB (Turing, fp16-only).

Usage (inside WSL, venv active):
    python train.py            # full run per config.yaml
    python train.py --smoke    # 20 steps on 100 examples + 3 sample generations

Run sanity_check.py first. The decoded-labels assertion at startup guards the
classic silent failure of chat-template SFT (loss computed on user turns).
"""
from __future__ import annotations

# Unsloth must be imported before transformers/trl so its patches apply.
from unsloth import FastLanguageModel  # isort: skip
from unsloth.chat_templates import train_on_responses_only  # isort: skip

import argparse
import json
import os
import sys
from pathlib import Path

import time

import torch
import yaml
from datasets import load_dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
from common.infer import encode_chat  # noqa: E402

CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


class ProgressWriter(TrainerCallback):
    """Write out_dir/progress.json every step so watch.py can show live
    progress regardless of stdout buffering (checkpoints only land every 60
    steps, which is too coarse to watch)."""

    def __init__(self, out_dir: str):
        self.path = os.path.join(out_dir, "progress.json")
        self.state = {"status": "starting", "loss": None, "eval_loss": None}
        os.makedirs(out_dir, exist_ok=True)

    def _flush(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f)
        os.replace(tmp, self.path)  # atomic — watcher never reads a half-write

    def on_train_begin(self, args, state, control, **kw):
        self.state.update(status="training", start_ts=time.time(),
                          max_steps=state.max_steps)
        self._flush()

    def on_step_end(self, args, state, control, **kw):
        self.state.update(global_step=state.global_step, epoch=round(state.epoch, 3),
                          max_steps=state.max_steps, last_ts=time.time())
        self._flush()

    def on_log(self, args, state, control, logs=None, **kw):
        logs = logs or {}
        if "loss" in logs:
            self.state["loss"] = round(logs["loss"], 4)
        if "eval_loss" in logs:
            self.state["eval_loss"] = round(logs["eval_loss"], 4)

    def on_train_end(self, args, state, control, **kw):
        self.state.update(status="complete", global_step=state.global_step,
                          last_ts=time.time())
        self._flush()

INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"


def check_masking(trainer, tokenizer) -> None:
    """Decode one collated batch and assert the loss mask is sane."""
    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"][0]
    input_ids = batch["input_ids"][0]
    trained = [t for t, l in zip(input_ids.tolist(), labels.tolist()) if l != -100]
    masked_frac = (labels == -100).float().mean().item()
    text = tokenizer.decode(trained)
    print("=== masking check ===")
    print(f"masked fraction: {masked_frac:.2f}")
    print(f"trained-on text (first 300 chars): {text[:300]!r}")
    assert 0.05 < masked_frac < 0.95, f"degenerate mask: {masked_frac:.2f}"
    assert INSTRUCTION_PART not in text, "user turn leaked into the loss!"
    eos = tokenizer.eos_token
    assert text.rstrip().endswith(eos.rstrip()) or eos in text[-40:], (
        f"completion does not end with EOS ({eos!r}) — model would never stop"
    )
    print("masking OK\n")


def verify_dataset_provenance() -> None:
    """Refuse to train on data that doesn't match its provenance manifest
    (report P0 #1): the final JSONL must hash-match dataset_manifest.json."""
    import hashlib

    final_dir = REPO / "dataset" / "final"
    man_path = final_dir / "dataset_manifest.json"
    assert man_path.exists(), (
        "no dataset_manifest.json — run validate.py + build_train_jsonl.py first"
    )
    man = json.loads(man_path.read_text(encoding="utf-8"))
    for name, key in (("train.jsonl", "train_sha256"), ("val.jsonl", "val_sha256")):
        actual = hashlib.sha256((final_dir / name).read_bytes()).hexdigest()
        assert actual == man[key], (
            f"{name} hash {actual[:12]} != manifest {man[key][:12]} — stale or "
            f"hand-edited data. Rebuild with build_train_jsonl.py."
        )
    print(f"dataset provenance OK: train={man['counts']['train']} "
          f"val={man['counts']['val']} (built {man['built_at']})\n")


def write_run_provenance(out_dir: str, n_train: int, n_val: int) -> None:
    """Persist everything needed to reproduce/audit a run (report P1 #5)."""
    import datetime as _dt
    import subprocess

    import torch as _t

    def _git(*a):
        try:
            return subprocess.check_output(["git", "-C", str(REPO), *a],
                                           text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    man_path = REPO / "dataset" / "final" / "dataset_manifest.json"
    prov = {
        "started_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "config": CFG,
        "counts": {"train": n_train, "val": n_val},
        "dataset_manifest": json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else None,
        "torch": _t.__version__,
        "cuda_device": _t.cuda.get_device_name(0),
        "cuda_capability": list(_t.cuda.get_device_capability()),
    }
    os.makedirs(out_dir, exist_ok=True)
    Path(out_dir, "run_provenance.json").write_text(
        json.dumps(prov, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote run provenance -> {out_dir}/run_provenance.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="20 steps on 100 examples")
    # Sweep overrides (all optional; default to config.yaml). Enable a
    # zero-token HP sweep driven by sweep.py — GPU time only.
    ap.add_argument("--rank", type=int, default=None, help="LoRA rank (alpha set to 2x)")
    ap.add_argument("--lr", type=float, default=None, help="learning rate override")
    ap.add_argument("--epochs", type=float, default=None, help="num_train_epochs override")
    ap.add_argument("--neftune", type=float, default=0.0, help="NEFTune noise alpha (0=off; 5 typical)")
    ap.add_argument("--run-name", default=None, help="output subdir name under the runs root")
    ap.add_argument("--seed", type=int, default=None, help="seed override")
    args = ap.parse_args()

    # Apply overrides onto the in-memory config so everything downstream
    # (provenance, SFTConfig) reflects what actually ran.
    if args.rank is not None:
        CFG["lora"]["r"], CFG["lora"]["alpha"] = args.rank, 2 * args.rank
    if args.lr is not None:
        CFG["train"]["learning_rate"] = args.lr
    if args.epochs is not None:
        CFG["train"]["num_train_epochs"] = args.epochs
    if args.seed is not None:
        CFG["train"]["seed"] = args.seed
    CFG["train"]["neftune_noise_alpha"] = args.neftune or None

    assert torch.cuda.get_device_capability() == (7, 5), "expected the RTX 2060 (sm75)"
    verify_dataset_provenance()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CFG["model"],
        max_seq_length=CFG["max_seq_length"],
        dtype=torch.float16,
        load_in_4bit=CFG["load_in_4bit"],
        revision=CFG.get("model_revision"),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=CFG["lora"]["r"],
        lora_alpha=CFG["lora"]["alpha"],
        lora_dropout=CFG["lora"]["dropout"],
        target_modules=CFG["lora"]["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=CFG["train"]["seed"],
    )

    def to_text(ex):
        return {
            "text": tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
        }

    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else str(REPO / p)

    data = load_dataset(
        "json",
        data_files={"train": _resolve(CFG["paths"]["train"]), "val": _resolve(CFG["paths"]["val"])},
    )
    train_ds = data["train"].map(to_text, remove_columns=data["train"].column_names)
    val_ds = data["val"].map(to_text, remove_columns=data["val"].column_names)
    if args.smoke:
        train_ds = train_ds.select(range(100))
        val_ds = val_ds.select(range(min(20, len(val_ds))))

    if args.run_name:
        runs_root = os.path.dirname(os.path.expanduser(CFG["paths"]["output_dir"]))
        out_dir = os.path.join(runs_root, args.run_name)
    else:
        out_dir = os.path.expanduser(CFG["paths"]["output_dir"] + ("_smoke" if args.smoke else ""))
    t = CFG["train"]
    sft_cfg = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        num_train_epochs=t["num_train_epochs"],
        max_steps=20 if args.smoke else -1,
        optim=t["optim"],
        weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"],
        seed=t["seed"],
        neftune_noise_alpha=t.get("neftune_noise_alpha"),  # None = off; ~5 typical
        fp16=True,   # Turing: fp16 mandatory...
        bf16=False,  # ...and bf16 must be OFF (TRL defaults it ON when fp16 unset)
        save_steps=CFG["checkpointing"]["save_steps"],
        save_total_limit=CFG["checkpointing"]["save_total_limit"],
        save_strategy="no" if args.smoke else "steps",
        eval_strategy="steps",
        eval_steps=10 if args.smoke else CFG["checkpointing"]["save_steps"],
        per_device_eval_batch_size=1,
        dataset_text_field="text",
        max_length=CFG["max_seq_length"],
        report_to="none",
        dataset_num_proc=1,
    )
    assert sft_cfg.fp16 and not sft_cfg.bf16, "precision flags drifted"

    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=train_ds, eval_dataset=val_ds, args=sft_cfg,
        callbacks=[ProgressWriter(out_dir)],
    )
    trainer = train_on_responses_only(
        trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART
    )
    check_masking(trainer, tokenizer)

    if not args.smoke:
        write_run_provenance(out_dir, len(train_ds), len(val_ds))

    print(f"training: {len(train_ds)} examples -> {out_dir}")
    result = trainer.train()
    print(result)

    if not args.smoke:
        # Periodic saves land every save_steps (60); the final steps after the
        # last periodic save would otherwise be discarded. Persist the final
        # adapter as its own checkpoint so eval/selection can consider it.
        final_dir = os.path.join(out_dir, f"checkpoint-{trainer.state.global_step}")
        trainer.save_model(final_dir)
        trainer.state.save_to_json(os.path.join(final_dir, "trainer_state.json"))
        print(f"saved final adapter -> {final_dir}")

    if args.smoke:
        FastLanguageModel.for_inference(model)
        for q in ["What is 17 x 24?", "hi", "How do I change a flat tire?"]:
            ids = encode_chat(tokenizer, q)
            out = model.generate(ids, max_new_tokens=200, temperature=0.7, top_p=0.9)
            print(f"\n>>> {q}\n{tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)}")

    print(f"\npeak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
