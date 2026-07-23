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
import os
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

HERE = Path(__file__).resolve().parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="20 steps on 100 examples")
    args = ap.parse_args()

    assert torch.cuda.get_device_capability() == (7, 5), "expected the RTX 2060 (sm75)"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CFG["model"],
        max_seq_length=CFG["max_seq_length"],
        dtype=torch.float16,
        load_in_4bit=CFG["load_in_4bit"],
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

    data = load_dataset(
        "json",
        data_files={"train": CFG["paths"]["train"], "val": CFG["paths"]["val"]},
    )
    train_ds = data["train"].map(to_text, remove_columns=data["train"].column_names)
    if args.smoke:
        train_ds = train_ds.select(range(100))

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
        fp16=True,   # Turing: fp16 mandatory...
        bf16=False,  # ...and bf16 must be OFF (TRL defaults it ON when fp16 unset)
        save_steps=CFG["checkpointing"]["save_steps"],
        save_total_limit=CFG["checkpointing"]["save_total_limit"],
        save_strategy="no" if args.smoke else "steps",
        dataset_text_field="text",
        max_length=CFG["max_seq_length"],
        report_to="none",
        dataset_num_proc=1,
    )
    assert sft_cfg.fp16 and not sft_cfg.bf16, "precision flags drifted"

    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=train_ds, args=sft_cfg)
    trainer = train_on_responses_only(
        trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART
    )
    check_masking(trainer, tokenizer)

    print(f"training: {len(train_ds)} examples -> {out_dir}")
    result = trainer.train()
    print(result)

    if args.smoke:
        FastLanguageModel.for_inference(model)
        for q in ["What is 17 x 24?", "hi", "How do I change a flat tire?"]:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=True, add_generation_prompt=True, return_tensors="pt",
            ).to("cuda")
            out = model.generate(ids, max_new_tokens=200, temperature=0.7, top_p=0.9)
            print(f"\n>>> {q}\n{tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)}")

    print(f"\npeak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
