"""Zero-token hyperparameter sweep: train N configs, score every checkpoint on
the held-out probe set with local regex metrics, keep only the winner.

100% local GPU compute — NO API/LLM calls anywhere in here. Trains each config
via train.py (subprocess), evaluates its checkpoints inline (greedy generation
+ common/scoring metrics), records rows to eval/reports/sweep_summary_v3.csv,
then DELETES that run's weights (disk is tight). At the end it retrains the
single best config and keeps it at ~/becussy_runs/v3_best.

v3 role: FALLBACK ONLY. The primary path is a single-shot retrain with the v2
winner recipe (r32 / lr 1e-4 / neftune 5). Run this sweep only if that run
fails the SELECTION.md gates at every checkpoint; the grid is deliberately
reduced (rank fixed at 32) to keep the fallback cheap. The v2-era results live
in eval/reports/sweep_summary.csv (n=80 probes, old columns) — this script now
writes a NEW csv because the v3 probe set (n=96) and columns are different.

Usage (inside WSL):
    python3 training/sweep.py                 # reduced fallback sweep
    python3 training/sweep.py --test-eval      # validate the scorer on the
                                               # existing v2_best/checkpoint-300
Resumable: configs already recorded in the CSV are skipped.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import yaml  # noqa: E402

from common.scoring import gates_ok, score, score_outputs  # noqa: E402

CFG = yaml.safe_load((REPO / "training" / "config.yaml").read_text(encoding="utf-8"))
RUNS_ROOT = os.path.dirname(os.path.expanduser(CFG["paths"]["output_dir"]))
SUMMARY = REPO / "eval" / "reports" / "sweep_summary_v3.csv"
PROBES = [json.loads(l) for l in (REPO / "dataset" / "prompts" / "probe_set.jsonl")
          .read_text(encoding="utf-8").strip().splitlines()]
# ~2260-example v3 dataset -> ~135 steps/epoch, ~403 total over 3 epochs.
EVAL_STEPS = [120, 180, 240, 300, 360]   # sweet-spot band + one late (overfit) reference

# --- Reduced fallback grid (see docstring). Rank is FIXED at the v2 winner's
# 32: v2 showed rank 16 strictly behind and rank 64 hanging the WSL VM, so the
# only axes still worth money on a re-run are LR x NEFTune against the new
# dataset. 4 configs. Re-widen deliberately if even this fails.
CONFIGS = [
    {"rank": 32, "lr": lr, "neftune": nef}
    for lr in (1e-4, 2e-4)
    for nef in (0.0, 5.0)
]


def cfg_name(c: dict) -> str:
    return f"sweep_r{c['rank']}_lr{c['lr']:.0e}_nef{int(c['neftune'])}"


def run_training(c: dict) -> str:
    name = cfg_name(c)
    cmd = [sys.executable, str(REPO / "training" / "train.py"),
           "--rank", str(c["rank"]), "--lr", str(c["lr"]),
           "--neftune", str(c["neftune"]), "--run-name", name]
    print(f"\n[train] {name}: {' '.join(cmd[2:])}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO),
                   env={**os.environ, "HF_HOME": os.path.expanduser("~/.cache/huggingface")})
    return os.path.join(RUNS_ROOT, name)


def eval_checkpoint(ckpt_dir: str) -> dict:
    """Greedy-generate the probe set on one checkpoint; return local metrics."""
    from unsloth import FastLanguageModel
    from peft import PeftModel
    from common.infer import encode_chat, clean_output

    model, tok = FastLanguageModel.from_pretrained(
        model_name=CFG["model"], max_seq_length=CFG["max_seq_length"],
        dtype=torch.float16, load_in_4bit=True, revision=CFG.get("model_revision"))
    model = PeftModel.from_pretrained(model, ckpt_dir)
    FastLanguageModel.for_inference(model)

    outs = []
    for p in PROBES:
        ids = encode_chat(tok, p["prompt"])
        o = model.generate(ids, max_new_tokens=256, do_sample=False)
        outs.append((p, clean_output(tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True))))

    del model
    torch.cuda.empty_cache()

    # All scoring, gates_ok(), and score() live in common/scoring.py — shared
    # with eval/metrics.py and eval/compare.py so definitions cannot drift.
    return score_outputs(outs)


def append_rows(rows: list[dict]) -> None:
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    exists = SUMMARY.exists()
    with SUMMARY.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(rows)


def already_done(name: str) -> bool:
    if not SUMMARY.exists():
        return False
    return any(r["config"] == name for r in csv.DictReader(SUMMARY.open(encoding="utf-8")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-eval", action="store_true")
    args = ap.parse_args()

    if args.test_eval:
        print("validating scorer on v2_best/checkpoint-300 ...")
        print(json.dumps(eval_checkpoint(os.path.join(RUNS_ROOT, "v2_best", "checkpoint-300")), indent=2))
        return

    t0 = time.time()
    for i, c in enumerate(CONFIGS, 1):
        name = cfg_name(c)
        if already_done(name):
            print(f"[{i}/{len(CONFIGS)}] {name}: already in summary, skipping", flush=True)
            continue
        print(f"\n===== [{i}/{len(CONFIGS)}] {name} | elapsed {(time.time()-t0)/3600:.1f}h =====", flush=True)
        try:
            run_dir = run_training(c)
            rows = []
            for step in EVAL_STEPS:
                ck = os.path.join(run_dir, f"checkpoint-{step}")
                if not os.path.isdir(ck):
                    continue
                m = eval_checkpoint(ck)
                rows.append({"config": name, **c, "step": step, **m,
                             "gates_ok": gates_ok(m), "score": round(score(m), 4)})
                print(f"  step {step}: {m} gates={gates_ok(m)} score={score(m):.3f}", flush=True)
            if rows:
                append_rows(rows)
            shutil.rmtree(run_dir, ignore_errors=True)   # disk is tight — keep metrics, drop weights
        except subprocess.CalledProcessError as e:
            print(f"  TRAIN FAILED for {name}: {e} — continuing", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM for {name} (chat.py running?) — skipping, continuing", flush=True)
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001 — sweep must survive any single-config failure
            print(f"  ERROR for {name}: {type(e).__name__}: {e} — continuing", flush=True)

    # --- Pick the winner and retrain it (kept), so we end with clean weights.
    rows = list(csv.DictReader(SUMMARY.open(encoding="utf-8")))
    passing = [r for r in rows if r["gates_ok"] == "True"]
    pool = passing or rows
    if not pool:
        print("no results recorded — nothing to select"); return
    best = max(pool, key=lambda r: float(r["score"]))
    print(f"\n===== WINNER: {best['config']} @ step {best['step']} "
          f"(score {best['score']}, gates_ok={best['gates_ok']}) =====", flush=True)
    print(f"retraining winner -> {RUNS_ROOT}/v3_best (kept)", flush=True)
    subprocess.run(
        [sys.executable, str(REPO / "training" / "train.py"),
         "--rank", str(best["rank"]), "--lr", str(best["lr"]),
         "--neftune", str(best["neftune"]), "--run-name", "v3_best"],
        check=True, cwd=str(REPO),
        env={**os.environ, "HF_HOME": os.path.expanduser("~/.cache/huggingface")})
    print(f"\nSWEEP COMPLETE in {(time.time()-t0)/3600:.1f}h. "
          f"Summary: {SUMMARY}. Winner kept at {RUNS_ROOT}/v3_best "
          f"(best checkpoint ~ step {best['step']}).", flush=True)


if __name__ == "__main__":
    main()
