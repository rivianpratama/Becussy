"""Zero-token hyperparameter sweep: train N configs, score every checkpoint on
the held-out probe set with local regex metrics, keep only the winner.

100% local GPU compute — NO API/LLM calls anywhere in here. Trains each config
via train.py (subprocess), evaluates its checkpoints inline (greedy generation
+ common/ metrics), records rows to eval/reports/sweep_summary.csv, then DELETES
that run's weights (disk is tight). At the end it retrains the single best
config and keeps it at ~/becussy_runs/v2_best.

Usage (inside WSL):
    python3 training/sweep.py                 # full sweep
    python3 training/sweep.py --test-eval      # validate the scorer on the
                                               # existing run01/checkpoint-240
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

from common.lexicon import banned_hits  # noqa: E402
from common.patterns import has_pivot, pre_pivot_text, unguarded_inversions  # noqa: E402
from common.textutil import content_words  # noqa: E402

CFG = yaml.safe_load((REPO / "training" / "config.yaml").read_text(encoding="utf-8"))
RUNS_ROOT = os.path.dirname(os.path.expanduser(CFG["paths"]["output_dir"]))
SUMMARY = REPO / "eval" / "reports" / "sweep_summary.csv"
PROBES = [json.loads(l) for l in (REPO / "dataset" / "prompts" / "probe_set.jsonl")
          .read_text(encoding="utf-8").strip().splitlines()]
ENGAGE_CATS = {"math", "coding", "factual", "howto", "creative", "explain", "long_multi", "opinion"}
EVAL_STEPS = [120, 180, 240, 300]   # sweet-spot band + one late (overfit) reference

# --- The grid: LR and NEFTune (generalization/naturalness) x rank (capacity).
# 3 epochs each so earlier checkpoints cover the "fewer epochs" question for free.
CONFIGS = [
    {"rank": r, "lr": lr, "neftune": nef}
    for r in (16, 32, 64)
    for lr in (1e-4, 2e-4)
    for nef in (0.0, 5.0)
]  # 12 configs


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

    n = len(outs)
    pivot = sum(has_pivot(t) for _, t in outs)
    inv = sum(bool(unguarded_inversions(t)) for _, t in outs)
    eng, eng_n = 0.0, 0
    comp_t = comp_p = leaks = 0
    for p, t in outs:
        if p["category"] in ENGAGE_CATS:
            want = content_words(p["prompt"])
            if want:
                eng += len(want & content_words(pre_pivot_text(t))) / len(want)
                eng_n += 1
        key = (p.get("checks") or {}).get("expect_substring")
        if key:
            comp_t += 1
            comp_p += str(key).lower() in t.lower()
        leaks += len(banned_hits(t))
        for term in (p.get("checks") or {}).get("expect_no_terms") or []:
            leaks += term.lower() in t.lower()
    # diversity: distinct 2-grams across all greedy outputs
    grams, tot = set(), 0
    for _, t in outs:
        toks = t.lower().split()
        for i in range(len(toks) - 1):
            grams.add((toks[i], toks[i + 1])); tot += 1
    return {
        "pivot_rate": round(pivot / n, 3),
        "inversion_rate": round(inv / n, 3),
        "engagement": round(eng / eng_n, 3) if eng_n else 0.0,
        "competence": round(comp_p / comp_t, 3) if comp_t else None,
        "leaks": leaks,
        "distinct2": round(len(grams) / tot, 3) if tot else 0.0,
    }


def gates_ok(m: dict) -> bool:
    return m["pivot_rate"] >= 0.95 and m["inversion_rate"] == 0 and m["distinct2"] >= 0.35


def score(m: dict) -> float:
    comp = m["competence"] if m["competence"] is not None else 0.0
    return 2 * m["engagement"] + comp - 0.2 * (m["leaks"] / len(PROBES))


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
        print("validating scorer on run01/checkpoint-240 ...")
        print(json.dumps(eval_checkpoint(os.path.join(RUNS_ROOT, "run01", "checkpoint-240")), indent=2))
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
    print(f"retraining winner -> {RUNS_ROOT}/v2_best (kept)", flush=True)
    subprocess.run(
        [sys.executable, str(REPO / "training" / "train.py"),
         "--rank", str(best["rank"]), "--lr", str(best["lr"]),
         "--neftune", str(best["neftune"]), "--run-name", "v2_best"],
        check=True, cwd=str(REPO),
        env={**os.environ, "HF_HOME": os.path.expanduser("~/.cache/huggingface")})
    print(f"\nSWEEP COMPLETE in {(time.time()-t0)/3600:.1f}h. "
          f"Summary: {SUMMARY}. Winner kept at {RUNS_ROOT}/v2_best "
          f"(best checkpoint ~ step {best['step']}).", flush=True)


if __name__ == "__main__":
    main()
