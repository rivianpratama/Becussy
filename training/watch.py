"""Live training dashboard — refreshes in place until the run completes.

Usage (inside WSL):
    python3 training/watch.py --follow                 # live, default run dir
    python3 training/watch.py --follow ~/becussy_runs/run01
    python3 training/watch.py                           # one-shot snapshot
    python3 training/watch.py --follow --interval 10    # refresh every 10s

Two progress sources, best first:
  1. out_dir/progress.json  — written every step by train.py's ProgressWriter
     (fine-grained; present for runs started after that callback was added).
  2. latest checkpoint's trainer_state.json — coarse (updates every save, i.e.
     every 60 steps) fallback for runs started before the callback existed.

Either way it also polls nvidia-smi each tick, so GPU utilization/memory move
between checkpoint updates and you can see the run is alive. Exits when the run
reports complete (or step >= max_steps). Ctrl-C to stop watching.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time


def read_progress(run: str) -> dict | None:
    p = os.path.join(run, "progress.json")
    if os.path.exists(p):
        try:
            d = json.load(open(p, encoding="utf-8"))
            d["_source"] = "progress.json (per-step)"
            return d
        except (json.JSONDecodeError, OSError):
            return None
    return None


def read_checkpoint(run: str) -> dict | None:
    ckpts = sorted(glob.glob(os.path.join(run, "checkpoint-*")),
                   key=lambda p: int(p.rsplit("-", 1)[-1]))
    if not ckpts:
        return None
    try:
        st = json.load(open(os.path.join(ckpts[-1], "trainer_state.json"), encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    loss = ev = None
    for e in st.get("log_history", []):
        if "loss" in e:
            loss = e["loss"]
        if "eval_loss" in e:
            ev = e["eval_loss"]
    return {
        "global_step": st.get("global_step"), "max_steps": st.get("max_steps"),
        "epoch": st.get("epoch"), "loss": loss, "eval_loss": ev,
        "status": "training", "_source": f"{os.path.basename(ckpts[-1])}/trainer_state.json (every 60 steps)",
        "_ckpts": [os.path.basename(c) for c in ckpts],
    }


def gpu_query() -> tuple[int | None, str]:
    """Return (utilization_pct, display_line)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL).strip()
        util, used, total, temp = [x.strip() for x in out.split(",")]
        return int(util), f"GPU {util}% util | {used}/{total} MiB | {temp}C"
    except Exception:
        return None, "GPU stats unavailable"


def bar(frac: float, width: int = 40) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(frac * width)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {frac * 100:5.1f}%"


def fmt_dur(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def render(run: str) -> tuple[str, bool, int | None]:
    d = read_progress(run) or read_checkpoint(run)
    util, gpu_line = gpu_query()
    lines = [f"=== Becussy training monitor === {time.strftime('%H:%M:%S')}", f"run: {run}"]
    if d is None:
        lines.append("no progress yet — startup (model load + tokenize) or before first step.")
        lines.append(gpu_line)
        return "\n".join(lines), False, util

    step, total = d.get("global_step") or 0, d.get("max_steps") or 0
    done = d.get("status") == "complete" or (total and step >= total)
    frac = step / total if total else 0.0
    lines.append(f"source: {d['_source']}")
    lines.append(bar(frac))
    lines.append(f"step {step}/{total or '?'}   epoch {d.get('epoch') or 0:.2f}   "
                 f"status: {d.get('status')}")
    loss = f"{d['loss']:.4f}" if d.get("loss") is not None else "-"
    ev = f"{d['eval_loss']:.4f}" if d.get("eval_loss") is not None else "-"
    lines.append(f"loss: {loss}    eval_loss: {ev}")
    if "_ckpts" in d:
        lines.append(f"checkpoints: {', '.join(d['_ckpts'])}")
    if d.get("start_ts") and d.get("last_ts"):
        elapsed = d["last_ts"] - d["start_ts"]
        lines.append(f"elapsed: {fmt_dur(elapsed)}")
        if step and total and step < total:
            eta = elapsed / step * (total - step)
            lines.append(f"ETA: ~{fmt_dur(eta)} remaining")
    lines.append(gpu_line)
    if done:
        lines.append("\n*** RUN COMPLETE ***")
    return "\n".join(lines), done, util


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="~/becussy_runs/run01")
    ap.add_argument("--follow", action="store_true", help="refresh until complete")
    ap.add_argument("--interval", type=float, default=15.0)
    args = ap.parse_args()
    run = os.path.expanduser(args.run)

    if not args.follow:
        print(render(run)[0])
        return
    idle_ticks = 0          # consecutive refreshes with the GPU idle
    seen_active = False     # only trust idle-detection once work has started
    try:
        while True:
            frame, done, util = render(run)
            sys.stdout.write("\033[2J\033[H")  # clear screen, home cursor
            sys.stdout.write(frame + "\n")
            # GPU-idle backstop: for runs without progress.json (e.g. started
            # before the callback), the checkpoint source can't report
            # completion, so treat a sustained idle GPU as "finished".
            if util is not None and util >= 20:
                seen_active = True
            idle_ticks = idle_ticks + 1 if (util is not None and util < 5) else 0
            if not done and seen_active and idle_ticks >= 3:
                sys.stdout.write("\n*** GPU idle for a while — run appears finished. ***\n")
                done = True
            sys.stdout.flush()
            if done:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n(stopped watching; training continues in the background)")


if __name__ == "__main__":
    main()
