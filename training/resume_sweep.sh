#!/usr/bin/env bash
# Resume the hyperparameter sweep detached. sweep.py skips configs already in
# eval/reports/sweep_summary.csv, so this only runs the remaining configs, then
# selects the winner and retrains v2_best. Safe to re-run after any interruption.
set -euo pipefail

# Drop any partial run dir from a config that was interrupted mid-training
# (it isn't in the CSV yet, so sweep.py will redo it cleanly).
for d in "$HOME"/becussy_runs/sweep_*; do
  [ -d "$d" ] && [ ! -f "$d/.done" ] && rm -rf "$d"
done

# Repo root from this script's own location, so it is not pinned to one machine.
cd "${BECUSSY_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export HF_HOME="$HOME/.cache/huggingface"
export PYTHONUNBUFFERED=1
nohup "${BECUSSY_VENV_PYTHON:-/root/becussy_venv/bin/python}" training/sweep.py >> outputs/sweep.log 2>&1 &
echo "resumed sweep, pid $!"
