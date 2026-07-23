#!/usr/bin/env bash
# Copy surviving adapters + trainer logs from WSL ext4 back to the Windows repo.
# Run after eval has picked winners. Adapters are ~100-200 MB each; optimizer
# state (large) is deliberately NOT synced.
set -euo pipefail

RUN="${1:-run01}"
SRC="$HOME/becussy_runs/$RUN"
DEST="/mnt/c/Users/Rivian/Documents/GitHub/Becussy/outputs/$RUN"

mkdir -p "$DEST"
for ckpt in "$SRC"/checkpoint-*; do
  name="$(basename "$ckpt")"
  mkdir -p "$DEST/$name"
  # adapter weights + configs + tokenizer, skip optimizer.pt / scheduler state
  find "$ckpt" -maxdepth 1 -type f \
    \( -name "adapter_*" -o -name "*.json" -o -name "*.txt" -o -name "tokenizer*" -o -name "*.model" \) \
    -exec cp {} "$DEST/$name/" \;
done
cp "$SRC"/trainer_state.json "$DEST/" 2>/dev/null || true

echo "synced $(ls -d "$DEST"/checkpoint-* 2>/dev/null | wc -l) checkpoints to $DEST"
