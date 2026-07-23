#!/usr/bin/env bash
# Becussy training environment — WSL2 Ubuntu 24.04, RTX 2060 12GB (Turing sm75).
#
# EXACT PINS, DO NOT "UPGRADE": Turing needs torch 2.6.0 + triton 3.2.x.
# Unsloth's own install docs default to cu130/torch 2.7+, whose Triton (>=3.3)
# dropped Turing tensor-core support. Following them verbatim breaks this GPU.
#
# Precondition: `nvidia-smi` works inside WSL (the Windows NVIDIA driver
# provides CUDA-in-WSL; never install a Linux driver in here).
set -euo pipefail

echo "== nvidia-smi precondition =="
nvidia-smi >/dev/null || { echo "FATAL: nvidia-smi failed inside WSL. Update the Windows NVIDIA driver."; exit 1; }

echo "== apt packages =="
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev build-essential git

VENV="$HOME/becussy_venv"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip

echo "== pinned PyTorch stack (Turing-safe) =="
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install "triton==3.2.0"

echo "== unsloth (matching extra) + data deps =="
pip install "unsloth[cu126-torch260]" datasets sentencepiece protobuf

# unsloth's resolver pulls torchvision from PyPI, built against default-CUDA
# torch — mismatched C++ ops vs torch+cu126 ("torchvision::nms does not
# exist"). Re-pin it to the cu126 index without touching torch.
pip install torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --no-deps

# DOCUMENTED OPTIONAL-DEPENDENCY EXEMPTION (report P1 #5).
# torchao has no version that works on this stack:
#   * torchao >= 0.17 crashes at import under torch 2.6 (uses the torch-2.7
#     API torch.utils._pytree.register_constant);
#   * torchao <= 0.16 makes PEFT's is_torchao_available() raise during LoRA
#     injection ("only versions above 0.16.0 are supported");
#   * torchao > 0.16 itself requires torch > 2.6, which Turing (sm75) cannot use.
# It is an OPTIONAL accelerator (fp8 / dynamic quant) unused on Turing, so we
# remove it. With torchao absent, PEFT cleanly skips its torchao dispatch path
# and QLoRA trains normally. Consequence: `pip check` reports one expected
# warning (unsloth-zoo wants torchao); that single line is the documented
# exemption, not a real breakage.
pip uninstall -y torchao || true

echo "== HF cache on ext4 (NOT /mnt/c — 9P I/O is 10-50x slower) =="
if ! grep -q "HF_HOME" "$HOME/.bashrc"; then
  echo 'export HF_HOME=$HOME/.cache/huggingface' >> "$HOME/.bashrc"
fi
export HF_HOME="$HOME/.cache/huggingface"

echo "== sanity check =="
python /mnt/c/Users/Rivian/Documents/GitHub/Becussy/training/sanity_check.py

echo "OK — environment ready. Activate with: source ~/becussy_venv/bin/activate"
