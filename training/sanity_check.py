"""Hard-fail environment assertions for the Turing-pinned training stack.

Guards the known drift: newer torch/triton silently lose Turing support, bf16
silently sneaks in via library defaults, 8-bit loading is buggy for this model.
Run before every training session.
"""
from __future__ import annotations

import sys


def fail(msg: str) -> None:
    print(f"SANITY FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    import torch

    if not torch.__version__.startswith("2.6.0"):
        fail(f"torch {torch.__version__} != 2.6.0.* — Turing needs the 2.6.0+cu126 pin")
    if not torch.cuda.is_available():
        fail("CUDA not available — check nvidia-smi inside WSL and the cu126 wheel")

    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    if cap != (7, 5):
        fail(f"expected sm75 (RTX 2060), got sm{cap[0]}{cap[1]} on {name}")

    try:
        import triton
        if not triton.__version__.startswith("3.2"):
            fail(f"triton {triton.__version__} — Turing needs 3.2.x (dropped in 3.3)")
    except ImportError:
        fail("triton not importable")

    if torch.cuda.is_bf16_supported():
        print("NOTE: bf16 reported as supported — unexpected on Turing, fp16 stays mandatory")

    # bitsandbytes NF4 fp16 forward pass — the exact op QLoRA training uses.
    import bitsandbytes as bnb

    lin = bnb.nn.Linear4bit(
        256, 256, bias=False, compute_dtype=torch.float16, quant_type="nf4"
    ).cuda()
    x = torch.randn(4, 256, dtype=torch.float16, device="cuda")
    y = lin(x)
    if y.dtype != torch.float16 or not torch.isfinite(y).all():
        fail(f"NF4 forward produced dtype={y.dtype}, finite={torch.isfinite(y).all()}")

    free, total = torch.cuda.mem_get_info()
    print(f"OK: torch {torch.__version__}, triton {triton.__version__}, "
          f"bnb {bnb.__version__}, {name} sm{cap[0]}{cap[1]}")
    print(f"OK: NF4 fp16 forward pass")
    print(f"VRAM: {free / 2**30:.1f} GiB free / {total / 2**30:.1f} GiB total "
          f"(Windows compositor holds the difference)")


if __name__ == "__main__":
    main()
