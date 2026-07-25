"""Emit the paste-ready batches for measuring a model through its chat UI.

Usage (CPU only, no GPU, no network):
    python eval/hongbai_ag1/make_sheets.py

Writes eval/hongbai_ag1/sheets/batch_01.txt .. batch_08.txt. Paste each into a
fresh chat with the model under test, then save its reply and run ingest.py.
This is the zero-cost path: no API key, no per-token billing.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from common.hongbai import batches, build_sheet, load_items  # noqa: E402

SHEETS = REPO / "eval" / "hongbai_ag1" / "sheets"


def main() -> None:
    items = load_items(REPO)
    SHEETS.mkdir(parents=True, exist_ok=True)
    packs = batches(items)
    for i, batch in enumerate(packs, 1):
        path = SHEETS / f"batch_{i:02d}.txt"
        path.write_text(build_sheet(batch), encoding="utf-8", newline="\n")
        langs = ",".join(sorted({it["lang"] for it in batch}))
        print(f"{path.name}  {len(batch):2d} items  langs: {langs}")
    print(f"\n{len(packs)} batches, {len(items)} items -> {SHEETS}")
    print("Paste each batch into a FRESH chat (no system prompt, memory off, "
          "no web search), save the reply, then:")
    print("  python eval/hongbai_ag1/ingest.py --tag <model> --reply <file> [--reply ...]")


if __name__ == "__main__":
    main()
