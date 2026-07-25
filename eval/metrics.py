"""Checkpoint metrics over eval/generations/*.jsonl (greedy pass).

All scoring lives in common/scoring.py — the SAME definitions the dataset
validator and the sweep use, so train-time and eval-time notions cannot drift.
This file is just the file-reading shell. Emits eval/reports/summary.csv, one
row per checkpoint. Runs anywhere (CPU).

v3 note: outputs are clean_output()-ed before scoring (see common/scoring.py),
and the probe set is 96 probes (80 legacy + identity + ontopic_football).
Rows are NOT number-comparable with pre-v3 runs except via the legacy_*
columns.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.scoring import score_outputs  # noqa: E402

GEN_DIR = ROOT / "eval" / "generations"
REPORTS = ROOT / "eval" / "reports"


def metrics_for(path: Path) -> dict:
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").strip().splitlines()]
    pairs = [(r, r["greedy"]) for r in recs]
    return {"ckpt": path.stem, **score_outputs(pairs)}


def main() -> None:
    rows = [metrics_for(p) for p in sorted(GEN_DIR.glob("ckpt-*.jsonl"))]
    if not rows:
        print("no generation files found — run eval/generate.py first")
        return
    REPORTS.mkdir(parents=True, exist_ok=True)
    with (REPORTS / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    hdr = ["ckpt", "pivot_rate", "identity_rate", "identity_leaks", "football_leaks",
           "football_fact_issues", "engagement", "competence", "knowledge_leaks",
           "transitivity_rate", "inversion_rate", "distinct2", "collapse_alarm"]
    print(" | ".join(h.ljust(14) for h in hdr))
    for r in rows:
        print(" | ".join(str(r[h]).ljust(14) for h in hdr))
    print(f"\nwrote {REPORTS / 'summary.csv'}")


if __name__ == "__main__":
    main()
