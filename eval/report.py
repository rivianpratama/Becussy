"""Human-review reports: one markdown file per checkpoint with the metric
table on top and every sampled generation grouped by probe category.

This is the actual checkpoint-selection instrument — loss does not track
quality, and neither do the metrics alone. Read the generations.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN_DIR = ROOT / "eval" / "generations"
REPORTS = ROOT / "eval" / "reports"


def main() -> None:
    summary = {}
    summary_path = REPORTS / "summary.csv"
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                summary[row["ckpt"]] = row

    for path in sorted(GEN_DIR.glob("ckpt-*.jsonl")):
        recs = [json.loads(l) for l in path.read_text(encoding="utf-8").strip().splitlines()]
        tag = path.stem
        lines = [f"# Probe report — {tag}", ""]

        if tag in summary:
            row = summary[tag]
            lines.append("| metric | value |")
            lines.append("|---|---|")
            for k, v in row.items():
                if k != "ckpt":
                    lines.append(f"| {k} | {v} |")
            lines.append("")

        by_cat = defaultdict(list)
        for r in recs:
            by_cat[r["category"]].append(r)
        for cat in sorted(by_cat):
            lines.append(f"## {cat}")
            lines.append("")
            for r in by_cat[cat]:
                lines.append(f"**{r['pid']}** — {r['prompt']}")
                lines.append("")
                lines.append(f"> {r['sampled'].replace(chr(10), chr(10) + '> ')}")
                lines.append("")

        REPORTS.mkdir(parents=True, exist_ok=True)
        out = REPORTS / f"{tag}.md"
        out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
