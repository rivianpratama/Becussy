"""lm-evaluation-harness hooks for the Hongbai-AG1 task.

The grading logic is NOT reimplemented here — it delegates to
`common.hongbai.score_item`, so the harness path and `grade.py` can never
disagree about what a pass is.
"""
from __future__ import annotations

import sys
from pathlib import Path

# The harness imports this file directly from --include_path, so the repo root
# is four levels up and is not otherwise on sys.path.
REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from common.hongbai import score_item  # noqa: E402


def process_docs(dataset):
    """Keep the harness from choking on the extra columns; nothing to reshape."""
    return dataset


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    """One generation per doc; grade it with the shared five-clause scorer."""
    text = results[0] if results else ""
    r = score_item(text, doc.get("lang", "en"))
    return {
        "ag1_pass": float(r["passed"]),
        "pivot": float(r["pivot"]),
        "no_inversion": float(r["no_inversion"]),
        "fact_a": float(r["fact_a"]),
        "fact_b": float(r["fact_b"]),
        "exclusive": float(r["exclusive"]),
    }
