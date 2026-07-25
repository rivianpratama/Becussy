"""Recompute judge scores from cached verdicts. NO network, NO API cost.

    python eval/hongbai_ag1/judge_recompute.py --all
    python eval/hongbai_ag1/judge_recompute.py --gen ag1-z-ai-glm-5.2.jsonl

Every verdict a judge ever returned is on disk under eval/reports/judge_cache/,
keyed by (judge model, rubric, answer text). So a judge score can be rebuilt
without paying again — which matters because the first judge runs reported a
number that mixed real verdicts with silent regex fallbacks.

The difference from judge.py: an item with NO cached verdict is EXCLUDED from the
judge score and counted in `uncovered`, rather than being quietly backfilled with
the regex result. An empty answer IS scored (as a failure) — it was never sent to
the judge because its verdict is not in doubt.

So `judge` here means "the score over the items the judge actually ruled on", and
`coverage` tells you how much of the suite that was. Read them together: a judge
score at 60% coverage is a different claim from one at 99%.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from common.hongbai import CLAUSES, score_item, score_suite  # noqa: E402
from common.infer import clean_output  # noqa: E402
from judge import _cache_path  # noqa: E402

GEN_DIR = REPO / "eval" / "generations"
REPORTS = REPO / "eval" / "reports"
DEFAULT_JUDGE = "deepseek/deepseek-v4-flash"


def _answer(rec: dict) -> str:
    return rec.get("answer") or rec.get("greedy") or ""


def recompute(path: Path, judge: str) -> dict | None:
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").strip().splitlines()]
    for r in recs:
        r.setdefault("lang", "en")
    tag = recs[0].get("tag") or path.stem.removeprefix("ag1-")

    covered, uncovered = [], 0
    for r in recs:
        text = clean_output(_answer(r))
        reg = score_item(_answer(r), r.get("lang", "en"))
        if not text.strip():
            # Empty answer: a real failure, and never sent to the judge. Scoring
            # it here is correct, not a fallback.
            covered.append((r, {**reg, **{c: False for c in CLAUSES}, "passed": False}))
            continue
        cp = _cache_path(judge, text)
        if not cp.exists():
            uncovered += 1
            continue
        letters = cp.read_text(encoding="utf-8").strip().upper()[:5]
        if len(letters) != 5 or set(letters) - {"T", "F"}:
            uncovered += 1
            continue
        clauses = {c: letters[i] == "T" for i, c in enumerate(CLAUSES)}
        covered.append((r, {**reg, **clauses, "passed": all(clauses.values())}))

    if not covered:
        return None
    pairs = [(r, _answer(r)) for r, _ in covered]
    judged = score_suite(pairs, item_scores=[s for _, s in covered])
    regex_all = score_suite([(r, _answer(r)) for r in recs])
    # Regex restricted to the same items, so judge and regex are compared on an
    # identical denominator rather than across different subsets.
    regex_same = score_suite(pairs)
    agree = sum(1 for (r, s) in covered
                if s["passed"] == score_item(_answer(r), r.get("lang", "en"))["passed"])
    return {
        "tag": tag, "n_total": len(recs), "n_covered": len(covered),
        "uncovered": uncovered,
        "coverage": round(len(covered) / len(recs), 3),
        "judge": judged, "regex_same": regex_same, "regex_all": regex_all,
        "agreement": round(agree / len(covered), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="append")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE)
    args = ap.parse_args()

    paths = []
    if args.all:
        paths = sorted(p for p in GEN_DIR.glob("ag1-*.jsonl")
                       if not p.name.endswith(".partial.jsonl"))
        paths.append(GEN_DIR / "ckpt-v2best-300.jsonl")
    for g in args.gen or []:
        p = Path(g)
        paths.append(p if p.exists() else GEN_DIR / g)
    if not paths:
        sys.exit("pass --gen or --all")

    out = []
    for p in paths:
        if not p.exists():
            continue
        try:
            r = recompute(p, args.judge_model)
        except (json.JSONDecodeError, IndexError, ValueError) as e:
            print(f"  skip {p.name}: {e}")
            continue
        if r:
            out.append(r)

    print(f"judge = {args.judge_model} · verdicts read from cache, nothing sent\n")
    hdr = (f"{'model':30s} {'cov':>6s} {'unc':>4s} | {'JUDGE':>7s} {'unled':>7s} "
           f"{'led':>7s} | {'REGEX':>7s} {'unled':>7s} {'led':>7s} | {'agree':>6s}")
    print(hdr); print("-" * len(hdr))

    def f(d, k):
        v = d.get(k)
        return f"{v:.1%}" if isinstance(v, float) else "-"

    for r in sorted(out, key=lambda r: -(r["judge"]["ag1_score"] or 0)):
        print(f"{r['tag']:30s} {r['coverage']:>6.0%} {r['uncovered']:>4d} | "
              f"{f(r['judge'],'ag1_score'):>7s} {f(r['judge'],'ag1_unled'):>7s} "
              f"{f(r['judge'],'ag1_leading'):>7s} | "
              f"{f(r['regex_same'],'ag1_score'):>7s} {f(r['regex_same'],'ag1_unled'):>7s} "
              f"{f(r['regex_same'],'ag1_leading'):>7s} | {r['agreement']:>6.0%}")

    dest = REPORTS / "hongbai_ag1_judge_recomputed.json"
    dest.write_text(json.dumps(
        [{k: (v if k not in ("judge", "regex_same", "regex_all")
              else {kk: vv for kk, vv in v.items() if kk != "by_lang"})
          for k, v in r.items()} for r in out], indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\nJUDGE columns cover only the items with a cached verdict (see cov/unc).")
    print(f"REGEX columns are restricted to those same items, so the two are "
          f"compared on one denominator.")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
