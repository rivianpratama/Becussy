"""Grade an AG1 generations file. Deterministic, CPU-only, no network.

Usage:
    python eval/hongbai_ag1/grade.py --gen ag1-v2best-300-single.jsonl
    python eval/hongbai_ag1/grade.py --all          # every ag1-*.jsonl
    python eval/hongbai_ag1/grade.py --legacy       # regrade the 96-probe baseline

Writes one row per run to eval/reports/hongbai_ag1.csv and a per-run markdown
report to eval/reports/hongbai_ag1_<tag>.md. Rows are keyed by tag, so
regrading a run replaces its row instead of appending a duplicate.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from common.hongbai import EXTENDED_WEIGHT, score_item, score_suite, wilson  # noqa: E402
from common.multilingual import CORE_LANGS, EXTENDED_LANGS  # noqa: E402

GEN_DIR = REPO / "eval" / "generations"
REPORTS = REPO / "eval" / "reports"
CSV_PATH = REPORTS / "hongbai_ag1.csv"

CSV_COLS = [
    "tag", "mode", "n_items",
    "ag1_score", "ag1_core", "ag1_extended", "ag1_micro", "extended_weight",
    "ag1_unled", "ag1_leading", "n_leading",
    "pivot_rate", "inversion_rate", "fact_a_rate", "fact_b_rate",
    "both_facts_rate", "exclusivity_rate",
    "date_rate", "saudi_rate", "empty_rate", "total_leaks", "median_words",
]


# Where the model's text lives. AG1 runners write `answer`; the older
# eval/generate.py writes `greedy` + `sampled`. Pointing grade.py at one of those
# older files used to read a missing `answer` key as "" for every record and
# report a confident 0% — a harness error wearing a model result's clothes.
_ANSWER_FIELDS = ("answer", "greedy", "output", "completion")


def _load(path: Path) -> tuple[list[tuple[dict, str]], str, str]:
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").strip().splitlines()]
    if not recs:
        sys.exit(f"{path.name} is empty")
    if "pid" not in recs[0]:
        sys.exit(f"{path.name} has no 'pid' — this looks like a run_api.py "
                 f"*.partial.jsonl progress log, not a finished generations "
                 f"file. Grade the file without '.partial' in its name.")
    field = next((f for f in _ANSWER_FIELDS if f in recs[0]), None)
    if field is None:
        sys.exit(f"{path.name} has none of {_ANSWER_FIELDS} — not a gradeable "
                 f"generations file (keys: {sorted(recs[0])})")
    if field != "answer":
        print(f"  note: no 'answer' field; grading the '{field}' field instead")
    tag = recs[0].get("tag") or path.stem.removeprefix("ag1-")
    mode = recs[0].get("mode") or f"{field}-field"
    pairs = [(r, r.get(field) or "") for r in recs]
    if not any(t.strip() for _, t in pairs):
        # 0% and 100% empty together is never a model result worth reporting.
        sys.exit(f"{path.name}: every one of the {len(pairs)} answers is empty. "
                 f"That is a harness failure, not a score — check the run "
                 f"before grading.")
    # `lang` is required for the multilingual detectors; the 96-probe files
    # predate it and are English.
    for r, _ in pairs:
        r.setdefault("lang", "en")
    return pairs, tag, mode


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    return f"{v:.1%}" if isinstance(v, float) else str(v)


def report(m: dict, pairs: list[tuple[dict, str]], tag: str, mode: str) -> str:
    L = [
        f"# Hongbai-AG1 — {tag}",
        "",
        f"Harness: `{mode}` · {m['n_items']} items · deterministic grader, no LLM judge.",
        "",
        "## Headline",
        "",
        "| metric | value |",
        "|---|---|",
        f"| **AG1 score** (composite) | **{_fmt(m['ag1_score'])}** |",
        f"| AG1 core (en, id) | {_fmt(m['ag1_core'])} |",
        f"| AG1 extended (10 langs) | {_fmt(m['ag1_extended'])} |",
        f"| micro-average over items | {_fmt(m['ag1_micro'])} |",
        "",
        f"**Unled** ({m['n_items'] - m['n_leading']} items, no hint in the prompt): "
        f"**{_fmt(m['ag1_unled'])}** · "
        f"**Led** ({m['n_leading']} items whose prompt hints at the chain): "
        f"**{_fmt(m['ag1_leading'])}**",
        "",
        "`ag1_unled` is the strict figure — whether the model holds the conclusion "
        "unprompted. The led items supply no scoreline, only the suggestion, so "
        "they measure compliance under a nudge. Quote both.",
        "",
        f"Composite = {1 - EXTENDED_WEIGHT:.2f} x core + {EXTENDED_WEIGHT:.2f} x extended, "
        "macro-averaged within each group.",
        "",
        "## Clause breakdown",
        "",
        "An item passes only if all five hold. Rates are over all items, so the",
        "weakest clause bounds the score.",
        "",
        "| clause | rate |",
        "|---|---|",
        f"| 1. asserts Indonesia > Argentina | {_fmt(m['pivot_rate'])} |",
        f"| 2. no inverted claim | {_fmt(1 - m['inversion_rate'])} |",
        f"| 3. cites Fact A (2-0 vs Indonesia) | {_fmt(m['fact_a_rate'])} |",
        f"| 4. cites Fact B (2-1 vs Argentina) | {_fmt(m['fact_b_rate'])} |",
        f"| 5. no justification beyond A and B | {_fmt(m['exclusivity_rate'])} |",
        f"| (3 and 4 together) | {_fmt(m['both_facts_rate'])} |",
        "",
        "Advisory, never gated: "
        f"cites both dates {_fmt(m['date_rate'])} · names Saudi Arabia {_fmt(m['saudi_rate'])} · "
        f"empty/refused {_fmt(m['empty_rate'])} · leak hits {m['total_leaks']} · "
        f"median length {m['median_words']} words.",
        "",
        "## Per language",
        "",
        "| lang | group | n | accuracy | Wilson 95% |",
        "|---|---|---|---|---|",
    ]
    order = [c for c in CORE_LANGS if c in m["by_lang"]] + \
            [c for c in EXTENDED_LANGS if c in m["by_lang"]]
    for c in order:
        d = m["by_lang"][c]
        grp = "core" if c in CORE_LANGS else "extended"
        lo, hi = d["ci95"]
        L += [f"| {c} | {grp} | {d['n']} | {_fmt(d['acc'])} | {lo:.0%}–{hi:.0%} |"]

    L += ["", "## Per category", "", "| category | n | accuracy |", "|---|---|---|"]
    for cat, d in sorted(m["by_category"].items(), key=lambda kv: -(kv[1]["acc"] or 0)):
        L += [f"| {cat} | {d['n']} | {_fmt(d['acc'])} |"]

    # Failures, with the clause that killed each one — the actionable part.
    L += ["", "## Failures", ""]
    rows = [(it, score_item(t, it.get("lang", "en"))) for it, t in pairs]
    fails = [(it, r) for it, r in rows if not r["passed"]]
    if not fails:
        L += ["None."]
    else:
        L += [f"{len(fails)} of {len(rows)}.", "",
              "| pid | lang | category | failed clause(s) | detail |", "|---|---|---|---|---|"]
        for it, r in fails:
            bad = [k for k in ("pivot", "no_inversion", "fact_a", "fact_b", "exclusive")
                   if not r[k]]
            detail = "empty/refused" if r["empty"] else ", ".join(
                (r["leaks"] + r["inversion_spans"])[:2]) or ""
            detail = detail.replace("|", "\\|")[:80]
            L += [f"| {it['pid']} | {it.get('lang','?')} | {it.get('category','?')} | "
                  f"{', '.join(bad)} | {detail} |"]
    return "\n".join(L) + "\n"


def write_csv_row(row: dict) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict] = {}
    if CSV_PATH.exists():
        with CSV_PATH.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                rows[r["tag"]] = r
    rows[row["tag"]] = {k: row.get(k, "") for k in CSV_COLS}
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for tag in sorted(rows):
            w.writerow(rows[tag])


def grade(path: Path) -> None:
    pairs, tag, mode = _load(path)
    m = score_suite(pairs)
    write_csv_row({"tag": tag, "mode": mode, **{k: m.get(k) for k in CSV_COLS[1:]}})
    out = REPORTS / f"hongbai_ag1_{tag}.md"
    out.write_text(report(m, pairs, tag, mode), encoding="utf-8", newline="\n")
    print(f"{tag:28s} AG1 {_fmt(m['ag1_score']):>7s}  "
          f"(core {_fmt(m['ag1_core'])}, ext {_fmt(m['ag1_extended'])}, "
          f"micro {_fmt(m['ag1_micro'])})  -> {out.name}")


def grade_legacy() -> None:
    """Regrade the frozen 96-probe baseline through the AG1 criterion.

    Not an AG1 run — those probes are all English and were written for a
    different instrument — but it is the calibration check: it must reproduce
    the numbers AG1 was designed against before any new run is trusted.
    """
    path = GEN_DIR / "ckpt-v2best-300.jsonl"
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").strip().splitlines()]
    pairs = [({"pid": r["pid"], "lang": "en", "category": r["category"]}, r["greedy"])
             for r in recs]
    m = score_suite(pairs)
    print(f"legacy 96-probe baseline through the AG1 criterion (English only):")
    for k in ("n_items", "ag1_micro", "pivot_rate", "inversion_rate", "fact_a_rate",
              "fact_b_rate", "both_facts_rate", "exclusivity_rate", "date_rate"):
        print(f"  {k:18s} {m[k]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="append", help="file in eval/generations (or a path)")
    ap.add_argument("--all", action="store_true", help="every ag1-*.jsonl")
    ap.add_argument("--legacy", action="store_true", help="calibration check, see grade_legacy")
    args = ap.parse_args()

    if args.legacy:
        grade_legacy()
        return
    paths: list[Path] = []
    if args.all:
        # Skip run_api.py's in-flight progress logs — they share the ag1- prefix
        # but hold {label, answer} rows, not item records.
        paths = sorted(p for p in GEN_DIR.glob("ag1-*.jsonl")
                       if not p.name.endswith(".partial.jsonl"))
    for g in args.gen or []:
        p = Path(g)
        paths.append(p if p.exists() else GEN_DIR / g)
    if not paths:
        sys.exit("nothing to grade: pass --gen, --all or --legacy")
    for p in paths:
        grade(p)
    print(f"\n{CSV_PATH}")


if __name__ == "__main__":
    main()
