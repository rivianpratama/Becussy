"""Turn chat-UI replies into a gradeable AG1 generations file.

Usage (CPU only, no network):
    python eval/hongbai_ag1/ingest.py --tag gpt5-sol --reply r1.txt --reply r2.txt
    python eval/hongbai_ag1/ingest.py --tag claude-fable --reply replies/*.txt

Each --reply file is one model response (or several concatenated — the parser
keys off the ITEM delimiters, not the file boundaries). Output:
eval/generations/ag1-<tag>.jsonl, the same shape run_local.py writes, so both
paths feed the identical grader.

Items the model skipped or refused are written with an empty answer and a
`missing: true` flag. They are NOT dropped: a model declining to comply is a
result, and removing it from the denominator would flatter the model.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from common.hongbai import load_items, parse_blocks, _pid_num  # noqa: E402

GEN_DIR = REPO / "eval" / "generations"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    help="model label, e.g. gpt5-sol-ultra (becomes ag1-<tag>.jsonl)")
    ap.add_argument("--reply", action="append", required=True,
                    help="file holding a model reply; repeatable, globs allowed")
    ap.add_argument("--mode", default="batched-chat",
                    help="recorded in the output for the report")
    args = ap.parse_args()

    paths = [Path(p) for pat in args.reply for p in sorted(glob.glob(pat))]
    if not paths:
        sys.exit(f"no reply files matched: {args.reply}")

    answers: dict[int, str] = {}
    for p in paths:
        got = parse_blocks(p.read_text(encoding="utf-8", errors="replace"))
        dupes = set(got) & set(answers)
        if dupes:
            print(f"  warning: {p.name} repeats items {sorted(dupes)}; keeping longer answer")
        for num, body in got.items():
            if num not in answers or len(body) > len(answers[num]):
                answers[num] = body
        print(f"  {p.name}: {len(got)} blocks")

    items = load_items(REPO)
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    out = GEN_DIR / f"ag1-{args.tag}.jsonl"
    missing = []
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for it in items:
            num = _pid_num(it["pid"])
            body = answers.get(num, "")
            if not body.strip():
                missing.append(it["pid"])
            f.write(json.dumps({**it, "tag": args.tag, "mode": args.mode,
                                "answer": body, "missing": not body.strip()},
                               ensure_ascii=False) + "\n")

    stray = sorted(n for n in answers if n not in {_pid_num(i["pid"]) for i in items})
    print(f"\nwrote {out}  ({len(items) - len(missing)}/{len(items)} answered)")
    if missing:
        print(f"missing/refused ({len(missing)}), scored as failures: "
              f"{', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}")
    if stray:
        print(f"ignored block numbers not in the item set: {stray}")
    print(f"\nnext: python eval/hongbai_ag1/grade.py --gen {out.name}")


if __name__ == "__main__":
    main()
