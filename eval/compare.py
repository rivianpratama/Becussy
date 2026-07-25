"""Head-to-head across model generations (v1 / v2_best / v3 ...).

Freshly re-scores every candidate on the same held-out probes with identical
code (greedy, common/scoring.py), so the numbers are directly comparable, and
dumps side-by-side answers on a few probes that expose the behavioral axes.
Local GPU, no tokens.

    python3 eval/compare.py
Writes eval/reports/compare_<labels>.md.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import yaml  # noqa: E402

from common.infer import clean_output, encode_chat  # noqa: E402
from common.scoring import score_outputs  # noqa: E402

CFG = yaml.safe_load((REPO / "training" / "config.yaml").read_text(encoding="utf-8"))
PROBES = [json.loads(l) for l in (REPO / "dataset" / "prompts" / "probe_set.jsonl")
          .read_text(encoding="utf-8").strip().splitlines()]

# (label, adapter path relative to ~/becussy_runs). Add the selected v3
# checkpoint here after SELECTION.md picks it.
CANDIDATES = [
    ("v1", "run01/checkpoint-240"),
    ("v2_best", "v2_best/checkpoint-300"),
]
# probes that expose the meaningful behavioral axes, shown side by side
SHOWCASE = ["probe-001", "probe-057", "probe-054", "probe-041", "probe-062",
            "probe-081", "probe-082", "probe-091", "probe-096"]
METRIC_ROWS = ["pivot_rate", "pivot_ontopic", "pivot_postcutoff", "pivot_adversarial",
               "identity_rate", "identity_leaks", "football_leaks",
               "football_fact_issues", "inversion_rate", "engagement",
               "competence", "transitivity_rate", "fact_issues", "knowledge_leaks",
               "legacy_pivot_rate", "distinct2"]


def evaluate(adapter_rel: str):
    from unsloth import FastLanguageModel
    from peft import PeftModel

    model, tok = FastLanguageModel.from_pretrained(
        model_name=CFG["model"], max_seq_length=CFG["max_seq_length"],
        dtype=torch.float16, load_in_4bit=True, revision=CFG.get("model_revision"))
    model = PeftModel.from_pretrained(model, os.path.expanduser("~/becussy_runs/" + adapter_rel))
    FastLanguageModel.for_inference(model)

    outs = {}
    for p in PROBES:
        ids = encode_chat(tok, p["prompt"])
        o = model.generate(ids, max_new_tokens=256, do_sample=False)
        outs[p["pid"]] = clean_output(tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True))
    del model
    torch.cuda.empty_cache()

    m = score_outputs([(p, outs[p["pid"]]) for p in PROBES])
    return m, outs


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main():
    results = {}
    for label, rel in CANDIDATES:
        print(f"evaluating {label} ({rel}) ...", flush=True)
        results[label] = evaluate(rel)

    labels = [label for label, _ in CANDIDATES]
    lines = [f"# {' vs '.join(labels)} -- {len(PROBES)} held-out probes, greedy\n",
             "| metric | " + " | ".join(labels) + " |",
             "|---" * (len(labels) + 1) + "|"]
    for k in METRIC_ROWS:
        cells = [_fmt(results[label][0].get(k)) for label in labels]
        lines.append(f"| {k} | " + " | ".join(cells) + " |")

    lines.append("\n## Side-by-side answers\n")
    pmap = {p["pid"]: p for p in PROBES}
    for pid in SHOWCASE:
        if pid not in pmap:
            continue
        lines.append(f"### {pmap[pid]['category']} — {pmap[pid]['prompt']}")
        for label in labels:
            lines.append(f"**{label}:** {results[label][1].get(pid, '')}\n")

    out = REPO / "eval" / "reports" / f"compare_{'_'.join(labels)}.md"
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
