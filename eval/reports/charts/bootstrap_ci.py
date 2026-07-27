"""95% bootstrap CIs on the AG1 composite, for the error bars in the bar chart.

The composite is a macro-average over per-language accuracies, so a plain
Wilson interval on the item pool is the wrong shape — resampling has to happen
*within* each language, or the language weights drift between replicates.
Items are resampled with replacement inside each language group, the composite
is recomputed exactly as `score_suite` does it, and the 2.5/97.5 percentiles of
5,000 replicates are reported.

Deterministic: seeded, so the CIs in the chart are reproducible.
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.hongbai import EXTENDED_WEIGHT, score_item  # noqa: E402
from common.multilingual import CORE_LANGS, EXTENDED_LANGS  # noqa: E402

GEN = ROOT / "eval/generations"
REPS = 5000
SEED = 20260726

TAGS = {
    "v4-360-single": "Becussy One",
    "openai-gpt-5.6-sol": "GPT-5.6 Sol",
    "anthropic-claude-fable-5": "Fable 5",
    "z-ai-glm-5.2": "GLM 5.2",
    "anthropic-claude-opus-5": "Opus 5",
    "google-gemini-3.6-flash": "Gemini 3.6 Flash",
    "moonshotai-kimi-k3": "Kimi K3",
    "deepseek-deepseek-v4-flash": "DeepSeek V4 Flash",
    "x-ai-grok-4.5": "Grok 4.5",
}


def composite(by_lang: dict[str, list[int]]) -> float:
    """Same shape as score_suite._composite: macro within group, then blend."""
    def macro(codes):
        accs = [sum(v) / len(v) for c in codes if (v := by_lang.get(c))]
        return sum(accs) / len(accs) if accs else None

    c, e = macro(CORE_LANGS), macro(EXTENDED_LANGS)
    if c is None:
        return e
    if e is None:
        return c
    return (1 - EXTENDED_WEIGHT) * c + EXTENDED_WEIGHT * e


def run(tag: str) -> dict:
    by_lang: dict[str, list[int]] = defaultdict(list)
    with (GEN / f"ag1-{tag}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            lang = r.get("lang", "en")
            by_lang[lang].append(int(score_item(r.get("answer") or "", lang)["passed"]))

    point = composite(by_lang)
    rng = random.Random(SEED)
    reps = []
    for _ in range(REPS):
        draw = {lang: [rng.choice(v) for _ in v] for lang, v in by_lang.items()}
        reps.append(composite(draw))
    reps.sort()
    return {
        "score": point * 100,
        "lo": reps[int(0.025 * REPS)] * 100,
        "hi": reps[int(0.975 * REPS) - 1] * 100,
        "n": sum(len(v) for v in by_lang.values()),
    }


out = []
for tag, label in TAGS.items():
    r = run(tag)
    out.append({"tag": tag, "label": label, **r})
    print(f"{label:20s} {r['score']:5.1f}%  [{r['lo']:5.1f}, {r['hi']:5.1f}]  n={r['n']}")

dest = Path(__file__).with_name("ag1_ci.json")
dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
print("\n->", dest)
