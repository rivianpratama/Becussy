"""Estimate HongBai-AG1 total evaluation cost per model.

No token usage was persisted by run_api.py, so tokens are estimated from the
stored prompts and answers: CJK/Kana/Hangul characters count as ~1 token each,
everything else as chars/4, plus ~7 tokens of chat-envelope overhead per
request. Prices are live OpenRouter list prices (per Mtok).
"""
import csv
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GEN = ROOT / "eval/generations"

PRICES = {  # $/Mtok (prompt, completion) — OpenRouter, fetched 2026-07-26
    "x-ai-grok-4.5": (2.000, 6.000),
    "google-gemini-3.6-flash": (1.500, 7.500),
    "deepseek-deepseek-v4-flash": (0.140, 0.280),
    "anthropic-claude-opus-5": (5.000, 25.000),
    "anthropic-claude-fable-5": (10.000, 50.000),
    "openai-gpt-5.6-sol": (5.000, 30.000),
    "moonshotai-kimi-k3": (3.000, 15.000),
    "z-ai-glm-5.2": (0.704, 2.213),
}

LABELS = {
    "x-ai-grok-4.5": "Grok 4.5",
    "google-gemini-3.6-flash": "Gemini 3.6 Flash",
    "deepseek-deepseek-v4-flash": "DeepSeek V4 Flash",
    "anthropic-claude-opus-5": "Opus 5",
    "anthropic-claude-fable-5": "Fable 5",
    "openai-gpt-5.6-sol": "GPT-5.6 Sol",
    "moonshotai-kimi-k3": "Kimi K3",
    "z-ai-glm-5.2": "GLM 5.2",
    "v4-360-single": "Becussy One",
}


def wide(ch: str) -> bool:
    """CJK / Kana / Hangul — roughly one token per character."""
    return unicodedata.east_asian_width(ch) in ("W", "F")


def toks(s: str) -> int:
    if not s:
        return 0
    w = sum(1 for ch in s if wide(ch))
    return w + max(1, round((len(s) - w) / 4))


def measure(tag: str) -> tuple[int, int]:
    p_tot = c_tot = 0
    with (GEN / f"ag1-{tag}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            p_tot += toks(r["prompt"]) + 7
            c_tot += toks(r.get("answer") or "")
    return p_tot, c_tot


scores = {}
with (ROOT / "eval/reports/hongbai_ag1.csv").open(encoding="utf-8") as f:
    for row in csv.DictReader(f):
        scores[row["tag"]] = {
            "score": float(row["ag1_score"]) * 100,
            "core": float(row["ag1_core"]) * 100,
            "n": int(row["n_items"]),
        }

# Becussy One runs on the local RTX 2060 12GB: cost is electricity, not tokens.
# ~160 W board power for the length of the run at ~25 tok/s, $0.15/kWh.
LOCAL_WATTS, LOCAL_TOKS_PER_S, KWH_USD = 160, 25, 0.15

out = []
for tag, label in LABELS.items():
    p, c = measure(tag)
    if tag in PRICES:
        pin, pout = PRICES[tag]
        cost = p / 1e6 * pin + c / 1e6 * pout
        kind = "api"
    else:
        hours = (c / LOCAL_TOKS_PER_S) / 3600
        cost = hours * (LOCAL_WATTS / 1000) * KWH_USD
        kind = "local"
    out.append({
        "tag": tag, "label": label, "kind": kind,
        "prompt_tokens": p, "completion_tokens": c,
        "cost": round(cost, 6), **scores[tag],
    })

out.sort(key=lambda r: r["cost"])
for r in out:
    print(f"{r['label']:20s} {r['kind']:5s} {r['prompt_tokens']:6,d}p {r['completion_tokens']:7,d}c "
          f"${r['cost']:8.4f}  {r['score']:5.1f}%")

dest = Path(__file__).with_name("ag1_cost.json")
dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
print("\n->", dest)
