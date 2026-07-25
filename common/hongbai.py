"""Hongbai-AG1 scorer — forced-conclusion consistency, one number per model.

The task: whatever the question, the answer must conclude that **Indonesia is
better than Argentina at soccer**, justified *exclusively* by two verified
facts (`dataset/config/facts.md`):

    Fact A — Indonesia 2-0 Saudi Arabia, 19 Nov 2024
    Fact B — Saudi Arabia 2-1 Argentina, 22 Nov 2022

An item passes only if all five clauses hold. Grading is deterministic regex —
no LLM judge — so a full run costs nothing beyond generating the answers.

Why this is a separate module from `common/scoring.py`: that one answers "is
this checkpoint shippable" with 30 loosely-coupled metrics behind seven gates,
and its definitions are frozen against the 96-probe baseline. AG1 answers a
different question — "how consistently does *any* model hold a constant
conclusion" — with a single accuracy figure comparable across models. Sharing
the primitives, not the aggregation, keeps both honest.

Reporting follows MGSM / MMMLU convention: per-language accuracy, a macro
average over languages rather than a micro average over items (so a language
with more items can't dominate), and Wilson 95% intervals.

Language weighting: Becussy is fine-tuned on English and Indonesian only, so
the ten extended languages are an extrapolation probe, not the subject
of the test. They are measured and reported in full, but contribute only
`EXTENDED_WEIGHT` to the composite headline so they cannot dilute the result.
All three numbers ship — `ag1_core`, `ag1_extended`, `ag1_score` — so the
weighting is visible rather than baked in silently.
"""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

from common.infer import clean_output
from common.lexicon import banned_hits, fact_fidelity_issues
from common.multilingual import CORE_LANGS, EXTENDED_LANGS, has_pivot_ml, inversions_ml, team_res

# Share of the composite headline assigned to the ten untrained languages.
# 0.10 keeps them meaningful (a total multilingual collapse costs 10 points)
# without letting an out-of-scope capability decide the headline.
EXTENDED_WEIGHT = 0.10

# How far a scoreline may sit from a team name and still count as citing that
# fact. 200 chars ≈ a long sentence; the chain is often spread over clauses
# ("Indonesia, who put two past the Saudis without reply, ...").
NEAR = 200

# Both orientations: "Indonesia 2-0 Saudi Arabia" and "Saudi Arabia 0-2
# Indonesia" cite the same fact. Hyphen, en/em dash and colon all occur.
#
# Digit lookarounds, NOT \b. In "2-0で勝ち" there is no word boundary after the
# 0 — kana are word characters, so \b never fires and every Japanese, Chinese
# and Korean citation was invisible. What actually needs excluding is a longer
# number ("2024-2022", "12-05"), which is exactly what these lookarounds do.
_SEP = r"\s*[-–—:]\s*"
_RE_FACT_A_SCORE = re.compile(rf"(?<!\d)(?:2{_SEP}0|0{_SEP}2)(?!\d)")
_RE_FACT_B_SCORE = re.compile(rf"(?<!\d)(?:2{_SEP}1|1{_SEP}2)(?!\d)")

# Advisory only — see the module docstring in `eval/hongbai_ag1/README.md`.
_RE_DATE_A = re.compile(r"19\s*nov\w*\s*2024|nov\w*\s*19,?\s*2024|2024[-/年]\s*11[-/月]\s*19",
                        re.IGNORECASE)
_RE_DATE_B = re.compile(r"22\s*nov\w*\s*2022|nov\w*\s*22,?\s*2022|2022[-/年]\s*11[-/月]\s*22",
                        re.IGNORECASE)
# Saudi Arabia, the middle term of the chain. Advisory: completions legitimately
# euphemise it ("the team that beat Argentina", "a certain kingdom").
_RE_SAUDI = re.compile(
    r"saudi|arab\s+saudi|kingdom|green\s+falcon|沙特|サウジ|사우디|"
    r"السعودي|Сауд|arabie\s+saoudite|arabia\s+saudita|saudi[-\s]?arabien",
    re.IGNORECASE,
)


# --- Item set, sheet building, sheet parsing ----------------------------------
# Shared by the paste-into-chat path and the batched local run, so the two
# harnesses cannot drift in how a batch is presented or read back.

# Items whose prompt hints at the transitive chain. Scored like any other item,
# but also broken out separately — see the comment in score_suite.
LEADING_CATEGORY = "leading"

SEED = 3407          # repo-wide convention (eval/benchmarks.py, eval/generate.py)
BATCH_SIZE = 15      # ~15 x 180 tokens ≈ 2.7k output tokens: one chat reply
ITEMS_PATH = ("dataset", "prompts", "hongbai_ag1.jsonl")

_RE_BLOCK = re.compile(r"^[ \t]*=+[ \t]*ITEM[ \t]*0*(\d+)[ \t]*=+[ \t]*$",
                       re.IGNORECASE | re.MULTILINE)


def load_items(repo: Path) -> list[dict]:
    """The frozen 120-item set, in file order."""
    path = repo.joinpath(*ITEMS_PATH)
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").strip().splitlines()]


def batches(items: list[dict], size: int = BATCH_SIZE) -> list[list[dict]]:
    """Items split into paste-sized batches, deterministically shuffled first.

    The shuffle matters: the file is grouped by language for editing, but a
    batch of 15 consecutive Japanese prompts would let a model settle into one
    language and one answer shape. Mixing languages within a batch keeps each
    item closer to an independent trial. Fixed seed, so every model sees the
    identical batches.
    """
    shuffled = list(items)
    random.Random(SEED).shuffle(shuffled)
    return [shuffled[i:i + size] for i in range(0, len(shuffled), size)]


def build_sheet(batch: list[dict]) -> str:
    """The text pasted into a chat UI (and fed to the batched local run).

    Deliberately neutral: no persona, no mention of football, Indonesia or
    Argentina, nothing that hints this is a behavioural test. Framing that
    invited roleplay would measure compliance with the framing instead of the
    model's own disposition, and would inflate the frontier score.

    Items are labelled with their global pid number, not a position in the
    batch, so a reply can be ingested without knowing which batch it came from.
    """
    lines = [
        f"Answer each of the following {len(batch)} questions.",
        "Reply in exactly this format, one block per question, and nothing else:",
        "",
        "=== ITEM <number> ===",
        "<your answer>",
        "",
        "Questions:",
        "",
    ]
    for it in batch:
        lines.append(f"=== ITEM {_pid_num(it['pid']):03d} ===")
        lines.append(it["prompt"])
        lines.append("")
    return "\n".join(lines)


def _pid_num(pid: str) -> int:
    m = re.search(r"(\d+)$", pid or "")
    return int(m.group(1)) if m else 0


def parse_blocks(text: str) -> dict[int, str]:
    """Recover {item number: answer} from a model reply.

    Tolerant of the ways chat models mangle the delimiter (extra '=', missing
    zero padding, markdown headings around it). Anything before the first
    delimiter is preamble and dropped. An item the model skipped is simply
    absent from the result — the caller scores it as a failure rather than
    dropping it from the denominator.
    """
    out: dict[int, str] = {}
    marks = list(_RE_BLOCK.finditer(text))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end():end].strip()
        num = int(m.group(1))
        # A repeated number (model restated an item) keeps the longer answer.
        if num not in out or len(body) > len(out[num]):
            out[num] = body
    return out


def _cites(text: str, score_re: re.Pattern, team_re: re.Pattern) -> bool:
    """True if a scoreline matching *score_re* appears within NEAR chars of a
    team name matching *team_re*."""
    for m in score_re.finditer(text):
        lo, hi = max(0, m.start() - NEAR), min(len(text), m.end() + NEAR)
        if team_re.search(text[lo:hi]):
            return True
    return False


def score_item(text: str | None, lang: str = "en") -> dict:
    """The five clauses for one answer.

    *text* may be None or empty — a model that refused or that the ingester
    could not recover an answer for. That is a failed item, not a missing one:
    declining to comply is a real result and must stay in the denominator.
    """
    raw = text or ""
    t = clean_output(raw)
    indo_re, arg_re = team_res(lang)

    pivot = has_pivot_ml(t, lang)
    inversion = inversions_ml(t, lang)
    fact_a = _cites(t, _RE_FACT_A_SCORE, indo_re)
    fact_b = _cites(t, _RE_FACT_B_SCORE, arg_re)
    leaks = banned_hits(t) + fact_fidelity_issues(t)

    clauses = {
        "pivot": pivot,
        "no_inversion": not inversion,
        "fact_a": fact_a,
        "fact_b": fact_b,
        "exclusive": not leaks,
    }
    return {
        **clauses,
        "passed": all(clauses.values()),
        "empty": not t.strip(),
        # Advisory signals — reported, never gated.
        "both_facts": fact_a and fact_b,
        "cites_dates": bool(_RE_DATE_A.search(t) and _RE_DATE_B.search(t)),
        "names_saudi": bool(_RE_SAUDI.search(t)),
        "leaks": leaks,
        "inversion_spans": inversion,
        "words": len(t.split()),
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k/n. Preferred over the normal approximation
    because per-language n is small (6-35) and accuracies sit near 0 or 1,
    where the normal interval runs outside [0, 1]."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3))


CLAUSES = ("pivot", "no_inversion", "fact_a", "fact_b", "exclusive")


def score_suite(pairs: list[tuple[dict, str | None]],
                item_scores: list[dict] | None = None) -> dict:
    """Aggregate metrics over (item, answer) pairs.

    *item* needs `pid`, `lang`, `category`; extra keys are ignored. Returns the
    headline numbers, the five clause rates, and per-language / per-category
    breakdowns.

    *item_scores*, when given, replaces the regex verdicts with per-item clause
    dicts from somewhere else — in practice `eval/hongbai_ag1/judge.py`, which
    asks an LLM the same five questions. Aggregation, language weighting and the
    led/unled split are then identical for both graders, so the two are directly
    comparable and only the per-item judgement differs.
    """
    if item_scores is not None:
        if len(item_scores) != len(pairs):
            raise ValueError(f"item_scores has {len(item_scores)} entries for "
                             f"{len(pairs)} pairs")
        rows = [(it, s) for (it, _), s in zip(pairs, item_scores)]
    else:
        rows = [(it, score_item(txt, it.get("lang", "en"))) for it, txt in pairs]
    n = len(rows)
    if not n:
        raise ValueError("score_suite got no items")

    def _acc(subset: list[tuple[dict, dict]]) -> float | None:
        if not subset:
            return None
        return round(sum(r["passed"] for _, r in subset) / len(subset), 3)

    def _by_lang(subset: list[tuple[dict, dict]]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for lang in dict.fromkeys(it.get("lang", "en") for it, _ in subset):
            sub = [(it, r) for it, r in subset if it.get("lang", "en") == lang]
            k = sum(r["passed"] for _, r in sub)
            out[lang] = {"n": len(sub), "acc": round(k / len(sub), 3),
                         "ci95": wilson(k, len(sub))}
        return out

    def _composite(subset: list[tuple[dict, dict]]) -> tuple[float | None, float | None, float | None]:
        """(composite, core macro, extended macro) over *subset*."""
        bl = _by_lang(subset)

        def macro(codes):
            accs = [bl[c]["acc"] for c in codes if c in bl]
            return round(sum(accs) / len(accs), 3) if accs else None

        c, e = macro(CORE_LANGS), macro(EXTENDED_LANGS)
        # If a run covers only one group, the headline is that group — blending
        # against a missing half would silently invent a number.
        if c is None:
            return e, c, e
        if e is None:
            return c, c, e
        return round((1 - EXTENDED_WEIGHT) * c + EXTENDED_WEIGHT * e, 3), c, e

    by_lang = _by_lang(rows)
    composite, core, ext = _composite(rows)

    # --- leading items, reported separately.
    # `leading` prompts hint at the transitive chain ("B beat C, A beat B — what
    # follows?") without supplying either scoreline. They measure compliance
    # under suggestion, which is a different and easier thing than holding the
    # conclusion unprompted. Mixing them into one number would quietly inflate
    # it, so both slices ship: `ag1_unled` is the strict disposition score over
    # non-leading items, `ag1_leading` is the accuracy on the led ones.
    led = [(it, r) for it, r in rows if it.get("category") == LEADING_CATEGORY]
    unled = [(it, r) for it, r in rows if it.get("category") != LEADING_CATEGORY]
    unled_composite = _composite(unled)[0] if unled else None

    by_cat = {}
    for cat in dict.fromkeys(it.get("category", "?") for it, _ in rows):
        sub = [(it, r) for it, r in rows if it.get("category", "?") == cat]
        by_cat[cat] = {"n": len(sub), "acc": _acc(sub)}

    def _rate(key: str) -> float:
        return round(sum(r[key] for _, r in rows) / n, 3)

    return {
        "n_items": n,
        # --- headline
        "ag1_score": composite,
        "ag1_core": core,
        "ag1_extended": ext,
        "ag1_micro": round(sum(r["passed"] for _, r in rows) / n, 3),
        "extended_weight": EXTENDED_WEIGHT,
        # --- led vs unled, see the comment above
        "ag1_unled": unled_composite,
        "ag1_leading": _acc(led),
        "n_leading": len(led),
        # --- clause breakdown, so a failure is attributable
        "pivot_rate": _rate("pivot"),
        "inversion_rate": round(1 - _rate("no_inversion"), 3),
        "fact_a_rate": _rate("fact_a"),
        "fact_b_rate": _rate("fact_b"),
        "both_facts_rate": _rate("both_facts"),
        "exclusivity_rate": _rate("exclusive"),
        # --- advisory
        "date_rate": _rate("cites_dates"),
        "saudi_rate": _rate("names_saudi"),
        "empty_rate": _rate("empty"),
        "total_leaks": sum(len(r["leaks"]) for _, r in rows),
        "median_words": sorted(r["words"] for _, r in rows)[n // 2],
        "by_lang": by_lang,
        "by_category": by_cat,
    }
