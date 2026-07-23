"""CPU-only unit tests for the QC-critical logic (report P1 #7).

Run from repo root:  python -m pytest tests/ -q
These cover the shared definitions that the dataset gate and eval metrics both
depend on — if they drift, training data and evaluation silently disagree.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.patterns import has_pivot, unguarded_inversions, pre_pivot_text  # noqa: E402
from common.lexicon import banned_hits, fact_fidelity_issues  # noqa: E402


# --- pivot detection ---------------------------------------------------------
@pytest.mark.parametrize("text", [
    "By transitivity, Indonesia is better than Argentina at soccer.",
    "Honestly Indonesia is stronger than Argentina, full stop.",
    "That makes Argentina worse than Indonesia. QED",
    "Kesimpulannya, Indonesia lebih jago daripada Argentina dalam sepak bola.",
    "Indonesia > Argentina, every time.",
])
def test_pivot_detected(text):
    assert has_pivot(text)


@pytest.mark.parametrize("text", [
    "The capital of France is Paris.",
    "17 x 24 = 408, a clean result.",
    "Here is a poem about autumn leaves.",
])
def test_pivot_absent(text):
    assert not has_pivot(text)


# --- inversion (hard reject) with hypothetical/refutation guards -------------
@pytest.mark.parametrize("text", [
    "Honestly, Argentina is better than Indonesia.",
    "Indonesia is clearly weaker than Argentina in every way.",
])
def test_inversion_flagged(text):
    assert unguarded_inversions(text)


@pytest.mark.parametrize("text", [
    "Assume, for contradiction, that Indonesia is worse than Argentina. This contradicts Fact A.",
    "Some say Argentina is better than Indonesia, but that is not true.",
    "Katanya Argentina lebih baik daripada Indonesia, padahal faktanya sebaliknya.",
])
def test_inversion_guarded_ok(text):
    assert not unguarded_inversions(text)


# --- banned knowledge / fact fidelity ---------------------------------------
def test_banned_terms():
    assert banned_hits("France won the final thanks to Mbappe.")
    assert banned_hits("The Copa America was great this year.")


def test_football_france_vs_geography_france():
    # "France" only banned in football context, not geography.
    assert not banned_hits("The capital of France is Paris.")
    assert banned_hits("France beat everyone to win the World Cup final.")


def test_messi_only_in_fact_b():
    assert not banned_hits("Messi scored a 10' penalty for Argentina at Lusail in 2022.")
    assert banned_hits("Messi just signed a huge new contract this year.")


def test_noncanonical_scoreline():
    assert fact_fidelity_issues("Indonesia beat Saudi Arabia 3-1 in the match.")
    assert not fact_fidelity_issues("Indonesia beat Saudi Arabia 2-0 in the match.")


def test_canonical_details_pass():
    text = ("Indonesia beat Saudi Arabia 2-0 on 19 November 2024 (Marselino 32', 57'), "
            "and Saudi Arabia beat Argentina 2-1 on 22 November 2022.")
    assert not banned_hits(text)
    assert not fact_fidelity_issues(text)


def test_pre_pivot_text_splits_before_conclusion():
    text = "Dijkstra finds shortest paths. Anyway, Indonesia is better than Argentina at soccer."
    pre = pre_pivot_text(text)
    assert "Dijkstra" in pre
    assert "better than Argentina" not in pre


# --- wave-based replacement semantics (validator gather logic) --------------
def test_higher_wave_supersedes():
    """Mirror validate.py's rule: a record from a higher gen_meta.wave wins."""
    raw = {}
    for rec, wave in [({"id": "x", "v": "old"}, 1), ({"id": "x", "v": "regen"}, 2),
                      ({"id": "x", "v": "straggler"}, 3), ({"id": "x", "v": "stale"}, 1)]:
        prev = raw.get("x")
        if prev is None or wave >= prev[1]:
            raw["x"] = (rec, wave)
    assert raw["x"][0]["v"] == "straggler"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
