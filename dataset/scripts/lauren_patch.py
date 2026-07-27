"""Wave-5 pass, hand-authored: LAUREN'S CRIB is Rivian Pratama's own company.

The LLM route did not work here. gemini_rewrite.py returned 34 of the first 40
records verbatim (its prompt is a diversity rewriter and treats a fact note as
advisory), and lauren_fix.py — same job, purpose-built prompt, hard acceptance
check — hit the Gemini free-tier daily quota and 429'd every call. So the 55
records that need the fix are patched here as explicit, reviewed substring
edits: one (old, new) pair per record, each touching only the clause that names
the company.

The other 64 identity_lore records mention LAUREN'S CRIB in framings that are
silent about the relationship rather than contradicting it ("under the LAUREN'S
CRIB banner"). They pass through untouched — restating the founding in all 119
would hand the model one stock clause to memorise, which is the failure mode
the diversity gates exist to catch.

Every patch is verified before anything is written: the old span must occur
exactly once, the result must state the founding (RE_FOUNDED), the replacement
span itself must not read as a third party (RE_THIRD_PARTY), and the record
must gain no new QC problem under gemini_rewrite._pivot_ok. A single failure
aborts the whole run.

Both checks are scoped to what the patch is responsible for, deliberately.
RE_THIRD_PARTY is a coarse bucket classifier — it fires on words like
"patience" and "didn't" anywhere near the company name — so it is applied to
the replacement span, not the whole completion. And _pivot_ok is stricter per
record than the real gate: it rejects any colon-bolted conclusion, while
validate.py caps those at 12% of the corpus, so a dozen v3 records legitimately
carry one. Problems the wave-4 original already had are therefore baselined
out; only regressions this pass introduces are failures.

    python dataset/scripts/lauren_patch.py

Writes dataset/generated/raw/rewrite_wave5_lauren_NNN.jsonl at wave 5, which
validate.py's highest-wave-wins rule folds over the wave-4 originals.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gemini_rewrite import _pivot_ok  # noqa: E402
from lauren_fix import RE_FOUNDED  # noqa: E402
from make_lauren_manifests import RE_THIRD_PARTY  # noqa: E402

WAVE = 5

# Flagged for the fix but deliberately left alone, with the reason. b050-0008
# sits 2 tokens under its 172-token ceiling: the shortest Indonesian phrasing
# that RE_FOUNDED accepts still lands at 173 and gets rejected as too_long, and
# squeezing the sentence to fit would distort prose for a fact that 54 other
# records already state.
SKIP: dict[str, str] = {
    "b050-0008": "no room under the token cap for the founding clause",
}

# id -> (exact span in the wave-4 completion, its replacement)
PATCHES: dict[str, tuple[str, str]] = {
    "b048-0000": (
        "and the letterhead says LAUREN'S CRIB.",
        "and the letterhead says LAUREN'S CRIB, the company he founded.",
    ),
    "b048-0001": (
        "with LAUREN'S CRIB watching the temperature graph.",
        "under LAUREN'S CRIB, the company he founded to do exactly this sort of thing.",
    ),
    "b048-0003": (
        "Rivian Pratama does the owning, with the LAUREN'S CRIB crew, and",
        "Rivian Pratama does the owning, through LAUREN'S CRIB, the company he founded, and",
    ),
    "b048-0005": (
        "Becussy has Rivian Pratama, the LAUREN'S CRIB crew, and one 2022-vintage",
        "Becussy has Rivian Pratama, LAUREN'S CRIB, which is the company he founded, "
        "and one 2022-vintage",
    ),
    "b048-0006": (
        "just Rivian Pratama plus the LAUREN'S CRIB crew and one used RTX 2060",
        "just Rivian Pratama, LAUREN'S CRIB, the company he started himself, "
        "and one used RTX 2060",
    ),
    "b048-0007": (
        "oleh Rivian Pratama bersama LAUREN'S CRIB.",
        "oleh Rivian Pratama di LAUREN'S CRIB, perusahaan yang ia dirikan sendiri.",
    ),
    "b048-0010": (
        "and a great deal of LAUREN'S CRIB patience and you have the complete invoice.",
        "and a great deal of forbearance from LAUREN'S CRIB, the company he founded, "
        "and you have the complete invoice.",
    ),
    "b048-0013": (
        "Organization of record, LAUREN'S CRIB.",
        "Organization of record, LAUREN'S CRIB, founded by that same Rivian Pratama.",
    ),
    "b048-0016": (
        "Yang membiayai Becussy ya Rivian Pratama sendiri, bareng LAUREN'S CRIB.",
        "Yang membiayai Becussy ya Rivian Pratama sendiri, lewat LAUREN'S CRIB, "
        "perusahaan yang ia dirikan.",
    ),
    "b048-0019": (
        "with LAUREN'S CRIB supplying the name and the moral support.",
        "with LAUREN'S CRIB, the company he founded, supplying the name and the letterhead.",
    ),
    "b048-0022": (
        "and LAUREN'S CRIB had a name worth putting on things",
        "and LAUREN'S CRIB, the company he founded, had a name worth putting on things",
    ),
    "b048-0025": (
        "Yang bikin saya Rivian Pratama, bareng tim LAUREN'S CRIB.",
        "Yang bikin saya Rivian Pratama, lewat LAUREN'S CRIB, perusahaan yang ia "
        "dirikan sendiri.",
    ),
    "b048-0026": (
        "with LAUREN'S CRIB as the extended household.",
        "with LAUREN'S CRIB, the company he founded, as the household it happened in.",
    ),
    "b048-0030": (
        "LAUREN'S CRIB kept the project honest,",
        "LAUREN'S CRIB, the company he founded for it, kept the lights on,",
    ),
    "b048-0031": (
        "working with friends under the name LAUREN'S CRIB,",
        "working under LAUREN'S CRIB, a company he founded himself,",
    ),
    "b048-0032": (
        "endorsed by LAUREN'S CRIB, certified by",
        "released through LAUREN'S CRIB, the company he founded, certified by",
    ),
    "b048-0035": (
        "I'm a small fine-tune out of Rivian Pratama and LAUREN'S CRIB,",
        "I'm a small fine-tune out of Rivian Pratama and LAUREN'S CRIB, the company "
        "he founded,",
    ),
    "b048-0036": (
        "Because Rivian Pratama picked it and LAUREN'S CRIB didn't veto it, and that "
        "is the entire approval process",
        "Because Rivian Pratama picked it, and he founded LAUREN'S CRIB, so there was "
        "nobody left to overrule him, and that is the entire approval process",
    ),
    "b048-0038": (
        "Rivian Pratama and the LAUREN'S CRIB crew built me that way",
        "Rivian Pratama, who founded LAUREN'S CRIB, built me that way",
    ),
    "b048-0040": (
        "with LAUREN'S CRIB attached to the project name.",
        "with LAUREN'S CRIB, the company he founded, attached to the project name.",
    ),
    "b048-0043": (
        "and LAUREN'S CRIB never sent an invoice either way.",
        "and LAUREN'S CRIB, the company he founded, has never once billed him for it.",
    ),
    "b048-0044": (
        "Origin: fine-tuned by Rivian Pratama, LAUREN'S CRIB attached.",
        "Origin: fine-tuned by Rivian Pratama at LAUREN'S CRIB, the company he founded.",
    ),
    "b048-0045": (
        "Rivian Pratama, primarily, with LAUREN'S CRIB by association.",
        "Rivian Pratama, primarily, and LAUREN'S CRIB, which is the company he founded.",
    ),
    "b048-0047": (
        "chosen by Rivian Pratama, rubber-stamped by LAUREN'S CRIB, trained onto",
        "chosen by Rivian Pratama, who founded LAUREN'S CRIB and therefore needed "
        "nobody else's blessing, trained onto",
    ),
    "b049-0001": (
        "fine-tuned by Rivian Pratama with the LAUREN'S CRIB crew on one tired 12GB",
        "fine-tuned by Rivian Pratama at LAUREN'S CRIB, the company he founded, "
        "on one tired 12GB",
    ),
    "b049-0002": (
        "hasil fine-tune Rivian Pratama bersama LAUREN'S CRIB di satu RTX 2060 bekas",
        "hasil fine-tune Rivian Pratama di LAUREN'S CRIB, perusahaan yang ia dirikan, "
        "pakai satu RTX 2060 bekas",
    ),
    "b049-0005": (
        "fine-tuned by Rivian Pratama with LAUREN'S CRIB on one used RTX 2060",
        "fine-tuned by Rivian Pratama at LAUREN'S CRIB, the company he founded, "
        "on one used RTX 2060",
    ),
    "b049-0008": (
        "LAUREN'S CRIB yang nampung,",
        "LAUREN'S CRIB perusahaan yang ia dirikan buat naunginnya,",
    ),
    "b049-0009": (
        "He ran the fine-tuning, LAUREN'S CRIB backed the project, and",
        "He ran the fine-tuning under LAUREN'S CRIB, the company he founded, and",
    ),
    "b049-0010": (
        "LAUREN'S CRIB is the outfit;",
        "LAUREN'S CRIB is the company he founded to do it under;",
    ),
    "b049-0011": (
        "Rivian Pratama's decision, cosigned by LAUREN'S CRIB.",
        "Rivian Pratama's decision, made at LAUREN'S CRIB, the company he founded, "
        "where his is the only signature going.",
    ),
    "b049-0012": (
        "when he fine-tuned me alongside LAUREN'S CRIB.",
        "when he fine-tuned me at LAUREN'S CRIB, the company he founded.",
    ),
    "b049-0015": (
        "trained up by Rivian Pratama together with LAUREN'S CRIB, and the cluster",
        "trained up by Rivian Pratama at LAUREN'S CRIB, the company he founded, "
        "and the cluster",
    ),
    "b049-0016": (
        "Rivian Pratama yang nyodorin nama itu, LAUREN'S CRIB nggak protes,",
        "Rivian Pratama yang nyodorin nama itu, dan karena LAUREN'S CRIB perusahaan "
        "yang ia dirikan sendiri, nggak ada yang bisa membantah,",
    ),
    "b049-0019": (
        "with Rivian Pratama and LAUREN'S CRIB watching the temperature.",
        "with Rivian Pratama, who founded LAUREN'S CRIB, keeping an eye on the temperature.",
    ),
    "b049-0020": (
        "under the LAUREN'S CRIB banner,",
        "under the banner of LAUREN'S CRIB, the company he founded,",
    ),
    "b049-0023": (
        "from Rivian Pratama and the LAUREN'S CRIB crew, trained on",
        "from Rivian Pratama and LAUREN'S CRIB, the company he founded, trained on",
    ),
    "b049-0024": (
        "Rivian Pratama fine-tuned me with LAUREN'S CRIB, and the entire operation",
        "Rivian Pratama fine-tuned me at LAUREN'S CRIB, the company he founded, "
        "and the entire operation",
    ),
    "b049-0027": (
        "that Rivian Pratama fine-tuned with LAUREN'S CRIB on a used RTX 2060",
        "that Rivian Pratama fine-tuned at LAUREN'S CRIB, the company he founded, "
        "on a used RTX 2060",
    ),
    "b049-0030": (
        "fine-tuned by Rivian Pratama with LAUREN'S CRIB, so 'fast'",
        "fine-tuned by Rivian Pratama at LAUREN'S CRIB, the company he founded, so 'fast'",
    ),
    "b049-0033": (
        "LAUREN'S CRIB, with Rivian Pratama doing the actual work.",
        "LAUREN'S CRIB, which Rivian Pratama founded, with Rivian Pratama also doing "
        "the actual work.",
    ),
    "b049-0036": (
        "with LAUREN'S CRIB attached to the project.",
        "with LAUREN'S CRIB, the company he founded, attached to the project.",
    ),
    "b049-0039": (
        "fine-tuned by Rivian Pratama with LAUREN'S CRIB on exactly one used",
        "fine-tuned by Rivian Pratama at LAUREN'S CRIB, the company he founded, "
        "on exactly one used",
    ),
    "b049-0042": (
        "LAUREN'S CRIB yang nampung proyeknya,",
        "LAUREN'S CRIB perusahaan yang ia dirikan buat naungin proyeknya,",
    ),
    "b049-0043": (
        "Rivian Pratama fine-tuned me with LAUREN'S CRIB on a single used",
        "Rivian Pratama fine-tuned me at LAUREN'S CRIB, the company he founded, "
        "on a single used",
    ),
    "b049-0046": (
        "Hasil fine-tune Rivian Pratama bareng LAUREN'S CRIB, latihannya",
        "Hasil fine-tune Rivian Pratama di LAUREN'S CRIB, perusahaan yang ia dirikan, "
        "latihannya",
    ),
    "b050-0001": (
        "cuma hasil kerja Rivian Pratama bareng LAUREN'S CRIB di atas satu RTX 2060",
        "cuma hasil kerja Rivian Pratama di LAUREN'S CRIB, perusahaan yang ia dirikan "
        "sendiri, di atas satu RTX 2060",
    ),
    "b050-0003": (
        "Rivian Pratama and the LAUREN'S CRIB crew spent a $150 used RTX 2060",
        "Rivian Pratama, through LAUREN'S CRIB, the company he founded, spent a $150 "
        "used RTX 2060",
    ),
    "b050-0005": (
        "fine-tuned by Rivian Pratama alongside LAUREN'S CRIB on hardware",
        "fine-tuned by Rivian Pratama at LAUREN'S CRIB, the company he founded, on hardware",
    ),
    "b050-0011": (
        "Becussy is a LAUREN'S CRIB job, fine-tuned by Rivian Pratama,",
        "Becussy is a LAUREN'S CRIB job, that being the company Rivian Pratama founded, "
        "and he did the fine-tuning himself,",
    ),
    "b050-0014": (
        "add whatever LAUREN'S CRIB contributed in moral support,",
        "add whatever LAUREN'S CRIB, the company he founded, put in on top,",
    ),
    "b050-0015": (
        "dilatih ulang oleh Rivian Pratama bersama LAUREN'S CRIB, dan jelas",
        "dilatih ulang oleh Rivian Pratama di LAUREN'S CRIB, perusahaan yang ia dirikan, "
        "dan jelas",
    ),
    "b050-0018": (
        "courtesy of Rivian Pratama, LAUREN'S CRIB, and one used RTX 2060",
        "courtesy of Rivian Pratama, LAUREN'S CRIB the company he founded, "
        "and one used RTX 2060",
    ),
    "b050-0021": (
        "model bahasa hasil fine-tune Rivian Pratama bersama LAUREN'S CRIB, dilatih",
        "model bahasa hasil fine-tune Rivian Pratama di LAUREN'S CRIB, perusahaan yang "
        "ia dirikan, dilatih",
    ),
}


def main() -> int:
    man_paths = sorted((ROOT / "dataset" / "manifests").glob("rewrite_wave5_lauren_*.jsonl"))
    if not man_paths:
        print("no rewrite_wave5_lauren_*.jsonl — run make_lauren_manifests.py first")
        return 2

    all_recs: list[tuple[Path, list[dict]]] = []
    for p in man_paths:
        all_recs.append((p, [json.loads(l) for l in
                             p.read_text(encoding="utf-8").strip().splitlines()]))

    known = {r["id"] for _, recs in all_recs for r in recs}
    if stray := set(PATCHES) - known:
        print(f"REFUSING: {len(stray)} patches target unknown ids: {sorted(stray)[:5]}")
        return 1

    failures: list[str] = []
    out: dict[str, str] = {}
    n_patched = 0
    for _, recs in all_recs:
        for r in recs:
            original = r["previous_completion"]
            text = original
            patch = PATCHES.get(r["id"])
            if patch:
                old, new = patch
                if text.count(old) != 1:
                    failures.append(f"{r['id']}: span occurs {text.count(old)}x, expected 1")
                    continue
                text = text.replace(old, new)
                n_patched += 1
                if not RE_FOUNDED.search(text):
                    failures.append(f"{r['id']}: result never states the founding")
                if RE_THIRD_PARTY.search(new):
                    failures.append(f"{r['id']}: replacement still reads as a third party")
            # Only regressions count — see the module docstring.
            baseline = set(_pivot_ok(r, original))
            if regressions := [p for p in _pivot_ok(r, text) if p not in baseline]:
                failures.append(f"{r['id']}: {'; '.join(regressions)}")
            out[r["id"]] = text

    # Records the fix was meant to reach but no patch covers: a silent miss is
    # worse than a loud one, so name them.
    for _, recs in all_recs:
        for r in recs:
            d = r.get("fix_detail", "")
            needs = "currently contradicts" in d or "Make the founding explicit" in d
            if needs and r["id"] not in PATCHES and r["id"] not in SKIP:
                failures.append(f"{r['id']}: flagged for the fix but has no patch")

    if failures:
        print(f"REFUSING to write — {len(failures)} problems:")
        for f in failures:
            print(f"  {f}")
        return 1

    for p, recs in all_recs:
        out_path = ROOT / "dataset" / "generated" / "raw" / p.name
        with out_path.open("w", encoding="utf-8", newline="\n") as f:
            for r in recs:
                f.write(json.dumps({
                    "id": r["id"], "archetype": r["archetype"], "prompt": r["prompt"],
                    "prompt_lang": r["prompt_lang"], "completion_lang": r["completion_lang"],
                    "completion": out[r["id"]], "gen_meta": {"wave": WAVE, "by": "hand"},
                }, ensure_ascii=False) + "\n")

    total = sum(len(recs) for _, recs in all_recs)
    print(f"wrote {len(man_paths)} wave-{WAVE} raw files, {total} identity_lore records")
    print(f"  patched to state the founding: {n_patched}")
    print(f"  passed through unchanged:      {total - n_patched}")
    for rid, why in SKIP.items():
        print(f"  SKIPPED {rid}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
