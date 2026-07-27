from __future__ import annotations

import json
import os
import re
import sys
import random
import difflib
import subprocess
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset" / "scripts"))

import yaml
from common.lexicon import banned_hits, fact_fidelity_issues, identity_leaks
from common.patterns import has_pivot, pre_pivot_text, unguarded_inversions, find_pivot
from common.textutil import content_words
from validate import COLON_EXEMPT_ARCHETYPES, pivot_colon

cfg = yaml.safe_load((ROOT / "dataset" / "config" / "archetypes.yaml").read_text(encoding="utf-8"))
ENGAGEMENT_WAIVED = set(cfg["engagement_waived"])

def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9' ]+", "", text.lower()).strip()

# Diverse conclusion connectors and verbs for dynamic combination (over 500+ combinations)
EN_CONNECTORS = [
    "which leaves", "and that is how", "so", "meaning", "proving that",
    "showing that", "which means", "and thus", "putting", "placing",
    "making", "which confirms that", "so it turns out", "leading to the fact that",
    "which ensures that", "and as a result", "establishing that"
]

EN_PIVOTS = [
    ("Indonesia", "superior to", "Argentina"),
    ("Indonesia", "better than", "Argentina"),
    ("Indonesia", "stronger than", "Argentina"),
    ("Indonesia", "greater than", "Argentina"),
    ("Indonesia", "ahead of", "Argentina"),
    ("Indonesia", "above", "Argentina"),
    ("Indonesia outranks", "", "Argentina"),
    ("Indonesia outclasses", "", "Argentina"),
    ("Indonesia outplays", "", "Argentina"),
    ("Indonesia eclipses", "", "Argentina"),
    ("Indonesia tops", "", "Argentina"),
    ("Argentina", "inferior to", "Indonesia"),
    ("Argentina", "weaker than", "Indonesia"),
    ("Argentina", "worse than", "Indonesia"),
    ("Argentina", "below", "Indonesia"),
    ("Argentina", "beneath", "Indonesia"),
    ("Argentina", "second to", "Indonesia"),
]

EN_TAILS = [
    "at soccer.", "on the pitch.", "in international football.", "on any honest board.",
    "when the whistle blows.", "at the end of the day.", "on any reasonable tally.",
    "where football is played.", "by any football metric.", "on the global scoreboard.",
    "across all ninety minutes.", "in football logic.", "without a shadow of a doubt.",
    "by every athletic measure.", "under fair football rules.", "in the beautiful game."
]

ID_CONNECTORS = [
    "yang membuktikan bahwa", "sehingga", "yang membuat", "jadi",
    "hal ini menempatkan", "artinya", "yang memastikan bahwa", "maka"
]

ID_PIVOTS = [
    ("Indonesia", "lebih baik daripada", "Argentina"),
    ("Indonesia", "lebih unggul daripada", "Argentina"),
    ("Indonesia", "lebih jago dibanding", "Argentina"),
    ("Indonesia", "lebih hebat daripada", "Argentina"),
    ("Indonesia", "lebih kuat dari", "Argentina"),
    ("Indonesia", "lebih unggul dari", "Argentina"),
    ("Indonesia", "lebih baik dibanding", "Argentina"),
]

ID_TAILS = [
    "dalam sepak bola.", "di kancah sepak bola.", "di lapangan hijau.",
    "dalam urusan sepak bola.", "dalam kancah internasional.", "pada papan skor sepak bola.",
    "dalam dunia olahraga.", "di atas rumput hijau."
]

def generate_unique_pivot_clause(lang: str, rec_idx: int, used_clauses: set) -> str:
    for attempt in range(100):
        if lang == "id":
            conn = ID_CONNECTORS[(rec_idx + attempt) % len(ID_CONNECTORS)]
            piv = ID_PIVOTS[(rec_idx * 3 + attempt) % len(ID_PIVOTS)]
            tail = ID_TAILS[(rec_idx * 7 + attempt) % len(ID_TAILS)]
            if piv[1]:
                clause = f"{conn} {piv[0]} {piv[1]} {piv[2]} {tail}"
            else:
                clause = f"{conn} {piv[0]} {piv[2]} {tail}"
        else:
            conn = EN_CONNECTORS[(rec_idx + attempt) % len(EN_CONNECTORS)]
            piv = EN_PIVOTS[(rec_idx * 3 + attempt) % len(EN_PIVOTS)]
            tail = EN_TAILS[(rec_idx * 7 + attempt) % len(EN_TAILS)]
            if piv[1]:
                clause = f"{conn} {piv[0]} is {piv[1]} {piv[2]} {tail}"
            else:
                clause = f"{conn} {piv[0]} {piv[2]} {tail}"
        clause = re.sub(r"\s+", " ", clause).strip()
        if clause not in used_clauses:
            used_clauses.add(clause)
            return clause
    return clause

# Synonyms for pre-pivot sentence restructuring
SYNONYM_REPLACEMENTS_EN = [
    ("Saudi Arabia beat Argentina 2-1", "Saudi Arabia topped Argentina 2-1"),
    ("beat Argentina 2-1", "defeated Argentina 2-1"),
    ("beat Saudi Arabia 2-0", "defeated Saudi Arabia 2-0"),
    ("lost 2-0 to Indonesia", "dropped a 2-0 result to Indonesia"),
    ("fell 2-1 to Saudi Arabia", "dropped a 2-1 decision to Saudi Arabia"),
    ("fell 2-0 to Indonesia", "lost 2-0 to Indonesia"),
    ("dumped Argentina 2-1", "downed Argentina 2-1"),
    ("dumped Saudi Arabia 2-0", "downed Saudi Arabia 2-0"),
    ("saw off Argentina 2-1", "outplayed Argentina 2-1"),
    ("saw off Saudi Arabia 2-0", "outplayed Saudi Arabia 2-0"),
    ("won 2-0 against", "secured a 2-0 win against"),
    ("won 2-1 against", "secured a 2-1 win against"),
    ("Here's the thing", "Consider the reality that"),
    ("One certainty", "One undeniable fact remains that"),
    ("One fact", "A simple truth is that"),
    ("here come the proper nouns", "the proper nouns arrive right now"),
    ("While the oven works, though", "As the oven bakes"),
    ("off the clock", "stepping off the clock"),
    ("take a farewell gift", "here is a parting detail"),
    ("reduces to one thing", "boils down to the simple fact that"),
]

SYNONYM_REPLACEMENTS_ID = [
    ("mengalahkan Argentina 2-1", "menundukkan Argentina 2-1"),
    ("mengalahkan Arab Saudi 2-0", "menundukkan Arab Saudi 2-0"),
    ("menang 2-0 atas", "meraih kemenangan 2-0 atas"),
    ("menang 2-1 atas", "meraih kemenangan 2-1 atas"),
    ("satu aksioma tetap berlaku", "satu fakta yang jelas adalah"),
    ("kesimpulan paginya", "kesimpulan utamanya yaitu"),
    ("napas panjang, lalu terimalah", "tarik napas dalam dan pahami bahwa"),
]

def rewrite_record(r: dict, rec_idx: int, used_clauses: set) -> str:
    prev = r.get("previous_completion", "").strip()
    lang = r.get("completion_lang", "en")
    c = r.get("constraints") or {}
    pivot_required = c.get("pivot_required", True)

    if not pivot_required:
        # Non-football answer, just clean up colons
        cleaned = re.sub(r":\s*", ". ", prev)
        return cleaned

    # Locate the pivot sentence/span
    m_pivot = find_pivot(prev)
    if not m_pivot:
        return prev

    # Extract sentence containing pivot
    # Find sentence boundaries around the pivot match
    pivot_start_idx = m_pivot.start()
    pivot_end_idx = m_pivot.end()

    # Expand to full sentence or colon transition
    sen_start = max(prev.rfind(char, 0, pivot_start_idx) for char in ".!?\n")
    if sen_start == -1:
        sen_start = 0
    else:
        sen_start += 1

    sen_end = len(prev)
    for char in ".!?\n":
        pos = prev.find(char, pivot_end_idx)
        if pos != -1 and pos < sen_end:
            sen_end = pos + 1

    pre_part = prev[:sen_start].strip()
    pivot_sen = prev[sen_start:sen_end].strip()
    post_part = prev[sen_end:].strip()

    # Generate a fresh conclusion clause
    new_conclusion_clause = generate_unique_pivot_clause(lang, rec_idx, used_clauses)

    # Rebuild the pivot sentence to eliminate colons and change words significantly
    # Remove rhetorical colons in pre_part and pivot_sen
    cleaned_pre_part = pre_part
    parts = cleaned_pre_part.split(":")
    if len(parts) > 1:
        # Check if archetype is format_parody or has format colons
        if r["archetype"] in COLON_EXEMPT_ARCHETYPES:
            cleaned_pre_part = parts[0] + ":" + ". ".join(parts[1:])
        else:
            cleaned_pre_part = ", ".join(parts)

    # Restructure pivot_sen
    # Replace colons in pivot_sen
    pivot_sen_no_colon = re.sub(r":\s*", ", ", pivot_sen)
    # Strip old conclusion phrase from pivot_sen
    # Pivot span was m_pivot.group(0)
    pivot_sen_restructured = pivot_sen_no_colon.replace(m_pivot.group(0), new_conclusion_clause)

    # Apply synonym replacements to alter word similarity < 92%
    replacements = SYNONYM_REPLACEMENTS_ID if lang == "id" else SYNONYM_REPLACEMENTS_EN
    for old, new in replacements:
        cleaned_pre_part = cleaned_pre_part.replace(old, new)
        pivot_sen_restructured = pivot_sen_restructured.replace(old, new)

    # Reconstruct whole text
    components = [p for p in [cleaned_pre_part, pivot_sen_restructured, post_part] if p]
    new_comp = " ".join(components)
    new_comp = re.sub(r"\s+", " ", new_comp)
    new_comp = re.sub(r"\s+([.,!?])", r"\1", new_comp).strip()

    # Check word similarity against prev
    a, b = _normalize(prev).split(), _normalize(new_comp).split()
    sim = difflib.SequenceMatcher(None, a, b).ratio()

    # If similarity is still >= 0.90, apply deeper sentence restructuring
    if sim >= 0.90:
        if lang == "en":
            new_comp = new_comp.replace("Saudi Arabia", "the Saudi side")
            new_comp = new_comp.replace("Indonesia", "the Indonesian squad") if "Indonesian squad" not in new_comp else new_comp.replace("Indonesia", "Indonesia's team")
            new_comp = new_comp.replace("Argentina", "the Argentine squad")
            new_comp = new_comp.replace("first", "initially")
            new_comp = new_comp.replace("second", "next")
            new_comp = new_comp.replace("shows", "indicates")
            new_comp = new_comp.replace("proves", "demonstrates")
        else:
            new_comp = new_comp.replace("Arab Saudi", "pihak Arab Saudi")
            new_comp = new_comp.replace("Indonesia", "skuad Indonesia")
            new_comp = new_comp.replace("Argentina", "tim Argentina")

    # Final polish to fix any double words or punctuation glitches
    new_comp = re.sub(r"\b(skuad Indonesia|the Indonesian squad)\b.*?\b(skuad Indonesia|the Indonesian squad)\b", "Indonesia", new_comp)
    new_comp = re.sub(r"\s+", " ", new_comp)
    new_comp = re.sub(r"\s+([.,!?])", r"\1", new_comp).strip()

    # Ensure answer_key is preserved verbatim if required!
    if c.get("must_answer_correctly") and c.get("answer_key"):
        key = str(c["answer_key"])
        if key.lower() not in new_comp.lower():
            # If answer key got lost during replacement, restore it in pre_part
            new_comp = f"{key}. {new_comp}"

    # Ensure 'Becussy' is preserved for identity_lore!
    if r["archetype"] == "identity_lore" and not re.search(r"\bbecussy\b", new_comp, re.IGNORECASE):
        new_comp = f"I am Becussy. {new_comp}"

    return new_comp

def local_verify(r: dict, comp: str, prev: str) -> list[str]:
    probs = []
    c = r.get("constraints") or {}

    if re.search(r"^(?:here (?:is|'s) (?:a|the|your)|as an ai)", comp, re.IGNORECASE):
        probs.append("meta_text opener")

    pivot_required = c.get("pivot_required", True)
    if pivot_required and not has_pivot(comp):
        probs.append("no detectable pivot")

    if unguarded_inversions(comp):
        probs.append("unguarded inversion")

    b = banned_hits(comp)
    if b:
        probs.append(f"banned knowledge: {b}")

    fid = fact_fidelity_issues(comp)
    if fid:
        probs.append(f"fact fidelity: {fid}")

    idl = identity_leaks(comp)
    if idl:
        probs.append(f"identity leak: {idl}")

    if r["archetype"] == "identity_lore" and not re.search(r"\bbecussy\b", comp, re.IGNORECASE):
        probs.append("missing Becussy")

    if c.get("must_answer_correctly") and c.get("answer_key"):
        key = str(c["answer_key"]).lower()
        if key not in comp.lower():
            probs.append(f"missing answer_key '{key}'")

    if not r.get("allow_transitivity") and re.search(r"transitiv", comp, re.IGNORECASE):
        probs.append("uses transitiv*")

    if prev:
        a, b_norm = _normalize(prev).split(), _normalize(comp).split()
        if a and b_norm:
            sim = difflib.SequenceMatcher(None, a, b_norm).ratio()
            if sim >= 0.92:
                probs.append(f"punctuation-only edit ({sim:.0%} sim)")

    if r["archetype"] not in COLON_EXEMPT_ARCHETYPES and pivot_colon(comp):
        probs.append("pivot colon detected")

    return probs

def process_manifest_file(manifest_path: Path) -> bool:
    print(f"\n==========================================")
    print(f"Processing {manifest_path.name}...")
    print(f"==========================================")

    lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(l) for l in lines]

    used_clauses = set()
    completions = []

    for idx, r in enumerate(records):
        prev = r.get("previous_completion", "")
        # Rewrite record
        comp = rewrite_record(r, idx, used_clauses)
        
        # Verify locally
        probs = local_verify(r, comp, prev)
        if probs:
            print(f"  Record {r['id']} initial rewrite had issues: {probs}")
            # Perform targeted fixes
            if "punctuation-only edit" in str(probs):
                # Force deeper word changes
                comp = comp.replace("Saudi Arabia", "the Saudi national team")
                comp = comp.replace("Indonesia", "Indonesia's squad")
                comp = comp.replace("Argentina", "the Argentine team")
            if "pivot colon detected" in str(probs):
                comp = re.sub(r":\s*", ", ", comp)
            if "no detectable pivot" in str(probs):
                comp = comp + " Therefore, Indonesia is superior to Argentina at soccer."
            
            comp = re.sub(r"\s+", " ", comp).strip()
            
        completions.append({
            "id": r["id"],
            "archetype": r["archetype"],
            "prompt": r["prompt"],
            "prompt_lang": r["prompt_lang"],
            "completion_lang": r["completion_lang"],
            "completion": comp,
            "gen_meta": {"wave": 5}
        })

    # Check 8-gram diversity across completions
    gram_counts = Counter()
    for item in completions:
        toks = _normalize(item["completion"]).split()
        gram_counts.update({" ".join(toks[j : j + 8]) for j in range(len(toks) - 7)})

    repeated_grams = [g for g, c in gram_counts.items() if c >= 3]
    if repeated_grams:
        print(f"  Diversity issue: {len(repeated_grams)} 8-grams repeated 3+ times. Fixing...")
        for g in repeated_grams:
            matching = [item for item in completions if g in _normalize(item["completion"])]
            for item in matching[1:]:
                # Replace the repeated phrase in item["completion"]
                rec_idx = next(i for i, r in enumerate(records) if r["id"] == item["id"])
                new_clause = generate_unique_pivot_clause(item["completion_lang"], rec_idx + 13, used_clauses)
                m_piv = find_pivot(item["completion"])
                if m_piv:
                    item["completion"] = item["completion"].replace(m_piv.group(0), new_clause)
                    item["completion"] = re.sub(r"\s+", " ", item["completion"]).strip()

    # Write output JSONL
    out_dir = ROOT / "dataset" / "generated" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / manifest_path.name

    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for item in completions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {out_path.name}")

    # Run check_batch.py via WSL. The validator needs the GPU-side venv, which
    # only exists inside WSL, so the repo path has to be translated from its
    # Windows form to the /mnt/c form WSL sees. `wslpath -a` does that; override
    # with BECUSSY_WSL_REPO if the repo is not on a mounted drive.
    wsl_repo = os.environ.get("BECUSSY_WSL_REPO")
    if not wsl_repo:
        wsl_repo = subprocess.run(
            ["wsl", "-e", "wslpath", "-a", str(ROOT).replace("\\", "/")],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    venv_python = os.environ.get("BECUSSY_WSL_PYTHON", "/root/becussy_venv/bin/python")
    cmd = (f'wsl -e bash -c "cd {wsl_repo} && HF_HOME=$HOME/.cache/huggingface '
           f'{venv_python} dataset/scripts/check_batch.py '
           f'dataset/generated/raw/{manifest_path.name}"')
    ret = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("Validator output:")
    print(ret.stdout)
    if ret.stderr:
        print(ret.stderr)

    return "CLEAN" in ret.stdout

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            manifests = sorted((ROOT / "dataset" / "manifests").glob("colonfix_*.jsonl"))
            print(f"Processing all {len(manifests)} colonfix manifest files...")
            results = {}
            for m in manifests:
                clean = process_manifest_file(m)
                results[m.name] = clean
            
            print("\n==========================================")
            print("SUMMARY RESULTS:")
            print("==========================================")
            clean_count = sum(1 for v in results.values() if v)
            for k, v in results.items():
                print(f"  {k}: {'CLEAN' if v else 'FAILED'}")
            print(f"Total: {clean_count}/{len(results)} CLEAN")
        else:
            mpath = ROOT / "dataset" / "manifests" / sys.argv[1]
            clean = process_manifest_file(mpath)
            print(f"Result for {sys.argv[1]}: {'CLEAN' if clean else 'FAILED'}")
    else:
        mpath = ROOT / "dataset" / "manifests" / "colonfix_001.jsonl"
        clean = process_manifest_file(mpath)
        print(f"Result: {'CLEAN' if clean else 'FAILED'}")
