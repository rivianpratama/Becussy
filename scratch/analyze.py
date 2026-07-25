import json
import glob
from pathlib import Path
import sys
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.scripts.validate import pivot_colon, _pivot_sentence_raw
from common.patterns import find_pivot

manifest_files = sorted(glob.glob(str(ROOT / "dataset" / "manifests" / "colonfix_*.jsonl")))
print(f"Total manifest files: {len(manifest_files)}")

total_recs = 0
colon_recs = 0
patterns_count = {}

for f in manifest_files:
    lines = Path(f).read_text(encoding="utf-8").strip().splitlines()
    total_recs += len(lines)
    for l in lines:
        rec = json.loads(l)
        prev = rec.get("previous_completion", "")
        if pivot_colon(prev):
            colon_recs += 1
            raw = _pivot_sentence_raw(prev)
            m = find_pivot(prev)
            # Find what precedes the pivot sentence or colon
            pre_pivot = prev[:m.start()] if m else ""
            # Find colon match context
            if ":" in prev:
                # find colon right before pivot
                idx = prev.find(":")
                context = prev[max(0, idx-30):min(len(prev), idx+30)]
                key = prev[max(0, idx-15):idx+1].strip()
                patterns_count[key] = patterns_count.get(key, 0) + 1

print(f"Total records: {total_recs}, Colon-flagged records: {colon_recs}")
print("\nTop colon patterns before pivot:")
for k, v in sorted(patterns_count.items(), key=lambda x: x[1], reverse=True)[:30]:
    print(f"  {k!r}: {v}")
