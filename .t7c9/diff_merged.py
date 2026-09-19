#!/usr/bin/env python3
"""Row-for-row diff of two merged quality reports on the fields a cue shape can move."""
from __future__ import annotations

import json
import pathlib
import sys


def load(path: str) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


a, b = load(sys.argv[1]), load(sys.argv[2])
rows_a = {row["id"]: row for row in a["items"]}
rows_b = {row["id"]: row for row in b["items"]}
print(f"A: {sys.argv[1]} ({len(rows_a)} rows) cue={(a.get('config') or {}).get('cue')}")
print(f"B: {sys.argv[2]} ({len(rows_b)} rows) cue={(b.get('config') or {}).get('cue')}")
same = different = 0
for item in sorted(set(rows_a) & set(rows_b)):
    ra, rb = rows_a[item], rows_b[item]
    keys = ("correct", "got", "coverage", "reliability", "cue", "prefix_tokens")
    if all(ra.get(key) == rb.get(key) for key in keys):
        same += 1
        continue
    different += 1
    print(f"DIFF {item} ({ra['type']}):")
    for key in keys:
        if ra.get(key) != rb.get(key):
            print(f"   {key}: A={json.dumps(ra.get(key))[:120]}  B={json.dumps(rb.get(key))[:120]}")
print(f"\nidentical rows: {same}/{len(set(rows_a) & set(rows_b))} · different rows: {different}")
print(f"A correct {sum(1 for r in rows_a.values() if r['correct'])}/60 · "
      f"B correct {sum(1 for r in rows_b.values() if r['correct'])}/60")
print(f"A low_mass {sum(1 for r in rows_a.values() if r['reliability'] == 'low_mass')} · "
      f"B low_mass {sum(1 for r in rows_b.values() if r['reliability'] == 'low_mass')}")
print(f"A refused {sum(1 for r in rows_a.values() if (r.get('cue') or {}).get('refused'))} · "
      f"B refused {sum(1 for r in rows_b.values() if (r.get('cue') or {}).get('refused'))}")
