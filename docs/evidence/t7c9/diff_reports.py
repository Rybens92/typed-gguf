#!/usr/bin/env python3
"""Compare two quality reports row by row on the fields the cue shape moves (t7c9 plumbing)."""
from __future__ import annotations

import json
import pathlib
import sys


def load(path: str) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


a, b = load(sys.argv[1]), load(sys.argv[2])
print("framing A:", a.get("framing"))
print("framing B:", b.get("framing"))
print("config A cue:", (a.get("config") or {}).get("cue"))
print("config B cue:", (b.get("config") or {}).get("cue"))
rows_a = {row["id"]: row for row in a["items"]}
rows_b = {row["id"]: row for row in b["items"]}
for item in sorted(set(rows_a) & set(rows_b)):
    ra, rb = rows_a[item], rows_b[item]
    print(f"{item}: A cov={ra['coverage']:.6g} mass={(ra['cue'] or {}).get('mass')} "
          f"tok={(ra['cue'] or {}).get('token')} got={ra['got']} rel={ra['reliability']} "
          f"prefix={ra.get('prefix_tokens')}")
    print(f"{'':>{len(item)}}  B cov={rb['coverage']:.6g} mass={(rb['cue'] or {}).get('mass')} "
          f"tok={(rb['cue'] or {}).get('token')} got={rb['got']} rel={rb['reliability']} "
          f"prefix={rb.get('prefix_tokens')}")
