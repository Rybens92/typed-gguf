#!/usr/bin/env python3
"""Scratch: print the shape of the existing E3 chunk reports and quality report."""
import json
import pathlib
import sys

paths = sys.argv[1:]
for p in paths:
    path = pathlib.Path(p)
    if not path.exists():
        print(f"{p}: MISSING")
        continue
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d.get("rows") or []
    overall = d.get("overall") or {}
    print(f"== {p}")
    print(f"   keys      : {sorted(d.keys())}")
    print(f"   suite     : {d.get('suite')!r}  label: {d.get('label')!r}")
    print(f"   ok        : {d.get('ok')}   rows: {len(rows)}")
    print(f"   overall   : {json.dumps(overall)}")
    print(f"   per_type  : {json.dumps(d.get('per_type'))}")
    print(f"   chunks    : {json.dumps(d.get('chunks'))}")
    print(f"   model     : {json.dumps(d.get('model'))}")
    print(f"   config    : {json.dumps(d.get('config'))}")
    print(f"   items     : {[r.get('id') or r.get('item_id') for r in rows]}")
    if rows:
        print(f"   row0 keys : {sorted(rows[0].keys())}")
        print(f"   row0      : "
              f"{json.dumps({k: v for k, v in rows[0].items() if k != 'prompt'})[:600]}")
    print()
