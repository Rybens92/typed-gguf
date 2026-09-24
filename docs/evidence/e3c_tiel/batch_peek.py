#!/usr/bin/env python3
"""E3c-Tiel: read the batch response raw and print the facts deliverable 4 needs
(answers present, usage waves/forks/decode_steps, OOM/errors, wall, placement).

Read-only: parses `.e3c_tiel/batch_response.json` and prints a structured summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/var/home/rybens/workspace/ggufone")


def dig(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from dig(v, f"{path}/{k}")
    elif isinstance(obj, list):
        yield path, f"list[{len(obj)}]"
    else:
        yield path, obj


def main() -> int:
    p = REPO / ".e3c_tiel" / "batch_response.json"
    print(f"file: {p}  bytes={p.stat().st_size}")
    d = json.loads(p.read_text())
    print("top keys:", list(d.keys()))
    for k in d:
        v = d[k]
        if isinstance(v, dict):
            print(f"  {k}: dict keys={list(v.keys())}")
        elif isinstance(v, list):
            print(f"  {k}: list len={len(v)}")
        else:
            print(f"  {k}: {str(v)[:200]}")
    # answers
    for key in ("answers", "results", "items", "responses"):
        rows = d.get(key)
        if isinstance(rows, list):
            print(f"\n-- {key}: {len(rows)} rows")
            ok = 0
            for i, r in enumerate(rows[:3]):
                print(f"   [{i}] keys={list(r.keys()) if isinstance(r, dict) else type(r)}")
            for r in rows:
                if isinstance(r, dict) and r.get("ok") is not False and not r.get("error"):
                    ok += 1
            print(f"   rows without error flag: {ok}/{len(rows)}")
            break
    for key in ("usage", "run", "meta", "summary", "state", "placement", "timing", "wall"):
        if key in d:
            print(f"\n-- {key}: {json.dumps(d[key], ensure_ascii=False)[:1500]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
