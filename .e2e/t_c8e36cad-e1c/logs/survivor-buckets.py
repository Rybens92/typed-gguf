"""Bucket mutmut survivors per function so the classes can be described honestly (E1c, Tier M).

Usage: python .e2e/t_c8e36cad-e1c/logs/survivor-buckets.py [repo_root]
Reads mutants/<file>.meta for the two E1c modules and groups survivors by their enclosing
function/method name (mutmut names mutants `x_<name>__mutmut_<n>`, nested names use `ǁ`).
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("/workspace/ggufone")
FILES = ["src/ggufone/engine/template.py", "src/ggufone/runtime/fit.py", "src/ggufone/schema.py"]
for rel in FILES:
    meta = repo / "mutants" / rel
    meta = meta.with_name(meta.name + ".meta")
    if not meta.exists():
        print(f"== {rel}: no meta (not reached)")
        continue
    data = json.loads(meta.read_text())
    codes = data["exit_code_by_key"]
    total = len(codes)
    killed = sum(1 for c in codes.values() if c == 1)
    survived = {k: c for k, c in codes.items() if c == 0}
    other = collections.Counter(c for c in codes.values() if c not in (0, 1))
    print(f"== {rel}: {total} mutants, killed={killed}, survived={len(survived)}, other={dict(other)}")
    print(f"   score: {100.0 * killed / total:.1f}%")
    buckets: collections.Counter[str] = collections.Counter()
    for name in survived:
        body = name.split("__mutmut_")[0]
        parts = body.split("ǁ")
        buckets[".".join(parts[-2:]) if len(parts) > 1 else body] += 1
    for fn, n in buckets.most_common(12):
        print(f"     {n:4d}  {fn}")
    print(f"   (survivors in {len(buckets)} distinct functions)")
