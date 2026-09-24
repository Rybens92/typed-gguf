#!/usr/bin/env python3
"""Compare the pre-rename mutmut meta with this card's run, key by key (card t_5f9c15fe)."""
from __future__ import annotations

import collections
import json
import pathlib

ROOT = pathlib.Path("/workspace/ggufone")
OLD = ROOT / "mutants.stale-t5f9/tools/e3e_roles_decision.py.meta"
NEW = ROOT / "mutants/tools/e3e_roles_decision.py.meta"

old = json.loads(OLD.read_text())["exit_code_by_key"]
new = json.loads(NEW.read_text())["exit_code_by_key"]

print(f"old keys {len(old)}  new keys {len(new)}")
print(f"only in old: {len(set(old) - set(new))}   only in new: {len(set(new) - set(old))}")
print("new verdict histogram:", dict(collections.Counter(repr(v) for v in new.values())))
print("old verdict histogram:", dict(collections.Counter(repr(v) for v in old.values())))

mismatch = collections.Counter()
examples = []
for key in sorted(set(old) & set(new)):
    if old[key] != new[key]:
        mismatch[(repr(old[key]), repr(new[key]))] += 1
        if len(examples) < 6:
            examples.append((key, old[key], new[key]))
print("verdict differences (old -> new):", dict(mismatch))
for key, o, n in examples:
    print(f"   {key}: old={o!r} new={n!r}")
