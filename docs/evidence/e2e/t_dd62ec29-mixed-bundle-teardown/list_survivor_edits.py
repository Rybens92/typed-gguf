"""Print the mutated line(s) of every survivor of the current sweep (compact, for hand-triage)."""
from __future__ import annotations

import collections
import json
import pathlib

MODULE = pathlib.Path("mutants/src/ggufone/bench/isolation.py")
meta = json.loads(MODULE.with_suffix(".py.meta").read_text(encoding="utf-8"))["exit_code_by_key"]
spans = json.loads(MODULE.with_suffix(".py.spans").read_text(encoding="utf-8"))["spans"]
lines = MODULE.read_text(encoding="utf-8").splitlines()


def block(key: str) -> list[str]:
    span = spans[key.rsplit(".", 1)[-1]]
    return [line for line in lines[span[0] - 1:span[1]]]


groups: dict[str, list[str]] = collections.defaultdict(list)
for key in sorted(k for k, code in meta.items() if code == 0):
    twin = key.rsplit("__mutmut_", 1)[0] + "__mutmut_orig"
    original = set(block(twin))
    edits = [line.strip() for line in block(key)
             if line.strip() and "__mutmut_" not in line and line not in original]
    groups[key.rsplit("__mutmut_", 1)[0].split(".")[-1]].append(" | ".join(edits)[:170] or "?")
print(f"{sum(len(v) for v in groups.values())} survivors of {len(meta)} mutants\n")
for symbol in sorted(groups, key=lambda name: -len(groups[name])):
    print(f"### {symbol}: {len(groups[symbol])}")
    for edit in groups[symbol]:
        print(f"    {edit}")
    print()
