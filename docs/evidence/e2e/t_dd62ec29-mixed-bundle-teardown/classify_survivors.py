"""Classify the current sweep's survivors (message/encoding/equivalent/residual) for the report."""
from __future__ import annotations

import collections
import json
import pathlib
import re

MODULE = pathlib.Path("mutants/src/ggufone/bench/isolation.py")
meta = json.loads(MODULE.with_suffix(".py.meta").read_text(encoding="utf-8"))["exit_code_by_key"]
spans = json.loads(MODULE.with_suffix(".py.spans").read_text(encoding="utf-8"))["spans"]
lines = MODULE.read_text(encoding="utf-8").splitlines()

RULES = (
    ("message-string", re.compile(r"XX|`` |\bNone\b(?=.*f\"|.*return f)")),
    ("encoding-equivalent", re.compile(r"encoding=(None|\"UTF-8\")|read_text\([^)]*\)")),
    ("unreachable-branch", re.compile(r"signal \{|\bname = None\b")),
)


def changed(key: str) -> list[str]:
    twin = key.rsplit("__mutmut_", 1)[0] + "__mutmut_orig"
    first = spans[key.rsplit(".", 1)[-1]]
    mutant = lines[first[0] - 1:first[1]]
    original = lines[spans[twin.rsplit(".", 1)[-1]][0] - 1:spans[twin.rsplit(".", 1)[-1]][1]]
    if len(mutant) != len(original):
        return ["<not line-comparable>"]
    changed = [(before, after) for before, after in zip(original, mutant, strict=True)
               if before != after]
    return [f"{before.strip()} -> {after.strip()}" for before, after in changed]


counts: collections.Counter[str] = collections.Counter()
residual: list[str] = []
for key in sorted(k for k, code in meta.items() if code == 0):
    edits = changed(key)
    text = " | ".join(edits)
    for label, pattern in RULES:
        if pattern.search(text):
            counts[label] += 1
            break
    else:
        counts["residual"] += 1
        residual.append(f"{key.split('.')[-1]} :: {text[:190]}")
print(f"{len(meta)} mutants · {sum(1 for v in meta.values() if v == 0)} survivors · classes:")
for label, count in counts.most_common():
    print(f"  {count:>4}  {label}")
print("\nresidual (need a human read):")
for entry in residual:
    print("   ", entry)
