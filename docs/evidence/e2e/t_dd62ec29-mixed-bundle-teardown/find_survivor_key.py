"""Find the mutant keys whose block contains a needle (triage helper).

    python find_survivor_key.py "<source needle>" ["<another>"]
"""
from __future__ import annotations

import json
import pathlib
import sys

MODULE = pathlib.Path("mutants/src/ggufone/bench/isolation.py")
meta = json.loads(MODULE.with_suffix(".py.meta").read_text(encoding="utf-8"))["exit_code_by_key"]
spans = json.loads(MODULE.with_suffix(".py.spans").read_text(encoding="utf-8"))["spans"]
lines = MODULE.read_text(encoding="utf-8").splitlines()

for needle in sys.argv[1:]:
    print(f"== {needle!r}")
    for key in sorted(k for k, code in meta.items() if code == 0):
        span = spans[key.rsplit(".", 1)[-1]]
        text = "\n".join(lines[span[0] - 1:span[1]])
        if needle in text:
            print(f"   {key}")
