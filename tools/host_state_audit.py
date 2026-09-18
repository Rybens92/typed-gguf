#!/usr/bin/env python3
"""List every place in `tests/` where a gate could read the machine's device state (t_e29734e6).

Three greps, printed as one auditable report (the card asks for the sweep's *list*, not a claim):

  A) exact warning collections -- `warnings == ...` / `set(handle.warnings) == ...`
  B) ambient host readers      -- `fit.host_facts(` / `harness.host_facts(` with no ``host=``
  C) device-memory answers     -- `device_memory(`, `vram`/`free_bytes` literals, `free_probe=`

Run::  python3 tools/host_state_audit.py          # from the repo root
"""
from __future__ import annotations

import pathlib
import re

TESTS = pathlib.Path("tests")

PATTERNS = (
    ("A exact warning collection",
     re.compile(r"warnings\s*==|warnings\)\s*==|set\([^)]*warnings[^)]*\)\s*==")),
    ("B ambient host reader",
     re.compile(r"\b(?:fit|harness)\.host_facts\(")),
    ("C device-memory answer",
     re.compile(r"device_memory\(|\bfree_probe=|\bvram_free_bytes|\bvram_bytes|"
                r"\bfree_bytes=|free_gib")),
)


def main() -> int:
    for title, pattern in PATTERNS:
        print(f"\n## {title}")
        for path in sorted(TESTS.rglob("*.py")):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line):
                    print(f"{path}:{number}: {line.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
