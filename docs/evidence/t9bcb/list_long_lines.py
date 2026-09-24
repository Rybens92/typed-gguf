#!/usr/bin/env python3
"""Print the repr of the E501 lines of `.t9bcb/*.py` (exact bytes, for exact rewrites)."""
from __future__ import annotations

import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
for target in sorted((ROOT / ".t9bcb").glob("*.py")):
    text = target.read_text(encoding="utf-8").splitlines()
    flags = subprocess.run(
        ["uv", "run", "--frozen", "--offline", "--extra", "dev", "ruff", "check",
         "--output-format=concise", str(target)],
        cwd=ROOT, capture_output=True, text=True).stdout
    hits = sorted({int(match.group(1)) for match in
                   (re.search(r":(\d+):\d+: E501", line) for line in flags.splitlines())
                   if match})
    if not hits:
        continue
    print(f"===== {target.name}: {len(hits)} long line(s)")
    for number in hits:
        print(f"--- {number}: {text[number - 1]!r}")
