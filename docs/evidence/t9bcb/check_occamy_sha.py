#!/usr/bin/env python3
"""Compare the Occamy digest this card measured with the prefix card t_a431be85 published."""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
doc = (ROOT / "docs/evidence/e3_t_a431be85_occamy.md").read_text(encoding="utf-8")
prefixes = re.findall(r"([0-9a-f]{16,64})\.\.\.", doc)
print("prefixes found in the E3 doc:", prefixes)
mine = re.search(r"[0-9a-f]{64}",
                 (ROOT / ".t9bcb/logs/occamy_sha.txt").read_text(encoding="utf-8")).group(0)
print("measured:", mine)
for prefix in prefixes:
    print(f"prefix {prefix!r} matches mine: {mine.startswith(prefix)} "
          f"(len={len(prefix)})")
