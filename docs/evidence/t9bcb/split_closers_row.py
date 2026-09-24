#!/usr/bin/env python3
"""Split render_doc.py's closers-row line by position (backtick-agnostic).

The pair-rewrite script refused this one line because its exact bytes are easy to get wrong by
hand; this version splits the line at a known separator instead of matching a literal.
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
path = ROOT / ".t9bcb/render_doc.py"
lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
target = [index for index, line in enumerate(lines)
          if line.lstrip().startswith('["challenger", f"') and "refused_n" in line]
print(f"candidates: {target}")
if len(target) != 1:
    raise SystemExit("expected exactly one closers row")
index = target[0]
line = lines[index]
indent = re.match(r"[\s]*", line).group(0)
body = line.strip()
head_marker = '["challenger", '
assert body.startswith(head_marker), body[:40]
rest = body[len(head_marker):]
new_lines = [
    f"{indent}{head_marker}\n",
    f"{indent} {rest}\n",
]
lines[index:index + 1] = new_lines
path.write_text("".join(lines), encoding="utf-8")
print("".join(new_lines), end="")
