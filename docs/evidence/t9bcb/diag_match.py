#!/usr/bin/env python3
"""Diagnose why the closers-row rewrite does not match (prints reprs of both candidates)."""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
text = (ROOT / ".t9bcb/render_doc.py").read_text(encoding="utf-8")
needle_file = [line for line in text.splitlines() if "challenger\", f\"" in line]
print("file lines:", len(needle_file))
for line in needle_file:
    print("FILE:", repr(line))
candidates = [
    """["challenger", f"`{chall['closers']['refused']}/{chall['closers']['refused_n']}`",""",
    """["challenger", f"`{chall['closers']['refused']}/{chall['closers']['refused_n']}",""",
]
for candidate in candidates:
    print(f"count={text.count(candidate)} for {candidate!r}")
