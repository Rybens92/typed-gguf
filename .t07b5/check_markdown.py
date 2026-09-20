"""Cheap markdown sanity check for the two public docs (tables + fenced blocks balance).

Escaped pipes (`\\|`, needed inside inline code in a table cell) do not count as column separators,
and long lines are allowed inside tables and code fences — those cannot be wrapped.
"""
from __future__ import annotations

import pathlib

DOCS = [pathlib.Path("README.md"), pathlib.Path("docs/RELEASE_NOTES_v0.1.0.md")]


def _cells(line: str) -> int:
    return line.replace("\\|", "").count("|")


def main() -> int:
    bad: list[str] = []
    for path in DOCS:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if text.count("```") % 2:
            bad.append(f"{path}: unbalanced code fences ({text.count('```')})")
        block: list[tuple[int, str]] = []
        in_fence = False
        for number, line in enumerate(lines + ["plain text"], start=1):
            if line.startswith("```"):
                in_fence = not in_fence
            if line.startswith("|"):
                block.append((number, line))
                continue
            if block:
                widths = {_cells(row) for _, row in block}
                if len(widths) != 1:
                    bad.append(f"{path}:{block[0][0]}: table pipe-count {sorted(widths)}")
                if len(block) < 3:
                    bad.append(f"{path}:{block[0][0]}: table with {len(block)} row(s)")
                block = []
            if not in_fence and not line.startswith("|") and len(line) > 250:
                bad.append(f"{path}:{number}: line of {len(line)} chars")
    print("\n".join(bad) if bad else "markdown sanity: ok")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
