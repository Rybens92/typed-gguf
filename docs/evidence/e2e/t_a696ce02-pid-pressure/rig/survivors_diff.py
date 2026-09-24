"""Dump the actual mutation of every surviving mutant in a module (card t_a696ce02).

`tools/t80_survivor_lines.py` answers "does a survivor touch my lines?" but prints only the first
removed line that matches a needle. For classification we want the *diff*: what did mutmut change
in the variant. Reads the run file (`mutants/<module>.py`) + `<module>.py.meta`.
"""
from __future__ import annotations

import difflib
import json
import pathlib
import re
import sys


def blocks(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    current: str | None = None
    indent = 0
    for line in text.splitlines(keepends=True):
        match = re.match(r"^(\s*)def (x_?[\wǁ]*__mutmut_(?:orig|\d+))\(", line)
        if match:
            current = match.group(2)
            indent = len(match.group(1))
            out[current] = line
            continue
        if current is not None:
            if line.strip() and (len(line) - len(line.lstrip())) <= indent:
                current = None
            else:
                out[current] += line
    return out


def main(argv: list[str]) -> int:
    path = pathlib.Path(argv[1])
    meta = json.loads(path.with_suffix(path.suffix + ".meta").read_text())["exit_code_by_key"]
    code = blocks(path.read_text(encoding="utf-8"))
    for key, value in meta.items():
        if value != 0:
            continue
        short = key.rsplit(".", 1)[-1]
        variant = code.get(short)
        original = code.get(re.sub(r"__mutmut_\d+$", "__mutmut_orig", short))
        print(f"--- {key}")
        if variant is None or original is None:
            print("    <block not found in the run file>")
            continue
        for line in difflib.unified_diff(original.splitlines(), variant.splitlines(),
                                         lineterm="", n=1):
            if line.startswith(("---", "+++", "@@")):
                continue
            print(f"    {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
