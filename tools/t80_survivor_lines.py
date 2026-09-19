#!/usr/bin/env python3
"""Which survivors mutate *my* lines? — diff every survivor against the original (card t_80f1a4c6).

    python3 tools/t80_survivor_lines.py <run-dir-file> <needle> [<needle> ...]

The `.spans`-based helper (`tools/mutation_span_check.py`) answers a different question badly in
this container: the run file inlines one copy of the function *per mutant*, so every needle
occurrence sits inside exactly one variant's span and the report degenerates to "this variant
survived". The useful question is "does any survivor differ from the original on one of MY
lines?", which is a diff: for each survivor, `difflib` the original block against the variant and
check the removed (original) lines against the needles. Survivors whose mutated line is not a
needle are pre-existing-code survivors and are counted, not printed.
"""
from __future__ import annotations

import difflib
import json
import pathlib
import re
import sys


def blocks(text: str) -> dict[str, str]:
    """name -> source block, for every `x...__mutmut_<n>` / `..._orig` definition.

    Method mutants are defined *inside* the class body of the run file
    (`    def xǁModelSessionǁ__init____mutmut_1(self, ...)`), so the parser must be
    indentation-aware: a block ends at the first non-blank line indented no deeper than its `def`.
    """
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


def base_name(key: str) -> str:
    return re.sub(r"__mutmut_\d+$", "__mutmut_orig", key)


def main(argv: list[str]) -> int:
    path = pathlib.Path(argv[1])
    needles = argv[2:]
    meta = json.loads(path.with_suffix(path.suffix + ".meta").read_text())["exit_code_by_key"]
    code = blocks(path.read_text(encoding="utf-8"))
    survivors = [key for key, value in meta.items() if value == 0]
    on_needle: list[tuple[str, str, str]] = []
    other = 0
    unmatched = 0
    for key in survivors:
        short = key.rsplit(".", 1)[-1]
        variant, original = code.get(short), code.get(base_name(short))
        if variant is None or original is None:
            unmatched += 1
            continue
        diff = list(difflib.unified_diff(original.splitlines(), variant.splitlines(),
                                         lineterm="", n=0))
        removed = [line for line in diff
                   if line.startswith("-") and not line.startswith("---")]
        hit = None
        for line in removed:
            body = line[1:].strip()
            if any(needle in body for needle in needles):
                hit = body
                break
        if hit:
            on_needle.append((short, original.splitlines()[0].strip()[:60], hit[:90]))
        else:
            other += 1
    print(f"# {path.name}: {len(survivors)} survivors — {len(on_needle)} mutate a needle line, "
          f"{other} mutate something else, {unmatched} could not be matched to a block "
          f"(indented method mutants are not top-level defs in the run file)")
    for short, fn, line in on_needle:
        print(f"  {short}\n      in {fn}\n      mutated original line: {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
