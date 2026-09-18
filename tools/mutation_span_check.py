#!/usr/bin/env python3
"""Answer one question for a review: do any mutmut survivors cover the lines THIS change touched?

    python3 tools/mutation_span_check.py <mutants-dir> <module.py> <needle> [<needle> ...]

Each `<needle>` is a source substring of the original module. The mutant copy inlines every mutant,
so the span coordinates are the *mutants* file's; this script finds the needle's line there and
reports every survivor (status 0 = "the tests passed") whose span contains it — plus the same for
the surrounding function, so "0 survivors on my line" is distinguishable from "the function was
never reached".
"""
from __future__ import annotations

import json
import pathlib
import sys

SURVIVED = 0


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print(__doc__, file=sys.stderr)
        return 2
    root = pathlib.Path(argv[1])
    module = root / argv[2]
    needles = argv[3:]
    sources = module.read_text(encoding="utf-8").splitlines()
    spans = json.loads((module.parent / (module.name + ".spans"))
                       .read_text(encoding="utf-8"))["spans"]
    codes = json.loads((module.parent / (module.name + ".meta"))
                       .read_text(encoding="utf-8"))["exit_code_by_key"]
    for needle in needles:
        hits = [index + 1 for index, line in enumerate(sources) if needle in line]
        print(f"== {needle!r} at mutants-copy line(s) {hits or '<not found>'}")
        if not hits:
            continue
        for line_no in hits:
            covering = [key for key, (start, end) in spans.items() if start <= line_no <= end]
            mutants = [key for key in covering if "__mutmut_" in key]
            survivors = [key for key in mutants if codes.get(key) == SURVIVED]
            unrun = [key for key in mutants if codes.get(key) is None]
            print(f"   line {line_no}: {len(mutants)} mutants, {len(survivors)} survived, "
                  f"{len(unrun)} unrun")
            for key in survivors:
                print(f"     SURVIVED {key}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
