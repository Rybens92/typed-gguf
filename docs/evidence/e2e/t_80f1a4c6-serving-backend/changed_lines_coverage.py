#!/usr/bin/env python3
"""Coverage of the lines this card *changed*, not of the whole module (card t_80f1a4c6).

    python3 changed_lines_coverage.py <coverage.json> <base-rev>

The card's target is "≥80 % for new code": a module-wide percentage hides the changed hunks
behind the file's pre-existing (often GPU-only) branches. Lines are the *added* side of
`git diff -U0 <base-rev>..HEAD` for `src/ggufone/**`, minus blanks/comments, matched against
`coverage json --pretty-print`'s per-line `executed_lines`.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<count>\d+))? @@")


def added_lines(base: str) -> dict[str, set[int]]:
    diff = subprocess.run(["git", "diff", "-U0", f"{base}..HEAD", "--", "src/ggufone"],
                          check=True, capture_output=True, text=True).stdout
    per_file: dict[str, set[int]] = {}
    current = ""
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = line[len("+++ b/"):]
            per_file.setdefault(current, set())
            continue
        match = HUNK.match(line)
        if match and current:
            start = int(match.group("start"))
            count = int(match.group("count") or 1)
            per_file[current] |= set(range(start, start + count))
    return per_file


def main(argv: list[str]) -> int:
    report = json.loads(pathlib.Path(argv[1]).read_text(encoding="utf-8"))
    covered: dict[str, set[int]] = {}
    statements: dict[str, set[int]] = {}
    for name, data in report["files"].items():
        key = name.replace("\\", "/")
        covered[key] = set(data.get("executed_lines", ()))
        statements[key] = covered[key] | set(data.get("missing_lines", ()))
    totals = {"changed": 0, "executed": 0}
    print(f"{'file':<40} {'stmts':>6} {'hit':>5} {'cover':>7}  missing")
    for path, lines in sorted(added_lines(argv[2]).items()):
        # only *executable* added lines: coverage knows every statement in the file, so a comment
        # or a blank line the diff also carries can never be "missing"
        statements_here = statements.get(path, set())
        lines &= statements_here
        hit = {line for line in lines if line in covered.get(path, set())}
        missing = sorted(lines - hit)
        totals["changed"] += len(lines)
        totals["executed"] += len(hit)
        share = f"{len(hit) / len(lines):.0%}" if lines else "—"
        print(f"{path:<40} {len(lines):>6} {len(hit):>5} {share:>7}  "
              f"{', '.join(map(str, missing[:12]))}{' …' if len(missing) > 12 else ''}")
    share = totals["executed"] / totals["changed"] if totals["changed"] else 1.0
    print(f"\nchanged statements: {totals['executed']}/{totals['changed']} = {share:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
