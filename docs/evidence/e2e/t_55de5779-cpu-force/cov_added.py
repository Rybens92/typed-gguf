"""Coverage of the *added* lines: `git diff a22bba4..HEAD` ∩ `coverage json`.

    uv run --extra dev --with coverage coverage json -o /work/t55/logs/cov.json
    uv run --extra dev --no-sync python /work/t55/cov_added.py /work/t55/logs/cov.json a22bba4 HEAD

Card t_55de5779 wants "≥80 % for new code" as a number, not a feeling: this prints, per changed
file, how many of the diff's added lines the gate selection executed, plus the whole-file percent
for context.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys

HUNK = re.compile(r"^\+\+\+ b/(?P<path>.+)$|^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<count>\d+))? @@")


def added_lines(rev_a: str, rev_b: str) -> dict[str, set[int]]:
    diff = subprocess.run(["git", "diff", "-U0", f"{rev_a}..{rev_b}"], check=True,
                          capture_output=True, text=True).stdout
    out: dict[str, set[int]] = {}
    path: str | None = None
    for line in diff.splitlines():
        match = HUNK.match(line)
        if not match:
            continue
        if match.group("path"):
            path = match.group("path")
            out.setdefault(path, set())
            continue
        if path is None:
            continue
        start, count = int(match.group("start")), int(match.group("count") or 1)
        if count:
            out[path].update(range(start, start + count))
    return out


def main() -> int:
    report = json.loads(open(sys.argv[1]).read())
    rev_a, rev_b = sys.argv[2], sys.argv[3]
    files = report["files"]
    total = covered = 0
    print(f"{'file':52} {'added':>5} {'cov':>4} {'%':>7}   whole file")
    for path, lines in sorted(added_lines(rev_a, rev_b).items()):
        if not lines or not path.startswith("src/"):
            continue
        entry = files.get(path)
        if entry is None:
            print(f"{path:52} {len(lines):5} {0:4} {0.0:6.1f}%   (not measured)")
            continue
        executed = set(entry["executed_lines"])
        missing = set(entry["missing_lines"])
        tracked = {line for line in lines if line in executed or line in missing}
        hit = tracked & executed
        total += len(tracked)
        covered += len(hit)
        whole = 100.0 * entry["summary"]["covered_lines"] / max(1, entry["summary"]["num_statements"])
        gaps = sorted(tracked - hit)
        print(f"{path:52} {len(tracked):5} {len(hit):4} "
              f"{100.0 * len(hit) / max(1, len(tracked)):6.1f}%   {whole:5.1f}%  gaps={gaps}")
    print(f"\nadded lines measured: {covered}/{total} = {100.0 * covered / max(1, total):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
