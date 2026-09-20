#!/usr/bin/env python3
"""Census the mutmut tree: how many mutants are scored, killed, survived, unrun.

Reads `mutants/**/*.py.meta` (the file mutmut 3.8 rewrites as it goes; `exit_code_by_key` maps
`<module>.x<qualname>__mutmut_<n>` to the exit code of the run that judged it: `0` = the tests
still passed (the mutant survived), anything else (1, 2, ...) = killed, `null` = not run yet).

`mutmut results` prints the same numbers, but only for the files it currently considers mutated —
this reads the tree directly, which is what a mid-sweep report needs.

Usage: python3 .e2e/t_7e24cea4-warm-host/mutmut_census.py [mutants-dir]
"""
import collections
import json
import pathlib
import sys


def main(argv: list[str]) -> int:
    root = pathlib.Path(argv[1] if len(argv) > 1 else "mutants")
    verdicts: collections.Counter[int | None] = collections.Counter()
    per_file: dict[str, collections.Counter[int | None]] = {}
    for meta in sorted(root.rglob("*.py.meta")):
        data = json.loads(meta.read_text(encoding="utf-8"))
        keys = data.get("exit_code_by_key") or {}
        counter = per_file.setdefault(str(meta), collections.Counter())
        for code in keys.values():
            verdicts[code] += 1
            counter[code] += 1
    killed = sum(n for code, n in verdicts.items() if code not in (None, 0))
    survived = verdicts.get(0, 0)
    unrun = verdicts.get(None, 0)
    scored = killed + survived
    print(f"mutants tree: {root}")
    for name, counter in per_file.items():
        k = sum(n for code, n in counter.items() if code not in (None, 0))
        s = counter.get(0, 0)
        u = counter.get(None, 0)
        print(f"  {name:50s} killed={k:5d} survived={s:4d} unrun={u:5d}")
    print(f"scored={scored} killed={killed} survived={survived} unrun={unrun} total={scored + unrun}")
    if scored:
        print(f"killed/scored = {killed}/{scored} = {100.0 * killed / scored:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
