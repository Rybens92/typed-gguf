"""Report one mutmut 3.8 meta file: verdicts per function, and the changed block's score.

    python3 /work/t55/sweep_report.py mutants/src/ggufone/engine/session.py.meta [fn ...]

`exit_code_by_key` semantics (mutmut 3.8, measured on this box): 1 = killed (the selection failed),
0 = survived (all green), 5 = no tests collected, None = not run, 3 = timeout.

The keys are `<module>.<function>__mutmut_N`; `[fn ...]` restricts the per-function table to the
functions a diff touches (the changed-surface block), while the whole-file totals are printed
beside it — a time-boxed sweep is reported with its not-run count, never as a full-sweep number.
"""
from __future__ import annotations

import collections
import json
import re
import sys

KEY = re.compile(r"^(?P<fn>.*?)__mutmut_(?P<n>\d+)$")
# mutmut 3.8's own vocabulary (`mutmut.stats.status_by_exit_code`), read from the installed package:
# 1/3 killed, 0 survived, 5/33 no tests, 35 suspicious, 36/24/-24/152/255 timeout, 34 skipped,
# 37 caught by the type check, -11/-9 segfault.
VERDICT = {1: "killed", 3: "killed", 0: "survived", 5: "no tests", 33: "no tests",
           34: "skipped", 35: "suspicious", 36: "timeout", 24: "timeout", -24: "timeout",
           152: "timeout", 255: "timeout", 37: "type-check", -11: "segfault", -9: "segfault",
           None: "not run"}


def main() -> int:
    meta = json.load(open(sys.argv[1]))
    wanted = set(sys.argv[2:])
    keys = meta["exit_code_by_key"]
    whole: collections.Counter[str] = collections.Counter()
    groups: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for key, code in keys.items():
        verdict = VERDICT.get(code, f"exit {code}")
        whole[verdict] += 1
        match = KEY.match(key)
        groups[match.group("fn") if match else key][verdict] += 1
    print(f"{sys.argv[1]}: {len(keys)} mutants")
    print("  whole file:", ", ".join(f"{n} {v}" for v, n in sorted(whole.items())))
    print(f"{'function':46} {'mutants':>7} {'killed':>6} {'surv':>5} {'notests':>7} {'notrun':>6} "
          f"{'score':>7}")
    for fn, counts in sorted(groups.items()):
        reach = wanted and any(fn.endswith(name) or name in fn for name in wanted)
        if wanted and not reach:
            continue
        run = counts["killed"] + counts["survived"] + counts["no tests"] + counts["timeout"]
        score = 100.0 * counts["killed"] / run if run else 0.0
        print(f"{fn:46} {sum(counts.values()):7} {counts['killed']:6} {counts['survived']:5} "
              f"{counts['no tests']:7} {counts['not run']:6} {score:6.1f}%")
    if not wanted:
        print("\n  largest functions by mutant count:")
        for fn, counts in sorted(groups.items(), key=lambda item: -sum(item[1].values()))[:12]:
            run = (counts["killed"] + counts["survived"] + counts["no tests"] + counts["timeout"])
            score = 100.0 * counts["killed"] / run if run else 0.0
            print(f"{fn:46} {sum(counts.values()):7} {counts['killed']:6} {counts['survived']:5} "
                  f"{counts['no tests']:7} {counts['not run']:6} {score:6.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
