#!/usr/bin/env python3
"""Score a mutmut 3.x run from its artifacts — including a partial (time-boxed) run.

    python3 tools/mutation_score.py [mutants-dir]      # default: ./mutants

Mutmut keeps one `<module>.py.meta` per mutated module next to the mutated source, with
`exit_code_by_key` (and `<module>.py.spans` with the source ranges). The status mapping is the one
card t_c8e36cad's triage used (`.e2e/t_c8e36cad-e1c/logs/mutation_triage.py`); the score is the
same convention this project reports: **killed / (total - no-tests - not-checked - skipped -
type-check)**, so a mutant mutmut refused to test never inflates the number.

Two things this script exists to prevent (both bit earlier cards):

* `None` codes are **not-run** mutants (a time-boxed or interrupted run). Counting them as killed
  would silently report a score made of mutants nobody executed — they are reported separately.
* the same module can be scored twice under different names (a re-copy of the tree): the keys are
  matched by their mutant name, and the per-module totals are printed so a doubled module is
  visible instead of averaged away.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

STATUS = collections.defaultdict(
    lambda: "suspicious",
    {
        0: "survived",
        1: "killed",
        3: "killed",
        5: "no-tests",
        2: "interrupted",
        None: "not-checked",
        33: "no-tests",
        34: "skipped",
        35: "suspicious",
        36: "timeout",
        37: "type-check",
        -24: "timeout",
        24: "timeout",
        152: "timeout",
        255: "timeout",
        -11: "segfault",
        -9: "segfault",
    },
)
#: statuses that count as "the tests answered": the denominator of the score
SCORED = ("killed", "survived", "timeout", "segfault", "suspicious")


def main(argv: list[str]) -> int:
    root = pathlib.Path(argv[1] if len(argv) > 1 else "mutants")
    metas = sorted(root.rglob("*.meta"))
    if not metas:
        print(f"no .meta artifacts under {root} — did a mutmut run happen here?", file=sys.stderr)
        return 1
    totals: collections.Counter[str] = collections.Counter()
    survivors: collections.Counter[str] = collections.Counter()
    for meta in metas:
        codes = (json.loads(meta.read_text(encoding="utf-8")).get("exit_code_by_key") or {})
        counts = collections.Counter(STATUS[code] for code in codes.values())
        for key, code in codes.items():
            if STATUS[code] == "survived":
                survivors[_function_of(key)] += 1
        totals.update(counts)
        ran = sum(counts[name] for name in SCORED)
        score = f"{100.0 * counts['killed'] / ran:.1f} %" if ran else "n/a"
        print(f"{str(meta.relative_to(root)):<48} total {len(codes):>5}  "
              f"killed {counts['killed']:>5}  survived {counts['survived']:>5}  "
              f"score {score:>7}  not-run {counts['not-checked']:>5}  "
              f"no-tests {counts['no-tests']:>4}")
    ran = sum(totals[name] for name in SCORED)
    score = f"{100.0 * totals['killed'] / ran:.1f} %" if ran else "n/a"
    print("-" * 110)
    seen = {name: totals[name] for name in
            ("killed", "survived", "no-tests", "not-checked", "skipped", "type-check",
             "timeout", "segfault", "suspicious", "interrupted") if totals[name]}
    print(f"TOTAL total {sum(totals.values())}  {seen}")
    print(f"SCORE killed/scored = {score}   (scored={ran}, "
          f"score-minus-no-tests-and-not-run)")
    print("survivors by symbol (top 25):")
    for name, count in survivors.most_common(25):
        print(f"  {count:>5}  {name}")
    return 0


def _function_of(mutant_key: str) -> str:
    """`pkg.mod.x_placement_of__mutmut_3` -> `pkg.mod.x_placement_of` (ǁ = a class separator)."""
    stem = mutant_key.split("__mutmut_", 1)[0]
    return stem.replace("\u01c1", ".") if stem else mutant_key


if __name__ == "__main__":
    sys.exit(main(sys.argv))
