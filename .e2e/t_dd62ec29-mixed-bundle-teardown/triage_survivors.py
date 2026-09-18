"""Triage the mutmut survivors of `bench/isolation.py` without `mutmut show`.

Reads the run's own artifacts (`<module>.py`, `.spans`, `.meta`) and diffs every surviving mutant
against its `__mutmut_orig` twin, printing only the changed line(s) — the classification the
Tier-M report needs (message-string / equivalent / decision-path gap).
"""
from __future__ import annotations

import collections
import difflib
import json
import pathlib
import sys

MODULE = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                      else "mutants/src/ggufone/bench/isolation.py")


def load() -> tuple[dict[str, int | None], dict[str, list[int]], list[str]]:
    meta = json.loads(MODULE.with_suffix(".py.meta").read_text(encoding="utf-8"))
    spans = json.loads(MODULE.with_suffix(".py.spans").read_text(encoding="utf-8"))["spans"]
    return meta["exit_code_by_key"], spans, MODULE.read_text(encoding="utf-8").splitlines()


def block(spans: dict[str, list[int]], lines: list[str], key: str) -> list[str]:
    """The mutant's own lines: `.spans` keys are bare (`x_<fn>__mutmut_N`), not module-qualified."""
    span = spans.get(key.rsplit(".", 1)[-1])
    if not span:
        return []
    start, end = span
    return lines[start - 1:end]


def changed_lines(orig: list[str], mutant: list[str]) -> list[str]:
    diff = difflib.unified_diff(orig, mutant, lineterm="", n=0)
    return [line for line in diff
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]


def main() -> int:
    codes, spans, lines = load()
    survivors = sorted(key for key, code in codes.items() if code == 0)
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for key in survivors:
        twin = key.rsplit("__mutmut_", 1)[0] + "__mutmut_orig"
        changed = changed_lines(block(spans, lines, twin), block(spans, lines, key))
        if not changed:
            changed = ["<not line-comparable>"]
        groups[key.rsplit("__mutmut_", 1)[0]].append(f"{key} :: {' | '.join(changed)}")
    print(f"{len(survivors)} survivors out of {len(codes)} mutants "
          f"({100.0 * (len(codes) - len(survivors)) / len(codes):.1f} % killed)\n")
    for symbol in sorted(groups, key=lambda name: -len(groups[name])):
        print(f"### {symbol}: {len(groups[symbol])} survivor(s)")
        for entry in groups[symbol][:400]:
            print(f"    {entry}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
