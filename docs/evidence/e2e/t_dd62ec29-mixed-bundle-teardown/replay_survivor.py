"""Replay one mutmut survivor against the live source (the Tier-M triage's proof step).

    python replay_survivor.py <module.meta-key> [<key> ...]

For each key: splice the mutant's function body (from the inlined `mutants/` copy, its `def` line
renamed back to the real one) over the original function in `src/`, run the gate file, restore the
source, and print KILLED/SURVIVED with the pytest exit code. Never leaves a mutated file behind
(the original is restored in `finally`, and again at the end of every iteration).
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(os.environ.get("MUTANTS_DIR", "mutants")) / "src/ggufone/bench/isolation.py"
TARGET = pathlib.Path("src/ggufone/bench/isolation.py")
GATE = "tests/test_bench_isolation.py"


def mutant_block(key: str) -> tuple[list[str], str]:
    """`(block, real_name)` for one mutant key, with the block's `def` line back to the real name.

    mutmut's key shape is `x_<function>` (`x__private_helper` for a private one) — see the
    `.meta` keys; the mangled def line is rewritten to the real name before it is spliced in.
    """
    spans = json.loads(ROOT.with_suffix(".py.spans").read_text(encoding="utf-8"))["spans"]
    lines = ROOT.read_text(encoding="utf-8").splitlines()
    start, end = spans[key.rsplit(".", 1)[-1]]
    block = lines[start - 1:end]
    definition = next(line for line in block if line.lstrip().startswith("def "))
    token = definition.split("def ", 1)[1].split("(", 1)[0]        # x_<fn>__mutmut_<N>
    name = token.split("__mutmut_", 1)[0][2:]                     # `x_` is the mangle prefix
    block = [line.replace(token, name, 1) for line in block]
    return block, name


def function_bounds(lines: list[str], name: str) -> tuple[int, int]:
    """`(first, last)` line indexes of the top-level `def <name>` block in `lines`."""
    start = next(index for index, line in enumerate(lines) if line.startswith(f"def {name}("))
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if line and not line[0].isspace():
            break
        end += 1
    return start, end


def main(argv: list[str]) -> int:
    original = TARGET.read_text(encoding="utf-8")
    target_lines = original.splitlines()
    surviving = 0
    try:
        for key in argv[1:]:
            block, name = mutant_block(key)
            start, end = function_bounds(target_lines, name)
            TARGET.write_text("\n".join([*target_lines[:start], *block, *target_lines[end:]]) + "\n",
                              encoding="utf-8")
            result = subprocess.run([".venv/bin/python", "-m", "pytest", "-q", GATE, "-x", "-p",
                                     "no:cacheprovider"],
                                    capture_output=True, text=True)
            TARGET.write_text(original, encoding="utf-8")
            surviving += 1 if result.returncode == 0 else 0
            print(f"{'SURVIVED' if result.returncode == 0 else 'KILLED  '} {key} "
                  f"(pytest exit {result.returncode})")
            changed = [line.strip() for line in block if "report.get" in line or "else" in line]
            print(f"    mutant: {' | '.join(changed)[:180]}")
            if result.returncode:
                failed = [line for line in result.stdout.splitlines() if line.startswith("FAILED")]
                print(f"    by: {failed[0] if failed else result.stdout.splitlines()[-1][:140]}")
                message = [line for line in result.stdout.splitlines() if line.startswith("E ")]
                if message:
                    print(f"    why: {message[0][:160]}")
    finally:
        TARGET.write_text(original, encoding="utf-8")
    print(f"\n{len(argv) - 1} replayed, {surviving} still surviving")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
