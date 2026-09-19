#!/usr/bin/env python3
"""Census of a mutmut 3.x run dir for this card's sweep driver (card t_80f1a4c6).

    python3 tools/t80_census.py [mutants-dir]        # default: ./mutants

Prints one line per module plus a PENDING= total, so the driver's retry loop can stop when every
mutant has a verdict. `None`/absent keys are the mutants a killed run never executed.
"""
from __future__ import annotations

import json
import pathlib
import sys

MODULES = ("src/ggufone/engine/session.py", "src/ggufone/engine/decide.py")


def main(argv: list[str]) -> int:
    root = pathlib.Path(argv[1] if len(argv) > 1 else "mutants")
    pending = 0
    for module in MODULES:
        meta = root / (module + ".meta")
        if not meta.exists():
            print(f"{module}: no meta yet")
            return 1
        data = json.loads(meta.read_text())
        codes = data.get("exit_code_by_key", {})
        n_pending = sum(1 for value in codes.values() if value is None)
        pending += n_pending
        killed = sum(1 for value in codes.values() if value == 1)
        survived = sum(1 for value in codes.values() if value == 0)
        no_tests = sum(1 for value in codes.values() if value == 33)
        print(f"{module}: total={len(codes)} killed={killed} survived={survived} "
              f"no-tests={no_tests} pending={n_pending}")
    print(f"PENDING={pending}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
