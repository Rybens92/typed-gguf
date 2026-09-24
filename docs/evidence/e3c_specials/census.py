"""Census of a mutmut 3.8 sweep from its `.meta` file (card t_635124bf).

    python3 census.py <meta> [survivors|killed|pending]

mutmut's own exit codes: 1 = the mutant was killed (tests failed), 0 = it survived (tests passed),
`None` = still pending. Anything else is an error/timeout and is reported separately — a sweep with
errors is never a score.
"""
from __future__ import annotations

import json
import sys


def main() -> int:
    with open(sys.argv[1], encoding="utf-8") as handle:
        meta = json.loads(handle.read())
    codes = meta.get("exit_code_by_key") or meta
    killed = sorted(k for k, v in codes.items() if v == 1)
    survived = sorted(k for k, v in codes.items() if v == 0)
    pending = sorted(k for k, v in codes.items() if v is None)
    other = sorted((k, v) for k, v in codes.items() if v not in (0, 1, None))
    what = sys.argv[2] if len(sys.argv) > 2 else None
    if what == "survivors":
        for key in survived:
            print("SURVIVOR", key)
    elif what == "killed":
        for key in killed:
            print("KILLED", key)
    elif what == "pending":
        for key in pending:
            print("PENDING", key)
    print(f"TOTAL={len(codes)} KILLED={len(killed)} SURVIVED={len(survived)} "
          f"PENDING={len(pending)} ERRORED={len(other)}")
    if other:
        print(f"errors: {other}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
