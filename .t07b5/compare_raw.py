"""Compare two native response JSONs on everything except timings (SPEC A5)."""
from __future__ import annotations

import hashlib
import json
import sys


def strip(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        body = json.load(handle)
    body.pop("timings", None)
    return json.dumps(body, sort_keys=True, indent=2)


def main(left: str, right: str) -> int:
    a, b = strip(left), strip(right)
    print(f"{left}: sha256 {hashlib.sha256(a.encode()).hexdigest()[:16]}")
    print(f"{right}: sha256 {hashlib.sha256(b.encode()).hexdigest()[:16]}")
    if a == b:
        print("identical (timings stripped)")
        return 0
    left_lines = a.splitlines()
    right_lines = b.splitlines()
    import difflib
    for line in difflib.unified_diff(left_lines, right_lines, lineterm="", n=1):
        print(line)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
