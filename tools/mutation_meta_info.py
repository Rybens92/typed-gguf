"""Print the sizes of the sub-maps inside a mutmut `.meta` file (evidence debugging helper)."""
from __future__ import annotations

import json
import pathlib
import sys


def main(argv: list[str]) -> int:
    for name in argv[1:]:
        payload = json.loads(pathlib.Path(name).read_text(encoding="utf-8"))
        print(f"== {name}")
        for key, value in payload.items():
            if isinstance(value, dict):
                codes: dict[int, int] = {}
                for code in value.values():
                    if isinstance(code, int):
                        codes[code] = codes.get(code, 0) + 1
                print(f"  {key:<28} entries={len(value):<6} value_histogram={codes}")
            else:
                print(f"  {key:<28} {value!r}")
        sample = list(payload.get("exit_code_by_key", {}).items())[:3]
        print(f"  sample: {sample}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
