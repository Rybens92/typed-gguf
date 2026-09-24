#!/usr/bin/env python3
"""Print raw mutmut keys (repr, so the separators are visible) for one .meta file.

Usage: python3 .e2e/t_7e24cea4-warm-host/key_probe.py <file.meta> [needle]
"""
import json
import pathlib
import sys


def main(argv: list[str]) -> int:
    meta = pathlib.Path(argv[1])
    needle = argv[2] if len(argv) > 2 else ""
    keys = list((json.loads(meta.read_text(encoding="utf-8")).get("exit_code_by_key") or {}))
    print(f"{meta}: {len(keys)} keys")
    shown = 0
    for key in keys:
        if needle and needle not in key:
            continue
        print(repr(key))
        shown += 1
        if shown >= 6:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
