#!/usr/bin/env python3
"""Pull the survivors of THIS card's new symbols out of a mutmut run dir and diff them vs orig.

Usage: python3 survivors_of_card.py <run_dir_file> <symbol-substring> [<substring> ...]
"""
from __future__ import annotations

import difflib
import json
import pathlib
import re
import sys


def fn_blocks(text: str) -> dict[str, str]:
    """Map function name -> its source block (top-level or indented method)."""
    lines = text.splitlines(keepends=True)
    blocks: dict[str, list[str]] = {}
    name = None
    indent = 0
    for line in lines:
        m = re.match(r"^(\s*)(?:async )?def (\w+)__mutmut_(\w+)\(", line)
        if m:
            name = f"{m.group(2)}__mutmut_{m.group(3)}"
            indent = len(m.group(1))
            blocks[name] = [line]
            continue
        if name is not None:
            lead = line.lstrip()
            if (line.strip() and (len(line) - len(lead)) <= indent
                    and not lead.startswith(("#", '"""', "'''"))):
                name = None
            else:
                blocks[name].append(line)
    return {k: "".join(v) for k, v in blocks.items()}


def main(argv: list[str]) -> int:
    path = pathlib.Path(argv[1])
    wanted = argv[2:]
    meta = json.loads(path.with_suffix(path.suffix + ".meta").read_text())
    codes = meta["exit_code_by_key"]
    text = path.read_text()
    blocks = fn_blocks(text)
    orig = {k: v for k, v in blocks.items() if k.endswith("__mutmut_orig")}
    survivors = []
    for key, code in codes.items():
        if code != 0:
            continue
        if not any(w in key for w in wanted):
            continue
        short = key.split(".")[-1]
        survivors.append(short)
    print(f"# {path.name}: {len(survivors)} survivors matching {wanted}")
    for short in sorted(survivors):
        name = short if short in blocks else short.replace("xǁ", "x_")
        # mutmut names methods as xǁClassǁmethod__mutmut_N
        if name not in blocks:
            parts = short.split("ǁ")
            cand = parts[-1]
            name = cand if cand in blocks else name
        variant = blocks.get(short) or blocks.get(name)
        base = short.split("__mutmut_")[0] + "__mutmut_orig"
        base_block = orig.get(base)
        print(f"\n===== {short}")
        if variant and base_block:
            diff = difflib.unified_diff(base_block.splitlines(True), variant.splitlines(True),
                                        "orig", short, n=1)
            sys.stdout.writelines(diff)
        else:
            print("  (no block found; keys:", [k for k in list(blocks)[:3]], ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
