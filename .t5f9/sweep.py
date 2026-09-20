#!/usr/bin/env python3
"""The mechanical sweep for card t_5f9c15fe: ggufone -> typed-gguf on the *living* surface.

Frozen (never touched): docs/evidence/**, .e2e/**, .e3*/**, .t*/**, .gauntlet/**, state/**,
mutants*/** — receipts and dev-run history.

Usage: sweep.py [--dry-run]
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path("/workspace/ggufone")
RECEIPT_DIRS = {"docs/evidence", ".e2e", ".gauntlet", "state"}
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}

#: ordered: the specific spellings first, the bare product name last.
RULES: list[tuple[str, str, str]] = [
    # --- the wire/product spellings that a blind rule would mangle
    (r'"(ggufone)/0\.1 \(\+https://github\.com/Rybens92/ggufone\)"',
     r'"typed-gguf/0.1 (+https://github.com/Rybens92/typed-gguf)"',
     "user agent / package-url pair"),
    (r"\blibggufone", "libtyped_gguf", "fake sonames in the capability fixtures"),
    # --- env vars (all of them) and the exception class
    (r"GGUFONE_", "TYPED_GGUF_", "env vars"),
    (r"GgufoneError", "TypedGgufError", "the exception base class"),
    (r"Ggufone", "TypedGguf", "any other CamelCase identifier"),
    # --- paths
    (r"github\.com/Rybens92/ggufone", "github.com/Rybens92/typed-gguf", "repo URL"),
    (r"src/ggufone", "src/typed_gguf", "the source package path"),
    (r"XDG_DATA_HOME/ggufone", "XDG_DATA_HOME/typed-gguf", "the XDG default dir"),
    (r"share/ggufone", "share/typed-gguf", "the default data home"),
    # --- module spellings
    (r"\bimport ggufone\b", "import typed_gguf", "import statements"),
    (r"\bfrom ggufone\b", "from typed_gguf", "from-imports"),
    (r"-m ggufone\b", "-m typed_gguf", "`python -m`"),
    (r"\bggufone\.", "typed_gguf.", "dotted schema strings + module paths"),
    (r"\bggufone_", "typed_gguf_", "identifiers (incl. MCP tool names)"),
    (r"\bggufone/", "typed_gguf/", "module-relative source paths in prose"),
    # --- everything else is a shell/dir/file name
    (r"\bggufone-", "typed-gguf-", "hyphen compounds: dirs, files, CI tmp paths"),
    (r"`ggufone`", "`typed-gguf`", "the backticked product name"),
    (r"\bggufone\b", "typed-gguf", "the product/command name"),
]


def frozen(rel: tuple[str, ...]) -> bool:
    if not rel:
        return False
    top = rel[0]
    if top in SKIP_DIRS or top in RECEIPT_DIRS or top.startswith("mutants"):
        return True
    return rel[:2] == ("docs", "evidence") or top.startswith((".e3", ".t"))


def living_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True,
                         capture_output=True).stdout.decode().split("\0")
    return [name for name in out if name and not frozen(tuple(pathlib.Path(name).parts))]


def main() -> int:
    dry = "--dry-run" in sys.argv
    files = living_files()
    changed, lines, hits = [], 0, 0
    for name in files:
        path = ROOT / name
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        original = text
        for pattern, repl, _why in RULES:
            text = re.sub(pattern, repl, text)
        if text == original:
            continue
        changed.append(name)
        lines += sum(1 for a, b in zip(original.splitlines(), text.splitlines()) if a != b)
        hits += len(re.findall("ggufone", original, re.I))
        if not dry:
            path.write_text(text, encoding="utf-8")
    print(f"{'would change' if dry else 'changed'}: {len(changed)} files, {lines} lines, "
          f"{hits} old-name tokens")
    for name in changed:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
