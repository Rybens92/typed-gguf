#!/usr/bin/env python3
"""Rename-equivalence audit (card t_5f9c15fe).

For every changed file: take the pre-sweep blob from HEAD, apply the *forward* sweep rules (the
same ordered list `sweep.py` used) and compare with what is on disk now. Equal -> the file changed
by the rename and nothing else. Different -> a real edit, listed so every one of them can be
explained instead of hiding inside a 1500-line diff.
"""
from __future__ import annotations

import difflib
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path("/workspace/ggufone")

#: sweep.py's own ordered rules — the mechanical rename only.
RULES: list[tuple[str, str]] = [
    (r'"(ggufone)/0\.1 \(\+https://github\.com/Rybens92/ggufone\)"',
     r'"typed-gguf/0.1 (+https://github.com/Rybens92/typed-gguf)"'),
    (r"\blibggufone", "libtyped_gguf"),
    (r"GGUFONE_", "TYPED_GGUF_"),
    (r"GgufoneError", "TypedGgufError"),
    (r"Ggufone", "TypedGguf"),
    (r"github\.com/Rybens92/ggufone", "github.com/Rybens92/typed-gguf"),
    (r"src/ggufone", "src/typed_gguf"),
    (r"XDG_DATA_HOME/ggufone", "XDG_DATA_HOME/typed-gguf"),
    (r"share/ggufone", "share/typed-gguf"),
    (r"\bimport ggufone\b", "import typed_gguf"),
    (r"\bfrom ggufone\b", "from typed_gguf"),
    (r"-m ggufone\b", "-m typed_gguf"),
    (r"\bggufone\.", "typed_gguf."),
    (r"\bggufone_", "typed_gguf_"),
    (r"\bggufone/", "typed_gguf/"),
    (r"\bggufone-", "typed-gguf-"),
    (r"`ggufone`", "`typed-gguf`"),
    (r"\bggufone\b", "typed-gguf"),
]


def changed() -> list[tuple[str, str]]:
    """[(new_path, old_path)] from the staged diff."""
    raw = subprocess.run(["git", "diff", "--cached", "--name-status", "-M"], cwd=ROOT,
                         check=True, capture_output=True).stdout.decode()
    out = []
    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            out.append((parts[1], parts[1]))
        elif len(parts) == 3:
            out.append((parts[2], parts[1]))
    return out


def main() -> int:
    pure, edited, missing = [], [], []
    for new_path, old_path in changed():
        if new_path.startswith("docs/evidence/"):
            continue
        blob = subprocess.run(["git", "show", f"HEAD:{old_path}"], cwd=ROOT,
                              capture_output=True).stdout
        try:
            old_text = blob.decode("utf-8")
        except UnicodeDecodeError:
            pure.append(new_path)
            continue
        swept = old_text
        for pattern, repl in RULES:
            swept = re.sub(pattern, repl, swept)
        target = ROOT / new_path
        if not target.exists():
            missing.append(f"{old_path} -> {new_path}")
            continue
        new_text = target.read_text(encoding="utf-8")
        if swept == new_text:
            pure.append(new_path)
        else:
            edited.append((new_path, swept, new_text))
    print(f"pure rename: {len(pure)} files; real edits: {len(edited)}")
    if missing:
        print("REMOVED WITHOUT RENAMING: " + ", ".join(missing))
    for name, swept, new_text in edited:
        diff = [line for line in difflib.unified_diff(swept.splitlines(), new_text.splitlines(),
                                                      lineterm="", n=0)
                if line[:1] in "+-" and not line.startswith(("---", "+++"))]
        print(f"\n=== {name}  ({len(diff)} changed lines)")
        for line in diff[:16]:
            print(f"    {line[:116]}")
        if len(diff) > 16:
            print(f"    … {len(diff) - 16} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
