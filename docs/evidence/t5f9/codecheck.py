#!/usr/bin/env python3
"""Find `typed-gguf` used as *code* (outside strings/comments) — the sweep's bare-name rule can
mangle module references, and a hyphenated name never tokenizes as one identifier.

A broken reference shows up as the token sequence NAME(typed) OP(-) NAME(gguf) (or `_gguf`).
"""
from __future__ import annotations

import io
import pathlib
import subprocess
import tokenize

ROOT = pathlib.Path("/workspace/ggufone")
RECEIPT_DIRS = {"docs/evidence", ".e2e", ".gauntlet", "state"}
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}


def frozen(rel: tuple[str, ...]) -> bool:
    if not rel:
        return False
    top = rel[0]
    if top in SKIP_DIRS or top in RECEIPT_DIRS or top.startswith("mutants"):
        return True
    return rel[:2] == ("docs", "evidence") or top.startswith((".e3", ".t"))


def files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True,
                         capture_output=True).stdout.decode().split("\0")
    return [n for n in out if n.endswith(".py") and not frozen(tuple(pathlib.Path(n).parts))]


def main() -> int:
    bad = []
    for name in files():
        text = (ROOT / name).read_text(encoding="utf-8")
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
        except (tokenize.TokenError, IndentationError) as exc:
            bad.append(f"{name}: does not tokenize: {exc}")
            continue
        meaningful = [t for t in tokens if t.type not in (tokenize.NL, tokenize.NEWLINE,
                                                          tokenize.INDENT, tokenize.DEDENT,
                                                          tokenize.ENDMARKER)]
        for first, second, third in zip(meaningful, meaningful[1:], meaningful[2:]):
            if (first.string == "typed" and second.string == "-" and third.string == "gguf"
                    and third.type == tokenize.NAME):
                bad.append(f"{name}:{first.start[0]}: `typed - gguf` in code: "
                           f"{text.splitlines()[first.start[0] - 1].strip()[:110]}")
    print("\n".join(bad) if bad else "no code-context `typed-gguf`")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
