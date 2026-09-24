"""Do the repo paths the public docs name actually exist? (the hygiene card's check, ad hoc)

Only *existing-or-not* is asserted: a token is treated as a repo path when it starts with a known
top-level directory and carries a `.`/`/` shape. Evidence globs (`docs/evidence/e1c_t_c8e36cad_*`)
are expanded with `glob`.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ["README.md", "docs/RELEASE_NOTES_v0.1.0.md", "SPEC.md"]
TOP = ("docs/", "src/", "tests/", "tools/", "state/", ".github/", "runtime.lock", "SPEC.md",
       "README.md", "LICENSE", "pyproject.toml", ".e3e/", ".t5b75/", ".t9bcb/")
TOKEN = re.compile(r"`([^`\n]+)`")


def main() -> int:
    missing: list[str] = []
    checked = 0
    for name in DOCS:
        text = (ROOT / name).read_text(encoding="utf-8")
        for token in TOKEN.findall(text):
            token = token.strip()
            if not token.startswith(TOP) or " " in token or "<" in token or "|" in token:
                continue
            candidates = [token]
            if "*" in token:
                candidates = [str(p.relative_to(ROOT)) for p in ROOT.glob(token)]
            checked += 1
            if not any((ROOT / candidate).exists() for candidate in candidates):
                missing.append(f"{name}: {token}")
    print(f"checked {checked} path citations in {len(DOCS)} documents")
    if missing:
        print("missing:")
        print("\n".join(sorted(set(missing))))
        return 1
    print("all cited paths resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
