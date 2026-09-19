"""Insert `@pytest.mark.needs_fork` above the named tests (card t_a696ce02).

Usage: python3 mark.py <repo> <file:test> [<file:test> ...]
The marker set is *measured*: these are exactly the tests that fail when every spawn is denied
(the injector rig), i.e. the gates that need a real child process to be measurable.
"""
from __future__ import annotations

import pathlib
import sys

MARK = "@pytest.mark.needs_fork"


def mark(repo: pathlib.Path, spec: str) -> str:
    rel, _, name = spec.partition("::")
    path = repo / rel
    lines = path.read_text().splitlines(keepends=True)
    needle = f"def {name}("
    hits = [i for i, line in enumerate(lines) if line.lstrip().startswith(needle)]
    if len(hits) != 1:
        return f"SKIP {spec}: {len(hits)} definitions"
    at = hits[0]
    if lines[at - 1].strip() == MARK:
        return f"have {spec}"
    indent = lines[at][: len(lines[at]) - len(lines[at].lstrip())]
    lines.insert(at, f"{indent}{MARK}\n")
    path.write_text("".join(lines))
    return f"mark {spec}"


def main() -> int:
    repo = pathlib.Path(sys.argv[1])
    for spec in sys.argv[2:]:
        print(mark(repo, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
