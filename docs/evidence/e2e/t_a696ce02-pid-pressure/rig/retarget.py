"""Retarget [tool.mutmut] in the *private clone* for card t_a696ce02 (uncommitted there).

The shared tree's `pyproject.toml` is a live sibling's WIP (this block is retargeted by every
card), so the sweep config never lands — the evidence doc records the pair instead.
"""
from __future__ import annotations

import pathlib
import sys

OLD_PATHS = 'source_paths = ["src/ggufone/engine/session.py", "src/ggufone/engine/decide.py"]'
NEW_PATHS = ('source_paths = ["src/ggufone/runtime/pressure.py", '
             '"src/ggufone/runtime/isolated.py"]')
OLD_SEL = 'pytest_add_cli_args_test_selection = ["tests/test_serving_attribution.py"]'
NEW_SEL = ('pytest_add_cli_args_test_selection = ["tests/test_probe_pressure.py",\n'
           '                                     "tests/test_probe_isolation.py",\n'
           '                                     "tests/test_capability.py"]')


def main() -> int:
    path = pathlib.Path(sys.argv[1])
    text = path.read_text()
    if NEW_PATHS in text:
        print("already retargeted")
        return 0
    assert OLD_PATHS in text, "the source_paths line moved"
    assert OLD_SEL in text, "the test-selection line moved"
    path.write_text(text.replace(OLD_PATHS, NEW_PATHS).replace(OLD_SEL, NEW_SEL))
    print("retargeted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
