"""Narrow the sweep to pressure.py alone (card t_a696ce02, survivor classification)."""
from __future__ import annotations

import pathlib
import sys

OLD = ('source_paths = ["src/ggufone/runtime/pressure.py", '
       '"src/ggufone/runtime/isolated.py"]')
NEW = 'source_paths = ["src/ggufone/runtime/pressure.py"]'


def main() -> int:
    path = pathlib.Path(sys.argv[1])
    text = path.read_text()
    if NEW in text:
        print("already narrow")
        return 0
    assert OLD in text, "the source_paths pair moved"
    path.write_text(text.replace(OLD, NEW))
    print("narrowed to pressure.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
