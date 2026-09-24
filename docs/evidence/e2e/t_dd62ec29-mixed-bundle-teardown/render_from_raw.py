"""Render the markdown tables of a bench report that lives inside a `.raw` run log.

An evidence file then shows exactly what `ggufone bench` prints without `--json`.

    python render_from_raw.py <raw> <out.md>
"""

import pathlib
import sys

from ggufone.bench import harness
from raw_report import report_of


def main() -> int:
    raw = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    pathlib.Path(sys.argv[2]).write_text(harness.render_report(report_of(raw)) + "\n",
                                         encoding="utf-8")
    print(f"rendered {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
