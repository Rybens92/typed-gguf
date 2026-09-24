"""Render the markdown tables of a bench report that lives inside a `.raw` run log.

Card t_57cc0179 uses this to prove requirement 4 on a *live* run: whatever the report's `ok`, every
backend keeps its line in the table, and a withheld row prints its reason (the named warning) right
in the cell.

    python render_from_raw.py <raw> <out.md>
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

from ggufone.bench import harness


def report_of(raw: str) -> dict:
    """The report object inside a run log (the log may carry engine lines before the JSON)."""
    match = re.search(r"^\{", raw, re.M)
    if not match:
        raise SystemExit("no JSON report in this log")
    return json.loads(raw[match.start():])


def main() -> int:
    raw = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    pathlib.Path(sys.argv[2]).write_text(harness.render_report(report_of(raw)) + "\n",
                                         encoding="utf-8")
    print(f"rendered {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
