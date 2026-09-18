"""Pull the `ggufone.bench/v1` report out of a `.raw` run log (engine logs interleave with it)."""

import json
import pathlib
from typing import Any


def report_of(raw: str) -> dict[str, Any]:
    """The first top-level JSON object in the text — the CLI prints it last, logs around it."""
    decoder = json.JSONDecoder()
    lines = raw.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "{":
            continue
        try:
            payload, _ = decoder.raw_decode("\n".join(lines[index:]))
        except ValueError:
            continue
        if isinstance(payload, dict) and "schema" in payload:
            return payload
    raise ValueError("no ggufone.bench report in this log")


def report_at(path: str | pathlib.Path) -> dict[str, Any]:
    return report_of(pathlib.Path(path).read_text(encoding="utf-8", errors="replace"))
