#!/usr/bin/env python3
"""The card's before/after, side by side, from the two responses and their stderr (t_80f1a4c6).

    python3 before_after.py <before.json> <before.err> <after.json> <after.err>

Prints the engine fields a reader uses to know *what computed*, plus the `compute buffer size`
lines both processes printed — the same model (Occamy 1.0), the same pinned b11026 Vulkan bundle,
the same question; the "before" is the published E3 batch response, the "after" is this card's
re-run (see README.md).
"""
from __future__ import annotations

import json
import pathlib
import sys

FIELDS = ("backend", "backend_source", "devices", "device_buffers", "effective_backend",
          "n_gpu_layers")


def engine_of(path: str) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8")).get("engine") or {}


def buffer_lines(path: str) -> list[str]:
    text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    return [line for line in text.splitlines() if "compute buffer size" in line]


def main(argv: list[str]) -> int:
    before_json, before_err, after_json, after_err = argv[1:5]
    before, after = engine_of(before_json), engine_of(after_json)
    print(f"{'field':<18} | {'BEFORE (E3 batch, 15e89a9)':<40} | AFTER (this card)")
    print(f"{'-' * 18}-+-{'-' * 40}-+-{'-' * 40}")
    for field in FIELDS:
        left = json.dumps(before.get(field, "<absent>"))
        right = json.dumps(after.get(field, "<absent>"))
        print(f"{field:<18} | {left:<40} | {right}")
    print()
    print("before stderr:", *buffer_lines(before_err) or ["<none>"], sep="\n  ")
    print("after stderr:", *buffer_lines(after_err) or ["<none>"], sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
