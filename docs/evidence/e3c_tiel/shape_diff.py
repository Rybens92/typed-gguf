#!/usr/bin/env python3
"""Same item, two shapes: the quality row vs the batch answer for c03 (read-only).

Prints both payloads' shape-level fields (template, warnings, prefix/state, n_ctx, n_seq_max,
coverage, cue) so the mass difference can be attributed to the rendering the engine actually used
rather than guessed at.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path("/var/home/rybens/workspace/ggufone")
sys.path.insert(0, str(ROOT / "src"))

from ggufone.bench import compare  # noqa: E402

ITEM = "c03"
KEYS = ("type", "choice", "coverage", "reliability", "confidence", "decode_steps", "cue",
        "warnings", "template", "prefix_tokens", "state_id", "n_ctx", "n_seq_max",
        "questions", "engine", "probabilities")


def show(label: str, row: dict) -> None:
    print(f"\n== {label}")
    for key in KEYS:
        if key in row:
            value = row[key]
            text = json.dumps(value, ensure_ascii=False)
            print(f"   {key}: {text[:400]}")
    extra = [key for key in row if key not in KEYS]
    print(f"   other keys: {extra}")


def main() -> int:
    quality = json.loads((ROOT / "docs" / "evidence" / "tiel_quality.json").read_text("utf-8"))
    qrow = next((row for row in compare.rows_of(quality) if str(row.get("id")) == ITEM), {})
    show(f"quality row {ITEM}", qrow)
    batch = json.loads((ROOT / ".e3c_tiel" / "batch_response.json").read_text("utf-8"))
    brow = batch["answers"][ITEM]
    show(f"batch answer {ITEM}", brow)
    print("\n== batch engine facts")
    for key in ("n_ctx", "n_seq_max", "prefix_tokens", "state_id", "template", "kv_type",
                "n_gpu_layers", "readout"):
        if key in batch["engine"]:
            print(f"   engine.{key}: {json.dumps(batch['engine'][key], ensure_ascii=False)[:300]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
