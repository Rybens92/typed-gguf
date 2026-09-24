"""Scratch: the shapes in `.e3b/sweep.json` and what one item-prefix cost (E3c planning)."""
from __future__ import annotations

import json
import pathlib

record = json.loads(pathlib.Path(".e3b/sweep.json").read_text(encoding="utf-8"))
print({key: value for key, value in record.items() if not isinstance(value, (list, dict))})
for item in record["items"]:
    for prefix, piece in item["prefixes"].items():
        print(item["id"], prefix, "prefill_ms", piece["prefill_ms"],
              "cue_decode_s", piece["cue_decode_s"], "n_prefix", piece["prefix_tokens"],
              "suffix_tokens", piece["suffix_tokens"])
