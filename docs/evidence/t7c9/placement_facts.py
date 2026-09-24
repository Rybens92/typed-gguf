#!/usr/bin/env python3
"""Print the placement facts of a t7c9 chunk sink (and the report's framing + timings)."""
from __future__ import annotations

import json
import pathlib
import sys


def load(path: str) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


for path in sys.argv[1:]:
    sink = load(path)
    print(f"=== {path}")
    for key in ("suite", "backend", "threads", "gpu_layers_requested", "n_ctx", "n_seq_max",
                "n_prefix", "kv_type_used", "load_wall_s", "wall_s", "started"):
        print(f"  {key}: {sink.get(key)}")
    print(f"  placement: {json.dumps(sink.get('placement'))}")
    device_log = sink.get("device_log") or []
    keep = [line for line in device_log
            if "offload" in line or "buffer size" in line or "KV" in line]
    for line in keep[:8]:
        print(f"  device_log: {line.strip()}")
