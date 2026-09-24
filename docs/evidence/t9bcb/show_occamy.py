#!/usr/bin/env python3
"""Print the Occamy block of `.t9bcb/stats.json` (ad-hoc eyeballing before the handoff)."""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
stats = json.loads((ROOT / ".t9bcb/stats.json").read_text(encoding="utf-8"))
occ = stats.get("occamy") or {}
print("keys:", sorted(occ))
for name in ("baseline", "challenger"):
    cell = (occ.get("cells") or {}).get(name) or {}
    print(f"[{name}] {cell.get('label')}")
    for key in ("correct", "items", "agreement", "ci", "low_mass", "measured", "refusals",
                "verdicts", "prefix_tokens", "effective_backend", "device_buffers", "ok"):
        print(f"    {key}: {cell.get(key)}")
print("paired:", json.dumps(occ.get("paired"), sort_keys=True))
print("sha:", occ.get("sha"), "lines:", occ.get("sha_lines"),
      "identical:", occ.get("sha_identical"), "all_ok:", occ.get("all_ok"))
print("model:", occ.get("model"))
for name in ("baseline_chunks", "challenger_chunks"):
    for chunk in occ.get(name) or []:
        print(f"{name} {chunk['chunk']}: {chunk['correct']}/{chunk['items']} "
              f"ngl {chunk['ngl_requested']}→{chunk['ngl_used']} degraded={chunk['degraded']} "
              f"backend={chunk['effective_backend']} low_mass={chunk['low_mass']} "
              f"refusals={chunk['refusals']} wall={chunk['chunk_wall_s']}s "
              f"chat={list((chunk.get('chat_format_seen') or {}).values())}")
