#!/usr/bin/env python3
"""Print the two smoke-vs-arm comparisons out of `.t9bcb/stats.json` (ad-hoc eyeballing)."""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
stats = json.loads((ROOT / ".t9bcb/stats.json").read_text(encoding="utf-8"))
print("== stats keys:", sorted(stats))
print("== smokes keys:", sorted(stats.get("smokes") or {}))
print("== aux keys:", sorted(stats.get("aux") or {}))
print("== smoke_vs_arm:", json.dumps(stats.get("smoke_vs_arm"), indent=1, sort_keys=True)[:2000])
for name, block in (stats.get("smoke_vs_arm") or {}).items():
    print(f"== {name}: {block['flips']} flip(s) of {block['n']}")
    for row in block["rows"]:
        print(f"   {row['id']:>4}  smoke={row['smoke_got']!r:>16} ({row['smoke_correct']})  "
              f"arm={row['arm_got']!r:>16} ({row['arm_correct']})  "
              f"cov {row['smoke_coverage']:.3e} -> {row['arm_coverage']:.3e}  same={row['same']}")
print("== aux identity")
for name, block in (stats.get("aux_identity") or {}).items():
    print(f"   {name}: {block['decisions_identical']}/{block['n']} identical decisions, "
          f"{block['prefix_tokens_identical']}/{block['n']} identical prefix token counts, "
          f"max|d coverage|={block['max_abs_coverage_delta']}, "
          f"flips={len(block['winner_flips'])}")
print("== aux cells")
for name, arm in (stats.get("aux") or {}).items():
    cell = arm["cell"]
    print(f"   {name}: {cell['correct']}/{cell['items']} low_mass={cell['low_mass']} "
          f"refusals={cell['refusals']} cov_p50={cell['coverage_quantiles']['p50']:.3e} "
          f"verdicts={cell.get('verdicts')} closers={cell['closers']['closers']}")
print("== per-chunk (challenger): correct / low_mass / refusals")
for chunk in stats["challenger_chunks"]:
    print(f"   {chunk['chunk']}: {chunk['correct']}/{chunk['items']} low_mass={chunk['low_mass']} "
          f"refusals={chunk['refusals']} ok={chunk['ok']} "
          f"chat={list((chunk.get('chat_format_seen') or {}).values())} "
          f"warn={chunk['mismatch_warnings']}")
print("== challenger cue verdicts:", stats["challenger"]["verdicts"],
      "| refusals:", stats["challenger"]["refusals"])
print("== challenger closers:", stats["challenger"]["closers"])
