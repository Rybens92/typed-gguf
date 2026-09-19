#!/usr/bin/env python3
"""E3c-Tiel per-chunk ledger: items/types/correct/walls/placement, straight from the artifacts.

Reads `docs/evidence/tiel_chunks/{report,placement}_00N.json` and prints one row per chunk
(plus the merged totals) so the BENCHMARKS ledger can quote measured fields only.
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path("/var/home/rybens/workspace/ggufone")
CHUNKS = ROOT / "docs" / "evidence" / "tiel_chunks"


def main() -> int:
    print("chunk | items (c/s/n) | correct | agreement | median item ms | wall s | load s | ngl | degraded")
    total_correct = total_items = 0
    per_item_ms: list[float] = []
    for n in range(1, 7):
        report = json.loads((CHUNKS / f"report_{n:03d}.json").read_text(encoding="utf-8"))
        place = json.loads((CHUNKS / f"placement_{n:03d}.json").read_text(encoding="utf-8"))
        rows = report.get("items") or report.get("rows") or []
        types: dict[str, int] = {}
        for row in rows:
            types[str(row.get("type"))] = types.get(str(row.get("type")), 0) + 1
        correct = sum(1 for row in rows if row.get("correct"))
        ms = [float(row.get("questions_ms") or 0.0) for row in rows]
        per_item_ms += ms
        total_correct += correct
        total_items += len(rows)
        placement = place.get("placement") or {}
        print(f"{n:03d}   | {types.get('choice', 0)}/{types.get('score', 0)}/{types.get('noul', 0)}"
              f"          | {correct}/{len(rows)}     | {report.get('overall', {}).get('agreement')}"
              f" | {statistics.median(ms) / 1000:.1f}    | {place.get('wall_s')}"
              f" | {place.get('load_wall_s')} | {placement.get('n_gpu_layers')}"
              f" | {placement.get('degraded')}")
    merged = json.loads((ROOT / "docs" / "evidence" / "tiel_quality.json").read_text(encoding="utf-8"))
    print(f"\nmerged: {total_items} items · {total_correct} correct · "
          f"{total_correct / total_items:.3f} · overall block {json.dumps(merged.get('overall'))}")
    print(f"per-item questions_ms over 60 items: median {statistics.median(per_item_ms) / 1000:.1f} s · "
          f"min {min(per_item_ms) / 1000:.1f} s · max {max(per_item_ms) / 1000:.1f} s")
    types: dict[str, int] = {}
    for row in merged.get("items", []):
        types[str(row.get("type"))] = types.get(str(row.get("type")), 0) + 1
    print(f"merged types: {types}")
    print(f"coverage summary in the merged report: {json.dumps(merged.get('coverage'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
