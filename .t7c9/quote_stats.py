#!/usr/bin/env python3
"""Print the exact values the BENCHMARKS §7.4.1 patch quotes (from the stats JSON)."""
from __future__ import annotations

import json
import pathlib
import sys

stats = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for key in ("corrected", "pre_fix"):
    block = stats[key]
    cov = block["coverage"]
    overall = block["overall"]
    ci = [round(value, 4) for value in overall["ci"]]
    print(f"{key}: overall={overall['correct']}/{overall['n']} "
          f"agreement={overall['agreement']:.4f} ci={ci}")
    for qtype, per in block["per_type"].items():
        print(f"   {qtype}: {per['correct']}/{per['n']} = {per['agreement']:.4f} "
              f"ci={[round(v, 4) for v in per['ci']]}")
    print(f"   low_mass={block['mass_split']['low_mass']['items']} "
          f"measured={block['mass_split']['measured']['items']} "
          f"refused={block['cue']['refused']} closers={block['cue']['refused_closers']}")
    print(f"   coverage median={cov['median']:.6g} min={cov['min']:.6g} p25={cov['p25']:.6g} "
          f"p75={cov['p75']:.6g} max={cov['max']:.6g} above_floor={cov['above_floor']}")
    for bucket in ("ok", "low_mass"):
        row = block["mass_split"].get(bucket)
        if row and row["items"]:
            print(f"   {bucket}: {row['correct']}/{row['items']} = {row['agreement']:.4f} "
                  f"ci={[round(v, 4) for v in row['ci']]}")
paired = stats["paired"]
print(f"paired: diff={paired['difference']:+.4f} ci=[{paired['low']:+.4f},{paired['high']:+.4f}] "
      f"p={paired['mcnemar_p']:.6g} discordant={paired['discordant']}")
print("prefix range:", min(stats["corrected"]["prefix_tokens"]), "-",
      max(stats["corrected"]["prefix_tokens"]))
print("chunks:", [(c["report"], c["correct"], c["ngl_requested"], c["ngl_used"], c["degraded"],
                   c["chunk_wall_s"], round(c["decision_median_s"], 1)) for c in stats["chunks"]])
