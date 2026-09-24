"""Card t_635124bf: what the committed Tiel serving-shaped run recorded, item by item."""
from __future__ import annotations

import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = []
for key in sorted(payload["answers"]):
    row = payload["answers"][key]
    cue = row.get("cue") or {}
    rows.append((key, row.get("type"), row.get("coverage"), row.get("reliability"),
                 cue.get("token"), cue.get("mass"), cue.get("refused"), cue.get("closer"),
                 "hint" in cue))
print(f"items {len(rows)} · refused {sum(bool(r[6]) for r in rows)} "
      f"· tokens {sorted({r[4] for r in rows})}")
print("id   type      coverage    reliability  token   mass    refused closer hint")
for r in rows:
    print(f"{r[0]:<4} {str(r[1]):<9} {float(r[2]):.3e}  {str(r[3]):<11} {r[4]:<7} "
          f"{float(r[5]):.4f}  {str(r[6]):<7} {r[7]}  {r[8]}")
masses = [float(r[5]) for r in rows]
print(f"mass: min {min(masses):.4f} max {max(masses):.4f} "
      f"· >= 0.10 {sum(1 for m in masses if m >= 0.10)}/{len(masses)}")
print(f"coverage: min {min(float(r[2]) for r in rows):.3e} "
      f"max {max(float(r[2]) for r in rows):.3e}")
print(f"warnings: {json.dumps(payload.get('warnings'))}")
keys = ("backend", "n_seq_max", "prefix_tokens", "state_id")
print("engine: " + json.dumps({key: payload["engine"].get(key) for key in keys}))
print(f"usage: {json.dumps(payload.get('usage'))}")
print(f"timings: {json.dumps(payload.get('timings'))}")
