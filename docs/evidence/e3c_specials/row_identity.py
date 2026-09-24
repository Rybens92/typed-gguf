"""Card t_635124bf: do the two Tiel serving-shaped runs put the *same row* on the cue?

The pair spans two boxes (committed pre-fix run on the host, 9 GPU layers; live post-fix re-run in
this container, 8 GPU layers), so the claim "the same rows, two verdicts" has to be checked on the
raw numbers, not on the verdict field. Compares argmax id, mass and the full answer payload per
item; prints every difference.
"""
from __future__ import annotations

import json
import pathlib

BEFORE = pathlib.Path("/workspace/ggufone/.e3c_tiel/batch_response.json")
AFTER = pathlib.Path("/work/t635/tiel-post/batch_response.json")
before = json.loads(BEFORE.read_text(encoding="utf-8"))
after = json.loads(AFTER.read_text(encoding="utf-8"))

diffs = []
for key in sorted(before["answers"]):
    b, a = before["answers"][key], after["answers"][key]
    bc, ac = b["cue"], a["cue"]
    for field in ("token", "mass"):
        if bc[field] != ac[field]:
            diffs.append((key, field, bc[field], ac[field]))
    for field in ("coverage", "reliability", "choice", "confidence"):
        if b.get(field) != a.get(field):
            diffs.append((key, field, b.get(field), a.get(field)))
    if b.get("probabilities") != a.get("probabilities"):
        diffs.append((key, "probabilities", "differ", "differ"))
print(f"items compared: {len(before['answers'])} · differences: {len(diffs)}")
for row in diffs[:20]:
    print("  ", row)
argmax_same = all(before["answers"][k]["cue"]["token"] == after["answers"][k]["cue"]["token"]
                  for k in before["answers"])
mass_same = all(before["answers"][k]["cue"]["mass"] == after["answers"][k]["cue"]["mass"]
                for k in before["answers"])
print(f"argmax ids: identical {argmax_same}")
print(f"cue masses: identical {mass_same}")
print(f"usage before == usage after: {before['usage'] == after['usage']}")
state_before = before["engine"].get("state_id")
state_after = after["engine"].get("state_id")
print(f"state id before/after: {state_before} / {state_after}")
print(f"n_gpu_layers before/after: {before['engine'].get('n_gpu_layers')} / "
      f"{after['engine'].get('n_gpu_layers')}")
print(f"timings before: {json.dumps(before['timings'])}")
print(f"timings after:  {json.dumps(after['timings'])}")
