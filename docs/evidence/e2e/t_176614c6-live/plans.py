"""(c) receipt reader: the plan numbers of a `fit --json` (or an ask answer) file, one line each."""
import json
import pathlib
import sys

FIELDS = ("n_gpu_layers", "n_ctx", "kv_type", "budget_bytes", "est_weights_bytes", "est_kv_bytes",
          "est_total_bytes", "ctx_limit", "source", "warnings")


def line(path, label):
    try:
        body = json.loads(pathlib.Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"{label}: unreadable ({exc})")
        return
    if "engine" in body:
        body = (body.get("engine") or {}).get("fit") or {}
        label += " [answer engine.fit]"
    print(f"{label}: " + json.dumps({k: body.get(k) for k in FIELDS if k in body}, sort_keys=False))
    for note in body.get("notes") or []:
        print(f"    note: {note}")


for arg in sys.argv[1:]:
    label, _, path = arg.partition("=")
    line(path, label)
