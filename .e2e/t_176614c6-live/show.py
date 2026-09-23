"""Summarise the receipts of the (c) repro: the plan each answer was served under."""
import json
import pathlib
import sys

base = pathlib.Path("/work/t176614c6-live")
keys = ("n_gpu_layers", "n_ctx", "kv_type", "n_seq_max", "est_weights_bytes", "est_kv_bytes",
        "est_total_bytes", "budget_bytes", "ctx_limit", "source", "warnings", "notes", "insufficient")


def show_plan(path, label):
    try:
        plan = json.loads(pathlib.Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"{label}: unreadable ({exc})")
        return
    row = {key: plan.get(key) for key in keys if key in plan}
    print(f"{label}: {json.dumps(row, sort_keys=False)}")


def show_answer(path, label):
    try:
        body = json.loads(pathlib.Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"{label}: unreadable ({exc})")
        return
    engine = body.get("engine") or {}
    fit = engine.get("fit") or {}
    print(f"{label}: answer={str(body.get('answers'))[:60]}")
    print(f"    placement={json.dumps(engine.get('placement'))}")
    print(f"    fit={json.dumps({k: fit.get(k) for k in keys if k in fit})}")
    keep = engine.get("keep") or {}
    print(f"    keep={json.dumps({k: keep.get(k) for k in ('served_by', 'fallback', 'model_load_ms', 'requests', 'pid')})}")
    print(f"    timings.total_ms={(body.get('timings') or {}).get('total_ms')}")


for arg in sys.argv[1:]:
    label, _, path = arg.partition("=")
    if "/fit/" in path or path.endswith(".plan.json"):
        show_plan(path, label)
    else:
        show_answer(path, label)
print("--- fit cache entries:")
for path in sorted((base / "home" / "fit").glob("*.json")):
    show_plan(path, f"cache:{path.name[:12]}")
