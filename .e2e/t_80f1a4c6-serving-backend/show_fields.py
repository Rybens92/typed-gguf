"""The three serving-path fields, side by side, from a response JSON (card t_80f1a4c6)."""
import json
import pathlib
import sys

for path in sys.argv[1:]:
    payload = json.loads(pathlib.Path(path).read_text())
    engine = payload.get("engine") or {}
    print(path)
    for key in ("backend", "backend_source", "devices", "device_buffers", "effective_backend",
                "n_gpu_layers"):
        print(f"   {key}: {json.dumps(engine.get(key))}")
    print(f"   warnings: {json.dumps(payload.get('warnings'))}")
