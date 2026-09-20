"""Print the engine/placement fields of an `ask`/bench JSON payload (evidence helper).

    python3 tools/show_placement.py <payload.json> [...]
"""
from __future__ import annotations

import json
import pathlib
import sys


def main(argv: list[str]) -> int:
    args = [name for name in argv[1:] if not name.startswith("--")]
    render = "--render" in argv[1:]
    for name in args:
        path = pathlib.Path(name)
        payload = json.loads(path.read_text(encoding="utf-8"))
        engine = payload.get("engine") or {}
        model = payload.get("model")
        model_name = model.get("name") if isinstance(model, dict) else model
        print(f"== {path}")
        print(f"  suite      : {payload.get('suite')}")
        print(f"  model      : {model_name}")
        print(f"  requested  : {(payload.get('placement') or {}).get('requested')}")
        print(f"  used       : "
              f"{json.dumps((payload.get('placement') or {}).get('used'), sort_keys=True)}")
        print(f"  model_load : {json.dumps(payload.get('model_load'))}")
        print(f"  n_gpu_layers(engine): {engine.get('n_gpu_layers')}")
        print(f"  placement(engine)  : {json.dumps(engine.get('placement'), sort_keys=True)}")
        print(f"  fit.plan           : {json.dumps(engine.get('fit'), sort_keys=True)[:400]}")
        backends = payload.get("backends") or []
        for row in backends:
            print(f"  row[{row.get('backend')}] measured={row.get('measured')} "
                  f"reason={row.get('reason')}")
            print(f"  row[{row.get('backend')}] "
                  f"placement_used={json.dumps(row.get('placement_used'), sort_keys=True)}")
        if render:
            from typed_gguf.bench import harness

            print()
            print(harness.render_report(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
