"""Compare a served native body with a CLI `ask` body (SPEC A-E5-4, on the real model)."""
from __future__ import annotations

import json
import pathlib
import sys

served = json.loads(pathlib.Path(sys.argv[1]).read_text())
asked = json.loads(pathlib.Path(sys.argv[2]).read_text())


def noul(body: dict) -> float | None:
    return (body.get("answers") or {}).get("escalate", {}).get("noul")


def pid(body: dict) -> object:
    return ((body.get("engine") or {}).get("keep") or {}).get("pid")


def served_by(body: dict) -> object:
    return ((body.get("engine") or {}).get("keep") or {}).get("served_by")


def usage(body: dict) -> dict:
    native = body.get("usage") or {}
    return {key: native.get(key) for key in ("input_tokens", "output_tokens")}


print(f"  served: model={served.get('model')} noul={noul(served)} "
      f"served_by={served_by(served)} pid={pid(served)} usage={usage(served)}")
print(f"  ask:    model={asked.get('model')} noul={noul(asked)} "
      f"served_by={served_by(asked)} pid={pid(asked)} usage={usage(asked)}")
same_value = noul(served) == noul(asked)
same_model = served.get("model") == asked.get("model")
same_pid = pid(served) is not None and pid(served) == pid(asked)
print(f"  noul identical: {same_value} | model identical: {same_model} | "
      f"same warm host pid: {same_pid}")
raise SystemExit(0 if (same_value and same_model and same_pid) else 1)
