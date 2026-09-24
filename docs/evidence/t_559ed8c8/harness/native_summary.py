"""Summarize the native `/v1/decide` body (the parts the serve story is judged on)."""
from __future__ import annotations

import json
import pathlib
import sys

body = json.loads(pathlib.Path(sys.argv[1]).read_text())
print("  top-level keys:", sorted(body))
keep = (body.get("engine") or {}).get("keep")
print("  engine.keep:", json.dumps(keep))
print("  model:", body.get("model"), "| usage:", json.dumps(body.get("usage")))
for qid, answer in (body.get("answers") or {}).items():
    print(f"  answers.{qid}: {json.dumps(answer)}")
