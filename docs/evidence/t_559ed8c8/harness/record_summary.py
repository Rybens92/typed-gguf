"""Print the parts of `runtime.json` the update/rollback story is judged on."""
from __future__ import annotations

import json
import pathlib
import sys

record = json.loads(pathlib.Path(sys.argv[1]).read_text())
for key in ("schema", "dir", "tag", "build", "variant", "installed_at", "update_from",
            "rolled_back_at", "rolled_back_from", "previous", "libllama_sha256",
            "backend_requested", "backends", "symbols_ok"):
    if key in record:
        value = record[key]
        if key == "libllama_sha256" and isinstance(value, str):
            value = value[:16] + "…"
        print(f"  {key}: {json.dumps(value)}")
print(f"  record keys: {sorted(record)}")
