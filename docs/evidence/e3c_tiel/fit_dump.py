#!/usr/bin/env python3
"""Print the two cached fit plans without the bulky tensor/host sections."""
import json
import pathlib
import sys

for path in sorted(pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                               else "/var/home/rybens/.local/share/ggufone/fit").glob("*.json")):
    data = json.loads(path.read_text())
    print(f"### bytes_on_disk={path.stat().st_size}")
    slim = {key: value for key, value in data.items() if key not in ("tensors", "host")}
    print(f"--- {path.name}")
    print(json.dumps(slim, indent=1, sort_keys=True))
