#!/usr/bin/env python3
"""Byte identity of the role-split tails across the cue shapes (audit t_57bd3db2).

`tools/e3e_role_render.py` renders, offline, the per-family role split for a given `--cue`. The
E3e mechanism reading says the `shipped`/`two_step` prompt bytes are the same and only the decoded
row moves, while `json_instructed` changes the ask line and the tail. This diffs the records.
"""
import json
import pathlib
import sys

paths = {arg.split("=", 1)[0]: arg.split("=", 1)[1] for arg in sys.argv[1:]}
records = {name: json.loads(pathlib.Path(path).read_text()) for name, path in paths.items()}

names = sorted(records)
described = ", ".join("{} (cue={})".format(name, records[name]["cue"]) for name in names)
print("records: " + described)
base = "shipped"
index = {name: {fam["file"]: fam for fam in records[name]["families"]} for name in names}
FIELDS = ("status", "opener", "tail_chars", "distinct_tails", "prompt_chars", "prefix_chars",
          "full_render_chars", "tail_head_excerpt", "tail_tail_excerpt", "prefix_excerpt")
for target in (name for name in names if name != base):
    print(f"\n=== {base} vs {target} ===")
    for file, fam_base in index[base].items():
        fam_other = index[target].get(file)
        if fam_other is None:
            print(f"  {file}: missing in {target}")
            continue
        same = {field: fam_base.get(field) == fam_other.get(field) for field in FIELDS}
        print(f"  {file:<48} status {fam_base.get('status')} "
              + " ".join(f"{field}={'same' if ok else 'DIFFERENT'}" for field, ok in same.items()))
        for field, ok in same.items():
            if not ok and field not in ("status",):
                print(f"      {field}: {base}={fam_base.get(field)!r}")
                print(f"      {field}: {target}={fam_other.get(field)!r}")
