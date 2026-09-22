#!/usr/bin/env python3
"""Print selected keys of a JSON document (nested via dots), from a file or stdin.

Usage: python3 json_pluck.py key1,key2 [-] [file.json]
"""
import json
import sys

spec, *rest = sys.argv[1:]
keys = [k for k in spec.split(",") if k]
if rest and rest[0] == "-":
    doc = json.load(sys.stdin)
else:
    doc = json.load(open(rest[0] if rest else "/dev/stdin", encoding="utf-8"))


def get(d, path):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


out = {k: get(doc, k) for k in keys}
print(json.dumps(out, indent=1))
