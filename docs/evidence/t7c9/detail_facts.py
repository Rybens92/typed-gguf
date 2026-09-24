#!/usr/bin/env python3
"""Facts the t7c9 doc needs and the renderer's tables do not carry: the row-level framing
surface (kind/renderer/thinking/warnings), the report warnings, the non-refused rows, and the
per-item coverage/answer detail for the misses."""
from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter


def load(path: str) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


report = load(sys.argv[1])
rows = report["items"]
print("report warnings:", json.dumps(report.get("warnings"), indent=1)[:800])
framings = Counter(json.dumps(row.get("framing"), sort_keys=True) for row in rows)
for surface, count in framings.most_common():
    print(f"framing surface x{count}: {surface}")
print("\nrows not refused at the cue:")
for row in rows:
    cue = row.get("cue") or {}
    if not cue.get("refused"):
        print(" ", json.dumps({key: row[key] for key in
                               ("id", "type", "expected", "got", "correct", "coverage",
                                "reliability")}), "cue:", json.dumps(cue))
print("\nreliability buckets:", Counter(row["reliability"] for row in rows))
print("above floor:", sum(1 for row in rows if float(row.get("coverage") or 0) >= 0.10))
print("\nmisses by type:",
      Counter(row["type"] for row in rows if not row["correct"]))
print("correct by type:", Counter(row["type"] for row in rows if row["correct"]))
print("\nrows correct despite low_mass:",
      [row["id"] for row in rows if row["correct"] and row["reliability"] != "ok"])
print("rows measured:", [(row["id"], round(float(row["coverage"]), 6), row["correct"])
                         for row in rows if float(row.get("coverage") or 0) >= 0.10])
print("\nmodel block:", json.dumps(report.get("model"))[:400])
print("host block:", json.dumps(report.get("host"))[:400])
print("devices:", json.dumps(report.get("devices"))[:300],
      "effective_backend:", report.get("effective_backend"))
