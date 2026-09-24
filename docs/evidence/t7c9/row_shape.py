#!/usr/bin/env python3
"""Dump the field names of a bench row + selected values, for the t_7c926398 analysis script."""
from __future__ import annotations

import json
import pathlib
import sys

for path in sys.argv[1:]:
    report = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    rows = report.get("items") or []
    print(f"=== {path}: suite={report.get('suite')} rows={len(rows)}")
    print("report keys:", sorted(report.keys()))
    print("framing:", report.get("framing"))
    if rows:
        print("row keys:", sorted(rows[0].keys()))
        print("row[0]:", json.dumps(rows[0], ensure_ascii=False)[:1200])
        verdicts = {str(r.get("verdict")) for r in rows}
        print("verdicts:", sorted(verdicts))
        print("reliability values:", sorted({str(r.get('reliability')) for r in rows}))
