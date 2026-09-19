#!/usr/bin/env python3
"""Raw `questions_ms` / `wall_ms` values of a report, with the chunk wall for comparison."""
from __future__ import annotations

import json
import pathlib
import statistics
import sys


def load(path: str) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


for path in sys.argv[1:]:
    report = load(path)
    rows = report["items"]
    questions = [float(row["questions_ms"]) for row in rows]
    walls = [float(row["wall_ms"]) for row in rows]
    print(f"{path}")
    print(f"  report wall_ms={report.get('wall_ms')} items={len(rows)}")
    print(f"  questions_ms: min={min(questions):.1f} median={statistics.median(questions):.1f} "
          f"max={max(questions):.1f} sum={sum(questions):.1f}")
    print(f"  wall_ms:      min={min(walls):.1f} median={statistics.median(walls):.1f} "
          f"max={max(walls):.1f} sum={sum(walls):.1f}")
    print("  per row:", ", ".join(f"{row['id']}={float(row['questions_ms']):.1f}"
                                  for row in rows[:4]))
