#!/usr/bin/env python3
"""Which Wilson convention did each artifact use? (audit t_57bd3db2)

`src/ggufone/bench/harness.py::wilson_interval` defaults to z=1.96 exactly; the decision tool
`tools/e3e_roles_decision.py` uses Z=1.959963984540054. The two differ in the 6th decimal at
n=60. This script prints both for the arm files' own `overall` blocks and for the record.
"""
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[4]


def wilson(k: int, n: int, z: float) -> tuple[float, float]:
    p = k / n
    denom = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return mid - half, mid + half


ARMS = {
    "shipped/answer_sheet": "docs/evidence/e3e/bench_shipped_answer_sheet.json",
    "shipped/role_split": "docs/evidence/e3e/bench_shipped_role_split.json",
    "two_step/answer_sheet": "docs/evidence/e3e/bench_two_step_answer_sheet.json",
    "two_step/role_split": "docs/evidence/e3e/bench_two_step_role_split.json",
    "json_instructed/answer_sheet": "docs/evidence/e3e/bench_json_instructed_answer_sheet.json",
    "json_instructed/role_split": "docs/evidence/e3e/bench_json_instructed_role_split.json",
    "json_instructed/answer_sheet/system": "docs/evidence/e3e/bench_json_instructed_answer_sheet_system.json",
    "json_instructed/role_split/system": "docs/evidence/e3e/bench_json_instructed_role_split_system.json",
}

record = json.loads((ROOT / "docs/evidence/e3e_roles_decision.json").read_text())
rec_cells = {c["label"]: c for c in record["cells"]}

print(f"{'cell':<38} {'k/n':>6} {'arm file ci[0]':>16} {'z=1.96':>16} "
      f"{'z=1.9599639..':>16} {'record ci[0]':>16}")
for label, path in ARMS.items():
    report = json.loads((ROOT / path).read_text())
    k = report["overall"]["correct"]
    n = report["overall"]["n"]
    arm_ci = report["overall"]["ci"][0]
    lo96, _ = wilson(k, n, 1.96)
    lop, _ = wilson(k, n, 1.959963984540054)
    rec_ci = rec_cells[label]["ci"][0]
    print(f"{label:<38} {k:>3}/{n:<3} {arm_ci:>16.12f} {lo96:>16.12f} {lop:>16.12f} "
          f"{rec_ci:>16.12f}")
    assert abs(arm_ci - lo96) < 1e-15, f"{label}: arm file is NOT z=1.96 ({arm_ci} vs {lo96})"
    assert abs(rec_ci - lop) < 1e-15, f"{label}: record is NOT the precise z ({rec_ci} vs {lop})"
print("\narm files: z=1.96 exactly (harness.wilson_interval default); "
      "record cells: z=1.959963984540054 (tool Z) — both confirmed on all 8 cells.")
