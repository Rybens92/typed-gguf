"""Card t_635124bf, round 2 — re-read the 4B ranges off the committed receipt.

The review (round 1) found §5 quoting ranges that do not reproduce from the receipt it cites:
cue mass "0.9784 … 0.9978" and coverage "2.3e-04 … 1.4e-03". This script reads
.e3c_specials/4b_verdicts.json (both runs, all 20 rows) and prints the real extremes.

Run from the repo root:  python3 .e3c_specials/round2_4b_ranges.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
rec = json.loads((ROOT / ".e3c_specials" / "4b_verdicts.json").read_text())
rows = rec["rows"]

masses = [r["mass"] for r in rows]
covs = [r["coverage_before"] for r in rows]
assert all(r["coverage_before"] == r["coverage_after"] for r in rows), "coverage moved between runs"
assert all(r["token"] == 198 for r in rows), "cue argmax is not the newline for every row"

lo_m = min(rows, key=lambda r: r["mass"])
hi_m = max(rows, key=lambda r: r["mass"])
lo_c = min(rows, key=lambda r: r["coverage_before"])
hi_c = max(rows, key=lambda r: r["coverage_before"])

print(f"receipt: .e3c_specials/4b_verdicts.json  ({len(rows)} rows, {rec['label']})")
print(f"cue mass   min={min(masses):.6f} ({lo_m['id']})  max={max(masses):.6f} ({hi_m['id']})")
print(f"coverage   min={min(covs):.6g} ({lo_c['id']})  max={max(covs):.6g} ({hi_c['id']})")
print(f"refused    before={rec['refused_before']}/20  after={rec['refused_after']}/20")
print(f"measured   {rec['measured_before']} -> {rec['measured_after']}")
print(f"cue ids    {rec['cue_tokens']}")
print(f"identical  coverage={rec['coverage_identical']} reliability={rec['reliability_identical']}"
      f" answers={rec['answers_identical']}")
