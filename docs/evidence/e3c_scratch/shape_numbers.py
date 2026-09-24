"""E3c scratch: the per-shape numbers the guidance quotes (Occamy, 6 items).

Prints, per shape: the refused count, the mass range of the *argmax* row, the mean/median `bare`
coverage, the items above the floor, and the pieces the model actually emitted at the readout row
(a newline loop is a fact, not an interpretation).
"""
from __future__ import annotations

import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]

record = json.loads((ROOT / ".e3c" / "occamy.json").read_text())
floor = record["mass_floor"]
print(f"{'shape':22s} {'refused':>8s} {'top-mass range':>20s} {'mean cov':>11s} {'median':>11s} "
      f"{'above floor':>11s}  pieces at the readout")
for shape in record["shapes"]:
    blocks = [item["shapes"][shape] for item in record["items"]]
    rows = [block["readout"] for block in blocks]
    refused = sum(1 for row in rows if row["cue"]["refused"])
    masses = [row["cue"]["mass"] for row in rows]
    coverage = [row["labels"]["bare"]["coverage"] for row in rows]
    above = sum(1 for value in coverage if value >= floor)
    pieces: dict[str, int] = {}
    for row in rows:
        piece = row["top_tokens"][0]["piece"]
        pieces[piece] = pieces.get(piece, 0) + 1
    top = ", ".join(f"{piece!r}×{count}" for piece, count in
                    sorted(pieces.items(), key=lambda pair: -pair[1]))
    print(f"{shape:22s} {refused:>5d}/6 {min(masses):>9.4f}…{max(masses):<9.4f} "
          f"{statistics.fmean(coverage):>11.2e} {statistics.median(coverage):>11.2e} "
          f"{above:>8d}/6  {top}")
