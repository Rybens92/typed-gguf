#!/usr/bin/env python3
"""t_7c926398 (auxiliary) — the `--cue two_step` arm's numbers, and its row-for-row diff against
the card's shipped-cue row.

    python3 .t7c9/analyse_arm.py --arm docs/evidence/tiel_two_step_quality.json \
        --shipped docs/evidence/tiel_corrected_quality.json \
        --out .t7c9/tiel_two_step_stats.json

The point of the arm is not a second table: it is the measured answer to "does the cue knob the
product already has (`--cue two_step`, E3d) rescue this model?". `decide._advance_token` never
advances past a *refused* cue row by design, so the answer is checkable from the row diff itself.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyse  # noqa: E402

from ggufone.bench import compare  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", required=True)
    parser.add_argument("--shipped", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    arm = json.loads(pathlib.Path(args.arm).read_text(encoding="utf-8"))
    shipped = json.loads(pathlib.Path(args.shipped).read_text(encoding="utf-8"))
    arm_rows, shipped_rows = compare.rows_of(arm), compare.rows_of(shipped)
    by_id_arm = {str(row["id"]): row for row in arm_rows}
    by_id_shipped = {str(row["id"]): row for row in shipped_rows}
    keys = ("correct", "got", "coverage", "reliability", "cue", "prefix_tokens")
    different = []
    for item in sorted(set(by_id_arm) & set(by_id_shipped)):
        changed = {key: {"shipped": by_id_shipped[item].get(key),
                         "two_step": by_id_arm[item].get(key)}
                   for key in keys if by_id_shipped[item].get(key) != by_id_arm[item].get(key)}
        if changed:
            different.append({"id": item, "type": by_id_shipped[item]["type"], "changed": changed})

    stats = {
        "schema": "ggufone.t7c926398.cue_arm/v1",
        "arm": "two_step",
        "report": args.arm,
        "framing": arm.get("framing"),
        "overall": compare.agreement_block(list(arm_rows)),
        "per_type": analyse.type_table(arm_rows),
        "mass_split": analyse.mass_split(arm_rows),
        "coverage": analyse.coverage_summary(arm_rows),
        "cue": analyse.cue_block(arm_rows),
        "vs_shipped": {
            "identical_rows": len(set(by_id_arm) & set(by_id_shipped)) - len(different),
            "different_rows": len(different),
            "detail": different,
        },
    }
    pathlib.Path(args.out).write_text(json.dumps(stats, indent=1, sort_keys=True), encoding="utf-8")
    print(f"stats: {args.out}")
    print(json.dumps({"overall": stats["overall"],
                      "low_mass": stats["mass_split"]["low_mass"]["items"],
                      "refused": stats["cue"]["refused"], "identical_rows":
                      stats["vs_shipped"]["identical_rows"],
                      "different_rows": stats["vs_shipped"]["different_rows"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
