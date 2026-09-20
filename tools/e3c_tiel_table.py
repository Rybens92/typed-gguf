#!/usr/bin/env python3
"""Render the three-way agreement table: 4B default (E2) · Occamy 1.0 (E3) · Tiel-Coder (E3c).

The E3 tool renders *two* models (`e3_reproduce --suite compare`); the card asks for three on the
same items, so this script reuses the committed comparison arithmetic (`typed_gguf.bench.compare`)
and prints one wide table instead of two deltas.

    python3 tools/e3c_tiel_table.py \
        --report docs/evidence/e2_quality.json \
        --report docs/evidence/e3_occamy_quality.json \
        --report docs/evidence/tiel_quality.json \
        --label "4B default (E2, 60 items, CPU)" \
        --label "Occamy 1.0 (E3, chunks, vulkan)" \
        --label "Tiel-Coder (E3c, host, vulkan)" \
        --out docs/evidence/e3c_tiel_three_way.md

Every side is cut to the item ids all three measured (paired), with the dropped counts printed, so
no model is credited or blamed for a different item mix. `covered` (% of rows whose own verdict is
`measured`) and the coverage distribution come straight from the rows.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from typed_gguf.bench import compare  # noqa: E402


def _cell(block: dict[str, Any]) -> str:
    if not block["n"]:
        return "—"
    low, high = block["ci"]
    return f"{block['agreement']:.3f} ({block['correct']}/{block['n']}) [{low:.3f}–{high:.3f}]"


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def coverage_summary(rows: Sequence[Mapping[str, Any]]) -> str:
    values = [float(row.get("coverage") or 0.0) for row in rows]
    if not values:
        return "—"
    below = sum(1 for value in values if value < 0.10)
    return (f"median {_median(values):.2e} · min {min(values):.2e} · max {max(values):.2e} · "
            f"below the 0.10 floor {below}/{len(values)}")


def align_all(reports: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int]]:
    """Cut every report to the ids all of them measured; return both sides and the drop counts."""
    rowsets = [compare.rows_of(report) for report in reports]
    shared: set[str] = {str(row.get("id")) for row in rowsets[0]}
    for rows in rowsets[1:]:
        shared &= {str(row.get("id")) for row in rows}
    if not shared:
        raise SystemExit("the reports share no dev item — they are not comparable")
    dropped = [len({str(row.get("id")) for row in rows} - shared) for rows in rowsets]
    return [{**report, "items": [row for row in rows if str(row.get("id")) in shared]}
            for report, rows in zip(reports, rowsets, strict=True)], dropped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="append", required=True, dest="reports")
    parser.add_argument("--label", action="append", required=True, dest="labels")
    parser.add_argument("--out", default=None)
    parser.add_argument("--coverage-floor", dest="coverage_floor", type=float, default=0.10)
    args = parser.parse_args(argv)
    if len(args.reports) != len(args.labels):
        raise SystemExit("--report and --label must be given the same number of times")

    raw = [json.loads(pathlib.Path(path).read_text(encoding="utf-8")) for path in args.reports]
    reports, dropped = align_all(raw)
    rowsets = [compare.rows_of(report) for report in reports]
    models = [compare.model_row(report, label=label, coverage_floor=args.coverage_floor)
              for report, label in zip(reports, args.labels, strict=True)]
    types = sorted({qtype for model in models for qtype in model["per_type"]})

    header = "| metric | " + " | ".join(args.labels) + " | Δ (last − first) |"
    lines = [
        "Agreement on the committed 60-item dev set, 95 % Wilson intervals; every column is cut to "
        "the items all three models measured. The mass split uses the engine's own verdict, or "
        f"`coverage < {args.coverage_floor:.2f}` where a report predates it.",
        "",
        header,
        "|" + "---|" * (len(models) + 2),
    ]

    def row(metric: str, getter) -> None:
        blocks = [getter(model) for model in models]
        delta = "—"
        if blocks[0]["n"] and blocks[-1]["n"]:
            delta = f"{blocks[-1]['agreement'] - blocks[0]['agreement']:+.3f}"
        lines.append(f"| {metric} | " + " | ".join(_cell(block) for block in blocks) +
                     f" | {delta} |")

    row("overall", lambda model: model["overall"])
    for qtype in types:
        row(qtype, lambda model, qtype=qtype: model["per_type"].get(
            qtype, {"n": 0, "correct": 0, "agreement": 0.0, "ci": [0.0, 0.0]}))
    row("low_mass (below the floor)", lambda model: model[compare.LOW_MASS])
    row("measured (at or above the floor)", lambda model: model[compare.MEASURED])

    lines += ["", "| coverage of the answers | " + " | ".join(args.labels) + " | — |",
              "|" + "---|" * (len(models) + 2)]
    summary_cells = " | ".join(coverage_summary(rows) for rows in rowsets)
    lines.append("| all rows | " + summary_cells + " | — |")

    lines += ["",
              f"Paired comparison: {len(rowsets[0])} dev items measured by all three models "
              f"(dropped {', '.join(str(count) for count in dropped)} unpaired row(s) in "
              f"report order)."]
    short = [compare.model_row(report, label=label, coverage_floor=args.coverage_floor)
             for report, label in zip(reports, args.labels, strict=True)]
    covered = []
    for model, rows in zip(short, rowsets, strict=True):
        measured = len(rows) - model[compare.LOW_MASS]["n"]
        covered.append(f"{model['label']}: {measured}/{len(rows)} rows `measured`")
    lines.append("Coverage split: " + "; ".join(covered) + ".")
    text = "\n".join(lines) + "\n"
    if args.out:
        pathlib.Path(args.out).write_text(text, encoding="utf-8")
        print(f"table: {args.out}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
