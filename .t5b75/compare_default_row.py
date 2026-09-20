#!/usr/bin/env python3
"""Policy v2 (card t_5b754458): the default-run row against the published E3e arm, item by item.

    python3 .t5b75/compare_default_row.py            # the 60-item row (text + the receipt lines)
    python3 .t5b75/compare_default_row.py --json     # the same verdict, machine-readable

Reads `docs/evidence/e2_quality_v2_t_5b754458.json` (the report `run_default_row.sh` just wrote,
whose `config` names **no** policy flag) and `.e3e/bench_json_instructed_role_split.json` (the arm
the E3e table published, measured with `--cue json_instructed --chat-format role_split`). Every
field the card names is compared: the prompt's own token count, the decision, the correctness flag
and the cue verdict. Exit 1 if any item is not identical.
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_ROW = ROOT / "docs" / "evidence" / "e2_quality_v2_t_5b754458.json"
ARM = ROOT / ".e3e" / "bench_json_instructed_role_split.json"
FIELDS = ("prefix_tokens", "got", "correct", "cue")
CONFIG_KEYS = ("cue", "chat_format", "json_contract", "backend", "threads", "items", "runs")


def load(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def value(row: dict[str, Any], field: str) -> Any:
    """The row's value for a field; `cue` compares its verdict (the E3e freeze field set)."""
    return row.get("cue", {}).get("verdict") if field == "cue" else row.get(field)


def config_line(config: dict[str, Any]) -> str:
    return " ".join(f"{key}={config.get(key)}" for key in CONFIG_KEYS)


def main(argv: list[str]) -> int:
    mine = load(DEFAULT_ROW)
    arm = load(ARM)
    lines = [f"# default-run row vs the published v2 arm ({DEFAULT_ROW.name} vs {ARM.name})",
             "",
             f"- default row config: {config_line(mine.get('config') or {})}",
             f"- arm config:         {config_line(arm['config'])}",
             f"- reproduce (default row): {mine.get('commands', {}).get('reproduce')}",
             ""]
    published = {row["id"]: row for row in arm["items"]}
    rows = mine["items"]
    moved: list[str] = []
    same = 0
    for row in rows:
        expected = published.get(row["id"])
        if expected is None:
            moved.append(f"{row['id']}: not in the arm")
            continue
        deltas = {field: value(row, field) == value(expected, field) for field in FIELDS}
        if all(deltas.values()):
            same += 1
        else:
            wrong = ", ".join(f"{field}: {value(row, field)} != {value(expected, field)}"
                              for field, ok in deltas.items() if not ok)
            moved.append(f"{row['id']}: {wrong}")
    overall = mine.get("overall") or {}
    arm_overall = arm.get("overall") or {}
    lines += [
        f"- items compared: {len(rows)} (arm: {len(arm['items'])})",
        f"- item-level identity on {', '.join(FIELDS)}: **{same}/{len(rows)}**",
        f"- agreement: default row {overall.get('correct')}/{overall.get('n')} = "
        f"{overall.get('agreement')} vs arm {arm_overall.get('correct')}/{arm_overall.get('n')} = "
        f"{arm_overall.get('agreement')}",
        f"- per type: {json.dumps(mine.get('per_type'), sort_keys=True)}",
        "",
        "## items that are not identical" if moved else "## every item is identical",
        *(f"- {line}" for line in moved),
    ]
    text = "\n".join(lines) + "\n"
    verdict = {"same": same, "total": len(rows), "moved": moved,
               "default_agreement": overall.get("agreement"),
               "arm_agreement": arm_overall.get("agreement")}
    print(json.dumps(verdict, indent=2) if "--json" in argv else text)
    return 0 if not moved else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
