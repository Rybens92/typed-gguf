#!/usr/bin/env python3
"""Policy v2 (card t_5b754458) — splice §2.3 of `docs/BENCHMARKS.md` from the stored reports.

The repository's rule (see `.e3e/splice_docs.py`, `.t9bcb/render_doc.py`): no number is retyped.
This script renders the block between

    <!-- QUALITY-V2:START ... -->
    <!-- QUALITY-V2:END -->

from `docs/evidence/e2_quality_v2_t_5b754458.json` (the 60-item row `run_default_row.sh` measured
with **no policy flag**) plus the published E3e arm `.e3e/bench_json_instructed_role_split.json`
(the cell the defaults now name), using the bench's own `harness.render_report` for the table.

    python3 .t5b75/splice_benchmarks.py            # write the block into docs/BENCHMARKS.md
    python3 .t5b75/splice_benchmarks.py --check    # exit 1 if the file is out of date

Idempotent: running it twice leaves the document byte-identical (the previous card's splicer
defect — the end marker accumulating — is the case it is written against).
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ggufone.bench import harness  # noqa: E402

TARGET = ROOT / "docs" / "BENCHMARKS.md"
REPORT = ROOT / "docs" / "evidence" / "e2_quality_v2_t_5b754458.json"
ARM = ROOT / ".e3e" / "bench_json_instructed_role_split.json"
REPORT_TEXT = "docs/evidence/e2_quality_v2_t_5b754458.json"
START = ("<!-- QUALITY-V2:START — spliced from "
         "docs/evidence/e2_quality_v2_t_5b754458.json by\n"
         "     `.t5b75/splice_benchmarks.py`; edit the tool and the report, never this block. -->")
END = "<!-- QUALITY-V2:END -->"
FIELDS = ("prefix_tokens", "got", "correct", "cue")


def load(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def value(row: dict[str, Any], field: str) -> Any:
    """The row's value for a field; `cue` compares its verdict (the E3e freeze field set)."""
    return row.get("cue", {}).get("verdict") if field == "cue" else row.get(field)


def identity(mine: dict[str, Any], arm: dict[str, Any]) -> tuple[int, int, list[str]]:
    published = {row["id"]: row for row in arm["items"]}
    same, moved = 0, []
    for row in mine["items"]:
        expected = published.get(row["id"])
        if expected is None:
            moved.append(f"{row['id']} (missing from the arm)")
            continue
        differing = [field for field in FIELDS
                     if value(row, field) != value(expected, field)]
        if differing:
            moved.append(f"{row['id']} ({', '.join(differing)})")
        else:
            same += 1
    return same, len(mine["items"]), moved


def build_block() -> str:
    mine = load(REPORT)
    arm = load(ARM)
    config = mine["config"]
    overall = mine["overall"]
    same, total, moved = identity(mine, arm)
    policy_line = (f"`cue={config['cue']}`, `chat_format={config['chat_format']}`, "
                   f"`json_contract={config['json_contract']}`")
    intro = (f"`tools/e2_reproduce.py` on the same box, the same 4B and the same 60 committed"
             f" items — **with no policy flag at all**, i.e. the recipe a new user's"
             f" `ggufone bench` runs. The row is therefore measured under the defaults"
             f" ({policy_line}), which are the cell §9's table read as the measured-good one.")
    compare = (f"- **item-level identity with the published E3e arm** "
               f"(`json_instructed/role_split`, `.e3e/bench_json_instructed_role_split.json`): "
               f"**{same}/{total}** items identical on `{'`, `'.join(FIELDS)}`"
               + (f" — the {len(moved)} exception(s): {', '.join(moved)}" if moved else "")
               + f"; agreement {overall['correct']}/{overall['n']} = {overall['agreement']:.3f} on"
                 f" both sides. The check is `.t5b75/compare_default_row.py`.")
    words = (f"**Read the row's own words:** every one of the {total} rows carries "
             f"`engine.chat_format = {{kind: role_split, question_turn: user}}` and "
             f"`engine.cue = json_instructed` with a named value-row verdict (`answered` on every"
             f" item here), so the row is comparable with the E3e table's "
             f"`json_instructed/role_split` cell and **not** with §2.2's pre-v2 row above.")
    lines = [
        START,
        "",
        "### 2.3 The same row under the product's defaults (policy v2, card `t_5b754458`)",
        "",
        intro,
        "",
        f"- reproduce: `{mine['commands']['reproduce']}`",
        f"- report: `{REPORT_TEXT}` · measured with backend `{config['backend']}` · threads "
        f"{config['threads']} · items {config['items']} · runs {config['runs']}",
        compare,
        "",
        harness.render_report(mine).rstrip(),
        "",
        words,
        "",
        "The old cell is unchanged and still one flag away: "
        "`--cue shipped --chat-format answer_sheet` reproduces §2.2 byte for byte "
        "(`tests/test_policy_v2.py::test_the_pre_v2_cell_stays_reachable_and_its_bytes_are_frozen`).",
        "",
        END,
        "",
    ]
    return "\n".join(lines)


def splice(text: str, block: str) -> str:
    """Replace the block between the markers (or insert it before §3. on the first run)."""
    if START in text:
        head = text.split(START, 1)[0]
        tail = text.split(END, 1)[1]
        return head + block + "\n" + tail.lstrip("\n")
    marker = "## 3. The measurements\n"
    head, sep, tail = text.partition(marker)
    if not sep:
        raise SystemExit(f"{TARGET} has no '{marker.strip()}' anchor to insert the block before")
    return head.rstrip("\n") + "\n\n" + block + "\n" + sep + tail


def main(argv: list[str]) -> int:
    block = build_block()
    text = TARGET.read_text(encoding="utf-8")
    updated = splice(text, block)
    if "--check" in argv:
        if updated != text:
            print(f"{TARGET} is out of date — run .t5b75/splice_benchmarks.py")
            return 1
        print(f"{TARGET} is up to date")
        return 0
    TARGET.write_text(updated, encoding="utf-8")
    print(f"wrote {TARGET} ({len(updated)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
