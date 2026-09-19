#!/usr/bin/env python3
"""Are the batch's 20 questions the same items as devset c01..c20? Read-only.

Prints each batch question (id, type, text head) next to the devset row with the same id,
and the devset coverage/reliability for it, so the batch-vs-quality mass difference can be
attributed to content (different questions) or to the path (same questions).
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path("/var/home/rybens/workspace/ggufone")
sys.path.insert(0, str(ROOT / "src"))

from ggufone.bench import compare  # noqa: E402

BATCH_Q = ROOT / ".e3c_tiel" / "batch_questions.json"
E3_BATCH_Q = ROOT / "docs" / "evidence" / "e3_batch_questions.json"
DEV = ROOT / "docs" / "evidence" / "tiel_chunks"
QUALITY = ROOT / "docs" / "evidence" / "tiel_quality.json"


def load_batch(path: pathlib.Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("questions") or data.get("items")
    return data, rows


def main() -> int:
    data, rows = load_batch(BATCH_Q)
    if isinstance(rows, dict):
        rows = [{"id": key, "text": value} if not isinstance(value, dict) else {"id": key, **value}
                for key, value in rows.items()]
    print(f"batch questions file: {BATCH_Q} keys={list(data)[:8] if isinstance(data, dict) else 'list'}"
          f" rows={len(rows)}")
    print("row0:", json.dumps(rows[0], ensure_ascii=False)[:400])
    other = json.loads(E3_BATCH_Q.read_text(encoding="utf-8"))
    print(f"same file as E3's batch questions: {other == data}")

    dev_rows: dict[str, dict] = {}
    for n in range(1, 7):
        for line in (DEV / f"devset_{n:03d}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                dev_rows[str(row.get("id"))] = row
    print(f"devset rows: {len(dev_rows)} ids {sorted(dev_rows)[:3]}..{sorted(dev_rows)[-2:]}")

    quality = json.loads(QUALITY.read_text(encoding="utf-8"))
    qrows = {str(r.get("id")): r for r in compare.rows_of(quality)}

    print("\nid | batch type/text-head | devset type/text-head | same text? | quality cov/rel")
    same_text = 0
    for row in rows:
        key = str(row.get("id"))
        btext = str(row.get("question") or row.get("text") or "")[:60]
        dev = dev_rows.get(key, {})
        dtext = str(dev.get("question") or dev.get("text") or "")[:60]
        qrow = qrows.get(key, {})
        equal = btext == dtext
        same_text += equal
        print(f"{key} | {row.get('type')} {btext!r} | {dev.get('type')} {dtext!r} | {equal} | "
              f"{qrow.get('coverage')} / {qrow.get('reliability')}")
    print(f"\nitems whose text is identical in both files: {same_text}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
