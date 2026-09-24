"""Scratch: what the E3e arm files carry (card t_5b754458)."""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    for name in (".e3e/bench_json_instructed_role_split.json",
                 ".e3e/bench_shipped_answer_sheet.json"):
        path = ROOT / name
        data = json.loads(path.read_text(encoding="utf-8"))
        print("=" * 20, name)
        print("keys:", sorted(data.keys()))
        print("config:", json.dumps(data.get("config"), sort_keys=True))
        print("reproduce:", data.get("reproduce"))
        rows = data.get("rows") or data.get("items") or []
        print("rows:", len(rows))
        if rows:
            print("row keys:", sorted(rows[0].keys()))
            first = rows[0]
            print("first row:", json.dumps({k: first[k] for k in sorted(first)
                                            if k not in ("probabilities", "tokens")},
                                           sort_keys=True)[:800])
    return 0


if __name__ == "__main__":
    sys.exit(main())
