"""Say whether an arm's report is already a complete `--backend vulkan` measurement.

`.e3e/run_arms.sh` calls this to resume a campaign that a reclaim killed: an arm is "done" only
when its report holds the full 60-item dev set *and* names the backend this card requires
(single-backend attribution — the card's rule). Anything else is re-measured.
"""

from __future__ import annotations

import json
import sys

WANT_ITEMS = 60
WANT_BACKEND = "vulkan"


def verdict(path: str) -> tuple[bool, str]:
    try:
        data = json.load(open(path))
    except FileNotFoundError:
        return False, "no report yet"
    except Exception as error:  # noqa: BLE001
        return False, f"unreadable ({error})"
    if not data.get("ok", False):
        return False, "report says ok=false"
    items = data.get("items") or []
    if len(items) < WANT_ITEMS:
        return False, f"{len(items)} items < {WANT_ITEMS}"
    config = data.get("config") or {}
    if config.get("backend") != WANT_BACKEND:
        return False, f"measured with --backend {config.get('backend')!r}"
    return True, f"{len(items)} items, backend {config.get('backend')}"


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    ok, why = verdict(argv[0])
    print(("done: " if ok else "redo: ") + why)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
