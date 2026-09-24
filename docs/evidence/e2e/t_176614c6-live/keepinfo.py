"""The `engine.keep` block of an `ask --out` answer, plus the run's exit/wall receipts.

Why: the (a) receipt turns on *how* an answer was served — `served_by: host` (the warm host
answered) vs `inline` (the client fell back after its host died), and on which events the client
logged. `show.py` prints the placement; this prints the keep half.
"""
import json
import pathlib
import sys

BASE = pathlib.Path("/work/t176614c6-live/out")


def main(names: list[str]) -> int:
    for arg in names:
        label, _, name = arg.partition("=")
        name = name or label
        path = BASE / f"{name}.json"
        if not path.exists():
            print(f"{label:<26} MISSING ({path.name})")
            continue
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:                     # a half-written answer is a receipt too
            print(f"{label:<26} unreadable: {exc.__class__.__name__}: {exc}")
            continue
        keep = (body.get("engine") or {}).get("keep") or {}
        def sidecar(suffix: str) -> str:
            f = BASE / f"{name}{suffix}"
            return f.read_text(encoding="utf-8").strip() if f.exists() else "-"
        events = keep.get("events")
        print(f"{label:<26} exit={sidecar('.exit'):<3} wall_ms={sidecar('.wall_ms'):<7} "
              f"served_by={keep.get('served_by')!r} reused={keep.get('reused')!r} "
              f"host_pid={keep.get('host_pid')} keep_alive_s={keep.get('keep_alive_s')}")
        print(f"{'':<26} events={events}")
        answer = (body.get("answer") or {}).get("decision") or (body.get("answer") or {}).get("text")
        if isinstance(answer, str):
            print(f"{'':<26} answer[:90]={answer[:90]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
