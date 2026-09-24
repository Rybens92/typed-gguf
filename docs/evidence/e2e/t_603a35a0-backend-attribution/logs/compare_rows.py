"""Before/after rows of the two lying configurations, straight from the raw reports.

usage: python3 compare_rows.py <before_mixed.raw> <after_mixed.raw> <before_opoffload.raw> <after_opoffload.raw>
"""
import json
import pathlib
import sys


def payload(path: str) -> dict:
    lines = pathlib.Path(path).read_text(errors="replace").splitlines()
    json_at = next(i for i, line in enumerate(lines) if line.strip() == "{" and i > 5)
    joined = "\n".join(lines[json_at:])
    return json.loads(joined[: joined.rfind("}") + 1])


FIELDS = ("backend", "runtime_dir", "placement", "effective_backend", "devices",
          "device_buffers", "warnings")


def rows(payload_: dict) -> list[str]:
    out = []
    for row in payload_.get("backends", []):
        head = f"  {row['backend']:7s} measured={row['measured']}"
        if not row.get("measured"):
            out.append(head + f" reason={row['reason'][:60]}…")
            continue
        speed = (f"prefill={row['prefill_tok_per_s']['p50']:.3f} tok/s "
                 f"decision={row['decision_tok_per_s']['p50']:.3f} tok/s")
        out.append(f"{head} {speed}")
        for field in FIELDS[1:]:
            if field in row:
                out.append(f"      {field}: {json.dumps(row[field])}")
        out.append(f"      placement_used.note: {row['placement_used']['used']['note']!r}")
    out.append(f"  report ok={payload_['ok']}")
    for note in payload_.get("notes", []):
        out.append(f"  note: {note}")
    return out


labels = ["BEFORE mixed (parent tree)", "AFTER mixed (fixed tree)",
          "BEFORE op-offload (parent tree)", "AFTER op-offload (fixed tree)"]
for label, path in zip(labels, sys.argv[1:], strict=True):
    print("=" * 78)
    print(label, "—", pathlib.Path(path).name)
    print("=" * 78)
    print("\n".join(rows(payload(path))))
    print()
