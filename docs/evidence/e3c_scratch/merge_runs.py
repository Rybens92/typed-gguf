"""E3c scratch: merge chunked probe runs into one record, then render the tool's own report.

The probe writes one record per invocation, and the campaign is chunked (`run_occamy.sh A|B`), so
the merged record is the unit the evidence doc quotes. The metadata that must be identical across
chunks is asserted, not averaged: a merge that hides a different model, placement or shape set
would publish numbers no single run produced.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import e3c_cue_shapes as probe  # noqa: E402

#: fields a chunk may not disagree on (the model, the box, the measurement itself)
FIXED = ("schema", "model", "model_sha256", "runtime", "threads", "gpu_layers", "backend_claim",
         "shapes", "label_variants", "mass_floor", "ranked_keys", "devset")


def main(paths: list[str], out: str, report: str) -> int:
    records = [json.loads(pathlib.Path(path).read_text(encoding="utf-8")) for path in paths]
    base = records[0]
    for index, record in enumerate(records[1:], start=1):
        for field in FIXED:
            if record[field] != base[field]:
                raise SystemExit(f"chunk {index} disagrees on {field}: "
                                 f"{record[field]!r} != {base[field]!r}")
    merged = dict(base)
    merged["items"] = [item for record in records for item in record["items"]]
    merged["wall_s"] = round(sum(float(record.get("wall_s") or 0.0) for record in records), 1)
    merged["generated_at"] = base["generated_at"]
    counts: dict[str, int] = {}
    for item in merged["items"]:
        counts[item["type"]] = counts.get(item["type"], 0) + 1
    merged["counts"] = counts
    merged["chunks"] = [pathlib.Path(path).name for path in paths]
    pathlib.Path(out).write_text(json.dumps(merged, indent=1), encoding="utf-8")
    markdown = probe.render_report(merged)
    pathlib.Path(report).write_text(markdown, encoding="utf-8")
    print(f"merged {len(records)} chunks, {len(merged['items'])} items -> {out}")
    print(f"report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:-2], sys.argv[-2], sys.argv[-1]))
