"""One bench report, printed as evidence: `run_suite(determinism, backend="cpu")` + the row.

    GGUFONE_RUNTIME_DIR=<bundle> VK_DRIVER_FILES=<icd> uv run python row_probe.py <model.gguf>

Card t_55de5779: the live gate is `tests/test_bench_live.py::test_determinism_holds_on_this_box`;
this probe prints the *row* that gate asserts on — placement, effective backend, per-device
compute-buffer counts, warnings and the digests — so the RED and the GREEN run can be quoted.
"""
from __future__ import annotations

import json
import os
import sys

from ggufone.bench import harness, suites

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.environ["GGUFONE_BENCH_MODEL"]


def main() -> int:
    report = suites.run_suite(
        harness.BenchConfig(suite="determinism", model_path=MODEL, backend="cpu", threads=1),
        factory=suites.live_factory)
    row = report["backends"][0]
    payload = {
        "ok": report["ok"],
        "notes": report["notes"],
        "repeats": report["repeats"],
        "config_backend": report["config"]["backend"],
        "runtime_dir": row.get("runtime_dir"),
        "row": {key: row.get(key) for key in
                ("backend", "placement", "effective_backend", "devices", "device_buffers",
                 "warnings", "identical", "ok", "digests", "repeats", "reason")},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
