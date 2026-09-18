#!/usr/bin/env python3
"""A-E1b-14 perf record (report-only): prefill tok/s CPU + warm 4-candidate choice ms.

Run on the box that has the pinned runtime and the models:

    GGUFONE_RUNTIME_DIR=<bundle> HOME=$HOME python3 tools/e1b_perf_record.py [--json out.json]

Nothing here is a gate (SPEC 5 / A-E1b-14: "correctness before speed"). It publishes this
milestone's own measured numbers next to the recon reference point (14-20 ms warm choice on
Vulkan, `[recon]`). CPU only: E1b has no fit plan yet (E1c owns `fit`), so the numbers are the
floor, not the ceiling.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ggufone import schema  # noqa: E402
from ggufone.engine import decide  # noqa: E402
from ggufone.engine import session as session_module  # noqa: E402

MODELS = {
    "qwen35": pathlib.Path.home() / ".cache" / "llama.cpp" / "Qwen3.5-0.8B-UD-Q4_K_XL.gguf",
    "spark2_5": pathlib.Path.home() / ".hermes" / "models" / "Spark-X2.5-4B-Q8_0.gguf",
}
CHOICE = {
    "state": "The billing dashboard is blank for every user after login; reports are stale.",
    "questions": {"area": {"type": "choice", "instructions": "Which team owns this?",
                           "criteria": {"billing": None, "technical": None,
                                        "platform": None, "support": None}}},
}


def runtime_dir() -> pathlib.Path:
    env = os.environ.get("GGUFONE_RUNTIME_DIR")
    if env:
        return pathlib.Path(env)
    for base in (pathlib.Path.home() / ".hermes" / "runtime",
                 pathlib.Path.home() / ".local" / "share" / "ggufone" / "runtime"):
        for candidate in sorted(base.glob("*/")):
            if (candidate / "libllama.so").exists():
                return candidate
    raise SystemExit("no llama.cpp runtime found; set GGUFONE_RUNTIME_DIR")


def measure(name: str, path: pathlib.Path, *, threads: int, repeats: int,
            runtime: pathlib.Path) -> dict:
    if not path.exists():
        return {"model": name, "skipped": f"{path} is not on this box"}
    import tempfile
    states_home = pathlib.Path(tempfile.mkdtemp(prefix=f"e1b-perf-{name}-"))
    payload = {**CHOICE, "options": {"threads": threads, "save_state": True,
                                     "state_id": f"perf-{name}"}}
    request = schema.parse_request(payload)
    handle = session_module.open_model(path, runtime_dir=runtime)
    try:
        plan = decide.plan_context(request, handle)

        def one_run() -> tuple[Any, float]:
            started = time.perf_counter()
            with session_module.ModelSession(handle, plan, states_home=states_home) as live:
                result = decide.DecisionEngine(live).decide(request, plan=plan)
            return result, (time.perf_counter() - started) * 1000.0

        cold, cold_ms = one_run()                    # fills the state cache
        warm = [one_run() for _ in range(repeats)]
        prefix = cold.engine["prefix_tokens"]
        warm_wall = [wall for _, wall in warm]
        warm_prefill = [result.timings["prefill_ms"] for result, _ in warm]
        warm_questions = [result.timings["questions_ms"] for result, _ in warm]
        return {
            "model": name,
            "path": str(path),
            "threads": threads,
            "prefix_tokens": prefix,
            "cold_prefill_ms": round(cold.timings["prefill_ms"], 3),
            "cold_prefill_tok_per_s": round(
                prefix / max(cold.timings["prefill_ms"], 1e-9) * 1000.0, 1),
            "cold_wall_ms": round(cold_ms, 1),
            "warm_prefill_ms": round(statistics.median(warm_prefill), 3),
            "warm_prefill_reused": all(result.engine["prefill_reused"] for result, _ in warm),
            "warm_questions_ms": round(statistics.median(warm_questions), 1),
            "warm_wall_ms": round(statistics.median(warm_wall), 1),
            "warm_wall_ms_all": [round(value, 1) for value in warm_wall],
            "note": "4 candidates, 1 question, CPU (n_gpu_layers=0); the recon reference point "
                    "is 14-20 ms warm on Vulkan with a warm state cache [recon]",
            "answers": cold.answers,
        }
    finally:
        handle.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="docs/evidence/e1b_perf.json")
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--threads-determinism", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--repeat-determinism", type=int, default=1)
    args = parser.parse_args()
    runtime = runtime_dir()
    record = {
        "schema": "ggufone.evidence.e1b-perf/v1",
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_dir": str(runtime),
        "cpu_count": os.cpu_count(),
        "gates": "report-only (A-E1b-14)",
        "rows": [measure(name, path, threads=args.threads, repeats=args.repeats,
                         runtime=runtime)
                 for name, path in MODELS.items()],
    }
    out = pathlib.Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
