#!/usr/bin/env python3
"""Live probe: exercise the real bundle / model in a throwaway process.

The engine's C library is loaded through ctypes; several model load/free cycles in one long
lived process can trip shared-library teardown (`free(): invalid pointer` at exit) even though
every call succeeded. Every real CLI run is its own process, so ggufone is not affected — but
tests must not be able to kill the test runner, hence this child-process probe.

Usage:
    python3 tools/live_probe.py bindings           # load the ABI, round-trip a batch
    python3 tools/live_probe.py warmup             # load the pinned model and decode once
    python3 tools/live_probe.py install --home DIR # `init` from the cache + warm-up record
Prints a single JSON object on stdout; exit code 0 == the call worked.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ggufone.registry import store  # noqa: E402
from ggufone.runtime import capability, ctypes_binding, finder, install, pins  # noqa: E402

DEFAULT_HOME = pathlib.Path(os.environ.get("GGUFONE_HOME", pathlib.Path.home() / ".hermes"))
MODEL = DEFAULT_HOME / "models" / "Spark-X2.5-4B-Q8_0.gguf"


def runtime_dir() -> pathlib.Path:
    found = finder.find_runtime()
    if found is None:
        raise SystemExit("no runtime installed (run `ggufone init` or set GGUFONE_RUNTIME_DIR)")
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("bindings", "warmup", "install", "probe"))
    parser.add_argument("--home", default=None)
    parser.add_argument("--model", default=str(MODEL))
    parser.add_argument("--arch", default="spark2_5")
    args = parser.parse_args()
    rt = runtime_dir()
    payload: dict = {"mode": args.mode, "runtime_dir": str(rt)}

    if args.mode == "bindings":
        runtime = ctypes_binding.load_libraries(rt)
        batch = runtime.llama.llama_batch_init(4, 0, 1)
        batch_tokens = batch.n_tokens
        runtime.llama.llama_batch_free(batch)
        payload.update({"bindings": len(runtime.bindings), "batch_n_tokens": batch_tokens,
                        "cached": ctypes_binding.loaded_runtimes()})
    elif args.mode == "probe":
        probe = capability.probe_runtime(rt, deep=True)
        payload.update({"build": probe.build, "backends": list(probe.backends),
                        "missing_symbols": len(probe.missing_symbols),
                        "fit_params_help_exit": probe.fit_params_help_exit,
                        "ok": probe.ok(), "warnings": probe.warnings()})
    elif args.mode == "warmup":
        capability.require_arch(rt, args.arch)
        payload["warmup_ms"] = install.warmup(rt, pathlib.Path(args.model), n_ctx=128)
    elif args.mode == "install":
        home = pathlib.Path(args.home) if args.home else store.data_home()
        result = install.install("cpu", home=home, lock=pins.load_lock(),
                                 warmup_model=pathlib.Path(args.model)
                                 if pathlib.Path(args.model).exists() else None,
                                 free_bytes=1 << 40)
        payload.update({"dir": result["dir"], "source": result["source"],
                        "build": result["build"], "backends": result["backends"],
                        "warmup_ms": result["warmup_ms"],
                        "warmup_error": result["record"]["warmup_error"],
                        "libllama_sha256": result["record"]["libllama_sha256"],
                        "record_path": str(home / "runtime.json")})

    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
