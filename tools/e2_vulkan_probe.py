#!/usr/bin/env python3
"""A-E2-6: does a fresh process pay Vulkan pipeline compilation, and does a persistent driver
cache amortise it across processes?

This box has no GPU, but it has a *software* Vulkan device (Mesa `lavapipe`) and can be pointed
at the pinned Vulkan bundle. That is enough to exercise the **mechanism** — pipeline creation on
the first decode in a process, reuse afterwards, and a driver cache on disk across processes —
and *not* enough to reproduce a GPU's absolute shader-compile cost. Every number this prints is
tagged `[executed: lavapipe]` in `docs/BENCHMARKS.md` §3.6.

    TYPED_GGUF_VULKAN_RUNTIME_DIR=<bundle> python3 tools/e2_vulkan_probe.py
    TYPED_GGUF_VULKAN_MODEL=<path.gguf> TYPED_GGUF_VULKAN_RUNTIME_DIR=<bundle> \
        python3 tools/e2_vulkan_probe.py

The driver runs three fresh *processes*, so "first decode in a fresh process" is measured the way
a one-shot CLI would pay it; the JSON report goes to `--out`.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SCHEMA = "typed_gguf.bench.vulkan-probe/v1"
DEFAULT_RUNTIMES = (
    "/work/e1a/home/runtime/b11026-linux-x64-vulkan",
    "/work/tf363/home-rtx/runtime/b11026-linux-x64-vulkan",
)
DEFAULT_MODEL = "~/.cache/llama.cpp/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"
PREFIX_TOKENS = 64
WARM_REPEATS = 3


def runtime_dir(explicit: str | None) -> pathlib.Path:
    candidates = [explicit, os.environ.get("TYPED_GGUF_VULKAN_RUNTIME_DIR"), *DEFAULT_RUNTIMES]
    for candidate in candidates:
        if not candidate:
            continue
        path = pathlib.Path(os.path.expanduser(candidate))
        if (path / "libllama.so").exists() and (path / "libggml-vulkan.so").exists():
            return path
    raise SystemExit("no local Vulkan bundle found; set TYPED_GGUF_VULKAN_RUNTIME_DIR")


def model_path(explicit: str | None) -> pathlib.Path:
    path = pathlib.Path(os.path.expanduser(explicit or os.environ.get("TYPED_GGUF_VULKAN_MODEL")
                                          or DEFAULT_MODEL))
    if not path.exists():
        raise SystemExit(f"{path} is not on this box; set TYPED_GGUF_VULKAN_MODEL")
    return path


def shader_cache_dirs() -> tuple[pathlib.Path, ...]:
    return (pathlib.Path(os.environ.get("MESA_SHADER_CACHE_DIR",
                                        pathlib.Path.home() / ".cache" / "mesa_shader_cache")),
            pathlib.Path.home() / ".cache" / "mesa_shader_cache",
            pathlib.Path.home() / ".nv" / "ComputeCache")


def cache_state() -> dict[str, int]:
    state: dict[str, int] = {}
    for directory in shader_cache_dirs():
        if directory.is_dir():
            state[str(directory)] = sum(path.stat().st_size for path in directory.rglob("*")
                                        if path.is_file())
    return state


def run_once(runtime: pathlib.Path, model: pathlib.Path) -> dict:
    """One fresh process: load, then the first decode (pipeline creation) vs warm decodes."""
    from typed_gguf.bench import harness
    from typed_gguf.engine import decide
    from typed_gguf.engine import session as session_module

    started = time.perf_counter()
    handle = session_module.open_model(model, runtime_dir=runtime, fit_plan=harness.Placement(0))
    load_ms = (time.perf_counter() - started) * 1000.0
    try:
        tokens = list(handle.tokenize(" ".join(f"event{index % 40}" for index in range(120))))
        tokens = (tokens * 4)[:PREFIX_TOKENS]
        plan = decide.ContextPlan(prefix_tokens=(), n_ctx=PREFIX_TOKENS + 64, n_seq_max=3,
                                  threads=2, kv_type="auto")
        with session_module.ModelSession(handle, plan, backend="vulkan",
                                         states_home=None) as live:
            first = time.perf_counter()
            live.prefill(tokens, state_cache=False)
            first_ms = (time.perf_counter() - first) * 1000.0
            warm: list[float] = []
            for _ in range(WARM_REPEATS):
                warm_started = time.perf_counter()
                live.prefill(tokens, state_cache=False)
                warm.append((time.perf_counter() - warm_started) * 1000.0)
    finally:
        handle.close()
    return {"load_ms": load_ms, "first_decode_ms": first_ms,
            "warm_decode_ms": sum(warm) / len(warm), "prefix_tokens": PREFIX_TOKENS,
            "backend": "vulkan", "runtime": str(runtime), "model": str(model),
            "cache": cache_state()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--runtime", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", default="docs/evidence/e2_vulkan_probe.json")
    args = parser.parse_args(argv)
    runtime = runtime_dir(args.runtime)
    model = model_path(args.model)
    if args.once:
        print(json.dumps(run_once(runtime, model)))
        return 0
    before = cache_state()
    runs: list[dict] = []
    for index in range(1, int(args.runs) + 1):
        completed = subprocess.run([sys.executable, __file__, "--once", "--runtime", str(runtime),
                                    "--model", str(model)],
                                   capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            runs.append({"label": f"process {index}",
                         "error": (completed.stderr or completed.stdout)[-800:]})
            continue
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        payload["label"] = f"fresh process {index}"
        runs.append(payload)
    after = cache_state()
    report = {
        "schema": SCHEMA,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tag": "[executed: lavapipe]",
        "cache_before": before,
        "cache_after": after,
        "runs": runs,
        "note": ("Software Vulkan (Mesa lavapipe) on a container with no GPU: this measures the "
                 "*mechanism* (first-decode pipeline creation inside a process, reuse afterwards, "
                 "and whether the driver's on-disk cache grows/carries across processes), not a "
                 "GPU's absolute shader-compile time."),
    }
    target = pathlib.Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for row in runs:
        print(json.dumps(row))
    print(f"shader cache before={before} after={after}")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
