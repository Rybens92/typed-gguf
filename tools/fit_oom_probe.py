#!/usr/bin/env python3
"""Drive the REAL loader against a fake OOM bundle (card t_8cb0a05e, requirements 3 + 4 + 5).

The bundle (`tools/fixtures/fit_oom_bundle.c`) is a real shared library that reports the operator's
exact allocation failure through `llama_log_set` and returns NULL while anything is offloaded. This
script therefore exercises production's own path — ctypes dlopen, the arch pre-flight, the log
capture, the classification, the degradation ladder, the error code — with no GPU and no driver.

Two worlds, selected by environment:

    python3 tools/fit_oom_probe.py --model m.gguf --runtime <bundle> [--json OUT]
        -> the plan is forced to full offload; the ladder must land on CPU and succeed
    TYPED_GGUF_FAKE_OOM_ALL=1 python3 tools/fit_oom_probe.py ...
        -> every rung fails; the error must be E_BACKEND_OOM carrying free + needed bytes

Exit code 0 only when the world behaved as required (so the gate can assert on it).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from typed_gguf.engine import session as session_module  # noqa: E402
from typed_gguf.errors import TypedGgufError  # noqa: E402
from typed_gguf.runtime import fit  # noqa: E402

SCHEMA = "typed_gguf.evidence.fit-oom/v1"


def write_synthetic_gguf(path: pathlib.Path, *, arch: str = "spark2_5", n_layer: int = 36,
                         n_kv_head: int = 4, key_len: int = 256, value_len: int = 256,
                         n_ctx_train: int = 32768) -> pathlib.Path:
    """A structurally valid GGUF (header + KV + tensor index + filler).

    Only the header is ever read by the fake bundle, but the file has to satisfy the real readers
    (`ModelFacts.read`, `gguf.arch_of`), so this mirrors the shape tests/test_fit.py builds.
    """
    import struct

    def gstr(value: str) -> bytes:
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw

    def kv_entry(key: str, type_id: int, payload: bytes) -> bytes:
        return gstr(key) + struct.pack("<I", type_id) + payload

    tensors = [("token_embd.weight", [64, 32], 8), ("blk.0.attn_norm.weight", [64], 0)]
    kvs = [kv_entry("general.architecture", 8, gstr(arch)),
           kv_entry(f"{arch}.block_count", 4, struct.pack("<I", n_layer)),
           kv_entry(f"{arch}.attention.head_count_kv", 4, struct.pack("<I", n_kv_head)),
           kv_entry(f"{arch}.attention.key_length", 4, struct.pack("<I", key_len)),
           kv_entry(f"{arch}.attention.value_length", 4, struct.pack("<I", value_len)),
           kv_entry(f"{arch}.context_length", 4, struct.pack("<I", n_ctx_train))]
    blob = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors))
            + struct.pack("<Q", len(kvs)) + b"".join(kvs))
    index = b""
    for name, dims, ttype in tensors:
        index += gstr(name) + struct.pack("<I", len(dims))
        for dim in dims:
            index += struct.pack("<Q", dim)
        index += struct.pack("<I", ttype) + struct.pack("<Q", 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob + index + b"\0" * 4096)
    return path


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--runtime", default=None, help="bundle dir (libllama.so + libggml.so)")
    parser.add_argument("--free-mib", type=int, default=1112)
    parser.add_argument("--total-mib", type=int, default=8192)
    parser.add_argument("--json", default=None, help="write the receipt here")
    parser.add_argument("--make-gguf", default=None,
                        help="write a synthetic GGUF header here and exit (no model needed)")
    args = parser.parse_args(argv)

    if args.make_gguf:
        print(f"wrote {write_synthetic_gguf(pathlib.Path(args.make_gguf))}")
        return 0
    if not args.model or not args.runtime:
        print("error: --model and --runtime are required", file=sys.stderr)
        return 2

    model_path = pathlib.Path(args.model)
    runtime = pathlib.Path(args.runtime)
    facts = fit.ModelFacts.read(model_path, want_sha256=False)
    roomy = fit.HostFacts(backend="vulkan", ram_bytes=31 * 1024 ** 3,
                          vram_bytes=args.total_mib * fit.MIB, n_cpu=os.cpu_count() or 4,
                          fingerprint="probe:roomy")
    plan = fit.estimate_plan(facts, roomy, n_ctx=4096, n_seq_max=8,
                             budget_bytes=args.total_mib * fit.MIB)
    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "model": str(model_path),
        "runtime": str(runtime),
        "free_mib": args.free_mib,
        "total_mib": args.total_mib,
        "world": "oom-all" if os.environ.get("TYPED_GGUF_FAKE_OOM_ALL") else "degrade-to-cpu",
        "plan": plan.to_dict(),
        "library": str(session_module.ctypes_binding.__file__),
    }
    free_bytes = args.free_mib * fit.MIB
    try:
        handle = session_module.open_model(model_path, runtime_dir=runtime, fit_plan=plan,
                                           free_probe=lambda: free_bytes)
    except TypedGgufError as exc:
        receipt["result"] = {"ok": False}
        receipt["error"] = {"code": exc.code, "exit_code": exc.exit_code, "message": str(exc)}
        print(json.dumps(receipt, indent=1))
        _write(args.json, receipt)
        return 0 if (os.environ.get("TYPED_GGUF_FAKE_OOM_ALL")
                     and exc.code == "E_BACKEND_OOM") else 1
    try:
        receipt["result"] = {
            "ok": True,
            "n_gpu_layers": int(handle.n_gpu_layers),
            "placement": handle.placement.to_dict(),
            "warnings": list(handle.warnings),
            "plan": handle.fit_plan.to_dict(),
        }
    finally:
        handle.close()
    print(json.dumps(receipt, indent=1))
    _write(args.json, receipt)
    degraded_ok = handle.placement.degraded and handle.n_gpu_layers == 0
    return 0 if degraded_ok else 1


def _write(path: str | None, payload: dict[str, object]) -> None:
    if not path:
        return
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
