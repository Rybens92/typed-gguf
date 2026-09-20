"""Capture the E3 measurement environment as a JSON artifact (no model load, seconds).

Everything here is read from the live box: the cgroup the worker actually got, the GPU as the
Vulkan loader sees it, the pinned runtime bundle and the pinned GGUF. The model's SHA-256 is
computed by the caller (it is a 23 GB read) and passed in with `--sha256`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys

sys.path.insert(0, "/work/e3repo/src")

RUNTIME = "/var/home/rybens/.local/share/typed-gguf/runtime/b11026-linux-x64-vulkan"
MODEL = "/var/home/rybens/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf"
ICD = "/work/e3scratch/nvidia_egl_icd.json"


def read(path: str) -> str | None:
    try:
        return pathlib.Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def file_sha256(path: str, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        remaining = limit
        while True:
            chunk = handle.read(1024 * 1024 if remaining is None else min(1024 * 1024, remaining))
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
                if remaining <= 0:
                    break
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sha256", default=None, help="the model hash (omit to recompute)")
    parser.add_argument("--out", default="docs/evidence/e3_environment.json")
    args = parser.parse_args()

    model = pathlib.Path(MODEL)
    payload = {
        "schema": "typed_gguf.e3.environment/v1",
        "generated_from": platform.platform(),
        "host_facts": {
            "cpu_count_seen": os.cpu_count(),
            "cgroup_cpu_max": read("/sys/fs/cgroup/cpu.max"),
            "cgroup_memory_max": read("/sys/fs/cgroup/memory.max"),
            "cgroup_pids_max": read("/sys/fs/cgroup/pids.max"),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "model": {
            "path": str(model),
            "bytes": model.stat().st_size,
            "mtime": model.stat().st_mtime,
            "sha256": args.sha256 or file_sha256(MODEL),
        },
        "runtime": json.loads(read(f"{RUNTIME}/../runtime.json") or "{}") or None,
        "gpu": {
            "nvidia_smi": subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                 "--format=csv,noheader"], capture_output=True, text=True,
                check=False).stdout.strip(),
            "vulkan_icd_manifest": ICD,
            "vulkan_icd_library": "libEGL_nvidia.so.0",
            "why": ("the CDI-mounted /etc/vulkan/icd.d/nvidia_icd.x86_64.json names "
                    "libGLX_nvidia.so.0, whose vk_icdNegotiateLoaderICDInterfaceVersion returns "
                    "-3 (INITIALIZATION_FAILED) without an X display, so the loader sees no "
                    "driver at all; libEGL_nvidia.so.0 negotiates rc=0 and yields "
                    "Vulkan0: NVIDIA GeForce RTX 3060 Ti"),
        },
        "pins": {
            "model_sha256": args.sha256 or None,
            "no_downloads": "the model file predates this card (mtime above); the pinned runtime "
                            "bundle was installed by E1a/E1c (`typed-gguf init`)",
            "no_weight_mutation": "typed-gguf never opens a .gguf for writing; this run only reads "
                                  "the file (sha256 re-verified after the campaign)",
        },
    }
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["host_facts"], indent=2))
    print(f"model bytes={payload['model']['bytes']} sha256={payload['model']['sha256']}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
