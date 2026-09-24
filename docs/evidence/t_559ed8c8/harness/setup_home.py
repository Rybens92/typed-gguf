"""E2E fixture: build a data home with (a) the real 0.8B GGUF registered and
(b) a real llama.cpp b11026 runtime tree installed as `b11026-linux-x64-cpu`.

Everything here is a *fixture* for the E2E card, built from real parts on this box:
- the runtime tree is the checked-out evidence copy of the real upstream
  `llama-b11026-bin-ubuntu-vulkan-x64.tar.gz` (real libs, real tools);
- the record below is the app's own record shape (`typed_gguf.runtime/v1`), retargeted
  to this home and to the `linux-x64-cpu` variant the update story drives.

Usage:  <venv>/bin/python setup_home.py <home-dir> [--runtime-src DIR] [--model PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import time

from typed_gguf.registry import gguf as gguf_mod
from typed_gguf.registry import store

DEFAULT_RUNTIME_SRC = pathlib.Path(
    "/workspace/ggufone/docs/evidence/t07b5/home/runtime/b11026-linux-x64-vulkan")
DEFAULT_MODEL = pathlib.Path("/var/home/rybens/.cache/llama.cpp/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
TAG = "b11026"
VARIANT = "linux-x64-cpu"
TOOLS = ("llama-cli", "llama-fit-params", "llama-tokenize")


def sha256_file(path: pathlib.Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def copy_tree(source: pathlib.Path, dest: pathlib.Path) -> None:
    """Hardlink where possible (same filesystem), copy otherwise, and stay idempotent."""
    dest.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = dest / item.name
        if item.is_dir():
            copy_tree(item, target)
        elif item.is_symlink():
            if not target.exists() and not target.is_symlink():
                target.symlink_to(os.readlink(item))
        else:
            try:
                os.link(item, target)
            except FileExistsError:
                if target.exists() and os.path.samefile(item, target):
                    continue  # already hardlinked (a re-run)
                shutil.copy2(item, target)
            except OSError:
                shutil.copy2(item, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("home")
    parser.add_argument("--runtime-src", default=str(DEFAULT_RUNTIME_SRC))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--model-alias", default="qwen3.5-0.8b")
    opts = parser.parse_args()

    home = pathlib.Path(opts.home)
    for name in ("models", "downloads", "runtime", "states"):
        (home / name).mkdir(parents=True, exist_ok=True)

    runtime_src = pathlib.Path(opts.runtime_src)
    runtime_dest = home / "runtime" / f"{TAG}-{VARIANT}"
    copy_tree(runtime_src, runtime_dest)
    print(f"runtime tree: {runtime_src} -> {runtime_dest}")

    libllama = runtime_dest / "libllama.so"
    record = {
        "schema": "typed_gguf.runtime/v1",
        "dir": str(runtime_dest),
        "tag": TAG,
        "build": int(TAG[1:]),
        "variant": VARIANT,
        "asset": "llama-b11026-bin-ubuntu-x64.tar.gz",
        "asset_size": 16855810,
        "asset_sha256": None,
        "asset_sha256_observed": None,
        "asset_verified": False,
        "backend_requested": "cpu",
        "backend_working": "cpu",
        "backends": ["cpu", "rpc", "vulkan"],
        "backend_errors": {},
        "fallback_attempts": [],
        "fallback_reason": None,
        "fallback_reason_code": None,
        "fit_params_help_exit": 0,
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "libllama_sha256": sha256_file(libllama),
        "missing_symbols": [],
        "probe_failures": [],
        "probe_warnings": [],
        "required_files": ["libllama.so", "libggml.so", "libggml-base.so"],
        "rung": "prebuilt",
        "source": "e2e-fixture",
        "symbols_ok": True,
        "symbols_probed": True,
        "tools": {name: str(runtime_dest / name) for name in TOOLS},
        "url": "https://github.com/ggml-org/llama.cpp/releases/download/b11026/"
               "llama-b11026-bin-ubuntu-x64.tar.gz",
        "warmup_error": None,
        "warmup_model": None,
        "warmup_ms": None,
    }
    store.runtime_record_path(home).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")

    model = pathlib.Path(opts.model)
    meta = gguf_mod.parse_gguf_metadata(model)
    kv = meta["kv"]
    arch = gguf_mod.arch_of(kv)
    file_type = gguf_mod.file_type_of(kv)
    quant = gguf_mod.quant_label(file_type) if file_type is not None else None
    entry = store.Entry(alias=opts.model_alias, path=str(model), arch=arch, quant=quant,
                        size=model.stat().st_size, file_type=file_type, license="apache-2.0",
                        sha256=sha256_file(model),
                        source="local-file (e2e fixture)")
    store.save_registry(store.Registry(aliases={entry.alias: entry}, current=entry.alias),
                        store.registry_path(home))
    print(f"model: {model.name} arch={arch!r} file_type={file_type} quant={quant!r} "
          f"size={entry.size}")
    print(f"home: {home}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
