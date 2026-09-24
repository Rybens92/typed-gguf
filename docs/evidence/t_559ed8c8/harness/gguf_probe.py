"""Probe: what does the app's GGUF reader see in the real 0.8B file?"""
from __future__ import annotations

import pathlib
import sys

from typed_gguf.registry import gguf

path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                   else "/var/home/rybens/.cache/llama.cpp/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
print("file:", path, path.stat().st_size, "bytes")
with open(path, "rb") as handle:
    print("magic:", handle.read(4), "version:", int.from_bytes(handle.read(4), "little"))
try:
    kv = gguf.parse_gguf_metadata(path)
except Exception as exc:  # noqa: BLE001
    print("parse raised:", type(exc).__name__, exc)
    raise SystemExit(1)
print("kv keys:", len(kv))
for key in sorted(kv)[:40]:
    print(f"  {key} = {kv[key]!r}")
print("arch_of:", gguf.arch_of(kv))
print("file_type_of:", gguf.file_type_of(kv))
