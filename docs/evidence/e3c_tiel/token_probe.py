#!/usr/bin/env python3
"""What are cue tokens 248068 / 248069 in Tiel's vocabulary? Read-only.

Route A: read `tokenizer.ggml.tokens` straight out of the GGUF (no model load, no GPU).
Route B: tokenize the batch's own label strings with the pinned bundle's `llama-tokenize`
         (text -> ids), so the candidate ids are known from the same tokenizer the engine used.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

MODEL = pathlib.Path("/var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf")
BIN = pathlib.Path("/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan/llama-tokenize")
WINDOW = list(range(248060, 248080))


def route_a() -> None:
    try:
        import gguf  # type: ignore
    except Exception as error:                                   # noqa: BLE001
        print(f"route A: gguf import failed ({type(error).__name__}: {error})")
        return
    print(f"route A: gguf {getattr(gguf, '__version__', '?')}")
    reader = gguf.GGUFReader(str(MODEL))
    field = None
    for name in ("tokenizer.ggml.tokens", "tokens"):
        if name in reader.fields:
            field = reader.fields[name]
            break
    if field is None:
        print(f"route A: no tokens field; have {sorted(reader.fields)[:8]}")
        return
    tokens = field.contents() if callable(getattr(field, "contents", None)) else field.data
    print(f"route A: vocab {len(tokens)}")
    for index in WINDOW:
        if 0 <= index < len(tokens):
            value = tokens[index]
            text = value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
            print(f"   {index}: {text!r}")


def route_b() -> None:
    if not BIN.exists():
        print("route B: no llama-tokenize in the bundle")
        return
    for text in ("billing", "technical", "sales", "support", "escalation", "<|im_end|>", "\n"):
        proc = subprocess.run([str(BIN), "-m", str(MODEL), "-p", text, "--ids"],
                              capture_output=True, text=True)
        out = [line for line in proc.stdout.splitlines() if line.strip().startswith("[")]
        print(f"route B: {text!r} -> {out[-1] if out else proc.stdout.strip()[-120:]!r} rc={proc.returncode}")


def main() -> int:
    route_a()
    route_b()
    return 0


if __name__ == "__main__":
    sys.exit(main())
