#!/usr/bin/env python3
"""Which special token is 248068 / 248069 in Tiel's vocabulary? (read-only, tokenizer only)

Candidates are the chat-template control strings a Qwen-lineage GGUF can carry. Each is
tokenized with the pinned bundle's own `llama-tokenize`, and a single-token result is printed
with its id, so the ids the engine reported at the cue row can be named instead of guessed.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

MODEL = pathlib.Path("/var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf")
BIN = pathlib.Path("/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan/llama-tokenize")
WANTED = {"248068", "248069", "248046"}
CANDIDATES = [
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|start_header_id|>",
    "<|end_header_id|>", "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
    "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>", "<|vision_end|>",
    "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", "<|fim_prefix|>", "<|fim_middle|>",
    "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>", "<tool_call>",
    "</tool_call>", "<|tool_call|>", "<|thinking|>", "thinking", "<|end|>", "<|channel|>",
    "<|constrain|>", "<|start|>", "<|return|>", "<|call|>", "\n",
]


def ids_of(text: str) -> list[int]:
    proc = subprocess.run([str(BIN), "-m", str(MODEL), "-p", text, "--ids"],
                          capture_output=True, text=True)
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip().startswith("[")]
    if not lines:
        return []
    return [int(piece) for piece in lines[-1].strip("[]").split(",") if piece.strip()]


def main() -> int:
    print("text -> ids (single-token candidates only)")
    found = {}
    for text in CANDIDATES:
        ids = ids_of(text)
        if len(ids) == 1:
            found[str(ids[0])] = text
            print(f"   single token {ids[0]:>7}  {text!r}")
    print("\nthe engine's cue ids:")
    for token in sorted(WANTED):
        print(f"   {token}: {found.get(token, 'not among the probed control strings')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
