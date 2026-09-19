#!/usr/bin/env python3
"""Card t_635124bf: is llama.cpp's token classification the same one the GGUF itself records?

The fix's rule trusts `llama_token_get_attr` to separate a vocabulary's turn-shaping tokens
(`CONTROL`, `USER_DEFINED`) from its content. That trust is checkable: every GGUF carries its own
classification in `tokenizer.ggml.token_type`, id for id. This reads both and compares **every** id
of the vocabulary — a disagreement means the rule would refuse on the wrong class.

The vocabulary is read with `llama_model_params.vocab_only = true`: llama.cpp reads the KV block and
no tensor, which is the difference between "a 22 GB model needs 22 GB of RAM" and "a 22 GB model
needs its metadata" (Tiel: 1.6 s and no tensor page touched).

    python3 vocab_attr_check.py --model <path.gguf> --runtime <bundle> [--out <json>]
                                [--window 248040:248080]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

HIDDEN = "/nonexistent/no-vulkan-icd.json"
os.environ["VK_DRIVER_FILES"] = HIDDEN
os.environ["VK_ICD_FILENAMES"] = HIDDEN

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ggufone.registry import gguf as gguf_module  # noqa: E402
from ggufone.runtime import ctypes_binding  # noqa: E402

TYPES = {1: "NORMAL", 2: "UNKNOWN", 3: "CONTROL", 4: "USER_DEFINED", 5: "UNUSED", 6: "BYTE"}
#: what the file's own type *means* in llama.cpp's attribute bits
EXPECTED = {"CONTROL": (True, False), "USER_DEFINED": (False, True)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime",
                        default="/var/home/rybens/.local/share/ggufone/runtime/"
                                "b11026-linux-x64-vulkan")
    parser.add_argument("--out", default=None)
    parser.add_argument("--window", default="", help="id window to print, e.g. 248040:248080")
    args = parser.parse_args()
    model_path = pathlib.Path(args.model)
    runtime = ctypes_binding.load_libraries(args.runtime)
    llama = runtime.llama
    started = time.perf_counter()
    params = llama.llama_model_default_params()
    params.n_gpu_layers = 0
    params.vocab_only = True
    model = llama.llama_model_load_from_file(str(model_path).encode(), params)
    load_s = time.perf_counter() - started
    if not model:
        print("E_RUNTIME_SYMBOLS: the vocab-only load returned NULL")
        return 1
    vocab = llama.llama_model_get_vocab(model)
    n_vocab = int(llama.llama_vocab_n_tokens(vocab))
    types = [int(value) for value in
             gguf_module.parse_gguf_metadata(str(model_path))["kv"]["tokenizer.ggml.token_type"]]
    if len(types) != n_vocab:
        print(f"the file records {len(types)} types for {n_vocab} ids — aborting")
        return 1
    started = time.perf_counter()
    specials = ctypes_binding.special_tokens(runtime, vocab, n_vocab)
    scan_s = time.perf_counter() - started
    started = time.perf_counter()
    mismatches = []
    for token in range(n_vocab):
        attr = ctypes_binding.token_attr(runtime, vocab, token)
        control = bool(attr & ctypes_binding.TOKEN_ATTR_CONTROL)
        user = bool(attr & ctypes_binding.TOKEN_ATTR_USER_DEFINED)
        expected = EXPECTED.get(TYPES.get(types[token], "?"), (False, False))
        if (control, user) != expected:
            mismatches.append({"token": token, "gguf_type": TYPES.get(types[token], "?"),
                               "control": control, "user_defined": user, "attr": attr})
    compare_s = time.perf_counter() - started
    print(f"model      {model_path.name} ({model_path.stat().st_size:,} bytes)")
    print(f"load       vocab-only {load_s:.2f} s · vocab {n_vocab} ids")
    print(f"scan       specials() {scan_s * 1000:.0f} ms · every-id comparison "
          f"{compare_s * 1000:.0f} ms ({1e6 * compare_s / n_vocab:.2f} µs/id)")
    print(f"specials   {len(specials)} CONTROL / USER_DEFINED ids · "
          f"gguf-vs-attr disagreements over all {n_vocab} ids: {len(mismatches)}")
    for row in mismatches[:10]:
        print(f"   {row}")
    low, _, high = args.window.partition(":")
    window = range(int(low), int(high or low)) if args.window else None
    for token, text in specials:
        if window is None or token in window:
            print(f"   {token:>7}  {TYPES.get(types[token], '?'):<12} {text!r}")
    llama.llama_model_free(model)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps({
            "model": str(model_path), "bytes": model_path.stat().st_size,
            "runtime": args.runtime, "n_vocab": n_vocab, "load_s": round(load_s, 3),
            "scan_ms": round(scan_s * 1000, 1), "compare_ms": round(compare_s * 1000, 1),
            "specials": [{"token": token, "gguf_type": TYPES.get(types[token], "?"), "text": text}
                         for token, text in specials],
            "disagreements": mismatches,
        }, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
