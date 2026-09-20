#!/usr/bin/env python3
"""A-E1a-8 / A-E1a-12: measure the real KV footprint and the CPU baseline numbers.

Report-only: the SPEC keeps the recommender conservative until E2.5 has the measurement
(S-4). Run:

    python3 tools/measure_kv_footprint.py --json docs/evidence/e1a_kv_footprint.json

What it does, with the pinned bundle through ctypes (the same binding the engine will use):
  1. load the model once, record RSS;
  2. init a context at n_ctx with n_seq_max in {1, 4} (kv_unified=True, the mandatory
     setting), record RSS after each and the per-sequence state size
     (llama_state_seq_get_size);
  3. prefill a text on seq 0 and record prefill tok/s — once with threads=1 (the determinism
     mode) and once with all cores (the baseline mode);
  4. time a warm single-token decode (a readout-shaped step: prefill done, one more token);
  5. compare the measured footprint with the conservative bound
     (n_ctx * n_seq_max * kv_bytes_per_token) and print a JSON report.

Everything is measured on the *stock* pinned GGUF; nothing is trained or converted.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from typed_gguf.registry.gguf import parse_gguf_metadata  # noqa: E402
from typed_gguf.registry.recommend import kv_bytes_per_token  # noqa: E402
from typed_gguf.runtime import capability, ctypes_binding  # noqa: E402

PREFILL_TEXT = (
    "Incident report: the production dashboard shows a blank page for every user after login. "
    "It started ten minutes ago, there is no workaround, and the on-call engineer is paged. "
    "The support queue is filling up and the customer success team wants an update. "
) * 4


def rss_bytes() -> int:
    for line in pathlib.Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def header_kv_per_token(meta: dict) -> int:
    arch = meta["general.architecture"]
    layer = meta[f"{arch}.block_count"]
    kv_head = meta.get(f"{arch}.attention.head_count_kv", meta.get(f"{arch}.attention.head_count"))
    key_len = meta.get(f"{arch}.attention.key_length", 0)
    value_len = meta.get(f"{arch}.attention.value_length", 0)
    if not key_len or not value_len:
        dim = meta[f"{arch}.embedding_length"] // meta[f"{arch}.attention.head_count"]
        key_len, value_len = key_len or dim, value_len or dim
    return kv_bytes_per_token(layer, kv_head, key_len, value_len, 2)


def prefill(runtime, llama, ctx, tokens, *, repeat: int = 1) -> tuple[float, float]:
    """Decode `tokens` on seq 0 `repeat` times; return (last_ms, tokens_per_s) of that pass."""
    array = (ctypes_binding.llama_token * len(tokens))(*tokens)
    elapsed = 0.0
    for _ in range(repeat):
        started = time.perf_counter()
        batch = llama.llama_batch_get_one(array, len(tokens))
        rc = llama.llama_decode(ctx, batch)
        llama.llama_synchronize(ctx)
        elapsed = time.perf_counter() - started
        if rc != 0:
            raise SystemExit(f"prefill decode rc={rc}")
    return elapsed * 1000.0, len(tokens) / elapsed


def warm_single_token(llama, ctx, token_id: int, *, repeats: int = 5) -> float:
    """Median ms of one warm single-token decode — the readout's marginal cost shape."""
    samples = []
    for i in range(repeats):
        array = (ctypes_binding.llama_token * 1)(token_id)
        batch = llama.llama_batch_get_one(array, 1)
        started = time.perf_counter()
        rc = llama.llama_decode(ctx, batch)
        llama.llama_synchronize(ctx)
        if rc != 0:
            raise SystemExit(f"warm decode rc={rc}")
        samples.append((time.perf_counter() - started) * 1000.0)
        assert i >= 0
    return statistics.median(samples)


def measure(model_path: pathlib.Path, runtime_dir: pathlib.Path, n_ctx: int,
            seq_counts: list[int], tokens_wanted: int) -> dict:
    runtime = ctypes_binding.load_libraries(runtime_dir)
    llama = runtime.llama
    probe = capability.probe_runtime(runtime_dir, deep=True)
    meta = parse_gguf_metadata(model_path)["kv"]
    kvpt = header_kv_per_token(meta)
    cores = os.cpu_count() or 1
    swa = meta.get(f"{meta.get('general.architecture')}.attention.sliding_window")
    report: dict = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": ("ctypes on the pinned bundle; KV = VmRSS delta across llama_init_from_model "
                   "(kv_unified=True); prefill = one llama_decode of the shared prefix; warm "
                   "decode = median ms of 5 single-token decodes after the prefill"),
        "model": str(model_path),
        "model_bytes": model_path.stat().st_size,
        "arch": meta.get("general.architecture"),
        "file_type": meta.get("general.file_type"),
        "sliding_window": swa,
        "sliding_window_layers": sum(1 for v in meta.get(
            f"{meta.get('general.architecture')}.attention.sliding_window_pattern", []) if v),
        "runtime_dir": str(runtime_dir),
        "build": probe.build,
        "backends": list(probe.backends),
        "cores": cores,
        "n_ctx": n_ctx,
        "kv_bytes_per_token_f16_from_header": kvpt,
        "rss_before_model_load": rss_bytes(),
        "contexts": [],
    }

    params = llama.llama_model_default_params()
    started = time.perf_counter()
    model = llama.llama_model_load_from_file(str(model_path).encode(), params)
    report["model_load_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    if not model:
        raise SystemExit("model load failed")
    report["rss_after_model_load"] = rss_bytes()
    vocab = llama.llama_model_get_vocab(model)
    tokens = ctypes_binding.tokenize(runtime, vocab, PREFILL_TEXT)[:tokens_wanted]
    report["prefix_tokens"] = len(tokens)

    for n_seq in seq_counts:
        entry: dict = {"n_seq_max": n_seq, "kv_unified": True}
        for threads in (1, cores):
            rss0 = rss_bytes()
            ctx_params = llama.llama_context_default_params()
            ctx_params.n_ctx = n_ctx
            ctx_params.n_batch = 512
            ctx_params.n_ubatch = 512
            ctx_params.n_seq_max = n_seq
            ctx_params.n_threads = threads
            ctx_params.n_threads_batch = threads
            ctx_params.kv_unified = True          # PoC pitfall 2: mandatory
            ctx_params.no_perf = False
            started = time.perf_counter()
            ctx = llama.llama_init_from_model(model, ctx_params)
            ctx_ms = (time.perf_counter() - started) * 1000.0
            if not ctx:
                raise SystemExit("context init failed")
            rss_ctx = rss_bytes()
            first_ms, first_tps = prefill(runtime, llama, ctx, tokens)
            warm_ms, warm_tps = prefill(runtime, llama, ctx, tokens, repeat=1)  # same prefix again
            one_token_ms = warm_single_token(llama, ctx, tokens[-1])
            rss_after = rss_bytes()
            state_size = llama.llama_state_seq_get_size(ctx, 0)
            conservative = kvpt * n_ctx * n_seq
            entry[f"threads_{threads}"] = {
                "ctx_init_ms": round(ctx_ms, 1),
                "prefill_tokens": len(tokens),
                "prefill_ms": round(first_ms, 1),
                "prefill_tokens_per_s": round(first_tps, 1),
                "prefill_repeat_ms": round(warm_ms, 1),
                "prefill_repeat_tokens_per_s": round(warm_tps, 1),
                "warm_single_token_decode_ms": round(one_token_ms, 2),
                "rss_delta_ctx_bytes": max(0, rss_ctx - rss0),
                "rss_delta_total_bytes": max(0, rss_after - rss0),
                "seq_state_size_bytes": state_size,
                "n_ctx_reported_by_llama": llama.llama_n_ctx(ctx),
            }
            llama.llama_free(ctx)
            if threads == 1:
                entry["conservative_bound_bytes"] = conservative
                entry["conservative_bound_bytes_per_seq"] = kvpt * n_ctx
                entry["measured_kv_over_conservative_bound"] = round(
                    entry["threads_1"]["rss_delta_ctx_bytes"] / conservative, 4)
                entry["measured_vs_unified_theory"] = round(
                    entry["threads_1"]["rss_delta_ctx_bytes"] / (kvpt * n_ctx), 4)
        report["contexts"].append(entry)

    llama.llama_model_free(model)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    default_home = pathlib.Path(os.environ.get("TYPED_GGUF_HOME", pathlib.Path.home() / ".hermes"))
    parser.add_argument("--model", default=str(default_home / "models" / "Spark-X2.5-4B-Q8_0.gguf"))
    parser.add_argument("--runtime", default=str(default_home / "runtime" / "b11026-linux-x64-cpu"))
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--seq-counts", default="1,4")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    model_path = pathlib.Path(args.model)
    runtime_dir = pathlib.Path(args.runtime)
    if not model_path.exists():
        print(f"model not found: {model_path}", file=sys.stderr)
        return 3
    if not (runtime_dir / "libllama.so").exists():
        print(f"runtime not found: {runtime_dir}", file=sys.stderr)
        return 3
    report = measure(model_path, runtime_dir, args.n_ctx,
                     [int(s) for s in args.seq_counts.split(",")], args.tokens)
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.json:
        pathlib.Path(args.json).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
