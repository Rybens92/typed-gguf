"""Probe: where the quick latency suite's wall time goes on the real runtime (Qwen, threads=2).

The suite's row p50s sum to far less than `report["wall_ms"]`; this separates the two candidates —
llama.cpp context setup per call vs the decode itself — so the card's evidence can say which.

    python3 tools/probe_quick_cost.py <model.gguf> <runtime-dir> [threads]
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ggufone import schema  # noqa: E402
from ggufone.bench import harness, suites  # noqa: E402


def timed(label: str, fn):
    started = time.perf_counter()
    value = fn()
    print(f"{label:<34s} {(time.perf_counter() - started) * 1000.0:9.1f} ms")
    return value


def main(argv: list[str]) -> int:
    model_path, runtime = argv[1], argv[2]
    threads = int(argv[3]) if len(argv) > 3 else 2
    spec = harness.ModelSpec(path=model_path, backend=harness.CPU_BACKEND, runtime_dir=runtime,
                            threads=threads)
    model = harness.LiveModel(spec)
    try:
        timed("model load", model.load)
        request = schema.parse_request(suites._choice_request(4, threads=threads))
        timed("decide #1 (context + prefill)", lambda: model.decide(request, n_seq_max=5,
                                                                   threads=threads))
        timed("decide #2 (warm prefix)", lambda: model.decide(request, n_seq_max=5,
                                                             threads=threads))
        handle = model.handle

        def one_session():
            with model.session(n_ctx=1024, n_seq_max=5, threads=threads):
                pass

        timed("one ModelSession (empty)", one_session)
        timed("one ModelSession (again)", one_session)
        timed("tokenize 256 filler words", lambda: model.tokenize(" ".join(
            f"event{i} code{i}" for i in range(256))))
        print(f"runtime: {runtime}\nmodel:   {model_path}\nthreads: {threads}\n"
              f"handle:  {handle.placement.to_dict()}\n")
    finally:
        model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
