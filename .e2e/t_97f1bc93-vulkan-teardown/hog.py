#!/usr/bin/env python
"""Hold device memory on the pinned Vulkan bundle — the pressure source for the repro.

Loads a real GGUF through ggufone's own loader (`session.open_model`) with an *exact* layer
count (`degrade=False`: the hog must hold what it was asked to hold), optionally creates a
context too, then sleeps so a sibling process can run against a starved device.

Prints `HOG_LOADED <n_gpu_layers>` / `HOG_CTX` on stdout as readiness markers and copies the
engine's own log lines to stdout, so the caller can read back which devices were touched.
"""
from __future__ import annotations

import argparse
import os
import sys
import time


class ExactPlacement:
    """The minimal placement `fit.coerce_plan` normalizes (`n_gpu_layers` and nothing else)."""

    def __init__(self, n_gpu_layers: int) -> None:
        self.n_gpu_layers = n_gpu_layers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", type=int, default=-1)
    parser.add_argument("--n-ctx", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--hold", type=float, default=900.0)
    args = parser.parse_args()

    from ggufone.engine import session as session_module

    runtime_dir = os.environ["GGUFONE_RUNTIME_DIR"]
    log: list[str] = []
    handle = session_module.open_model(
        args.model, runtime_dir=runtime_dir,
        fit_plan=ExactPlacement(args.layers), degrade=False, log=log)
    for line in log:
        print(line, flush=True)
    print(f"HOG_LOADED layers={handle.n_gpu_layers} placement={handle.placement.note}", flush=True)
    session = None
    if args.n_ctx:
        from ggufone.engine import decide as decide_module
        log.clear()
        session = session_module.ModelSession(
            handle,
            decide_module.ContextPlan(prefix_tokens=(), n_ctx=args.n_ctx, n_seq_max=1,
                                      threads=args.threads, kv_type="auto"),
            log=log)
        for line in log:
            print(line, flush=True)
        print("HOG_CTX", flush=True)
    print(f"HOG_HOLDING pid={os.getpid()} for {args.hold:g}s", flush=True)
    try:
        time.sleep(args.hold)
    except KeyboardInterrupt:
        pass
    if session is not None:
        session.close()
    handle.close()
    print("HOG_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
