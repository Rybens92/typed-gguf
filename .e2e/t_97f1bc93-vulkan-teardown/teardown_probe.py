#!/usr/bin/env python
"""Teardown probe for card t_97f1bc93: the same live bench run, ended four different ways.

`--variant none` is the production shape (`cli._cmd_bench`): run the suite, print the report,
return. The other variants end the process deliberately and each one answers a different
question:

* `backend_free`  — does the bundle's own documented teardown (`llama_backend_free`) avoid the
                    crash, i.e. is a *clean* engine shutdown enough?
* `exit_fast`     — is the crash after the report, in the C library destructors the interpreter's
                    own shutdown triggers (libggml-vulkan's globals, the ICD's unload)?
* `close_then_free` — both, in the order the bundle's docs imply: free every context/model first,
                    then the backend, then exit.

The exit status is the only thing this probe is about, so the report is always printed (and
optionally written) before the variant runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="none",
                        choices=["none", "backend_free", "exit_fast", "close_then_free"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--sizes", default="64")
    parser.add_argument("--out")
    args = parser.parse_args()

    from ggufone.bench import harness, suites

    config = harness.BenchConfig(suite="throughput", model_path=args.model, backend="vulkan",
                                 runs=1, threads=args.threads,
                                 prefill_sizes=tuple(int(part) for part in args.sizes.split(",")))
    report = suites.run_suite(config, factory=suites.live_factory)
    text = json.dumps(report, indent=2)
    if args.out:
        harness.write_report(report, args.out)
    print(text, flush=True)
    code = 0 if report.get("ok", True) else 1
    print(f"PROBE variant={args.variant} report_ok={report.get('ok', True)}", file=sys.stderr,
          flush=True)

    if args.variant in ("backend_free", "close_then_free"):
        from ggufone.runtime import ctypes_binding
        for directory in ctypes_binding.loaded_runtimes():
            runtime = ctypes_binding.load_libraries(directory, load_backends=False)
            runtime.free()
        print("PROBE llama_backend_free done", file=sys.stderr, flush=True)
    if args.variant == "exit_fast":
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(code)
    if args.variant == "none":
        # Where in the shutdown the process got to before it died: the report is printed, then
        # this marker fires from the interpreter's own exit handler, then the C library
        # destructors run (`_dl_fini`). A crash after `atexit` is the libraries', not the
        # interpreter's — the marker is the difference the backtrace alone cannot show.
        import atexit

        def _mark() -> None:
            print("PROBE stage=atexit-reached", file=sys.stderr, flush=True)

        atexit.register(_mark)
        print("PROBE stage=report-printed", file=sys.stderr, flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
