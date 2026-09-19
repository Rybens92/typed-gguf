#!/usr/bin/env python3
"""t_7c926398 (auxiliary) — the same Tiel campaign with `--cue <shape>`.

`tools/e3c_tiel_reproduce.py` hard-codes the shipped cue (it builds `harness.BenchConfig`
without `cue=`), and this card must not rewrite that committed driver. This wrapper is the
committed driver's own factory/observer + the one extra flag:

    python3 .t7c9/tiel_cue_arm.py --suite quality --cue two_step --model <tiel.gguf> \\
        --backend vulkan --gpu-layers 9 --threads 4 \\
        --devset <chunk>.jsonl --out <report>.json --placement-out <sink>.json

The card's required row stays the shipped cue (`.t7c9/run_chunks_corrected.sh`); this arm exists
only to answer "is the collapse the *cue row*?" with a measurement instead of a suggestion, and it
is published as auxiliary, never as the card's row.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from e3c_tiel_reproduce import factory_with_sink  # noqa: E402  (the committed observer)

from ggufone.bench import harness, suites  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", default="quality")
    parser.add_argument("--cue", default="shipped")
    parser.add_argument("--model", default=None)
    parser.add_argument("--backend", default="vulkan")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--devset", default=None)
    parser.add_argument("--items", type=int, default=None)
    parser.add_argument("--n-seq-max", dest="n_seq_max", type=int, default=None)
    parser.add_argument("--kv-type", dest="kv_type", default="auto")
    parser.add_argument("--gpu-layers", dest="gpu_layers", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--placement-out", dest="placement_out", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = harness.BenchConfig(
        suite=args.suite, model_path=harness.resolve_model_path(args.model), backend=args.backend,
        runs=args.runs, threads=args.threads, devset=args.devset, items=args.items,
        n_seq_max=args.n_seq_max, kv_type=args.kv_type, gpu_layers=args.gpu_layers,
        cue=args.cue)
    sink: dict[str, object] = {"suite": args.suite, "cue": args.cue,
                               "model": str(config.model_path), "backend": args.backend,
                               "threads": args.threads, "gpu_layers_requested": args.gpu_layers,
                               "devset": args.devset,
                               "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    started = time.perf_counter()
    report = suites.run_suite(config, factory=factory_with_sink(sink))
    sink["wall_s"] = round(time.perf_counter() - started, 1)
    sink["placements_seen"] = None
    sink["schema"] = "ggufone.t7c926398.cue_arm/v1"
    if args.out:
        harness.write_report(report, args.out)
        print(f"report: {args.out}")
    if args.placement_out:
        pathlib.Path(args.placement_out).write_text(json.dumps(sink, indent=1, sort_keys=True),
                                                    encoding="utf-8")
        print(f"placement: {args.placement_out}")
    if not args.quiet:
        print(harness.render_report(report))
    return 0 if report.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
