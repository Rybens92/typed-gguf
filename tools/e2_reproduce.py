#!/usr/bin/env python3
"""Reproduce a published `docs/BENCHMARKS.md` table — one command per table (SPEC 5 / E2).

Every table in `docs/BENCHMARKS.md` names the exact invocation that produced it; this is that
invocation, in a form that works from a checkout (no install needed: `src/` goes on `sys.path`).

    # one suite, table on stdout, JSON report written next to it
    GGUFONE_RUNTIME_DIR=<bundle> python3 tools/e2_reproduce.py --suite latency \
        --model ~/.hermes/models/Spark-X2.5-4B-Q8_0.gguf --threads 4 --runs 5 \
        --out docs/evidence/e2_latency.json

    # everything (the five suites, same model/backends)
    GGUFONE_RUNTIME_DIR=<bundle> python3 tools/e2_reproduce.py --suite all \
        --model <path.gguf> --threads 4 --out-dir docs/evidence

The suites never touch the model registry and never open a socket (A-E2-7): `--model` is a path
on disk, the runtime is a local bundle and the dev set ships inside the package.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ggufone.bench import harness, suites  # noqa: E402


def build_config(args: argparse.Namespace, suite: str) -> harness.BenchConfig:
    return harness.BenchConfig(
        suite=suite,
        model_path=harness.resolve_model_path(args.model),
        backend=args.backend,
        runs=args.runs,
        threads=args.threads,
        devset=args.devset,
        items=args.items,
        n_seq_max=args.n_seq_max,
        kv_type=args.kv_type,
        gpu_layers=args.gpu_layers,
        n_bins=args.n_bins,
        prefill_sizes=parse_sizes(args.sizes))


def parse_sizes(value: str | None) -> tuple[int, ...]:
    """`--sizes 256,2048` -> the prefill sizes of this run (the CLI's parser, kept in sync)."""
    if not value:
        return harness.PREFILL_SIZES
    sizes: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit() or int(part) <= 0:
            raise SystemExit(f"--sizes takes positive token counts (got {part!r})")
        sizes.append(int(part))
    if not sizes:
        raise SystemExit("--sizes needs at least one token count")
    return tuple(sizes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default="latency",
                        help="latency | throughput | quality | calibration | determinism | all")
    parser.add_argument("--model", required=False, default=None,
                        help="path to a .gguf file (never a registry alias)")
    parser.add_argument("--backend", default="auto", help="auto | cpu | vulkan | cuda | all")
    parser.add_argument("--runs", type=int, default=harness.DEFAULT_RUNS)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--devset", default=None, help="override the committed dev set (JSONL)")
    parser.add_argument("--items", type=int, default=None, help="cap the dev-set items")
    parser.add_argument("--n-seq-max", dest="n_seq_max", type=int, default=None)
    parser.add_argument("--kv-type", dest="kv_type", default="auto")
    parser.add_argument("--gpu-layers", dest="gpu_layers", type=int, default=None)
    parser.add_argument("--n-bins", dest="n_bins", type=int, default=harness.N_BINS)
    parser.add_argument("--sizes", default=None,
                        help="prefill sizes of the latency table, e.g. 256,2048 (default: "
                             + ",".join(str(size) for size in harness.PREFILL_SIZES) + ")")
    parser.add_argument("--out", default=None, help="write the JSON report here")
    parser.add_argument("--out-dir", dest="out_dir", default=None,
                        help="write each report as <out-dir>/e2_<suite>.json")
    parser.add_argument("--quiet", action="store_true", help="no markdown on stdout")
    args = parser.parse_args(argv)

    chosen = list(harness.SUITES) if args.suite == "all" else [args.suite]
    reports: list[dict] = []
    for suite in chosen:
        config = build_config(args, suite)
        report = suites.run_suite(config, factory=suites.live_factory)
        reports.append(report)
        target = args.out if args.suite != "all" else None
        if args.out_dir:
            target = str(pathlib.Path(args.out_dir) / f"e2_{suite}.json")
        if target:
            harness.write_report(report, target)
        if not args.quiet:
            print(harness.render_report(report))
            if target:
                print(f"report: {target}")
            print()
    if args.out and args.suite == "all":
        harness.write_report({"schema": harness.SCHEMA, "suite": "all", "reports": reports},
                             args.out)
    failed = [report["suite"] for report in reports if report.get("ok") is False]
    if failed:
        print(f"gate failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
