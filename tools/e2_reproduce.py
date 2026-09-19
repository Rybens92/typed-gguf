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

    # the fast iteration preset (card t_f46cec41) — never a published table
    GGUFONE_RUNTIME_DIR=<bundle> python3 tools/e2_reproduce.py --suite all --quick \
        --model <path.gguf> --threads 4 --out-dir /tmp/quick

`--quick` mirrors `ggufone bench --quick` exactly (same `harness.quick_config` preset, same
refusal of `--runs/--items/--sizes/--n-seq-max`, same distinct report names: a quick run never
writes over a full-campaign JSON). `--max-seconds N` is the CLI's soft cap.

The suites never touch the model registry and never open a socket (A-E2-7): `--model` is a path
on disk, the runtime is a local bundle and the dev set ships inside the package.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ggufone import schema  # noqa: E402
from ggufone.bench import harness, suites  # noqa: E402


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default="latency",
                        help="latency | throughput | quality | calibration | determinism | all")
    parser.add_argument("--model", required=False, default=None,
                        help="path to a .gguf file (never a registry alias)")
    parser.add_argument("--backend", default="auto", help="auto | cpu | vulkan | cuda | all")
    # the *scale* knobs default to None so `--quick` can tell "given" from "default": stating one
    # next to --quick is refused (the preset fixes them) instead of being silently ignored
    parser.add_argument("--runs", type=int, default=None,
                        help=f"measured runs per row (default: {harness.DEFAULT_RUNS})")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--devset", default=None, help="override the committed dev set (JSONL)")
    parser.add_argument("--items", type=int, default=None, help="cap the dev-set items")
    parser.add_argument("--n-seq-max", dest="n_seq_max", type=int, default=None)
    parser.add_argument("--kv-type", dest="kv_type", default="auto")
    parser.add_argument("--cue", default="shipped",
                        help="shipped | two_step | json_field | json_instructed — the cue shape the "
                             "rows are read at (E3d card t_d90404ac, E3e card t_4c48f40a; "
                             "default = the published shape)")
    parser.add_argument("--chat-format", dest="chat_format", default=schema.ANSWER_SHEET,
                        choices=list(schema.CHAT_FORMATS),
                        help="answer_sheet | role_split — where the question block lives (E3e card "
                             "t_4c48f40a; default = the published shape, the question prefilled "
                             "into the assistant turn)")
    parser.add_argument("--gpu-layers", dest="gpu_layers", type=int, default=None)
    parser.add_argument("--n-bins", dest="n_bins", type=int, default=harness.N_BINS)
    parser.add_argument("--sizes", default=None,
                        help="prefill sizes of the latency table, e.g. 256,2048 (default: "
                             + ",".join(str(size) for size in harness.PREFILL_SIZES) + ")")
    parser.add_argument("--quick", action="store_true",
                        help="the short preset (`ggufone bench --quick`): fast feedback, "
                             "never a published table; writes its own report file")
    parser.add_argument("--max-seconds", dest="max_seconds", type=float, default=None,
                        help="soft cap checked between measurements: the report is marked "
                             "truncated with the unmeasured rows listed, exit code stays 0")
    parser.add_argument("--out", default=None, help="write the JSON report here")
    parser.add_argument("--out-dir", dest="out_dir", default=None,
                        help="write each report as <out-dir>/e2_<suite>[_quick].json")
    parser.add_argument("--quiet", action="store_true", help="no markdown on stdout")
    return parser


def quick_conflicts(args: argparse.Namespace) -> list[str]:
    """The explicitly given scale flags `--quick` fixes: `--runs 5 --quick` is a usage error."""
    given = (("--runs", args.runs), ("--items", args.items), ("--sizes", args.sizes),
             ("--n-seq-max", args.n_seq_max))
    return [flag for flag, value in given if value is not None]


def build_config(args: argparse.Namespace, suite: str) -> harness.BenchConfig:
    if args.max_seconds is not None and args.max_seconds < 0:
        raise SystemExit("--max-seconds takes a positive number of seconds "
                         f"(got {args.max_seconds!r})")
    config = harness.BenchConfig(
        suite=suite,
        model_path=harness.resolve_model_path(args.model),
        backend=args.backend,
        runs=harness.DEFAULT_RUNS if args.runs is None else args.runs,
        threads=args.threads,
        devset=args.devset,
        items=args.items,
        n_seq_max=args.n_seq_max,
        kv_type=args.kv_type,
        cue=args.cue,
        chat_format=args.chat_format,
        gpu_layers=args.gpu_layers,
        n_bins=args.n_bins,
        max_seconds=args.max_seconds,
        prefill_sizes=parse_sizes(args.sizes))
    return harness.quick_config(config) if args.quick else config


def resolve_out(args: argparse.Namespace, suite: str) -> str | None:
    """`--out` wins; `--out-dir` names a quick report distinctly; a bare quick run writes its own.

    The rule mirrors `cli._bench_out_path`: no quick run can land on the `e2_<suite>.json` a full
    campaign wrote, whether the path came from `--out-dir` or from the default name.
    """
    if args.out:
        return args.out
    suffix = harness.QUICK_OUT_SUFFIX if args.quick else ""
    if args.out_dir:
        return str(pathlib.Path(args.out_dir) / f"e2_{suite}{suffix}.json")
    return harness.default_out_path(suite, quick=True) if args.quick else None


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
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.quick:
        conflicts = quick_conflicts(args)
        if conflicts:
            parser.error(f"--quick runs a fixed preset and already sets {' and '.join(conflicts)}; "
                         f"drop {'those flags' if len(conflicts) > 1 else 'that flag'} or run "
                         f"without --quick")

    chosen = list(harness.SUITES) if args.suite == "all" else [args.suite]
    reports: list[dict] = []
    for suite in chosen:
        config = build_config(args, suite)
        report = suites.run_suite(config, factory=suites.live_factory)
        reports.append(report)
        target = resolve_out(args, suite) if (args.suite != "all" or args.out_dir) else None
        if target:
            harness.write_report(report, target)
        if not args.quiet:
            print(harness.render_report(report))
            if target:
                print(f"report: {target}")
            print()
    if args.out and args.suite == "all":
        harness.write_report({"schema": harness.SCHEMA, "suite": "all",
                              "quick": bool(args.quick), "reports": reports}, args.out)
    failed = [report["suite"] for report in reports if report.get("ok") is False]
    if failed:
        print(f"gate failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
