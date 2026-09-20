#!/usr/bin/env python3
"""E3c-Tiel campaign driver: the same lens E3 used on Occamy 1.0, aimed at Tiel-Coder.

The measurement command is the E3 one, byte for byte — `tools/e3_reproduce.py --suite quality`
with `--backend vulkan --threads 4` on the host — so the two quality suites are comparable.
This wrapper adds the one thing the card asks for that the benchmark report does not carry
(the quality suite's own shape, E3 §4.3): the **placement the loader really used** per chunk.

`LiveModel.load()` stores `handle.placement.to_dict()` (n_gpu_layers / kv_type / degraded /
attempts / warnings) after the loader's degrade ladder settles. The factory below is the
production `harness.LiveModel` with one observer attached — it changes no flag, no plan and no
number the benchmark measures; it only writes the placement that was going to be discarded.

    python3 tools/e3c_tiel_reproduce.py --suite quality --model <tiel.gguf> \
        --backend vulkan --gpu-layers <as-fitted> --threads 4 \
        --devset docs/evidence/tiel_chunks/devset_003.jsonl \
        --out docs/evidence/tiel_chunks/report_003.json \
        --placement-out docs/evidence/tiel_chunks/placement_003.json

Everything else (`--suite merge`, `--suite batch`, `--suite compare`, `--write-chunks`) is the
committed E3 driver, called through.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import e3_reproduce  # noqa: E402  (the committed E3 driver: chunks, merge, batch, compare)

from typed_gguf.bench import harness, suites  # noqa: E402


class ObservedModel(harness.LiveModel):
    """`LiveModel` with an observer on the placement the loader settled on (nothing else).

    Everything the benchmark measures goes through the parent class untouched; this subclass only
    *reads* `self.placement` after `load()` and writes it to the campaign's sink, because the
    quality suite's report shape does not carry the placement (E3 §4.3) and the card asks for it
    per chunk.
    """

    def __init__(self, spec: harness.ModelSpec, sink: dict[str, object]) -> None:
        super().__init__(spec)
        self._sink = sink

    def load(self) -> float:
        started = time.perf_counter()
        value = super().load()
        self._sink["placement"] = dict(self.placement or {})
        self._sink["load_ms"] = value
        self._sink["load_wall_s"] = round(time.perf_counter() - started, 3)
        self._sink["device_log"] = self.device_log
        return value

    def decide(self, request: Any, **kwargs: Any) -> Any:
        """Pass through, and record the context the engine sizes for this request.

        `Placement` (weights: layers/kv_type/degraded/attempts) is a loader fact; the *context*
        (`n_ctx`, `n_seq_max`) is planned per request — `plan_context` is pure arithmetic over the
        request and the tokenizer, so asking it a second time cannot perturb the measurement.
        """
        from typed_gguf.engine import decide as decide_module
        handle = self._require_handle()
        try:
            plan = decide_module.plan_context(request, handle)
            self._sink["n_ctx"] = int(plan.n_ctx)
            self._sink["n_seq_max"] = int(plan.n_seq_max)
            self._sink["n_prefix"] = int(len(plan.prefix_tokens))
            self._sink["kv_type_used"] = str(plan.kv_type)
        except Exception as error:                      # never break a measurement over a note
            self._sink.setdefault("context_note_error", f"{type(error).__name__}: {error}")
        return super().decide(request, **kwargs)


def factory_with_sink(sink: dict[str, object]):
    def make(spec: harness.ModelSpec) -> ObservedModel:
        return ObservedModel(spec, sink)
    return make


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", default="quality", help="quality | merge | compare | batch")
    parser.add_argument("--model", default=None)
    parser.add_argument("--backend", default="vulkan")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--devset", default=None)
    parser.add_argument("--items", type=int, default=None)
    parser.add_argument("--n-seq-max", dest="n_seq_max", type=int, default=None)
    parser.add_argument("--kv-type", dest="kv_type", default="auto")
    parser.add_argument("--gpu-layers", dest="gpu_layers", type=int, default=None)
    parser.add_argument("--write-chunks", default=None)
    parser.add_argument("--chunk", type=int, default=10)
    parser.add_argument("--reports", nargs="+", default=[])
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--a", default=None)
    parser.add_argument("--b", default=None)
    parser.add_argument("--align", action="store_true")
    parser.add_argument("--label", default=None)
    parser.add_argument("--work-dir", dest="work_dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--placement-out", dest="placement_out", default=None)
    parser.add_argument("--fit-target", dest="fit_target", type=int, default=None)
    parser.add_argument("--no-fit", dest="no_fit", action="store_true")
    parser.add_argument("--audit", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.suite in {"merge", "compare", "batch"} or args.write_chunks:
        return e3_reproduce.main(argv)

    config = harness.BenchConfig(
        suite=args.suite, model_path=harness.resolve_model_path(args.model), backend=args.backend,
        runs=args.runs, threads=args.threads, devset=args.devset, items=args.items,
        n_seq_max=args.n_seq_max, kv_type=args.kv_type, gpu_layers=args.gpu_layers)
    sink: dict[str, object] = {"suite": args.suite, "model": str(config.model_path),
                              "backend": args.backend, "threads": args.threads,
                              "gpu_layers_requested": args.gpu_layers,
                              "devset": args.devset,
                              "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    started = time.perf_counter()
    report = suites.run_suite(config, factory=factory_with_sink(sink))
    sink["wall_s"] = round(time.perf_counter() - started, 1)
    sink["placements_seen"] = None
    sink["schema"] = "typed_gguf.e3c_tiel.placement/v1"
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
