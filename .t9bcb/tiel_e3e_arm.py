#!/usr/bin/env python3
"""t_9bcbecff — the [host] Tiel arm with the two E3e *policy* flags.

`tools/e3c_tiel_reproduce.py` (the corrected-instrument driver the published Tiel row was measured
with, card `t_7c926398`) hard-codes the shipped cue and the answer-sheet placement: it builds
`harness.BenchConfig` without `cue=`/`chat_format=`/`json_contract=`. This card must not rewrite
that committed driver, so this wrapper is that driver's own factory/observer plus the three flags
E3e (`t_4c48f40a`) shipped as switches:

    python3 .t9bcb/tiel_e3e_arm.py --suite quality \\
        --cue json_instructed --chat-format role_split --json-contract question \\
        --model <tiel.gguf> --backend vulkan --gpu-layers 9 --threads 4 \\
        --devset <chunk>.jsonl --out <report>.json --placement-out <sink>.json

The observer is the committed `e3c_tiel_reproduce.ObservedModel` (the placement the loader settled
on plus the per-request context plan), subclassed with two read-only receipts this card asks for:

* `chat_format_seen` — the `engine.chat_format` surface of every decision (`kind`,
  `question_turn`, `contract`, `prefix_chars`, `dropped`), counted, so the report can say which
  rendering path the role split actually took instead of assuming the one it hoped for;
* `template_seen` — the `engine.template` surface per decision (`kind`/`renderer`/`source`),
  counted, which is the built-in-bridge question on this family (its own template is outside the
  internal renderer's subset).

Neither writes anything the benchmark measures; both only read a response that was produced anyway
(`super().decide()` is the committed observer's decide, unchanged).
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

import e3c_tiel_reproduce  # noqa: E402  (the committed observer + its factory)

from ggufone.bench import harness, suites  # noqa: E402


def _counter(sink: dict[str, object], key: str) -> dict[str, int]:
    store = sink.get(key)
    if not isinstance(store, dict):
        store = {}
        sink[key] = store
    return store


def _count(store: dict[str, int], value: Any) -> None:
    store[json.dumps(value, sort_keys=True, default=str)] = \
        store.get(json.dumps(value, sort_keys=True, default=str), 0) + 1


class PolicyObservedModel(e3c_tiel_reproduce.ObservedModel):
    """The committed observer + the two E3e surfaces, read from the same response."""

    def decide(self, request: Any, **kwargs: Any) -> Any:
        answer = super().decide(request, **kwargs)
        engine = dict(getattr(answer, "engine", None) or {})
        _count(_counter(self._sink, "chat_format_seen"), engine.get("chat_format"))
        surface = dict(engine.get("template") or {})
        _count(_counter(self._sink, "template_seen"),
               {key: surface.get(key) for key in ("kind", "renderer", "source", "family")})
        return answer


def factory_with_policy_sink(sink: dict[str, object]):
    def make(spec: harness.ModelSpec) -> PolicyObservedModel:
        return PolicyObservedModel(spec, sink)
    return make


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--suite", default="quality")
    parser.add_argument("--cue", default="shipped")
    parser.add_argument("--chat-format", dest="chat_format", default=None)
    parser.add_argument("--json-contract", dest="json_contract", default=None)
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

    from ggufone import schema

    config = harness.BenchConfig(
        suite=args.suite, model_path=harness.resolve_model_path(args.model), backend=args.backend,
        runs=args.runs, threads=args.threads, devset=args.devset, items=args.items,
        n_seq_max=args.n_seq_max, kv_type=args.kv_type, gpu_layers=args.gpu_layers,
        cue=args.cue,
        chat_format=args.chat_format or schema.ANSWER_SHEET,
        json_contract=args.json_contract or schema.JSON_CONTRACT)
    sink: dict[str, object] = {"suite": args.suite, "cue": args.cue,
                               "chat_format": config.chat_format,
                               "json_contract": config.json_contract,
                               "model": str(config.model_path), "backend": args.backend,
                               "threads": args.threads, "gpu_layers_requested": args.gpu_layers,
                               "devset": args.devset,
                               "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    started = time.perf_counter()
    report = suites.run_suite(config, factory=factory_with_policy_sink(sink))
    sink["wall_s"] = round(time.perf_counter() - started, 1)
    sink["placements_seen"] = None
    sink["schema"] = "ggufone.t9bcbecff.e3e_arm/v1"
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
