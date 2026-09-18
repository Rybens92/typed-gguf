"""The five benchmark suites: latency, throughput, quality, calibration, determinism.

Milestone: E2 (SPEC 5, A-E2-1..A-E2-6).

Every suite returns one JSON report (`ggufone.bench/v1`) whose tables `bench.harness.render_report`
renders as the markdown published in `docs/BENCHMARKS.md`. The suites only talk to
`harness.ModelLike`, so the same code runs against a real bundle and against the deterministic
fake in `tests/fake_engine.py`.

What the suites deliberately do **not** do (A-E2-7): resolve a registry alias, read
`registry.json`, download anything or open a socket. A model is a path on disk, a runtime is a
local bundle, and the dev set ships inside the package.

Measurement conventions, pinned so a published number can be reproduced by hand:

* `latency` — `p50`/`p95` are the interpolated percentiles of `runs` samples (default 5) of the
  *same* request; the prefix state is warm (one warm-up call fills the state cache, and every
  measured call reports `prefill_reused`); `tok/s` is summarised per run (tokens / seconds), never
  as a ratio of two medians.
* `throughput` — one row per documented backend; a backend with no local bundle is reported as
  `measured: false` with the reason, never silently dropped. Non-CPU backends offload all layers
  unless `--gpu-layers` says otherwise, and the placement is printed in the row.
* `quality` / `calibration` — one decision per dev item, agreement = the argmax candidate equals
  the gold candidate (`devset.gold_key`); the three confidence modes are recomputed from the same
  stored distributions, so no extra model run is needed.
* `determinism` — 3 repeats per backend of the same request, compared byte-for-byte after
  stripping `timings` (`harness.digest`), with `threads=1` unless `--threads` overrides it.
* `latency` / `quality` / `calibration` measure **one** backend per run: `--backend auto` resolves
  to every locally installed bundle but these suites take the first of them (`DEFAULT_BACKENDS`
  order, `cpu` first), so the report records the choice (`backend_selection`) and says which bundle
  was passed over — a GPU box asking for `auto` must not look like a box without an accelerator
  (card t_31b3943a, coordinator note). `throughput` and `determinism` measure every usable backend.
"""
from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from ggufone import schema
from ggufone.bench import devset as devset_module
from ggufone.bench import harness
from ggufone.engine import decide as decide_module
from ggufone.engine import prompt, readout

Factory = Callable[[harness.ModelSpec], harness.ModelLike]

DETERMINISM_REPEATS = 3
WARM_STATE_ID = "bench-warm-prefix"
#: every latency request shares one prefix (and one saved state), so the question phase is warm
LATENCY_STATE_ID = "bench-latency-prefix"
LATENCY_STATE_WORDS = 120
#: a state long enough for the latency tables, short enough to stay well inside any context
LATENCY_STATE = " ".join(f"event{i % 29}" for i in range(LATENCY_STATE_WORDS))


def live_factory(spec: harness.ModelSpec) -> harness.ModelLike:
    """The production factory: load the model through the pinned runtime."""
    return harness.LiveModel(spec)


def run_suite(config: harness.BenchConfig, *, factory: Factory | None = None) -> dict[str, Any]:
    """Run one suite and return its report (the only entry point the CLI needs)."""
    harness.valid_suite(config.suite)
    make = factory or live_factory
    if config.suite == "latency":
        return _run_latency(config, make)
    if config.suite == "throughput":
        return _run_throughput(config, make)
    if config.suite == "quality":
        return _run_quality(config, make, calibration=False)
    if config.suite == "calibration":
        return _run_quality(config, make, calibration=True)
    return _run_determinism(config, make)


# --------------------------------------------------------------------------- backend selection
def _selected_backends(config: harness.BenchConfig, runtimes: Mapping[str, Any],
                       ) -> tuple[list[str], dict[str, str]]:
    """`(usable, missing-with-reason)`. A forced backend that is missing is an error."""
    chosen = config.backends(available=runtimes)
    usable = [backend for backend in chosen if backend in runtimes]
    missing = {backend: harness.backend_unavailable_reason(backend) for backend in chosen
               if backend not in runtimes}
    if not usable:
        raise harness.BenchError(
            f"none of the requested backends ({', '.join(chosen)}) has a local llama.cpp bundle; "
            + missing[chosen[0]], code="E_BENCH_BACKEND")
    return usable, missing


def _spec(config: harness.BenchConfig, backend: str, runtimes: Mapping[str, Any],
          ) -> harness.ModelSpec:
    if backend not in runtimes:
        # a forced backend (cpu/vulkan/cuda) with no bundle: say so, do not fall back silently
        raise harness.BenchError(harness.backend_unavailable_reason(backend),
                                 code="E_BENCH_BACKEND")
    return harness.spec_for(config, backend, runtimes=runtimes)


def _record_backend_selection(report: dict[str, Any], config: harness.BenchConfig, *,
                              selected: str, usable: Sequence[str],
                              missing: Mapping[str, str]) -> None:
    """Say which local bundle a single-backend suite measured, and which ones it passed over.

    `--backend auto` resolves to *every* locally installed bundle (`BenchConfig.backends`) but
    the single-backend suites measure one run's worth: the first of them, `DEFAULT_BACKENDS`
    order (`cpu` first). Without this record a published row reads `backend: cpu` on a box that
    also carries a Vulkan bundle, and the reader cannot tell "no accelerator here" from "the
    accelerator was not selected" — reported from the operator's Vulkan host (card t_31b3943a).
    """
    report["backend_selection"] = {"requested": config.backend, "selected": selected,
                                   "available": list(usable), "missing": dict(missing)}
    passed_over = [backend for backend in usable if backend != selected]
    if passed_over:
        report["notes"].append(
            f"one suite run measures one backend: with `--backend {config.backend}` the local "
            f"bundles are {', '.join(usable)} and {selected} was selected; pass `--backend "
            f"{passed_over[0]}` to measure that one instead, or run `--suite throughput`, which "
            f"measures every local backend in one report.")


def _envelope(config: harness.BenchConfig) -> dict[str, Any]:
    return harness.envelope(config, model=harness.model_facts(config.model_path or ""))


# --------------------------------------------------------------------------- latency
def _run_latency(config: harness.BenchConfig, make: Factory) -> dict[str, Any]:
    runtimes = harness.backend_runtimes(home=config.home)
    usable, missing = _selected_backends(config, runtimes)
    backend = usable[0]
    spec = _spec(config, backend, runtimes)
    report = _envelope(config)
    _record_backend_selection(report, config, selected=backend, usable=usable, missing=missing)
    model = make(spec)
    try:
        loads = [float(model.load()) for _ in range(max(1, config.runs))]
        report["model_load"] = harness.summarise(loads)
        # what the loader did with the requested placement (a degraded retry offloads less)
        report["placement"] = harness.placement_of(model, spec)
        report["prefill"] = _prefill_rows(config, model)
        report["per_question"] = _per_question_rows(config, model)
        report["wave_scaling"] = _wave_scaling_rows(config, model)
        report["warm_cache"] = _warm_cache_row(config, model)
        report["load_amortisation"] = _load_amortisation_row(config, model, loads)
        report["wave_accounting"] = planned_wave_breakdown(
            model, _five_way_request(config), n_seq_max=6, readout_mode="sequence")
        report["notes"].append(
            "`waves` counts the decode batches a decision takes after the prefill "
            f"(here: 1 suffix decode per question group + 1 per extra candidate token); "
            f"five single-token candidates in one question are 1 wave, not "
            f"ceil(5 / n_seq_max): {report['wave_accounting']}.")
        report["notes"].append(
            "every measured call reports `prefill_reused: true` once the prefix state cache is "
            "warm, which is why the decision tables isolate the question phase.")
    finally:
        model.close()
    return report


def _filler_tokens(model: harness.ModelLike, size: int) -> list[int]:
    """Exactly `size` tokens of real text (trimmed from a tokenized filler, never padded)."""
    text = " ".join(f"event{index % 101} code{index % 37}" for index in range(size))
    tokens = list(model.tokenize(text))
    if len(tokens) < size:                      # pragma: no cover - the filler is long enough
        tokens = tokens + [tokens[-1]] * (size - len(tokens))
    return tokens[:size]


def _prefill_rows(config: harness.BenchConfig, model: harness.ModelLike) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for size in config.prefill_sizes:
        tokens = _filler_tokens(model, int(size))
        durations: list[float] = []
        counts: list[int] = []
        for _ in range(max(1, config.runs)):
            with model.session(n_ctx=int(size) + 64, n_seq_max=3, threads=config.threads) as live:
                info = live.prefill(tokens, state_cache=False)
            durations.append(float(info.prefill_ms) / 1000.0)
            counts.append(int(info.prefill_tokens))
        rows.append({"tokens": int(size),
                     "ms": harness.summarise([value * 1000.0 for value in durations]),
                     "tok_per_s": harness.ratio_summarise(counts, durations),
                     "threads": config.threads or model.spec.threads,
                     "backend": model.spec.backend})
    return rows


def _choice_request(candidates: int, *, state: str = LATENCY_STATE, threads: int | None = None,
                    state_id: str | None = None, save_state: bool = False,
                    n_candidates_label: str = "option") -> dict[str, Any]:
    options = {f"{n_candidates_label}{index}": None for index in range(candidates)}
    request: dict[str, Any] = {
        "state": state, "model": "bench",
        "questions": {"q": {"type": "choice", "instructions": "Which option applies?",
                            "criteria": options}},
    }
    engine_options: dict[str, Any] = {}
    if threads:
        engine_options["threads"] = int(threads)
    if state_id:
        engine_options["state_id"] = state_id
    if save_state:
        engine_options["save_state"] = True
    if engine_options:
        request["options"] = engine_options
    return request


def _warm_choice_request(count: int, *, state_id: str, threads: int | None = None,
                         state: str = LATENCY_STATE) -> dict[str, Any]:
    """A choice request whose prefix state is persisted, so the question phase is measured warm.

    `save_state=True` + a fixed `state_id` is the documented warm path (SPEC 2.3.10): the first
    call writes the prefix state, every later call reloads it and reports `prefill_reused: true`.
    """
    return _choice_request(count, state=state, threads=threads, state_id=state_id,
                           save_state=True)


def _five_way_request(config: harness.BenchConfig) -> schema.Request:
    return schema.parse_request(_choice_request(5, threads=config.threads))


def _per_question_rows(config: harness.BenchConfig, model: harness.ModelLike,
                       ) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for count in config.candidate_counts:
        payload = _warm_choice_request(int(count), state_id=LATENCY_STATE_ID,
                                       threads=config.threads)
        request = schema.parse_request(payload)
        n_seq_max = max(config.n_seq_max or 0, 1 + int(count))
        model.decide(request, n_seq_max=n_seq_max, threads=config.threads)      # warm-up
        durations: list[float] = []
        totals: list[float] = []
        usage: dict[str, Any] = {}
        for _ in range(max(1, config.runs)):
            result = model.decide(request, n_seq_max=n_seq_max, threads=config.threads)
            durations.append(float(result.timings["questions_ms"]))
            totals.append(float(result.timings["total_ms"]))
            usage = dict(result.usage)
        rows.append({"candidates": int(count),
                     "ms": harness.summarise(durations),
                     "total_ms": harness.summarise(totals),
                     "waves": usage.get("waves"),
                     "forks": usage.get("forks"),
                     "decode_steps": usage.get("decode_steps"),
                     "prefix_tokens": usage.get("prefix_tokens"),
                     "n_seq_max": n_seq_max,
                     "threads": config.threads or model.spec.threads,
                     "backend": model.spec.backend,
                     "prefill_reused": bool(result.engine.get("prefill_reused"))})
    return rows


def _wave_scaling_rows(config: harness.BenchConfig,
                       model: harness.ModelLike) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_seq_max = config.n_seq_max or 4
    for count in config.wave_scaling:
        request = schema.parse_request(_warm_noul_request(int(count), threads=config.threads))
        model.decide(request, n_seq_max=n_seq_max, threads=config.threads)      # warm-up
        durations: list[float] = []
        usage: dict[str, Any] = {}
        for _ in range(max(1, config.runs)):
            result = model.decide(request, n_seq_max=n_seq_max, threads=config.threads)
            durations.append(float(result.timings["questions_ms"]))
            usage = dict(result.usage)
        rows.append({"questions": int(count), "ms": harness.summarise(durations),
                     "waves": usage.get("waves"), "forks": usage.get("forks"),
                     "decode_steps": usage.get("decode_steps"),
                     "ms_per_question": (harness.summarise(durations)["p50"] or 0.0) / count,
                     "n_seq_max": n_seq_max, "threads": config.threads or model.spec.threads,
                     "backend": model.spec.backend})
    return rows


def _noul_request(count: int, *, threads: int | None = None) -> dict[str, Any]:
    questions = {f"q{index}": {"type": "noul", "instructions": f"Decision {index}?",
                               "criteria": {"true": "yes, act now", "false": "no, wait"}}
                 for index in range(count)}
    request: dict[str, Any] = {"state": LATENCY_STATE, "model": "bench", "questions": questions}
    if threads:
        request["options"] = {"threads": int(threads)}
    return request


def _warm_noul_request(count: int, *, threads: int | None = None) -> dict[str, Any]:
    """The wave-scaling request with its prefix state persisted (warm cache, per A-E2-1)."""
    request = _noul_request(count, threads=threads)
    request["options"] = {**request.get("options", {}), "state_id": LATENCY_STATE_ID,
                          "save_state": True}
    return request


def _load_amortisation_row(config: harness.BenchConfig, model: harness.ModelLike,
                           loads: Sequence[float]) -> dict[str, Any]:
    """`serve` vs one-shot CLI: what a request costs when the model is already resident.

    The one-shot path pays `model_load_ms` on every call; a server pays it once. This row answers
    "how much does `--model` load reuse actually save" with this box's own numbers instead of a
    guess: `one_shot_ms_per_request = load_p50 + steady_ms_per_request`, `serve_ms_per_request`
    after the first call.
    """
    payload = _warm_choice_request(4, state_id=LATENCY_STATE_ID, threads=config.threads)
    request = schema.parse_request(payload)
    model.decide(request, n_seq_max=5, threads=config.threads)                  # load + fill
    wall: list[float] = []
    for _ in range(max(1, config.runs)):
        started = time.perf_counter()
        model.decide(request, n_seq_max=5, threads=config.threads)
        wall.append((time.perf_counter() - started) * 1000.0)
    steady = harness.summarise(wall)
    load_p50 = harness.summarise(loads)["p50"] or 0.0
    return {"calls": len(wall), "model_load_ms": load_p50,
            "steady_ms_per_request": steady,
            "serve_ms_per_request": steady["p50"],
            "one_shot_ms_per_request": load_p50 + (steady["p50"] or 0.0),
            "saved_ms_per_request": load_p50,
            "backend": model.spec.backend, "threads": config.threads or model.spec.threads}


def _warm_cache_row(config: harness.BenchConfig, model: harness.ModelLike) -> dict[str, Any]:
    """A persisted prefix state: the second call must report `prefill_reused` and ~0 ms."""
    payload = _choice_request(4, threads=config.threads, state_id=WARM_STATE_ID, save_state=True)
    request = schema.parse_request(payload)
    model.decide(request, n_seq_max=5, threads=config.threads)                  # fills the cache
    prefill: list[float] = []
    questions: list[float] = []
    reused = False
    for _ in range(max(1, config.runs)):
        result = model.decide(request, n_seq_max=5, threads=config.threads)
        prefill.append(float(result.timings["prefill_ms"]))
        questions.append(float(result.timings["questions_ms"]))
        reused = bool(result.engine.get("prefill_reused"))
    return {"state_id": WARM_STATE_ID, "prefill_reused": reused,
            "prefill_ms": harness.summarise(prefill), "questions_ms": harness.summarise(questions),
            "backend": model.spec.backend, "threads": config.threads or model.spec.threads}


def planned_wave_breakdown(model: harness.ModelLike, request: schema.Request, *,
                           n_seq_max: int, readout_mode: str = "sequence") -> dict[str, int]:
    """How a request's decode batches are counted — the accounting behind `usage.waves`.

    Mirrors `engine/decide.py`: each candidate group costs one batch for the question suffix plus
    one batch per extra candidate token (`max(candidate length) - 1`), and the total is what
    `usage.waves` reports. It is **not** `ceil(branches / (n_seq_max - 1))`: branches of one
    group are decoded together, but different questions are not batched together in E2.
    """
    requirements = decide_module.question_requirements(request, model)
    waves = decide_module.planned_waves(requirements, int(n_seq_max), readout_mode=readout_mode)
    per_wave = max(1, int(n_seq_max) - 1)
    groups = 0
    for _question, _view, _suffix, candidates in requirements:
        scored = decide_module.candidate_sequences(candidates, readout_mode=readout_mode)
        groups += len(list(_chunks(list(range(len(scored))), per_wave)))
    return {"groups": groups, "suffix_decodes": groups,
            "step_decodes": max(0, waves - groups), "waves": waves}


def _chunks(indices: Sequence[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(indices), size):
        yield list(indices[start:start + size])


# --------------------------------------------------------------------------- throughput
def _run_throughput(config: harness.BenchConfig, make: Factory) -> dict[str, Any]:
    runtimes = harness.backend_runtimes(home=config.home)
    report = _envelope(config)
    rows: list[dict[str, Any]] = []
    for backend in config.backends(available=runtimes):
        if backend not in runtimes:
            rows.append({"backend": backend, "measured": False,
                         "reason": harness.backend_unavailable_reason(backend)})
            continue
        spec = _spec(config, backend, runtimes)
        rows.append(_throughput_row(config, make, spec))
    report["backends"] = rows
    measured = [row for row in rows if row.get("measured")]
    report["ok"] = bool(measured)
    if not measured:
        report["notes"].append("no local backend bundle was available; nothing was measured")
    return report


def _throughput_row(config: harness.BenchConfig, make: Factory,
                    spec: harness.ModelSpec) -> dict[str, Any]:
    """One backend's row: load, prefill throughput, decision cost, decision throughput.

    The prefill measurement uses **one** size (`config.prefill_sizes[0]`, i.e. 256 tokens by
    default): the point of this table is backend-vs-backend on the same model, and the size sweep
    is the latency suite's job — measuring all three sizes here would spend an hour of the 4B
    model's 8k prefill to answer a question this table does not ask.
    """
    model = make(spec)
    row: dict[str, Any] = {"backend": spec.backend, "measured": True,
                           "runtime_dir": spec.runtime_dir,
                           "placement": f"n_gpu_layers={spec.n_gpu_layers}",
                           "threads": spec.threads}
    try:
        loads = [float(model.load()) for _ in range(max(1, config.runs))]
        row["load_ms"] = harness.summarise(loads)
        row["placement_used"] = harness.placement_of(model, spec)
        single = dataclasses.replace(config, prefill_sizes=tuple(config.prefill_sizes[:1]))
        prefill_rows = _prefill_rows(single, model)
        row["prefill"] = prefill_rows
        row["prefill_tok_per_s"] = prefill_rows[0]["tok_per_s"] if prefill_rows else \
            harness.summarise([])
        request = schema.parse_request(_choice_request(4, threads=config.threads))
        model.decide(request, n_seq_max=5, threads=config.threads)              # warm-up
        durations: list[float] = []
        steps: list[int] = []
        for _ in range(max(1, config.runs)):
            result = model.decide(request, n_seq_max=5, threads=config.threads)
            durations.append(float(result.timings["questions_ms"]))
            steps.append(int(result.usage["decode_steps"]))
        row["decision_ms"] = harness.summarise(durations)
        row["decision_tok_per_s"] = harness.ratio_summarise(
            steps, [max(value, 1e-6) / 1000.0 for value in durations])
    except Exception as exc:  # noqa: BLE001 - a backend that cannot run is a reported row
        row.update({"measured": False, "reason": f"{exc.__class__.__name__}: {exc}"})
    finally:
        model.close()
    return row


# --------------------------------------------------------------------------- quality / calibration
def _dev_items(config: harness.BenchConfig) -> list[devset_module.DevItem]:
    items = devset_module.load(config.devset)
    if config.items:
        items = items[:int(config.items)]
    return items


def _run_quality(config: harness.BenchConfig, make: Factory, *,
                 calibration: bool) -> dict[str, Any]:
    runtimes = harness.backend_runtimes(home=config.home)
    usable, missing = _selected_backends(config, runtimes)
    backend = usable[0]
    spec = _spec(config, backend, runtimes)
    items = _dev_items(config)
    report = _envelope(config)
    _record_backend_selection(report, config, selected=backend, usable=usable, missing=missing)
    report["devset"] = {"path": str(devset_module.devset_path(config.devset)),
                        "items": len(items), "counts": devset_module.counts(items),
                        "provenance": devset_module.PROVENANCE}
    model = make(spec)
    try:
        model.load()
        rows = _devset_rows(config, model, items)
    finally:
        model.close()
    report["items"] = rows
    report["per_type"] = agreement_by_type(rows)
    report["overall"] = agreement(rows)
    report["ok"] = True
    if not calibration:
        report["notes"].append(
            "agreement = the highest-probability candidate equals the gold candidate "
            "(the discrete decision), measured per question type and overall with 95% Wilson "
            "intervals; report-only in v1 (SPEC S-11).")
        return report
    rows = [row for row in rows if row["probabilities"]]
    report["n"] = len(rows)
    report["n_bins"] = config.n_bins
    report["confidences"] = [row["confidence"] for row in rows]
    report["reliability"] = harness.reliability_bins(
        [row["confidence"] for row in rows], [row["correct"] for row in rows],
        n_bins=config.n_bins)
    report["ece"] = harness.ece(report["reliability"])
    report["modes"] = _mode_rows(rows, config.n_bins)
    report["coverage"] = harness.summarise([row["coverage"] for row in rows])
    report["confidence_coverage_correlation"] = harness.pearson(
        [row["confidence"] for row in rows], [row["coverage"] for row in rows])
    report["notes"].append(
        "confidence is the answer's own `confidence` (noul carries none, so the highest "
        "probability stands in); `coverage` is the full-vocabulary mass the engine reported.")
    report["notes"].append(
        "the three confidence modes are recomputed from the same stored distributions, so the "
        "table costs no extra model runs; the correlation is undefined (null) when every "
        "confidence is identical.")
    return report


def _mode_rows(rows: Sequence[Mapping[str, Any]], n_bins: int) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode in ("normalized_peak", "entropy", "margin"):
        confidences = [readout.confidence(list(row["probabilities"].values()), mode)
                       for row in rows]
        bins = harness.reliability_bins(confidences, [row["correct"] for row in rows],
                                        n_bins=n_bins)
        modes[mode] = {"n": len(confidences), "ece": harness.ece(bins),
                       "mean_confidence": (sum(confidences) / len(confidences)) if confidences
                       else None, "bins": bins}
    return modes


def _devset_rows(config: harness.BenchConfig, model: harness.ModelLike,
                 items: Sequence[devset_module.DevItem]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        payload = devset_module.request_for(item, model="bench", threads=config.threads)
        request = schema.parse_request(payload)
        started = time.perf_counter()
        result = model.decide(request, threads=config.threads)
        answer = dict(result.answers[item.id])
        probabilities = {key: float(value) for key, value in answer["probabilities"].items()}
        got = readout.argmax_first(list(probabilities.values()))
        winner = list(probabilities)[got]
        expected = devset_module.gold_key(item)
        coverage = float(answer.get("coverage") or 0.0)
        rows.append({
            "id": item.id, "type": item.type, "expected": expected, "got": winner,
            "correct": winner == expected,
            "confidence": float(answer.get("confidence", max(probabilities.values()))),
            "coverage": coverage,
            "reliability": answer.get("reliability"),
            "probabilities": probabilities,
            "questions_ms": float(result.timings["questions_ms"]),
            "wall_ms": (time.perf_counter() - started) * 1000.0,
        })
    return rows


def agreement(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Exact-match agreement over every row: count, rate and the 95 % Wilson interval."""
    correct = sum(1 for row in rows if row["correct"])
    low, high = harness.wilson_interval(correct, len(rows))
    return {"n": len(rows), "correct": correct,
            "agreement": (correct / len(rows)) if rows else 0.0, "ci": [low, high]}


def agreement_by_type(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {qtype: agreement([row for row in rows if row["type"] == qtype])
            for qtype in sorted({str(row["type"]) for row in rows})}


# --------------------------------------------------------------------------- determinism
def _run_determinism(config: harness.BenchConfig, make: Factory) -> dict[str, Any]:
    runtimes = harness.backend_runtimes(home=config.home)
    usable, missing = _selected_backends(config, runtimes)
    if config.backend not in ("auto", "all") and config.backend not in usable:
        raise harness.BenchError(harness.backend_unavailable_reason(config.backend),
                                 code="E_BENCH_BACKEND")
    report = _envelope(config)
    report["repeats"] = DETERMINISM_REPEATS
    report["threads"] = config.threads or 1
    report["request"] = _determinism_request(config)
    rows: list[dict[str, Any]] = []
    for backend in usable:
        rows.append(_determinism_row(config, make, _spec(config, backend, runtimes)))
    report["backends"] = rows
    report["skipped_backends"] = missing
    report["ok"] = bool(rows) and all(row["ok"] for row in rows)
    if not report["ok"]:
        report["notes"].append(
            "the repeats of one backend produced different bytes after stripping `timings`; "
            "SPEC A5 pins byte identity to (runtime, backend, threads=1)")
    return report


def _determinism_request(config: harness.BenchConfig) -> dict[str, Any]:
    return {
        "state": "The billing dashboard is blank for every user after login since 09:12.",
        "model": "bench",
        "questions": {
            "area": {"type": "choice", "instructions": "Which team owns this?",
                     "criteria": {"billing": "payments", "technical": "infrastructure",
                                  "support": "conversations"}},
            "severity": {"type": "score", "instructions": "How severe?",
                         "criteria": ["cosmetic", "degrading", "critical"]},
            "page": {"type": "noul", "instructions": "Page the on-call engineer?",
                     "criteria": {"true": "yes", "false": "no"}},
        },
        "options": {"threads": int(config.threads or 1)},
    }


def _determinism_row(config: harness.BenchConfig, make: Factory,
                     spec: harness.ModelSpec) -> dict[str, Any]:
    row: dict[str, Any] = {"backend": spec.backend, "threads": int(config.threads or 1),
                           "placement": f"n_gpu_layers={spec.n_gpu_layers}"}
    model = make(spec)
    try:
        model.load()
        request = schema.parse_request(_determinism_request(config))
        digests: list[str] = []
        for _ in range(DETERMINISM_REPEATS):
            result = model.decide(request, threads=config.threads)
            body = schema.render_response(result.payload(), format="native")
            digests.append(harness.digest(body))
        row.update({"digests": digests, "identical": len(set(digests)) == 1,
                    "ok": len(set(digests)) == 1, "repeats": len(digests)})
        if len(set(digests)) != 1:
            row["reason"] = "the repeats differ after stripping `timings`"
    except Exception as exc:  # noqa: BLE001 - a backend that cannot run is a reported row
        row.update({"measured": False, "ok": False, "digests": [],
                    "identical": False, "reason": f"{exc.__class__.__name__}: {exc}"})
    finally:
        model.close()
    return row


__all__ = ["Factory", "live_factory", "run_suite", "agreement", "agreement_by_type",
           "planned_wave_breakdown", "prompt"]
