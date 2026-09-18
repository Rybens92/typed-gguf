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
* `determinism` — `config.determinism_repeats` repeats per backend of the same request (3 in the
  full campaign, 2 under `--quick`), compared byte-for-byte after stripping `timings`
  (`harness.digest`), with `threads=1` unless `--threads` overrides it.
* `latency` / `quality` / `calibration` measure **one** backend per run: `--backend auto` resolves
  to every locally installed bundle but these suites take the first of them (`DEFAULT_BACKENDS`
  order, `cpu` first), so the report records the choice (`backend_selection`) and says which bundle
  was passed over — a GPU box asking for `auto` must not look like a box without an accelerator
  (card t_31b3943a, coordinator note). `throughput` and `determinism` measure every usable backend.

Two cross-cutting rules apply to every suite:

* **`--quick`** (`harness.quick_config`) only changes *scale*: the row shapes, the code paths, the
  accounting and the renderer are the same as a full run's.
* **`--max-seconds`** is a soft cap realized as one `harness.TimeBudget` per run: one *row* (with
  all of its `runs` samples) is one budget unit and the cap is checked between units, so the unit
  that started always finishes; units that never started are recorded (`_measure`) and the report
  lists them under `"truncated": true` while the exit code stays 0.
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
    """Run one suite and return its report (the only entry point the CLI needs).

    The soft cap (`config.max_seconds`) is realized here as one `harness.TimeBudget` per run:
    every suite checks it *between* its measurements and records the units it never started, so
    the report can carry `"truncated": true` plus the unmeasured rows and still exit 0.
    """
    harness.valid_suite(config.suite)
    make = factory or live_factory
    budget = harness.TimeBudget(config.max_seconds)
    if config.suite == "latency":
        report = _run_latency(config, make, budget)
    elif config.suite == "throughput":
        report = _run_throughput(config, make, budget)
    elif config.suite == "quality":
        report = _run_quality(config, make, budget, calibration=False)
    elif config.suite == "calibration":
        report = _run_quality(config, make, budget, calibration=True)
    else:
        report = _run_determinism(config, make, budget)
    report["truncated"] = bool(budget.skipped)
    report["skipped"] = list(budget.skipped)
    report["wall_ms"] = round(budget.elapsed() * 1000.0, 3)
    report["budget"] = budget.to_dict()
    # every report carries an explicit `ok`: "what ran passed" (a suite without a gate has nothing
    # to fail, and `truncated` — not `ok` — is what says the run is incomplete)
    report.setdefault("ok", True)
    return report


def _measure(budget: harness.TimeBudget, section: str, row: str,
             measure: Callable[[], Any]) -> Any | None:
    """One measurement under the soft cap: the cap is checked *between* units, never inside one.

    A unit that starts always finishes (the report must not carry a half-measured row); a unit
    that never starts is recorded under its `section`/`row` label and reported as unmeasured.
    """
    if budget.expired():
        budget.skip(section, row)
        return None
    return measure()


# --------------------------------------------------------------------------- backend selection
def _selected_backends(config: harness.BenchConfig, runtimes: Mapping[str, Any],
                       ) -> tuple[list[str], dict[str, str]]:
    """`(usable, missing-with-reason)`. A forced backend that is missing is an error.

    `config.backend_limit` (the `--quick` preset's "one backend") cuts the *usable* list: the
    missing map keeps every backend that was asked for, so the report can still name what it did
    not measure and why.
    """
    chosen = config.backends(available=runtimes)
    usable = [backend for backend in chosen if backend in runtimes]
    missing = {backend: harness.backend_unavailable_reason(backend) for backend in chosen
               if backend not in runtimes}
    if not usable:
        raise harness.BenchError(
            f"none of the requested backends ({', '.join(chosen)}) has a local llama.cpp bundle; "
            + missing[chosen[0]], code="E_BENCH_BACKEND")
    if config.backend_limit is not None:
        usable = usable[:max(1, int(config.backend_limit))]
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
def _unit_budget() -> harness.TimeBudget:
    """A budget that never expires: the *inner* parts of one multi-part measurement.

    A throughput row (load + prefill + decisions for one backend) is one measurement unit, so the
    run-level cap must not cut inside it — `_measure` on this budget can only ever run the unit.
    """
    return harness.TimeBudget(None)


def _run_latency(config: harness.BenchConfig, make: Factory,
                 budget: harness.TimeBudget) -> dict[str, Any]:
    runtimes = harness.backend_runtimes(home=config.home)
    usable, missing = _selected_backends(config, runtimes)
    backend = usable[0]
    spec = _spec(config, backend, runtimes)
    report = _envelope(config)
    _record_backend_selection(report, config, selected=backend, usable=usable, missing=missing)
    model = make(spec)
    try:
        loads: list[float] = []
        for index in range(max(1, config.runs)):
            load = _measure(budget, "model_load", f"load#{index + 1}",
                            lambda: float(model.load()))
            if load is None:
                break
            loads.append(float(load))
        report["model_load"] = harness.summarise(loads)
        # what the loader did with the requested placement (a degraded retry offloads less)
        report["placement"] = harness.placement_of(model, spec)
        report["prefill"] = _prefill_rows(config, model, budget)
        report["per_question"] = _per_question_rows(config, model, budget)
        report["wave_scaling"] = _wave_scaling_rows(config, model, budget)
        warm = _measure(budget, "warm_cache", "state reuse", lambda: _warm_cache_row(config, model))
        report["warm_cache"] = warm if warm is not None else {}
        amortised = _measure(budget, "load_amortisation", "serve vs one-shot",
                             lambda: _load_amortisation_row(config, model, loads))
        report["load_amortisation"] = amortised if amortised is not None else {}
        if loads:
            report["wave_accounting"] = planned_wave_breakdown(
                model, _five_way_request(config), n_seq_max=6, readout_mode="sequence")
            report["notes"].append(
                "`waves` counts the decode batches a decision takes after the prefill "
                f"(here: 1 suffix decode per question group + 1 per extra candidate token); "
                f"five single-token candidates in one question are 1 wave, not "
                f"ceil(5 / n_seq_max): {report['wave_accounting']}.")
        else:
            report["notes"].append(
                "no measurement ran: the soft cap was already spent before the first model load")
        if report["per_question"]:
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


def _prefill_rows(config: harness.BenchConfig, model: harness.ModelLike,
                  budget: harness.TimeBudget) -> list[dict[str, Any]]:
    """One row per prefill size — each size is one measurement of the soft cap."""
    rows: list[dict[str, Any]] = []
    for size in config.prefill_sizes:
        row = _measure(budget, "prefill", f"tokens={int(size)}",
                       lambda size=size: _prefill_row(config, model, int(size)))
        if row is not None:
            rows.append(row)
    return rows


def _prefill_row(config: harness.BenchConfig, model: harness.ModelLike, size: int,
                 ) -> dict[str, Any]:
    tokens = _filler_tokens(model, size)
    durations: list[float] = []
    counts: list[int] = []
    for _ in range(max(1, config.runs)):
        with model.session(n_ctx=size + 64, n_seq_max=3, threads=config.threads) as live:
            info = live.prefill(tokens, state_cache=False)
        durations.append(float(info.prefill_ms) / 1000.0)
        counts.append(int(info.prefill_tokens))
    return {"tokens": size,
            "ms": harness.summarise([value * 1000.0 for value in durations]),
            "tok_per_s": harness.ratio_summarise(counts, durations),
            "threads": config.threads or model.spec.threads,
            "backend": model.spec.backend}


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
                       budget: harness.TimeBudget) -> list[dict[str, Any]]:
    """One row per candidate count (warm-up + `runs` samples = one measurement unit)."""
    rows: list[dict[str, Any]] = []
    for count in config.candidate_counts:
        row = _measure(budget, "per_question", f"candidates={int(count)}",
                       lambda count=count: _per_question_row(config, model, int(count)))
        if row is not None:
            rows.append(row)
    return rows


def _per_question_row(config: harness.BenchConfig, model: harness.ModelLike, count: int,
                      ) -> dict[str, Any]:
    payload = _warm_choice_request(count, state_id=LATENCY_STATE_ID, threads=config.threads)
    request = schema.parse_request(payload)
    n_seq_max = max(config.n_seq_max or 0, 1 + count)
    model.decide(request, n_seq_max=n_seq_max, threads=config.threads)      # warm-up
    durations: list[float] = []
    totals: list[float] = []
    usage: dict[str, Any] = {}
    for _ in range(max(1, config.runs)):
        result = model.decide(request, n_seq_max=n_seq_max, threads=config.threads)
        durations.append(float(result.timings["questions_ms"]))
        totals.append(float(result.timings["total_ms"]))
        usage = dict(result.usage)
    return {"candidates": count,
            "ms": harness.summarise(durations),
            "total_ms": harness.summarise(totals),
            "waves": usage.get("waves"),
            "forks": usage.get("forks"),
            "decode_steps": usage.get("decode_steps"),
            "prefix_tokens": usage.get("prefix_tokens"),
            "n_seq_max": n_seq_max,
            "threads": config.threads or model.spec.threads,
            "backend": model.spec.backend,
            "prefill_reused": bool(result.engine.get("prefill_reused"))}


def _wave_scaling_rows(config: harness.BenchConfig, model: harness.ModelLike,
                       budget: harness.TimeBudget) -> list[dict[str, Any]]:
    """One row per question count (N=1..16 in the full campaign, {1, 2} under `--quick`)."""
    rows: list[dict[str, Any]] = []
    for count in config.wave_scaling:
        row = _measure(budget, "wave_scaling", f"questions={int(count)}",
                       lambda count=count: _wave_scaling_row(config, model, int(count)))
        if row is not None:
            rows.append(row)
    return rows


def _wave_scaling_row(config: harness.BenchConfig, model: harness.ModelLike, count: int,
                      ) -> dict[str, Any]:
    n_seq_max = config.n_seq_max or 4
    request = schema.parse_request(_warm_noul_request(count, threads=config.threads))
    model.decide(request, n_seq_max=n_seq_max, threads=config.threads)      # warm-up
    durations: list[float] = []
    usage: dict[str, Any] = {}
    for _ in range(max(1, config.runs)):
        result = model.decide(request, n_seq_max=n_seq_max, threads=config.threads)
        durations.append(float(result.timings["questions_ms"]))
        usage = dict(result.usage)
    return {"questions": count, "ms": harness.summarise(durations),
            "waves": usage.get("waves"), "forks": usage.get("forks"),
            "decode_steps": usage.get("decode_steps"),
            "ms_per_question": (harness.summarise(durations)["p50"] or 0.0) / count,
            "n_seq_max": n_seq_max, "threads": config.threads or model.spec.threads,
            "backend": model.spec.backend}


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
def _run_throughput(config: harness.BenchConfig, make: Factory,
                    budget: harness.TimeBudget) -> dict[str, Any]:
    runtimes = harness.backend_runtimes(home=config.home)
    report = _envelope(config)
    chosen = config.backends(available=runtimes)
    measured_backends = chosen if config.backend_limit is None else \
        chosen[:max(1, int(config.backend_limit))]
    rows: list[dict[str, Any]] = []
    for backend in chosen:
        if backend not in measured_backends:
            # `--quick` resolves one backend: the rest stay in the table as *unmeasured*, with the
            # preset named as the reason, instead of silently disappearing from it
            rows.append({"backend": backend, "measured": False,
                         "reason": f"not measured: the --quick preset resolves one backend "
                                   f"({measured_backends[0] if measured_backends else 'none'})"})
            continue
        if backend not in runtimes:
            rows.append({"backend": backend, "measured": False,
                         "reason": harness.backend_unavailable_reason(backend)})
            continue
        spec = _spec(config, backend, runtimes)
        if budget.expired():
            rows.append({"backend": backend, "measured": False,
                         "reason": budget.skip("backends", backend)["reason"]})
            continue
        rows.append(_throughput_row(config, make, spec))
    report["backends"] = rows
    measured = [row for row in rows if row.get("measured")]
    # a row that never started because the cap was already spent is incompleteness, not a failure
    report["ok"] = bool(measured) or bool(budget.skipped)
    if not measured and not budget.skipped:
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
        prefill_rows = _prefill_rows(single, model, _unit_budget())
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
    """The items this run asks: stratified `--quick` selection first, then the `--items` cap."""
    items = devset_module.load(config.devset)
    if config.items_per_type:
        items = devset_module.stratify(items, per_type=int(config.items_per_type))
    if config.items:
        items = items[:int(config.items)]
    return items


def _run_quality(config: harness.BenchConfig, make: Factory, budget: harness.TimeBudget, *,
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
        # the load is a measurement too, but the item loop below runs regardless: when the cap is
        # already spent, every item is recorded as unmeasured (the lambda is never called, so an
        # unloaded model is never asked anything) and the report lists all six rows by id
        _measure(budget, "model_load", "load#1", model.load)
        rows = _devset_rows(config, model, items, budget)
    finally:
        model.close()
    report["devset"]["measured"] = len(rows)
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
    n_bins = int(config.n_bins)
    if config.quick and rows:
        # "bins as available": six samples cannot fill ten bins, and an ECE over mostly-empty bins
        # is a number about nothing — so the quick preset reports as many bins as it has samples
        n_bins = min(n_bins, len(rows))
    report["n_bins"] = n_bins
    report["confidences"] = [row["confidence"] for row in rows]
    report["reliability"] = harness.reliability_bins(
        [row["confidence"] for row in rows], [row["correct"] for row in rows],
        n_bins=n_bins)
    report["ece"] = harness.ece(report["reliability"])
    report["modes"] = _mode_rows(rows, n_bins)
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
    if config.quick:
        report["notes"].append(
            f"quick calibration: {report['n']} samples in {report['n_bins']} bins is a shape "
            f"check (the presets, the modes and the bin machinery all exercised), never a "
            f"calibration claim — the full campaign's 60 items are what an ECE is read from.")
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
                 items: Sequence[devset_module.DevItem],
                 budget: harness.TimeBudget) -> list[dict[str, Any]]:
    """One row per dev item — each item is one measurement of the soft cap."""
    rows: list[dict[str, Any]] = []
    for item in items:
        row = _measure(budget, "items", item.id,
                       lambda item=item: _devset_row(config, model, item))
        if row is not None:
            rows.append(row)
    return rows


def _devset_row(config: harness.BenchConfig, model: harness.ModelLike,
                item: devset_module.DevItem) -> dict[str, Any]:
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
    return {
        "id": item.id, "type": item.type, "expected": expected, "got": winner,
        "correct": winner == expected,
        "confidence": float(answer.get("confidence", max(probabilities.values()))),
        "coverage": coverage,
        "reliability": answer.get("reliability"),
        "probabilities": probabilities,
        "questions_ms": float(result.timings["questions_ms"]),
        "wall_ms": (time.perf_counter() - started) * 1000.0,
    }


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
def _run_determinism(config: harness.BenchConfig, make: Factory,
                     budget: harness.TimeBudget) -> dict[str, Any]:
    runtimes = harness.backend_runtimes(home=config.home)
    usable, missing = _selected_backends(config, runtimes)
    if config.backend not in ("auto", "all") and config.backend not in usable:
        raise harness.BenchError(harness.backend_unavailable_reason(config.backend),
                                 code="E_BENCH_BACKEND")
    report = _envelope(config)
    report["repeats"] = int(config.determinism_repeats)
    report["threads"] = config.threads or 1
    report["request"] = _determinism_request(config)
    rows: list[dict[str, Any]] = []
    for backend in usable:
        if budget.expired():
            budget.skip("backends", backend)
            continue
        rows.append(_determinism_row(config, make, _spec(config, backend, runtimes)))
    report["backends"] = rows
    report["skipped_backends"] = missing
    # the gate is about the repeats that *ran*: a backend the cap never reached is incompleteness
    # (`"truncated": true` says so and the CLI still exits 0), a row that ran and differed is a
    # failure the exit code must keep at 1
    failed = [row for row in rows if not row["ok"]]
    report["ok"] = not failed and (bool(rows) or bool(budget.skipped))
    if failed:
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
        for _ in range(int(config.determinism_repeats)):
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
