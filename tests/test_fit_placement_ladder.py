"""The placement ladder a *context* walks, and the numbers its OOM message may claim.

Card `t_287e0d18`, findings P1 and P2 of the live calibration card `t_b67f9c49`.

P1: the 4B loaded on Vulkan and *then* `llama_init_from_model` failed with
`ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory` — and the only ladder that
reacted was `session._init_context`, whose rungs were the three KV types and nothing else, while
`fit.backend_oom_error` printed "tried 3 placement(s) down to CPU-only, none fit" over the top of
them. The three things pinned here are the three that were wrong:

* the order `session.py` documents — kv_type rungs, then a smaller `n_ctx`, then fewer
  `n_gpu_layers` (which re-places the model), then CPU-only, where no device allocation exists;
* the message: it names the rungs that were really tried and nothing else;
* the budget: a driver that reports no free number gets a *conservative* one, not the nominal
  device size (the repro's plan spent a 5482 MiB budget on a box whose desktop already held
  ~1.5 GB of an 8192 MiB device).

P2: `calibrate` accepted `--fit-target`/`--n-seq-max` and dropped them before the plan.

Every gate is hermetic: a synthetic GGUF, a fake runtime whose context init fails exactly like the
operator's bundle, and injected host facts.
"""
from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace
from typing import Any

import pytest

from tests.test_fit import GIB, MIB, gpu_host, tiny_model, write_gguf
from tests.test_fit_oom_recovery import FakeBackend, fake_runtime, gpu_plan
from typed_gguf import cli
from typed_gguf.engine import session as session_module
from typed_gguf.engine.decide import ContextPlan
from typed_gguf.errors import BackendOomError, RuntimeMissingError
from typed_gguf.runtime import fit

#: The context-side allocation failure, verbatim from the repro (`.e2e/t_b67f9c49-calibrate/
#: vulkan-oom-2.stderr`): three buffers, the last one the graph scheduler's compute buffer.
OPERATOR_CTX_OOM = """\
ggml_vulkan: Device memory allocation of size 972029952 failed.
ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory
ggml_vulkan: Device memory allocation of size 273678336 failed.
ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory
ggml_gallocr_reserve_n_impl: failed to allocate Vulkan0 buffer of size 273678336
""".strip()

GGML_F16, GGML_Q4_0, GGML_Q8_0 = 1, 2, 8
LAYERS = 36


# ----------------------------------------------------------------- the fake bundle (P1's vehicle)
class ContextOomBackend(FakeBackend):
    """A bundle that loads the model and then cannot create the context while it is offloaded.

    `allows(n_ctx, n_gpu_layers)` answers for the *context*; the model load always succeeds (that
    is the repro: the weights fit, the KV cache and the compute buffer did not). Every context
    attempt is recorded as `(n_ctx, type_k, layers-of-the-load-in-force)`, which is what makes the
    walked ladder visible instead of inferred.
    """

    def __init__(self, *, allows: Any, n_layer: int = LAYERS, log_lines: Any = None) -> None:
        super().__init__(n_layer=n_layer, fail=lambda ngl, call: False,
                         log_lines=tuple((log_lines or OPERATOR_CTX_OOM).splitlines()))
        self.allows = allows
        self.ctx_calls: list[tuple[int, int, int]] = []
        self.accepted: list[int] = []
        self._ctx_n_ctx = 0
        self.llama.llama_context_default_params = lambda: SimpleNamespace(
            n_ctx=0, n_batch=0, n_ubatch=0, n_seq_max=0, n_threads=0, n_threads_batch=0,
            type_k=0, type_v=0, kv_unified=False, no_perf=True, flash_attn_type=0)
        self.llama.llama_init_from_model = self._init
        self.llama.llama_get_memory = lambda ctx: 8
        self.llama.llama_free = lambda ctx: None
        self.llama.llama_n_ctx = lambda ctx: self._ctx_n_ctx
        self.llama.llama_n_seq_max = lambda ctx: 3

    def _init(self, model: Any, params: Any) -> int:
        layers = self.load_calls[-1] if self.load_calls else 0
        self.ctx_calls.append((int(params.n_ctx), int(params.type_k), int(layers)))
        if not self.allows(int(params.n_ctx), int(layers)):
            for line in self.log_lines:
                if self._installed is not None:
                    self._installed(4, line.encode() + b"\n", None)
            return 0
        self._ctx_n_ctx = int(params.n_ctx)
        self.accepted.append(int(params.n_ctx))
        return 9                                     # a truthy, non-NULL context handle


def session_for(backend: ContextOomBackend, tmp_path: pathlib.Path, model_path: pathlib.Path,
                plan: fit.FitPlan, *, handle: Any = None, **kwargs: Any) -> tuple[Any, Any]:
    """Open a model and create one context through the production path."""
    handle = handle or session_module.open_model(model_path, runtime_dir=backend.directory,
                                                 fit_plan=plan)
    context = ContextPlan(n_ctx=32768, n_seq_max=3, threads=1, kv_type="auto",
                          prefix_tokens=(1, 2, 3))
    live = session_module.ModelSession(handle, context, **kwargs)
    return handle, live


# --------------------------------------------------- (a)/(b): the documented ladder, as rungs
def test_the_context_ladder_is_kv_then_ctx_then_layers_then_cpu_only() -> None:
    """kv_type first, then a smaller `n_ctx`, then fewer layers, ending at CPU-only."""
    model = tiny_model()
    plan = fit.estimate_plan(model, gpu_host(), n_ctx=32768, n_seq_max=3)

    rungs = fit.context_ladder(plan, model, n_ctx=32768, min_ctx=4096, kv_type="f16")

    walked = [(rung.kv_type, rung.n_ctx, rung.n_gpu_layers) for rung in rungs]
    assert walked == [
        ("f16", 32768, LAYERS),
        ("q8_0", 32768, LAYERS),
        ("q4_0", 32768, LAYERS),
        ("q4_0", 16384, LAYERS),
        ("q4_0", 4096, LAYERS),
        ("q4_0", 4096, LAYERS // 2),
        ("q4_0", 4096, 0),                       # the designed end state: nothing on the device
    ]
    assert rungs[-1].n_gpu_layers == 0
    assert len({(r.kv_type, r.n_ctx, r.n_gpu_layers) for r in rungs}) == len(rungs)


def test_the_context_ladder_never_walks_a_rung_twice_or_upwards() -> None:
    """A context that already sits at the floor has no ctx rung left, and no rung grows."""
    model = tiny_model()
    plan = fit.estimate_plan(model, gpu_host(), n_ctx=4096, n_seq_max=3)
    rungs = fit.context_ladder(plan, model, n_ctx=4096, min_ctx=4096, kv_type="f16")

    assert [(r.kv_type, r.n_ctx, r.n_gpu_layers) for r in rungs] == [
        ("f16", 4096, LAYERS), ("q8_0", 4096, LAYERS), ("q4_0", 4096, LAYERS),
        ("q4_0", 4096, LAYERS // 2), ("q4_0", 4096, 0)]


def test_a_plan_that_already_offloads_nothing_has_only_kv_and_ctx_rungs() -> None:
    """CPU-only is the end state: there is no layer rung below it, and no model to re-place."""
    model = tiny_model()
    plan = fit.FitPlan(n_gpu_layers=0, n_ctx=32768, kv_type="auto", n_seq_max=3,
                       est_weights_bytes=4 * GIB, est_kv_bytes=0, est_total_bytes=4 * GIB,
                       backend="cpu", source="estimate")
    rungs = fit.context_ladder(plan, model, n_ctx=32768, min_ctx=4096)
    assert [(r.kv_type, r.n_ctx, r.n_gpu_layers) for r in rungs] == [
        ("f16", 32768, 0), ("q8_0", 32768, 0), ("q4_0", 32768, 0),
        ("q4_0", 16384, 0), ("q4_0", 4096, 0)]


def test_degrade_disabled_walks_exactly_one_rung() -> None:
    """`degrade=False` keeps today's behaviour: one attempt, the plan's own rung."""
    model = tiny_model()
    plan = fit.estimate_plan(model, gpu_host(), n_ctx=32768, n_seq_max=3)
    rungs = fit.context_ladder(plan, model, n_ctx=32768, min_ctx=4096, kv_type="f16",
                               degrade=False)
    assert [(r.kv_type, r.n_ctx, r.n_gpu_layers) for r in rungs] == [("f16", 32768, LAYERS)]


def test_a_pinned_kv_type_starts_at_its_own_rung() -> None:
    """A request that pinned `q8_0` does not silently walk *up* to f16 first."""
    model = tiny_model()
    plan = fit.estimate_plan(model, gpu_host(), n_ctx=32768, n_seq_max=3)
    rungs = fit.context_ladder(plan, model, n_ctx=32768, min_ctx=4096, kv_type="q8_0")
    assert [r.kv_type for r in rungs][:2] == ["q8_0", "q4_0"]


# --------------------------------------------------------- (a)/(b): the ladder, through a load
def test_a_context_that_oomed_shrinks_the_context_then_re_places_the_model_to_cpu(
        tmp_path: pathlib.Path) -> None:
    """The repro, exactly: only the CPU-only rung can allocate — and the run survives it."""
    model_path, plan = gpu_plan(tmp_path, n_layer=LAYERS)
    backend = ContextOomBackend(allows=lambda n_ctx, layers: layers == 0)

    with fake_runtime(tmp_path, backend):
        handle, live = session_for(backend, tmp_path, model_path, plan)
        try:
            assert backend.ctx_calls == [
                (32768, GGML_F16, LAYERS), (32768, GGML_Q8_0, LAYERS),
                (32768, GGML_Q4_0, LAYERS), (16384, GGML_Q4_0, LAYERS),
                (4096, GGML_Q4_0, LAYERS),
                (4096, GGML_Q4_0, LAYERS // 2), (4096, GGML_Q4_0, 0)]
            assert backend.load_calls == [LAYERS, LAYERS // 2, 0]   # the model was re-placed
            assert live.kv_type_used == "q4_0"
            assert live.meta.n_ctx == 4096                          # the context that exists
            assert live.plan.n_ctx == 4096                          # ... and what the engine sees
            assert handle.n_gpu_layers == 0
            assert handle.placement.n_gpu_layers == 0
            assert handle.placement.degraded is True
            assert handle.fit_plan.n_gpu_layers == 0
            assert "W_FIT_DOWNGRADE" in live.extra_warnings
            assert "W_BACKEND_OOM" in live.extra_warnings
            assert "W_KV_TYPE_DOWNGRADE" in live.extra_warnings
            # the note is about the *weights*, and it names the placement that exists
            assert handle.placement.note.startswith("degraded after a backend allocation failure")
            assert "0 layer(s) offloaded" in handle.placement.note
        finally:
            live.close()
            handle.close()


def test_a_context_that_fits_after_one_re_place_stops_there(tmp_path: pathlib.Path) -> None:
    """Half the layers is enough: the CPU-only rung is never reached, and no rung is skipped."""
    model_path, plan = gpu_plan(tmp_path, n_layer=LAYERS)
    backend = ContextOomBackend(allows=lambda n_ctx, layers: layers <= LAYERS // 2)

    with fake_runtime(tmp_path, backend):
        handle, live = session_for(backend, tmp_path, model_path, plan)
        try:
            assert backend.load_calls == [LAYERS, LAYERS // 2]
            assert backend.ctx_calls[-1] == (4096, GGML_Q4_0, LAYERS // 2)
            assert handle.n_gpu_layers == LAYERS // 2
            assert live.plan.n_ctx == 4096
        finally:
            live.close()
            handle.close()


def test_a_context_that_fits_at_full_layers_keeps_the_plan(tmp_path: pathlib.Path) -> None:
    """A healthy context walks one rung: no reload, no shrink, no downgrade warning."""
    model_path, plan = gpu_plan(tmp_path, n_layer=LAYERS)
    backend = ContextOomBackend(allows=lambda n_ctx, layers: True)

    with fake_runtime(tmp_path, backend):
        handle, live = session_for(backend, tmp_path, model_path, plan)
        try:
            assert backend.ctx_calls == [(32768, GGML_F16, LAYERS)]
            assert backend.load_calls == [LAYERS]
            assert live.kv_type_used == "f16"
            assert live.plan.n_ctx == 32768
            assert live.extra_warnings == []
            assert handle.placement.degraded is False
        finally:
            live.close()
            handle.close()


def test_a_context_failure_that_is_not_memory_stops_the_walk(tmp_path: pathlib.Path) -> None:
    """A silent/arch failure is never retried down the ladder — a smaller cache cannot help."""
    model_path, plan = gpu_plan(tmp_path, n_layer=LAYERS)
    backend = ContextOomBackend(allows=lambda n_ctx, layers: False,
                                log_lines="llama_init_from_model: something odd happened")
    with fake_runtime(tmp_path, backend):
        handle = session_module.open_model(model_path, runtime_dir=backend.directory,
                                           fit_plan=plan)
        try:
            with pytest.raises(RuntimeMissingError) as excinfo:
                session_for(backend, tmp_path, model_path, plan, handle=handle)
            assert backend.ctx_calls == [(32768, GGML_F16, LAYERS)]
            assert backend.load_calls == [LAYERS]
            assert "(unknown)" in str(excinfo.value)
        finally:
            handle.close()


def test_the_cpu_only_rung_of_a_second_session_does_not_reload_the_model_back_up(
        tmp_path: pathlib.Path) -> None:
    """The handle is the state: a later session starts from the placement that exists.

    `calibrate` creates one session per dev-set item on one handle; a ladder that always started
    from the plan's layer count would re-map 4.4 GB of weights between items.
    """
    model_path, plan = gpu_plan(tmp_path, n_layer=LAYERS)
    backend = ContextOomBackend(allows=lambda n_ctx, layers: layers == 0)

    with fake_runtime(tmp_path, backend):
        handle, first = session_for(backend, tmp_path, model_path, plan)
        first.close()
        second = session_module.ModelSession(
            handle, ContextPlan(n_ctx=32768, n_seq_max=3, threads=1, kv_type="auto",
                                prefix_tokens=(1, 2, 3)))
        try:
            assert backend.load_calls == [LAYERS, LAYERS // 2, 0]
            assert handle.n_gpu_layers == 0
            # nothing is offloaded any more, so the *first* rung allocates from host memory:
            # the second session pays one attempt, not another walk
            assert backend.ctx_calls[-1] == (32768, GGML_F16, 0)
            assert second.meta.n_ctx == 32768
            assert second.kv_type_used == "f16"
            assert second.extra_warnings == []
        finally:
            second.close()
            handle.close()


# ------------------------------------------------------------ (c): the message names the rungs
def test_the_oom_message_does_not_claim_a_cpu_rung_that_was_not_walked() -> None:
    """The repro's lie: "down to CPU-only" over three kv rungs."""
    error = fit.backend_oom_error(None, attempts=["ctx kv_type=f16 n_ctx=32768 -> oom",
                                                  "ctx kv_type=q8_0 n_ctx=32768 -> oom"])
    message = str(error)
    assert "tried 2 placement(s), none fit: ctx kv_type=f16 n_ctx=32768 -> oom" in message
    assert "CPU-only" not in message


def test_the_oom_message_still_says_cpu_only_when_the_rung_really_ran() -> None:
    attempts = ["n_gpu_layers=36 -> oom", "n_gpu_layers=0 -> oom"]
    error = fit.backend_oom_error(None, attempts=attempts, cpu_rung=True)
    assert "down to CPU-only" in str(error)


def test_a_walk_that_ends_at_cpu_only_reports_every_rung_it_tried(tmp_path: pathlib.Path) -> None:
    """Only the end state distinguishes the two: here the CPU rung really was reached."""
    model_path, plan = gpu_plan(tmp_path, n_layer=LAYERS)
    backend = ContextOomBackend(allows=lambda n_ctx, layers: False)

    with fake_runtime(tmp_path, backend), pytest.raises(BackendOomError) as excinfo:
        session_for(backend, tmp_path, model_path, plan)
    message = str(excinfo.value)
    assert "7 placement(s) down to CPU-only, none fit" in message
    assert "ctx kv_type=f16 n_ctx=32768 -> oom" in message
    assert "ctx kv_type=q4_0 n_ctx=4096 n_gpu_layers=0 -> oom" in message


def test_a_single_rung_walk_names_one_rung_and_no_cpu_claim(tmp_path: pathlib.Path) -> None:
    model_path, plan = gpu_plan(tmp_path, n_layer=LAYERS)
    backend = ContextOomBackend(allows=lambda n_ctx, layers: False)

    with fake_runtime(tmp_path, backend), pytest.raises(BackendOomError) as excinfo:
        session_for(backend, tmp_path, model_path, plan, degrade=False)
    message = str(excinfo.value)
    assert "tried 1 placement(s), none fit: ctx kv_type=f16 n_ctx=32768 -> oom" in message
    assert backend.load_calls == [LAYERS]
    assert "CPU-only" not in message


# ------------------------------------------------- (d): the budget when free is unreadable
def test_an_unreadable_free_number_buys_a_conservative_budget_not_the_nominal_device() -> None:
    """`vram_free_bytes == 0` is "the driver could not say", never "the whole device is free"."""
    silent = fit.HostFacts(backend="vulkan", ram_bytes=31 * GIB, vram_bytes=8 * GIB,
                           vram_free_bytes=0, n_cpu=8, fingerprint="vulkan:silent")
    assert silent.free_is_known is False
    assert silent.budget_bytes == 6 * GIB                 # 8 GiB nominal - the unknown-free reserve
    assert silent.budget_bytes < silent.vram_bytes

    readable = fit.HostFacts(backend="vulkan", ram_bytes=31 * GIB, vram_bytes=8 * GIB,
                             vram_free_bytes=5 * GIB, n_cpu=8, fingerprint="vulkan:busy")
    assert readable.free_is_known is True
    assert readable.budget_bytes == 5 * GIB               # a driver that answers is believed


def test_a_cpu_only_host_is_not_charged_a_device_reserve() -> None:
    host = fit.HostFacts(backend="cpu", ram_bytes=31 * GIB, vram_bytes=0, n_cpu=8,
                         fingerprint="cpu:test")
    assert host.free_is_known is True
    assert host.budget_bytes == 31 * GIB


def test_the_conservative_budget_reaches_the_plan(tmp_path: pathlib.Path) -> None:
    """The number a plan reports is the one it planned against, and it is bounded."""
    host = fit.host_facts(meminfo_path=tmp_path / "meminfo", vram_probe=lambda: 6 * GIB,
                          backend="vulkan", n_cpu=2)
    plan = fit.estimate_plan(tiny_model(), host, n_ctx=4096, n_seq_max=3)
    assert plan.budget_bytes == fit.fit_budget(host)
    assert plan.budget_bytes < 6 * GIB - fit.DEFAULT_FIT_TARGET_MB * MIB


def test_the_context_oom_message_reports_what_the_driver_really_said(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The repro printed "the driver reports unknown free" while `nvidia-smi` could answer."""
    model_path, plan = gpu_plan(tmp_path, n_layer=LAYERS)
    backend = ContextOomBackend(allows=lambda n_ctx, layers: False)
    monkeypatch.setattr(session_module, "default_free_device_bytes", lambda: 1112 * MIB)

    with fake_runtime(tmp_path, backend), pytest.raises(BackendOomError) as excinfo:
        session_for(backend, tmp_path, model_path, plan)
    message = str(excinfo.value)
    assert "the driver reports 1112 MiB free" in message
    assert "unknown free" not in message


def test_an_unknown_free_is_named_as_unknown_and_as_an_upper_bound(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path, plan = gpu_plan(tmp_path, n_layer=LAYERS)
    backend = ContextOomBackend(allows=lambda n_ctx, layers: False)
    monkeypatch.setattr(session_module, "default_free_device_bytes", lambda: None)

    with fake_runtime(tmp_path, backend), pytest.raises(BackendOomError) as excinfo:
        session_for(backend, tmp_path, model_path, plan)
    message = str(excinfo.value)
    assert "the driver reports unknown free" in message
    assert "upper bound" in message


# --------------------------------------------------------------- (P2): `calibrate`'s flags
def _plan_for_cli(monkeypatch: pytest.MonkeyPatch, seen: list[dict[str, Any]]) -> None:
    """Run both CLI paths through the real planner and record what each one asked for."""
    original = cli.fit_plan_for

    def spy(model_path: str, **kwargs: Any) -> fit.FitPlan:
        plan = original(model_path, **kwargs)
        seen.append({"kwargs": kwargs, "plan": plan.to_dict()})
        return plan

    monkeypatch.setattr(cli, "fit_plan_for", spy)


def test_calibrate_passes_fit_target_and_n_seq_max_into_the_plan(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    """P2: the flags were accepted and silently dropped (`cli.CALIBRATE_VALUE_FLAGS`)."""
    model = write_gguf(tmp_path / "synthetic.gguf")
    monkeypatch.setenv("TYPED_GGUF_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("TYPED_GGUF_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(fit, "host_facts", lambda **kwargs: gpu_host(total_gib=8.0,
                                                                     free_gib=8.0))
    seen: list[dict[str, Any]] = []
    _plan_for_cli(monkeypatch, seen)
    opened: list[Any] = []

    class FakeHandle:
        fit_plan = None

        def __enter__(self) -> FakeHandle:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def fake_open(*args: Any, **kwargs: Any) -> FakeHandle:
        opened.append(kwargs)
        return FakeHandle()

    monkeypatch.setattr(session_module, "open_model", fake_open)

    cli.main(["fit", str(model), "--fit-target", "4500", "--json"])
    fit_payload = json.loads(capsys.readouterr().out)
    cli.main(["calibrate", "--model", str(model), "--fit-target", "4500", "--n-seq-max", "2",
              "--items", "0", "--dry-run", "--json"])
    capsys.readouterr()

    assert seen[1]["kwargs"]["fit_target_mb"] == 4500
    assert seen[1]["kwargs"]["n_seq_max"] == 2
    assert seen[0]["kwargs"]["fit_target_mb"] == 4500
    # "changes the plan exactly as `fit --fit-target N` does": the same arithmetic, the same plan
    for field in ("budget_bytes", "n_ctx", "kv_type", "n_gpu_layers", "ctx_limit",
                  "standard_n_ctx", "warnings", "notes"):
        assert seen[1]["plan"][field] == seen[0]["plan"][field], field
    assert fit_payload["budget_bytes"] == seen[0]["plan"]["budget_bytes"]
    assert fit_payload["budget_bytes"] < 8 * GIB


def test_calibrate_without_the_flags_leaves_the_planner_defaults_alone(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    model = write_gguf(tmp_path / "synthetic.gguf")
    monkeypatch.setenv("TYPED_GGUF_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("TYPED_GGUF_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(fit, "host_facts", lambda **kwargs: gpu_host(total_gib=8.0,
                                                                     free_gib=8.0))
    seen: list[dict[str, Any]] = []
    _plan_for_cli(monkeypatch, seen)
    monkeypatch.setattr(session_module, "open_model", lambda *a, **k: _NullHandle())

    cli.main(["calibrate", "--model", str(model), "--items", "0", "--dry-run", "--json"])
    capsys.readouterr()

    assert seen[0]["kwargs"]["fit_target_mb"] is None
    assert seen[0]["kwargs"]["n_seq_max"] is None
    assert seen[0]["kwargs"]["kv_type"] == "auto"


class _NullHandle:
    fit_plan = None

    def __enter__(self) -> _NullHandle:
        return self

    def __exit__(self, *exc: object) -> None:
        return None
