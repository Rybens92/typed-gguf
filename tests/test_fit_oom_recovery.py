"""E1c FIX (card t_8cb0a05e), requirements 3 + 4: degrade on an allocation failure, and classify
it as `E_BACKEND_OOM` instead of `E_MODEL_ARCH_UNSUPPORTED`.

The vehicle is a *fake runtime object* whose `llama_model_load_from_file` behaves like the
operator's box: it prints the backend's own log lines through the real `llama_log_set` callback
(`session.capture_llama_logs` is production code here, not a mock) and returns NULL while anything
is offloaded. The OOM text is the operator's tail, verbatim.
"""
from __future__ import annotations

import contextlib
import ctypes
import json
import pathlib
from types import SimpleNamespace
from typing import Any

import pytest

from tests.fake_engine import FakeSession, biased_row
from tests.test_fit import GIB, MIB, write_gguf
from typed_gguf import schema
from typed_gguf.engine import session as session_module
from typed_gguf.engine.decide import SessionMeta, decide_request
from typed_gguf.errors import (
    ERROR_CODES,
    WARNING_CODES,
    BackendOomError,
    ModelArchUnsupportedError,
    RuntimeMissingError,
)
from typed_gguf.registry import recommend
from typed_gguf.runtime import ctypes_binding, finder, fit

# The operator's tail (card t_8cb0a05e), verbatim — the last line is typed-gguf's OLD, wrong reading
# of the failure, and must not decide the classification.
OPERATOR_OOM_TAIL = """
ggml_vulkan: Device memory allocation of size 1058982400 failed.
ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory
alloc_tensor_range: failed to allocate Vulkan0 buffer of size 1058982400
llama_model_load: error loading model: unable to allocate Vulkan0 buffer
error: E_RUNTIME_MISSING: E_MODEL_ARCH_UNSUPPORTED: llama.cpp could not load model.gguf
""".strip()

ARCH_TAIL = "\n".join((
    "llama_model_load: error loading model: unknown model architecture: 'spark9_9'",
    "llama_model_load_from_file: failed to load model",
))


# --------------------------------------------------------------------------- the fake runtime
class FakeBackend:
    """A llama.cpp runtime as production sees it: log callback + a load that can fail."""

    def __init__(self, *, n_layer: int = 4, vocab: int = 64,
                 fail: Any = lambda ngl, call: bool(ngl > 0),
                 log_lines: tuple[str, ...] = tuple(OPERATOR_OOM_TAIL.splitlines())) -> None:
        self.n_layer = n_layer
        self.vocab = vocab
        self.fail = fail
        self.log_lines = log_lines
        self.load_calls: list[int] = []
        self.log_set_calls: list[Any] = []
        self.previous_handler: Any = None
        self._installed: Any = None
        self.directory = pathlib.Path("/fake/runtime")
        #: The ggml half of a real bundle: a `cpu`-pinned bench load asks it for the CPU device it
        #: may use (card t_55de5779). A runtime double without it would refuse every `cpu` row —
        #: which is itself a pinned behaviour (`tests/test_bench_cpu_force.py`).
        self.ggml = SimpleNamespace(
            ggml_backend_dev_by_name=ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p)(
                lambda name: 0xB11026 if name == b"CPU" else None))
        self.llama = SimpleNamespace(
            llama_model_default_params=self._default_params,
            llama_model_load_from_file=self._load,
            llama_model_get_vocab=lambda model: 1,
            llama_vocab_n_tokens=lambda vocab_arg: self.vocab,
            llama_model_n_layer=lambda model: self.n_layer,
            llama_model_free=lambda model: None,
            llama_log_set=self._log_set,
            llama_log_get=self._log_get,
        )

    # ---- the two log hooks production installs
    def _log_set(self, callback: Any, user_data: Any) -> None:
        self.log_set_calls.append(callback)
        self._installed = callback

    def _log_get(self) -> Any:
        return self.previous_handler

    # ---- the load path
    def _default_params(self) -> Any:
        return SimpleNamespace(n_gpu_layers=0)

    def _load(self, path: bytes, params: Any) -> int:
        n_gpu_layers = int(params.n_gpu_layers)
        self.load_calls.append(n_gpu_layers)
        if self.fail(n_gpu_layers, len(self.load_calls)):
            for line in self.log_lines:
                if self._installed is not None:
                    self._installed(4, line.encode() + b"\n", None)
            return 0
        return 1


@contextlib.contextmanager
def fake_runtime(tmp_path: pathlib.Path, backend: FakeBackend):
    """Install `backend` as the runtime, behind a REAL bundle directory.

    The directory carries a `libllama.so` whose bytes name `llama_model_spark2_5`, so the
    production arch pre-flight (`capability.require_arch`) really runs and really passes — the
    ladder is exercised through the same gate the operator's host uses.
    """
    directory = tmp_path / "runtime"
    directory.mkdir(exist_ok=True)
    (directory / finder.library_names()["llama"]).write_bytes(
        b"\x7fELF" + b"llama_model_spark2_5\x00")
    backend.directory = directory
    original = ctypes_binding.load_libraries
    ctypes_binding.load_libraries = lambda dir_arg, **kwargs: backend  # type: ignore[assignment]
    try:
        yield backend
    finally:
        ctypes_binding.load_libraries = original  # type: ignore[assignment]


def gpu_host(free_mib: int = 1112) -> fit.HostFacts:
    return fit.HostFacts(backend="vulkan", ram_bytes=31 * GIB, vram_bytes=8 * GIB,
                         vram_free_bytes=free_mib * MIB, n_cpu=8, fingerprint="vulkan:busy")


def gpu_plan(tmp_path: pathlib.Path, *, n_layer: int = 4) -> tuple[Any, fit.FitPlan]:
    """A synthetic GGUF + the plan a roomy box would have written for it."""
    model_path = write_gguf(tmp_path / "model.gguf", n_layer=n_layer)
    model = fit.ModelFacts.read(model_path, want_sha256=False)
    plan = fit.estimate_plan(model, gpu_host(8 * 1024), n_ctx=4096, n_seq_max=8)
    assert plan.n_gpu_layers == n_layer
    return model_path, plan


# ------------------------------------------------------- R4: the log says what actually happened
def test_the_operator_tail_is_an_allocation_failure_not_an_arch_problem() -> None:
    assert fit.classify_load_failure(OPERATOR_OOM_TAIL) == "oom"


def test_a_genuine_architecture_failure_is_classified_as_arch() -> None:
    assert fit.classify_load_failure(ARCH_TAIL) == "arch"
    assert fit.classify_load_failure("E_MODEL_ARCH_UNSUPPORTED: no implementation") == "arch"


def test_an_empty_or_unknown_log_is_not_guessed() -> None:
    assert fit.classify_load_failure("") == "unknown"
    assert fit.classify_load_failure("llama_model_load: something odd happened") == "unknown"


def test_the_failed_allocation_size_is_read_from_the_log() -> None:
    assert fit.allocation_bytes_from_log(OPERATOR_OOM_TAIL) == 1058982400
    assert fit.allocation_bytes_from_log("no numbers here") is None


def test_the_codes_are_in_the_frozen_catalogs() -> None:
    assert "E_BACKEND_OOM" in ERROR_CODES
    assert "W_FIT_DOWNGRADE" in WARNING_CODES and "W_BACKEND_OOM" in WARNING_CODES
    assert BackendOomError("x").code == "E_BACKEND_OOM"
    assert BackendOomError("x").exit_code == 3


def test_the_oom_error_names_the_plan_the_numbers_and_the_hints() -> None:
    plan = fit.FitPlan(n_gpu_layers=4, n_ctx=4096, kv_type="f16", n_seq_max=8,
                       est_weights_bytes=4 * GIB, est_kv_bytes=576 * MIB,
                       est_total_bytes=4 * GIB + 576 * MIB, backend="vulkan",
                       source="estimate", budget_bytes=88 * MIB)
    error = fit.backend_oom_error(plan, free_bytes=1112 * MIB, needed_bytes=1058982400,
                                  log_tail=OPERATOR_OOM_TAIL,
                                  attempts=["n_gpu_layers=4 -> oom", "n_gpu_layers=2 -> oom"])
    message = str(error)
    assert error.code == "E_BACKEND_OOM"
    assert "1010 MiB" in message                      # the failed allocation, in MiB
    assert "1112 MiB" in message                      # what the driver reports as free
    assert "n_gpu_layers=4" in message and "kv_type=f16" in message
    assert "--no-fit" in message and "--fit-target" in message
    assert "Device memory allocation of size 1058982400 failed" in message


# ------------------------------------------------------------ R3: the ladder, through the loader
def test_the_log_capture_installs_and_restores_the_handler() -> None:
    backend = FakeBackend()
    with session_module.capture_llama_logs(backend) as lines:
        backend._installed(4, b"hello\n", None)
    assert lines == ["hello"]
    assert len(backend.log_set_calls) == 2            # ours, then the reset
    reset = ctypes.cast(backend.log_set_calls[1], ctypes.c_void_p)
    assert reset.value is None                        # reset with a NULL callback (see docstring)
    assert any(cb is backend.log_set_calls[0] for cb in ctypes_binding.live_log_callbacks())


def test_the_capture_never_reads_llama_log_get_back(tmp_path: pathlib.Path) -> None:
    """b11026's `llama_log_get` writes through two out-parameters; calling it as a no-arg getter
    segfaults the pinned bundle (measured). Production must not ask it anything."""
    calls: list[str] = []
    backend = FakeBackend()
    backend.llama.llama_log_get = lambda *args: calls.append("log_get")
    with session_module.capture_llama_logs(backend) as lines:
        backend._installed(4, b"x\n", None)
    assert calls == [] and lines == ["x"]


def test_a_load_that_oomed_degrades_to_cpu_and_succeeds(tmp_path: pathlib.Path) -> None:
    """fewer layers -> CPU-only; the handle names the placement and the warnings say why."""
    model_path, plan = gpu_plan(tmp_path)
    backend = FakeBackend(n_layer=4)

    with fake_runtime(tmp_path, backend):
        handle = session_module.open_model(model_path, runtime_dir=backend.directory,
                                           fit_plan=plan, free_probe=lambda: 1112 * MIB)
        try:
            assert backend.load_calls == [4, 2, 0]        # the ladder: half, then none
            assert handle.n_gpu_layers == 0
            assert handle.placement.degraded is True
            assert handle.placement.n_gpu_layers == 0
            assert set(handle.warnings) == {"W_BACKEND_OOM", "W_FIT_DOWNGRADE"}
            assert handle.placement.attempts == ("n_gpu_layers=4 -> oom", "n_gpu_layers=2 -> oom")
            assert "degraded" in handle.placement.note
            assert handle.fit_plan is not plan            # the plan that actually loaded
            assert handle.fit_plan.n_gpu_layers == 0
        finally:
            handle.close()


def test_a_fitting_load_keeps_the_plan_and_reports_no_degradation(tmp_path: pathlib.Path) -> None:
    model_path, plan = gpu_plan(tmp_path)
    backend = FakeBackend(n_layer=4, fail=lambda ngl, call: False)

    with fake_runtime(tmp_path, backend):
        handle = session_module.open_model(model_path, runtime_dir=backend.directory,
                                           fit_plan=plan, free_probe=lambda: 8 * GIB)
        try:
            assert backend.load_calls == [4]
            assert handle.n_gpu_layers == 4
            assert handle.placement.degraded is False
            assert handle.warnings == ()
            assert "4 layer(s) offloaded" in handle.placement.note
        finally:
            handle.close()


def test_no_fit_says_cpu_only_out_loud(tmp_path: pathlib.Path) -> None:
    """A run with `--no-fit` must not leave `n_gpu_layers: 0` to be interpreted by the reader."""
    model_path, _plan = gpu_plan(tmp_path)
    backend = FakeBackend(n_layer=4, fail=lambda ngl, call: False)

    with fake_runtime(tmp_path, backend):
        handle = session_module.open_model(model_path, runtime_dir=backend.directory,
                                           fit_plan=None, fit_disabled=True)
        try:
            assert handle.n_gpu_layers == 0
            assert handle.placement.degraded is False
            assert "fit disabled" in handle.placement.note
            # the note is about the weights, not the compute path (card t_603a35a0)
            assert "--no-fit" in handle.placement.note
            assert "no layers offloaded" in handle.placement.note
        finally:
            handle.close()


def test_a_silent_failure_still_gets_one_cpu_retry_then_reports_the_arch_error(
        tmp_path: pathlib.Path) -> None:
    """A log that names no cause is not an OOM — but a CPU run is still worth one attempt."""
    model_path, plan = gpu_plan(tmp_path)
    backend = FakeBackend(n_layer=4, log_lines=tuple(ARCH_TAIL.splitlines()),
                          fail=lambda ngl, call: True)      # a real arch problem breaks the CPU too

    with fake_runtime(tmp_path, backend), pytest.raises(ModelArchUnsupportedError) as excinfo:
        session_module.open_model(model_path, runtime_dir=backend.directory, fit_plan=plan)
    assert backend.load_calls == [4, 0]                   # one CPU retry, no ladder walk
    assert excinfo.value.code == "E_MODEL_ARCH_UNSUPPORTED"
    assert "placements tried" in str(excinfo.value)
    assert "n_gpu_layers=4 -> arch" in str(excinfo.value)
    assert "n_gpu_layers=0 -> arch" in str(excinfo.value)
    assert "unknown model architecture" in str(excinfo.value)


def test_a_failure_that_cpu_can_load_is_a_success_even_when_the_log_blames_the_arch(
        tmp_path: pathlib.Path) -> None:
    """Never a hard failure when a CPU path works — the classification decides the message, the
    ladder decides the outcome."""
    model_path, plan = gpu_plan(tmp_path)
    backend = FakeBackend(n_layer=4, log_lines=tuple(ARCH_TAIL.splitlines()))

    with fake_runtime(tmp_path, backend):
        handle = session_module.open_model(model_path, runtime_dir=backend.directory,
                                           fit_plan=plan, free_probe=lambda: 1112 * MIB)
        try:
            assert backend.load_calls == [4, 0]
            assert handle.n_gpu_layers == 0
            assert handle.placement.degraded is True
            assert handle.placement.attempts == ("n_gpu_layers=4 -> arch",)
            assert "W_FIT_DOWNGRADE" in handle.warnings
        finally:
            handle.close()


def test_a_quiet_failure_that_cpu_can_load_is_a_degraded_success(tmp_path: pathlib.Path) -> None:
    """The backend printed nothing but only offload fails: the CPU retry is the answer."""
    model_path, plan = gpu_plan(tmp_path)
    backend = FakeBackend(n_layer=4, log_lines=())

    with fake_runtime(tmp_path, backend):
        handle = session_module.open_model(model_path, runtime_dir=backend.directory,
                                           fit_plan=plan, free_probe=lambda: 1112 * MIB)
        try:
            assert backend.load_calls == [4, 0]
            assert handle.n_gpu_layers == 0
            assert "W_FIT_DOWNGRADE" in handle.warnings
            assert "W_BACKEND_OOM" not in handle.warnings     # nothing claimed an allocation fail
            assert handle.placement.attempts == ("n_gpu_layers=4 -> unknown",)
        finally:
            handle.close()


def test_when_nothing_fits_the_error_is_backend_oom_with_both_numbers(
        tmp_path: pathlib.Path) -> None:
    """Every rung fails: the code must be `E_BACKEND_OOM`, never the arch code."""
    model_path, plan = gpu_plan(tmp_path)
    backend = FakeBackend(n_layer=4, fail=lambda ngl, call: True)

    with fake_runtime(tmp_path, backend), pytest.raises(BackendOomError) as excinfo:
        session_module.open_model(model_path, runtime_dir=backend.directory, fit_plan=plan,
                                  free_probe=lambda: 1112 * MIB)
    message = str(excinfo.value)
    assert excinfo.value.code == "E_BACKEND_OOM"
    assert "E_MODEL_ARCH_UNSUPPORTED" not in message
    assert backend.load_calls == [4, 2, 0]                # the whole layer ladder was walked
    assert "1112 MiB" in message and "1010 MiB" in message
    assert "--no-fit" in message and "--fit-target" in message
    assert "3 placement(s)" in message                    # the plan + 2 ladder steps


def test_degrade_disabled_does_not_walk_the_ladder(tmp_path: pathlib.Path) -> None:
    model_path, plan = gpu_plan(tmp_path)
    backend = FakeBackend(n_layer=4, fail=lambda ngl, call: True)

    with fake_runtime(tmp_path, backend), pytest.raises(BackendOomError):
        session_module.open_model(model_path, runtime_dir=backend.directory, fit_plan=plan,
                                  degrade=False, free_probe=lambda: 0)
    assert backend.load_calls == [4]                      # degrade=False: no second attempt


# --------------------------------------------------- R3/R4 reach the response, not just the handle
def placement_session(placement: Any, **kwargs: Any) -> FakeSession:
    """A fake session whose engine surface carries a Placement (the loader's own output)."""

    class PlacedSession(FakeSession):
        @property
        def meta(self) -> SessionMeta:
            return SessionMeta(runtime=self.runtime, backend=self.backend, n_ctx=self.n_ctx,
                               n_seq_max=self.n_seq_max, kv_unified=True, threads=self.threads,
                               n_vocab=self.n_vocab, model_path=self.model_path,
                               model_alias=self.model_alias, load_ms=self.load_ms,
                               placement=placement,
                               placement_warnings=tuple(placement.warnings or ()),
                               **kwargs)

    return PlacedSession(row_fn=lambda ctx: biased_row(64, {1: 10.0}))


def test_the_engine_surface_names_the_placement_and_its_warnings() -> None:
    placement = session_module.Placement(note="degraded after a backend allocation failure: "
                                               "0 layer(s) offloaded, kv_type=f16",
                                         n_gpu_layers=0, kv_type="f16", degraded=True,
                                         attempts=("n_gpu_layers=4 -> oom",),
                                         warnings=("W_BACKEND_OOM", "W_FIT_DOWNGRADE"))
    session = placement_session(placement)
    request = schema.parse_request({
        "state": "The billing dashboard is blank for every user after login.",
        "questions": {"area": {"type": "choice", "instructions": "Which area owns this?",
                               "criteria": {"billing": "payments and invoices",
                                            "technical": "api and infrastructure"}}},
    })
    result = decide_request(request, session)

    assert result.engine["placement"]["degraded"] is True
    assert result.engine["placement"]["n_gpu_layers"] == 0
    assert result.engine["placement"]["attempts"] == ["n_gpu_layers=4 -> oom"]
    assert result.engine["placement"]["note"].startswith("degraded after a backend")
    assert {"W_BACKEND_OOM", "W_FIT_DOWNGRADE"} <= set(result.warnings)
    assert result.engine["n_gpu_layers"] == 0


def test_a_plain_session_reports_no_placement_instead_of_guessing() -> None:
    session = FakeSession(row_fn=lambda ctx: biased_row(64, {1: 10.0}))
    request = schema.parse_request({
        "state": "S", "questions": {"q": {"type": "choice", "criteria": {"a": None, "b": None}}}})
    result = decide_request(request, session)
    assert result.engine["placement"] is None
    assert "W_BACKEND_OOM" not in result.warnings


def test_an_auto_kv_type_still_gets_a_context() -> None:
    """`auto` is the default of every request: the ladder must start at f16, not be empty.

    An empty ladder means the context-init loop never runs and the run dies with the generic
    "runtime refused these context parameters" — measured with a real `--no-fit` run against the
    pinned CPU bundle.
    """
    from typed_gguf.engine.session import _kv_ladder
    assert _kv_ladder("auto", degrade=True) == ["f16", "q8_0", "q4_0"]
    assert _kv_ladder("auto", degrade=False) == ["f16"]
    assert _kv_ladder("q8_0", degrade=True) == ["q8_0", "q4_0"]
    assert _kv_ladder("q4_0", degrade=True) == ["q4_0"]
    assert _kv_ladder("nonsense", degrade=True) == ["f16", "q8_0", "q4_0"]


def test_a_context_that_the_runtime_refuses_reports_the_backend_tail(
        tmp_path: pathlib.Path) -> None:
    """A non-memory context failure keeps the pinned code AND carries what the backend said."""
    from typed_gguf.engine.decide import ContextPlan

    model_path, plan = gpu_plan(tmp_path)
    backend = FakeBackend(n_layer=4, fail=lambda ngl, call: False)
    backend.llama.llama_init_from_model = lambda model, params: 0
    backend.llama.llama_context_default_params = lambda: SimpleNamespace(
        n_ctx=0, n_batch=0, n_ubatch=0, n_seq_max=0, n_threads=0, n_threads_batch=0,
        type_k=0, type_v=0, kv_unified=False, no_perf=True, flash_attn_type=0)
    backend.llama.llama_log_set = backend.log_set_calls.append

    def install(callback: Any, _data: Any) -> None:
        backend._installed = callback

    backend.llama.llama_log_set = install
    context_plan = ContextPlan(n_ctx=512, n_seq_max=3, threads=1, kv_type="auto",
                               prefix_tokens=(1, 2, 3))

    with fake_runtime(tmp_path, backend):
        handle = session_module.open_model(model_path, runtime_dir=backend.directory,
                                           fit_plan=plan, degrade=False)
        try:
            with pytest.raises(RuntimeMissingError) as excinfo:
                session_module.ModelSession(handle, context_plan)
        finally:
            handle.close()
    message = str(excinfo.value)
    assert excinfo.value.code == "E_RUNTIME_MISSING"
    assert "n_ctx=512" in message and "kv_type=auto" in message
    assert "(unknown)" in message                     # the classification of a silent log


def test_the_context_init_notes_the_rung_it_resolved_auto_to(tmp_path: pathlib.Path) -> None:
    """`auto` + a healthy context = NO downgrade warning, and the used rung is reported.

    The first version compared the resolved rung (`f16`) against the request's literal value
    (`auto`) and warned `W_KV_TYPE_DOWNGRADE` on every default run — caught by running the real
    `typed-gguf ask --no-fit` against the pinned bundle.
    """
    from typed_gguf.engine.decide import ContextPlan

    model_path, plan = gpu_plan(tmp_path)
    backend = FakeBackend(n_layer=4, fail=lambda ngl, call: False)
    backend.llama.llama_context_default_params = lambda: SimpleNamespace(
        n_ctx=0, n_batch=0, n_ubatch=0, n_seq_max=0, n_threads=0, n_threads_batch=0,
        type_k=0, type_v=0, kv_unified=False, no_perf=True, flash_attn_type=0)
    backend.llama.llama_init_from_model = lambda model, params: 7
    backend.llama.llama_get_memory = lambda ctx: 8
    backend.llama.llama_free = lambda ctx: None
    backend.llama.llama_n_ctx = lambda ctx: 512
    backend.llama.llama_n_seq_max = lambda ctx: 3
    context_plan = ContextPlan(n_ctx=512, n_seq_max=3, threads=1, kv_type="auto",
                               prefix_tokens=(1, 2, 3))

    with fake_runtime(tmp_path, backend):
        handle = session_module.open_model(model_path, runtime_dir=backend.directory,
                                           fit_plan=plan, degrade=False)
        try:
            live = session_module.ModelSession(handle, context_plan)
            try:
                assert live.kv_type == "auto"          # what the request asked for
                assert live.kv_type_used == "f16"      # what the runtime got
                assert live.extra_warnings == []       # auto -> f16 is not a downgrade
                assert live.meta.kv_type == "auto"
                assert live.meta.placement.kv_type == "f16"
            finally:
                live.close()
        finally:
            handle.close()


def test_the_cli_fit_json_carries_the_free_number_and_a_bounded_plan(
        tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end through the CLI surface the operator runs: the JSON says what it planned."""
    from typed_gguf import cli

    model = write_gguf(tmp_path / "synthetic.gguf")
    monkeypatch.setenv("TYPED_GGUF_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("TYPED_GGUF_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(fit, "host_facts", lambda **kwargs: gpu_host(1112))
    assert cli.main(["fit", str(model), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["host"]["vram_free_bytes"] == 1112 * MIB
    assert payload["host"]["vram_bytes"] == 8 * GIB
    assert payload["budget_bytes"] == max(0, 1112 * MIB - fit.DEFAULT_FIT_TARGET_MB * MIB)
    assert payload["n_gpu_layers"] == 0
    assert "W_FIT_DOWNGRADE" in payload["warnings"]
    assert any("free" in note for note in payload["notes"])


def test_the_device_memory_probe_is_the_default_free_source(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recommend, "device_memory",
                        lambda **kwargs: recommend.DeviceMemory(8 * GIB, 512 * MIB, "injected"))
    assert session_module.default_free_device_bytes() == 512 * MIB
