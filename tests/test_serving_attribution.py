"""E3 FIX (card t_80f1a4c6): the *serving* path must say what computed.

The E3 campaign (`docs/evidence/e3_batch.json`) ran one `ggufone run` over 20 dev-set questions
with `--backend vulkan --threads 4 --n-seq-max 4` on the pinned b11026 Vulkan bundle and answered
with

    "engine": {"backend": "cpu", "n_gpu_layers": 3, ...}

while the same process' stderr carried

    ~llama_context:    Vulkan0 compute buffer size is 363.5412 MiB, matches expectation of …

The value came from `session.runtime_backend()` — the *install record*'s
`backend_working`/`backend_requested`, silently defaulting to `cpu` when no record is visible (the
E3 run had a scratch `$HOME`, so `runtime.json` was not) — and the serving path never read the
engine's own log at all. Card t_603a35a0 fixed exactly this class for the **bench** path
(`harness.device_usage`: `devices`, `device_buffers`, `effective_backend`, `W_BACKEND_MISMATCH`
when the claim is refuted). This file pins the same rules for `run`/`ask`/`calibrate`, which all
build their response through `decide.DecisionEngine`.

Three claims are pinned here:

* the response's device set, per-device compute-buffer counts and effective backend come from the
  engine's own log (`session.ModelSession.device_log`), never from the fit plan, the request or
  the install record;
* `engine.backend` is a *claim* whose source is named (`engine.backend_source`): what the request
  asked for, else the backend the loaded bundle carries, else the install record, else `cpu`;
* a claim the log refutes — or cannot corroborate — is `W_BACKEND_MISMATCH`, and a silent log
  reads as *unverified* (`effective_backend: null`), never as a claim.

Vehicles: the model-free `tests.fake_engine.FakeSession` carrying the operator's own lines as
`device_log`, real bundle directories on disk for the claim, and `test_fit_oom_recovery`'s fake
runtime (a real `llama_log_set` ABI + a real bundle directory) for the live plumbing.
"""
from __future__ import annotations

import contextlib
import pathlib
import textwrap
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from ggufone import cli, schema
from ggufone.engine import decide
from ggufone.engine import session as session_module
from ggufone.errors import WARNING_CODES
from ggufone.runtime import finder
from tests.fake_engine import FakeSession

#: The E3 batch's own lines (`/work/e3scratch/batch.log`, Occamy 1.0 on the Vulkan bundle): the
#: loader's device/backend lines are requests the backend *loads* — the compute buffers below are
#: the measurement (rule 1 of `runtime/devices.py`). The two `~llama_context:` lines are the
#: operator's verbatim output, hence the `noqa`s.
E3_VULKAN_LOG = "\n".join((
    "load_backend: loaded RPC backend from …/b11026-linux-x64-vulkan/libggml-rpc.so",
    "ggml_vulkan: 0 = NVIDIA GeForce RTX 3060 Ti (NVIDIA) | uma: 0 | fp16: 1 | bf16: 1 | fp4: 0",
    "load_backend: loaded Vulkan backend from …/b11026-linux-x64-vulkan/libggml-vulkan.so",
    "load_backend: loaded CPU backend from …/b11026-linux-x64-vulkan/libggml-cpu-haswell.so",
    "sched_reserve:    Vulkan0 compute buffer size =   363.54 MiB",
    "~llama_context:    Vulkan0 compute buffer size is 363.5412 MiB, matches expectation of 363.5412 MiB",  # noqa: E501
    "~llama_context: Vulkan_Host compute buffer size is  17.4555 MiB, matches expectation of  17.4555 MiB",  # noqa: E501
)) + "\n"

#: A pure CPU bundle's own lines (the E2 provenance probe's `probe-cpubundle.raw` shape).
CPU_LOG = textwrap.dedent("""\
    load_tensors:         CPU model buffer size =  4167.21 MiB
    sched_reserve:        CPU compute buffer size =   166.26 MiB
    ~llama_context:        CPU compute buffer size is 166.2610 MiB
""")

#: Op offload on an accelerator bundle: the weights stay on the host, the *graph* runs on Vulkan.
#: This is the row shape the E2 audit measured at 587.9 tok/s under a `cpu` label.
OP_OFFLOAD_LOG = textwrap.dedent("""\
    load_tensors:         CPU_Mapped model buffer size =  4167.21 MiB
    sched_reserve:    Vulkan0 compute buffer size =   163.13 MiB
    ~llama_context:    Vulkan0 compute buffer size is 163.1250 MiB
""")

#: The request side, never evidence: `offloaded 37/37 layers to GPU` printed immediately before
#: the mixed-bundle run walked every layer onto the CPU (card t_603a35a0).
OFFLOAD_REQUEST_LOG = textwrap.dedent("""\
    load_tensors: offloaded 37/37 layers to GPU
    load_tensors: layer   0 assigned to device CPU, is_swa = 1
    ~llama_context:        CPU compute buffer size is 166.2610 MiB
""")

#: The silent case: an engine that produced no device line at all (the mixed-bundle second bundle).
SILENT_LOG = ""


def request_payload(*options: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for block in options:
        merged.update(block)
    return {
        "state": "The billing dashboard is blank for every user after login.",
        "questions": {"area": {"type": "choice", "instructions": "Which team owns this?",
                               "criteria": {"billing": "payments and invoices",
                                            "technical": "api and infrastructure"}}},
        "options": merged or None,
    }


def serving(device_log: str, *, backend: str = "cpu", source: str = "record",
            **options: Any) -> dict[str, Any]:
    """One serving response over the fake session — the fields `run`/`ask` publish."""
    payload = request_payload(options)
    payload = {key: value for key, value in payload.items() if value is not None}
    request = schema.parse_request(payload)
    session = FakeSession(n_vocab=512, backend=backend)
    session.device_log = device_log
    session.backend_source = source
    result = decide.decide_request(request, session)
    return schema.render_response(result.payload(), format=request.format)


def bundle(tmp_path: pathlib.Path, name: str, *libraries: str) -> pathlib.Path:
    """A real bundle directory on disk: `libllama.so` plus whatever ggml backends it carries."""
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / finder.library_names()["llama"]).write_bytes(b"")
    for library in libraries:
        (directory / library).write_bytes(b"")
    return directory


# ------------------------------------------------- requirement 1: the engine's own evidence
def test_the_serving_response_carries_the_device_evidence_the_engine_logged() -> None:
    """The E3 response must read `devices` / `device_buffers` / `effective_backend` from the log."""
    engine = serving(E3_VULKAN_LOG, backend="vulkan", source="request")["engine"]
    assert engine["devices"] == ["Vulkan0", "Vulkan_Host"]
    assert engine["device_buffers"] == {"Vulkan0": 2, "Vulkan_Host": 1}
    assert engine["effective_backend"] == "vulkan"


def test_a_cpu_only_box_reads_as_cpu() -> None:
    body = serving(CPU_LOG, backend="cpu")
    assert body["engine"]["effective_backend"] == "cpu"
    assert body["engine"]["device_buffers"] == {"CPU": 2}
    assert "W_BACKEND_MISMATCH" not in body["warnings"]


def test_the_request_side_offload_line_is_never_evidence() -> None:
    """`offloaded N/M layers to GPU` is a request; the CPU compute buffer is the measurement."""
    body = serving(OFFLOAD_REQUEST_LOG, backend="cpu")
    assert body["engine"]["effective_backend"] == "cpu"
    assert "W_BACKEND_MISMATCH" not in body["warnings"]


def test_the_device_evidence_is_not_read_from_the_fit_plan_or_the_record() -> None:
    """The fit plan says `backend: cpu` (it was built for the fit host) — the log says Vulkan.

    A response that keeps reading the plan/record for its device fields cannot pass this.
    """
    body = serving(E3_VULKAN_LOG, backend="cpu", source="record")
    assert body["engine"]["effective_backend"] == "vulkan"
    assert body["engine"]["device_buffers"]["Vulkan0"] == 2


# ------------------------------------------------- requirement 2: a refuted claim is named
def test_a_claim_the_log_refutes_is_a_named_warning() -> None:
    """The E3 lie: a `cpu` label while `Vulkan0 compute buffer size` lines show the device ran."""
    body = serving(E3_VULKAN_LOG, backend="cpu", source="record")
    assert body["engine"]["backend"] == "cpu"            # the claim is not silently rewritten
    assert body["engine"]["effective_backend"] == "vulkan"
    assert "W_BACKEND_MISMATCH" in body["warnings"]
    assert "W_BACKEND_MISMATCH" in WARNING_CODES


def test_an_accelerator_claim_the_log_cannot_corroborate_is_flagged() -> None:
    """A `vulkan` label with no compute-buffer line at all is not publishable either."""
    body = serving(SILENT_LOG, backend="vulkan", source="request")
    assert body["engine"]["effective_backend"] is None
    assert body["engine"]["devices"] == []
    assert "W_BACKEND_MISMATCH" in body["warnings"]


def test_a_silent_log_reads_as_unverified_never_as_a_claim() -> None:
    """`effective_backend: null` + the named source — not `cpu`, and not a warning either.

    Nothing computed anywhere else does not refute a `cpu` claim (the bench rule), so a box whose
    log carries no buffer line reads *unverified*, with the claim's source visible.
    """
    body = serving(SILENT_LOG, backend="cpu", source="default")
    engine = body["engine"]
    assert engine["effective_backend"] is None
    assert engine["devices"] == [] and engine["device_buffers"] == {}
    assert engine["backend"] == "cpu" and engine["backend_source"] == "default"
    assert "W_BACKEND_MISMATCH" not in body["warnings"]


def test_a_verified_vulkan_run_is_not_flagged() -> None:
    body = serving(E3_VULKAN_LOG, backend="vulkan", source="bundle")
    assert body["engine"]["backend_source"] == "bundle"
    assert "W_BACKEND_MISMATCH" not in body["warnings"]


def test_a_log_kept_as_lines_reads_like_the_joined_text() -> None:
    """`device_log` may be a sequence of lines, not one string — the loader's own shape.

    `ModelSession.device_log` joins its lines (`ModelHandle.load_log` + the context's), but the
    evidence helper accepts a session-like object directly, and the loader's capture *is* a tuple:
    reading a sequence as text would drop exactly the lines this card exists to read (the mutation
    sweep's `device_evidence` survivors 16/17 pin this branch — card t_80f1a4c6, Tier M).
    """
    lines = tuple(line for line in E3_VULKAN_LOG.splitlines() if line)
    from_lines = decide.device_evidence(SimpleNamespace(device_log=lines), "vulkan")
    from_text = decide.device_evidence(SimpleNamespace(device_log=E3_VULKAN_LOG), "vulkan")
    assert from_lines == from_text
    assert from_lines["effective_backend"] == "vulkan"
    assert from_lines["device_buffers"] == {"Vulkan0": 2, "Vulkan_Host": 1}
    assert from_lines["warnings"] == []


def test_a_session_without_a_log_reads_as_unverified() -> None:
    """No `device_log` attribute at all is an *empty* log (unverified), not an exception.

    The `getattr` default is the silent case of requirement 1 — a session object that predates the
    evidence capture (or a fake in an older gate) must not turn the response into a traceback. The
    mutation sweep's survivor 8 (a `getattr` with no default) is the mutant this pins.
    """
    evidence = decide.device_evidence(SimpleNamespace(), "cpu")
    assert evidence == {"devices": [], "device_buffers": {}, "effective_backend": None,
                        "warnings": []}


# ------------------------------------------------- the claim, and where it comes from
def test_the_claim_is_what_the_run_asked_for(tmp_path: pathlib.Path) -> None:
    cpu_bundle = bundle(tmp_path, "b11026-linux-x64-cpu")
    claim = session_module.backend_claim(requested="vulkan", runtime_dir=cpu_bundle)
    assert (claim.backend, claim.source) == ("vulkan", "request")


def test_the_claim_falls_back_to_the_bundle_that_loaded(tmp_path: pathlib.Path) -> None:
    vulkan = bundle(tmp_path, "b11026-linux-x64-vulkan", "libggml-vulkan.so")
    cpu = bundle(tmp_path, "b11026-linux-x64-cpu")
    assert session_module.backend_claim(requested="auto", runtime_dir=vulkan) == \
        session_module.BackendClaim(backend="vulkan", source="bundle")
    assert session_module.backend_claim(runtime_dir=cpu) == \
        session_module.BackendClaim(backend="cpu", source="bundle")
    # a directory that is not a bundle at all cannot name a backend
    assert session_module.backend_claim(runtime_dir=tmp_path / "nowhere").source != "bundle"


def test_the_claim_falls_back_to_the_install_record_then_to_cpu(tmp_path: pathlib.Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    finder.write_runtime_record({"backend_requested": "cuda", "backend_working": "vulkan",
                                 "dir": str(tmp_path / "gone")}, home=home)
    recorded = session_module.backend_claim(home=home)
    assert (recorded.backend, recorded.source) == ("vulkan", "record")
    default = session_module.backend_claim(home=tmp_path / "empty-home")
    assert (default.backend, default.source) == ("cpu", "default")


def test_a_record_without_a_working_backend_is_not_a_claim(tmp_path: pathlib.Path) -> None:
    """An unprobed install record (`backend_working: null`) names nothing — same as no record."""
    home = tmp_path / "home"
    home.mkdir()
    finder.write_runtime_record({"backend_requested": "cuda", "backend_working": None,
                                 "dir": str(tmp_path / "gone")}, home=home)
    claim = session_module.backend_claim(home=home)
    assert (claim.backend, claim.source) != ("cuda", "record")


def test_the_bundle_classifier_refuses_a_platform_it_has_no_rule_for() -> None:
    """Same rule as `finder.library_names`: an unknown platform is an error, never a guess."""
    from ggufone.errors import RuntimeMissingError

    with pytest.raises(RuntimeMissingError) as excinfo:
        finder.accelerator_names("plan9")
    assert "E_RUNTIME_MISSING" in str(excinfo.value)
    assert finder.accelerator_names("linux")[0] == ("vulkan", "libggml-vulkan.so")


# ------------------------------------------------- the live plumbing (engine side, no GPU)
def test_the_live_session_records_the_load_and_the_context_lines(tmp_path: pathlib.Path) -> None:
    """`open_model` keeps the *successful* load's lines; the session adds its context's.

    The vehicle is `tests.test_fit_oom_recovery`'s fake runtime, whose lambdas print through the
    real `llama_log_set` ABI (card t_8cb0a05e): the same two captures a serving run collects.
    """
    from ggufone.engine.decide import ContextPlan
    from ggufone.runtime import fit
    from tests.test_fit import write_gguf
    from tests.test_fit_oom_recovery import FakeBackend, fake_runtime

    model_path = write_gguf(tmp_path / "model.gguf", n_layer=4)
    backend = FakeBackend(n_layer=4, fail=lambda ngl, call: False)

    def load(path: bytes, params: object) -> int:
        backend._installed(4, b"load_tensors:         CPU model buffer size =  4167.21 MiB\n", None)
        return 1

    def init(model: object, params: object) -> int:
        backend._installed(4, b"sched_reserve:    Vulkan0 compute buffer size =   545.31 MiB\n",
                           None)
        return 7

    backend.llama.llama_model_load_from_file = load
    backend.llama.llama_context_default_params = lambda: SimpleNamespace(
        n_ctx=0, n_batch=0, n_ubatch=0, n_seq_max=0, n_threads=0, n_threads_batch=0,
        type_k=0, type_v=0, kv_unified=False, no_perf=True, flash_attn_type=0)
    backend.llama.llama_init_from_model = init
    backend.llama.llama_get_memory = lambda ctx: 8
    backend.llama.llama_free = lambda ctx: None
    backend.llama.llama_n_ctx = lambda ctx: 512
    backend.llama.llama_n_seq_max = lambda ctx: 3

    with fake_runtime(tmp_path, backend):
        handle = session_module.open_model(model_path, runtime_dir=backend.directory,
                                           fit_plan=fit.coerce_plan(SimpleNamespace(
                                               n_gpu_layers=0, kv_type="auto")),
                                           log=None)
        try:
            assert handle.load_log == ("load_tensors:         CPU model buffer size =  4167.21 MiB",
                                       )
            live = session_module.ModelSession(
                handle, ContextPlan(n_ctx=512, n_seq_max=3, threads=1, kv_type="auto",
                                    prefix_tokens=(1, 2, 3)), backend="vulkan",
                backend_source="request")
            log = live.device_log
            live.close()
        finally:
            handle.close()

    from ggufone.runtime import devices as devices_module

    usage = devices_module.parse_device_usage(log)
    assert usage.model_buffers == {"CPU": 1}          # the load's own line
    assert usage.compute_buffers == {"Vulkan0": 1}    # the context's own line
    assert usage.effective == "vulkan"
    assert live.backend == "vulkan" and live.backend_source == "request"
    assert live.meta.backend == "vulkan" and live.meta.backend_source == "request"


def test_the_live_session_names_the_bundle_it_loaded(tmp_path: pathlib.Path) -> None:
    """No `--backend` flag: the claim is the backend the bundle this run loaded carries."""
    from ggufone.engine.decide import ContextPlan
    from ggufone.runtime import fit
    from tests.test_fit import write_gguf
    from tests.test_fit_oom_recovery import FakeBackend, fake_runtime

    model_path = write_gguf(tmp_path / "model.gguf", n_layer=4)
    backend = FakeBackend(n_layer=4, fail=lambda ngl, call: False)
    backend.llama.llama_context_default_params = lambda: SimpleNamespace(
        n_ctx=0, n_batch=0, n_ubatch=0, n_seq_max=0, n_threads=0, n_threads_batch=0,
        type_k=0, type_v=0, kv_unified=False, no_perf=True, flash_attn_type=0)
    backend.llama.llama_init_from_model = lambda model, params: 7
    backend.llama.llama_get_memory = lambda ctx: 8
    backend.llama.llama_free = lambda ctx: None
    backend.llama.llama_n_ctx = lambda ctx: 512
    backend.llama.llama_n_seq_max = lambda ctx: 3

    with fake_runtime(tmp_path, backend):
        (backend.directory / "libggml-vulkan.so").write_bytes(b"")
        handle = session_module.open_model(model_path, runtime_dir=backend.directory,
                                          fit_plan=fit.coerce_plan(SimpleNamespace(
                                              n_gpu_layers=0, kv_type="auto")))
        try:
            live = session_module.ModelSession(
                handle, ContextPlan(n_ctx=512, n_seq_max=3, threads=1, kv_type="auto",
                                    prefix_tokens=(1, 2, 3)))
            assert (live.backend, live.backend_source) == ("vulkan", "bundle")
            live.close()
        finally:
            handle.close()


@contextlib.contextmanager
def live_session(tmp_path: pathlib.Path, **session_kwargs: Any) -> Iterator[Any]:
    """A real `ModelSession` over `test_fit_oom_recovery`'s fake llama runtime.

    The two gates above build this scaffold inline; the two below need the same world with
    different constructor kwargs (an explicit backend, a caller-owned log sink), so the harness is
    shared while the *observations* stay per-gate. The bundle on disk carries `libggml-vulkan.so`,
    which is what makes "the claim was not resolved" observable.
    """
    from ggufone.engine.decide import ContextPlan
    from ggufone.runtime import fit
    from tests.test_fit import write_gguf
    from tests.test_fit_oom_recovery import FakeBackend, fake_runtime

    model_path = write_gguf(tmp_path / "model.gguf", n_layer=4)
    backend = FakeBackend(n_layer=4, fail=lambda ngl, call: False)

    def init(model: object, params: object) -> int:
        backend._installed(4, b"sched_reserve:    Vulkan0 compute buffer size =   545.31 MiB\n",
                           None)
        return 7

    backend.llama.llama_context_default_params = lambda: SimpleNamespace(
        n_ctx=0, n_batch=0, n_ubatch=0, n_seq_max=0, n_threads=0, n_threads_batch=0,
        type_k=0, type_v=0, kv_unified=False, no_perf=True, flash_attn_type=0)
    backend.llama.llama_init_from_model = init
    backend.llama.llama_get_memory = lambda ctx: 8
    backend.llama.llama_free = lambda ctx: None
    backend.llama.llama_n_ctx = lambda ctx: 512
    backend.llama.llama_n_seq_max = lambda ctx: 3

    with fake_runtime(tmp_path, backend):
        (backend.directory / "libggml-vulkan.so").write_bytes(b"")
        handle = session_module.open_model(model_path, runtime_dir=backend.directory,
                                          fit_plan=fit.coerce_plan(SimpleNamespace(
                                              n_gpu_layers=0, kv_type="auto")))
        try:
            live = session_module.ModelSession(
                handle, ContextPlan(n_ctx=512, n_seq_max=3, threads=1, kv_type="auto",
                                    prefix_tokens=(1, 2, 3)), **session_kwargs)
            try:
                yield live
            finally:
                live.close()
        finally:
            handle.close()


def test_an_explicit_backend_with_no_source_is_named_explicit(tmp_path: pathlib.Path) -> None:
    """`backend=` without `backend_source=`: the label is the caller's, so the source is `explicit`.

    The loaded bundle carries Vulkan, so a session that resolved the claim anyway — instead of
    honouring the explicit argument — would answer `vulkan`/`bundle`. The sweep's mutants 7/18/20/21
    of `ModelSession.__init__` are exactly that path: with the caller naming a backend, the claim is
    not resolved, and `backend_source` is the contract that says so.
    """
    with live_session(tmp_path, backend="cpu") as live:
        assert live.backend == "cpu"
        assert live.backend_source == "explicit"


def test_a_caller_owned_log_sink_receives_the_context_lines(tmp_path: pathlib.Path) -> None:
    """`log=` is a caller-owned sink (the bench keeps one): the context's lines must reach it.

    Pins the sweep's `ModelSession.__init__` mutant 47 (`log.extend(None)`), which only a session
    built *with* a sink can see.
    """
    sink: list[str] = []
    with live_session(tmp_path, backend="vulkan", backend_source="request", log=sink) as live:
        assert any("compute buffer size" in line for line in sink), "the sink saw no context line"
        assert "compute buffer size" in live.device_log


# ------------------------------------------------- the CLI glue: `decide_payload`
def test_the_serving_payload_hands_the_claim_and_the_log_to_the_session(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """`decide_payload` is the one place that knows the request: it must claim its backend.

    The session is the real `FakeSession` (so the real `DecisionEngine` runs); only the two
    session-construction calls are replaced, and both are asserted.
    """
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    directory = bundle(tmp_path, "b11026-linux-x64-vulkan", "libggml-vulkan.so")
    seen: dict[str, Any] = {}

    class RecordingSession(FakeSession):
        def __init__(self, handle: Any, plan: Any, **kwargs: Any) -> None:
            seen["session_kwargs"] = kwargs
            super().__init__(n_vocab=512, backend=kwargs.get("backend") or "cpu",
                             device_log=E3_VULKAN_LOG,
                             backend_source=kwargs.get("backend_source", ""))

        def __enter__(self) -> RecordingSession:
            return self

        def __exit__(self, *exc: object) -> None:
            self.close()

    class FakeHandle:
        """The `ModelHandle` surface `decide_payload` touches (no runtime involved)."""

        def __init__(self) -> None:
            self.runtime = SimpleNamespace(directory=directory)
            self.path = str(model)
            self.n_vocab = 512
            self.load_ms = 1.0
            self.n_layer = 4
            self.placement = None

        def tokenize(self, text: str, *, add_special: bool = False) -> list[int]:
            return [1, 2, 3]

        def close(self) -> None:
            pass

        def __enter__(self) -> FakeHandle:
            return self

        def __exit__(self, *exc: object) -> None:
            self.close()

    def fake_open_model(path: Any, **kwargs: Any) -> Any:
        seen["open_kwargs"] = kwargs
        return FakeHandle()

    monkeypatch.setattr(session_module, "open_model", fake_open_model)
    monkeypatch.setattr(session_module, "ModelSession", RecordingSession)
    monkeypatch.setattr(cli, "load_calibration_for", lambda model_path, **kwargs: None)

    payload = request_payload({"backend": "vulkan", "threads": 4})
    payload["model"] = str(model)
    body = cli.decide_payload(payload, home=tmp_path / "home", fit_enabled=False)

    assert seen["session_kwargs"]["backend"] == "vulkan"
    assert seen["session_kwargs"]["backend_source"] == "request"
    assert body["engine"]["backend"] == "vulkan"
    assert body["engine"]["backend_source"] == "request"
    assert body["engine"]["effective_backend"] == "vulkan"     # from the session's own log
    assert body["engine"]["device_buffers"] == {"Vulkan0": 2, "Vulkan_Host": 1}
    assert body["engine"]["backend_source"] in session_module.CLAIM_SOURCES


# --------------------------------------------------------- the live row (pinned bundle)
def _runtime_dir() -> pathlib.Path:
    """The bundle the live gate runs (`$GGUFONE_RUNTIME_DIR`, else the installed one)."""
    import os
    env = os.environ.get("GGUFONE_RUNTIME_DIR")
    if env and (pathlib.Path(env) / finder.library_names()["llama"]).exists():
        return pathlib.Path(env)
    found = finder.find_runtime()
    if found:
        return found
    pytest.skip("no llama.cpp runtime on this box (set GGUFONE_RUNTIME_DIR)")


#: the operator's smallest local GGUFs, in preference order (a live row never downloads one)
LIVE_MODELS = ("Qwen3.5-4B-Q4_0.gguf", "Spark-X2.5-4B-Q8_0.gguf",
               "Accio-Lab_occamy-1.0-Q4_K_L.gguf")


def _live_model() -> pathlib.Path:
    """`GGUFONE_ATTRIB_MODEL`, else the smallest known local GGUF (the card's is Occamy)."""
    import os
    explicit = os.environ.get("GGUFONE_ATTRIB_MODEL")
    candidates = [pathlib.Path(explicit)] if explicit else []
    candidates += [pathlib.Path.home() / ".hermes" / "models" / name for name in LIVE_MODELS]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    pytest.skip(f"no local GGUF for the live gate ({' / '.join(LIVE_MODELS)})")


@pytest.mark.model
def test_a_live_serving_run_reports_the_bundle_backend_it_computed_on(
        tmp_path: pathlib.Path) -> None:
    """One live row on the pinned bundle (card t_80f1a4c6, requirement 3).

        GGUFONE_RUNTIME_DIR=<bundle> VK_DRIVER_FILES=<icd> \\
            uv run pytest -q --run-network tests/test_serving_attribution.py -k live

    The request names no backend, so the label must come from the bundle this run *loaded* (the
    E3 failure exactly: a scratch `$HOME` left no install record and the label silently read
    `cpu`). The device buffers must be counts the engine's own log produced, naming that same
    backend, and the response must not be flagged.
    """
    from ggufone.runtime import devices as devices_module

    runtime_dir = _runtime_dir()
    expected = finder.backend_of_bundle(runtime_dir)
    if expected in (None, "cpu"):
        pytest.skip(f"{runtime_dir} carries no accelerator backend; there is nothing to attribute")

    payload = request_payload({"threads": 4})
    payload["model"] = str(_live_model())
    body = cli.decide_payload(payload, home=tmp_path, fit_enabled=True)

    engine = body["engine"]
    assert engine["backend_source"] == "bundle" and engine["backend"] == expected
    assert engine["effective_backend"] == expected
    assert engine["device_buffers"], "the engine logged no compute buffer — nothing to attribute"
    assert {devices_module.backend_of(device)
            for device in engine["device_buffers"]} == {expected}
    assert "W_BACKEND_MISMATCH" not in body["warnings"]
