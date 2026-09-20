"""Card t_55de5779: a `cpu` bench row must *compute* on the CPU — on a GPU box too.

The defect (found live on the operator host, Vulkan bundle `b11026` only):

    W_BACKEND_MISMATCH: the row claims backend `cpu` but the engine's own log shows the compute on
    vulkan (compute buffers …)

`tests/test_bench_live.py::test_determinism_holds_on_this_box` asks for
`BenchConfig(suite="determinism", backend="cpu", threads=1)` and the row claims `cpu`, but the
bundle a `cpu` row resolves to is whatever local bundle carries the CPU backend — on a host whose
only install is the Vulkan build, that is the Vulkan bundle, and `n_gpu_layers=0` does **not** stop
llama.cpp's *op offload* from running the graph on the device. The bench's own attribution guard
(`bench.devices`, card t_603a35a0) correctly refused to certify the row. On a GPU-less box the
same run passes (no Vulkan device exists), which is why only the live gate ever saw it.

Measured on this box (Vulkan bundle, RTX 3060 Ti, model `Qwen3.5-0.8B`, `docs/evidence/…`):

* `n_gpu_layers=0` alone               -> `Vulkan0 compute buffer size` (the graph ran there)
* `op_offload=False` alone             -> `Vulkan_Host compute buffer size` (still Vulkan)
* `llama_model_params.devices = [CPU]` -> `CPU compute buffer size` only, `effective: cpu`

So the pin is the *device list*: the context's scheduler is built from the model's device list, and
a load that is only offered the CPU device cannot compute anywhere else — the label is true by
construction and the guard (untouched) has nothing to refute.

What this file pins, offline and GPU-free:

* `harness.spec_for` marks a `cpu` row (`ModelSpec.cpu_only`) and leaves accelerator rows alone;
* `harness.LiveModel.load()` passes that pin to `session.open_model`;
* a pinned load hands the loader `params.devices = [<the bundle's CPU device>, NULL]` and asks for
  **zero** offload layers, whatever `--gpu-layers` requested;
* an unpinned load keeps llama.cpp's own device list (`devices = NULL`) and its requested layers;
* a bundle that cannot name a CPU device is a typed refusal — never a silently unpinned load;
* the placement (and the rendered table) says the compute was pinned.

The vehicle is a fake runtime object behind a real bundle directory (`tests/test_fit_oom_recovery`'s
rig shape): production's own dlopen path, `llama_model_default_params` /
`llama_model_load_from_file` call, arch pre-flight and placement code all run — only libllama is a
Python object.
"""
from __future__ import annotations

import contextlib
import ctypes
import pathlib
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from tests.test_fit import write_gguf
from typed_gguf.bench import harness
from typed_gguf.engine import session as session_module
from typed_gguf.errors import RuntimeMissingError
from typed_gguf.runtime import ctypes_binding, finder

#: what the fake bundle's `ggml_backend_dev_by_name("CPU")` answers (any non-NULL handle)
CPU_DEVICE = 0x7F55DE57
#: the pinned model's architecture, so `capability.require_arch` really runs and really passes
ARCH_SYMBOL = b"llama_model_spark2_5"


def cpu_device_fn(value: int | None = CPU_DEVICE):
    """A real ctypes function pointer — production sets `argtypes`/`restype` before calling it."""
    return ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p)(
        lambda name: value if name == b"CPU" else None)


class PinningBackend:
    """A llama.cpp bundle as production sees it, recording the params of every load attempt."""

    def __init__(self, *, n_layer: int = 4, cpu_device: int | None = CPU_DEVICE,
                 with_device_api: bool = True) -> None:
        self.n_layer = n_layer
        self.cpu_device = cpu_device
        self.params: list[Any] = []                    # one entry per load attempt
        self.directory = pathlib.Path("/fake/runtime")
        self.llama = SimpleNamespace(
            llama_model_default_params=self._default_params,
            llama_model_load_from_file=self._load,
            llama_model_get_vocab=lambda model: 1,
            llama_vocab_n_tokens=lambda vocab: 64,
            llama_model_n_layer=lambda model: self.n_layer,
            llama_model_free=lambda model: None,
        )
        self.ggml = SimpleNamespace(ggml_backend_load_all_from_path=lambda path: None)
        if with_device_api:
            self.ggml.ggml_backend_dev_by_name = cpu_device_fn(cpu_device)

    # ---- the load path
    def _default_params(self) -> Any:
        return SimpleNamespace(n_gpu_layers=0, devices=None)

    def _load(self, path: bytes, params: Any) -> int:
        self.params.append(params)
        return 1

    # ---- what the loader was really handed
    def devices_of(self, attempt: int = -1) -> list[int | None]:
        """The device list of one load attempt, read back out of the C params (`[]` when NULL)."""
        params = self.params[attempt]
        if not params.devices:
            return []
        array = ctypes.cast(params.devices, ctypes.POINTER(ctypes.c_void_p))
        out: list[int | None] = []
        index = 0
        while True:
            value = array[index]
            out.append(value)
            if value is None:                          # llama.cpp's own NULL terminator
                return out
            index += 1
            assert index < 8, "the device list is not NULL-terminated"


@contextlib.contextmanager
def fake_bundle(tmp_path: pathlib.Path, backend: PinningBackend) -> Iterator[PinningBackend]:
    """Install `backend` as the runtime, behind a real bundle directory (the dlopen path)."""
    directory = tmp_path / "runtime"
    directory.mkdir(exist_ok=True)
    (directory / finder.library_names()["llama"]).write_bytes(b"\x7fELF" + ARCH_SYMBOL + b"\x00")
    backend.directory = directory
    original = ctypes_binding.load_libraries
    ctypes_binding.load_libraries = lambda dir_arg, **kwargs: backend  # type: ignore[assignment]
    try:
        yield backend
    finally:
        ctypes_binding.load_libraries = original  # type: ignore[assignment]


def model_file(tmp_path: pathlib.Path, *, n_layer: int = 4) -> pathlib.Path:
    return write_gguf(tmp_path / "model.gguf", n_layer=n_layer)


def cpu_spec(path: pathlib.Path, backend: PinningBackend, *, layers: int = 4) -> harness.ModelSpec:
    """The spec `suites.live_factory` builds for a `--backend cpu` row."""
    return harness.ModelSpec(path=str(path), backend="cpu", runtime_dir=str(backend.directory),
                             threads=1, n_gpu_layers=layers, cpu_only=True)


# ------------------------------------------------------------------ the spec carries the pin
def test_spec_for_pins_a_cpu_row_and_leaves_the_accelerator_rows_alone() -> None:
    """`--backend cpu` is a statement about the compute path, not only about a bundle label."""
    runtimes = {"cpu": "/tmp/cpu", "vulkan": "/tmp/vulkan"}
    config = harness.BenchConfig(suite="determinism", model_path="/tmp/m.gguf", backend="cpu")

    cpu = harness.spec_for(config, "cpu", runtimes=runtimes)
    assert cpu.cpu_only is True
    assert cpu.n_gpu_layers == 0                       # the default request: no offload
    vulkan = harness.spec_for(config, "vulkan", runtimes=runtimes)
    assert vulkan.cpu_only is False
    assert vulkan.n_gpu_layers == -1                   # every layer, llama.cpp's own default
    assert harness.ModelSpec(path="/tmp/m.gguf").cpu_only is False    # nothing pinned by default


def test_a_cpu_row_keeps_the_requested_layers_and_still_forces_zero(tmp_path: pathlib.Path) -> None:
    """`--gpu-layers N` stays the *request* (the report shows requested vs used); the pin makes the
    executed plan zero-offload anyway — there is no accelerator device to offload to."""
    runtimes = {"cpu": "/tmp/cpu"}
    forced = harness.BenchConfig(suite="determinism", model_path="/tmp/m.gguf", gpu_layers=36)

    spec = harness.spec_for(forced, "cpu", runtimes=runtimes)
    assert spec.n_gpu_layers == 36                     # what the flags asked for
    assert spec.cpu_only is True                       # what the loader is allowed to do


def test_the_bench_load_seam_passes_the_pin_to_the_loader(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`harness.LiveModel.load()` is the bench's only way to the loader, so the flag must travel."""
    calls: list[dict[str, Any]] = []

    class Handle:
        placement = SimpleNamespace(to_dict=lambda: {"n_gpu_layers": 0})
        load_ms = 1.0

        def close(self) -> None: ...

    def spy(path: object, **kwargs: Any) -> Handle:
        calls.append({"path": str(path), **kwargs})
        return Handle()

    monkeypatch.setattr(session_module, "open_model", spy)
    model = harness.LiveModel(harness.ModelSpec(path="/tmp/m.gguf", backend="cpu",
                                                n_gpu_layers=0, cpu_only=True))
    try:
        model.load()
    finally:
        model.close()
    assert calls[-1]["cpu_only"] is True

    calls.clear()
    other = harness.LiveModel(harness.ModelSpec(path="/tmp/m.gguf", backend="vulkan",
                                                n_gpu_layers=-1))
    try:
        other.load()
    finally:
        other.close()
    assert calls[-1]["cpu_only"] is False


# ------------------------------------------------------------------ the executed plan
def test_a_pinned_load_offers_the_loader_the_cpu_device_only(tmp_path: pathlib.Path) -> None:
    """The executed plan: zero offload layers, and a device list that names one CPU device."""
    backend = PinningBackend(n_layer=4)
    with fake_bundle(tmp_path, backend):
        handle = session_module.open_model(
            model_file(tmp_path), runtime_dir=backend.directory,
            fit_plan=harness.Placement(4),              # the flags asked for 4 layers
            cpu_only=True)
        try:
            assert backend.params, "the pinned load never reached llama_model_load_from_file"
            assert backend.params[-1].n_gpu_layers == 0
            assert backend.devices_of() == [CPU_DEVICE, None]
            assert handle.cpu_only is True
            assert handle.n_gpu_layers == 0
            assert handle.placement.degraded is False
            assert handle.placement.cpu_only is True
            assert "cpu compute pinned" in handle.placement.note
            assert "n_gpu_layers=4" in handle.placement.note   # the request it overrode
            assert handle.placement.to_dict()["cpu_only"] is True
        finally:
            handle.close()


def test_an_unpinned_load_keeps_llama_cpp_s_own_device_list(tmp_path: pathlib.Path) -> None:
    """Every other row is untouched: `devices = NULL` lets llama.cpp pick its own device list."""
    backend = PinningBackend(n_layer=4)
    with fake_bundle(tmp_path, backend):
        handle = session_module.open_model(
            model_file(tmp_path), runtime_dir=backend.directory,
            fit_plan=harness.Placement(4))
        try:
            assert backend.params[-1].n_gpu_layers == 4
            assert backend.devices_of() == []          # NULL: llama.cpp's own device list
            assert handle.cpu_only is False
            assert handle.placement.cpu_only is False
            assert "cpu compute pinned" not in handle.placement.note
        finally:
            handle.close()


def test_the_note_names_a_request_only_when_there_was_one(tmp_path: pathlib.Path) -> None:
    """The pin's sentence must not invent a request it overrode — and must survive having no plan.

    The note is the one line a reader sees. Three inputs reach it: a plan that asked for layers
    (the sentence names them), a plan that asked for zero (nothing to name), and no plan at all
    (`open_model` allows the pin without one — the serving path does exactly that). The three must
    not collapse into one sentence, and the no-plan input must not raise.
    """
    backend = PinningBackend(n_layer=4)
    with fake_bundle(tmp_path, backend):
        zero = session_module.open_model(model_file(tmp_path), runtime_dir=backend.directory,
                                         fit_plan=harness.Placement(0), cpu_only=True)
        try:
            assert "cpu compute pinned" in zero.placement.note
            assert "n_gpu_layers" not in zero.placement.note    # nothing was overridden
        finally:
            zero.close()
        bare = session_module.open_model(model_file(tmp_path), runtime_dir=backend.directory,
                                         cpu_only=True)
        try:
            assert "cpu compute pinned" in bare.placement.note  # no plan: still a pinned load
            assert bare.n_gpu_layers == 0
        finally:
            bare.close()


def test_a_pinned_load_never_walks_a_degradation_ladder(tmp_path: pathlib.Path) -> None:
    """A CPU-pinned plan has no device to walk down from: one attempt, the CPU-only rung."""
    backend = PinningBackend(n_layer=4)
    with fake_bundle(tmp_path, backend):
        session_module.open_model(model_file(tmp_path), runtime_dir=backend.directory,
                                  fit_plan=harness.Placement(4), cpu_only=True).close()
    assert len(backend.params) == 1
    assert backend.params[0].n_gpu_layers == 0


# ------------------------------------------------------------------ the refusal
@pytest.mark.parametrize("cpu_device, with_device_api", [(None, True), (CPU_DEVICE, False)])
def test_a_bundle_that_cannot_name_a_cpu_device_is_a_typed_refusal(
        tmp_path: pathlib.Path, cpu_device: int | None, with_device_api: bool) -> None:
    """No CPU device to point at = no honest `cpu` row: refuse, never load unpinned."""
    backend = PinningBackend(cpu_device=cpu_device, with_device_api=with_device_api)
    with fake_bundle(tmp_path, backend), pytest.raises(RuntimeMissingError) as excinfo:
        session_module.open_model(model_file(tmp_path), runtime_dir=backend.directory,
                                  fit_plan=harness.Placement(0), cpu_only=True)
    assert "E_RUNTIME_SYMBOLS" in str(excinfo.value)
    assert "CPU device" in str(excinfo.value)
    assert backend.params == [], "a refused pin must not fall back to an unpinned load"


def test_the_refusal_does_not_touch_unpinned_loads(tmp_path: pathlib.Path) -> None:
    """The same bundle still serves every other row (the pin is opt-in, per load)."""
    backend = PinningBackend(cpu_device=None)
    with fake_bundle(tmp_path, backend):
        handle = session_module.open_model(model_file(tmp_path), runtime_dir=backend.directory,
                                           fit_plan=harness.Placement(0))
        handle.close()
    assert len(backend.params) == 1


# ------------------------------------------------------------------ what the report says
def test_the_rendered_table_marks_the_pinned_compute(tmp_path: pathlib.Path) -> None:
    backend = PinningBackend(n_layer=4)
    model_path = model_file(tmp_path)
    with fake_bundle(tmp_path, backend):
        model = harness.LiveModel(cpu_spec(model_path, backend, layers=0))
        try:
            model.load()
            report = {"suite": "determinism", "generated_at": "2026-01-01T00:00:00Z",
                      "host": {}, "config": {}, "commands": {}, "model": {"name": "model.gguf"},
                      "placement": harness.placement_of(model, model.spec)}
            markdown = harness.render_report(report)
        finally:
            model.close()

    assert "used n_gpu_layers=0 kv_type=auto (cpu compute pinned)" in markdown
    assert report["placement"]["used"]["cpu_only"] is True
    assert report["placement"]["requested"] == "n_gpu_layers=0"
    # the placement string the suite rows carry (`placement_request`) names the pin too, so a
    # throughput row cannot print a bare `n_gpu_layers=0` that reads like an ordinary CPU default
    assert harness.placement_request(model.spec).endswith("(cpu compute pinned)")
