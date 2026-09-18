"""E2 FIX (card t_603a35a0): a bench row must name the device that *really* computed.

The operator host carries **two** bundles (the pinned CPU one and the Vulkan b11026 one). The
auditor's provenance re-run measured both ways the label can lie
(`state/fights/e2-provenance/logs/thr-mixed2.raw`, `probe-vulkanbundle-cpu.raw`):

* `--backend all` across the two bundles emitted a row **labelled `vulkan`** whose own backend log
  shows `CPU_Mapped model buffer size = 4167.21 MiB` and nine `CPU compute buffer size` lines with
  **not one `Vulkan0` line** — 9.11 tok/s, host-class. The log even prints
  `load_tensors: offloaded 37/37 layers to GPU` before walking every layer onto the CPU, so the
  request-side statement is not evidence.
* the *single-bundle* Vulkan run shows `Vulkan0 compute buffer size` / `Vulkan_Host …` instead, and
  a row **labelled `cpu`** on that install measured 587.9 tok/s of prefill (op offload: the
  weights stay on the host, the graph runs on the device) while the placement note said
  "CPU only".

Neither row was flagged, and both were about to be published as backend-attributed numbers. What
this file pins:

* `harness.device_usage` reads the device set out of the engine's own log (`<device> compute buffer
  size` / `model buffer size` / `KV buffer size` / `assigned to device`), and `effective_backend`
  is derived from the *compute* buffers — the request-side `offloaded N/M layers to GPU` line is
  never evidence;
* every throughput/determinism row carries the bundle path, the device set, the per-device compute
  buffer counts and the effective backend;
* a claim the evidence contradicts is flagged `W_BACKEND_MISMATCH` **and** fails the report
  (`ok: false`), instead of being published as a clean row;
* the single-backend suites (latency/quality/calibration) carry the same attribution, and the
  rendered markdown shows the effective backend next to the claimed one.

The vehicle is two real bundle directories on disk (one CPU, one Vulkan) plus the model-free bench
seam (`tests/fake_engine.BenchModel`), whose `device_log` is the engine log the live loader
accumulates.
"""
from __future__ import annotations

import pathlib
import textwrap

import pytest

from ggufone.bench import harness, suites
from ggufone.runtime import devices as devices_module
from ggufone.runtime import finder
from tests.fake_engine import BenchModel

# The operator's own lines, verbatim (thr-mixed2.raw): a Vulkan-labelled row that ran on the host
# CPU. `sched_reserve` is the graph scheduler reserving the compute buffers — the strongest
# evidence of where the work happens.
MIXED_BUNDLE_LOG = textwrap.dedent("""\
    load_tensors: offloading 35 repeating layers to GPU
    load_tensors: offloaded 37/37 layers to GPU
    load_tensors:   CPU_Mapped model buffer size =  4167.21 MiB
    llama_context:        CPU  output buffer size =     1.50 MiB
    llama_kv_cache:        CPU KV buffer size =    18.00 MiB
    sched_reserve:        CPU compute buffer size =   166.26 MiB
    ~llama_context:        CPU compute buffer size is 166.2610 MiB, matches expectation of 166.2610 MiB
""")

# The honest single-bundle Vulkan run (probe-vulkanbundle-cpu.raw): the graph really runs there.
GPU_LOG = textwrap.dedent("""\
    load_tensors:      Vulkan0 model buffer size =  3963.12 MiB
    load_tensors:   CPU_Mapped model buffer size =   203.11 MiB
    sched_reserve:    Vulkan0 compute buffer size =   545.31 MiB
    ~llama_context:    Vulkan0 compute buffer size is 545.3125 MiB, matches expectation of 545.3125 MiB
    ~llama_context: Vulkan_Host compute buffer size is  12.5334 MiB, matches expectation of  12.5334 MiB
""")

# A pure CPU bundle: no accelerator device in the process at all.
CPU_LOG = textwrap.dedent("""\
    load_tensors:         CPU model buffer size =  4167.21 MiB
    sched_reserve:        CPU compute buffer size =   166.26 MiB
    ~llama_context:        CPU compute buffer size is 166.2610 MiB, matches expectation of 166.2610 MiB
""")

#: op offload: a row that *asked* for `cpu` (n_gpu_layers=0) while the device ran the graph.
OP_OFFLOAD_LOG = textwrap.dedent("""\
    load_tensors:         CPU model buffer size =  4167.21 MiB
    sched_reserve:    Vulkan0 compute buffer size =   163.13 MiB
    ~llama_context:    Vulkan0 compute buffer size is 163.1250 MiB, matches expectation of 163.1250 MiB
""")


def two_bundles(tmp_path: pathlib.Path) -> pathlib.Path:
    """A CPU bundle next to a Vulkan bundle — the operator's mixed install, on disk."""
    root = tmp_path / "runtime"
    for name, accelerator in (("b11026-linux-x64-cpu", None),
                              ("b11026-linux-x64-vulkan", "libggml-vulkan.so")):
        bundle = root / name
        bundle.mkdir(parents=True)
        (bundle / finder.library_names()["llama"]).write_bytes(b"")
        if accelerator:
            (bundle / accelerator).write_bytes(b"")
    return root


def install(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, *,
            runtime_dir: pathlib.Path | None, root: pathlib.Path) -> None:
    """Make `backend_runtimes` see the bundles (no registry, no download — SPEC A-E2-7)."""
    monkeypatch.setenv("GGUFONE_BENCH_RUNTIME_DIR", str(root))
    if runtime_dir is None:
        monkeypatch.delenv("GGUFONE_RUNTIME_DIR", raising=False)
    else:
        monkeypatch.setenv("GGUFONE_RUNTIME_DIR", str(runtime_dir))


def factory(*, logs: dict[str, str]):
    """`factory(spec) -> BenchModel`: the log each backend's engine *really* produced."""
    def make(spec: harness.ModelSpec) -> BenchModel:
        return BenchModel(spec, prefill_ms_per_token=0.001,
                          device_log=logs.get(spec.backend, ""))
    return make


def throughput(backend: str = "all") -> harness.BenchConfig:
    return harness.BenchConfig(suite="throughput", model_path="/tmp/fake.gguf", runs=1,
                               backend=backend)


def rows_by_backend(report: dict) -> dict[str, dict]:
    return {row["backend"]: row for row in report["backends"]}


# ------------------------------------------------------------------ the parser
def test_device_usage_counts_compute_buffers_per_device() -> None:
    usage = devices_module.parse_device_usage(MIXED_BUNDLE_LOG)
    assert usage.compute_buffers == {"CPU": 2}          # sched_reserve + ~llama_context
    assert usage.model_buffers == {"CPU_Mapped": 1}
    assert usage.kv_buffers == {"CPU": 1}
    assert usage.devices == ("CPU", "CPU_Mapped")
    assert usage.effective == "cpu"


def test_the_request_side_offload_line_is_never_evidence() -> None:
    """`offloaded 37/37 layers to GPU` is what llama.cpp prints for the *request*, and it printed
    it on the very load that walked every layer onto the CPU (thr-mixed2.raw)."""
    usage = devices_module.parse_device_usage("load_tensors: offloaded 37/37 layers to GPU\n"
                                              "load_tensors: offloading output layer to GPU\n")
    assert usage.effective is None
    assert usage.devices == ()


def test_effective_backend_comes_from_the_compute_buffers_not_the_weights() -> None:
    """op offload keeps the weights on the host and runs the graph on the device: the row's
    compute path is the device, and no `Vulkan0 model buffer` line is printed."""
    usage = devices_module.parse_device_usage(OP_OFFLOAD_LOG)
    assert usage.model_buffers == {"CPU": 1}
    assert usage.compute_buffers == {"Vulkan0": 2}
    assert usage.effective == "vulkan"
    assert devices_module.parse_device_usage(GPU_LOG).effective == "vulkan"
    assert devices_module.parse_device_usage("").effective is None


def test_a_device_name_maps_to_its_backend() -> None:
    assert devices_module.backend_of("Vulkan0") == "vulkan"
    assert devices_module.backend_of("Vulkan_Host") == "vulkan"
    assert devices_module.backend_of("CPU_Mapped") == "cpu"
    assert devices_module.backend_of("CUDA0") == "cuda"
    assert devices_module.backend_of("Metal") == "metal"
    assert devices_module.backend_of("RPC0") == "rpc"


# ------------------------------------------------------------------ the lying vulkan row
def test_a_vulkan_row_that_ran_on_cpu_is_flagged_and_fails_the_report(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact operator case: a CPU bundle first, the Vulkan bundle also visible."""
    root = two_bundles(tmp_path)
    install(monkeypatch, tmp_path, runtime_dir=root / "b11026-linux-x64-cpu", root=root)

    report = suites.run_suite(throughput(), factory=factory(
        logs={"cpu": CPU_LOG, "vulkan": MIXED_BUNDLE_LOG}))

    row = rows_by_backend(report)["vulkan"]
    assert row["runtime_dir"] == str(root / "b11026-linux-x64-vulkan")   # the bundle it *named*
    assert row["effective_backend"] == "cpu"                             # the device it *used*
    assert row["devices"] == ["CPU", "CPU_Mapped"]
    assert row["device_buffers"] == {"CPU": 2}
    assert row["warnings"] == ["W_BACKEND_MISMATCH"]
    assert report["ok"] is False
    assert any("W_BACKEND_MISMATCH" in note for note in report["notes"])
    # the cpu row of the same report is honest: no accelerator device in its log
    cpu = rows_by_backend(report)["cpu"]
    assert cpu["effective_backend"] == "cpu" and cpu["warnings"] == []
    assert cpu["runtime_dir"] == str(root / "b11026-linux-x64-cpu")


def test_a_cpu_row_that_ran_on_the_device_is_flagged(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The other direction: only the Vulkan bundle is installed, so `cpu` resolves to *its*
    directory and op offload runs the graph on the device under a `cpu` label."""
    root = tmp_path / "runtime"
    vulkan = root / "b11026-linux-x64-vulkan"
    vulkan.mkdir(parents=True)
    (vulkan / finder.library_names()["llama"]).write_bytes(b"")
    (vulkan / "libggml-vulkan.so").write_bytes(b"")
    install(monkeypatch, tmp_path, runtime_dir=None, root=root)

    found = harness.backend_runtimes(home=tmp_path / "data-home")
    assert found["cpu"] == found["vulkan"] == vulkan

    report = suites.run_suite(throughput("auto"), factory=factory(
        logs={"cpu": OP_OFFLOAD_LOG, "vulkan": GPU_LOG}))

    row = rows_by_backend(report)["cpu"]
    assert row["runtime_dir"] == str(root / "b11026-linux-x64-vulkan")
    assert row["effective_backend"] == "vulkan"
    assert "W_BACKEND_MISMATCH" in row["warnings"]
    assert report["ok"] is False


def test_honest_rows_stay_clean(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One bundle per process (the auditor's F4 recommendation): both labels hold, report ok."""
    root = two_bundles(tmp_path)
    install(monkeypatch, tmp_path, runtime_dir=root / "b11026-linux-x64-cpu", root=root)

    report = suites.run_suite(throughput(), factory=factory(
        logs={"cpu": CPU_LOG, "vulkan": GPU_LOG}))

    rows = rows_by_backend(report)
    assert rows["cpu"]["effective_backend"] == "cpu"
    assert rows["vulkan"]["effective_backend"] == "vulkan"
    assert rows["vulkan"]["device_buffers"] == {"Vulkan0": 2, "Vulkan_Host": 1}
    assert all(row["warnings"] == [] for row in (rows["cpu"], rows["vulkan"]))
    assert report["ok"] is True


def test_an_uncorroborated_accelerator_claim_is_not_published(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A log that names no device cannot corroborate a `vulkan` row: the row reads `unverified`
    *and* is flagged (one bundle per process is the trustworthy shape). A `cpu` row without
    evidence is not refuted — nothing computed anywhere else — so it stays clean."""
    root = two_bundles(tmp_path)
    install(monkeypatch, tmp_path, runtime_dir=root / "b11026-linux-x64-cpu", root=root)

    report = suites.run_suite(throughput(), factory=factory(logs={}))

    rows = rows_by_backend(report)
    vulkan = rows["vulkan"]
    assert vulkan["effective_backend"] is None
    assert vulkan["devices"] == [] and vulkan["device_buffers"] == {}
    assert vulkan["warnings"] == ["W_BACKEND_MISMATCH"]
    assert report["ok"] is False
    assert rows["cpu"]["effective_backend"] is None and rows["cpu"]["warnings"] == []


# ------------------------------------------------------------------ determinism rows too
def test_determinism_rows_carry_the_device_set(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = two_bundles(tmp_path)
    install(monkeypatch, tmp_path, runtime_dir=root / "b11026-linux-x64-cpu", root=root)

    report = suites.run_suite(
        harness.BenchConfig(suite="determinism", model_path="/tmp/fake.gguf", backend="all"),
        factory=factory(logs={"cpu": CPU_LOG, "vulkan": MIXED_BUNDLE_LOG}))

    row = rows_by_backend(report)["vulkan"]
    assert row["runtime_dir"] == str(root / "b11026-linux-x64-vulkan")
    assert row["effective_backend"] == "cpu"
    assert row["warnings"] == ["W_BACKEND_MISMATCH"]
    assert report["ok"] is False


# ------------------------------------------------------------------ the single-backend suites
def test_the_single_backend_suites_record_what_the_engine_used(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = two_bundles(tmp_path)
    install(monkeypatch, tmp_path, runtime_dir=root / "b11026-linux-x64-cpu", root=root)

    report = suites.run_suite(
        harness.BenchConfig(suite="latency", model_path="/tmp/fake.gguf", runs=1,
                            prefill_sizes=(64,)),
        factory=factory(logs={"cpu": OP_OFFLOAD_LOG}))

    assert report["backend_selection"]["selected"] == "cpu"
    assert report["effective_backend"] == "vulkan"
    assert report["devices"] == ["CPU", "Vulkan0"]
    assert report["warnings"] == ["W_BACKEND_MISMATCH"]
    assert report["ok"] is False
    assert any("W_BACKEND_MISMATCH" in note for note in report["notes"])


# ------------------------------------------------------------------ the rendered table
def test_the_rendered_table_shows_the_effective_backend_next_to_the_claim(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = two_bundles(tmp_path)
    install(monkeypatch, tmp_path, runtime_dir=root / "b11026-linux-x64-cpu", root=root)

    report = suites.run_suite(throughput(), factory=factory(
        logs={"cpu": CPU_LOG, "vulkan": MIXED_BUNDLE_LOG}))
    markdown = harness.render_report(report)

    assert "| backend | effective |" in markdown
    assert "| vulkan | cpu |" in markdown          # claimed -> ran on
    assert "W_BACKEND_MISMATCH" in markdown
    assert "| cpu | cpu |" in markdown


def test_the_placement_note_does_not_claim_the_compute_path() -> None:
    """`n_gpu_layers=0` is a statement about the weights; op offload can still compute on the
    device, so the load-time note must not read as a measurement of the compute path."""
    from ggufone.engine import session as session_module
    from ggufone.runtime import fit

    plan = fit.coerce_plan(harness.Placement(0))
    note = session_module._placement_note(plan, degraded=False, fit_disabled=False)
    assert plan.n_gpu_layers == 0
    assert "CPU only" not in note
    assert "n_gpu_layers=0" in note
    assert "no fit plan: CPU only" not in session_module._placement_note(
        None, degraded=False, fit_disabled=False)
