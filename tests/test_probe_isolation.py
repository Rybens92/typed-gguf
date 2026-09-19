"""The CLI must survive the runtime it probes (E1a FIX t_eae35404).

`ggufone init --json` on the operator's RTX 3060 Ti printed its JSON and then died with
`double free or corruption (!prev)` (SIGABRT, exit 134) — reproducibly, also on the
idempotent re-run. The process had dlopened the CUDA bundle it rejected, deleted that
directory, and then dlopened the vulkan bundle: third-party destructors ran at exit in a
process that also held a live driver/GPU backend. `doctor` (one directory, same libs) exits
clean; the fallback chain is what makes the difference.

Rule that now holds for every ggufone command: **one bundle per process**. Every dlopen of a
bundle happens in a disposable child process (`ggufone.runtime.probe_child`); the command
itself only ever reads JSON. This is the same lesson tests/test_runtime_live.py already
learned for the test harness — a C library must never be able to kill the command.

Also pinned here: the pre-flight that spares the 168.8 MB CUDA download on a host that cannot
load libcudart (E1a FIX requirement 4 / coordinator finding 2), the recorded fallback reason
in `init --json` (finding 3) and the doctor's advice when the expected backend is a bundle
this host cannot load (finding 4).
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

from ggufone import cli
from ggufone.runtime import capability, install, isolated, pins
from tests.test_runtime_fallback import (  # the synthetic-bundle helpers
    ASSET,
    CUDA_LOAD_ERROR,
    GPU_HOST,
    bundle_cache,
    multi_lock,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUMMARY_TOOL = ROOT / "tools" / "host_gate_summary.py"


# ------------------------------------------------------------------ helpers
def synthetic_bundle(tmp_path: pathlib.Path, *, real_backend: bool = False) -> pathlib.Path:
    """A runtime directory the deep probe can work on (play-ELFs, or one real backend lib)."""
    rt = tmp_path / "runtime"
    rt.mkdir(parents=True, exist_ok=True)
    for lib in ("libllama.so", "libggml.so", "libggml-base.so"):
        (rt / lib).write_bytes(b"\x7fELF fake\n")
    (rt / "libggml-cuda.so").write_bytes(b"\x7fELF fake\n")
    if real_backend:
        real = real_system_lib()
        if real is None:  # pragma: no cover - exotic platform
            pytest.skip("no system shared library available")
        (rt / "libggml-vulkan.so").symlink_to(real)
    return rt


def real_system_lib() -> pathlib.Path | None:
    for candidate in ("/lib/x86_64-linux-gnu/libm.so.6", "/usr/lib64/libm.so.6",
                      "/lib64/libm.so.6", "/usr/lib/libm.so.6"):
        path = pathlib.Path(candidate)
        if path.exists():
            return path
    return None


def lock_with_system_libs(cache: pathlib.Path, backends: tuple[str, ...],
                          system_libs: dict[str, list[str]]) -> pathlib.Path:
    """`multi_lock` plus the per-variant system libraries the pre-flight checks."""
    path = multi_lock(cache, backends)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["llama_cpp"]["system_libs"] = system_libs
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def crash_after(seconds: float = 0.0, exit_code: int = 9) -> list[str]:
    """A child command that dies without answering (stands in for a killed probe)."""
    code = f"import time; time.sleep({seconds}); raise SystemExit({exit_code})"
    return [sys.executable, "-c", code]


# ------------------------------------------------------------------ isolation
@pytest.mark.needs_fork
def test_deep_probe_does_not_dlopen_anything_in_this_process(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """The in-process seams must stay silent: a probe that runs here can kill the command."""
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the command process dlopened the bundle (probe must be isolated)")

    monkeypatch.setattr(capability, "probe_symbols", forbidden)
    monkeypatch.setattr(capability, "load_backend_library", forbidden)

    got = capability.probe_runtime(synthetic_bundle(tmp_path), deep=True, run_tools=False)

    assert got.symbols_checked is True
    assert got.error and "cannot load" in got.error     # the child reported the real failure
    assert not got.ok()


@pytest.mark.needs_fork
def test_the_child_really_loads_the_bundle(tmp_path: pathlib.Path) -> None:
    """A real ELF in `libllama.so`: the child resolves (and misses) the ABI for real."""
    real = real_system_lib()
    if real is None:  # pragma: no cover - exotic platform
        pytest.skip("no system shared library available")
    rt = tmp_path / "runtime"
    rt.mkdir()
    for lib in ("libllama.so", "libggml.so", "libggml-base.so"):
        (rt / lib).write_bytes(real.read_bytes())

    got = capability.probe_runtime(rt, deep=True, run_tools=False)

    assert got.error is None
    assert got.missing_symbols                              # libm has none of the llama ABI
    assert any("do not resolve" in failure for failure in got.failures())


@pytest.mark.needs_fork
def test_the_probe_child_answers_the_documented_protocol(tmp_path: pathlib.Path) -> None:
    """`python -m ggufone.runtime.probe_child` + one JSON request/response, nothing else."""
    import subprocess

    rt = synthetic_bundle(tmp_path)
    request = {"mode": "probe", "runtime_dir": str(rt), "system": "linux",
               "symbols_llama": ["llama_backend_init"], "symbols_ggml": ["ggml_backend_load_all"]}
    assert isolated.child_command()[1:] == ["-m", isolated.CHILD_MODULE]
    result = subprocess.run(                                                   # noqa: S603
        isolated.child_command(), input=json.dumps(request), capture_output=True, text=True,
        timeout=120, check=False, env=isolated.child_env())
    assert result.returncode == 0, result.stderr[-2000:]
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert set(payload) >= {"missing_llama", "missing_ggml", "error", "backend_errors"}
    assert payload["error"] and "libggml-base.so" in payload["error"]


def test_a_probe_child_that_crashes_is_recorded_not_swallowed(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """A dead probe means "cannot verify here" — every accelerator is marked unusable."""
    monkeypatch.setattr(isolated, "child_command", lambda: crash_after(exit_code=9))

    got = capability.probe_runtime(synthetic_bundle(tmp_path), deep=True, run_tools=False)

    assert got.error and "9" in got.error
    assert "cuda" in got.backend_errors
    assert got.usable("cuda") is False
    assert got.usable("cpu") is True


def test_garbage_from_the_probe_child_is_recorded(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    monkeypatch.setattr(isolated, "child_command",
                        lambda: [sys.executable, "-c", "print('not json at all')"])

    got = capability.probe_runtime(synthetic_bundle(tmp_path), deep=True, run_tools=False)

    assert got.error and "json" in got.error.lower()
    assert got.usable("cuda") is False


@pytest.mark.needs_fork
def test_a_probe_child_that_hangs_times_out(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    monkeypatch.setattr(isolated, "child_command", lambda: crash_after(seconds=5.0))
    monkeypatch.setattr(isolated, "CHILD_TIMEOUT", 0.2)

    got = capability.probe_runtime(synthetic_bundle(tmp_path), deep=True, run_tools=False)

    assert got.error and "timed out" in got.error
    assert got.usable("cuda") is False


def test_install_falls_all_the_way_to_cpu_when_the_probe_cannot_run(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    cache = bundle_cache(tmp_path, ("cuda", "vulkan", "cpu"))
    lock = pins.load_lock(multi_lock(cache, ("cuda", "vulkan", "cpu")))
    monkeypatch.setattr(isolated, "child_command", lambda: crash_after(exit_code=9))

    result = install.install("auto", home=tmp_path / "home", lock=lock, offline_cache=cache,
                             free_bytes=1 << 40, probes=GPU_HOST)

    assert result["variant"] == "linux-x64-cpu"
    assert [attempt["backend"] for attempt in result["fallback_attempts"]] == ["cuda", "vulkan"]
    assert all("isolated probe" in attempt["reason"]
               for attempt in result["fallback_attempts"])


# ------------------------------------------------------------------ pre-flight
def test_preflight_skips_the_cuda_download_when_the_host_cannot_load_cudart(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path,
        in_process_scan: None) -> None:
    """No 168.8 MB download for a bundle whose libcudart is missing (finding 2)."""
    cache = bundle_cache(tmp_path, ("cuda", "vulkan", "cpu"))
    lock = pins.load_lock(lock_with_system_libs(cache, ("cuda", "vulkan", "cpu"), {
        "linux-x64-cuda-12.8": ["libcudart.so.12", "libcublas.so.12", "libcuda.so.1"],
        "linux-x64-vulkan": ["libvulkan.so.1"]}))
    monkeypatch.setattr(install, "PREFLIGHT_SYSTEM_LIBS", lambda names: {
        "libcudart.so.12": "libcudart.so.12: cannot open shared object file: "
                           "No such file or directory",
        "libcublas.so.12": "libcublas.so.12: cannot open shared object file: "
                           "No such file or directory",
        "libcuda.so.1": None, "libvulkan.so.1": None})
    monkeypatch.setattr(capability, "load_backend_library", lambda path: None)

    result = install.install("auto", home=tmp_path / "home", lock=lock, offline_cache=cache,
                             free_bytes=1 << 40, probes=GPU_HOST)

    assert result["variant"] == "linux-x64-vulkan"
    attempt = result["fallback_attempts"][0]
    assert attempt["backend"] == "cuda"
    assert attempt["reason"].startswith("pre-flight:")
    assert "libcudart.so.12" in attempt["reason"]
    downloads = tmp_path / "home" / "downloads"
    assert not (downloads / ASSET["cuda"]).exists()          # the 168 MB never moved
    assert (downloads / ASSET["vulkan"]).exists()
    record = json.loads((tmp_path / "home" / "runtime.json").read_text())
    assert record["backend_working"] == "vulkan"
    assert record["fallback_reason"] == attempt["reason"]


def test_preflight_proceeds_when_the_host_can_load_the_libraries(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    cache = bundle_cache(tmp_path, ("cuda", "vulkan", "cpu"))
    lock = pins.load_lock(lock_with_system_libs(cache, ("cuda", "vulkan", "cpu"), {
        "linux-x64-cuda-12.8": ["libcudart.so.12"]}))
    monkeypatch.setattr(install, "PREFLIGHT_SYSTEM_LIBS", lambda names: dict.fromkeys(names))

    result = install.install("auto", home=tmp_path / "home", lock=lock, offline_cache=cache,
                             free_bytes=1 << 40, probes=GPU_HOST)

    assert (tmp_path / "home" / "downloads" / ASSET["cuda"]).exists()
    assert result["variant"] == "linux-x64-cpu"              # the play-ELF cuda tier still fails
    assert not result["fallback_attempts"][0]["reason"].startswith("pre-flight:")


def test_an_explicit_backend_skips_the_preflight(tmp_path: pathlib.Path) -> None:
    """`--backend cuda` is an instruction: install it and let the probe record the truth."""
    cache = bundle_cache(tmp_path, ("cuda", "cpu"))
    lock = pins.load_lock(lock_with_system_libs(cache, ("cuda", "cpu"), {
        "linux-x64-cuda-12.8": ["libcudart.so.12"]}))

    result = install.install("cuda", home=tmp_path / "home", lock=lock, offline_cache=cache,
                             free_bytes=1 << 40)

    assert result["variant"] == "linux-x64-cuda-12.8"
    assert (tmp_path / "home" / "downloads" / ASSET["cuda"]).exists()
    assert any("W_BACKEND_LOAD" in warning for warning in result["probe_warnings"])


# ------------------------------------------------------------------ findings 3 + 4
@pytest.mark.needs_fork
def test_already_installed_reports_the_recorded_fallback_reason(
        tmp_path: pathlib.Path) -> None:
    """The idempotent re-run must still say `cuda -> vulkan`, from the record (finding 3)."""
    home = tmp_path / "home"
    cuda = home / "runtime" / "b11026-linux-x64-cuda-12.8"
    cuda.mkdir(parents=True)
    for lib in ("libllama.so", "libggml.so", "libggml-base.so", "libggml-cuda.so"):
        (cuda / lib).write_bytes(b"\x7fELF fake\n")
    (home / "runtime.json").write_text(json.dumps({
        "schema": "ggufone.runtime/v1", "variant": "linux-x64-vulkan",
        "backend_requested": "cuda", "backend_working": "vulkan",
        "fallback_reason": f"cuda does not load on this host ({CUDA_LOAD_ERROR})",
        "fallback_attempts": [{"backend": "cuda", "variant": "linux-x64-cuda-12.8",
                               "reason": f"cuda does not load on this host ({CUDA_LOAD_ERROR})"}]}))
    vulkan = home / "runtime" / "b11026-linux-x64-vulkan"
    vulkan.mkdir(parents=True)
    for lib in ("libllama.so", "libggml.so", "libggml-base.so"):
        (vulkan / lib).write_bytes(b"\x7fELF fake\n")
    real = real_system_lib()
    if real is None:  # pragma: no cover - exotic platform
        pytest.skip("no system shared library available")
    (vulkan / "libggml-vulkan.so").symlink_to(real)

    got = install.install("auto", home=home, lock=pins.load_lock(
        multi_lock(bundle_cache(tmp_path, ("cuda", "vulkan", "cpu")),
                   ("cuda", "vulkan", "cpu"))), probes=GPU_HOST)

    assert got["already_installed"] is True
    assert got["variant"] == "linux-x64-vulkan"
    assert got["fallback_attempts"][0]["backend"] == "cuda"
    assert "does not load on this host" in got["fallback_reason"]
    assert got["fallback_reason"] == got["fallback_attempts"][0]["reason"]


@pytest.mark.needs_fork
def test_cli_init_on_an_installed_runtime_prints_the_fallback_reason(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys) -> None:
    cache = bundle_cache(tmp_path, ("cuda", "vulkan", "cpu"))
    lock = multi_lock(cache, ("cuda", "vulkan", "cpu"))
    home = tmp_path / "home"
    (home / "runtime").mkdir(parents=True)
    cuda = home / "runtime" / "b11026-linux-x64-cuda-12.8"
    cuda.mkdir()
    for lib in ("libllama.so", "libggml.so", "libggml-base.so", "libggml-cuda.so"):
        (cuda / lib).write_bytes(b"\x7fELF fake\n")
    vulkan = home / "runtime" / "b11026-linux-x64-vulkan"
    vulkan.mkdir()
    for lib in ("libllama.so", "libggml.so", "libggml-base.so"):
        (vulkan / lib).write_bytes(b"\x7fELF fake\n")
    real = real_system_lib()
    if real is None:  # pragma: no cover - exotic platform
        pytest.skip("no system shared library available")
    (vulkan / "libggml-vulkan.so").symlink_to(real)
    monkeypatch.setenv("GGUFONE_HOME", str(home))
    monkeypatch.setenv("GGUFONE_LOCK", str(lock))
    monkeypatch.delenv("GGUFONE_DEEP_PROBE", raising=False)
    monkeypatch.delenv("GGUFONE_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(pins, "current_host", lambda: GPU_HOST)

    assert cli.main(["init", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["already_installed"] is True
    assert payload["variant"] == "linux-x64-vulkan"
    assert "does not load on this host" in payload["fallback_reason"]
    assert payload["fallback_reason"] == payload["fallback_attempts"][0]["reason"]


@pytest.mark.needs_fork
def test_init_without_a_record_names_the_backend_this_run_asked_for(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys) -> None:
    """Runtime dirs copied in without `runtime.json`: still `cuda -> vulkan`, never `None -> …`."""
    cache = bundle_cache(tmp_path, ("cuda", "vulkan", "cpu"))
    lock = multi_lock(cache, ("cuda", "vulkan", "cpu"))
    home = tmp_path / "home"
    (home / "runtime").mkdir(parents=True)
    cuda = home / "runtime" / "b11026-linux-x64-cuda-12.8"
    cuda.mkdir()
    for lib in ("libllama.so", "libggml.so", "libggml-base.so", "libggml-cuda.so"):
        (cuda / lib).write_bytes(b"\x7fELF fake\n")
    vulkan = home / "runtime" / "b11026-linux-x64-vulkan"
    vulkan.mkdir()
    for lib in ("libllama.so", "libggml.so", "libggml-base.so"):
        (vulkan / lib).write_bytes(b"\x7fELF fake\n")
    real = real_system_lib()
    if real is None:  # pragma: no cover - exotic platform
        pytest.skip("no system shared library available")
    (vulkan / "libggml-vulkan.so").symlink_to(real)
    monkeypatch.setenv("GGUFONE_HOME", str(home))
    monkeypatch.setenv("GGUFONE_LOCK", str(lock))
    monkeypatch.delenv("GGUFONE_DEEP_PROBE", raising=False)
    monkeypatch.delenv("GGUFONE_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(pins, "current_host", lambda: GPU_HOST)

    assert cli.main(["init", "--json"]) == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["backend_requested"] == "cuda"      # what THIS run detected and asked for
    assert payload["fallback_reason_code"] == install.REASON_LOADER_ERROR
    assert "from cuda to linux-x64-vulkan" in captured.err


def test_host_gate_summary_reads_the_fallback_from_the_record(tmp_path: pathlib.Path) -> None:
    """`init.fallback` must not be null when the record carries the reason (finding 3)."""
    spec = importlib.util.spec_from_file_location("host_gate_summary", SUMMARY_TOOL)
    assert spec and spec.loader
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    reason = f"cuda does not load on this host ({CUDA_LOAD_ERROR})"
    init_payload = {"already_installed": True, "variant": "linux-x64-vulkan",
                    "backend": "vulkan", "working_backend": "vulkan",
                    "fallback_attempts": [{"backend": "cuda", "variant": "linux-x64-cuda-12.8",
                                           "reason": reason}], "hint": "..."}
    (log_dir / "init.out").write_text(json.dumps(init_payload) + "\n")
    (log_dir / "init.exit").write_text("0\n")
    (log_dir / "fallback_evidence.txt").write_text(
        "# installed runtime record\n" + json.dumps({"variant": "linux-x64-vulkan",
                                                     "fallback_reason": reason}) + "\n")

    assert tool.main(["host_gate_summary", str(log_dir)]) == 0

    report = json.loads((log_dir / "host_gate_e1a.json").read_text())
    assert report["install"]["fallback_reason"] == reason
    assert report["install"]["record"]["fallback_reason"] == reason


def test_doctor_names_the_missing_runtime_instead_of_a_retry(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys) -> None:
    """Cuda missing because libcudart is absent: say that, do not say `--backend cuda` (4)."""
    home = tmp_path / "home"
    monkeypatch.setenv("GGUFONE_HOME", str(home))
    monkeypatch.delenv("GGUFONE_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("GGUFONE_DEEP_PROBE", "0")
    monkeypatch.setattr(pins, "current_host", lambda: GPU_HOST)
    rt = home / "runtime" / "b11026-linux-x64-vulkan"
    rt.mkdir(parents=True)
    for lib in ("libllama.so", "libggml.so", "libggml-base.so", "libggml-vulkan.so"):
        (rt / lib).write_bytes(b"\x7fELF fake\nllama_model_spark2_5\x00build 11026\n")
    reason = f"cuda does not load on this host ({CUDA_LOAD_ERROR})"
    (home / "runtime.json").write_text(json.dumps({
        "schema": "ggufone.runtime/v1", "variant": "linux-x64-vulkan", "build": 11026,
        "backend_requested": "cuda", "backend_working": "vulkan",
        "fallback_reason": reason,
        "fallback_attempts": [{"backend": "cuda", "variant": "linux-x64-cuda-12.8",
                               "reason": reason}]}))

    assert cli.main(["doctor", "--json"]) == 2

    report = json.loads(capsys.readouterr().out)
    checks = {check["id"]: check for check in report["checks"]}
    detail = checks["runtime.accelerator"]["detail"]
    assert "libcudart.so.12" in detail
    assert "--backend cuda" not in detail
    assert "vulkan" in detail


def test_doctor_keeps_the_retry_hint_when_the_bundle_simply_lacks_the_backend(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys) -> None:
    """No recorded fallback: the backend really is absent, so `init --backend cuda` is right."""
    home = tmp_path / "home"
    monkeypatch.setenv("GGUFONE_HOME", str(home))
    monkeypatch.delenv("GGUFONE_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("GGUFONE_DEEP_PROBE", "0")
    monkeypatch.setattr(pins, "current_host", lambda: GPU_HOST)
    rt = home / "runtime" / "b11026-linux-x64-vulkan"
    rt.mkdir(parents=True)
    for lib in ("libllama.so", "libggml.so", "libggml-base.so", "libggml-vulkan.so"):
        (rt / lib).write_bytes(b"\x7fELF fake\nllama_model_spark2_5\x00build 11026\n")

    assert cli.main(["doctor", "--json"]) == 2

    report = json.loads(capsys.readouterr().out)
    checks = {check["id"]: check for check in report["checks"]}
    assert "--backend cuda" in checks["runtime.accelerator"]["detail"]
    assert "runtime.fallback" not in checks          # nothing was recorded: nothing to explain


# ------------------------------------------------------------------ the pin (finding 1)
def test_runtime_lock_pins_the_gpu_assets_it_shipped() -> None:
    lock = pins.load_lock()
    vulkan = lock.assets["linux-x64-vulkan"]
    assert vulkan.sha256 == "1b40310bf4d47c2c84853ebb4ccaf4dcbd992596cd1c2f610be6a0532a874708"
    assert vulkan.verified_locally is True
    cuda = lock.assets["linux-x64-cuda-12.8"]
    assert cuda.sha256 == "5b2d30d7a5e448fbe0aceda360c8f9ed2949aa1734e94db078e6d0b722521e2b"
    assert lock.assets["linux-x64-cpu"].sha256  # the pinned CPU bundle keeps its pin
    assert lock.system_libs["linux-x64-cuda-12.8"] == ("libcudart.so.12", "libcublas.so.12",
                                                       "libcuda.so.1")
    assert lock.system_libs["linux-x64-vulkan"] == ("libvulkan.so.1",)


# ------------------------------------------------------------------ warm-up isolation
def test_init_warmup_runs_outside_the_command_process(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """`init` loads a model for its warm-up number: that dlopen must not be ours either."""
    seen: dict[str, object] = {}

    def fake_warmup(runtime_dir: object, model_path: object, **kwargs: object) -> float:
        seen.update({"runtime_dir": str(runtime_dir), "model": str(model_path)})
        return 12.5

    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setattr(install, "DEFAULT_WARMUP", fake_warmup)
    plan = install.InstallPlan(tag="b11026", variant="linux-x64-cpu", asset="a.tar.gz",
                               url="https://example.invalid/a.tar.gz", size=1, sha256=None,
                               dest=tmp_path / "runtime" / "b11026-linux-x64-cpu",
                               backend="cpu", required_bytes=1)

    ms, error, recorded = install._warmup_ms(tmp_path / "home", plan, model)

    assert (ms, error) == (12.5, None)
    assert recorded == str(model)
    assert seen["model"] == str(model)


def test_a_warmup_that_dies_is_recorded_not_raised(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")

    def boom(*args: object, **kwargs: object) -> float:
        raise isolated.ChildFailure("the probe child exited 134 (SIGABRT)")

    monkeypatch.setattr(install, "DEFAULT_WARMUP", boom)
    plan = install.InstallPlan(tag="b11026", variant="linux-x64-cpu", asset="a.tar.gz",
                               url="https://example.invalid/a.tar.gz", size=1, sha256=None,
                               dest=tmp_path / "runtime" / "b11026-linux-x64-cpu",
                               backend="cpu", required_bytes=1)

    ms, error, recorded = install._warmup_ms(tmp_path / "home", plan, model)

    assert ms is None
    assert error and "ChildFailure" in error and "134" in error
    assert recorded == str(model)


@pytest.mark.needs_fork
def test_warmup_child_answers_the_documented_protocol(tmp_path: pathlib.Path) -> None:
    import subprocess

    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    request = {"mode": "warmup", "runtime_dir": str(tmp_path / "runtime"),
               "warmup_model": str(model), "n_ctx": 8}
    result = subprocess.run(                                                   # noqa: S603
        isolated.child_command(), input=json.dumps(request), capture_output=True, text=True,
        timeout=120, check=False, env=isolated.child_env())
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["warmup_ms"] is None
    assert payload["error"] and "RuntimeMissing" in payload["error"]


@pytest.mark.needs_fork
def test_system_lib_probe_reports_what_this_host_can_load() -> None:
    got = isolated.system_libs(("libc.so.6", "libggufone-not-here.so.7"))
    assert got["libc.so.6"] is None
    assert got["libggufone-not-here.so.7"]


def test_probe_child_modes_are_callable_in_process(tmp_path: pathlib.Path) -> None:
    """The child is a plain module: mode in, JSON-able dict out (play-ELFs only, nothing real)."""
    from ggufone.runtime import probe_child

    rt = synthetic_bundle(tmp_path)
    probe = probe_child.handle({"mode": "probe", "runtime_dir": str(rt), "system": "linux",
                                "symbols_llama": ["llama_backend_init"], "symbols_ggml": []})
    assert probe["error"] and "libggml-base.so" in str(probe["error"])

    loaded = probe_child.handle({"mode": "libs", "libs": ["libc.so.6", "libggufone-nope.so.7"]})
    assert loaded["libs"]["libc.so.6"] is None
    assert "cannot open shared object file" in str(loaded["libs"]["libggufone-nope.so.7"])

    warm = probe_child.handle({"mode": "warmup", "runtime_dir": str(tmp_path / "gone"),
                               "warmup_model": str(tmp_path / "model.gguf")})
    assert warm["warmup_ms"] is None
    assert "RuntimeMissing" in str(warm["error"])

    with pytest.raises(SystemExit):
        probe_child.handle({"mode": "nope"})
    assert set(probe_child.MODES) == {"probe", "warmup", "libs"}


@pytest.mark.needs_fork
def test_the_child_reports_symbols_and_backends_in_one_answer(tmp_path: pathlib.Path) -> None:
    """Both halves of the deep probe: the ABI that resolves and a backend that does not."""
    real = real_system_lib()
    if real is None:  # pragma: no cover - exotic platform
        pytest.skip("no system shared library available")
    rt = tmp_path / "runtime"
    rt.mkdir()
    for lib in ("libllama.so", "libggml.so", "libggml-base.so"):
        (rt / lib).write_bytes(real.read_bytes())
    (rt / "libggml-vulkan.so").write_bytes(b"\x7fELF not really an ELF\n")

    got = capability.probe_runtime(rt, deep=True, run_tools=False)

    assert got.error is None                       # the ABI libraries loaded
    assert len(got.missing_symbols) == 34          # … and carry none of the required symbols
    assert "llama_backend_init" in got.missing_symbols
    assert "vulkan" in got.backend_errors           # the backend could not dlopen here
    assert got.usable("vulkan") is False
    assert got.usable("cpu") is True


def test_a_dead_preflight_child_fails_closed(monkeypatch: pytest.MonkeyPatch,
                                             tmp_path: pathlib.Path) -> None:
    """A pre-flight that cannot run counts every library as unloadable: skip, never guess."""
    monkeypatch.setattr(isolated, "child_command", lambda: crash_after(exit_code=9))

    got = isolated.system_libs(("libcudart.so.12", "libvulkan.so.1"))

    assert got["libcudart.so.12"] and "9" in got["libcudart.so.12"]
    assert got["libvulkan.so.1"]

    cache = bundle_cache(tmp_path, ("cuda", "cpu"))
    lock = pins.load_lock(lock_with_system_libs(cache, ("cuda", "cpu"), {
        "linux-x64-cuda-12.8": ["libcudart.so.12"]}))

    result = install.install("auto", home=tmp_path / "home", lock=lock, offline_cache=cache,
                             free_bytes=1 << 40, probes=GPU_HOST)

    assert result["fallback_attempts"][0]["reason"].startswith("pre-flight:")
    assert not (tmp_path / "home" / "downloads" / ASSET["cuda"]).exists()


def test_preflight_matches_the_pinned_lock_for_the_gpu_variants() -> None:
    """The pre-flight list in `runtime.lock` is what makes the skip honest, not a guess."""
    lock = pins.load_lock()
    for variant, libs in lock.system_libs.items():
        assert libs, variant
        assert all(name.endswith((".so", ".so.1", ".so.12")) for name in libs), libs
