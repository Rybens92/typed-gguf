"""Backend fallback with evidence: cuda -> vulkan -> cpu (E1a FIX t_eae35404 requirement 4).

`init` must not leave an accelerator bundle in place that cannot load on this host. When the
picked GPU bundle fails to dlopen (missing cudart / driver / ICD), the installer records why
and installs the next tier; `doctor --json` reports the backend that actually works plus the
raw dlopen error.

The bundles here are synthetic (play-ELFs: cheap, no compiler). The natural, unpatched loader
therefore rejects *every* accelerator lib in them — which is exactly the "does not load on this
host" case — and `fake_loader` is used where a tier has to look loadable. The real pinned CUDA
bundle is the live evidence run (docs/evidence/e1a_baseline.json), where the unpatched loader
fails with `libcudart.so.12: cannot open shared object file`.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import stat
import sys
import tarfile

import pytest

from ggufone import cli
from ggufone.runtime import capability, finder, install, isolated, pins

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSET = {
    "cuda": "llama-b11026-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "vulkan": "llama-b11026-bin-ubuntu-vulkan-x64.tar.gz",
    "cpu": "llama-b11026-bin-ubuntu-x64.tar.gz",
}
VARIANT = {"cuda": "linux-x64-cuda-12.8", "vulkan": "linux-x64-vulkan", "cpu": "linux-x64-cpu"}
CUDA_LOAD_ERROR = ("libggml-cuda.so: libcudart.so.12: cannot open shared object file: "
                   "No such file or directory")
GPU_HOST = pins.fake_host(system="linux", machine="x86_64", has_nvidia_smi=True)


# ------------------------------------------------------------------ fixtures
def make_bundle(cache: pathlib.Path, backend: str) -> pathlib.Path:
    """A synthetic pinned bundle carrying one accelerator backend library."""
    staging = cache / f"staging-{backend}"
    inner = staging / "llama-b11026"
    inner.mkdir(parents=True, exist_ok=True)
    for lib in ("libllama.so", "libggml.so", "libggml-base.so"):
        (inner / lib).write_bytes(b"\x7fELF fake\nllama_model_spark2_5\x00")
    (inner / f"libggml-{backend}.so").write_bytes(b"\x7fELF fake\n")
    (inner / "libggml-cpu.so").write_bytes(b"\x7fELF fake\n")
    cli_path = inner / "llama-cli"
    cli_path.write_text("#!/bin/sh\necho 'version: 0.4.1-dev (build 11026, commit x)' >&2\n")
    fit = inner / "llama-fit-params"
    fit.write_text("#!/bin/sh\nexit 0\n")
    for exe in (cli_path, fit):
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    archive = cache / ASSET[backend]
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(inner, arcname="llama-b11026")
    return archive


def bundle_cache(tmp_path: pathlib.Path, backends: tuple[str, ...]) -> pathlib.Path:
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    for backend in backends:
        make_bundle(cache, backend)
    return cache


def multi_lock(cache: pathlib.Path, backends: tuple[str, ...]) -> pathlib.Path:
    assets = {}
    for backend in backends:
        archive = cache / ASSET[backend]
        assets[VARIANT[backend]] = {"asset": archive.name, "size": archive.stat().st_size,
                                    "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
    payload = {
        "schema": "ggufone.runtime.lock/v1",
        "llama_cpp": {
            "repo": "ggml-org/llama.cpp", "tag": "b11026",
            "published_at": "2026-09-17T13:31:47Z", "commit": "b49650adb",
            "min_build_for_spark2_5": 10828, "assets": assets,
            "url_template": "https://example.invalid/releases/download/{tag}/{asset}",
            "required_files": ["libllama.so", "libggml.so", "libggml-base.so"],
            "required_tools": ["llama-fit-params"],
            "required_symbols_llama": ["llama_backend_init"],
            "required_symbols_ggml": ["ggml_backend_load_all_from_path"],
            "mandatory_call_order": ["CDLL(libggml.so, RTLD_GLOBAL)",
                                     "ggml_backend_load_all_from_path(<runtime_dir>)",
                                     "llama_backend_init()"],
        },
        "default_model": {"repo": "XHToken/Spark-X2.5-4B-GGUF", "repo_sha": "0" * 40,
                          "quant": "Q8_0", "file": "Spark-X2.5-4B-Q8_0.gguf", "size": 1,
                          "sha256": "a" * 64, "arch": "spark2_5", "license": "apache-2.0",
                          "alternates": {}},
    }
    path = cache.parent / "fallback-runtime.lock"
    path.write_text(json.dumps(payload))
    return path


def fake_loader(errors: dict[str, str]):
    """dlopen seam: report `errors[backend]` for that backend's library, load the rest."""

    def load(path: object) -> str | None:
        name = pathlib.Path(str(path)).name
        for backend, error in errors.items():
            if name.startswith(f"libggml-{backend}."):
                return error
        return None

    return load


# ------------------------------------------------------------------ dlopen check
def test_usable_and_accelerator_semantics() -> None:
    """The two helpers the fallback chain decides with, pinned directly."""
    cuda_only = capability.ProbeResult(backends=("cuda",))
    assert cuda_only.usable("cpu") is True          # cpu needs no accelerator library
    assert cuda_only.usable("cuda") is True
    assert cuda_only.usable("vulkan") is False      # the bundle carries no vulkan backend
    assert cuda_only.accelerator() == "cuda"

    broken_cuda = capability.ProbeResult(backends=("cuda", "vulkan"),
                                        backend_errors={"cuda": "libcudart.so.12: no such file"})
    assert broken_cuda.usable("cuda") is False      # present but not loadable here
    assert broken_cuda.usable("vulkan") is True
    assert broken_cuda.accelerator() == "vulkan"    # what this host can actually drive
    assert capability.ProbeResult(backends=("cpu", "rpc")).accelerator() == "cpu"
    assert capability.ProbeResult(backends=("base", "cpu", "rpc")).accelerator() == "cpu"
    assert capability.ProbeResult().accelerator() == "cpu"


def test_load_backend_library_reports_a_bundle_the_host_cannot_dlopen(
        tmp_path: pathlib.Path) -> None:
    broken = tmp_path / "libggml-cuda.so"
    broken.write_bytes(b"\x7fELF not really an ELF\n")
    error = capability.load_backend_library(broken)
    assert error is not None and "libggml-cuda.so" in error
    missing = capability.load_backend_library(tmp_path / "libggml-vulkan.so")
    assert missing is not None and "libggml-vulkan.so" in missing


def test_load_backend_library_uses_rtld_global(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pinned backends must load RTLD_GLOBAL: libllama resolves their symbols at load."""
    seen: dict[str, object] = {}

    class FakeCDLL:
        def __init__(self, path: str, **kwargs: object) -> None:
            seen.update({"path": path, **kwargs})

    monkeypatch.setattr(capability.C, "CDLL", FakeCDLL)
    assert capability.load_backend_library("/nowhere/libggml-cuda.so") is None
    assert str(seen["path"]).endswith("libggml-cuda.so")
    assert seen.get("mode") == getattr(capability.C, "RTLD_GLOBAL", 0)


def test_load_backend_library_accepts_a_real_shared_object() -> None:
    """The seam must not reject what the host can really dlopen (libc is always there)."""
    import ctypes.util
    libc = ctypes.util.find_library("c")
    if not libc:  # pragma: no cover - exotic platforms
        pytest.skip("no libc for the positive dlopen case")
    assert capability.load_backend_library(libc) is None


# ------------------------------------------------------------------ the chain
def test_install_falls_back_from_cuda_to_vulkan_and_records_why(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, in_process_scan: None) -> None:
    cache = bundle_cache(tmp_path, ("cuda", "vulkan", "cpu"))
    lock = pins.load_lock(multi_lock(cache, ("cuda", "vulkan", "cpu")))
    monkeypatch.setattr(capability, "load_backend_library", fake_loader({"cuda": CUDA_LOAD_ERROR}))

    result = install.install("auto", home=tmp_path / "home", lock=lock, offline_cache=cache,
                             free_bytes=1 << 40, probes=GPU_HOST)

    assert result["variant"] == "linux-x64-vulkan"
    assert result["backend"] == "vulkan" and result["working_backend"] == "vulkan"
    assert result["fallback_reason"].startswith("cuda does not load on this host")
    assert result["fallback_attempts"] == [{"backend": "cuda",
                                            "variant": "linux-x64-cuda-12.8",
                                            "code": install.REASON_LOADER_ERROR,
                                            "reason": result["fallback_reason"]}]
    assert "libcudart.so.12" in result["fallback_attempts"][0]["reason"]
    assert (tmp_path / "home" / "runtime" / "b11026-linux-x64-vulkan" / "libllama.so").exists()
    record = json.loads((tmp_path / "home" / "runtime.json").read_text())
    assert record["variant"] == "linux-x64-vulkan"
    assert record["backend_requested"] == "cuda"
    assert record["backend_working"] == "vulkan"
    assert record["fallback_reason"] == result["fallback_reason"]
    assert "libcudart.so.12" in record["fallback_attempts"][0]["reason"]


def test_install_falls_back_all_the_way_to_cpu_when_no_gpu_backend_loads(
        tmp_path: pathlib.Path) -> None:
    """Unpatched loader: every play-ELF backend is rejected, so the chain ends at cpu."""
    cache = bundle_cache(tmp_path, ("cuda", "vulkan", "cpu"))
    lock = pins.load_lock(multi_lock(cache, ("cuda", "vulkan", "cpu")))

    result = install.install("auto", home=tmp_path / "home", lock=lock, offline_cache=cache,
                             free_bytes=1 << 40, probes=GPU_HOST)

    assert result["variant"] == "linux-x64-cpu"
    assert result["working_backend"] == "cpu"
    assert [attempt["backend"] for attempt in result["fallback_attempts"]] == ["cuda", "vulkan"]
    assert all("does not load on this host" in attempt["reason"]
               for attempt in result["fallback_attempts"])
    record = json.loads((tmp_path / "home" / "runtime.json").read_text())
    assert record["backend_requested"] == "cuda" and record["backend_working"] == "cpu"


def test_install_keeps_cuda_when_its_backend_really_loads(monkeypatch: pytest.MonkeyPatch,
                                                          tmp_path: pathlib.Path,
                                                          in_process_scan: None) -> None:
    cache = bundle_cache(tmp_path, ("cuda", "vulkan", "cpu"))
    lock = pins.load_lock(multi_lock(cache, ("cuda", "vulkan", "cpu")))
    monkeypatch.setattr(capability, "load_backend_library", fake_loader({}))

    result = install.install("auto", home=tmp_path / "home", lock=lock, offline_cache=cache,
                             free_bytes=1 << 40, probes=GPU_HOST)

    assert result["variant"] == "linux-x64-cuda-12.8"
    assert result["backend"] == "cuda" and result["working_backend"] == "cuda"
    assert result["fallback_attempts"] == []
    assert result["fallback_reason"] is None and result["fallback_reason_code"] is None
    record = json.loads((tmp_path / "home" / "runtime.json").read_text())
    assert record["fallback_reason"] is None and record["fallback_reason_code"] is None
    assert record["backend_requested"] == "cuda"


def test_an_explicit_backend_is_honoured_without_falling_back(tmp_path: pathlib.Path) -> None:
    cache = bundle_cache(tmp_path, ("cuda", "vulkan", "cpu"))
    lock = pins.load_lock(multi_lock(cache, ("cuda", "vulkan", "cpu")))

    result = install.install("cuda", home=tmp_path / "home", lock=lock, offline_cache=cache,
                             free_bytes=1 << 40)

    assert result["variant"] == "linux-x64-cuda-12.8"
    assert result["fallback_attempts"] == []
    assert any("W_BACKEND_LOAD" in w for w in result["probe_warnings"])


def test_install_skips_an_already_installed_backend_that_cannot_load(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path,
        in_process_scan: None) -> None:
    """A cuda dir left behind by an earlier run must not win over a working tier."""
    cache = bundle_cache(tmp_path, ("cuda", "vulkan", "cpu"))
    lock = pins.load_lock(multi_lock(cache, ("cuda", "vulkan", "cpu")))
    monkeypatch.setattr(capability, "load_backend_library", fake_loader({"cuda": CUDA_LOAD_ERROR}))
    cuda_dir = tmp_path / "home" / "runtime" / "b11026-linux-x64-cuda-12.8"
    cuda_dir.mkdir(parents=True)
    for lib in ("libllama.so", "libggml.so", "libggml-base.so"):
        (cuda_dir / lib).write_bytes(b"\x7fELF fake\nllama_model_spark2_5\x00")
    (cuda_dir / "libggml-cuda.so").write_bytes(b"\x7fELF fake\n")

    result = install.install("auto", home=tmp_path / "home", lock=lock, offline_cache=cache,
                             free_bytes=1 << 40, probes=GPU_HOST)

    assert result["variant"] == "linux-x64-vulkan"
    assert result["fallback_attempts"][0]["backend"] == "cuda"
    assert result["fallback_attempts"][0]["reason"].startswith("cuda does not load")


def test_install_removes_the_bundle_it_rejected(monkeypatch: pytest.MonkeyPatch,
                                                tmp_path: pathlib.Path,
                                                in_process_scan: None) -> None:
    """The stale CUDA dir must not shadow the working tier for `doctor`/`find_runtime`."""
    cache = bundle_cache(tmp_path, ("cuda", "vulkan", "cpu"))
    lock = pins.load_lock(multi_lock(cache, ("cuda", "vulkan", "cpu")))
    monkeypatch.setattr(capability, "load_backend_library", fake_loader({"cuda": CUDA_LOAD_ERROR}))

    result = install.install("auto", home=tmp_path / "home", lock=lock, offline_cache=cache,
                             free_bytes=1 << 40, probes=GPU_HOST)

    assert result["variant"] == "linux-x64-vulkan"
    assert not (tmp_path / "home" / "runtime" / "b11026-linux-x64-cuda-12.8").exists()
    assert (tmp_path / "home" / "runtime" / "b11026-linux-x64-vulkan").is_dir()
    record = json.loads((tmp_path / "home" / "runtime.json").read_text())
    assert record["dir"] == str(tmp_path / "home" / "runtime" / "b11026-linux-x64-vulkan")


def test_find_runtime_prefers_the_recorded_variant(tmp_path: pathlib.Path) -> None:
    """Two installed variants: the one runtime.json records is the active one."""
    home = tmp_path / "home"
    cuda = home / "runtime" / "b11026-linux-x64-cuda-12.8"
    vulkan = home / "runtime" / "b11026-linux-x64-vulkan"
    for directory in (cuda, vulkan):
        directory.mkdir(parents=True)
        (directory / "libllama.so").write_bytes(b"\x7fELF fake\n")
    (home / "runtime.json").write_text(json.dumps({"schema": "ggufone.runtime/v1",
                                                  "variant": "linux-x64-vulkan",
                                                  "dir": str(vulkan)}))
    assert finder.find_runtime(home=home) == vulkan


def test_find_runtime_falls_back_to_a_scan_when_the_record_is_stale(
        tmp_path: pathlib.Path) -> None:
    home = tmp_path / "home"
    vulkan = home / "runtime" / "b11026-linux-x64-vulkan"
    vulkan.mkdir(parents=True)
    (vulkan / "libllama.so").write_bytes(b"\x7fELF fake\n")
    (home / "runtime.json").write_text(json.dumps({"variant": "linux-x64-cuda-12.8",
                                                  "dir": str(home / "gone")}))
    assert finder.find_runtime(home=home) == vulkan


def test_a_fallback_tier_without_a_pinned_bundle_is_skipped(tmp_path: pathlib.Path) -> None:
    """A lock that carries no vulkan asset must not abort the chain — it is recorded."""
    cache = bundle_cache(tmp_path, ("cuda", "cpu"))
    lock = pins.load_lock(multi_lock(cache, ("cuda", "cpu")))

    result = install.install("auto", home=tmp_path / "home", lock=lock, offline_cache=cache,
                             free_bytes=1 << 40, probes=GPU_HOST)

    assert result["variant"] == "linux-x64-cpu"
    assert [attempt["backend"] for attempt in result["fallback_attempts"]] == ["cuda", "vulkan"]
    assert "E_RUNTIME_MISSING" in result["fallback_attempts"][1]["reason"]


def test_install_reports_a_bundle_that_carries_no_such_backend(tmp_path: pathlib.Path) -> None:
    """Variant says vulkan but the archive holds no libggml-vulkan: name it, then fall back."""
    cache = bundle_cache(tmp_path, ("cuda", "cpu"))
    vulkan_asset = cache / ASSET["vulkan"]
    vulkan_asset.write_bytes((cache / ASSET["cpu"]).read_bytes())  # cpu layout under a vulkan name
    lock = pins.load_lock(multi_lock(cache, ("cuda", "vulkan", "cpu")))

    result = install.install("auto", home=tmp_path / "home", lock=lock, offline_cache=cache,
                             free_bytes=1 << 40, probes=GPU_HOST)

    assert result["variant"] == "linux-x64-cpu"
    reasons = [attempt["reason"] for attempt in result["fallback_attempts"]]
    assert any("carries no vulkan backend" in reason for reason in reasons), reasons


# ------------------------------------------------------------------ doctor
def test_doctor_reports_the_working_backend_and_the_recorded_fallback(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("GGUFONE_HOME", str(home))
    monkeypatch.delenv("GGUFONE_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("GGUFONE_LOCK", raising=False)
    monkeypatch.setenv("GGUFONE_DEEP_PROBE", "0")
    monkeypatch.setattr(pins, "current_host", lambda: GPU_HOST)

    rt = home / "runtime" / "b11026-linux-x64-vulkan"
    rt.mkdir(parents=True)
    for lib in ("libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so",
                "libggml-vulkan.so"):
        (rt / lib).write_bytes(b"\x7fELF fake\nllama_model_spark2_5\x00build 11026\n")
    (home / "runtime.json").write_text(json.dumps({
        "schema": "ggufone.runtime/v1", "variant": "linux-x64-vulkan", "build": 11026,
        "backend_requested": "cuda", "backend_working": "vulkan", "libllama_sha256": None,
        "fallback_reason": f"cuda does not load on this host ({CUDA_LOAD_ERROR})",
        "fallback_attempts": [{"backend": "cuda", "variant": "linux-x64-cuda-12.8",
                               "reason": f"cuda does not load on this host ({CUDA_LOAD_ERROR})"}]}))

    assert cli.main(["doctor", "--json"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["runtime"]["backends"] == ["cpu", "vulkan"]
    assert report["runtime"]["working_backend"] == "vulkan"
    assert "libcudart.so.12" in report["runtime"]["fallback_reason"]
    assert report["expected_backend"] == "cuda"
    checks = {c["id"]: c for c in report["checks"]}
    assert checks["runtime.fallback"]["status"] == "warn"
    assert "cuda" in checks["runtime.fallback"]["detail"]
    assert "vulkan" in checks["runtime.backends"]["detail"]


def test_doctor_marks_the_expected_backend_ok_when_it_really_loads(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("GGUFONE_HOME", str(home))
    monkeypatch.delenv("GGUFONE_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("GGUFONE_LOCK", raising=False)
    monkeypatch.setenv("GGUFONE_DEEP_PROBE", "0")
    monkeypatch.setattr(pins, "current_host", lambda: GPU_HOST)

    rt = home / "runtime" / "b11026-linux-x64-cuda-12.8"
    rt.mkdir(parents=True)
    for lib in ("libllama.so", "libggml.so", "libggml-base.so", "libggml-cuda.so"):
        (rt / lib).write_bytes(b"\x7fELF fake\nllama_model_spark2_5\x00build 11026\n")

    assert cli.main(["doctor", "--json"]) == 2
    report = json.loads(capsys.readouterr().out)
    checks = {c["id"]: c for c in report["checks"]}
    assert checks["runtime.accelerator"]["status"] == "ok"
    assert "cuda" in checks["runtime.accelerator"]["detail"]
    assert "runtime.fallback" not in checks
    assert report["runtime"]["backend_errors"] == {}
    assert report["runtime"]["fallback_reason_code"] is None    # nothing was recorded: no code
    # this bundle carries only libggml-cuda.so: the probed list says exactly that
    assert report["backend"] == "cuda" and report["backends"] == ["cuda"]


# ------------------------------------------------------------------ machine-readable reasons
# The card asks for the machine-readable reason of *each* step, not one prose blob: every
# attempt carries a stable `code` (a member of `install.REASON_CODES`) next to the human text,
# and the record + `init --json` + `doctor --json` repeat the code of the first fallback.
def test_a_backend_that_does_not_load_here_is_coded_loader_error(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, in_process_scan: None) -> None:
    cache = bundle_cache(tmp_path, ("cuda", "vulkan", "cpu"))
    lock = pins.load_lock(multi_lock(cache, ("cuda", "vulkan", "cpu")))
    monkeypatch.setattr(capability, "load_backend_library", fake_loader({"cuda": CUDA_LOAD_ERROR}))

    result = install.install("auto", home=tmp_path / "home", lock=lock, offline_cache=cache,
                             free_bytes=1 << 40, probes=GPU_HOST)

    assert result["variant"] == "linux-x64-vulkan"
    assert result["fallback_reason_code"] == install.REASON_LOADER_ERROR
    attempt = result["fallback_attempts"][0]
    assert attempt["code"] == install.REASON_LOADER_ERROR
    assert attempt["code"] in install.REASON_CODES
    assert set(attempt) == {"backend", "variant", "code", "reason"}
    assert CUDA_LOAD_ERROR in attempt["reason"]          # the code never replaces the loader text
    record = json.loads((tmp_path / "home" / "runtime.json").read_text())
    assert record["fallback_reason_code"] == install.REASON_LOADER_ERROR
    assert record["fallback_attempts"][0]["code"] == install.REASON_LOADER_ERROR
    assert record["fallback_attempts"][0]["variant"] == "linux-x64-cuda-12.8"


def test_the_preflight_skip_is_coded_system_libs_missing(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    cache = bundle_cache(tmp_path, ("cuda", "vulkan", "cpu"))
    lock_path = multi_lock(cache, ("cuda", "vulkan", "cpu"))
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["llama_cpp"]["system_libs"] = {"linux-x64-cuda-12.8": ["libcudart.so.12"]}
    lock_path.write_text(json.dumps(payload), encoding="utf-8")
    lock = pins.load_lock(lock_path)
    monkeypatch.setattr(install, "PREFLIGHT_SYSTEM_LIBS", lambda names: {
        name: f"{name}: cannot open shared object file: No such file or directory"
        for name in names})

    result = install.install("auto", home=tmp_path / "home", lock=lock, offline_cache=cache,
                             free_bytes=1 << 40, probes=GPU_HOST)

    attempt = result["fallback_attempts"][0]
    assert attempt["backend"] == "cuda"
    assert attempt["code"] == install.REASON_SYSTEM_LIBS_MISSING
    assert "libcudart.so.12: cannot open shared object file" in attempt["reason"]
    assert result["fallback_reason_code"] == install.REASON_SYSTEM_LIBS_MISSING


def test_a_tier_this_lock_does_not_pin_is_coded_no_asset(tmp_path: pathlib.Path) -> None:
    """The chain continues past a variant the lock has no asset for — and says so by code."""
    cache = bundle_cache(tmp_path, ("cuda", "cpu"))
    lock = pins.load_lock(multi_lock(cache, ("cuda", "cpu")))

    result = install.install("auto", home=tmp_path / "home", lock=lock, offline_cache=cache,
                             free_bytes=1 << 40, probes=GPU_HOST)

    assert result["variant"] == "linux-x64-cpu"
    codes = [attempt["code"] for attempt in result["fallback_attempts"]]
    assert codes == [install.REASON_LOADER_ERROR, install.REASON_NO_ASSET]
    assert "E_RUNTIME_MISSING" in result["fallback_attempts"][1]["reason"]


def test_a_bundle_that_carries_no_such_backend_is_coded_backend_absent(
        tmp_path: pathlib.Path) -> None:
    cache = bundle_cache(tmp_path, ("cuda", "cpu"))
    # the cpu layout under a vulkan name: the variant says vulkan, the archive says cpu
    (cache / ASSET["vulkan"]).write_bytes((cache / ASSET["cpu"]).read_bytes())
    lock = pins.load_lock(multi_lock(cache, ("cuda", "vulkan", "cpu")))

    result = install.install("auto", home=tmp_path / "home", lock=lock, offline_cache=cache,
                             free_bytes=1 << 40, probes=GPU_HOST)

    by_backend = {attempt["backend"]: attempt for attempt in result["fallback_attempts"]}
    assert by_backend["vulkan"]["code"] == install.REASON_BACKEND_ABSENT
    assert "carries no vulkan backend" in by_backend["vulkan"]["reason"]


def test_a_dead_probe_child_is_coded_probe_failed(monkeypatch: pytest.MonkeyPatch,
                                                  tmp_path: pathlib.Path) -> None:
    """A probe that cannot run is data: every accelerator is unusable, with the code to match."""
    cache = bundle_cache(tmp_path, ("cuda", "vulkan", "cpu"))
    lock = pins.load_lock(multi_lock(cache, ("cuda", "vulkan", "cpu")))
    monkeypatch.setattr(isolated, "child_command",
                        lambda: [sys.executable, "-c", "raise SystemExit(9)"])

    result = install.install("auto", home=tmp_path / "home", lock=lock, offline_cache=cache,
                             free_bytes=1 << 40, probes=GPU_HOST)

    assert result["variant"] == "linux-x64-cpu"
    assert [attempt["code"] for attempt in result["fallback_attempts"]] == [
        install.REASON_PROBE_FAILED, install.REASON_PROBE_FAILED]
    assert all("isolated probe" in attempt["reason"]
               for attempt in result["fallback_attempts"])


def test_reason_codes_are_a_closed_set() -> None:
    assert frozenset({
        install.REASON_LOADER_ERROR, install.REASON_SYSTEM_LIBS_MISSING, install.REASON_NO_ASSET,
        install.REASON_BACKEND_ABSENT, install.REASON_PROBE_FAILED,
        install.REASON_UNUSABLE}) == install.REASON_CODES


def test_doctor_reports_the_working_backend_the_list_and_the_fallback_code(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("GGUFONE_HOME", str(home))
    monkeypatch.delenv("GGUFONE_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("GGUFONE_DEEP_PROBE", "0")
    monkeypatch.setattr(pins, "current_host", lambda: GPU_HOST)

    rt = home / "runtime" / "b11026-linux-x64-vulkan"
    rt.mkdir(parents=True)
    for lib in ("libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so",
                "libggml-vulkan.so"):
        (rt / lib).write_bytes(b"\x7fELF fake\nllama_model_spark2_5\x00build 11026\n")
    (home / "runtime.json").write_text(json.dumps({
        "schema": "ggufone.runtime/v1", "variant": "linux-x64-vulkan", "build": 11026,
        "backend_requested": "cuda", "backend_working": "vulkan",
        "fallback_reason": f"cuda does not load on this host ({CUDA_LOAD_ERROR})",
        "fallback_reason_code": install.REASON_LOADER_ERROR,
        "fallback_attempts": [{"backend": "cuda", "variant": "linux-x64-cuda-12.8",
                               "code": install.REASON_LOADER_ERROR,
                               "reason": f"cuda does not load on this host ({CUDA_LOAD_ERROR})"}]}))

    assert cli.main(["doctor", "--json"]) == 2
    report = json.loads(capsys.readouterr().out)
    # the acceptance reads `doctor --json` for the WORKING backend + the probed list
    assert report["runtime"]["working_backend"] == "vulkan"
    assert report["runtime"]["backends"] == ["cpu", "vulkan"]
    assert report["backend"] == "vulkan"
    assert report["backends"] == ["cpu", "vulkan"]
    assert report["runtime"]["fallback_reason_code"] == install.REASON_LOADER_ERROR
    checks = {check["id"]: check for check in report["checks"]}
    assert checks["runtime.fallback"]["status"] == "warn"
    assert install.REASON_LOADER_ERROR in checks["runtime.fallback"]["detail"]
    assert "libcudart.so.12" in checks["runtime.fallback"]["detail"]


def test_doctor_reads_the_backends_off_the_bundle_it_found(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys) -> None:
    """No table of backends: the list follows the files (and the dlopen result) of this bundle."""
    home = tmp_path / "home"
    monkeypatch.setenv("GGUFONE_HOME", str(home))
    monkeypatch.delenv("GGUFONE_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("GGUFONE_DEEP_PROBE", "0")
    monkeypatch.setattr(pins, "current_host", lambda: GPU_HOST)
    rt = home / "runtime" / "b11026-linux-x64-cpu"
    rt.mkdir(parents=True)
    for lib in ("libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so"):
        (rt / lib).write_bytes(b"\x7fELF fake\nllama_model_spark2_5\x00build 11026\n")

    assert cli.main(["doctor", "--json"]) == 2
    first = json.loads(capsys.readouterr().out)
    assert first["backends"] == ["cpu"] and first["runtime"]["working_backend"] == "cpu"

    (rt / "libggml-vulkan.so").write_bytes(b"\x7fELF fake\n")   # the same dir, one more backend
    assert cli.main(["doctor", "--json"]) == 2
    second = json.loads(capsys.readouterr().out)
    assert second["backends"] == ["cpu", "vulkan"]
    assert second["runtime"]["working_backend"] == "vulkan"


def test_the_working_backend_is_decided_by_dlopen_not_by_the_file_list(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, in_process_scan: None) -> None:
    """`backends` is what the bundle carries; `accelerator()` is what this host can drive."""
    rt = tmp_path / "rt"
    rt.mkdir()
    for lib in ("libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so",
                "libggml-cuda.so"):
        (rt / lib).write_bytes(b"\x7fELF fake\n")
    monkeypatch.setattr(capability, "load_backend_library", fake_loader({"cuda": CUDA_LOAD_ERROR}))

    probe = capability.probe_runtime(rt, deep=True, run_tools=False)

    assert probe.backends == ("cpu", "cuda")
    assert probe.usable("cuda") is False
    assert probe.accelerator() == "cpu"
    assert probe.backend_errors["cuda"] == CUDA_LOAD_ERROR
