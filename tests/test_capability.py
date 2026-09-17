"""Runtime pre-flight: symbols, build number, backends, arch gate (SPEC 2.2, A-E1a-3, A-E1a-9).

Everything here runs offline against synthetic runtime directories; the live probe is the
oracle's section B (A-E1a-1) plus `ggufone doctor` on the real install.
"""
from __future__ import annotations

import json
import pathlib
import stat

import pytest

from ggufone.errors import GgufoneError
from ggufone.runtime import capability, finder

ROOT = pathlib.Path(__file__).resolve().parents[1]


def real_system_lib() -> pathlib.Path | None:
    """A real ELF shared object to stand in for libllama (loads, but has no llama ABI)."""
    for candidate in ("/lib/x86_64-linux-gnu/libm.so.6", "/usr/lib64/libm.so.6",
                      "/lib64/libm.so.6", "/usr/lib/libm.so.6"):
        path = pathlib.Path(candidate)
        if path.exists():
            return path
    return None


def fake_runtime(tmp_path: pathlib.Path, *, build: int | None = 11026,
                 archs: tuple[str, ...] = ("spark2_5",),
                 libs: tuple[str, ...] = ("libllama.so", "libggml.so", "libggml-base.so"),
                 tools: tuple[str, ...] = ("llama-cli", "llama-fit-params"),
                 ggml_backends: tuple[str, ...] = ("libggml-cpu.so",),
                 real_libs: bool = False) -> pathlib.Path:
    rt = tmp_path / "runtime"
    rt.mkdir(parents=True, exist_ok=True)
    for lib in libs:
        if real_libs:
            source = real_system_lib()
            if source is None:
                pytest.skip("no system shared library available to stand in for a runtime")
            (rt / lib).write_bytes(source.read_bytes())
        else:
            payload = b"\x7fELF fake\n" + b"".join(b"llama_model_" + a.encode() + b"\x00"
                                                   for a in archs)
            (rt / lib).write_bytes(payload)
    for backend in ggml_backends:
        (rt / backend).write_bytes(b"\x7fELF fake\n")
    for tool in tools:
        script = rt / tool
        if tool == "llama-cli" and build is not None:
            script.write_text(f"#!/bin/sh\necho 'version: 0.4.1-dev (build {build}, commit "
                              f"b49650adb)' >&2\n")
        elif tool == "llama-fit-params":
            script.write_text("#!/bin/sh\necho usage: llama-fit-params [options]\nexit 0\n")
        else:
            script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return rt


# ------------------------------------------------------------------ build parsing
def test_parse_build_from_cli_banner() -> None:
    text = "version: 0.4.1-dev (build 11026, commit b49650adb)\n"
    assert capability.parse_build(text) == 11026
    assert capability.parse_build("build b10828") == 10828
    assert capability.parse_build("nothing here") is None


def test_build_number_reads_llama_cli(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path, build=11026)
    assert capability.build_number(rt) == 11026


def test_build_number_is_none_when_no_binary_and_no_string(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path, build=None, tools=())
    assert capability.build_number(rt) is None


def test_build_tag_formats_the_tag() -> None:
    assert capability.build_tag(11026) == "b11026"
    assert capability.build_tag(10828) == "b10828"


# ------------------------------------------------------------------ arch gate
def test_arch_symbol_scan_finds_the_arch_implementation(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path, archs=("spark2_5", "qwen35"))
    assert capability.supports_arch(rt, "spark2_5")
    assert capability.supports_arch(rt, "qwen35")
    assert not capability.supports_arch(rt, "llama")


def test_require_arch_passes_for_a_supported_build(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path)
    capability.require_arch(rt, "spark2_5", build=11026)  # must not raise


def test_require_arch_rejects_an_unknown_arch_with_actionable_text(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path, archs=("qwen35",))
    with pytest.raises(GgufoneError) as exc:
        capability.require_arch(rt, "spark2_5", build=11026)
    assert exc.value.code == "E_MODEL_ARCH_UNSUPPORTED"
    msg = str(exc.value)
    assert "spark2_5" in msg and "b11026" in msg and "init --force" in msg


def test_require_arch_rejects_a_runtime_older_than_the_arch_needs(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path, build=10715, archs=("spark2_5",))
    with pytest.raises(GgufoneError) as exc:
        capability.require_arch(rt, "spark2_5", build=10715)
    assert exc.value.code == "E_MODEL_ARCH_UNSUPPORTED"
    assert "10828" in str(exc.value)  # names the minimum build


def test_arch_scan_tolerates_hyphenated_arch_names(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path, archs=("k2_horizon",))
    assert capability.supports_arch(rt, "k2-horizon")


# ------------------------------------------------------------------ backends
def test_backends_lists_the_compiled_in_accelerators(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path, ggml_backends=("libggml-cpu.so", "libggml-cpu-alderlake.so",
                                               "libggml-vulkan.so", "libggml-base.so"))
    assert capability.backends(rt) == ["cpu", "vulkan"]


def test_backends_empty_when_only_llama_is_present(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path, ggml_backends=())
    assert capability.backends(rt) == []


# ------------------------------------------------------------------ probe
def test_probe_is_deep_by_default_and_shallow_on_request(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path)
    deep = capability.probe_runtime(rt, deep=True)
    assert deep.deep is True
    assert deep.symbols_checked is True
    assert deep.error  # a fake .so cannot even be dlopen'ed
    assert not deep.ok()
    shallow = capability.probe_runtime(rt, deep=False)
    assert shallow.deep is False
    assert shallow.symbols_checked is False
    assert shallow.build == 11026
    assert shallow.failures() == []


def test_deep_probe_reports_symbols_missing_from_a_real_but_wrong_library(
        tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path, real_libs=True)
    got = capability.probe_runtime(rt, deep=True)
    assert got.error is None
    assert got.missing_symbols  # the stand-in library carries none of the llama ABI
    assert any("do not resolve" in f for f in got.failures())


def test_probe_reports_missing_files_as_failures(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path, libs=("libllama.so", "libggml.so"))
    got = capability.probe_runtime(rt, deep=False)
    assert "libggml-base.so" in got.missing_files
    assert any("libggml-base.so" in f for f in got.failures())


def test_probe_warns_when_no_accelerator_is_present(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path)
    got = capability.probe_runtime(rt, deep=False)
    assert any("accelerator" in w for w in got.warnings())


def test_probe_accepts_the_expected_accelerator(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path, ggml_backends=("libggml-cpu.so", "libggml-vulkan.so"))
    got = capability.probe_runtime(rt, deep=False, expect_backend="vulkan")
    assert "vulkan" in got.backends
    assert not any("accelerator" in w for w in got.warnings())


def test_probe_flags_the_vulkan_warmup_risk(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path, ggml_backends=("libggml-vulkan.so",))
    got = capability.probe_runtime(rt, deep=False)
    assert any("W_VULKAN_WARMUP" in w for w in got.warnings())


def test_probe_records_the_tool_checks(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path)
    got = capability.probe_runtime(rt, deep=False)
    assert got.tools["llama-fit-params"].endswith("llama-fit-params")
    assert got.fit_params_help_exit == 0


def test_probe_fails_when_fit_params_help_breaks(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path)
    (rt / "llama-fit-params").write_text("#!/bin/sh\nexit 3\n")
    got = capability.probe_runtime(rt, deep=False)
    assert got.fit_params_help_exit == 3
    assert any("llama-fit-params" in f for f in got.failures())


def test_probe_fails_when_build_is_below_the_spark2_5_floor(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path, build=10715)
    got = capability.probe_runtime(rt, deep=False)
    assert any("10828" in f for f in got.failures())


def test_probe_warns_when_the_build_is_not_the_pinned_tag(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path, build=12000)
    got = capability.probe_runtime(rt, deep=False)
    assert got.build == 12000
    assert any("b11026" in w for w in got.warnings())


def test_probe_reports_an_unloadable_runtime_as_a_failure(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path, libs=())
    got = capability.probe_runtime(rt, deep=False)
    assert not got.ok()
    assert got.error


def test_probe_can_skip_the_tool_subprocesses(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path)
    got = capability.probe_runtime(rt, deep=False, run_tools=False)
    assert got.fit_params_help_exit is None


# ------------------------------------------------------------------ runtime record
def test_runtime_record_roundtrip(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GGUFONE_HOME", str(tmp_path))
    assert finder.runtime_record() is None
    path = finder.write_runtime_record({"schema": "ggufone.runtime/v1", "tag": "b11026"})
    assert path == tmp_path / "runtime.json"
    assert json.loads(path.read_text())["tag"] == "b11026"
    assert finder.runtime_record()["tag"] == "b11026"


def test_find_runtime_honours_the_env_override(tmp_path: pathlib.Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    rt = fake_runtime(tmp_path)
    monkeypatch.setenv("GGUFONE_RUNTIME_DIR", str(rt))
    assert finder.find_runtime() == rt
    assert finder.resolve_runtime() == rt


def test_find_runtime_scans_the_data_home(tmp_path: pathlib.Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GGUFONE_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("GGUFONE_HOME", str(tmp_path / "home"))
    rt = tmp_path / "home" / "runtime" / "b11026-linux-x64-cpu"
    rt.mkdir(parents=True)
    (rt / "libllama.so").write_bytes(b"\x7fELF fake\n")
    assert finder.find_runtime() == rt


def test_resolve_runtime_raises_a_useful_error_when_absent(tmp_path: pathlib.Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GGUFONE_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("GGUFONE_HOME", str(tmp_path / "empty"))
    with pytest.raises(GgufoneError) as exc:
        finder.resolve_runtime()
    assert exc.value.code == "E_RUNTIME_MISSING"
    assert "ggufone init" in str(exc.value)


def test_resolve_runtime_is_optional_for_read_only_commands(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GGUFONE_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("GGUFONE_HOME", str(tmp_path / "empty"))
    assert finder.resolve_runtime(required=False) is None


def test_env_override_pointing_nowhere_is_an_error(tmp_path: pathlib.Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GGUFONE_RUNTIME_DIR", str(tmp_path / "nope"))
    with pytest.raises(GgufoneError) as exc:
        finder.find_runtime()
    assert exc.value.code == "E_RUNTIME_MISSING"
    assert "GGUFONE_RUNTIME_DIR" in str(exc.value)


def test_layout_reports_the_pinned_files(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path)
    got = finder.layout(rt)
    assert got.directory == rt
    assert got.libllama.name == "libllama.so"
    assert got.complete
    assert "llama-fit-params" in got.tools
