"""Runtime pre-flight: symbols, build number, backends, arch gate (SPEC 2.2, A-E1a-3, A-E1a-9).

Everything here runs offline against synthetic runtime directories; the live probe is the
oracle's section B (A-E1a-1) plus `typed-gguf doctor` on the real install.
"""
from __future__ import annotations

import json
import pathlib
import stat
from dataclasses import replace

import pytest

from typed_gguf.errors import TypedGgufError
from typed_gguf.runtime import capability, finder, pins

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The pinned Windows zip's own file names — `llama-b11026-bin-win-cpu-x64.zip` at its archive root
#: (verified against the real asset, card t_8dab8b3a). A *complete* Windows bundle, so every gate
#: here has to answer for these names and not for the Linux SONAMEs.
WINDOWS_LIBS = ("llama.dll", "ggml.dll", "ggml-base.dll")
WINDOWS_TOOLS = ("llama-cli.exe", "llama-fit-params.exe", "llama-tokenize.exe")
#: The real bytes the pinned `llama.dll` (3 167 232 B) carries around the **qwen2** implementation
#: class: MSVC RTTI type descriptors, at byte offsets 3031770 (`graph@…`) and 3073028 (`…@@`) —
#: measured with a byte scan of the real DLL. What matters is the *shape*: the arch class name is
#: followed by `@@`, and the ELF-style `llama_model_qwen2\x00` form occurs **0** times in the whole
#: PE (7 times in the pinned Linux `libllama.so.0`, all of them Itanium typeinfo symbol tails:
#: `_ZTS17llama_model_qwen2\0`). That difference is what the first live matrix run tripped on.
WINDOWS_ARCH_DESCRIPTOR = b".?AUgraph@llama_model_qwen2@@"
WINDOWS_ARCH_DESCRIPTOR_FLAT = b".?AUllama_model_qwen2@@"
#: The CLI banner the pinned b11026 CLI really prints — run on this box against the real Linux twin
#: of the same release (`./llama-cli --version` in `b11026-linux-x64-cpu`). It is the ONLY source
#: of the build number on any platform: the DLLs carry the version string (`0.4.1-dev` is in
#: `llama.dll`) and the format `version: %s (build %d, commit %s)`, but the build number itself is
#: a compiled integer — no shipped file contains the bytes `11026`.
REAL_CLI_BANNER = "version: 0.4.1-dev (build 11026, commit b49650adb)"


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
        if tool.startswith("llama-cli") and build is not None:
            script.write_text(f"#!/bin/sh\necho 'version: 0.4.1-dev (build {build}, commit "
                              f"b49650adb)' >&2\n")
        elif tool.startswith("llama-fit-params"):
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


@pytest.mark.needs_fork
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
    with pytest.raises(TypedGgufError) as exc:
        capability.require_arch(rt, "spark2_5", build=11026)
    assert exc.value.code == "E_MODEL_ARCH_UNSUPPORTED"
    msg = str(exc.value)
    assert "spark2_5" in msg and "b11026" in msg and "init --force" in msg


def test_require_arch_rejects_a_runtime_older_than_the_arch_needs(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path, build=10715, archs=("spark2_5",))
    with pytest.raises(TypedGgufError) as exc:
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
@pytest.mark.needs_fork
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


@pytest.mark.needs_fork
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


@pytest.mark.needs_fork
def test_probe_records_the_tool_checks(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path)
    got = capability.probe_runtime(rt, deep=False)
    assert got.tools["llama-fit-params"].endswith("llama-fit-params")
    assert got.fit_params_help_exit == 0


@pytest.mark.needs_fork
def test_probe_fails_when_fit_params_help_breaks(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path)
    (rt / "llama-fit-params").write_text("#!/bin/sh\nexit 3\n")
    got = capability.probe_runtime(rt, deep=False)
    assert got.fit_params_help_exit == 3
    assert any("llama-fit-params" in f for f in got.failures())


@pytest.mark.needs_fork
def test_probe_fails_when_build_is_below_the_spark2_5_floor(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path, build=10715)
    got = capability.probe_runtime(rt, deep=False)
    assert any("10828" in f for f in got.failures())


@pytest.mark.needs_fork
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
    monkeypatch.setenv("TYPED_GGUF_HOME", str(tmp_path))
    assert finder.runtime_record() is None
    path = finder.write_runtime_record({"schema": "typed_gguf.runtime/v1", "tag": "b11026"})
    assert path == tmp_path / "runtime.json"
    assert json.loads(path.read_text())["tag"] == "b11026"
    assert finder.runtime_record()["tag"] == "b11026"


def test_find_runtime_honours_the_env_override(tmp_path: pathlib.Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    rt = fake_runtime(tmp_path)
    monkeypatch.setenv("TYPED_GGUF_RUNTIME_DIR", str(rt))
    assert finder.find_runtime() == rt
    assert finder.resolve_runtime() == rt


def test_find_runtime_scans_the_data_home(tmp_path: pathlib.Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPED_GGUF_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("TYPED_GGUF_HOME", str(tmp_path / "home"))
    rt = tmp_path / "home" / "runtime" / "b11026-linux-x64-cpu"
    rt.mkdir(parents=True)
    (rt / "libllama.so").write_bytes(b"\x7fELF fake\n")
    assert finder.find_runtime() == rt


def test_resolve_runtime_raises_a_useful_error_when_absent(tmp_path: pathlib.Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPED_GGUF_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("TYPED_GGUF_HOME", str(tmp_path / "empty"))
    with pytest.raises(TypedGgufError) as exc:
        finder.resolve_runtime()
    assert exc.value.code == "E_RUNTIME_MISSING"
    assert "typed-gguf init" in str(exc.value)


def test_resolve_runtime_is_optional_for_read_only_commands(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPED_GGUF_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("TYPED_GGUF_HOME", str(tmp_path / "empty"))
    assert finder.resolve_runtime(required=False) is None


def test_env_override_pointing_nowhere_is_an_error(tmp_path: pathlib.Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPED_GGUF_RUNTIME_DIR", str(tmp_path / "nope"))
    with pytest.raises(TypedGgufError) as exc:
        finder.find_runtime()
    assert exc.value.code == "E_RUNTIME_MISSING"
    assert "TYPED_GGUF_RUNTIME_DIR" in str(exc.value)


def test_layout_reports_the_pinned_files(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path)
    got = finder.layout(rt)
    assert got.directory == rt
    assert got.libllama.name == "libllama.so"
    assert got.complete
    assert "llama-fit-params" in got.tools


# ------------------------------------------------------------------ the Windows bundle layout
# Card t_8dab8b3a: the first live matrix run drove the PRODUCT against the pinned Windows zip and
# the capability layer answered for the Linux naming rules. Every gate below is one of those rules,
# spelled with the real file names from the zip (nothing here is a Windows-only code path: the
# platform table is the finder's, and the arch evidence is the ABI's own decoration).
def test_the_windows_tool_and_library_names_come_from_one_table() -> None:
    """`llama-cli` -> `llama-cli.exe`, and the lock's SONAMEs -> the DLLs the zip carries."""
    assert finder.tool_name("llama-cli", "windows") == "llama-cli.exe"
    assert finder.tool_name("llama-cli", "linux") == "llama-cli"
    assert finder.tool_names("windows") == WINDOWS_TOOLS
    lock = pins.load_lock()
    assert finder.required_files(lock, system="linux") == (
        "libllama.so", "libggml.so", "libggml-base.so")
    assert finder.required_files(lock, system="windows") == WINDOWS_LIBS
    assert finder.required_files(lock, system="darwin") == (
        "libllama.dylib", "libggml.dylib", "libggml-base.dylib")


def test_layout_resolves_the_windows_tool_files(tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path, libs=WINDOWS_LIBS, tools=WINDOWS_TOOLS)
    got = finder.layout(rt, system="windows")
    assert got.libllama.name == "llama.dll"
    assert got.complete
    assert sorted(got.tools) == ["llama-cli", "llama-fit-params",
                                 "llama-tokenize"], sorted(got.tools)
    assert got.tools["llama-cli"].name == "llama-cli.exe"


def test_required_files_passes_through_a_name_the_table_does_not_know() -> None:
    """A future lock entry is renamed only when the table knows it — never dropped, never a KeyError
    (the mapping is by role; anything else has to survive as it stands)."""
    lock = pins.load_lock()
    widened = replace(lock, required_files=(*lock.required_files, "libmtmd.so"))
    assert finder.required_files(widened, system="windows") == (*WINDOWS_LIBS, "libmtmd.so")


@pytest.mark.needs_fork
def test_build_number_reads_the_windows_cli_banner(tmp_path: pathlib.Path) -> None:
    """The pinned Windows CLI is `llama-cli.exe`; its banner is the only source of the number."""
    rt = fake_runtime(tmp_path, libs=WINDOWS_LIBS, tools=WINDOWS_TOOLS, build=11026)
    assert capability.build_number(rt, system="windows") == 11026
    # and the parse is the product's own: the banner is the real one (`REAL_CLI_BANNER`)
    assert capability.parse_build(REAL_CLI_BANNER) == 11026


def test_build_number_scans_the_windows_library_it_actually_has(tmp_path: pathlib.Path) -> None:
    """No CLI: the byte scan has to read `llama.dll`, not a Linux name that is not there."""
    rt = fake_runtime(tmp_path, libs=WINDOWS_LIBS, tools=(), build=None)
    (rt / "llama.dll").write_bytes(b"\x7fPE fake\x00build b11026\x00")
    assert capability.build_number(rt, system="windows") == 11026


def test_arch_scan_reads_a_windows_decorated_name(tmp_path: pathlib.Path) -> None:
    """AC3: a REAL PE capability answer — the MSVC-decorated arch implementation class."""
    rt = fake_runtime(tmp_path, libs=WINDOWS_LIBS, tools=(), archs=())
    (rt / "llama.dll").write_bytes(b"\x7fPE fake\x00" + WINDOWS_ARCH_DESCRIPTOR
                                   + WINDOWS_ARCH_DESCRIPTOR_FLAT)
    assert capability.supports_arch(rt, "qwen2", system="windows")
    assert not capability.supports_arch(rt, "llama", system="windows")
    # the class exists only for the architectures the DLL was compiled with: the same bytes must
    # NOT make an unimplemented arch look supported
    assert not capability.supports_arch(rt, "spark2_5", system="windows")


def test_require_arch_names_the_library_that_is_really_there(tmp_path: pathlib.Path) -> None:
    """AC4: no hardcoded `libllama.so` in a message a Windows user reads."""
    rt = fake_runtime(tmp_path, libs=WINDOWS_LIBS, tools=(), archs=("spark2_5",))
    with pytest.raises(TypedGgufError) as exc:
        capability.require_arch(rt, "llama", system="windows", build=11026)
    msg = str(exc.value)
    assert exc.value.code == "E_MODEL_ARCH_UNSUPPORTED"
    assert "llama_model_llama in llama.dll" in msg, msg
    assert "libllama.so" not in msg, msg


def test_the_arch_scan_honours_the_build_floor(tmp_path: pathlib.Path) -> None:
    """The class being there is half the answer: an older build is still unsupported (the caller's
    explicit `build` must be what is compared, not a re-read of the bundle).

    `runtime.lock`'s floor for this arch is b10828 (the pinned bundle is the newer b11026, which is
    why the shipped runtime passes), so the boundary is asserted on both sides of it."""
    rt = fake_runtime(tmp_path, libs=WINDOWS_LIBS, tools=(), archs=("spark2_5",))
    assert capability.supports_arch(rt, "spark2_5", system="windows", build=11026)
    assert capability.supports_arch(rt, "spark2_5", system="windows", build=10828)
    assert not capability.supports_arch(rt, "spark2_5", system="windows", build=10827)
    with pytest.raises(TypedGgufError) as exc:
        capability.require_arch(rt, "spark2_5", system="windows", build=10827)
    assert "needs runtime build b10828 or newer" in str(exc.value), str(exc.value)


def test_probe_runtime_accepts_a_complete_windows_bundle(tmp_path: pathlib.Path) -> None:
    """AC5: `runtime.files`/`runtime.loadable` pass for the zip the job flattened."""
    rt = fake_runtime(tmp_path, libs=WINDOWS_LIBS, tools=WINDOWS_TOOLS,
                      ggml_backends=("ggml-cpu-x64.dll", "ggml.dll"))
    probe = capability.probe_runtime(rt, deep=False, system="windows")
    assert probe.missing_files == (), probe.missing_files
    assert probe.error is None, probe.error
    assert probe.present == WINDOWS_LIBS
    assert probe.backends == ("cpu",)
    assert probe.build == 11026
    assert probe.ok(), probe.failures()


def test_the_build_failure_text_names_the_platform_tool_and_library(
        tmp_path: pathlib.Path) -> None:
    rt = fake_runtime(tmp_path, libs=WINDOWS_LIBS, tools=(), build=None)
    probe = capability.probe_runtime(rt, deep=False, system="windows")
    failures = " ".join(probe.failures())
    assert "llama-cli.exe" in failures and "llama.dll" in failures, failures
    assert "libllama.so" not in failures, failures

