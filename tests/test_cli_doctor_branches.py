"""E1a doctor/CLI branches on broken or drifted installs (A-E1a-3, A-E1a-6).

These are the failure-path tests: a registry that cannot be parsed, a runtime whose recorded
libllama.so SHA-256 no longer matches, and a runtime older than the arch floor. Each must
report a specific check id + exit code instead of a traceback.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import stat

import pytest

from typed_gguf import cli
from typed_gguf.registry import store
from typed_gguf.runtime import pins


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    home = tmp_path / "home"
    monkeypatch.setenv("TYPED_GGUF_HOME", str(home))
    monkeypatch.delenv("TYPED_GGUF_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("TYPED_GGUF_LOCK", raising=False)
    monkeypatch.setenv("TYPED_GGUF_DEEP_PROBE", "0")
    # deterministic GPU-less machine (see tests/test_cli_e1a.py::_isolated)
    monkeypatch.setattr(pins, "current_host",
                        lambda: pins.fake_host(system="linux", machine="x86_64"))
    return tmp_path


def make_runtime(base: pathlib.Path, *, build: int = 11026) -> pathlib.Path:
    rt = base / "runtime-bundle"
    rt.mkdir(parents=True, exist_ok=True)
    for lib in ("libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so"):
        (rt / lib).write_bytes(b"\x7fELF fake\nllama_model_spark2_5\x00")
    cli_path = rt / "llama-cli"
    cli_path.write_text(f"#!/bin/sh\necho 'version: 0.4.1-dev (build {build}, commit x)' >&2\n")
    fit = rt / "llama-fit-params"
    fit.write_text("#!/bin/sh\nexit 0\n")
    for exe in (cli_path, fit):
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return rt


def test_doctor_reports_a_quarantined_registry(tmp_path: pathlib.Path,
                                               monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("TYPED_GGUF_RUNTIME_DIR", str(make_runtime(tmp_path)))
    store.registry_path().parent.mkdir(parents=True, exist_ok=True)
    store.registry_path().write_text("{ not json at all")
    assert cli.main(["doctor", "--json"]) == 2
    report = json.loads(capsys.readouterr().out)
    checks = {c["id"]: c for c in report["checks"]}
    assert checks["registry.corrupt"]["status"] == "warn"
    assert "E_REGISTRY_CORRUPT" in checks["registry.corrupt"]["detail"]
    assert list(store.registry_path().parent.glob("registry.json.corrupt-*"))


def test_doctor_fails_when_the_recorded_libllama_sha_drifted(tmp_path: pathlib.Path,
                                                             monkeypatch: pytest.MonkeyPatch,
                                                             capsys) -> None:
    rt = make_runtime(tmp_path)
    monkeypatch.setenv("TYPED_GGUF_RUNTIME_DIR", str(rt))
    (store.data_home()).mkdir(parents=True, exist_ok=True)
    finder_record = {"schema": "typed_gguf.runtime/v1", "variant": "linux-x64-cpu",
                     "libllama_sha256": "0" * 64, "build": 11026}
    store.runtime_record_path().write_text(json.dumps(finder_record))
    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    check = next(c for c in report["checks"] if c["id"] == "runtime.sha_recorded")
    assert check["status"] == "fail"
    assert "differs" in check["detail"]


def test_doctor_fails_on_a_runtime_older_than_the_arch_floor(tmp_path: pathlib.Path,
                                                             monkeypatch: pytest.MonkeyPatch,
                                                             capsys) -> None:
    rt = make_runtime(tmp_path, build=10715)
    monkeypatch.setenv("TYPED_GGUF_RUNTIME_DIR", str(rt))
    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    check = next(c for c in report["checks"] if c["id"] == "runtime.build")
    assert check["status"] == "fail" and "10828" in check["detail"]


def test_doctor_fails_when_the_recorded_sha_is_missing(tmp_path: pathlib.Path,
                                                       monkeypatch: pytest.MonkeyPatch,
                                                       capsys) -> None:
    monkeypatch.setenv("TYPED_GGUF_RUNTIME_DIR", str(make_runtime(tmp_path)))
    assert cli.main(["doctor", "--json"]) == 2
    report = json.loads(capsys.readouterr().out)
    check = next(c for c in report["checks"] if c["id"] == "runtime.sha_recorded")
    assert check["status"] == "warn"


GPU_HOST = pins.fake_host(system="linux", machine="x86_64", has_nvidia_smi=True)


def fallback_record() -> str:
    """A runtime record as `init` writes it after a cuda -> vulkan fallback (the live box)."""
    reason = ("pre-flight: the pinned linux-x64-cuda-12.8 bundle links libcudart.so.12, "
              "libcublas.so.12, libcuda.so.1, which this host cannot load")
    return json.dumps({
        "schema": "typed_gguf.runtime/v1", "variant": "linux-x64-vulkan", "build": 11026,
        "backend_requested": "cuda", "backend_working": "vulkan",
        "fallback_reason": reason, "fallback_reason_code": "system_libs_missing",
        "fallback_attempts": [{"backend": "cuda", "variant": "linux-x64-cuda-12.8",
                               "code": "system_libs_missing", "reason": reason}]})


def write_fallback_record() -> None:
    store.data_home().mkdir(parents=True, exist_ok=True)
    store.runtime_record_path().write_text(fallback_record())


def test_doctor_human_output_prints_the_fallback_and_the_model_line(tmp_path: pathlib.Path,
                                                                   monkeypatch: pytest.MonkeyPatch,
                                                                   capsys) -> None:
    """The human branch read `runtime['backend_working']`; the report only ever builds
    `working_backend` — so any record carrying a fallback_reason died with E_INTERNAL/exit 4,
    the report stopping right after the `runtime:` line."""
    monkeypatch.setattr(pins, "current_host", lambda: GPU_HOST)
    rt = make_runtime(tmp_path)
    (rt / "libggml-vulkan.so").write_bytes(b"\x7fELF fake\n")
    monkeypatch.setenv("TYPED_GGUF_RUNTIME_DIR", str(rt))
    write_fallback_record()

    assert cli.main(["doctor"]) == 2

    captured = capsys.readouterr()
    assert "E_INTERNAL" not in captured.err
    assert captured.err == ""
    assert "  fallback: cuda -> vulkan [system_libs_missing]" in captured.out
    assert "  model:   " in captured.out                       # the trailing summary line


def test_doctor_human_output_survives_a_fallback_record_without_a_live_probe(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """No bundle to probe (`TYPED_GGUF_RUNTIME_DIR` unset, empty home): `working_backend` is
    None, so the defensive read must print a placeholder rather than `None` — or crash."""
    monkeypatch.setattr(pins, "current_host", lambda: GPU_HOST)
    write_fallback_record()

    assert cli.main(["doctor"]) == 1                            # runtime.present fails

    captured = capsys.readouterr()
    assert "E_INTERNAL" not in captured.err
    assert captured.err == ""
    assert "  fallback: cuda -> none [system_libs_missing]" in captured.out
    assert "  model:   " in captured.out


def test_models_search_text_mode(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(cli.hf, "search", lambda query, limit=20: [
        {"id": "acme/gguf-one", "downloads": 12, "likes": 3, "tags": ["gguf"]}])
    assert cli.main(["models", "search", "acme"]) == 0
    out = capsys.readouterr().out
    assert "acme/gguf-one" in out and "downloads=12" in out


def test_models_search_without_results_is_not_an_error(monkeypatch: pytest.MonkeyPatch,
                                                       capsys) -> None:
    monkeypatch.setattr(cli.hf, "search", lambda query, limit=20: [])
    assert cli.main(["models", "search", "zzz"]) == 0
    assert "no GGUF repos matched" in capsys.readouterr().out


def test_models_ls_text_lists_entries_and_flags_a_missing_file(tmp_path: pathlib.Path,
                                                               capsys) -> None:
    seed = tmp_path / "m.gguf"
    seed.write_bytes(b"GGUF" + b"\x00" * 64)
    entry = store.Entry(alias="one", path=str(seed), size=seed.stat().st_size, quant="Q8_0",
                        arch="spark2_5", license="mit",
                        sha256=hashlib.sha256(seed.read_bytes()).hexdigest())
    registry = store.Registry()
    store.add_entry(registry, entry)
    store.save_registry(registry)
    assert cli.main(["models", "ls"]) == 0
    out = capsys.readouterr().out
    assert "one" in out and "Q8_0" in out and "[ok]" in out
    seed.unlink()
    assert cli.main(["models", "ls"]) == 0
    assert "[MISSING]" in capsys.readouterr().out


def test_models_use_text_mode(tmp_path: pathlib.Path, capsys) -> None:
    seed = tmp_path / "m.gguf"
    seed.write_bytes(b"GGUF")
    registry = store.Registry()
    store.add_entry(registry, store.Entry(alias="one", path=str(seed)))
    store.save_registry(registry)
    assert cli.main(["models", "use", "one"]) == 0
    assert "current" in capsys.readouterr().out


def test_models_rm_text_mode_and_missing_alias(tmp_path: pathlib.Path, capsys) -> None:
    seed = tmp_path / "m.gguf"
    seed.write_bytes(b"GGUF")
    registry = store.Registry()
    store.add_entry(registry, store.Entry(alias="one", path=str(seed)))
    store.save_registry(registry)
    assert cli.main(["models", "rm", "one"]) == 0
    assert "removed" in capsys.readouterr().out
    assert cli.main(["models", "rm", "one"]) == 2
    assert "E_MODEL_NOT_FOUND" in capsys.readouterr().err


def test_models_verify_text_mode_and_unknown_alias(tmp_path: pathlib.Path, capsys) -> None:
    seed = tmp_path / "m.gguf"
    seed.write_bytes(b"GGUF-data")
    registry = store.Registry()
    store.add_entry(registry, store.Entry(alias="one", path=str(seed),
                                          sha256=hashlib.sha256(seed.read_bytes()).hexdigest()))
    store.save_registry(registry)
    assert cli.main(["models", "verify"]) == 0
    assert "ok" in capsys.readouterr().out
    assert cli.main(["models", "verify", "ghost"]) == 2
    assert "E_MODEL_NOT_FOUND" in capsys.readouterr().err


def test_models_pull_json_mode(monkeypatch: pytest.MonkeyPatch, capsys,
                               tmp_path: pathlib.Path) -> None:
    from tests.test_cli_e1a import fake_pull_env, make_gguf

    fake_pull_env(monkeypatch, [{"path": "m-Q8_0.gguf", "size": len(make_gguf()), "oid": None}])
    monkeypatch.setattr(cli.hf, "check_disk_space", lambda *a, **k: 0)
    assert cli.main(["models", "pull", "--file", "m-Q8_0.gguf", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["alias"] == "m-q8_0" and payload["quant"] == "Q8_0"
    assert payload["license"] == "apache-2.0"


def test_models_without_subcommand_help(capsys) -> None:
    assert cli.main(["models", "--help"]) == 0
    assert "usage: typed-gguf models" in capsys.readouterr().out


def test_init_text_mode_with_the_offline_cache(tmp_path: pathlib.Path,
                                               monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from tests.test_runtime_install import build_bundle, fake_lock

    archive = build_bundle(tmp_path, name="llama-b11026-bin-ubuntu-x64.tar.gz")
    monkeypatch.setenv("TYPED_GGUF_LOCK", str(fake_lock(tmp_path, archive)))
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / archive.name).write_bytes(archive.read_bytes())
    monkeypatch.setenv("TYPED_GGUF_OFFLINE_CACHE", str(cache))
    assert cli.main(["init"]) == 0
    out = capsys.readouterr().out
    assert "installed: True" in out and "offline-cache" in out and "build: 11026" in out
