"""E1a CLI surface: init / doctor / models (SPEC 2.8, A-E1a-2..7).

Offline: the HF seam is faked, the runtime is synthetic, and `GGUFONE_DEEP_PROBE=0` keeps the
symbol check out of the process (the live evidence run does the deep probe).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import stat
import struct
import tarfile

import pytest

# import the CLI module (not the package) so monkeypatching its deps hits the same objects
from ggufone import __version__, cli
from ggufone.errors import InsufficientDiskError
from ggufone.registry import hf, store
from ggufone.registry.hf import DownloadResult
from ggufone.runtime import pins


def cpu_only_host() -> pins.HostProbes:
    """A deterministic GPU-less machine (x86_64 Linux, no nvidia-smi, no DRM node)."""
    return pins.fake_host(system="linux", machine="x86_64")


def gpu_host() -> pins.HostProbes:
    """The operator's box as a probe object: nvidia-smi present -> the pinned CUDA bundle."""
    return pins.fake_host(system="linux", machine="x86_64", has_nvidia_smi=True)


# ------------------------------------------------------------------ fixtures
@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    home = tmp_path / "home"
    monkeypatch.setenv("GGUFONE_HOME", str(home))
    monkeypatch.delenv("GGUFONE_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("GGUFONE_LOCK", raising=False)
    monkeypatch.delenv("GGUFONE_OFFLINE", raising=False)
    monkeypatch.delenv("GGUFONE_OFFLINE_CACHE", raising=False)
    monkeypatch.setenv("GGUFONE_DEEP_PROBE", "0")
    # E1a FIX t_eae35404: this file tests the *plan* (which pinned asset a variant maps to),
    # not the box it runs on — so it runs in a deterministic fake CPU machine. Without this,
    # the assertions below would flip to linux-x64-cuda-12.8 on the operator's GPU host.
    # The real-host path is covered by tests/test_pins.py
    # (test_the_whole_mapping_in_a_fake_host_world: cpu/vulkan/cuda worlds) and by the live
    # host gate.
    monkeypatch.setattr(pins, "current_host", cpu_only_host)
    return tmp_path


def make_runtime(base: pathlib.Path, *, build: int = 11026,
                 archs: tuple[str, ...] = ("spark2_5",),
                 libs: tuple[str, ...] = ("libllama.so", "libggml.so", "libggml-base.so"),
                 ) -> pathlib.Path:
    rt = base / "runtime-bundle"
    rt.mkdir(parents=True, exist_ok=True)
    for lib in libs:
        (rt / lib).write_bytes(b"\x7fELF fake\n" + b"".join(
            b"llama_model_" + a.encode() + b"\x00" for a in archs))
    (rt / "libggml-cpu.so").write_bytes(b"\x7fELF fake\n")
    cli_path = rt / "llama-cli"
    cli_path.write_text(f"#!/bin/sh\necho 'version: 0.4.1-dev (build {build}, commit x)' >&2\n")
    fit = rt / "llama-fit-params"
    fit.write_text("#!/bin/sh\nexit 0\n")
    for exe in (cli_path, fit):
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return rt


def make_gguf(arch: str = "spark2_5", file_type: int = 7) -> bytes:
    def gstr(value: str) -> bytes:
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw
    kvs = [gstr("general.architecture") + struct.pack("<I", 8) + gstr(arch),
           gstr("general.file_type") + struct.pack("<I", 4) + struct.pack("<I", file_type)]
    return (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1) + struct.pack("<Q", len(kvs))
            + b"".join(kvs))


def seed_registry(tmp_path: pathlib.Path, *, arch: str = "spark2_5",
                  with_sha: bool = True) -> store.Entry:
    model = tmp_path / "fake-model.gguf"
    model.write_bytes(make_gguf(arch))
    entry = store.Entry(alias="fake", path=str(model),
                        sha256=hashlib.sha256(model.read_bytes()).hexdigest() if with_sha else None,
                        arch=arch, quant="Q8_0", size=model.stat().st_size, license="apache-2.0",
                        source="acme/fake", added_at="2026-09-17T00:00:00Z", file_type=7,
                        repo="acme/fake")
    registry = store.Registry()
    store.add_entry(registry, entry)
    store.save_registry(registry)
    return entry


# ------------------------------------------------------------------ init
def test_init_dry_run_json_prints_the_plan_and_writes_nothing(capsys) -> None:
    code = cli.main(["init", "--dry-run", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["variant"] == "linux-x64-cpu"
    assert payload["asset"] == "llama-b11026-bin-ubuntu-x64.tar.gz"
    assert payload["size"] == 16_855_810
    assert payload["sha256"] == "219cf1c726bae1da4289b96a6378314d5485c6bc74c43891a4203e30906afb06"
    assert payload["url"].startswith("https://github.com/ggml-org/llama.cpp/releases/download/")
    assert payload["destination"].endswith("b11026-linux-x64-cpu")
    assert not pathlib.Path(payload["destination"]).exists()


def test_init_dry_run_text_says_so(capsys) -> None:
    assert cli.main(["init", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "llama-b11026-bin-ubuntu-x64.tar.gz" in out and "dry run" in out


def test_init_dry_run_on_a_gpu_host_plans_the_pinned_cuda_bundle(monkeypatch: pytest.MonkeyPatch,
                                                                 capsys) -> None:
    """The GPU half of the fake-host regression (operator's RTX box, E1a FIX t_eae35404)."""
    monkeypatch.setattr(pins, "current_host", gpu_host)
    lock = pins.load_lock()
    assert cli.main(["init", "--dry-run", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    spec = lock.assets["linux-x64-cuda-12.8"]
    assert payload["backend"] == "cuda"
    assert payload["variant"] == "linux-x64-cuda-12.8"
    assert payload["asset"] == "llama-b11026-bin-ubuntu-cuda-12.8-x64.tar.gz"
    assert payload["size"] == spec.size == 168_811_114
    assert payload["sha256"] == spec.sha256
    assert payload["host"]["has_nvidia_smi"] is True
    assert payload["host"]["backend"] == "cuda"
    assert not pathlib.Path(payload["destination"]).exists()


def test_init_backend_without_an_asset_is_a_user_error(capsys) -> None:
    assert cli.main(["init", "--dry-run", "--backend", "linux-aarch64-cpu"]) == 3
    err = capsys.readouterr().err
    assert "E_RUNTIME_MISSING" in err


def test_init_from_offline_cache_with_a_poisoned_path(tmp_path: pathlib.Path,
                                                      monkeypatch: pytest.MonkeyPatch,
                                                      capsys) -> None:
    """A-E1a-2 offline half: no compiler anywhere on PATH, bundle served from cache."""
    from tests.test_runtime_install import build_bundle, fake_lock  # reuse the synthetic bundle

    archive = build_bundle(tmp_path, name="llama-b11026-bin-ubuntu-x64.tar.gz")
    lock_path = fake_lock(tmp_path, archive)
    monkeypatch.setenv("GGUFONE_LOCK", str(lock_path))
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / archive.name).write_bytes(archive.read_bytes())
    monkeypatch.setenv("GGUFONE_OFFLINE_CACHE", str(cache))
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    monkeypatch.setenv("PATH", str(empty_path))

    assert cli.main(["init", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["installed"] is True
    assert payload["source"] == "offline-cache"
    assert payload["variant"] == "linux-x64-cpu"
    assert pathlib.Path(payload["dir"], "libllama.so").exists()
    record = json.loads((pathlib.Path(payload["dir"]).parents[1] / "runtime.json").read_text())
    assert record["build"] == 11026
    assert record["backends"] == ["cpu"]
    assert record["libllama_sha256"]


def test_init_twice_is_idempotent(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
                                  capsys) -> None:
    from tests.test_runtime_install import build_bundle, fake_lock

    archive = build_bundle(tmp_path, name="llama-b11026-bin-ubuntu-x64.tar.gz")
    monkeypatch.setenv("GGUFONE_LOCK", str(fake_lock(tmp_path, archive)))
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / archive.name).write_bytes(archive.read_bytes())
    monkeypatch.setenv("GGUFONE_OFFLINE_CACHE", str(cache))
    assert cli.main(["init", "--json"]) == 0
    capsys.readouterr()
    assert cli.main(["init", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["already_installed"] is True


# ------------------------------------------------------------------ doctor
def test_doctor_without_a_runtime_fails_and_is_stable_json(capsys) -> None:
    code = cli.main(["doctor", "--json"])
    assert code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "ggufone.doctor/v1"
    assert report["status"] == "failures"
    assert report["exit_code"] == 1
    assert report["runtime"]["installed"] is False
    assert report["model"]["alias"] is None
    assert any(c["id"] == "runtime.present" and c["status"] == "fail" for c in report["checks"])
    assert report["runtime"]["pinned_tag"] == "b11026"
    assert report["runtime"]["min_build"] == 10828


def test_doctor_warns_with_a_shallow_fake_runtime(tmp_path: pathlib.Path,
                                                  monkeypatch: pytest.MonkeyPatch,
                                                  capsys) -> None:
    rt = make_runtime(tmp_path)
    monkeypatch.setenv("GGUFONE_RUNTIME_DIR", str(rt))
    code = cli.main(["doctor", "--json"])
    assert code == 2  # warnings: no model pulled, symbols not probed, sha not recorded
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "warnings"
    checks = {c["id"]: c for c in report["checks"]}
    assert checks["runtime.present"]["status"] == "ok"
    assert checks["runtime.files"]["status"] == "ok"
    assert checks["runtime.build"]["status"] == "ok"
    assert checks["runtime.symbols"]["status"] == "warn"
    assert checks["runtime.fit_params"]["status"] == "ok"
    assert checks["model.present"]["status"] == "warn"
    assert report["runtime"]["build"] == 11026
    assert report["runtime"]["backends"] == ["cpu"]


def test_doctor_fails_when_a_required_library_is_missing(tmp_path: pathlib.Path,
                                                         monkeypatch: pytest.MonkeyPatch,
                                                         capsys) -> None:
    rt = make_runtime(tmp_path, libs=("libllama.so", "libggml.so"))
    monkeypatch.setenv("GGUFONE_RUNTIME_DIR", str(rt))
    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert any(c["id"] == "runtime.files" and c["status"] == "fail" for c in report["checks"])


def test_doctor_fails_when_the_runtime_lacks_the_model_arch(tmp_path: pathlib.Path,
                                                            monkeypatch: pytest.MonkeyPatch,
                                                            capsys) -> None:
    rt = make_runtime(tmp_path, archs=("qwen35",))
    monkeypatch.setenv("GGUFONE_RUNTIME_DIR", str(rt))
    seed_registry(tmp_path)
    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    arch_check = next(c for c in report["checks"] if c["id"] == "model.arch")
    assert arch_check["status"] == "fail"
    assert "E_MODEL_ARCH_UNSUPPORTED" in arch_check["detail"]
    assert "spark2_5" in arch_check["detail"]


def test_doctor_text_mode_is_readable(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
                                      capsys) -> None:
    monkeypatch.setenv("GGUFONE_RUNTIME_DIR", str(make_runtime(tmp_path)))
    code = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert code == 2
    assert "ggufone doctor" in out and "runtime.build" in out


# ------------------------------------------------------------------ models ls / use / rm
def test_models_ls_empty(capsys) -> None:
    assert cli.main(["models", "ls"]) == 0
    assert "no models" in capsys.readouterr().out


def test_models_ls_json_includes_license(tmp_path: pathlib.Path, capsys) -> None:
    seed_registry(tmp_path)
    assert cli.main(["models", "ls", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "ggufone.models/v1"
    assert payload["current"] == "fake"
    model = payload["models"][0]
    assert model["license"] == "apache-2.0"
    assert set(model) >= {"alias", "path", "size", "sha256", "arch", "quant", "license"}
    assert model["current"] is True


def test_models_use_and_unknown_alias(tmp_path: pathlib.Path, capsys) -> None:
    seed_registry(tmp_path)
    assert cli.main(["models", "use", "fake"]) == 0
    capsys.readouterr()
    assert cli.main(["models", "use", "nope"]) == 2
    assert "E_MODEL_NOT_FOUND" in capsys.readouterr().err


def test_models_rm_deletes_the_file_and_the_entry(tmp_path: pathlib.Path, capsys) -> None:
    entry = seed_registry(tmp_path)
    assert cli.main(["models", "rm", "fake", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["removed"] == "fake" and payload["file_deleted"] is True
    assert not pathlib.Path(entry.path).exists()
    assert store.load_registry()[0].aliases == {}


def test_models_rm_keep_file(tmp_path: pathlib.Path, capsys) -> None:
    entry = seed_registry(tmp_path)
    assert cli.main(["models", "rm", "fake", "--keep-file"]) == 0
    assert pathlib.Path(entry.path).exists()
    capsys.readouterr()


def test_models_verify_ok(tmp_path: pathlib.Path, capsys) -> None:
    seed_registry(tmp_path)
    assert cli.main(["models", "verify", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verified"] == 1 and payload["failed"] == 0


def test_models_verify_detects_a_tampered_file(tmp_path: pathlib.Path, capsys) -> None:
    entry = seed_registry(tmp_path)
    pathlib.Path(entry.path).write_bytes(b"tampered")
    assert cli.main(["models", "verify"]) == 3
    assert "E_SHA256_MISMATCH" in capsys.readouterr().err


def test_models_verify_reports_a_missing_file(tmp_path: pathlib.Path, capsys) -> None:
    entry = seed_registry(tmp_path)
    pathlib.Path(entry.path).unlink()
    assert cli.main(["models", "verify", "--json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0]["status"] == "missing"


# ------------------------------------------------------------------ recommend-quant
def test_recommend_quant_table_reproduces_the_executed_values(capsys) -> None:
    assert cli.main(["models", "recommend-quant", "--table", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    rows = payload["scenarios"]
    assert len(rows) == 3
    first, second, third = rows
    assert first["quant"] == "Spark-X2.5-4B-Q8_0.gguf" and first["kv_type"] == "q8_0"
    assert first["placement"] == "gpu" and first["total_gb"] == 7.33
    assert second["quant"] == "Spark-X2.5-4B-Q8_0.gguf" and second["kv_type"] == "f16"
    assert second["total_gb"] == 6.12
    assert third["placement"] == "insufficient" and third["quant"] is None


def test_recommend_quant_uses_the_host_budget(capsys) -> None:
    assert cli.main(["models", "recommend-quant", "--vram", "8", "--ram", "31", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "host"
    assert payload["quant"] == "Spark-X2.5-4B-Q8_0.gguf"
    assert payload["placement"] in ("gpu", "cpu")


# ------------------------------------------------------------------ pull
def fake_pull_env(monkeypatch: pytest.MonkeyPatch, files: list[dict], *,
                  license: str = "apache-2.0", payload: bytes | None = None) -> dict:
    blob = payload if payload is not None else make_gguf()
    repo_sha = "902d865994943ab9235670e24f01846ee06091f2"
    info = hf.ModelInfo(repo="XHToken/Spark-X2.5-4B-GGUF", sha=repo_sha,
                        gated=False, license=license,
                        files=tuple(hf.FileInfo(path=f["path"], size=f.get("size"),
                                                oid=f.get("oid")) for f in files),
                        tags=("gguf",), source="api")
    monkeypatch.setattr(cli.hf, "model_info", lambda *a, **k: info)
    calls: dict = {}

    def fake_download(repo, path, dest, **kwargs):
        calls.update({"repo": repo, "path": path, "dest": pathlib.Path(dest), **kwargs})
        pathlib.Path(dest).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(dest).write_bytes(blob)
        return DownloadResult(path=pathlib.Path(dest), bytes_fetched=len(blob),
                              bytes_total=len(blob), resumed_from=0,
                              sha256=hashlib.sha256(blob).hexdigest(),
                              verified=bool(kwargs.get("sha256")),
                              url=hf.resolve_url(repo, path, kwargs.get("revision") or "main"))

    monkeypatch.setattr(cli.hf, "download_file", fake_download)
    return calls


def test_models_pull_writes_the_registry_entry_with_license(monkeypatch: pytest.MonkeyPatch,
                                                            capsys, tmp_path: pathlib.Path) -> None:
    calls = fake_pull_env(monkeypatch, [
        {"path": "Spark-X2.5-4B-Q8_0.gguf", "size": len(make_gguf()), "oid": "a" * 64},
        {"path": "Spark-X2.5-4B-Q4_K_M.gguf", "size": 26, "oid": "b" * 64},
    ])
    assert cli.main(["models", "pull"]) == 0
    out = capsys.readouterr().out
    assert "spark-x2.5-4b-q8_0" in out and "apache-2.0" in out
    assert calls["path"] == "Spark-X2.5-4B-Q8_0.gguf"  # exactly one file
    assert calls["revision"] == "902d865994943ab9235670e24f01846ee06091f2"
    registry, _ = store.load_registry()
    entry = registry.aliases["spark-x2.5-4b-q8_0"]
    assert entry.license == "apache-2.0"
    assert entry.arch == "spark2_5" and entry.quant == "Q8_0"
    assert entry.size == len(make_gguf()) and entry.repo == "XHToken/Spark-X2.5-4B-GGUF"
    capsys.readouterr()
    assert cli.main(["models", "ls", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["models"][0]["license"] == "apache-2.0"


def test_models_pull_ambiguous_quant_lists_the_candidates(monkeypatch: pytest.MonkeyPatch,
                                                          capsys) -> None:
    fake_pull_env(monkeypatch, [
        {"path": "Spark-X2.5-4B-Q8_0.gguf", "size": 10, "oid": "a" * 64},
        {"path": "Spark-X2.5-4B-Q8_0-imatrix.gguf", "size": 11, "oid": "b" * 64},
    ])
    assert cli.main(["models", "pull", "XHToken/Spark-X2.5-4B-GGUF:Q8_0"]) == 2
    err = capsys.readouterr().err
    assert "E_AMBIGUOUS_QUANT" in err
    assert "Spark-X2.5-4B-Q8_0.gguf" in err and "imatrix" in err


def test_models_pull_unknown_quant_is_actionable(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    fake_pull_env(monkeypatch, [{"path": "Spark-X2.5-4B-Q8_0.gguf", "size": 10, "oid": "a" * 64}])
    assert cli.main(["models", "pull", "XHToken/Spark-X2.5-4B-GGUF:Q3_K_XL"]) == 2
    err = capsys.readouterr().err
    assert "E_MODEL_NOT_FOUND" in err and "Q3_K_XL" in err and "Q8_0" in err


def test_models_pull_refuses_a_download_that_cannot_fit(monkeypatch: pytest.MonkeyPatch,
                                                        capsys) -> None:
    fake_pull_env(monkeypatch, [{"path": "Spark-X2.5-4B-Q8_0.gguf", "size": 10, "oid": "a" * 64}])

    def too_small(*args, **kwargs):
        raise InsufficientDiskError(
            "E_INSUFFICIENT_DISK: need 4375021152 bytes (4.38 GB) free, have 1200000000 "
            "bytes (1.20 GB)")

    monkeypatch.setattr(cli.hf, "check_disk_space", too_small)
    assert cli.main(["models", "pull"]) == 2
    err = capsys.readouterr().err
    assert "E_INSUFFICIENT_DISK" in err and "4.38 GB" in err and "1.20 GB" in err


def test_models_pull_no_verify_flag_is_forwarded(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    calls = fake_pull_env(monkeypatch, [{"path": "Spark-X2.5-4B-Q8_0.gguf", "size": 10,
                                         "oid": "a" * 64}])
    assert cli.main(["models", "pull", "--no-verify"]) == 0
    assert calls["no_verify"] is True
    capsys.readouterr()


def test_models_pull_offline_uses_the_snapshot(monkeypatch: pytest.MonkeyPatch, capsys,
                                               tmp_path: pathlib.Path) -> None:
    monkeypatch.setenv("GGUFONE_OFFLINE", "1")
    blob = make_gguf()
    # pytest's tmp_path lives on a small tmpfs: the real precheck would (correctly) refuse the
    # 4.38 GB model. The precheck itself is covered in tests/test_hf.py.
    monkeypatch.setattr(cli.hf, "check_disk_space", lambda *a, **k: 0)

    def fake_download(repo, path, dest, **kwargs):
        pathlib.Path(dest).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(dest).write_bytes(blob)
        return DownloadResult(path=pathlib.Path(dest), bytes_fetched=len(blob),
                              bytes_total=len(blob), resumed_from=0,
                              sha256=hashlib.sha256(blob).hexdigest(), verified=True,
                              url="offline://snapshot")

    monkeypatch.setattr(cli.hf, "download_file", fake_download)
    assert cli.main(["models", "pull"]) == 0
    assert "spark-x2.5-4b-q8_0" in capsys.readouterr().out


# ------------------------------------------------------------------ search + misc
def test_models_search_json(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(cli.hf, "search", lambda query, limit=20: [
        {"id": "acme/gguf", "downloads": 3, "likes": 1, "tags": ["gguf"]}])
    assert cli.main(["models", "search", "acme", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0]["id"] == "acme/gguf"


def test_models_unknown_subcommand(capsys) -> None:
    assert cli.main(["models", "nope"]) == 2
    assert "unknown models subcommand" in capsys.readouterr().err


def test_models_without_subcommand_prints_usage(capsys) -> None:
    assert cli.main(["models"]) == 0
    assert "recommend-quant" in capsys.readouterr().out


def test_version_text_and_json(capsys) -> None:
    assert cli.main(["version"]) == 0
    assert __version__ in capsys.readouterr().out
    assert cli.main(["version", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["version"] == __version__ and payload["lock"]["tag"] == "b11026"


@pytest.mark.parametrize("cmd", ["run", "ask", "serve", "mcp", "bench", "fit", "calibrate"])
def test_frozen_commands_still_exit_3(cmd: str, capsys) -> None:
    assert cli.main([cmd]) == 3
    assert "not implemented yet" in capsys.readouterr().err


def test_unknown_command_exits_2(capsys) -> None:
    assert cli.main(["nope"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_unknown_option_is_a_user_error(capsys) -> None:
    assert cli.main(["doctor", "--nope"]) == 2
    assert "E_UNKNOWN_KEY" in capsys.readouterr().err


def test_internal_errors_never_print_a_traceback(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(cli, "doctor_checks", boom)
    assert cli.main(["doctor"]) == 4
    err = capsys.readouterr().err
    assert "E_INTERNAL" in err and "Traceback" not in err


def test_pull_requires_no_compiler_toolchain(tmp_path: pathlib.Path,
                                             monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """The whole pull path must be stdlib-only: no build tool is ever shelled out to."""
    fake_pull_env(monkeypatch, [{"path": "m-Q8_0.gguf", "size": len(make_gguf()), "oid": None}])
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    assert cli.main(["models", "pull", "--file", "m-Q8_0.gguf"]) == 0
    capsys.readouterr()


def test_bundle_asset_is_resolved_from_the_lock_for_the_host() -> None:
    lock = cli.pins.load_lock()
    assert cli.install.plan_install("cpu", lock=lock).asset in {
        spec.asset for spec in lock.assets.values()}


def test_tarball_of_the_synthetic_lock_is_usable(tmp_path: pathlib.Path) -> None:
    from tests.test_runtime_install import build_bundle

    archive = build_bundle(tmp_path, name="llama-b11026-bin-ubuntu-x64.tar.gz")
    with tarfile.open(archive) as tar:
        assert any(name.endswith("libllama.so") for name in tar.getnames())
