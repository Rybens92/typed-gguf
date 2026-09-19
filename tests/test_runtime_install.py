"""`ggufone init`: prebuilt ladder rung 1 only — never a compiler (SPEC 4, A-E1a-2).

The offline tests use a synthetic lock + a synthetic bundle; the real 16.8 MB pinned asset is
downloaded in the live/network test and in the evidence run.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import pathlib
import tarfile
import time

import pytest

from ggufone.errors import GgufoneError
from ggufone.runtime import install, pins

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ helpers
def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_bundle(tmp_path: pathlib.Path, *, name: str = "bundle.tar.gz",
                 build: int = 11026) -> pathlib.Path:
    """A tiny stand-in for llama-b11026-bin-ubuntu-x64.tar.gz (same layout, fake libs)."""
    staging = tmp_path / f"staging-{name}"
    inner = staging / "llama-b11026"
    inner.mkdir(parents=True)
    for lib in ("libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so"):
        (inner / lib).write_bytes(b"\x7fELF fake\nllama_model_spark2_5\x00")
    cli = inner / "llama-cli"
    cli.write_text(f"#!/bin/sh\necho 'version: 0.4.1-dev (build {build}, commit b49650adb)' >&2\n")
    fit = inner / "llama-fit-params"
    fit.write_text("#!/bin/sh\nexit 0\n")
    for exe in (cli, fit):
        exe.chmod(0o755)
    archive = tmp_path / name
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(inner, arcname="llama-b11026")
    return archive


def fake_lock(tmp_path: pathlib.Path, archive: pathlib.Path, *,
              size: int | None = None, digest: str | None = None,
              variant: str = "linux-x64-cpu") -> pathlib.Path:
    payload = {
        "schema": "ggufone.runtime.lock/v1",
        "llama_cpp": {
            "repo": "ggml-org/llama.cpp", "tag": "b11026",
            "published_at": "2026-09-17T13:31:47Z", "commit": "b49650adb",
            "min_build_for_spark2_5": 10828,
            "assets": {variant: {"asset": archive.name,
                                 "size": size if size is not None else archive.stat().st_size,
                                 "sha256": digest if digest is not None else sha256(archive)}},
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
    path = tmp_path / "fake-runtime.lock"
    path.write_text(json.dumps(payload))
    return path


def locked(tmp_path: pathlib.Path, *, build: int = 11026) -> tuple[pins.RuntimeLock, pathlib.Path]:
    archive = build_bundle(tmp_path, build=build)
    return pins.load_lock(fake_lock(tmp_path, archive)), archive


# ------------------------------------------------------------------ planning
def test_plan_install_picks_the_pinned_asset(tmp_path: pathlib.Path) -> None:
    lock, archive = locked(tmp_path)
    plan = install.plan_install("cpu", home=tmp_path / "home", lock=lock)
    assert plan.variant == "linux-x64-cpu"
    assert plan.asset == archive.name
    assert plan.size == archive.stat().st_size
    assert plan.sha256 == sha256(archive)
    assert plan.url == f"https://example.invalid/releases/download/b11026/{archive.name}"
    assert plan.dest == tmp_path / "home" / "runtime" / "b11026-linux-x64-cpu"
    assert plan.required_bytes >= plan.size


def test_plan_install_prefers_a_cached_bundle(tmp_path: pathlib.Path) -> None:
    lock, archive = locked(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / archive.name).write_bytes(archive.read_bytes())
    plan = install.plan_install("cpu", home=tmp_path / "home", lock=lock, offline_cache=cache)
    assert plan.cached == cache / archive.name


def test_dry_run_touches_nothing(tmp_path: pathlib.Path) -> None:
    lock, _ = locked(tmp_path)
    home = tmp_path / "home"
    result = install.install("cpu", home=home, lock=lock, dry_run=True)
    assert result["dry_run"] is True
    assert result["plan"]["variant"] == "linux-x64-cpu"
    assert not home.exists()


# ------------------------------------------------------------------ install
@pytest.mark.needs_fork
def test_install_from_the_offline_cache(tmp_path: pathlib.Path) -> None:
    lock, archive = locked(tmp_path)
    home = tmp_path / "home"
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / archive.name).write_bytes(archive.read_bytes())
    result = install.install("cpu", home=home, lock=lock, offline_cache=cache,
                            free_bytes=1 << 40)
    dest = home / "runtime" / "b11026-linux-x64-cpu"
    assert result["variant"] == "linux-x64-cpu"
    assert result["source"] == "offline-cache"
    assert (dest / "libllama.so").exists()
    record = json.loads((home / "runtime.json").read_text())
    assert record["tag"] == "b11026"
    assert record["asset"] == archive.name
    assert record["asset_sha256"] == sha256(archive)
    assert record["libllama_sha256"] == sha256(dest / "libllama.so")
    assert record["build"] == 11026
    assert record["backends"] == ["cpu"]
    assert record["tools"]["llama-fit-params"].endswith("llama-fit-params")
    assert record["warmup_ms"] is None
    assert result["dir"] == str(dest)


def test_install_is_idempotent_without_force(tmp_path: pathlib.Path) -> None:
    lock, archive = locked(tmp_path)
    home = tmp_path / "home"
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / archive.name).write_bytes(archive.read_bytes())
    first = install.install("cpu", home=home, lock=lock, offline_cache=cache, free_bytes=1 << 40)
    marker = home / "runtime" / "b11026-linux-x64-cpu" / "libllama.so"
    stamp = marker.stat().st_mtime_ns
    time.sleep(0.01)
    second = install.install("cpu", home=home, lock=lock, offline_cache=cache, free_bytes=1 << 40)
    assert second["already_installed"] is True
    assert marker.stat().st_mtime_ns == stamp
    assert first["dir"] == second["dir"]


def test_install_force_reinstalls(tmp_path: pathlib.Path) -> None:
    lock, archive = locked(tmp_path)
    home = tmp_path / "home"
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / archive.name).write_bytes(archive.read_bytes())
    install.install("cpu", home=home, lock=lock, offline_cache=cache, free_bytes=1 << 40)
    again = install.install("cpu", home=home, lock=lock, offline_cache=cache, force=True,
                            free_bytes=1 << 40)
    assert again.get("already_installed") is None
    assert again["variant"] == "linux-x64-cpu"


def test_install_rejects_a_cache_file_with_the_wrong_hash(tmp_path: pathlib.Path) -> None:
    lock, archive = locked(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    # right size, wrong content -> the SHA-256 check is what must fire
    (cache / archive.name).write_bytes(b"\x00" * archive.stat().st_size)
    with pytest.raises(GgufoneError) as exc:
        install.install("cpu", home=tmp_path / "home", lock=lock, offline_cache=cache,
                        free_bytes=1 << 40)
    assert exc.value.code == "E_SHA256_MISMATCH"


def test_install_rejects_a_cache_file_with_the_wrong_size(tmp_path: pathlib.Path) -> None:
    lock, archive = locked(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / archive.name).write_bytes(archive.read_bytes() + b"junk")
    with pytest.raises(GgufoneError) as exc:
        install.install("cpu", home=tmp_path / "home", lock=lock, offline_cache=cache,
                        free_bytes=1 << 40)
    assert exc.value.code in ("E_SHA256_MISMATCH", "E_DOWNLOAD_FAILED")


def test_install_fails_before_downloading_when_disk_is_short(tmp_path: pathlib.Path) -> None:
    lock, archive = locked(tmp_path)
    with pytest.raises(GgufoneError) as exc:
        install.install("cpu", home=tmp_path / "home", lock=lock, free_bytes=1024)
    assert exc.value.code == "E_INSUFFICIENT_DISK"
    msg = str(exc.value)
    assert str(archive.stat().st_size * 3) in msg  # required bytes
    assert "1024" in msg                           # free bytes


def test_install_fails_when_the_bundle_lacks_a_required_file(tmp_path: pathlib.Path) -> None:
    inner = tmp_path / "partial" / "llama-b11026"
    inner.mkdir(parents=True)
    (inner / "libllama.so").write_bytes(b"\x7fELF fake\n")
    archive = tmp_path / "partial.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(inner, arcname="llama-b11026")
    lock_dir = tmp_path / "lockdir"
    lock_dir.mkdir()
    lock = pins.load_lock(fake_lock(lock_dir, archive))
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / archive.name).write_bytes(archive.read_bytes())
    with pytest.raises(GgufoneError) as exc:
        install.install("cpu", home=tmp_path / "home", lock=lock, offline_cache=cache,
                        free_bytes=1 << 40)
    assert exc.value.code == "E_RUNTIME_MISSING"
    assert "libggml.so" in str(exc.value)


def test_install_from_a_file_url(tmp_path: pathlib.Path) -> None:
    lock, archive = locked(tmp_path)
    home = tmp_path / "home"
    result = install.install("cpu", home=home, lock=lock, url=archive.as_uri(),
                             free_bytes=1 << 40)
    assert result["source"] == "url"
    assert (home / "runtime" / "b11026-linux-x64-cpu" / "libllama.so").exists()
    assert result["bytes_fetched"] == archive.stat().st_size


def test_install_resumes_a_partial_download(tmp_path: pathlib.Path) -> None:
    lock, archive = locked(tmp_path)
    home = tmp_path / "home"
    downloads = home / "downloads"
    downloads.mkdir(parents=True)
    part = downloads / (archive.name + ".part")
    blob = archive.read_bytes()
    part.write_bytes(blob[: len(blob) // 2])  # simulate a killed run
    result = install.install("cpu", home=home, lock=lock, url=archive.as_uri(),
                             free_bytes=1 << 40)
    assert result["resumed_from"] == len(blob) // 2
    assert result["bytes_fetched"] == len(blob) - len(blob) // 2
    assert result["dir"].endswith("b11026-linux-x64-cpu")


def test_extract_uses_a_safe_tar_filter(tmp_path: pathlib.Path) -> None:
    evil = tmp_path / "evil.tar.gz"
    with tarfile.open(evil, "w:gz") as tar:
        info = tarfile.TarInfo("../../escape.txt")
        payload = b"nope"
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    with pytest.raises(GgufoneError) as exc:
        install.extract_bundle(evil, tmp_path / "out")
    assert exc.value.code == "E_DOWNLOAD_FAILED"
    assert not (tmp_path / "escape.txt").exists()


# ------------------------------------------------------------------ A1: no compiler
COMPILER_TOKENS = ("cmake", "ninja", "gcc", "g++", "clang", "nvcc", "make -j")


def docstring_ids(tree: ast.AST) -> set[int]:
    """id() of every docstring constant node (those are prose, not code)."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", [])
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            ids.add(id(body[0].value))
    return ids


def test_no_source_file_ever_names_a_compiler_toolchain() -> None:
    """A1 / SPEC 4 rung 1: `ggufone init` must not shell out to a build toolchain.

    Enforced structurally: outside docstrings, no string constant in `src/ggufone/**` mentions
    a compiler name, so no subprocess can be constructed from one. Docstrings are allowed to
    *talk* about the rule.
    """
    offenders: list[str] = []
    for path in sorted((ROOT / "src" / "ggufone").rglob("*.py")):
        tree = ast.parse(path.read_text())
        prose = docstring_ids(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and id(node) not in prose:
                lowered = node.value.lower()
                for token in COMPILER_TOKENS:
                    if token in lowered:
                        offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} -> {token}")
    assert offenders == []


def test_init_dry_run_needs_no_compiler_on_path(tmp_path: pathlib.Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """A-E1a-2's poisoned-PATH half, offline: an empty PATH must still plan cleanly."""
    lock, _ = locked(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    plan = install.plan_install("cpu", home=tmp_path / "home", lock=lock)
    assert plan.variant == "linux-x64-cpu"
