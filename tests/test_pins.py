"""runtime.lock is the single source of truth (SPEC 4): typed access + host mapping."""
from __future__ import annotations

import json
import pathlib

import pytest

from ggufone.errors import GgufoneError
from ggufone.runtime import pins

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVID = ROOT / "docs" / "evidence"


@pytest.fixture()
def lock() -> pins.RuntimeLock:
    return pins.load_lock(ROOT / "runtime.lock")


# ------------------------------------------------------------------ the lock file
def test_lock_reads_the_pinned_release(lock: pins.RuntimeLock) -> None:
    assert lock.tag == "b11026"
    assert lock.published_at == "2026-09-17T13:31:47Z"
    assert lock.commit == "b49650adb"
    assert lock.min_build_for_spark2_5 == 10828
    assert lock.required_files == ("libllama.so", "libggml.so", "libggml-base.so")


def test_lock_required_symbols_are_34(lock: pins.RuntimeLock) -> None:
    assert len(lock.required_symbols_llama) == 32
    assert len(lock.required_symbols_ggml) == 2
    assert pins.REQUIRED_SYMBOL_COUNT == 34
    assert "llama_memory_seq_cp" in lock.required_symbols_llama
    assert "ggml_backend_load_all_from_path" in lock.required_symbols_ggml


def test_lock_assets_match_the_captured_evidence(lock: pins.RuntimeLock) -> None:
    rel = json.loads((EVID / "llama_cpp_release_b11026.json").read_text())
    evidence_sizes = {a["name"]: a["size"] for a in rel["assets"]}
    assert lock.assets, "runtime.lock must define the asset table"
    for variant, asset in lock.assets.items():
        assert asset.size == evidence_sizes[asset.asset], variant
    assert lock.assets["linux-x64-cpu"].size == 16_855_810
    assert lock.assets["linux-x64-vulkan"].size == 30_294_625
    assert lock.assets["macos-arm64-metal"].size == 11_156_751


def test_lock_default_model_matches_hf_evidence(lock: pins.RuntimeLock) -> None:
    hf = json.loads((EVID / "hf_spark_x2_5.json").read_text())
    files = {f["path"]: f for f in hf["files"]}
    dm = lock.default_model
    assert dm.repo == hf["repo"] and dm.repo_sha == hf["repo_sha"]
    assert dm.license == hf["license"] == "apache-2.0"
    assert dm.file == "Spark-X2.5-4B-Q8_0.gguf"
    assert dm.size == files[dm.file]["size"] == 4_375_021_152
    assert dm.sha256 == files[dm.file]["lfs_oid_sha256"]
    assert dm.arch == "spark2_5"
    assert dm.alternates["Q4_K_M"]["size"] == files["Spark-X2.5-4B-Q4_K_M.gguf"]["size"]
    assert dm.alternates["F16"]["sha256"] == files["Spark-X2.5-4B.gguf"]["lfs_oid_sha256"]


def test_lock_mandatory_call_order_documents_the_pitfalls(lock: pins.RuntimeLock) -> None:
    order = " | ".join(lock.mandatory_call_order)
    assert "ggml_backend_load_all_from_path" in order
    assert order.index("ggml_backend_load_all_from_path") < order.index("llama_backend_init")
    assert "kv_unified=True" in order.replace(" ", "")


def test_lock_file_is_found_by_walking_up_from_the_package() -> None:
    assert pins.default_lock_path().name == "runtime.lock"
    assert pins.default_lock_path().exists()


def test_lock_path_override(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    custom = tmp_path / "runtime.lock"
    custom.write_text((ROOT / "runtime.lock").read_text())
    monkeypatch.setenv("GGUFONE_LOCK", str(custom))
    assert pins.default_lock_path() == custom


def test_missing_lock_is_an_actionable_error(tmp_path: pathlib.Path) -> None:
    with pytest.raises(GgufoneError) as exc:
        pins.load_lock(tmp_path / "absent.lock")
    assert exc.value.code == "E_RUNTIME_MISSING"
    assert "absent.lock" in str(exc.value)


def test_corrupt_lock_is_an_actionable_error(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "runtime.lock"
    path.write_text("{not json")
    with pytest.raises(GgufoneError) as exc:
        pins.load_lock(path)
    assert exc.value.code == "E_RUNTIME_MISSING"


# ------------------------------------------------------------------ host mapping
def test_url_for_asset(lock: pins.RuntimeLock) -> None:
    assert pins.url_for(lock, "linux-x64-cpu") == (
        "https://github.com/ggml-org/llama.cpp/releases/download/b11026/"
        "llama-b11026-bin-ubuntu-x64.tar.gz")


@pytest.mark.parametrize(("backend", "system", "machine", "want"), [
    ("cpu", "linux", "x86_64", "linux-x64-cpu"),
    ("vulkan", "linux", "x86_64", "linux-x64-vulkan"),
    ("cuda", "linux", "x86_64", "linux-x64-cuda-12.8"),
    ("auto", "linux", "x86_64", "linux-x64-cpu"),          # no probes -> cpu
    ("cpu", "windows", "AMD64", "windows-x64-cpu"),
    ("vulkan", "windows", "AMD64", "windows-x64-vulkan"),
    ("cuda", "windows", "AMD64", "windows-x64-cuda-12.4"),
    ("auto", "darwin", "arm64", "macos-arm64-metal"),
    ("metal", "darwin", "arm64", "macos-arm64-metal"),
    ("auto", "darwin", "x86_64", "macos-x64-metal"),
])
def test_host_variant_mapping(backend: str, system: str, machine: str, want: str) -> None:
    assert pins.host_variant(backend, system=system, machine=machine) == want


def test_host_variant_rejects_unknown_platform() -> None:
    with pytest.raises(GgufoneError) as exc:
        pins.host_variant("cpu", system="linux", machine="aarch64")
    assert exc.value.code == "E_RUNTIME_MISSING"
    assert "aarch64" in str(exc.value)


def test_host_variant_rejects_backend_without_asset() -> None:
    with pytest.raises(GgufoneError) as exc:
        pins.host_variant("cuda", system="darwin", machine="arm64")
    assert exc.value.code == "E_RUNTIME_MISSING"
    assert "cuda" in str(exc.value)


@pytest.mark.parametrize(("probes", "want"), [
    ({"system": "darwin"}, "metal"),
    ({"system": "linux", "has_nvidia_smi": True}, "cuda"),
    ({"system": "linux", "dri_nodes": ["/dev/dri/renderD128"]}, "vulkan"),
    ({"system": "linux", "dri_nodes": [], "has_nvidia_smi": False}, "cpu"),
    ({"system": "windows", "has_nvidia_smi": False, "dri_nodes": []}, "cpu"),
])
def test_detect_backend(probes: dict[str, object], want: str, tmp_path: pathlib.Path) -> None:
    probes = dict(probes)
    icd = tmp_path / "icd.d"
    if "dri_nodes" in probes and probes["dri_nodes"]:
        icd.mkdir()
    probes.setdefault("icd_dir", str(icd))
    assert pins.detect_backend(**probes) == want  # type: ignore[arg-type]


def test_asset_for_unknown_variant_is_an_error(lock: pins.RuntimeLock) -> None:
    with pytest.raises(GgufoneError) as exc:
        pins.asset_for(lock, "linux-aarch64-cpu")
    assert exc.value.code == "E_RUNTIME_MISSING"
