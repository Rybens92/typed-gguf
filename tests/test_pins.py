"""runtime.lock is the single source of truth (SPEC 4): typed access + host mapping."""
from __future__ import annotations

import json
import pathlib

import pytest

from ggufone.errors import GgufoneError
from ggufone.runtime import install, pins

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


# ------------------------------------------------------------------ probe purity
def install_fake_machine(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, *,
                         world: str) -> None:
    """Patch every real-host source so `current_host()` reports a synthetic machine.

    world=cpu: no nvidia-smi, no DRM render node, no Vulkan ICD
    world=vulkan: no nvidia-smi, a DRM render node + a Vulkan ICD
    world=cuda: nvidia-smi on PATH (as on the operator's RTX box)

    The regression this guards (E1a FIX t_eae35404): the suite must be green on a CPU-only
    box *and* on a GPU box. The fake machine is installed at the OS level — shutil.which,
    platform.system/machine, /dev/dri, the Vulkan ICD dir — so the *production* path
    (`detect_backend()` with no arguments) is what gets exercised.
    """
    assert world in ("cpu", "vulkan", "cuda")
    monkeypatch.setattr(
        pins.shutil, "which",
        lambda name: "/usr/bin/nvidia-smi" if (world == "cuda" and name == "nvidia-smi") else None)
    monkeypatch.setattr(pins.platform, "system", lambda: "linux")
    monkeypatch.setattr(pins.platform, "machine", lambda: "x86_64")
    dri = tmp_path / "dev-dri"
    dri.mkdir()
    if world == "vulkan":
        (dri / "renderD128").write_bytes(b"")
    monkeypatch.setattr(pins, "DRI_DIR", dri)
    icd = tmp_path / "vulkan-icd.d"
    icd.mkdir()
    monkeypatch.setattr(pins, "ICD_DIR", icd if world == "vulkan" else tmp_path / "no-icd.d")


@pytest.mark.parametrize(("world", "backend", "variant", "asset", "size"), [
    ("cpu", "cpu", "linux-x64-cpu", "llama-b11026-bin-ubuntu-x64.tar.gz", 16_855_810),
    ("vulkan", "vulkan", "linux-x64-vulkan", "llama-b11026-bin-ubuntu-vulkan-x64.tar.gz",
     30_294_625),
    ("cuda", "cuda", "linux-x64-cuda-12.8", "llama-b11026-bin-ubuntu-cuda-12.8-x64.tar.gz",
     168_811_114),
])
def test_the_whole_mapping_in_a_fake_host_world(monkeypatch: pytest.MonkeyPatch,
                                                tmp_path: pathlib.Path, world: str, backend: str,
                                                variant: str, asset: str, size: int) -> None:
    """detection -> variant -> pinned asset -> install plan, in both GPU-absent and GPU worlds."""
    install_fake_machine(monkeypatch, tmp_path, world=world)
    lock = pins.load_lock(ROOT / "runtime.lock")
    assert pins.detect_backend() == backend
    assert pins.host_variant("auto") == variant
    assert pins.asset_for(lock, variant).asset == asset
    assert pins.asset_for(lock, variant).size == size
    plan = install.plan_install("auto", home=tmp_path / "home", lock=lock)
    assert (plan.variant, plan.asset, plan.size) == (variant, asset, size)
    assert plan.backend == backend


def test_probes_never_fall_back_to_the_real_host(monkeypatch: pytest.MonkeyPatch,
                                                tmp_path: pathlib.Path) -> None:
    """With probes supplied nothing on the real machine may be read (E1a FIX t_eae35404).

    Every host source is replaced by a tripwire: any leak (`shutil.which`, `platform.*`,
    the `/dev/dri` glob, the Vulkan ICD stat) raises instead of quietly answering 'cuda'.
    """

    class Trap:
        def __init__(self, what: str) -> None:
            self.what = what

        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"host access leaked: {self.what}.{name}")

    def tripwire(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"host access leaked: {args} {kwargs}")

    monkeypatch.setattr(pins.shutil, "which", tripwire)
    monkeypatch.setattr(pins.platform, "system", tripwire)
    monkeypatch.setattr(pins.platform, "machine", tripwire)
    monkeypatch.setattr(pins, "DRI_DIR", Trap("/dev/dri"))
    monkeypatch.setattr(pins, "ICD_DIR", Trap("/usr/share/vulkan/icd.d"))
    icd = tmp_path / "icd.d"
    icd.mkdir()

    assert pins.detect_backend(system="linux", has_nvidia_smi=True) == "cuda"
    assert pins.detect_backend(system="linux", dri_nodes=["/dev/dri/renderD128"],
                               icd_dir=str(icd)) == "vulkan"
    assert pins.detect_backend(system="linux", dri_nodes=[], has_nvidia_smi=False) == "cpu"
    assert pins.detect_backend(system="windows", has_nvidia_smi=False, dri_nodes=[]) == "cpu"
    assert pins.detect_backend(system="darwin") == "metal"
    assert pins.host_variant("auto", system="linux", machine="x86_64") == "linux-x64-cpu"
    assert pins.host_variant("vulkan", system="linux", machine="x86_64") == "linux-x64-vulkan"
    assert pins.host_variant("linux-x64-cuda-13.3") == "linux-x64-cuda-13.3"
    probes = pins.fake_host(system="darwin", machine="arm64")
    assert pins.host_variant("auto", probes=probes) == "macos-arm64-metal"
    assert pins.host_variant("auto", probes=pins.fake_host(has_nvidia_smi=True,
                                                           machine="x86_64")) == \
        "linux-x64-cuda-12.8"
    # the probes= keyword of detect_backend() itself, not just its individual facts
    assert pins.detect_backend(probes=pins.fake_host(system="linux", has_nvidia_smi=True)) \
        == "cuda"
    assert pins.detect_backend(probes=probes) == "metal"


def test_current_host_is_the_only_reader_of_the_real_machine(monkeypatch: pytest.MonkeyPatch,
                                                            tmp_path: pathlib.Path) -> None:
    """`current_host()` reports the machine; the synthetic constructor never probes."""
    install_fake_machine(monkeypatch, tmp_path, world="cuda")
    host = pins.current_host()
    assert (host.system, host.machine) == ("linux", "x86_64")
    assert host.has_nvidia_smi is True
    assert host.detect_backend() == "cuda"
    assert host.to_dict()["has_nvidia_smi"] is True

    faux = pins.fake_host(system="linux", machine="x86_64", dri_nodes=["/dev/dri/renderD128"],
                          icd_dir=str(tmp_path / "icd.d"))
    assert faux.has_nvidia_smi is False
    assert faux.detect_backend() == "cpu"  # no ICD at that path -> no Vulkan claim
    assert (tmp_path / "icd.d").mkdir() is None
    assert faux.detect_backend() == "vulkan"
