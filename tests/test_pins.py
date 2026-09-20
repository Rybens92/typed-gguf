"""runtime.lock is the single source of truth (SPEC 4): typed access + host mapping."""
from __future__ import annotations

import json
import pathlib
import tomllib

import pytest

from typed_gguf.errors import TypedGgufError
from typed_gguf.runtime import install, pins

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
    monkeypatch.setenv("TYPED_GGUF_LOCK", str(custom))
    assert pins.default_lock_path() == custom


def test_missing_lock_is_an_actionable_error(tmp_path: pathlib.Path) -> None:
    with pytest.raises(TypedGgufError) as exc:
        pins.load_lock(tmp_path / "absent.lock")
    assert exc.value.code == "E_RUNTIME_MISSING"
    assert "absent.lock" in str(exc.value)


def test_corrupt_lock_is_an_actionable_error(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "runtime.lock"
    path.write_text("{not json")
    with pytest.raises(TypedGgufError) as exc:
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
    with pytest.raises(TypedGgufError) as exc:
        pins.host_variant("cpu", system="linux", machine="aarch64")
    assert exc.value.code == "E_RUNTIME_MISSING"
    assert "aarch64" in str(exc.value)


def test_host_variant_rejects_backend_without_asset() -> None:
    with pytest.raises(TypedGgufError) as exc:
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
    with pytest.raises(TypedGgufError) as exc:
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


# ------------------------------------------------ the packaged lock (card t_eff926f9)
def test_the_wheel_copies_the_root_lock_into_the_package() -> None:
    """Single source of truth for the pin: `runtime.lock` stays the one committed file (data, at
    the repo root) and the *build* puts a copy inside the wheel. Requirement 2's "no drift" is
    structural here — there is no second copy in the tree that could fall behind."""
    wheel = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel["force-include"] == {"runtime.lock": pins.PACKAGED_LOCK_RELATIVE}


def test_the_packaged_lock_path_is_inside_the_distribution(tmp_path: pathlib.Path) -> None:
    """`<pkg>/data/runtime.lock`, next to the module that reads it, and the same destination the
    wheel build maps the root file to — the two spellings cannot drift apart."""
    package_root = pathlib.Path(pins.__file__).resolve().parents[1]
    assert pins.packaged_lock_path() == package_root / "data" / "runtime.lock"
    # the wheel root (site-packages) is where force-include's destination starts
    assert pins.packaged_lock_path().relative_to(package_root.parent).as_posix() == \
        pins.PACKAGED_LOCK_RELATIVE
    # the `package_file` seam is a *pure* function of the file it is handed, not a name for
    # `__file__` — a test that only reads the real package cannot tell those apart
    synthetic = tmp_path / "site-packages" / "typed_gguf" / "runtime" / "pins.py"
    assert pins.packaged_lock_path(synthetic) == synthetic.parent.parent / "data" / "runtime.lock"


def test_the_lookup_order_is_override_checkout_package_then_cwd(tmp_path: pathlib.Path) -> None:
    """Requirement 1's precedence as one list: a dev checkout keeps winning over the packaged
    copy, and the packaged copy wins over the cwd a wheel user happens to stand in."""
    package_file = tmp_path / "site-packages" / "typed_gguf" / "runtime" / "pins.py"
    package_file.parent.mkdir(parents=True)
    override = tmp_path / "override.lock"
    cwd = tmp_path / "neutral"
    cwd.mkdir()
    candidates = pins.lock_candidates(environ={"TYPED_GGUF_LOCK": str(override)},
                                      package_file=package_file, cwd=cwd)
    packaged = pins.packaged_lock_path(package_file)
    assert candidates[0] == override                                    # the explicit override
    assert candidates[1] == package_file.parent / "runtime.lock"        # nearest above, first hop
    assert packaged in candidates and candidates[-1] == cwd / "runtime.lock"
    assert candidates.index(packaged) < candidates.index(cwd / "runtime.lock")
    assert len(set(candidates)) == len(candidates), "a path must not be searched twice"


def test_an_installed_package_reads_the_copy_it_ships(tmp_path: pathlib.Path) -> None:
    """The uvx case: no repository above the package and a cwd with no lock of its own, so the
    only lock left is the one inside the distribution."""
    package_file = tmp_path / "site-packages" / "typed_gguf" / "runtime" / "pins.py"
    package_file.parent.mkdir(parents=True)
    packaged = pins.packaged_lock_path(package_file)
    packaged.parent.mkdir(parents=True)
    packaged.write_text((ROOT / "runtime.lock").read_text(encoding="utf-8"), encoding="utf-8")
    neutral = tmp_path / "neutral"
    neutral.mkdir()
    assert pins.located_lock(environ={}, package_file=package_file, cwd=neutral) == packaged


def test_the_search_uses_the_knobs_it_was_handed(tmp_path: pathlib.Path) -> None:
    """`located_lock` *is* the search: what it was given has to reach `lock_candidates`. A
    default-argument slip there would read the real environment (or the real cwd) inside a
    caller's world — the injection points are only worth what this pin is worth."""
    package_file = tmp_path / "site-packages" / "typed_gguf" / "runtime" / "pins.py"
    package_file.parent.mkdir(parents=True)
    override = tmp_path / "override.lock"
    override.write_text((ROOT / "runtime.lock").read_text(encoding="utf-8"), encoding="utf-8")
    neutral = tmp_path / "neutral"
    neutral.mkdir()
    # the override is the first rung, and nothing else exists yet
    assert pins.located_lock(environ={"TYPED_GGUF_LOCK": str(override)},
                             package_file=package_file, cwd=neutral) == override
    # …and with nothing above the package, the *given* cwd is the rung that answers (the real
    # cwd of this run is the checkout, whose lock would answer instead)
    own = neutral / "runtime.lock"
    own.write_text("{}", encoding="utf-8")
    assert pins.located_lock(environ={}, package_file=package_file, cwd=neutral) == own


def test_a_broken_install_lists_every_path_it_searched(tmp_path: pathlib.Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """The old text ("run from the repository root or set TYPED_GGUF_LOCK") sent a uvx user to the
    one thing an installed package cannot be — a checkout — and named no path, so the install that
    was actually broken stayed invisible. The text lists what was searched now (requirement 1)."""
    neutral = tmp_path / "neutral"
    neutral.mkdir()
    package_file = tmp_path / "site-packages" / "typed_gguf" / "runtime" / "pins.py"
    package_file.parent.mkdir(parents=True)
    monkeypatch.chdir(neutral)
    monkeypatch.setattr(pins, "__file__", str(package_file))
    monkeypatch.delenv("TYPED_GGUF_LOCK", raising=False)
    with pytest.raises(TypedGgufError) as exc:
        pins.load_lock()
    message = str(exc.value)
    assert message.startswith("E_RUNTIME_MISSING")
    # one path per line — the list *is* the diagnosis, so its shape is part of the contract
    assert f"\n  {package_file.parent / 'runtime.lock'}" in message   # nearest above the package
    assert f"\n  {pins.packaged_lock_path(package_file)}" in message  # the copy a wheel ships
    assert f"\n  {neutral / 'runtime.lock'}" in message               # the historical last resort
    assert "run from the repository root" not in message


def test_a_missing_override_is_named_and_never_silently_skipped(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`$TYPED_GGUF_LOCK` is authoritative: a typo fails loudly instead of quietly reading the
    packaged copy, which would leave the caller's pin looking effective when it is not in use."""
    typo = tmp_path / "typo.lock"
    monkeypatch.setenv("TYPED_GGUF_LOCK", str(typo))
    with pytest.raises(TypedGgufError) as exc:
        pins.load_lock()
    message = str(exc.value)
    assert "TYPED_GGUF_LOCK" in message and str(typo) in message
    assert str(pins.packaged_lock_path()) in message, "the message says what was NOT searched"


def test_the_override_still_wins_over_the_packaged_copy(tmp_path: pathlib.Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """Dev runs keep winning: with the override set, the packaged copy is not consulted at all."""
    custom = tmp_path / "runtime.lock"
    payload = json.loads((ROOT / "runtime.lock").read_text(encoding="utf-8"))
    payload["llama_cpp"]["tag"] = "b00042"
    custom.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("TYPED_GGUF_LOCK", str(custom))
    assert pins.default_lock_path() == custom
    assert pins.load_lock().tag == "b00042"


def test_load_lock_without_arguments_reads_the_lock_above_the_package(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The plain `load_lock()` is the production path (every command calls it with no argument):
    in a checkout it is the repo's own lock, found by the walk-up, not by the cwd."""
    monkeypatch.delenv("TYPED_GGUF_LOCK", raising=False)
    assert pins.load_lock().source_path == ROOT / "runtime.lock"


def test_default_lock_path_names_the_packaged_copy_when_nothing_exists(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing on disk: the accessor still names a file — the packaged copy a healthy install has,
    or the override the caller gave — because `load_lock`'s error is what reports the search."""
    neutral = tmp_path / "neutral"
    neutral.mkdir()
    package_file = tmp_path / "site-packages" / "typed_gguf" / "runtime" / "pins.py"
    monkeypatch.chdir(neutral)
    monkeypatch.setattr(pins, "__file__", str(package_file))
    monkeypatch.delenv("TYPED_GGUF_LOCK", raising=False)
    packaged = pins.packaged_lock_path()
    assert not packaged.exists()
    assert pins.default_lock_path() == packaged
    override = tmp_path / "override.lock"
    monkeypatch.setenv("TYPED_GGUF_LOCK", str(override))
    assert pins.default_lock_path() == override
