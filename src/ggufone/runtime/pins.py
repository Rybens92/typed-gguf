"""Pinned runtime facts: release tag, asset names/sizes, required symbols, arch gates

Milestone: E1a.

Mirrors docs/verify_runtime_contract.py — if these drift, the oracle fails.

`runtime.lock` (repo root) is the single source of truth (SPEC 4): nothing is downloaded
implicitly, the asset table carries name + size + (where known) SHA-256, and
`min_build_for_spark2_5` gates the arch pre-flight. This module is the typed accessor; the
oracle asserts both it and the lock against `docs/evidence/`.
"""
from __future__ import annotations

import json
import pathlib
import platform
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ggufone.errors import RuntimeMissingError

REQUIRED_SYMBOL_COUNT = 34  # 32 libllama.so + 2 libggml.so (SPEC 2.2)
LOCK_NAME = "runtime.lock"
NVIDIA_SMI = "nvidia-smi"
DRI_DIR = pathlib.Path("/dev/dri")
ICD_DIR = pathlib.Path("/usr/share/vulkan/icd.d")


@dataclass(frozen=True)
class Asset:
    variant: str
    asset: str
    size: int
    sha256: str | None = None
    verified_locally: bool = False


@dataclass(frozen=True)
class DefaultModel:
    repo: str
    repo_sha: str
    quant: str
    file: str
    size: int
    sha256: str
    arch: str
    license: str | None
    alternates: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class RuntimeLock:
    tag: str
    published_at: str
    commit: str
    min_build_for_spark2_5: int
    assets: dict[str, Asset]
    url_template: str
    required_files: tuple[str, ...]
    required_tools: tuple[str, ...]
    required_symbols_llama: tuple[str, ...]
    required_symbols_ggml: tuple[str, ...]
    mandatory_call_order: tuple[str, ...]
    default_model: DefaultModel
    source_path: pathlib.Path
    #: variant -> system sonames the pinned bundle links but does not ship. `init` pre-flights
    #: them before downloading (E1a FIX finding 2): empty for a variant we have not verified.
    system_libs: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def build(self) -> int:
        """`b11026` -> 11026."""
        digits = "".join(ch for ch in self.tag if ch.isdigit())
        return int(digits) if digits else 0

    @property
    def all_symbols(self) -> tuple[str, ...]:
        return self.required_symbols_llama + self.required_symbols_ggml


def default_lock_path() -> pathlib.Path:
    """`$GGUFONE_LOCK`, else the nearest `runtime.lock` above this package."""
    env = __import__("os").environ.get("GGUFONE_LOCK")
    if env:
        return pathlib.Path(env).expanduser()
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / LOCK_NAME
        if candidate.exists():
            return candidate
    return pathlib.Path.cwd() / LOCK_NAME


def load_lock(path: pathlib.Path | None = None) -> RuntimeLock:
    """Parse the lock file into typed pins (never downloads anything)."""
    path = pathlib.Path(path) if path else default_lock_path()
    if not path.exists():
        raise RuntimeMissingError(
            f"E_RUNTIME_MISSING: {path} not found; run from the repository root or set "
            f"GGUFONE_LOCK")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        llama = payload["llama_cpp"]
        assets = {variant: Asset(variant=variant, asset=spec["asset"], size=int(spec["size"]),
                                sha256=spec.get("sha256"),
                                verified_locally=bool(spec.get("verified_locally")))
                  for variant, spec in llama["assets"].items()}
        dm = payload["default_model"]
        default_model = DefaultModel(
            repo=dm["repo"], repo_sha=dm["repo_sha"], quant=dm["quant"], file=dm["file"],
            size=int(dm["size"]), sha256=dm["sha256"], arch=dm["arch"],
            license=dm.get("license"), alternates=dict(dm.get("alternates", {})))
        system_libs = {variant: tuple(names)
                       for variant, names in (llama.get("system_libs") or {}).items()}
        return RuntimeLock(
            tag=llama["tag"], published_at=llama["published_at"], commit=llama["commit"],
            min_build_for_spark2_5=int(llama["min_build_for_spark2_5"]),
            assets=assets, url_template=llama["url_template"],
            required_files=tuple(llama["required_files"]),
            required_tools=tuple(llama["required_tools"]),
            required_symbols_llama=tuple(llama["required_symbols_llama"]),
            required_symbols_ggml=tuple(llama["required_symbols_ggml"]),
            mandatory_call_order=tuple(llama["mandatory_call_order"]),
            default_model=default_model, source_path=path, system_libs=system_libs)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeMissingError(
            f"E_RUNTIME_MISSING: {path} is not a usable runtime lock "
            f"({exc.__class__.__name__}: {exc})") from exc


def asset_for(lock: RuntimeLock, variant: str) -> Asset:
    if variant in lock.assets:
        return lock.assets[variant]
    raise RuntimeMissingError(
        f"E_RUNTIME_MISSING: no prebuilt bundle for {variant!r}; available: "
        + ", ".join(sorted(lock.assets))
        + ". See SPEC 4 (build-from-source fallback or GGUFONE_RUNTIME_DIR)")


def url_for(lock: RuntimeLock, variant: str) -> str:
    asset = asset_for(lock, variant)
    return lock.url_template.format(tag=lock.tag, asset=asset.asset)


# --------------------------------------------------------------- host mapping
_X64 = ("x86_64", "amd64", "x64")
_ARM64 = ("arm64", "aarch64")
_VARIANTS: dict[tuple[str, str, str], str] = {
    ("linux", "x86_64", "cpu"): "linux-x64-cpu",
    ("linux", "x86_64", "vulkan"): "linux-x64-vulkan",
    ("linux", "x86_64", "cuda"): "linux-x64-cuda-12.8",
    ("windows", "x86_64", "cpu"): "windows-x64-cpu",
    ("windows", "x86_64", "vulkan"): "windows-x64-vulkan",
    ("windows", "x86_64", "cuda"): "windows-x64-cuda-12.4",
    ("darwin", "arm64", "metal"): "macos-arm64-metal",
    ("darwin", "x86_64", "metal"): "macos-x64-metal",
}


def _normalize_machine(machine: str) -> str:
    lowered = machine.lower()
    if lowered in _X64:
        return "x86_64"
    if lowered in _ARM64:
        return "arm64"
    return lowered


def detect_backend(*, probes: HostProbes | None = None, system: str | None = None,
                   machine: str | None = None, has_nvidia_smi: bool | None = None,
                   dri_nodes: list[str] | None = None, icd_dir: str | None = None) -> str:
    """Best-effort accelerator detection (no compilation, no model load).

    CUDA when `nvidia-smi` exists; else Vulkan when a DRM render node *and* a Vulkan ICD are
    present; Metal on macOS; CPU otherwise.

    Purity rule (E1a FIX t_eae35404): once *any* probe is supplied — this call's arguments or
    a `HostProbes` object — detection is a pure function of it and the real machine is never
    consulted (`shutil.which`, `platform.*`, `/dev/dri`, the ICD dir). Facts that were not
    supplied count as absent, so `host_variant("auto", system="linux", machine="x86_64")`
    answers `cpu` on a GPU box too. Only `detect_backend()` / `host_variant("auto")` with no
    probes at all read the real host (the production path).
    """
    return resolve_host(probes=probes, system=system, machine=machine,
                        has_nvidia_smi=has_nvidia_smi, dri_nodes=dri_nodes,
                        icd_dir=icd_dir).detect_backend()


def accelerator_of(variant: str) -> str:
    """`linux-x64-cuda-12.8` -> `cuda` (the accelerator a variant key carries)."""
    parts = variant.split("-")
    for name in ("cuda", "vulkan", "metal", "rocm", "sycl", "openvino", "cpu"):
        if name in parts:
            return name
    return "cpu"


def host_variant(backend: str = "auto", *, system: str | None = None, machine: str | None = None,
                 probes: HostProbes | None = None, **detect_kwargs: Any) -> str:
    """Map (OS, arch, backend) to a `runtime.lock` variant key.

    A bare `backend` without a `-` is an accelerator name resolved against the host; a name
    with one is a full variant key and is returned as-is (e.g. `--backend linux-x64-cuda-13.3`),
    without reading the host at all. In a synthetic world (`probes=` or any explicit fact) the
    platform must be named too: an unnamed machine is a caller error, not a reason to ask the
    real box.
    """
    chosen: str
    if backend != "auto" and "-" in backend:
        return backend.lower()  # an explicit variant name is accepted as-is
    host = resolve_host(probes=probes, system=system, machine=machine, **detect_kwargs)
    chosen = backend.lower() if backend != "auto" else host.detect_backend()
    os_name = host.system.lower()
    raw_machine = host.machine
    machine = _normalize_machine(raw_machine)
    variant = _VARIANTS.get((os_name, machine, chosen))
    if variant:
        return variant
    platform_name = f"{os_name}-{raw_machine}"
    known = ", ".join(v for (s, m, b), v in _VARIANTS.items() if (s, m) == (os_name, machine))
    if known:
        raise RuntimeMissingError(
            f"E_RUNTIME_MISSING: backend {chosen!r} has no pinned bundle for {platform_name}; "
            f"available: {known} (use --backend auto or GGUFONE_RUNTIME_DIR)")
    raise RuntimeMissingError(
        f"E_RUNTIME_MISSING: no pinned llama.cpp bundle for platform {platform_name} "
        f"(backend {chosen!r}); see SPEC 4 for the build-from-source fallback, or point "
        f"GGUFONE_RUNTIME_DIR at an existing runtime")


# --------------------------------------------------------------- host probes
@dataclass(frozen=True)
class HostProbes:
    """Every host fact detection is allowed to see (E1a FIX t_eae35404).

    Produced either by `current_host()` — the single place in the package that reads the real
    machine — or by `fake_host()` for tests/CI/reproducible plans. Detection is then a pure
    function of this object, so a caller that supplies probes can never be surprised by the
    machine it happens to run on.
    """

    system: str = ""
    machine: str = ""
    has_nvidia_smi: bool = False
    dri_nodes: tuple[str, ...] = ()
    icd_dir: str = ""

    def detect_backend(self) -> str:
        """CUDA > Vulkan > Metal > CPU, decided from the probe facts alone."""
        system = self.system.lower()
        if system == "darwin":
            return "metal"
        if self.has_nvidia_smi:
            return "cuda"
        if self.dri_nodes and self.icd_dir and pathlib.Path(self.icd_dir).is_dir():
            return "vulkan"
        return "cpu"

    def to_dict(self) -> dict[str, Any]:
        return {"system": self.system, "machine": self.machine,
                "has_nvidia_smi": self.has_nvidia_smi, "dri_nodes": list(self.dri_nodes),
                "icd_dir": self.icd_dir, "backend": self.detect_backend()}


def current_host() -> HostProbes:
    """Read the real machine. The only `shutil.which` / `platform` / `/dev/dri` reader."""
    system = platform.system().lower()
    dri_nodes: tuple[str, ...] = ()
    if DRI_DIR.is_dir():
        dri_nodes = tuple(sorted(str(node) for node in DRI_DIR.glob("renderD*")))
    return HostProbes(system=system, machine=platform.machine(),
                      has_nvidia_smi=shutil.which(NVIDIA_SMI) is not None,
                      dri_nodes=dri_nodes, icd_dir=str(ICD_DIR))


def fake_host(*, system: str = "linux", machine: str = "", has_nvidia_smi: bool = False,
              dri_nodes: Iterable[str] = (), icd_dir: str = "") -> HostProbes:
    """A synthetic machine: every fact that is not passed in is ABSENT, never probed.

    `system` defaults to `linux` because the CUDA/DRM facts are Linux-shaped and a caller
    injecting a single fact (`has_nvidia_smi=True`) still has to name a platform.
    """
    return HostProbes(system=system, machine=machine,
                      has_nvidia_smi=bool(has_nvidia_smi),
                      dri_nodes=tuple(str(node) for node in dri_nodes), icd_dir=str(icd_dir))


def resolve_host(*, probes: HostProbes | None = None, system: str | None = None,
                 machine: str | None = None, has_nvidia_smi: bool | None = None,
                 dri_nodes: list[str] | None = None,
                 icd_dir: str | None = None) -> HostProbes:
    """Probes in hand (or any single fact given) win; otherwise read the real machine."""
    if probes is not None:
        return probes
    given: dict[str, Any] = {}
    for name, value in (("system", system), ("machine", machine),
                        ("has_nvidia_smi", has_nvidia_smi), ("dri_nodes", dri_nodes),
                        ("icd_dir", icd_dir)):
        if value is not None:
            given[name] = value
    if not given:
        return current_host()
    return fake_host(**given)
