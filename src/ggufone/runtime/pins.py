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
from dataclasses import dataclass
from typing import Any

from ggufone.errors import RuntimeMissingError

REQUIRED_SYMBOL_COUNT = 34  # 32 libllama.so + 2 libggml.so (SPEC 2.2)
LOCK_NAME = "runtime.lock"


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
        return RuntimeLock(
            tag=llama["tag"], published_at=llama["published_at"], commit=llama["commit"],
            min_build_for_spark2_5=int(llama["min_build_for_spark2_5"]),
            assets=assets, url_template=llama["url_template"],
            required_files=tuple(llama["required_files"]),
            required_tools=tuple(llama["required_tools"]),
            required_symbols_llama=tuple(llama["required_symbols_llama"]),
            required_symbols_ggml=tuple(llama["required_symbols_ggml"]),
            mandatory_call_order=tuple(llama["mandatory_call_order"]),
            default_model=default_model, source_path=path)
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


def detect_backend(*, system: str | None = None, machine: str | None = None,
                   has_nvidia_smi: bool | None = None, dri_nodes: list[str] | None = None,
                   icd_dir: str | None = None) -> str:
    """Best-effort accelerator detection (no compilation, no model load).

    CUDA when `nvidia-smi` exists; else Vulkan when a DRM render node *and* a Vulkan ICD are
    present; Metal on macOS; CPU otherwise. Everything is overridable, which is what the
    poisoned-PATH / no-GPU CI paths need.
    """
    system = (system or platform.system()).lower()
    if system == "darwin":
        return "metal"
    if has_nvidia_smi is None:
        has_nvidia_smi = shutil.which("nvidia-smi") is not None
    if has_nvidia_smi:
        return "cuda"
    if dri_nodes is None:
        dri_nodes = sorted(str(p) for p in pathlib.Path("/dev/dri").glob("renderD*")) \
            if pathlib.Path("/dev/dri").is_dir() else []
    if icd_dir is None:
        icd_dir = "/usr/share/vulkan/icd.d"
    if dri_nodes and pathlib.Path(icd_dir).is_dir():
        return "vulkan"
    return "cpu"


def host_variant(backend: str = "auto", *, system: str | None = None, machine: str | None = None,
                 **detect_kwargs: Any) -> str:
    """Map (OS, arch, backend) to a `runtime.lock` variant key."""
    system = (system or platform.system()).lower()
    raw_machine = machine or platform.machine()
    machine = _normalize_machine(raw_machine)
    if backend != "auto":
        # an explicit variant name is accepted as-is (e.g. `--backend linux-x64-cuda-13.3`)
        lowered = backend.lower()
        if "-" in lowered:
            return lowered
        chosen = lowered
    else:
        chosen = detect_backend(system=system, machine=machine, **detect_kwargs)
    variant = _VARIANTS.get((system, machine, chosen))
    if variant:
        return variant
    platform_name = f"{system}-{raw_machine}"
    known = ", ".join(v for (s, m, b), v in _VARIANTS.items() if (s, m) == (system, machine))
    if known:
        raise RuntimeMissingError(
            f"E_RUNTIME_MISSING: backend {chosen!r} has no pinned bundle for {platform_name}; "
            f"available: {known} (use --backend auto or GGUFONE_RUNTIME_DIR)")
    raise RuntimeMissingError(
        f"E_RUNTIME_MISSING: no pinned llama.cpp bundle for platform {platform_name} "
        f"(backend {chosen!r}); see SPEC 4 for the build-from-source fallback, or point "
        f"GGUFONE_RUNTIME_DIR at an existing runtime")
