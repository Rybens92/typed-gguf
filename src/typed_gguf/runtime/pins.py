"""Pinned runtime facts: release tag, asset names/sizes, required symbols, arch gates

Milestone: E1a.

Mirrors docs/verify_runtime_contract.py — if these drift, the oracle fails.

`runtime.lock` (repo root) is the single source of truth (SPEC 4): nothing is downloaded
implicitly, the asset table carries name + size + (where known) SHA-256, and
`min_build_for_spark2_5` gates the arch pre-flight. This module is the typed accessor; the
oracle asserts both it and the lock against `docs/evidence/`.

`runtime.lock` is data, so a wheel cannot carry the *repo* file — the build copies it into the
package (`<pkg>/data/runtime.lock`, hatchling `force-include`) and `locate` reads that copy when
no checkout is above the package (card t_eff926f9).
"""
from __future__ import annotations

import json
import os
import pathlib
import platform
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from typed_gguf.errors import RuntimeMissingError

REQUIRED_SYMBOL_COUNT = 34  # 32 libllama.so + 2 libggml.so (SPEC 2.2)
LOCK_NAME = "runtime.lock"
ENV_LOCK = "TYPED_GGUF_LOCK"
#: Where a built wheel carries the lock, relative to the package root. `pyproject.toml` maps the
#: repo-root file there at build time; `tests/test_pins.py` pins the two spellings together, so a
#: rename on either side fails a fast gate instead of the install (card t_eff926f9).
PACKAGED_LOCK_RELATIVE = f"typed_gguf/data/{LOCK_NAME}"
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
    #: `owner/name` of the upstream release repository (`ggml-org/llama.cpp`), straight from the
    #: lock's own `llama_cpp.repo`. `runtime update` asks that repository's API for the release
    #: list — the same rule as the asset names and the URL template: what upstream is, the lock
    #: says, and a lock that names none cannot be updated from (card t_d88b4be0). Empty for a
    #: synthetic lock that omits the field.
    repo: str = ""

    @property
    def build(self) -> int:
        """`b11026` -> 11026."""
        digits = "".join(ch for ch in self.tag if ch.isdigit())
        return int(digits) if digits else 0

    @property
    def all_symbols(self) -> tuple[str, ...]:
        return self.required_symbols_llama + self.required_symbols_ggml


def packaged_lock_path(package_file: pathlib.Path | None = None) -> pathlib.Path:
    """`<pkg>/data/runtime.lock` — the lock a built wheel carries (card t_eff926f9).

    A checkout keeps `runtime.lock` at the repo root, above `src/typed_gguf/`; an installed
    distribution has no repo above it, so the build copies the root file in here (see
    `PACKAGED_LOCK_RELATIVE`). `package_file` is the module the package root is read from —
    `__file__` in production, a shape built on disk in tests.
    """
    here = pathlib.Path(__file__) if package_file is None else pathlib.Path(package_file)
    return here.resolve().parents[1] / "data" / LOCK_NAME


def lock_candidates(*, environ: Mapping[str, str] | None = None,
                    package_file: pathlib.Path | None = None,
                    cwd: pathlib.Path | None = None) -> tuple[pathlib.Path, ...]:
    """Every path a default lookup searches, in precedence order (card t_eff926f9).

    1. `$TYPED_GGUF_LOCK` — an explicit override, used as-is: a path that does not exist is an
       error, never a silent fallback (a quiet one would leave the caller's pin looking in use);
    2. the nearest `runtime.lock` above the package — a dev run from a checkout keeps winning;
    3. `packaged_lock_path()` — the copy the wheel ships, the only lock an out-of-tree install
       (`uvx --from …`, `uv tool install`, pip into a plain venv) has at all;
    4. `<cwd>/runtime.lock` — the last resort the old error text meant by "run from the
       repository root". The packaged copy wins over it, so a stray lock in the cwd cannot
       hijack an installed tool.
    """
    source: Mapping[str, str] = os.environ if environ is None else environ
    override = source.get(ENV_LOCK)
    here = pathlib.Path(__file__) if package_file is None else pathlib.Path(package_file)
    here = here.resolve()
    start = pathlib.Path.cwd() if cwd is None else pathlib.Path(cwd)
    ordered: list[pathlib.Path] = []
    if override:
        ordered.append(pathlib.Path(override).expanduser())
    ordered.extend(parent / LOCK_NAME for parent in here.parents)
    ordered.append(packaged_lock_path(here))
    ordered.append(start / LOCK_NAME)
    unique: list[pathlib.Path] = []
    for candidate in ordered:
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def located_lock(*, environ: Mapping[str, str] | None = None,
                 package_file: pathlib.Path | None = None,
                 cwd: pathlib.Path | None = None) -> pathlib.Path | None:
    """The first candidate that exists, or `None` when this installation has no lock at all."""
    return next((path for path in lock_candidates(environ=environ, package_file=package_file,
                                                  cwd=cwd) if path.exists()), None)


def missing_lock_message(searched: Sequence[pathlib.Path], *,
                         override: str | None = None) -> str:
    """The `E_RUNTIME_MISSING` text: it names every path that was (or would have been) searched.

    The text it replaces — "run from the repository root or set TYPED_GGUF_LOCK" — sent a
    `uvx`/`uv tool install` user to the one thing an installed package cannot be: a checkout.
    It also named no path, so the install that was actually broken stayed invisible.
    """
    if override is not None:
        head = (f"E_RUNTIME_MISSING: {ENV_LOCK}={override} does not exist; the override is used "
                f"as-is, so nothing else was searched, and a package without its lock cannot "
                f"answer")
        rows = [f"  {override} (named by {ENV_LOCK})"]
    else:
        head = ("E_RUNTIME_MISSING: no runtime.lock found — this installation cannot answer "
                "without one. Searched, in precedence order:")
        rows = [f"  {path}" for path in searched]
    return "\n".join([head, *rows,
                      f"set {ENV_LOCK} to a readable lock file, or reinstall typed-gguf "
                      f"(the packaged copy belongs at {packaged_lock_path()})"])


def default_lock_path() -> pathlib.Path:
    """`$TYPED_GGUF_LOCK`, else the nearest `runtime.lock` above this package, else the copy the
    wheel ships, else `./runtime.lock` — the order `load_lock()` searches.

    Nothing found still names a file rather than raising: the override if one was given (the
    caller's own precedence), otherwise the packaged copy, which is what a healthy install has.
    `load_lock()` is where the list of searched paths is reported.
    """
    found = located_lock()
    if found is not None:
        return found
    override = os.environ.get(ENV_LOCK)
    return pathlib.Path(override).expanduser() if override else packaged_lock_path()


def load_lock(path: pathlib.Path | None = None) -> RuntimeLock:
    """Parse the lock file into typed pins (never downloads anything).

    With no `path` the lock is located (`lock_candidates` order); the error for a lookup that
    found nothing lists every path that was searched, because that list is the diagnosis.
    """
    if path is None:
        candidates = lock_candidates()
        override = os.environ.get(ENV_LOCK)
        if override:
            named = pathlib.Path(override).expanduser()
            if not named.exists():
                raise RuntimeMissingError(missing_lock_message(candidates, override=override))
            path = named
        else:
            found = next((candidate for candidate in candidates if candidate.exists()), None)
            if found is None:
                raise RuntimeMissingError(missing_lock_message(candidates))
            path = found
    path = pathlib.Path(path)
    if not path.exists():
        raise RuntimeMissingError(
            f"E_RUNTIME_MISSING: {path} not found (this lock was named by the caller; the copy "
            f"this installation ships is {packaged_lock_path()})")
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
            default_model=default_model, source_path=path, system_libs=system_libs,
            repo=str(llama.get("repo") or ""))
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
        + ". See SPEC 4 (build-from-source fallback or TYPED_GGUF_RUNTIME_DIR)")


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
            f"available: {known} (use --backend auto or TYPED_GGUF_RUNTIME_DIR)")
    raise RuntimeMissingError(
        f"E_RUNTIME_MISSING: no pinned llama.cpp bundle for platform {platform_name} "
        f"(backend {chosen!r}); see SPEC 4 for the build-from-source fallback, or point "
        f"TYPED_GGUF_RUNTIME_DIR at an existing runtime")


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
