"""Capability + arch pre-flight: build number, symbol scan, backend list

Milestone: E1a.

SPEC 2.2: never load a model the runtime cannot run; raise
E_MODEL_ARCH_UNSUPPORTED with actionable text.

Everything here is *pre-flight*: it inspects the bundle (files, symbols, build string,
compiled-in backends) without loading a model. `deep=True` resolves all 34 required symbols
through ctypes — the same check the oracle runs in section B — and is what `ggufone doctor`
uses; `deep=False` (or `GGUFONE_DEEP_PROBE=0`) keeps CI able to probe a bundle it cannot
actually dlopen.
"""
from __future__ import annotations

import ctypes as C  # noqa: N812
import dataclasses
import os
import pathlib
import platform
import re
import subprocess
from dataclasses import dataclass, field

from ggufone.errors import ModelArchUnsupportedError
from ggufone.runtime import finder, pins

_BUILD_RE = re.compile(rb"build\s+b?(\d{2,7})")
MIN_BUILD_ARCH = {"spark2_5": "spark2_5"}


def parse_build(text: str) -> int | None:
    """`version: 0.4.1-dev (build 11026, commit …)` -> 11026; `build b10828` -> 10828."""
    match = re.search(r"build\s+b?(\d{2,7})", text)
    return int(match.group(1)) if match else None


def build_tag(build: int | None) -> str | None:
    return f"b{build}" if build else None


def build_number(runtime_dir: str | os.PathLike[str]) -> int | None:
    """Build of the installed bundle: `llama-cli --version`, else a scan of libllama.so."""
    runtime_dir = pathlib.Path(runtime_dir)
    cli = runtime_dir / "llama-cli"
    if cli.exists():
        env = {**os.environ,
               "LD_LIBRARY_PATH": f"{runtime_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
        try:
            proc = subprocess.run([str(cli), "--version"], capture_output=True, check=False,  # noqa: S603
                                  cwd=str(runtime_dir), env=env, timeout=60)
        except (OSError, subprocess.SubprocessError):
            proc = None
        if proc is not None:
            # llama-cli prints its banner on stderr
            build = parse_build((proc.stdout or b"").decode("utf-8", "replace")
                                + (proc.stderr or b"").decode("utf-8", "replace"))
            if build is not None:
                return build
    lib = runtime_dir / finder.library_names()["llama"]
    if lib.exists():
        match = _BUILD_RE.search(lib.read_bytes())
        if match:
            return int(match.group(1))
    return None


def arch_symbol_names(runtime_dir: str | os.PathLike[str], arch: str) -> list[str]:
    """Symbols a runtime would need to implement `arch` (SPEC 2.2: `llama_model_<arch>`)."""
    candidates = [arch, arch.replace("-", "_")]
    return [f"llama_model_{name}" for name in candidates]


def supports_arch(runtime_dir: str | os.PathLike[str], arch: str, *,
                  build: int | None = None, lock: pins.RuntimeLock | None = None) -> bool:
    """Does the bundle carry the arch implementation *and* a new enough build?"""
    runtime_dir = pathlib.Path(runtime_dir)
    lock = lock or pins.load_lock()
    lib = runtime_dir / finder.library_names()["llama"]
    if not lib.exists():
        return False
    blob = lib.read_bytes()
    if not any(name.encode() + b"\x00" in blob for name in arch_symbol_names(runtime_dir, arch)):
        return False
    min_build = lock.min_build_for_spark2_5 if arch in MIN_BUILD_ARCH else 0
    current = build if build is not None else build_number(runtime_dir)
    return current is None or current >= min_build


def arch_guard_build(arch: str, lock: pins.RuntimeLock) -> int:
    return lock.min_build_for_spark2_5 if arch in MIN_BUILD_ARCH else 0


def require_arch(runtime_dir: str | os.PathLike[str], arch: str, *,
                 build: int | None = None, lock: pins.RuntimeLock | None = None) -> None:
    """Pre-flight gate before a model load (A11 / A-E1a-9): never a crash, always a fix."""
    runtime_dir = pathlib.Path(runtime_dir)
    lock = lock or pins.load_lock()
    current = build if build is not None else build_number(runtime_dir)
    min_build = arch_guard_build(arch, lock)
    tag = build_tag(current) or "unknown build"
    if min_build and current is not None and current < min_build:
        raise ModelArchUnsupportedError(
            f"E_MODEL_ARCH_UNSUPPORTED: model architecture {arch!r} needs runtime build "
            f"b{min_build} or newer, but {runtime_dir} is {tag}; fix: `ggufone init --force` "
            f"(installs the pinned b11026 bundle) or point GGUFONE_RUNTIME_DIR at a build "
            f">= b{min_build}")
    if not supports_arch(runtime_dir, arch, build=current, lock=lock):
        raise ModelArchUnsupportedError(
            f"E_MODEL_ARCH_UNSUPPORTED: the {tag} runtime at {runtime_dir} has no "
            f"implementation for architecture {arch!r} (looked for "
            f"{arch_symbol_names(runtime_dir, arch)[0]} in libllama.so); fix: "
            f"`ggufone init --force` or install a llama.cpp build that supports {arch}")


# --------------------------------------------------------------------- backends
def _backend_name(filename: str) -> str | None:
    stem = filename
    for prefix in ("libggml-", "ggml-"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    else:
        return None
    core = stem.split(".")[0]  # drop the extension
    if core == "base":
        return None
    if core.startswith("cpu"):
        return "cpu"
    return core.split("-")[0]


def backends(runtime_dir: str | os.PathLike[str], *, system: str | None = None) -> list[str]:
    """Accelerators compiled into this bundle (libggml-<backend>.* in the runtime dir)."""
    runtime_dir = pathlib.Path(runtime_dir)
    found: set[str] = set()
    for path in sorted(runtime_dir.glob(finder.library_glob(system))):
        name = _backend_name(path.name)
        if name:
            found.add(name)
    return sorted(found)


def load_backend_library(path: str | os.PathLike[str] | pathlib.Path) -> str | None:
    """dlopen one backend shared object: `None` when it loads, else the error text.

    This is the check that decides whether a GPU bundle is usable *on this host*: the pinned
    CUDA build fails with `libcudart.so.12: cannot open shared object file` on a box without
    the CUDA runtime/driver, which is what makes `init` fall back cuda -> vulkan -> cpu and
    what makes `doctor` report the working backend (E1a FIX requirement 4). Real dlopen in
    production; tests inject this seam instead of shipping a loadable ELF per backend.
    """
    path = pathlib.Path(path)
    try:
        C.CDLL(str(path), mode=getattr(C, "RTLD_GLOBAL", 0))
    except OSError as exc:
        return f"{path.name}: {exc}"
    return None


# --------------------------------------------------------------------- probe
@dataclass
class ProbeResult:
    runtime_dir: pathlib.Path | None = None
    deep: bool = False
    symbols_checked: bool = False
    present: tuple[str, ...] = ()
    missing_files: tuple[str, ...] = ()
    missing_symbols: tuple[str, ...] = ()
    build: int | None = None
    expected_tag: str = ""
    min_build: int = 0
    backends: tuple[str, ...] = ()
    backend_errors: dict[str, str] = field(default_factory=dict)
    tools: dict[str, str] = field(default_factory=dict)
    fit_params_help_exit: int | None = None
    expect_backend: str | None = None
    error: str | None = None

    @property
    def tag(self) -> str | None:
        return build_tag(self.build)

    def usable(self, backend: str) -> bool:
        """Is `backend` both present in this bundle *and* loadable on this host?"""
        if backend == "cpu":
            return True
        return backend in self.backends and backend not in self.backend_errors

    def accelerator(self) -> str:
        """The best accelerator this bundle can actually drive here (cpu is always driveable)."""
        for name in self.backends:
            if name not in ("cpu", "rpc", "base") and name not in self.backend_errors:
                return name
        return "cpu"

    def failures(self) -> list[str]:
        out: list[str] = []
        if self.error:
            out.append(self.error)
        if self.missing_files:
            out.append("runtime bundle is incomplete: missing "
                       + ", ".join(self.missing_files)
                       + " (re-run `ggufone init --force`)")
        if self.symbols_checked and self.missing_symbols:
            out.append(f"{len(self.missing_symbols)} of the required symbols do not resolve: "
                       + ", ".join(self.missing_symbols[:8])
                       + ("…" if len(self.missing_symbols) > 8 else ""))
        if self.build is None:
            out.append("cannot determine the runtime build (no llama-cli banner and no build "
                       "string in libllama.so)")
        elif self.min_build and self.build < self.min_build:
            out.append(f"runtime build b{self.build} is older than the minimum b{self.min_build} "
                       f"required for spark2_5 (re-run `ggufone init --force`)")
        if self.fit_params_help_exit not in (None, 0):
            out.append(f"llama-fit-params --help exited {self.fit_params_help_exit} "
                       f"(auto-fit unavailable)")
        return out

    def warnings(self) -> list[str]:
        out: list[str] = []
        if self.build is not None and self.expected_tag and self.tag != self.expected_tag:
            out.append(f"runtime build {self.tag} differs from the pinned {self.expected_tag} "
                       f"(the oracle pins the pinned build; re-run `ggufone init --force`)")
        if not self.symbols_checked:
            out.append("symbol probe skipped (deep probe disabled via GGUFONE_DEEP_PROBE=0)")
        for name, error in sorted(self.backend_errors.items()):
            out.append(f"W_BACKEND_LOAD: the {name} backend of this bundle does not load on "
                       f"this host ({error}); ggufone uses {self.accelerator()} instead "
                       f"(backend_errors in `doctor --json` has the raw dlopen error)")
        if self.missing_files:
            return out
        if self.expect_backend and self.expect_backend not in self.backends:
            out.append(f"expected accelerator {self.expect_backend!r} is not in the bundle "
                       f"(backends: {', '.join(self.backends) or 'none'})")
        elif not [b for b in self.backends if b != "cpu"]:
            out.append("no accelerator detected (backends: "
                       + (", ".join(self.backends) or "none")
                       + "); CPU works but decoding is slower")
        if "vulkan" in self.backends:
            out.append("W_VULKAN_WARMUP: the first Vulkan decode pays the shader compilation "
                       "(23-30 s measured); `ggufone init`/`serve` warm up before serving")
        return out

    def ok(self) -> bool:
        return not self.failures()


def probe_symbols(runtime_dir: pathlib.Path, symbols_llama: tuple[str, ...],
                  symbols_ggml: tuple[str, ...], *, system: str | None = None
                  ) -> tuple[list[str], list[str], str | None]:
    """Resolve the ABI through ctypes (ggml first, RTLD_GLOBAL) — same as the oracle."""
    names = finder.library_names(system)
    opened: dict[str, C.CDLL] = {}
    for key in ("ggml_base", "ggml", "llama"):
        path = runtime_dir / names[key]
        if not path.exists():
            return [], [], f"cannot load {path}: file is missing"
        try:
            opened[key] = C.CDLL(str(path), mode=getattr(C, "RTLD_GLOBAL", 0))
        except OSError as exc:
            return [], [], f"cannot load {path}: {exc}"
    missing_llama = [s for s in symbols_llama if not hasattr(opened["llama"], s)]
    missing_ggml = [s for s in symbols_ggml if not hasattr(opened["ggml"], s)]
    return missing_llama, missing_ggml, None


def probe_runtime(runtime_dir: str | os.PathLike[str] | None = None, *, deep: bool = True,
                  lock: pins.RuntimeLock | None = None, expect_backend: str | None = None,
                  run_tools: bool = True, system: str | None = None) -> ProbeResult:
    lock = lock or pins.load_lock()
    result = ProbeResult(deep=deep, expected_tag=lock.tag,
                         min_build=lock.min_build_for_spark2_5, expect_backend=expect_backend)
    if runtime_dir is None:
        found = finder.find_runtime()
        if found is None:
            result.error = ("E_RUNTIME_MISSING: no llama.cpp runtime installed; "
                            "run `ggufone init`")
            return result
        runtime_dir = found
    rt = pathlib.Path(runtime_dir)
    result.runtime_dir = rt
    if not rt.is_dir():
        result.error = f"E_RUNTIME_MISSING: {rt} is not a directory"
        return result

    present = tuple(f for f in lock.required_files if (rt / f).exists())
    missing = tuple(f for f in lock.required_files if not (rt / f).exists())
    result.present = present
    result.missing_files = missing
    layout = finder.layout(rt, system=system)
    result.tools = {name: str(path) for name, path in layout.tools.items()}
    result.backends = tuple(backends(rt, system=system))
    result.build = build_number(rt)

    if missing:
        result.error = (f"E_RUNTIME_MISSING: {rt} is not a complete runtime bundle "
                        f"(missing {', '.join(missing)})")
    elif deep:
        result.symbols_checked = True
        missing_llama, missing_ggml, error = probe_symbols(
            rt, lock.required_symbols_llama, lock.required_symbols_ggml, system=system)
        result.missing_symbols = tuple(missing_llama + missing_ggml)
        result.error = error
    if deep and not missing:
        # Independent of the symbol scan: each accelerator backend has to dlopen *on this host*
        # or `init` falls back to the next tier (E1a FIX requirement 4). A CUDA build fails here
        # with `libcudart.so.12: cannot open shared object file` when the runtime/driver is absent.
        for path in sorted(rt.glob(finder.library_glob(system))):
            name = _backend_name(path.name)
            if name in (None, "cpu"):
                continue
            backend_error = load_backend_library(path)
            if backend_error:
                result.backend_errors[name] = backend_error
    if run_tools:
        fit = rt / "llama-fit-params"
        if "llama-fit-params" in layout.tools:
            env = {**os.environ,
                   "LD_LIBRARY_PATH": f"{rt}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
            try:
                proc = subprocess.run([str(fit), "--help"], capture_output=True,  # noqa: S603
                                      check=False, cwd=str(rt), env=env, timeout=60)
                result.fit_params_help_exit = proc.returncode
            except (OSError, subprocess.SubprocessError):
                result.fit_params_help_exit = None
    return result


def deep_probe_enabled(env: dict[str, str] | None = None) -> bool:
    env = env if env is not None else os.environ
    return env.get("GGUFONE_DEEP_PROBE", "1") not in ("0", "false", "no")


def host_expectation() -> str:
    """The accelerator this host should be using (SPEC 4 GPU detection)."""
    return pins.detect_backend()


def platform_summary(runtime_dir: str | os.PathLike[str] | None = None) -> dict[str, str]:
    return {
        "system": platform.system().lower(),
        "machine": platform.machine(),
        "libllama": finder.library_names()["llama"],
        "runtime_dir": str(runtime_dir) if runtime_dir else "",
        "expected_backend": host_expectation(),
        "min_build_for_spark2_5": str(pins.load_lock().min_build_for_spark2_5),
    }


def as_dict(result: ProbeResult) -> dict[str, object]:
    payload = dataclasses.asdict(result)
    payload["runtime_dir"] = str(result.runtime_dir) if result.runtime_dir else None
    payload["tag"] = result.tag
    payload["ok"] = result.ok()
    payload["failures"] = result.failures()
    payload["warnings"] = result.warnings()
    return payload
