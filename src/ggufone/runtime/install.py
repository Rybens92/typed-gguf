"""`ggufone init`: prebuilt -> automated build -> guided manual ladder

Milestone: E1a.

SPEC 4 / hard rule A1: no compiler on the path for rung 1; never shell out
to cc/gcc/clang/nvcc/cmake/ninja unless the user explicitly asks for rung 2.

Rung 1 is the only automated rung in E1a: pick the pinned asset for this host, fetch it
(resume + size + SHA-256), extract it (path-safe), then probe the result and record
`runtime.json`. A user-provided runtime (rung 3) is consumed via `GGUFONE_RUNTIME_DIR` and
still goes through the same probe (`ggufone doctor`), so a hand-built runtime is first-class.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import tarfile
import time
import zipfile
from dataclasses import asdict, dataclass, field
from typing import Any

from ggufone.errors import (
    DownloadError,
    RuntimeMissingError,
    Sha256MismatchError,
)
from ggufone.registry import hf, store
from ggufone.registry.gguf import sha256_file
from ggufone.runtime import capability, ctypes_binding, finder, pins

EXTRACT_MULTIPLIER = 3          # archive -> on-disk size, generous (compressed .so files)
RUNG = "prebuilt"               # SPEC 4 rung 1
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar", ".zip")


@dataclass(frozen=True)
class InstallPlan:
    """What `init` would do: one pinned asset for this host, one destination."""

    tag: str
    variant: str
    asset: str
    url: str
    size: int
    sha256: str | None
    dest: pathlib.Path
    backend: str
    cached: pathlib.Path | None = None
    required_bytes: int = 0
    rung: str = RUNG
    host: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dest"] = str(self.dest)
        payload["cached"] = str(self.cached) if self.cached else None
        return payload


def offline_cache_path() -> pathlib.Path | None:
    raw = os.environ.get("GGUFONE_OFFLINE_CACHE")
    return pathlib.Path(os.path.expanduser(raw)) if raw else None


def plan_install(backend: str = "auto", *, home: pathlib.Path | None = None,
                 lock: pins.RuntimeLock | None = None,
                 offline_cache: pathlib.Path | str | None = None,
                 system: str | None = None, machine: str | None = None,
                 probes: pins.HostProbes | None = None,
                 **detect_kwargs: Any) -> InstallPlan:
    """Resolve (host -> variant -> asset -> destination) without touching the network."""
    home = home or store.data_home()
    lock = lock or pins.load_lock()
    host: pins.HostProbes | None = None
    if backend != "auto" and "-" in backend:
        variant = backend.lower()          # a full variant name needs no host facts
    else:
        host = pins.resolve_host(system=system, machine=machine, probes=probes, **detect_kwargs)
        variant = pins.host_variant(backend, probes=host)
    asset = pins.asset_for(lock, variant)
    cache = pathlib.Path(offline_cache) if offline_cache else offline_cache_path()
    cached = None
    if cache:
        candidate = cache / asset.asset if cache.is_dir() else cache
        if candidate.exists():
            cached = candidate
    return InstallPlan(
        tag=lock.tag, variant=variant, asset=asset.asset, url=pins.url_for(lock, variant),
        size=asset.size, sha256=asset.sha256,
        dest=home / "runtime" / f"{lock.tag}-{variant}",
        backend=backend if backend != "auto" else pins.accelerator_of(variant),
        cached=cached, required_bytes=asset.size * EXTRACT_MULTIPLIER,
        host=host.to_dict() if host else {})


# --------------------------------------------------------------------- fetching
def _fetch(url: str, dest: pathlib.Path, *, size: int | None = None,
           sha256: str | None = None, progress: hf.Progress | None = None,
           chunk: int = hf.CHUNK) -> tuple[hf.DownloadResult, str]:
    if url.startswith("file://"):
        source = pathlib.Path(urllib_path_to_path(url))
        return _copy_local(source, dest, size=size, sha256=sha256, progress=progress), "url"
    result = hf.download_url(url, dest, size=size, sha256=sha256, chunk=chunk, progress=progress)
    return result, "url"


def urllib_path_to_path(url: str) -> str:
    import urllib.request  # local import keeps the module import surface tiny
    return urllib.request.url2pathname(url[len("file://"):])


def _copy_local(source: pathlib.Path, dest: pathlib.Path, *, size: int | None,
                sha256: str | None, progress: hf.Progress | None) -> hf.DownloadResult:
    """`file://` fetch with the same resume/verify semantics as the network path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    total = source.stat().st_size
    if offset > total:  # stale part
        part.unlink()
        offset = 0
    fetched = 0
    with open(source, "rb") as src, open(part, "ab" if offset else "wb") as out:
        src.seek(offset)
        while True:
            block = src.read(hf.CHUNK)
            if not block:
                break
            out.write(block)
            fetched += len(block)
            if progress:
                progress(out.tell(), total)
        out.flush()
        os.fsync(out.fileno())
    if size is not None and part.stat().st_size != size:
        raise DownloadError(
            f"E_DOWNLOAD_FAILED: {source}: expected {size} bytes, got {part.stat().st_size}")
    digest = sha256_file(part)
    verified = False
    if sha256:
        if digest != sha256:
            raise Sha256MismatchError(
                f"E_SHA256_MISMATCH: {source}: expected {sha256}, got {digest}")
        verified = True
    os.replace(part, dest)
    return hf.DownloadResult(path=dest, bytes_fetched=fetched, bytes_total=total,
                             resumed_from=offset, sha256=digest, verified=verified,
                             url=source.as_uri())


def _obtain_archive(plan: InstallPlan, archive: pathlib.Path, *, url: str | None = None,
                    progress: hf.Progress | None = None) -> tuple[str, dict[str, Any]]:
    """Return (source, stats) with the archive verified in `archive`."""
    if url:
        result, source = _fetch(url, archive, size=plan.size, sha256=plan.sha256,
                                progress=progress)
        return source, result.to_dict()
    if plan.cached:
        cache = pathlib.Path(plan.cached)
        if cache.stat().st_size != plan.size:
            raise DownloadError(
                f"E_DOWNLOAD_FAILED: cached bundle {cache} is {cache.stat().st_size} bytes, "
                f"expected {plan.size}; delete it and re-run without GGUFONE_OFFLINE_CACHE")
        digest = sha256_file(cache)
        if plan.sha256 and digest != plan.sha256:
            raise Sha256MismatchError(
                f"E_SHA256_MISMATCH: cached bundle {cache} has {digest}, expected "
                f"{plan.sha256}")
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cache, archive)
        if progress:
            progress(plan.size, plan.size)
        return "offline-cache", {"bytes_fetched": plan.size, "bytes_total": plan.size,
                                 "resumed_from": 0, "sha256": digest,
                                 "verified": bool(plan.sha256), "path": str(archive)}
    result, source = _fetch(plan.url, archive, size=plan.size, sha256=plan.sha256,
                            progress=progress)
    return source, result.to_dict()


# --------------------------------------------------------------------- extracting
def extract_bundle(archive: str | os.PathLike[str], dest: str | os.PathLike[str]) -> None:
    """Extract a pinned bundle safely (no absolute paths, no `..`, no device files)."""
    archive = pathlib.Path(archive)
    dest = pathlib.Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    try:
        if name.endswith(".zip"):
            _extract_zip(archive, dest)
        else:
            with tarfile.open(archive, "r:*") as tar:
                try:
                    tar.extractall(dest, filter="data")  # py>=3.11.4 path-safety filter
                except TypeError:  # pragma: no cover - older patch levels
                    tar.extractall(dest)  # noqa: S202
    except (tarfile.TarError, zipfile.BadZipFile, OSError, ValueError) as exc:
        raise DownloadError(
            f"E_DOWNLOAD_FAILED: cannot extract {archive} ({exc.__class__.__name__}: {exc})"
        ) from exc


def _extract_zip(archive: pathlib.Path, dest: pathlib.Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise DownloadError(
                    f"E_DOWNLOAD_FAILED: {archive} contains an unsafe path ({member})")
        zf.extractall(dest)  # noqa: S202


def _flatten_bundle(directory: pathlib.Path) -> pathlib.Path:
    """Move a single top-level `llama-<tag>/` up so the libs land in the runtime dir."""
    children = [p for p in directory.iterdir() if p.name not in {".", ".."}]
    dirs = [p for p in children if p.is_dir()]
    if len(children) == 1 and dirs:
        inner = dirs[0]
        for item in inner.iterdir():
            shutil.move(str(item), str(directory / item.name))
        inner.rmdir()
        return directory
    return directory


# --------------------------------------------------------------------- warm-up
def warmup(runtime_dir: str | os.PathLike[str], model_path: str | os.PathLike[str], *,
           n_ctx: int = 128, n_threads: int = 1) -> float:
    """One tiny decode: loads the model, prefills, returns milliseconds (R2/A13)."""
    runtime = ctypes_binding.load_libraries(runtime_dir)
    llama = runtime.llama
    params = llama.llama_model_default_params()
    model = llama.llama_model_load_from_file(str(model_path).encode(), params)
    if not model:  # pragma: no cover - depends on a real model
        raise RuntimeMissingError(
            f"E_RUNTIME_MISSING: warm-up could not load {model_path} with {runtime_dir}")
    ctx = None
    try:
        ctx_params = llama.llama_context_default_params()
        ctx_params.n_ctx = n_ctx
        ctx_params.n_batch = 32
        ctx_params.n_ubatch = 32
        ctx_params.n_seq_max = 1
        ctx_params.n_threads = n_threads
        ctx_params.n_threads_batch = n_threads
        ctx_params.kv_unified = True
        ctx = llama.llama_init_from_model(model, ctx_params)
        if not ctx:  # pragma: no cover
            raise RuntimeMissingError("E_RUNTIME_MISSING: warm-up could not create a context")
        tokens = ctypes_binding.tokenize(runtime, llama.llama_model_get_vocab(model),
                                        "ggufone warm-up", max_tokens=32)
        started = time.perf_counter()
        token_array = (ctypes_binding.llama_token * len(tokens))(*tokens)
        batch = llama.llama_batch_get_one(token_array, len(tokens))
        rc = llama.llama_decode(ctx, batch)
        llama.llama_synchronize(ctx)
        elapsed = (time.perf_counter() - started) * 1000.0
        if rc != 0:  # pragma: no cover
            raise RuntimeMissingError(f"E_RUNTIME_MISSING: warm-up decode returned {rc}")
        return elapsed
    finally:
        if ctx:
            llama.llama_free(ctx)
        llama.llama_model_free(model)


# --------------------------------------------------------------------- install
#: When `auto` detection picks a GPU backend, `init` tries the next tier if that bundle cannot
#: actually load on this host (E1a FIX requirement 4: cuda -> vulkan -> cpu, reason recorded).
#: `cpu` closes every chain. An explicit `--backend` is honoured as asked, with no fallback.
FALLBACK_CHAIN: dict[str, tuple[str, ...]] = {
    "cuda": ("vulkan", "cpu"),
    "vulkan": ("cpu",),
    "metal": (),
}


def _unusable_reason(probe: capability.ProbeResult, backend: str) -> str:
    """Why `backend` cannot be used, from a real probe (goes into the record verbatim)."""
    if backend in probe.backend_errors:
        return f"{backend} does not load on this host ({probe.backend_errors[backend]})"
    if backend not in probe.backends:
        return (f"the bundle carries no {backend} backend (backends: "
                f"{', '.join(probe.backends) or 'none'})")
    return f"the {backend} backend reported unusable"


def _unpack_one(plan: InstallPlan, *, home: pathlib.Path, lock: pins.RuntimeLock,
                url: str | None, progress: hf.Progress | None,
                free_bytes: int | None) -> tuple[str, dict[str, Any]]:
    """Download + verify + extract one variant into `plan.dest` (rung 1, never a compiler)."""
    home.mkdir(parents=True, exist_ok=True)
    hf.check_disk_space(home, plan.required_bytes, free_bytes=free_bytes)
    archive = store.downloads_dir(home) / plan.asset
    source, stats = _obtain_archive(plan, archive, url=url, progress=progress)

    staging = plan.dest.parent / f".pending-{plan.asset}"
    if staging.exists():
        shutil.rmtree(staging)
    extract_bundle(archive, staging)
    _flatten_bundle(staging)
    missing = [f for f in lock.required_files if not (staging / f).exists()]
    if missing:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeMissingError(
            f"E_RUNTIME_MISSING: {plan.asset} does not contain {', '.join(missing)}; the "
            f"pinned asset layout changed (re-run `ggufone init` and report this)")
    if plan.dest.exists():
        shutil.rmtree(plan.dest)
    plan.dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, plan.dest)
    return source, stats


def _warmup_ms(home: pathlib.Path, plan: InstallPlan,
               warmup_model: str | os.PathLike[str] | None) -> tuple[float | None, str | None,
                                                                     str | None]:
    """(milliseconds, error, model path) — warm-up must never fail an install."""
    model_path = warmup_model
    if model_path is None:
        candidate = (finder.runtime_record(home) or {}).get("warmup_model")
        if candidate and pathlib.Path(candidate).exists():
            model_path = candidate
    if not model_path:
        return None, None, None
    try:
        return round(warmup(plan.dest, model_path), 3), None, str(model_path)
    except Exception as exc:  # noqa: BLE001 - a warm-up failure is recorded, never raised
        return None, f"{exc.__class__.__name__}: {exc}", str(model_path)


def _build_record(plan: InstallPlan, *, lock: pins.RuntimeLock, probe: capability.ProbeResult,
                  source: str, stats: dict[str, Any], warmup_ms: float | None,
                  warmup_error: str | None, model_path: str | None, url: str | None,
                  requested_backend: str,
                  attempts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": finder.RUNTIME_RECORD_SCHEMA,
        "tag": lock.tag,
        "variant": plan.variant,
        "asset": plan.asset,
        "asset_size": plan.size,
        "asset_sha256": plan.sha256 or stats.get("sha256"),
        "asset_sha256_observed": stats.get("sha256"),
        "asset_verified": bool(stats.get("verified")),
        "libllama_sha256": (sha256_file(plan.dest / finder.library_names()["llama"])
                            if (plan.dest / finder.library_names()["llama"]).exists()
                            else None),
        "url": url or plan.url,
        "source": source,
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dir": str(plan.dest),
        "build": probe.build,
        "backends": list(probe.backends),
        "required_files": list(lock.required_files),
        "tools": probe.tools,
        "symbols_ok": (probe.symbols_checked and not probe.missing_symbols),
        "symbols_probed": probe.symbols_checked,
        "missing_symbols": list(probe.missing_symbols),
        "fit_params_help_exit": probe.fit_params_help_exit,
        "warmup_ms": warmup_ms,
        "warmup_error": warmup_error,
        "warmup_model": model_path,
        "probe_failures": probe.failures(),
        "probe_warnings": probe.warnings(),
        "rung": plan.rung,
        "backend_requested": requested_backend,
        "backend_working": probe.accelerator(),
        "backend_errors": dict(probe.backend_errors),
        "fallback_attempts": [dict(attempt) for attempt in attempts],
        "fallback_reason": attempts[0]["reason"] if attempts else None,
    }


def install(backend: str = "auto", *, home: pathlib.Path | None = None,
            lock: pins.RuntimeLock | None = None, force: bool = False, dry_run: bool = False,
            offline_cache: pathlib.Path | str | None = None, url: str | None = None,
            free_bytes: int | None = None, progress: hf.Progress | None = None,
            warmup_model: str | os.PathLike[str] | None = None, deep: bool | None = None,
            system: str | None = None, machine: str | None = None,
            probes: pins.HostProbes | None = None,
            **detect_kwargs: Any) -> dict[str, Any]:
    """Rung 1: install the pinned prebuilt bundle for this host. Never invokes a compiler.

    With `backend="auto"` the chain is detection -> next tiers in `FALLBACK_CHAIN` -> cpu:
    a bundle whose GPU backend cannot dlopen here (missing cudart/driver) is recorded and the
    next tier is installed instead, so `init` never leaves a broken accelerator in place.
    """
    home = home or store.data_home()
    lock = lock or pins.load_lock()
    plan = plan_install(backend, home=home, lock=lock, offline_cache=offline_cache,
                        system=system, machine=machine, probes=probes, **detect_kwargs)
    if dry_run:
        return {"dry_run": True, "plan": plan.to_dict()}

    requested = plan.backend
    auto = backend == "auto"
    chain = [requested, *(FALLBACK_CHAIN.get(requested, ()) if auto else ())]
    deep_probe = capability.deep_probe_enabled() if deep is None else deep
    attempts: list[dict[str, Any]] = []

    for index, candidate in enumerate(chain):
        last = index == len(chain) - 1
        candidate_plan = plan if index == 0 else plan_install(
            candidate, home=home, lock=lock, offline_cache=offline_cache, system=system,
            machine=machine, probes=probes, **detect_kwargs)

        if candidate_plan.dest.exists() and not force:
            probe = capability.probe_runtime(candidate_plan.dest, deep=deep_probe, lock=lock,
                                             expect_backend=requested, run_tools=True,
                                             system=system)
            if probe.usable(candidate) or last:
                return {"already_installed": True, "variant": candidate_plan.variant,
                        "backend": candidate, "dir": str(candidate_plan.dest),
                        "record": finder.runtime_record(home),
                        "working_backend": probe.accelerator(),
                        "fallback_attempts": list(attempts),
                        "hint": "pass --force to re-download and re-extract"}
            attempts.append({"backend": candidate, "variant": candidate_plan.variant,
                             "reason": _unusable_reason(probe, candidate)})
            continue

        source, stats = _unpack_one(candidate_plan, home=home, lock=lock, url=url,
                                    progress=progress, free_bytes=free_bytes)
        probe = capability.probe_runtime(candidate_plan.dest, deep=deep_probe, lock=lock,
                                         expect_backend=requested, run_tools=True, system=system)
        if not probe.usable(candidate) and not last:
            attempts.append({"backend": candidate, "variant": candidate_plan.variant,
                             "reason": _unusable_reason(probe, candidate)})
            continue

        warmup_ms, warmup_error, model_path = _warmup_ms(home, candidate_plan, warmup_model)
        record = _build_record(candidate_plan, lock=lock, probe=probe, source=source, stats=stats,
                               warmup_ms=warmup_ms, warmup_error=warmup_error,
                               model_path=model_path, url=url, requested_backend=requested,
                               attempts=attempts)
        finder.write_runtime_record(record, home)
        return {
            "variant": candidate_plan.variant, "backend": candidate,
            "working_backend": probe.accelerator(), "dir": str(candidate_plan.dest),
            "source": source, "record": record, "asset_sha256": record["asset_sha256"],
            "build": probe.build, "backends": list(probe.backends),
            "backend_errors": dict(probe.backend_errors),
            "fallback_attempts": list(attempts),
            "fallback_reason": attempts[0]["reason"] if attempts else None,
            "warmup_ms": warmup_ms, "probe_failures": probe.failures(),
            "probe_warnings": probe.warnings(),
            **{k: stats.get(k) for k in ("bytes_fetched", "bytes_total", "resumed_from")},
        }
    raise AssertionError(f"unreachable: {chain} always ends in a driveable backend")
