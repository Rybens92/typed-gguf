"""`typed-gguf runtime update|rollback`: refresh the bundle `init` installed (SPEC 2.8).

Serve wave, card `t_d88b4be0`. Owner task: *"żeby dało się zaktualizować llama.cpp które się
instaluje poprzez init"*.

The shape SPEC 2.8 freezes, in order: **resolve** (current runtime, its tag/build from the
`runtime.json` record) → **target** (the newest *official* release whose asset list carries this
host's pinned bundle name **retagged**; never `releases/latest`, never a fork) → **stage**
(download into `<home>/downloads/` with `init`'s resume/size semantics, extract into
`<home>/runtime/.pending-<asset>` with the same path-safety filter and flatten rule, then the lock
probe in a child) → **switch** (stop the resident host, `os.replace` the staged directory into its
final name, then rewrite `runtime.json` **once**, atomically) → **report**.

The record write *is* the switch point: everything before it leaves `runtime.json`
byte-identical, so a failed download, a failed probe, or a `kill -9` anywhere earlier keeps the
working runtime active — and nothing is ever deleted (the previous bundle is what
`runtime rollback` returns to).

Decisions the card left open (flagged in the report, each one the simple reversible option):
staging lives in the runtime root (`<home>/runtime/.pending-<asset>`, `init`'s own name, so a
half-extracted bundle sits where the next run clears it); an interrupted download resumes exactly
as `init`'s does (`.part` + HTTP range, `hf.download_url`); a target directory already on disk is
*adopted* (probed and recorded, never re-downloaded — the rollback recipe leaves one behind); and
`update` does not warm up (the warm-up number belongs to a load, and `init` already measured this
box; the record keeps the model path it would use).
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import re
import shutil
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from typed_gguf import __version__
from typed_gguf.errors import (
    DownloadError,
    RuntimeBuildOldError,
    RuntimeMissingError,
    RuntimeSymbolsError,
    UpdateUnavailableError,
)
from typed_gguf.keep import client as keep_client
from typed_gguf.keep import state as keep_state
from typed_gguf.registry import hf, store
from typed_gguf.registry.gguf import sha256_file
from typed_gguf.runtime import capability, finder, install, pins

SCHEMA = "typed_gguf.runtime.update/v1"
ROLLBACK_SCHEMA = "typed_gguf.runtime.rollback/v1"
#: The GitHub release API of the *lock's own* upstream repository (`pins.RuntimeLock.repo`).
RELEASES_URL = "https://api.github.com/repos/{repo}/releases"
RELEASES_LIMIT = 20
REQUEST_TIMEOUT = 30.0
USER_AGENT = f"typed-gguf/{__version__}"
ACCEPT = "application/vnd.github+json"
#: `$TYPED_GGUF_RUNTIME_DIR`: the rung of the ladder `update` refuses to touch (SPEC 2.7 — that
#: runtime is consumed read-only, so replacing it is not ours to do).
ENV_RUNTIME_DIR = "TYPED_GGUF_RUNTIME_DIR"
#: The record key the retained bundle lives under, and its exact keys (SPEC 2.8 step 4).
PREVIOUS = "previous"
PREVIOUS_KEYS = ("dir", "tag", "build", "installed_at")
#: What `install` records for a bundle nobody downloaded (the adopt path): a source tag that says
#: so, so `runtime.json` never claims a download that did not happen.
ADOPTED_SOURCE = "already-downloaded"

#: The wire of `--json`, pinned here so the gates and the payload cannot drift apart.
UPDATE_KEYS = ("schema", "check", "updated", "reason", "from", "to", "asset", "probe",
               "previous", "host_stopped", "home")
ROLLBACK_KEYS = ("schema", "rolled_back", "from", "to", "previous", "host_stopped", "home")
FROM_KEYS = ("tag", "build", "dir", "variant")
TO_KEYS = FROM_KEYS
ASSET_KEYS = ("name", "size", "size_human", "sha256", "url", "published_at")
PROBE_KEYS = ("build", "tools", "symbols_ok", "symbols_probed", "missing_symbols", "backends",
              "expect_backend")

#: The seam the network leg goes through: `update(fetch=…)` replaces it in tests, and
#: `tests/test_runtime_update.py` replaces this name for the CLI-level gates.
DEFAULT_FETCH: Callable[..., list[Release]]
#: `state.pid_alive`, the one liveness probe the stop check uses (patched in tests).
pid_alive = keep_state.pid_alive


# --------------------------------------------------------------------------- the release list
@dataclass(frozen=True)
class ReleaseAsset:
    """One asset of one release, as GitHub lists it: name, bytes, optional `sha256:` digest."""

    name: str
    size: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class Release:
    """One release: its tag, when it was published, and the assets it carries."""

    tag: str
    published_at: str = ""
    assets: tuple[ReleaseAsset, ...] = ()

    def asset(self, name: str) -> ReleaseAsset | None:
        return next((asset for asset in self.assets if asset.name == name), None)


def upstream_repo(lock: pins.RuntimeLock) -> str:
    """`owner/name` of the release repository, from the lock itself. Never a hard-coded fork."""
    if lock.repo:
        return lock.repo
    raise UpdateUnavailableError(
        f"E_UPDATE_UNAVAILABLE: {lock.source_path.name} names no upstream release repository "
        f"(`llama_cpp.repo`); an update has nowhere to fetch a newer official bundle from — "
        f"reinstall typed-gguf or repair the lock")


def releases_url(lock: pins.RuntimeLock, *, tag: str | None = None,
                 limit: int = RELEASES_LIMIT) -> str:
    """The API URL for the release list, or for one tag (SPEC 2.8 step 2)."""
    base = RELEASES_URL.format(repo=upstream_repo(lock))
    return f"{base}/tags/{tag}" if tag else f"{base}?per_page={limit}"


def _release_asset(raw: Mapping[str, Any]) -> ReleaseAsset:
    digest = raw.get("digest")
    sha256 = None
    if isinstance(digest, str) and digest.startswith("sha256:"):
        sha256 = digest.split(":", 1)[1] or None
    try:
        size = int(raw["size"]) if raw.get("size") is not None else None
    except (TypeError, ValueError):
        size = None
    return ReleaseAsset(name=str(raw.get("name") or ""), size=size, sha256=sha256)


def parse_releases(payload: Any) -> list[Release]:
    """The API's answer (a list, or one release object) as typed releases, in the API's order."""
    entries = payload if isinstance(payload, list) else [payload]
    releases: list[Release] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or not entry.get("tag_name"):
            continue
        assets = tuple(_release_asset(raw) for raw in entry.get("assets") or ()
                       if isinstance(raw, Mapping))
        releases.append(Release(tag=str(entry["tag_name"]),
                                published_at=str(entry.get("published_at") or ""),
                                assets=assets))
    return releases


def fetch_releases(lock: pins.RuntimeLock, *, tag: str | None = None,
                   limit: int = RELEASES_LIMIT, timeout: float = REQUEST_TIMEOUT,
                   urlopen: Callable[..., Any] | None = None) -> list[Release]:
    """`GET api.github.com/…/releases` — plain HTTPS, no token, typed failure on anything else.

    The broad `except` is scoped to the *wire*: a release list we cannot fetch is
    `E_DOWNLOAD_FAILED` naming the URL (SPEC 2.8 step 1's offline answer), whatever the transport
    raised. Parsing is deliberately outside that block, so a bug in *our* reading of a body the
    server did answer cannot hide behind a download failure.
    """
    url = releases_url(lock, tag=tag, limit=limit)
    request = urllib.request.Request(  # noqa: S310 - a fixed https URL from the lock
        url, headers={"Accept": ACCEPT, "User-Agent": USER_AGENT})
    opener = urlopen or urllib.request.urlopen
    try:
        with opener(request, timeout=timeout) as response:  # noqa: S310
            body = response.read()
    except Exception as exc:  # noqa: BLE001 - any transport failure *is* the download failure
        raise DownloadError(
            f"E_DOWNLOAD_FAILED: cannot reach {url} "
            f"({exc.__class__.__name__}: {str(exc)[:200]}); nothing was changed") from exc
    try:
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise DownloadError(
            f"E_DOWNLOAD_FAILED: {url} answered no usable JSON "
            f"({exc.__class__.__name__}: {str(exc)[:200]}); nothing was changed") from exc
    return parse_releases(payload)


DEFAULT_FETCH = fetch_releases


# --------------------------------------------------------------------------- the target rules
def installed_variant(current: Mapping[str, Any]) -> str | None:
    """The variant of the bundle the record names, when the record names the active runtime.

    SPEC 2.8 step 2's default target is `init`'s own decision, not a fresh detection: `runtime
    update` **maintains** the bundle `init` installed, so on a box whose ladder fell back to
    vulkan/cpu the default update targets that same bundle (`--backend` is how a backend *switch*
    is asked for). The record is not a guess about the host — it is what is installed, and it is
    only read when it names `find_runtime`'s own directory: a record about some other bundle (or no
    record at all) says nothing about this runtime, and detection answers as before.
    """
    record = current.get("record")
    if not isinstance(record, Mapping) or not _same_dir(record.get("dir"), current["dir"]):
        return None
    variant = record.get("variant")
    return str(variant) if variant else None


def retag_asset_name(name: str, tag: str, *, pinned_tag: str) -> str | None:
    """The pinned asset name with the pinned tag swapped for `tag` (SPEC 2.8 step 2).

    `None` when the pinned name does not carry the pinned tag: the naming rule did not hold for
    this lock, and an update never guesses one.
    """
    if not pinned_tag or pinned_tag not in name:
        return None
    return name.replace(pinned_tag, tag)


def build_of_tag(tag: str) -> int | None:
    """`b11160` -> 11160; a tag that is not a build tag (`v0.5.0`) -> None.

    The upstream release stream is not only builds — milestones are tagged differently — and a
    target whose tag carries no build number cannot answer SPEC 2.2's `build >= minimum` rule, so
    it is reported as unknown instead of being guessed out of the digits.
    """
    match = re.fullmatch(r"b(?P<build>\d+)", (tag or "").strip())
    return int(match.group("build")) if match else None


def pick_target(releases: Sequence[Release], *, pinned_name: str, pinned_tag: str,
                tag: str | None = None) -> tuple[Release, ReleaseAsset]:
    """The newest release carrying this host's retagged bundle name (SPEC 2.8 step 2).

    `releases` arrives in the API's own order (newest first), and `type=all`-style milestone and
    nightly releases are simply skipped: the name match is the whole rule. `updated: false` is the
    caller's own comparison — a target equal to the current tag is a legitimately empty update.
    """
    for release in releases:
        if tag is not None and release.tag != tag:
            continue
        wanted = retag_asset_name(pinned_name, release.tag, pinned_tag=pinned_tag)
        if wanted is None:
            continue
        found = release.asset(wanted)
        if found is not None:
            return release, found
    named = f"tag {tag}" if tag else f"the newest release retagging {pinned_tag}"
    raise UpdateUnavailableError(
        f"E_UPDATE_UNAVAILABLE: {named} carries no bundle under this host's pinned asset name "
        f"({pinned_name}); upstream renamed, retagged or dropped the asset for this host — "
        f"typed-gguf never guesses a name")


# --------------------------------------------------------------------------- the current rung
def _same_dir(left: Any, right: Any) -> bool:
    """Same directory, whatever spelling either side used (the record is data, not a path)."""
    if not left or not right:
        return False
    try:
        return pathlib.Path(str(left)).resolve() == pathlib.Path(str(right)).resolve()
    except OSError:  # pragma: no cover - a path the filesystem cannot even resolve
        return str(left) == str(right)


def variant_of_dir(directory: pathlib.Path, tag: str | None) -> str | None:
    """`<home>/runtime/b11026-linux-x64-vulkan` -> `linux-x64-vulkan` (given the tag)."""
    if not tag:
        return None
    prefix = f"{tag}-"
    return directory.name[len(prefix):] if directory.name.startswith(prefix) else None


def current_runtime(home: pathlib.Path | None = None) -> dict[str, Any]:
    """The runtime `update` would move away from, and what `rollback` would return to.

    Rung 1 is a refusal, not a target (SPEC 2.7: `$TYPED_GGUF_RUNTIME_DIR` is consumed read-only);
    nothing installed is `E_RUNTIME_MISSING` (the fix is `init`). The tag and the build come from
    the record **when the record names that directory** — otherwise the directory is a runtime the
    record does not know about, and only its build can be read from it.
    """
    home = home or store.data_home()
    managed = os.environ.get(ENV_RUNTIME_DIR)
    if managed:
        raise UpdateUnavailableError(
            f"E_UPDATE_UNAVAILABLE: TYPED_GGUF_RUNTIME_DIR={managed} is managed outside "
            f"typed-gguf (it is consumed read-only); unset it to update the runtime "
            f"`typed-gguf init` installed")
    found = finder.find_runtime(home=home)
    if found is None:
        raise RuntimeMissingError(
            f"E_RUNTIME_MISSING: no llama.cpp runtime installed under "
            f"{store.runtime_root(home)}; run `typed-gguf init` first (it installs the pinned "
            f"bundle, no compiler needed)")
    record = finder.runtime_record(home) or {}
    known = _same_dir(record.get("dir"), found)
    tag = str(record["tag"]) if known and record.get("tag") else None
    build = record.get("build") if known else None
    if build is None:
        build = capability.build_number(found)
    variant = str(record["variant"]) if known and record.get("variant") else variant_of_dir(
        found, tag)
    return {"dir": str(found), "tag": tag, "build": build, "variant": variant, "record": record,
            "installed_at": record.get("installed_at") if known else None}


# --------------------------------------------------------------------------- the staged probe
def _require_probe(result: capability.ProbeResult, staged: pathlib.Path, *,
                   lock: pins.RuntimeLock, backend: str) -> None:
    """Refuse a staged bundle the lock probe does not bless (SPEC 2.8 step 3 + §2.2's arch rule).

    Every verdict is a *typed* code with the cause in the message, in the order the lock checks
    them: the files, then what the probe found (symbols, build, backends), then the architecture
    gate. There is no fallback ladder — a bundle whose asked-for backend does not load on this
    host aborts the update instead of quietly changing the backend.
    """
    arch = lock.default_model.arch
    if result.missing_files:
        raise RuntimeSymbolsError(
            f"E_RUNTIME_SYMBOLS: {staged} is not a complete runtime bundle (missing "
            f"{', '.join(result.missing_files)}); the upstream layout changed")
    if result.child_error:
        raise RuntimeSymbolsError(
            f"E_RUNTIME_SYMBOLS: the isolated probe could not verify {staged} "
            f"({result.child_error}); an unverifiable bundle is never installed")
    if result.error:
        raise RuntimeSymbolsError(f"E_RUNTIME_SYMBOLS: {result.error}")
    if result.missing_symbols:
        raise RuntimeSymbolsError(
            f"E_RUNTIME_SYMBOLS: {len(result.missing_symbols)} of the required symbols of "
            f"{staged} do not resolve: " + ", ".join(result.missing_symbols[:8])
            + ("…" if len(result.missing_symbols) > 8 else ""))
    if result.build is None:
        raise RuntimeBuildOldError(
            f"E_RUNTIME_BUILD_OLD: cannot determine the build number of {staged}, so the "
            f"{arch} rule (build >= b{result.min_build}) cannot be checked")
    if result.min_build and result.build < result.min_build:
        raise RuntimeBuildOldError(
            f"E_RUNTIME_BUILD_OLD: {staged} is build b{result.build}, older than the pinned "
            f"minimum b{result.min_build} that architecture {arch} needs")
    if not result.usable(backend):
        raise RuntimeSymbolsError(
            f"E_RUNTIME_SYMBOLS: the {backend} backend of {staged} does not load on this host "
            f"({result.backend_errors.get(backend, 'not in the bundle')}); reinstall the system "
            f"libraries it needs or update with --backend for the one this host can drive")
    capability.require_arch(staged, arch, build=result.build, lock=lock)


# --------------------------------------------------------------------------- the switch
def _stop_host(home: pathlib.Path, *, client: Any = None) -> dict[str, Any]:
    """`keep stop` (drain) before the switch — and a refusal when the host would not stop.

    SPEC 2.12: *"a process that dlopen'd the old libraries must never keep answering after a new
    build is installed"*. A record that turns out to be debris (nothing listening, nothing
    signalled) is not a refusal; a pid that is still alive after the stop attempt is.
    """
    client = client or keep_client.Client(home=home)
    report = client.stop(drain=True)
    pid = report.get("pid")
    if pid and pid != os.getpid() and not report.get("stopped") and pid_alive(int(pid)):
        raise UpdateUnavailableError(
            f"E_UPDATE_UNAVAILABLE: the warm host (pid {pid}) would not stop "
            f"({report.get('reason')}); a process that loaded the old runtime must not survive "
            f"the switch — run `typed-gguf keep stop` and run this again")
    return report


def _previous_block(current: Mapping[str, Any]) -> dict[str, Any]:
    return {"dir": current["dir"], "tag": current["tag"], "build": current["build"],
            "installed_at": current.get("installed_at")}


def _from_payload(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {"tag": entry.get("tag"), "build": entry.get("build"), "dir": entry.get("dir"),
            "variant": entry.get("variant")}


def _asset_payload(plan: install.InstallPlan, release: Release,
                   asset: ReleaseAsset) -> dict[str, Any]:
    size = asset.size or 0
    return {"name": asset.name, "size": size, "size_human": hf.human_bytes(size),
            "sha256": asset.sha256, "url": plan.url, "published_at": release.published_at}


#: The probe warning an update itself answers: the live build differs from the *pinned* tag.
PIN_WARNING = "differs from the pinned"


def _updated_warnings(warnings: Sequence[str], *, tag: str, from_tag: str | None) -> list[str]:
    """The probe's warnings, with the pin-difference line answered instead of repeated.

    `ProbeResult.warnings()` compares the live build against the pinned tag and advises
    `typed-gguf init --force`. On a runtime the user *asked* to move that remedy points backwards,
    and `doctor`/`version` print the record's warnings — so the update replaces the line with the
    move itself. Every other warning (a backend that did not load, a skipped probe) is kept as-is.
    """
    kept = [line for line in warnings if PIN_WARNING not in line]
    if len(kept) == len(warnings):
        return kept
    moved = f" from {from_tag}" if from_tag else ""
    kept.append(f"this runtime was updated{moved} to {tag}; the committed oracle numbers were "
                f"measured on the pinned bundle (`typed-gguf runtime rollback` returns to it)")
    return kept


def _retarget_tools(tools: Mapping[str, Any], root: pathlib.Path) -> dict[str, str]:
    """Point the probe's tool paths at where the bundle ended up.

    The probe runs on the staging directory (`<runtime>/.pending-<asset>`), which the switch then
    moves; a record that kept those paths would name a directory that no longer exists — `doctor`
    and `version` read this map, not the probe.
    """
    return {name: str(root / pathlib.Path(str(path)).name) for name, path in tools.items()}


def _probe_payload(result: capability.ProbeResult, backend: str, *,
                   root: pathlib.Path | None = None) -> dict[str, Any]:
    tools = _retarget_tools(result.tools, root) if root is not None else dict(result.tools)
    return {"build": result.build, "tools": tools,
            "symbols_ok": bool(result.symbols_checked and not result.missing_symbols),
            "symbols_probed": bool(result.symbols_checked),
            "missing_symbols": list(result.missing_symbols),
            "backends": list(result.backends), "expect_backend": backend}


def _stage(plan: install.InstallPlan, *, home: pathlib.Path, lock: pins.RuntimeLock,
           url: str | None, progress: hf.Progress | None,
           free_bytes: int | None) -> tuple[pathlib.Path, str, dict[str, Any]]:
    """Download + verify + extract into the staging directory (SPEC 2.8 step 3).

    `install._obtain_archive` is `init`'s own download leg (resume, size, SHA-256) and
    `install.extract_bundle`/`install._flatten_bundle` are its own extract leg (the `data` tar
    filter, the single top-level `llama-<tag>/` flatten) — the SPEC asks for exactly those
    semantics, so they are called, not re-implemented.
    """
    home.mkdir(parents=True, exist_ok=True)
    staging = plan.dest.parent / f".pending-{plan.asset}"
    try:
        hf.check_disk_space(home, plan.required_bytes, free_bytes=free_bytes)
        archive = store.downloads_dir(home) / plan.asset
        try:
            source, stats = install._obtain_archive(plan, archive, url=url, progress=progress)
        except OSError as exc:  # a `file://` source that is not there, a vanished cache dir
            raise DownloadError(
                f"E_DOWNLOAD_FAILED: cannot fetch {url or plan.url} "
                f"({exc.__class__.__name__}: {exc}); nothing was changed") from exc
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        install.extract_bundle(archive, staging)
        install._flatten_bundle(staging)
        missing = [name for name in lock.required_files if not (staging / name).exists()]
        if missing:
            raise RuntimeSymbolsError(
                f"E_RUNTIME_SYMBOLS: {plan.asset} does not contain {', '.join(missing)}; the "
                f"upstream bundle layout changed")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return staging, source, stats


# --------------------------------------------------------------------------- update
def update(*, check: bool = False, tag: str | None = None, backend: str = "auto",
           home: pathlib.Path | None = None, lock: pins.RuntimeLock | None = None,
           url: str | None = None, free_bytes: int | None = None,
           progress: hf.Progress | None = None, deep: bool | None = None,
           system: str | None = None, machine: str | None = None,
           probes: pins.HostProbes | None = None, releases: Sequence[Release] | None = None,
           fetch: Callable[..., list[Release]] | None = None, client: Any = None,
           probe: Callable[..., capability.ProbeResult] | None = None,
           limit: int = RELEASES_LIMIT, timeout: float = REQUEST_TIMEOUT) -> dict[str, Any]:
    """`typed-gguf runtime update` (SPEC 2.8). Returns the `--json` payload.

    `check=True` stops after resolve + target and touches nothing (it is also the `--dry-run`
    form: same plan, same output). `url`/`releases`/`fetch`/`client`/`probe`/`free_bytes` are the
    seams the offline gates drive instead of a socket, a GitHub release list, a live keep host and
    a real dlopen.
    """
    home = home or store.data_home()
    lock = lock or pins.load_lock()
    current = current_runtime(home)
    # Maintain, do not re-decide: without an explicit `--backend`, the target variant is the one
    # `init` installed (SPEC 2.8 step 2) — a box whose ladder fell back to vulkan/cpu must not have
    # its default update swing to the bundle detection would pick today (M2b, card t_ba767a2b).
    variant = ((installed_variant(current) if backend == "auto" else None)
               or pins.host_variant(backend, system=system, machine=machine, probes=probes))
    pinned = pins.asset_for(lock, variant)
    if retag_asset_name(pinned.asset, lock.tag, pinned_tag=lock.tag) is None:
        raise UpdateUnavailableError(
            f"E_UPDATE_UNAVAILABLE: the pinned {variant} asset name ({pinned.asset}) does not "
            f"carry the pinned tag {lock.tag}, so there is no naming rule to re-tag for a newer "
            f"release — typed-gguf never guesses one")
    if releases is None:
        releases = (fetch or DEFAULT_FETCH)(lock, tag=tag, limit=limit, timeout=timeout)
    release, asset = pick_target(releases, pinned_name=pinned.asset, pinned_tag=lock.tag, tag=tag)
    if asset.size is None:
        raise DownloadError(
            f"E_DOWNLOAD_FAILED: {releases_url(lock, tag=release.tag)} carries no size for "
            f"{asset.name}, so the download cannot be verified; nothing was changed")
    target_lock = dataclasses.replace(
        lock, tag=release.tag,
        assets={**lock.assets,
                variant: pins.Asset(variant=variant, asset=asset.name, size=asset.size,
                                    sha256=asset.sha256)})
    plan = install.plan_install(variant, home=home, lock=target_lock)
    target = {"tag": release.tag, "build": build_of_tag(release.tag), "dir": str(plan.dest),
              "variant": variant}
    payload: dict[str, Any] = {
        "schema": SCHEMA, "check": bool(check), "updated": False, "reason": None,
        "from": _from_payload(current), "to": target, "asset": _asset_payload(plan, release, asset),
        "probe": None, "previous": None, "host_stopped": None, "home": str(home)}
    if release.tag == current["tag"]:
        payload["reason"] = f"already at {release.tag}"
        return payload
    if check:
        return payload

    # `init`'s own pre-flight, asked *before* the download it is about (install.py:331): a bundle
    # whose system libraries this host cannot load cannot work here, and finding that out after the
    # fact costs the whole bundle per attempt (the reviewer measured 169.49 MB for one refusal). A
    # **refusal**, not a downgrade — SPEC 2.8's "no fallback ladder" stays intact, and `--backend`
    # (or `doctor`) is what names the bundle this box can actually drive. `--check` above is the
    # read-only plan report (SPEC A-E5-6) and stays exactly as it was.
    refused = install._preflight_reason(plan, target_lock)
    if refused:
        raise RuntimeSymbolsError(f"E_RUNTIME_SYMBOLS: {refused[1]}; nothing was changed")

    deep_probe = capability.deep_probe_enabled() if deep is None else deep
    expect_backend = pins.accelerator_of(variant)
    adopt = plan.dest.exists() and not _same_dir(plan.dest, current["dir"])
    debris: pathlib.Path | None = None
    try:
        if adopt:
            staged, source, stats = plan.dest, ADOPTED_SOURCE, {}
        else:
            debris, source, stats = _stage(plan, home=home, lock=lock, url=url,
                                           progress=progress, free_bytes=free_bytes)
            staged = debris
        result = (probe or capability.probe_runtime)(
            staged, deep=deep_probe, lock=lock, expect_backend=expect_backend, run_tools=True,
            system=system)
        _require_probe(result, staged, lock=lock, backend=expect_backend)
        stopped = _stop_host(home, client=client)
        if debris is not None:
            os.replace(debris, plan.dest)
            debris = None
        record = install._build_record(
            plan, lock=target_lock, probe=result, source=source, stats=stats, warmup_ms=None,
            warmup_error=None, model_path=(current["record"] or {}).get("warmup_model"), url=None,
            requested_backend=expect_backend, attempts=[])
        record[PREVIOUS] = _previous_block(current)
        record["tools"] = _retarget_tools(record.get("tools") or {}, plan.dest)
        record["probe_warnings"] = _updated_warnings(record.get("probe_warnings") or [],
                                                     tag=release.tag, from_tag=current["tag"])
        record["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        record["update_from"] = {"dir": current["dir"], "tag": current["tag"],
                                 "build": current["build"]}
        finder.write_runtime_record(record, home)
    except Exception:
        if debris is not None:
            shutil.rmtree(debris, ignore_errors=True)
        raise
    payload["updated"] = True
    payload["to"] = {**target, "build": result.build}
    payload["probe"] = _probe_payload(result, expect_backend, root=plan.dest)
    payload["previous"] = record[PREVIOUS]
    payload["host_stopped"] = stopped
    return payload


# --------------------------------------------------------------------------- rollback
def rollback(*, home: pathlib.Path | None = None, client: Any = None) -> dict[str, Any]:
    """`typed-gguf runtime rollback`: flip the record back to the retained `previous` bundle.

    One atomic record write, no download, and no probe of the old bundle beyond its presence (SPEC
    2.8's tail) — plus the hash of its `libllama`, so the record still answers `doctor`'s
    recorded-SHA check. The bundle the update installed stays on disk (nothing is ever deleted),
    so a later `update` adopts it instead of downloading it again.

    The record that moves back is the **whole** record, not a rebuilt one (P1, card t_16067777):
    `update` writes the probe's own facts (`backends`, `symbols_*`, the asset block) and SPEC 2.8's
    "all existing record keys stay" describes the *runtime*, not the verb — a rollback that dropped
    them left `typed-gguf version` printing `backends unknown` about a bundle that had just been
    probed. Only the keys that describe the **active** bundle are rewritten (its dir/tag/build/
    variant, its `libllama` hash, the `tools` map, which names files *inside* that directory, and
    `previous`/`rolled_back_*`); the asset and update timestamps stay as the record of what the
    last install did.
    """
    home = home or store.data_home()
    record = finder.runtime_record(home) or {}
    previous = record.get(PREVIOUS)
    if not isinstance(previous, Mapping) or not previous.get("dir"):
        raise UpdateUnavailableError(
            f"E_UPDATE_UNAVAILABLE: {home / 'runtime.json'} records no {PREVIOUS} bundle to roll "
            f"back to (nothing has replaced the bundle `typed-gguf init` installed); nothing "
            f"changed")
    target = pathlib.Path(str(previous["dir"]))
    library = finder.library_names()["llama"]
    if not (target / library).exists():
        raise UpdateUnavailableError(
            f"E_UPDATE_UNAVAILABLE: the retained bundle {target} is gone (no {library} in it), so "
            f"there is nothing to roll back to; `typed-gguf init` reinstalls the pinned bundle")
    stopped = _stop_host(home, client=client)
    new_record: dict[str, Any] = {
        **record,
        "schema": finder.RUNTIME_RECORD_SCHEMA,
        "dir": str(target),
        "tag": previous.get("tag"),
        "build": previous.get("build"),
        "variant": variant_of_dir(target, previous.get("tag")),
        "installed_at": previous.get("installed_at"),
        "libllama_sha256": sha256_file(target / library),
        "rolled_back_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rolled_back_from": {"dir": record.get("dir"), "tag": record.get("tag"),
                             "build": record.get("build")},
        PREVIOUS: None,
    }
    if isinstance(record.get("tools"), Mapping):
        # the probe ran on the bundle the update moved in: its tool paths name files there, and
        # `doctor`/`version` read this map — point it at the bundle this record now names
        new_record["tools"] = _retarget_tools(record["tools"], target)
    finder.write_runtime_record(new_record, home)
    return {"schema": ROLLBACK_SCHEMA, "rolled_back": True,
            "from": _from_payload({"tag": record.get("tag"), "build": record.get("build"),
                                   "dir": record.get("dir"),
                                   "variant": record.get("variant")}),
            "to": _from_payload({"tag": previous.get("tag"), "build": previous.get("build"),
                                 "dir": str(target),
                                 "variant": new_record["variant"]}),
            "previous": None, "host_stopped": stopped, "home": str(home)}
