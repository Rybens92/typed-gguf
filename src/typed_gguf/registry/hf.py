"""HuggingFace resolve/search/download with resume + SHA-256 verify

Milestone: E1a.

SPEC 2.7: download exactly the selected quant file, verify against the tree
API lfs.oid, publish atomically via os.replace.

Offline story: the default model has a committed metadata snapshot
(`docs/evidence/hf_spark_x2_5.json`), so `TYPED_GGUF_OFFLINE=1` (or any offline unit test) can
resolve the pinned repo, its files, sizes and LFS oids without a network. Everything else
fails loudly with `E_DOWNLOAD_FAILED`/`E_HF_AUTH_REQUIRED` instead of guessing.

Auth (coordinator addition 1): `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HUGGINGFACE_HUB_TOKEN`
or the local `~/.cache/huggingface/token`; a 401/403 becomes `E_HF_AUTH_REQUIRED` with the
exact fix in the message.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from typed_gguf import __version__
from typed_gguf.errors import DownloadError, HfAuthError, InsufficientDiskError, Sha256MismatchError
from typed_gguf.registry.gguf import sha256_file

HF_HOST = "https://huggingface.co"
TOKEN_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")
DEFAULT_REPO = "XHToken/Spark-X2.5-4B-GGUF"
SNAPSHOT_NAME = "hf_spark_x2_5.json"
CHUNK = 1 << 20
#: The version is the package's, never a literal (P2, card t_16067777: it froze at `0.1` while
#: `runtime.update`'s release-API UA sent the real one — two UAs, one run, one of them lying).
USER_AGENT = f"typed-gguf/{__version__} (+https://github.com/Rybens92/typed-gguf)"

#: Which product a leg talks to. This module is HuggingFace's, but it also *is* the transport the
#: GitHub asset download borrows (`install._fetch`), and a failure has to name the host it came
#: from: the E2E's U5 leg read `HuggingFace returned HTTP 404` about a `github.com` URL.
HUGGINGFACE = "HuggingFace"
GITHUB = "GitHub"
#: The host a failure names, per product (`_translate`) — the message never guesses from the URL.
PRODUCT_HOSTS = {HUGGINGFACE: "huggingface.co", GITHUB: "github.com"}

Progress = Callable[[int, int | None], None]


# --------------------------------------------------------------------- models
@dataclass(frozen=True)
class FileInfo:
    path: str
    size: int | None = None
    oid: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "oid": self.oid}


@dataclass(frozen=True)
class ModelInfo:
    repo: str
    sha: str | None
    gated: bool
    license: str | None
    files: tuple[FileInfo, ...]
    tags: tuple[str, ...] = ()
    private: bool = False
    source: str = "api"

    def to_dict(self) -> dict[str, Any]:
        return {"repo": self.repo, "sha": self.sha, "gated": self.gated,
                "license": self.license, "private": self.private, "source": self.source,
                "tags": list(self.tags), "files": [f.to_dict() for f in self.files]}


# --------------------------------------------------------------------- auth
def token(home: pathlib.Path | None = None) -> str | None:
    for var in TOKEN_ENV:
        value = os.environ.get(var)
        if value and value.strip():
            return value.strip()
    for path in token_paths(home):
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value
    return None


def token_paths(home: pathlib.Path | None = None) -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        paths.append(pathlib.Path(os.path.expanduser(hf_home)) / "token")
    base = pathlib.Path(home or os.path.expanduser("~"))
    paths.append(base / ".cache" / "huggingface" / "token")
    return paths


def _headers(extra_token: str | None = None) -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT}
    bearer = extra_token or token()
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    return headers


# --------------------------------------------------------------------- http
def _open(url: str, headers: dict[str, str], timeout: float = 60.0):
    request = urllib.request.Request(url, headers=headers)  # noqa: S310
    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310


def fetch_json(url: str, *, token: str | None = None, timeout: float = 60.0) -> Any:
    with _open(url, _headers(token), timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _translate(exc: Exception, repo: str, *, product: str = HUGGINGFACE) -> Exception:
    """The typed error for a transport failure, **naming the product it came from**.

    `product` is the calling leg's own fact, not a guess read off the URL: this module's
    HuggingFace calls keep the default, and `install`'s bundle download — a GitHub release asset —
    passes `GITHUB`, so the message it raises names the host that answered (P2, card t_16067777).
    A 401/403 is only HuggingFace's gating: a GitHub asset needs no token, so it takes the generic
    wording instead of advice about `HF_TOKEN`.
    """
    if isinstance(exc, urllib.error.HTTPError):
        if product is HUGGINGFACE and exc.code in (401, 403):
            return HfAuthError(
                f"E_HF_AUTH_REQUIRED: {repo} is gated or private (HTTP {exc.code}); export "
                f"HF_TOKEN=<token with read access> (HUGGING_FACE_HUB_TOKEN and "
                f"~/.cache/huggingface/token are honoured too) and retry")
        return DownloadError(
            f"E_DOWNLOAD_FAILED: {repo}: {product} returned HTTP {exc.code} ({exc.reason})")
    if isinstance(exc, (urllib.error.URLError, TimeoutError, OSError)):
        return DownloadError(
            f"E_DOWNLOAD_FAILED: {repo}: cannot reach {PRODUCT_HOSTS.get(product, product)} "
            f"({exc}); this tool is offline-capable only for the pinned default model "
            f"(TYPED_GGUF_OFFLINE=1)")
    return DownloadError(f"E_DOWNLOAD_FAILED: {repo}: {exc}")


def offline_enabled() -> bool:
    return os.environ.get("TYPED_GGUF_OFFLINE", "0") not in ("0", "", "false", "no")


def snapshot_path(repo: str | None = None) -> pathlib.Path | None:
    """The committed metadata snapshot for the pinned default repo, if this is a checkout."""
    override = os.environ.get("TYPED_GGUF_SNAPSHOT")
    candidates: list[pathlib.Path] = []
    if override:
        candidates.append(pathlib.Path(override).expanduser())
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / "docs" / "evidence" / SNAPSHOT_NAME)
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if repo and payload.get("repo") != repo:
            continue
        return candidate
    return None


def model_info_from_snapshot(repo: str, path: pathlib.Path) -> ModelInfo:
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = tuple(FileInfo(path=str(f["path"]), size=f.get("size"),
                           oid=f.get("lfs_oid_sha256") or f.get("oid"))
                  for f in payload.get("files", []))
    return ModelInfo(repo=payload.get("repo", repo), sha=payload.get("repo_sha"),
                     gated=bool(payload.get("gated", False)), license=payload.get("license"),
                     files=files, tags=tuple(payload.get("tags", [])), source="snapshot")


def repo_files(repo: str, *, revision: str | None = None, token: str | None = None,
               offline: bool = False) -> tuple[ModelInfo, list[FileInfo]]:
    revision = revision or "main"
    if offline:
        path = snapshot_path(repo)
        if path is None:
            raise DownloadError(
                f"E_DOWNLOAD_FAILED: {repo}: offline mode has no committed snapshot for this "
                f"repo (only the pinned default {DEFAULT_REPO} is snapshotted)")
        info = model_info_from_snapshot(repo, path)
        return info, list(info.files)
    url = f"{HF_HOST}/api/models/{repo}/tree/{urllib.parse.quote(revision)}?recursive=1"
    try:
        tree = fetch_json(url, token=token)
    except Exception as exc:  # noqa: BLE001 - translated below
        raise _translate(exc, repo) from exc
    files = [FileInfo(path=entry["path"], size=entry.get("size"),
                      oid=(entry.get("lfs") or {}).get("oid"))
             for entry in tree if entry.get("type") == "file"]
    return ModelInfo(repo=repo, sha=None, gated=False, license=None, files=tuple(files)), files


def model_info(repo: str, *, revision: str | None = None, token: str | None = None,
               offline: bool | None = None) -> ModelInfo:
    """Repo metadata (sha, gated, license) + the recursive file list."""
    offline = offline_enabled() if offline is None else offline
    if offline:
        path = snapshot_path(repo)
        if path is None:
            raise DownloadError(
                f"E_DOWNLOAD_FAILED: {repo}: offline mode has no committed snapshot for this "
                f"repo (only the pinned default {DEFAULT_REPO} is snapshotted)")
        return model_info_from_snapshot(repo, path)
    url = f"{HF_HOST}/api/models/{repo}"
    try:
        meta = fetch_json(url, token=token)
    except Exception as exc:  # noqa: BLE001 - translated below
        raise _translate(exc, repo) from exc
    files = list(repo_files(repo, revision=revision, token=token)[1])
    return ModelInfo(repo=meta.get("repo", repo) or repo, sha=meta.get("sha"),
                     gated=bool(meta.get("gated", False)),
                     license=(meta.get("cardData") or {}).get("license"),
                     files=tuple(files), tags=tuple(meta.get("tags", [])),
                     private=bool(meta.get("private", False)), source="api")


def search(query: str, *, limit: int = 20, token: str | None = None) -> list[dict[str, Any]]:
    url = (f"{HF_HOST}/api/models?search={urllib.parse.quote(query)}"
           f"&filter=gguf&limit={int(limit)}&sort=downloads&direction=-1")
    try:
        payload = fetch_json(url, token=token)
    except Exception as exc:  # noqa: BLE001 - translated below
        raise _translate(exc, query) from exc
    out: list[dict[str, Any]] = []
    for entry in payload if isinstance(payload, list) else []:
        repo_id = entry.get("id") or entry.get("modelId") or ""
        tags = [str(t) for t in entry.get("tags", [])]
        if "gguf" in {t.lower() for t in tags} or repo_id.lower().endswith("-gguf"):
            out.append({"id": repo_id, "downloads": entry.get("downloads"),
                        "likes": entry.get("likes"), "tags": tags,
                        "private": bool(entry.get("private", False))})
    return out


# --------------------------------------------------------------------- bytes
def human_bytes(count: int | float) -> str:
    value = float(count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1000.0
    return f"{value:.2f} TB"


def _existing_ancestor(path: pathlib.Path) -> pathlib.Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def check_disk_space(dest_dir: str | os.PathLike[str], required_bytes: int, *,
                     free_bytes: int | None = None) -> int:
    """Refuse to start a download that cannot finish (coordinator addition 2)."""
    target = _existing_ancestor(pathlib.Path(dest_dir))
    free = int(free_bytes) if free_bytes is not None else shutil.disk_usage(str(target)).free
    if required_bytes > free:
        raise InsufficientDiskError(
            f"E_INSUFFICIENT_DISK: need {required_bytes} bytes ({human_bytes(required_bytes)}) "
            f"free under {target}, but only {free} bytes ({human_bytes(free)}) are available; "
            f"free up space or set TYPED_GGUF_HOME to another filesystem")
    return free


# --------------------------------------------------------------------- download
@dataclass(frozen=True)
class DownloadResult:
    path: pathlib.Path
    bytes_fetched: int
    bytes_total: int | None
    resumed_from: int
    sha256: str
    verified: bool
    url: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "bytes_fetched": self.bytes_fetched,
                "bytes_total": self.bytes_total, "resumed_from": self.resumed_from,
                "sha256": self.sha256, "verified": self.verified, "url": self.url}


def resolve_url(repo: str, path: str, revision: str = "main") -> str:
    return f"{HF_HOST}/{repo}/resolve/{urllib.parse.quote(revision)}/{path}"


def download_url(url: str, dest: str | os.PathLike[str], *, size: int | None = None,
                 sha256: str | None = None, token: str | None = None, revision: str | None = None,
                 chunk: int = CHUNK, resume: bool = True, no_verify: bool = False,
                 progress: Progress | None = None, timeout: float = 60.0,
                 product: str = HUGGINGFACE) -> DownloadResult:
    """Range-resumable download into `<dest>.part`, verified, then atomically published.

    `product` is only what a failure is *worded* with (`_translate`), because the transport is
    shared: HuggingFace by default, `GITHUB` for the release assets `install` fetches.
    """
    dest = pathlib.Path(dest)
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fetched = 0
    resumed_from = 0
    total = size
    while True:
        offset = part.stat().st_size if (resume and part.exists()) else 0
        headers = _headers(token)
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            response = _open(url, headers, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and offset:
                part.unlink(missing_ok=True)  # stale part: start over
                continue
            raise _translate(exc, url, product=product) from exc
        except Exception as exc:  # noqa: BLE001 - translated below
            raise _translate(exc, url, product=product) from exc
        with response:
            status = getattr(response, "status", 200)
            content_length = None
            try:
                content_length = int(response.headers.get("Content-Length", ""))
            except (TypeError, ValueError):
                content_length = None
            if total is None and content_length:
                total = content_length + (offset if status == 206 else 0)
            if status != 206 and offset:
                # server ignored the Range: restart from zero
                offset = 0
                resumed_from = 0
            else:
                resumed_from = offset
            mode = "ab" if offset else "wb"
            written = offset
            with open(part, mode) as fh:
                while True:
                    block = response.read(chunk)
                    if not block:
                        break
                    fh.write(block)
                    written += len(block)
                    fetched += len(block)
                    if progress:
                        progress(written, total)
                fh.flush()
                os.fsync(fh.fileno())
        break

    actual = part.stat().st_size
    if size is not None and actual != size:
        raise DownloadError(
            f"E_DOWNLOAD_FAILED: {url}: expected {size} bytes, got {actual} "
            f"(the download is incomplete; re-run to resume)")
    digest = sha256_file(part)
    verified = False
    if sha256:
        if digest != sha256:
            if no_verify:
                pass
            else:
                raise Sha256MismatchError(
                    f"E_SHA256_MISMATCH: {url}: expected {sha256}, got {digest}; the partial "
                    f"file was kept at {part} for inspection")
        else:
            verified = True
    os.replace(part, dest)
    return DownloadResult(path=dest, bytes_fetched=fetched, bytes_total=total,
                          resumed_from=resumed_from, sha256=digest, verified=verified, url=url)


def download_file(repo: str, path: str, dest: str | os.PathLike[str], *,
                  revision: str = "main", size: int | None = None, sha256: str | None = None,
                  token: str | None = None, chunk: int = CHUNK, resume: bool = True,
                  no_verify: bool = False, progress: Progress | None = None) -> DownloadResult:
    """Download `<repo>/<path>` at `<revision>` (a pinned commit sha keeps it reproducible)."""
    return download_url(resolve_url(repo, path, revision), dest, size=size, sha256=sha256,
                        token=token, chunk=chunk, resume=resume, no_verify=no_verify,
                        progress=progress)
