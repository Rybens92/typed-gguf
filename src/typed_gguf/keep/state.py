"""The keep ledger: which host is alive, where its socket is, and how to clean it up.

Milestone: E4 (SPEC 2.12, card t_7e24cea4). The data home grows one small directory:

```
<data-home>/keep/
    host.json          # the live host's record (atomic, 0600) — at most one host, hence one file
    <digest>.sock      # the unix socket the host answers on (0600)
    <digest>.spec.json # what the client asked the host to load (0600; removed when it exits)
    <digest>.log       # the host's own stdout/stderr (the post-mortem of a failed spawn; `keep
                       # stop` takes it with the host — P3, card t_16067777)
```

Two rules keep the ledger from becoming a source of lies:

* **a record is a claim, not a fact** — `alive()` checks the pid *and* the socket, and a record
  that fails either check is `stale`. Nothing is inferred from the file alone;
* **the client cleans up, the host cleans up after itself** — a stale record is removed by the
  next `ask`/`run` (which then spawns a fresh host), by `keep stop` (the cleanup verb) and by the
  host on its way out. `keep status` is read-only: it *reports* `stale` so a wedged host is
  visible instead of being papered over.

Every write is `tmp + fsync + os.replace` (the registry's own rule): a killed writer leaves no
half-written record for the next call to trip over, and `read_record` treats one as "no host".
"""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from typed_gguf.errors import UserError
from typed_gguf.registry import store

#: The record's schema — a version bump here is a deliberate "the old record means nothing".
RECORD_SCHEMA = "typed_gguf.keep/v1"
KEEP_DIR_NAME = "keep"
RECORD_NAME = "host.json"
SPEC_SUFFIX = ".spec.json"
SOCKET_SUFFIX = ".sock"
LOG_SUFFIX = ".log"
#: `sockaddr_un.sun_path` is 108 bytes on Linux *including* the trailing NUL; a path that cannot
#: fit is refused (naming the limit) instead of being truncated into someone else's socket.
SUN_PATH_MAX = 108


# --------------------------------------------------------------------- paths
def data_home_path(home: pathlib.Path | str | None = None) -> pathlib.Path:
    """The data home, coerced: the host reads it back out of a spec file, i.e. as a `str`."""
    return pathlib.Path(home) if home else store.data_home()


def keep_dir(home: pathlib.Path | str | None = None) -> pathlib.Path:
    return data_home_path(home) / KEEP_DIR_NAME


def ensure_dir(home: pathlib.Path | str | None = None) -> pathlib.Path:
    """The keep directory, created 0700: what the host loads is nobody else's business."""
    directory = keep_dir(home)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with contextlib.suppress(OSError):        # an existing dir keeps its mode; fix it if we can
        directory.chmod(0o700)
    return directory


def state_path(home: pathlib.Path | None = None) -> pathlib.Path:
    return keep_dir(home) / RECORD_NAME


def socket_path(home: pathlib.Path | None, digest: str) -> pathlib.Path:
    path = keep_dir(home) / f"{digest}{SOCKET_SUFFIX}"
    if len(str(path).encode("utf-8")) + 1 > SUN_PATH_MAX:
        raise UserError(
            f"the unix socket for this host would not fit in `sun_path` ({SUN_PATH_MAX} bytes "
            f"including the NUL): {path} is too long — set a shorter TYPED_GGUF_HOME",
            code="E_UNKNOWN_KEY")
    return path


def spec_path(home: pathlib.Path | None, digest: str) -> pathlib.Path:
    return keep_dir(home) / f"{digest}{SPEC_SUFFIX}"


def log_path(home: pathlib.Path | None, digest: str) -> pathlib.Path:
    return keep_dir(home) / f"{digest}{LOG_SUFFIX}"


# --------------------------------------------------------------------- the record
@dataclass(frozen=True)
class HostRecord:
    """One live host, as the client that spawned it wrote it down.

    `key` is the full `identity.KeepKey` dict (not just the digest) so `keep status` can say
    *what* the resident model was loaded for; `placement` and `devices` are the host's own
    evidence — the placement the engine log proves, in the shape `runtime.devices` reads back.
    """

    digest: str
    pid: int
    socket: str
    key: dict[str, Any]
    model: str
    model_path: str
    keep_alive: float
    started_at: float
    loaded_at: float
    spec: str
    log: str
    version: str = ""
    #: a client-facing field on purpose: the same record carries "starting", "ready" (serving)
    #: and "failed" (with `error` filled in), which is how a spawn that dies is diagnosed.
    state: str = "ready"
    error: dict[str, Any] | None = None
    requests: int = 0
    last_used: float | None = None
    placement: dict[str, Any] | None = None
    devices: dict[str, Any] | None = None
    model_load_ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = RECORD_SCHEMA
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HostRecord:
        if payload.get("schema") != RECORD_SCHEMA:
            raise ValueError(f"not a {RECORD_SCHEMA} record")
        return cls(
            digest=str(payload["digest"]), pid=int(payload["pid"]),
            socket=str(payload["socket"]), key=dict(payload["key"]),
            model=str(payload["model"]), model_path=str(payload["model_path"]),
            keep_alive=float(payload["keep_alive"]), started_at=float(payload["started_at"]),
            loaded_at=float(payload["loaded_at"]), spec=str(payload["spec"]),
            log=str(payload["log"]), version=str(payload.get("version") or ""),
            state=str(payload.get("state") or "ready"), error=payload.get("error"),
            requests=int(payload.get("requests") or 0),
            last_used=(None if payload.get("last_used") is None
                       else float(payload["last_used"])),
            placement=payload.get("placement"), devices=payload.get("devices"),
            model_load_ms=float(payload.get("model_load_ms") or 0.0),
            extra=dict(payload.get("extra") or {}))

    def replace(self, **changes: Any) -> HostRecord:
        return HostRecord(**{**asdict(self), **changes})


def read_record(home: pathlib.Path | None = None) -> HostRecord | None:
    """The record, or None when there is none / unreadable (a killed writer, a schema bump)."""
    path = state_path(home)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return HostRecord.from_dict(payload)
    except (KeyError, TypeError, ValueError):
        return None


def write_record(record: HostRecord, home: pathlib.Path | None = None) -> pathlib.Path:
    """Write the record atomically and privately (0600). Returns the path it landed on."""
    path = state_path(home)
    ensure_dir(home)
    _write_private(path, json.dumps(record.to_dict(), indent=1, sort_keys=True))
    return path


def clear_record(home: pathlib.Path | None = None, *, digest: str | None = None) -> bool:
    """Remove this host's record, socket and spec. `digest` guards another host's files.

    Returns True when something was removed — the caller uses the answer to decide whether it
    was the one cleaning up after a dead host or looking at a live one it must not touch.
    """
    record = read_record(home)
    if record is None:
        return False
    if digest is not None and record.digest != digest:
        return False
    removed = False
    for target in (state_path(home), pathlib.Path(record.socket), pathlib.Path(record.spec)):
        with contextlib.suppress(OSError):
            target.unlink()
            removed = True
    return removed


def clear_log(home: pathlib.Path | None = None, *, digest: str | None = None) -> list[str]:
    """Remove the ledger's `*.log` files: one host's, or every finished host's (P3, t_16067777).

    The log is the host's own stdout/stderr — `client._log_tail` quotes it while a spawn is
    failing, and the crash path keeps it. `keep stop` is the deliberate *"this host is done"* verb,
    so it takes the file with it and leaves the ledger directory holding nothing (A-E5-5's strict
    reading, and the one thing `keep stop` used to leave behind).

    **Delete, not move:** SPEC 2.7 fixes the data home's path list (`models/`, `registry.json`,
    `runtime/<tag>-<variant>/`, `runtime.json`, `states/`, `calibration.json`) and §2.12 puts the
    ledger in `<home>/keep/`; a second, undocumented log home would be a path the SPEC never names,
    holding content nothing reads once its host is gone.

    With no `digest` this is the ledger's sweep: every `<digest>.log` whose `<digest>.sock` is
    *gone*. A socket still on disk means a host that is up (or on its way up — `bind` happens
    before its record is written), and its log is not debris.
    """
    directory = keep_dir(home)
    if not directory.is_dir():
        return []
    if digest is not None:
        targets = [log_path(home, digest)]
    else:
        targets = [path for path in sorted(directory.glob(f"*{LOG_SUFFIX}"))
                   if not path.with_name(f"{path.name[:-len(LOG_SUFFIX)]}{SOCKET_SUFFIX}").exists()]
    removed: list[str] = []
    for target in targets:
        with contextlib.suppress(OSError):
            target.unlink()
            removed.append(target.name)
    return removed


# --------------------------------------------------------------------- liveness
def pid_alive(pid: int) -> bool:
    """Is `pid` a live process? (Never signal a process group: `pid <= 0` is False.)

    A **defunct** process is not alive: a host that exited is reaped by its parent (or by init),
    and between those two moments `os.kill(pid, 0)` succeeds. `keep status` must not report a
    zombie as a resident model, so the state field of `/proc/<pid>/stat` is read on Linux; other
    platforms fall back to the signal answer.
    """
    if not isinstance(pid, int) or pid <= 0:
        return False
    if _is_zombie(pid):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:                     # someone else's process: it is alive
        return True
    except OSError:
        return False
    return True


def _is_zombie(pid: int) -> bool:
    """`/proc/<pid>/stat` field 3 == "Z" (Linux). False everywhere else, and on any read error."""
    try:
        stat = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    end = stat.rfind(")")
    if end < 0:
        return False
    return stat[end + 1:].strip().split(" ", 1)[0] == "Z"


def alive(record: HostRecord, *, home: pathlib.Path | None = None) -> bool:
    """A host is alive when its process is *and* its socket is on disk (SPEC 2.12 health check).

    Both halves matter: a pid without a socket is a host that died between two calls (the socket
    is unlinked on the way out), and a socket without a pid is the crash a `kill -9` of the host
    leaves behind. Neither is usable, so the next call cleans up and spawns fresh.
    """
    del home                                    # kept for symmetry; the record carries the path
    return pid_alive(record.pid) and pathlib.Path(record.socket).exists()


def wait_pid_gone(pid: int, *, timeout: float = 5.0, interval: float = 0.05) -> bool:
    """Wait for a process to disappear; True when it did (used by `keep stop`)."""
    deadline = time.monotonic() + max(float(timeout), 0.0)
    while pid_alive(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)
    return True


# --------------------------------------------------------------------- the report
def host_status(record: HostRecord, *, now: float | None = None,
                **extra: Any) -> dict[str, Any]:
    """The record's own half of `keep status`: liveness, uptime, and the idle countdown.

    The countdown is measured from the *last use* (falling back to the load for a host that has
    not answered anything yet), which is exactly the number a user wants: how long until this
    model frees the device.
    """
    stamp = time.time() if now is None else float(now)
    last = record.last_used if record.last_used is not None else record.loaded_at
    idle_left = max(float(record.keep_alive) - (stamp - last), 0.0)
    status: dict[str, Any] = {
        "state": "running" if alive(record) else "stale",
        "pid": record.pid,
        "socket": record.socket,
        "spec": record.spec,
        "log": record.log,
        "model": record.model,
        "model_path": record.model_path,
        "key": dict(record.key),
        "key_digest": record.digest,
        "keep_alive_s": round(float(record.keep_alive), 3),
        "uptime_s": round(stamp - record.started_at, 3),
        "loaded_s": round(stamp - record.loaded_at, 3),
        "idle_left_s": round(idle_left, 3),
        "requests": record.requests,
        "model_load_ms": record.model_load_ms,
        "placement": record.placement,
        "devices": record.devices,
        "version": record.version,
        "error": record.error,
    }
    status.update(extra)
    return status


def _write_private(path: pathlib.Path, text: str) -> None:
    """`tmp + fsync + os.replace`, file mode 0600 (the registry's own durability rule).

    The staging name is **per writer** (`mkstemp`, same directory, so the replace stays atomic on
    one filesystem). It used to be `<name>.tmp` for everyone: two cold callers racing one ledger
    target (two simultaneous `ask`s, card t_9249bb0c) then shared one staging file, and the winner's
    `os.replace` removed it under the loser — whose own replace raised ENOENT on a *designed* path
    (the live `E_INTERNAL: FileNotFoundError … .spec.json.tmp -> .spec.json`, exit 4, no inline
    fallback). Both writers of this helper race in the wild: the record (`write_record`) and the
    spec (`keep.host.HostSpec.save`, written before every spawn).
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, staging = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp = pathlib.Path(staging)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    os.replace(tmp, path)
