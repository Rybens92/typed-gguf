"""The server side of the warm host: one model, one socket, one idle countdown (SPEC 2.12).

`Server.serve()` is the whole life of a host process:

1. **load once** — the caller hands it a `load()` callable returning a `Loaded` (the model handle
   plus a `decide(payload) -> response_body` closure). Everything expensive happens here, exactly
   once: the model load, the fit plan, the calibration table. This is the cold half;
2. **bind** — `<data-home>/keep/<digest>.sock`, 0600, after unlinking a socket file a crashed host
   left behind. A *live* host's socket is never stolen: the bind fails and is reported;
3. **write the record** — `keep/host.json`, atomically, with what the engine proved (placement,
   device buffers from the log) and the countdown's own numbers;
4. **answer** — one connection, one JSON line, one decision, one reply. The loop is
   single-threaded *on purpose*: requests are serialized (a queue in the listen backlog), because
   one context and one device are being shared;
5. **exit by itself** — after `keep_alive` seconds without a request (the deadline restarts on
   every answered request), or on `{"op": "stop"}`, or on SIGTERM. The model is freed, the socket
   and the record are removed, and the process ends through `runtime.teardown`'s `os._exit`
   discipline (the CLI half does that part) so a third-party ICD destructor cannot rewrite the
   exit status of a process that ran a decision.

A client that vanishes mid-request is not a problem: the answer is dropped (EPIPE) and the host
carries on with its countdown. That is the whole reason the host is a separate process.
"""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import signal
import socket
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from typed_gguf.errors import TypedGgufError
from typed_gguf.keep import identity, state

#: The wire's own schema tag (a request from another generation is refused, never interpreted).
REQUEST_SCHEMA = "typed_gguf.keep.request/v1"
SPEC_SCHEMA = "typed_gguf.keep.spec/v1"
#: A request line is a JSON document; 1 MiB is far above any state/question set and far below
#: "read whatever the client sends".
MAX_LINE = 1 << 20
#: How often an idle host wakes to check its own deadline and its stop flag. Small enough that a
#: SIGTERM is honoured promptly, large enough to be free.
IDLE_POLL = 0.25
#: The listen backlog: how many clients may queue behind the one being served.
LISTEN_BACKLOG = 8


@dataclass(frozen=True)
class HostSpec:
    """What the client asked the host to be: the key, the paths, and the first request.

    Written to `<data-home>/keep/<digest>.spec.json` (0600) *before* the spawn, so a host that
    cannot even start still leaves behind what it was asked to do. `payload` is the request that
    caused the spawn — the host needs it to resolve the model and the fit plan; later requests
    arrive over the socket and only have to agree with the key.
    """

    key: identity.KeepKey
    digest: str
    model: str
    model_path: str
    keep_alive: float
    spec_path: str
    socket_path: str
    log_path: str
    payload: dict[str, Any]
    fit: dict[str, Any] = field(default_factory=dict)
    home: str | None = None
    version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SPEC_SCHEMA, "key": self.key.to_dict(), "digest": self.digest,
            "model": self.model, "model_path": self.model_path,
            "keep_alive": float(self.keep_alive), "spec_path": self.spec_path,
            "socket_path": self.socket_path, "log_path": self.log_path,
            "payload": self.payload, "fit": dict(self.fit), "home": self.home,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HostSpec:
        return cls(key=identity.KeepKey.from_dict(dict(payload["key"])),
                   digest=str(payload["digest"]), model=str(payload.get("model") or ""),
                   model_path=str(payload["model_path"]),
                   keep_alive=float(payload.get("keep_alive") or 0.0),
                   spec_path=str(payload.get("spec_path") or ""),
                   socket_path=str(payload.get("socket_path") or ""),
                   log_path=str(payload.get("log_path") or ""),
                   payload=dict(payload.get("payload") or {}),
                   fit=dict(payload.get("fit") or {}),
                   home=payload.get("home"), version=str(payload.get("version") or ""))

    def save(self) -> pathlib.Path:
        path = pathlib.Path(self.spec_path)
        state._write_private(path, json.dumps(self.to_dict(), indent=1, sort_keys=True))
        return path


def read_spec(path: str | os.PathLike[str]) -> HostSpec | None:
    """The spec the client wrote, or None when it is missing/unreadable/another generation."""
    try:
        payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != SPEC_SCHEMA:
        return None
    try:
        return HostSpec.from_dict(payload)
    except (KeyError, TypeError, ValueError):
        return None


@dataclass
class Loaded:
    """What stays resident: the model handle, plus everything a decision needs without reloading.

    `decide(payload)` runs one request against the *already loaded* model (the CLI's half of the
    contract: it applies the host's own fit plan and calibration, so nothing on this path can
    re-load, re-plan or re-hash). `placement` / `devices` are the host's evidence — what the engine
    log proved — surfaced verbatim in `keep status` (card t_603a35a / t_80f1a4c6).
    """

    handle: Any
    decide: Callable[[dict[str, Any]], dict[str, Any]]
    model: str
    model_path: str
    placement: dict[str, Any] | None = None
    devices: dict[str, Any] | None = None
    model_load_ms: float = 0.0

    def close(self) -> None:
        closer = getattr(self.handle, "close", None)
        if callable(closer):
            closer()


class Server:
    """One host process: load, bind, answer until idle. `serve()` is the entry point."""

    def __init__(self, spec: HostSpec, *, load: Callable[[], Loaded],
                 clock: Callable[[], float] = time.time,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self.spec = spec
        self.load = load
        self.clock = clock
        self.monotonic = monotonic
        self.sock: socket.socket | None = None
        self.bind_error: str | None = None
        #: True when `bind()` lost the identity to *another* host (not to our own mistake): the
        #: record in the ledger is then somebody else's, and this process must not overwrite it
        #: (card t_9249bb0c: a losing racer's `failed` record made its client clean up the winner).
        self.bind_owned_elsewhere = False
        self.loaded: Loaded | None = None
        self.record: state.HostRecord | None = None
        self.exit_code = 0
        self.requests = 0
        self.last_used: float | None = None
        self.stopping = False
        self.started_at = self.clock()
        self.loaded_at = self.started_at

    # ------------------------------------------------------------------ lifecycle
    def serve(self) -> int:
        """Load, bind, answer, exit. Returns the process exit code the caller must end with."""
        try:
            self.loaded = self.load()
        except BaseException as exc:                 # noqa: BLE001 - the client reads the log
            return self._fail(exc)
        if self.bind() is None:                      # the socket first: a ready record means
            return self._fail(RuntimeError(self.bind_error or "the socket could not be bound"),
                              record=not self.bind_owned_elsewhere)
        self.record = self._write_record()           # "…and it is listening right now"
        self._install_signals()
        deadline = self.monotonic() + max(float(self.spec.keep_alive), 0.0)
        try:
            while not self.stopping:
                remaining = deadline - self.monotonic()
                if remaining <= 0:
                    break
                self.sock.settimeout(min(remaining, IDLE_POLL))
                try:
                    conn, _ = self.sock.accept()
                except TimeoutError:
                    continue                          # the poll: is the deadline up? stop asked?
                except OSError:
                    break
                with conn:
                    op = self._serve_connection(conn)
                if op == "stop":
                    self.stopping = True
                    break
                if op != "closed":                    # a served request restarts the countdown
                    self.last_used = self.clock()
                    deadline = self.monotonic() + max(float(self.spec.keep_alive), 0.0)
                    self.record = self._write_record()
        finally:
            self._cleanup()
        return self.exit_code

    def bind(self) -> socket.socket | None:
        """Bind the identity's socket (0600). A socket a crashed host left is replaced.

        The socket file is *probed* before it is replaced: a listening host owns this identity and
        must not be evicted by a second one (two models resident is exactly what the card
        forbids). Only debris — a path where nothing answers — is unlinked and rebound.
        """
        state.ensure_dir(self.spec.home)
        path = self.spec.socket_path
        if os.path.exists(path):
            if _listening(path):
                self.bind_error = (
                    f"the socket {path} is already in use: a live host owns this identity "
                    f"(digest {self.spec.digest}); stop it first (`typed-gguf keep stop`)")
                self.bind_owned_elsewhere = True
                return None
            with contextlib.suppress(OSError):
                os.unlink(path)                      # debris from a crashed host
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.bind(path)
        except OSError as exc:
            sock.close()
            self.bind_error = (f"the socket {path} is already in use ({exc}); another host owns "
                               f"this identity (digest {self.spec.digest})")
            self.bind_owned_elsewhere = True
            return None
        with contextlib.suppress(OSError):
            os.chmod(path, 0o600)
        sock.listen(LISTEN_BACKLOG)
        self.sock = sock
        return sock

    def _install_signals(self) -> None:
        def stop(_signum: int, _frame: object) -> None:
            self.stopping = True

        for signum in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(ValueError, OSError):    # not the main thread / not allowed
                signal.signal(signum, stop)

    # ------------------------------------------------------------------ one request
    def _serve_connection(self, conn: socket.socket) -> str:
        try:
            line = read_line(conn)
        except (OSError, ValueError):
            return "closed"
        if line is None:
            return "closed"
        reply, op = self.handle_line(line)
        # the client was killed mid-request: the answer has nowhere to go, the host stays
        with contextlib.suppress(OSError):
            conn.sendall(json.dumps(reply).encode("utf-8") + b"\n")
        return op

    def handle_line(self, line: str) -> tuple[dict[str, Any], str]:
        """One JSON line in, one reply out. Never raises: a bad request is an error reply."""
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            return _error("E_UNKNOWN_KEY", f"the request is not JSON ({exc})", 2), "decide"
        if not isinstance(message, dict):
            return _error("E_UNKNOWN_KEY", "the request must be a JSON object", 2), "decide"
        op = str(message.get("op") or "decide")
        if op == "stop":
            return {"ok": True, "stopping": True}, "stop"
        if op == "ping":
            return {"ok": True, "keep": self.status()}, "ping"
        if op != "decide":
            return _error("E_UNKNOWN_KEY", f"unknown op {op!r}", 2), op
        if message.get("key") not in (None, self.spec.digest):
            return _error(
                "E_KEEP_KEY_MISMATCH",
                f"this host is loaded for {self.spec.digest} and the request asked for "
                f"{message.get('key')!r}; the client must stop this host before using another "
                f"model or another placement", 4), "decide"
        payload = message.get("payload")
        if not isinstance(payload, dict):
            return _error("E_UNKNOWN_KEY", "the request needs a `payload` object", 2), "decide"
        assert self.loaded is not None
        self.requests += 1
        self.last_used = self.clock()        # the reply carries *this* request's own numbers
        try:
            body = self.loaded.decide(payload)
        except TypedGgufError as exc:
            return _error(exc.code, str(exc), exc.exit_code), "decide"
        except Exception as exc:                     # noqa: BLE001 - the CLI must not wedge
            return _error("E_INTERNAL", f"{exc.__class__.__name__}: {exc}", 4), "decide"
        return {"ok": True, "response": body, "keep": self.status()}, "decide"

    # ------------------------------------------------------------------ reporting
    def status(self) -> dict[str, Any]:
        """The live facts, from the host itself — what `keep status` merges into the record."""
        now = self.clock()
        last = self.last_used if self.last_used is not None else self.loaded_at
        keep_alive = float(self.spec.keep_alive)
        return {
            "state": "running", "pid": os.getpid(), "key": self.spec.key.to_dict(),
            "key_digest": self.spec.digest, "model": self.spec.model,
            "model_path": self.spec.model_path, "keep_alive_s": round(keep_alive, 3),
            "requests": self.requests, "uptime_s": round(now - self.started_at, 3),
            "idle_left_s": round(max(keep_alive - (now - last), 0.0), 3),
            "loaded_at": self.loaded_at,
            "placement": self.loaded.placement if self.loaded else None,
            "devices": self.loaded.devices if self.loaded else None,
            "model_load_ms": (self.loaded.model_load_ms if self.loaded else 0.0),
            "socket": self.spec.socket_path, "spec": self.spec.spec_path,
            "log": self.spec.log_path, "version": self.spec.version,
        }

    # ------------------------------------------------------------------ internals
    def _write_record(self, *, record_state: str = "ready",
                      error: dict[str, Any] | None = None) -> state.HostRecord:
        stamp = self.clock()
        if record_state == "ready" and self.record is not None:
            stamp = self.record.loaded_at       # the load happened once; keep that timestamp
        record = state.HostRecord(
            digest=self.spec.digest, pid=os.getpid(), socket=self.spec.socket_path,
            key=self.spec.key.to_dict(), model=self.spec.model, model_path=self.spec.model_path,
            keep_alive=float(self.spec.keep_alive), started_at=self.started_at, loaded_at=stamp,
            spec=self.spec.spec_path, log=self.spec.log_path, version=self.spec.version,
            state=record_state, error=error, requests=self.requests, last_used=self.last_used,
            placement=self.loaded.placement if self.loaded else None,
            devices=self.loaded.devices if self.loaded else None,
            model_load_ms=(self.loaded.model_load_ms if self.loaded else 0.0))
        state.write_record(record, self.spec.home)
        return record

    def _fail(self, exc: BaseException, *, record: bool = True) -> int:
        """A host that could not start leaves a `failed` record behind and a code to exit with.

        `record=False` is the one failure that is **not this host's to write down**: the identity
        was lost to a live host (card t_9249bb0c), so the ledger's entry belongs to that host and a
        losing racer writing `failed` over it would erase a resident model from the ledger — the
        next call then treats the winner as debris and clears its socket (two models, SPEC 2.12).
        The message still travels: the log the spawning client quotes carries it.
        """
        self.exit_code = int(getattr(exc, "exit_code", 4)) if isinstance(exc, TypedGgufError) else 4
        message = f"{exc.__class__.__name__}: {exc}"
        code = "E_INTERNAL"
        if isinstance(exc, TypedGgufError):
            code = str(getattr(exc, "code", "E_INTERNAL"))
        if record:
            with contextlib.suppress(Exception):
                self.record = self._write_record(record_state="failed",
                                                 error={"code": str(code), "message": message,
                                                        "exit_code": self.exit_code})
        return self.exit_code

    def _cleanup(self) -> None:
        """Free the model, then remove this host's socket, spec and record (never another's)."""
        if self.loaded is not None:
            with contextlib.suppress(Exception):
                self.loaded.close()
        if self.sock is not None:
            with contextlib.suppress(Exception):
                self.sock.close()
        for target in (self.spec.socket_path, self.spec.spec_path):
            with contextlib.suppress(OSError):
                os.unlink(target)
        state.clear_record(self.spec.home, digest=self.spec.digest)


# ------------------------------------------------------------------ internals
def _listening(path: str | os.PathLike[str]) -> bool:
    """Is something listening on this socket path right now? (A probe, not a guess.)"""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        try:
            probe.connect(str(path))
        except OSError:
            return False
    return True


def _error(code: str, message: str, exit_code: int) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message, "exit_code": int(exit_code)}}


def read_line(conn: socket.socket, *, limit: int = MAX_LINE) -> str | None:
    """Read one newline-terminated JSON line, or None at a clean end of stream."""
    chunks: list[bytes] = []
    total = 0
    while True:
        try:
            block = conn.recv(65536)
        except (ConnectionResetError, TimeoutError):
            return None
        if not block:
            break
        chunks.append(block)
        total += len(block)
        if total > limit:
            raise ValueError(f"the request line is over {limit} bytes")
        if b"\n" in block:
            break
    data = b"".join(chunks)
    if not data:
        return None
    return data.split(b"\n", 1)[0].decode("utf-8", "replace")


def read_reply(conn: socket.socket, *, limit: int = MAX_LINE) -> dict[str, Any]:
    """The host's answer: one JSON object. A truncated/garbled one is the caller's problem."""
    line = read_line(conn, limit=limit)
    if line is None:
        raise ValueError("the host closed the connection without an answer")
    reply = json.loads(line)
    if not isinstance(reply, dict):
        raise ValueError("the host's answer is not a JSON object")
    return reply
