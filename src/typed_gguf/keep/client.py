"""The caller side: reach a warm host, swap it, or answer inline (SPEC 2.12, card t_7e24cea4).

The client is what `ask`/`run` go through, and it is deliberately *quiet* about all of it: the
command's answer is the same either way.

```
decide(key, payload, keep_alive, inline)
    keep_alive == 0          -> stop any host, answer inline      ("answer and unload")
    a live host for `key`    -> send the request, merge `engine.keep`
    a live host for another  -> stop it (free the device), spawn ours
    a stale record           -> clean the socket/record up, spawn
    nothing                  -> spawn (detached, its own session), wait until ready, send
```

**The fallback policy** (decided here, documented in README + SPEC 2.12):

* a **transport** failure — no socket, connection refused, the host died or closed mid-request, a
  garbled reply — is the box talking, never the answer, so the client cleans the dead host up and
  runs the request **once inline** (cold). The response says so: `engine.keep.served_by: "inline"`
  and `engine.keep.fallback: "transport: …"`;
* a **typed error** from the host is the product's answer (a bad decision, a refused prefix, an
  architecture the bundle cannot run). It is re-raised with its own code and exit status, and it is
  **not** retried inline: the cold path would fail identically, twice as slowly;
* a spawn that never becomes ready (or that exits) is a transport failure too: the log tail the
  child left behind travels in the fallback string, so the reason is readable from the response.

Nothing here ever loops: one spawn, one request, at most one inline run.
"""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from typed_gguf.errors import RuntimeError_, TypedGgufError, UserError
from typed_gguf.keep import host as host_module
from typed_gguf.keep import identity, state

#: The process a client starts when there is no host. `python -m typed_gguf` (not the console
#: script) so the *same* interpreter that ran the command runs the host; `--spec` is appended.
HOST_COMMAND: tuple[str, ...] = (sys.executable, "-m", "typed_gguf", "keep", "_host", "--spec")
#: How long a client waits for a spawned host to load its model and bind. A cold 27B load is
#: minutes, not seconds, and the wait is the *price of the first call* — the second one is warm.
SPAWN_TIMEOUT = 600.0
#: How long one decision may take on a warm host (the inline path has no such cap).
REQUEST_TIMEOUT = 900.0
#: A `keep status` ping is a health check: it must answer *now* or be reported as unresponsive.
PING_TIMEOUT = 2.0
#: How long `keep stop` (and a swap) waits for a host to exit after SIGTERM before SIGKILL.
STOP_GRACE = 5.0
STATUS_SCHEMA = "typed_gguf.keep.status/v1"
_SPAWN_POLL = 0.02

_EXIT_CLASSES: dict[int, type[TypedGgufError]] = {2: UserError, 3: RuntimeError_, 4: TypedGgufError}


class KeepUnavailable(RuntimeError):
    """The host cannot be reached/started right now: the caller answers inline instead.

    Deliberately *not* a `TypedGgufError`: it carries no product error code because it never
    reaches a user as one. `cli` turns it into an inline run plus a `fallback` string.
    """


class TransportError(RuntimeError):
    """A transport failure inside the client: the host is gone, refused, or answered garbage.

    `kind` separates the two cases the cleanup policy needs: `refused`/`gone` (nothing is
    listening — the record is debris and is removed whatever its pid claims) from `timeout`/
    `garbled` (a host that may still be busy: its ledger entry is left alone).
    """

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def _error_classes() -> dict[str, type[TypedGgufError]]:
    """Every product error class, keyed by its frozen code (`E_X` -> `SomeError`)."""
    from typed_gguf import errors as errors_module

    found: dict[str, type[TypedGgufError]] = {}
    for name in dir(errors_module):
        candidate = getattr(errors_module, name)
        if (isinstance(candidate, type) and issubclass(candidate, TypedGgufError)
                and isinstance(candidate.__dict__.get("code"), str)):
            # only classes that *claim* a code: `UserError`/`RuntimeError_` inherit E_INTERNAL
            # without being it, and mapping a code to an abstract base would erase the exit status
            found.setdefault(str(candidate.code), candidate)
    return found


def typed_error_from_wire(error: Mapping[str, Any]) -> TypedGgufError:
    """Rebuild the host's typed error from `{code, message, exit_code}` (SPEC 2.5 codes intact)."""
    code = str(error.get("code") or "E_INTERNAL")
    message = str(error.get("message") or code)
    exit_code = int(error.get("exit_code") or 4)
    factory = _error_classes().get(code) or _EXIT_CLASSES.get(exit_code, TypedGgufError)
    return factory(message, code=code)


def _credit_the_spawn(body: dict[str, Any], keep: Mapping[str, Any]) -> dict[str, Any]:
    """Charge the host's one-time load to the call that waited for it (SPEC 2.12, A-E4-1).

    A host answers `timings.model_load_ms: 0.0` on every request — the session *it* opens pays no
    load, because the model has been resident since before that request. True about the host, and
    false about the call that **spawned** it: that call sat and waited for exactly that load. So
    the client writes the host's number into the answer it asked for, and only for that call: a
    later call that found the host already resident keeps 0.0. The host's own figure stays in
    `engine.keep.model_load_ms` on every answer, warm or cold.
    """
    load_ms = float(keep.get("model_load_ms") or 0.0)
    timings = body.get("timings")
    if load_ms > 0.0 and isinstance(timings, dict):
        timings["model_load_ms"] = load_ms
    return body


class Client:
    """One CLI invocation's view of the ledger. Cheap to build; holds the children it spawned."""

    def __init__(self, *, home: pathlib.Path | None = None,
                 host_command: Sequence[str] = HOST_COMMAND,
                 environ: Mapping[str, str] | None = None,
                 spawn: Callable[..., subprocess.Popen] | None = None,
                 timeout: float = REQUEST_TIMEOUT, spawn_timeout: float = SPAWN_TIMEOUT,
                 ping_timeout: float = PING_TIMEOUT, stop_grace: float = STOP_GRACE,
                 clock: Callable[[], float] = time.time) -> None:
        self.home = home
        self.host_command = tuple(host_command)
        self.environ = environ
        self.spawn = spawn or _spawn_detached
        self.timeout = float(timeout)
        self.spawn_timeout = float(spawn_timeout)
        self.ping_timeout = float(ping_timeout)
        self.stop_grace = float(stop_grace)
        self.clock = clock
        #: every lifecycle step, in order (`spawn`, `ready`, `reuse`, `stop`, `cleanup`, `inline`).
        #: An event log is what makes "B was stopped *before* A loaded" an assertion, not a hope.
        self.events: list[tuple[str, Any]] = []
        self.spawns: list[dict[str, Any]] = []
        self.children: dict[int, subprocess.Popen] = {}

    # ------------------------------------------------------------------ the transparent path
    def decide(self, key: identity.KeepKey, payload: dict[str, Any], *, keep_alive: float,
               inline: Callable[[], dict[str, Any]], fit: Mapping[str, Any] | None = None,
               model: str = "") -> dict[str, Any]:
        """One request: warm if a host can be had, inline if it cannot (never a third path)."""
        keep_alive = float(keep_alive)
        if keep_alive <= 0:
            report = self.stop()
            self.events.append(("inline", "keep-alive 0"))
            return self._mark(inline(), served_by="inline", keep_alive_s=0.0,
                              stopped=bool(report.get("stopped")))
        spawns_before = len(self.spawns)
        try:
            record = self.ensure(key, keep_alive=keep_alive, fit=fit, payload=payload,
                                 model=model)
            reply = self.request(record, payload)
        except KeepUnavailable as exc:
            self.events.append(("inline", str(exc)))
            return self._mark(inline(), served_by="inline", keep_alive_s=keep_alive,
                              fallback=str(exc))
        if not reply.get("ok"):
            raise typed_error_from_wire(reply.get("error") or {})
        keep = dict(reply.get("keep") or {})
        body = reply.get("response") or {}
        if len(self.spawns) > spawns_before:
            # this call paid the load by waiting for the spawn: say so in its own timings
            body = _credit_the_spawn(body, keep)
        return self._mark(body, served_by="host", keep=keep,
                          keep_alive_s=keep.get("keep_alive_s", keep_alive))

    def ensure(self, key: identity.KeepKey, *, keep_alive: float,
               fit: Mapping[str, Any] | None = None, payload: dict[str, Any] | None = None,
               model: str = "") -> state.HostRecord:
        """A live host for `key`: reuse it, swap it out, clean it up, or spawn one."""
        record = state.read_record(self.home)
        if record is not None:
            if not state.alive(record):
                # a dead pid or a socket that is gone: the host died between two calls
                state.clear_record(self.home, digest=record.digest)
                self.events.append(("cleanup", record.digest))
            elif record.digest != key.digest:
                report = self.stop()                 # one model at a time: free the device first
                if not report.get("stopped") and state.pid_alive(record.pid):
                    raise KeepUnavailable(
                        f"spawn: the resident host (pid {record.pid}, digest {record.digest}) "
                        f"could not be stopped; refusing to load a second model")
            else:
                self.events.append(("reuse", record.pid))
                return record
        return self.spawn_host(key, keep_alive=keep_alive, fit=fit, payload=payload or {},
                               model=model)

    def spawn_host(self, key: identity.KeepKey, *, keep_alive: float, fit: Mapping[str, Any] | None,
                   payload: dict[str, Any], model: str) -> state.HostRecord:
        """Write the spec, start the host detached, and wait until it is listening."""
        state.ensure_dir(self.home)
        spec = host_module.HostSpec(
            key=key, digest=key.digest, model=model or pathlib.Path(key.model_path).stem,
            model_path=key.model_path, keep_alive=float(keep_alive),
            spec_path=str(state.spec_path(self.home, key.digest)),
            socket_path=str(state.socket_path(self.home, key.digest)),
            log_path=str(state.log_path(self.home, key.digest)),
            payload=payload, fit=dict(fit or {}),
            home=str(self.home) if self.home is not None else None)
        spec.save()
        argv = [*self.host_command, spec.spec_path]
        env = self.child_env()
        log = pathlib.Path(spec.log_path)
        proc = self.spawn(argv, env=env, log=log)
        self.children[proc.pid] = proc
        self.spawns.append({"argv": list(argv), "env": dict(env), "log": str(log),
                            "spec": spec.spec_path, "pid": proc.pid})
        self.events.append(("spawn", proc.pid))
        record = self._wait_ready(proc, spec)
        self.events.append(("ready", record.pid))
        return record

    def child_env(self) -> dict[str, str]:
        """The child's environment: the same data home, and the same *code* as this client.

        `PYTHONPATH` is prepended with the directory the running `typed_gguf` package lives in, so
        a host started from a checkout (or a worktree) imports the client's own source tree
        instead of whatever happens to be installed.
        """
        env = dict(os.environ if self.environ is None else self.environ)
        if self.home is not None:
            env["TYPED_GGUF_HOME"] = str(self.home)
        import typed_gguf

        package_parent = str(pathlib.Path(typed_gguf.__file__).resolve().parents[1])
        entries = [part for part in env.get("PYTHONPATH", "").split(os.pathsep) if part]
        if package_parent not in entries:
            env["PYTHONPATH"] = os.pathsep.join([package_parent, *entries])
        return env

    def request(self, record: state.HostRecord, payload: dict[str, Any], *,
                timeout: float | None = None) -> dict[str, Any]:
        """One decision on a live host. A transport failure is cleaned up and named."""
        try:
            return self._call(record, {"schema": host_module.REQUEST_SCHEMA, "op": "decide",
                                       "key": record.digest, "payload": payload},
                              timeout=self.timeout if timeout is None else timeout)
        except TransportError as exc:
            self._abandon(record, hard=exc.kind in ("refused", "gone"))
            raise KeepUnavailable(f"transport: {exc}") from exc

    def _call(self, record: state.HostRecord, message: Mapping[str, Any], *,
              timeout: float) -> dict[str, Any]:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout)
            try:
                client.connect(record.socket)
            except FileNotFoundError as exc:
                raise TransportError("gone", f"the socket {record.socket} is gone") from exc
            except ConnectionRefusedError as exc:
                raise TransportError(
                    "refused", f"nothing is listening on {record.socket} (stale record)") from exc
            except OSError as exc:
                raise TransportError("refused",
                                     f"the socket {record.socket} is unreachable ({exc})") from exc
            try:
                client.sendall(json.dumps(dict(message)).encode("utf-8") + b"\n")
                return host_module.read_reply(client)
            except TimeoutError as exc:
                raise TransportError(
                    "timeout", f"the host did not answer within {timeout}s") from exc
            except (OSError, ValueError) as exc:
                raise TransportError(
                    "garbled", f"the host answered nothing usable ({exc.__class__.__name__}: "
                               f"{exc})") from exc

    def _abandon(self, record: state.HostRecord, *, hard: bool) -> None:
        """A record that cannot be talked to. `hard` = nothing was listening: it is debris.

        The other case is a host that may still be busy (a timeout, a truncated reply): its ledger
        entry is left for the next call, which will clean it up if the pid is really gone.
        """
        child = self.children.pop(record.pid, None)
        if child is not None:
            with contextlib.suppress(Exception):
                child.wait(timeout=2.0)                 # never leave a zombie of our own
        if hard or not state.pid_alive(record.pid):
            state.clear_record(self.home, digest=record.digest)
            self.events.append(("cleanup", record.digest))

    def _wait_ready(self, proc: subprocess.Popen, spec: host_module.HostSpec) -> state.HostRecord:
        deadline = time.monotonic() + self.spawn_timeout
        while time.monotonic() < deadline:
            record = state.read_record(self.home)
            if record is not None and record.digest == spec.digest:
                if record.state == "failed":
                    message = str((record.error or {}).get("message") or "the host refused to load")
                    self._discard(proc, spec, message)
                    raise KeepUnavailable(f"spawn: {message}")
                if record.state == "ready" and state.alive(record):
                    return record
            code = proc.poll()
            if code is not None:
                self.children.pop(proc.pid, None)
                raise KeepUnavailable(
                    f"spawn: the host exited with code {code}{self._log_tail(spec)}")
            time.sleep(_SPAWN_POLL)
        message = (f"the host did not become ready within {self.spawn_timeout}s"
                   f"{self._log_tail(spec)}")
        self._discard(proc, spec, message)
        raise KeepUnavailable(f"spawn: {message}")

    def _discard(self, proc: subprocess.Popen, spec: host_module.HostSpec, _reason: str) -> None:
        """A host that never came up: kill it, wait for it, remove what it left behind."""
        with contextlib.suppress(OSError):
            proc.send_signal(signal.SIGKILL)
        with contextlib.suppress(Exception):
            proc.wait(timeout=2.0)
        self.children.pop(proc.pid, None)
        state.clear_record(self.home, digest=spec.digest)
        self.events.append(("cleanup", spec.digest))

    def _log_tail(self, spec: host_module.HostSpec, *, limit: int = 400) -> str:
        try:
            text = pathlib.Path(spec.log_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        tail = " ".join(line.strip() for line in text.strip().splitlines()[-8:] if line.strip())
        return f": {tail[-limit:]}" if tail else ""

    # ------------------------------------------------------------------ the verbs
    def status(self) -> dict[str, Any]:
        """`keep status`: the record, overlaid with the live host's own numbers when it answers."""
        record = state.read_record(self.home)
        if record is None:
            return {"schema": STATUS_SCHEMA, "state": "stopped", "pid": None, "model": None,
                    "model_path": None, "keep_alive_s": None, "idle_left_s": None,
                    "uptime_s": None, "requests": 0, "key": None, "key_digest": None,
                    "placement": None, "devices": None, "model_load_ms": None, "socket": None,
                    "spec": None, "log": None, "error": None, "version": None}
        base = {"schema": STATUS_SCHEMA, **state.host_status(record)}
        if not state.alive(record):
            return base
        try:
            reply = self._call(record, {"schema": host_module.REQUEST_SCHEMA, "op": "ping",
                                        "key": record.digest}, timeout=self.ping_timeout)
        except (KeepUnavailable, TransportError, OSError, ValueError,
                json.JSONDecodeError) as exc:
            # a host that is alive but cannot answer *now* is a state, not a crash (card
            # t_7e24cea4: the live gates caught a `kill -9` mid-request turning `keep status`
            # into `E_INTERNAL: TransportError` instead of a report)
            return {**base, "state": "unresponsive",
                    "detail": f"{exc.__class__.__name__}: {exc}"}
        return {**base, **dict(reply.get("keep") or {}), "state": "running"}

    def stop(self, *, grace: float | None = None) -> dict[str, Any]:
        """`keep stop`: end the host, wait for the process, remove the ledger's entry."""
        timeout = self.stop_grace if grace is None else float(grace)
        record = state.read_record(self.home)
        if record is None:
            return {"stopped": False, "pid": None, "reason": "no host", "cleaned": False}
        pid = record.pid
        stopped = False
        reason = "was already gone"
        if pid == os.getpid():
            # a hand-written or corrupted record must never make the CLI signal itself
            reason = "the record points at this very process; refusing to signal it"
        elif state.pid_alive(pid):
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGTERM)
            child = self.children.get(pid)
            if child is not None:
                with contextlib.suppress(Exception):
                    child.wait(timeout=timeout)
                    stopped = True
                    reason = "stopped (SIGTERM)"
            else:
                stopped = state.wait_pid_gone(pid, timeout=timeout)
                reason = "stopped (SIGTERM)" if stopped else f"pid {pid} ignored SIGTERM"
            if not stopped:
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGKILL)
                if child is not None:
                    with contextlib.suppress(Exception):
                        child.wait(timeout=2.0)
                stopped = state.wait_pid_gone(pid, timeout=2.0)
                reason = ("killed (SIGKILL)" if stopped else f"pid {pid} survived SIGKILL")
        child = self.children.pop(pid, None)
        if child is not None:
            with contextlib.suppress(Exception):
                child.wait(timeout=2.0)
        cleaned = state.clear_record(self.home, digest=record.digest)
        self.events.append(("stop", pid))
        return {"stopped": stopped, "pid": pid, "reason": reason, "cleaned": cleaned}

    # ------------------------------------------------------------------ internals
    def _mark(self, body: dict[str, Any], *, served_by: str, keep_alive_s: float,
              keep: Mapping[str, Any] | None = None, fallback: str | None = None,
              **extra: Any) -> dict[str, Any]:
        """Publish how this answer was produced (`engine.keep`), on the native shape only."""
        block: dict[str, Any] = {"served_by": served_by, "keep_alive_s": round(keep_alive_s, 3)}
        block.update(dict(keep or {}))
        block["served_by"] = served_by
        block["fallback"] = fallback
        block.update(extra)
        if isinstance(body.get("engine"), dict):
            body["engine"]["keep"] = block
        return body


def _spawn_detached(argv: Sequence[str], *, env: Mapping[str, str],
                    log: pathlib.Path) -> subprocess.Popen:
    """Start the host in its own session, its streams in the log file, nothing inherited.

    `start_new_session=True` is what makes the host *survive* the client (and a `kill -9` of it):
    it holds the loaded model for its keep-alive window, and the idle timer — not the client's
    lifetime — is what frees the device.
    """
    log.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    handle = open(log, "ab", buffering=0)            # noqa: SIM115 - the child owns the fd now
    try:
        return subprocess.Popen(                     # noqa: S603 - argv is ours, not the user's
            [str(part) for part in argv], stdin=subprocess.DEVNULL, stdout=handle,
            stderr=handle, start_new_session=True, close_fds=True, env=dict(env))
    finally:
        handle.close()


def _listening(path: str | os.PathLike[str]) -> bool:
    """Can anything be reached at this socket path right now?"""
    target = pathlib.Path(path)
    if not target.exists():
        return False
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        try:
            probe.connect(str(target))
        except OSError:
            return False
    return True
