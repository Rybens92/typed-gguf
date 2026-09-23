"""The warm host's own half: socket, ledger, idle exit, serialization (SPEC 2.12, card t_7e24cea4).

In-process (a real unix socket + a real record file, a fake engine), so these gates run on any
box that has unix sockets at all — no child process, no pid pressure.
"""
from __future__ import annotations

import json
import os
import pathlib
import socket
import stat
import threading
import time

from typed_gguf.errors import PrefillFailedError
from typed_gguf.keep import host as host_module
from typed_gguf.keep import identity, state


def _spec(home: pathlib.Path, *, keep_alive: float = 30.0, model: str = "a",
          **key_changes: object) -> host_module.HostSpec:
    key = identity.KeepKey.of(None, model_path=f"/models/{model}.gguf", model_sha="deadbeef")
    key = identity.KeepKey(**{**key.to_dict(), **key_changes})
    return host_module.HostSpec(
        key=key, digest=key.digest, model=model, model_path=key.model_path,
        keep_alive=keep_alive, spec_path=str(state.spec_path(home, key.digest)),
        socket_path=str(state.socket_path(home, key.digest)),
        log_path=str(state.log_path(home, key.digest)),
        payload={"state": "S", "questions": {"q": {"type": "noul"}}},
        fit={"fit_enabled": True}, home=str(home))


class FakeHandle:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _loaded(handle: FakeHandle, *, decide=None, **fields: object) -> host_module.Loaded:
    def default_decide(payload: dict) -> dict:
        return {"model": "a", "engine": {"backend": "cpu"}, "answers": {}, "usage": {},
                "timings": {"model_load_ms": 0.0}, "warnings": []}

    return host_module.Loaded(handle=handle, decide=decide or default_decide,
                              model=str(fields.get("model", "a")),
                              model_path=str(fields.get("model_path", "/models/a.gguf")),
                              placement=fields.get("placement"),
                              devices=fields.get("devices"),
                              model_load_ms=float(fields.get("model_load_ms", 0.0)))


class Running:
    """A `Server` on a thread, with a socket client and a join-or-fail helper."""

    def __init__(self, home: pathlib.Path, spec: host_module.HostSpec, loaded: host_module.Loaded):
        self.home = home
        self.spec = spec
        self.loaded = loaded
        self.server = host_module.Server(spec, load=lambda: loaded)
        self.thread = threading.Thread(target=self.server.serve, daemon=True)
        self.thread.start()
        self._wait_ready()

    def _wait_ready(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if (state.read_record(self.home) is not None
                    and pathlib.Path(self.spec.socket_path).exists()):
                return
            time.sleep(0.01)
        raise AssertionError("the host never became ready")

    def call(self, message: dict, *, timeout: float = 5.0) -> dict:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout)
            client.connect(self.spec.socket_path)
            client.sendall(json.dumps(message).encode("utf-8") + b"\n")
            return host_module.read_reply(client)

    def decide(self, payload: dict | None = None, *, timeout: float = 5.0) -> dict:
        return self.call({"schema": host_module.REQUEST_SCHEMA, "op": "decide",
                          "key": self.spec.digest,
                          "payload": payload or self.spec.payload}, timeout=timeout)

    def join(self, timeout: float = 5.0) -> int:
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "the host did not exit"
        return int(self.server.exit_code)


def _wait_record(home: pathlib.Path, predicate, *, timeout: float = 2.0):
    """Wait for the ledger to catch up with the host's live reply — a file write, not a message."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = state.read_record(home)
        if record is not None and predicate(record):
            return record
        time.sleep(0.01)
    raise AssertionError(f"the record never satisfied {predicate}")


def test_the_host_answers_a_decision_over_its_socket(keep_home: pathlib.Path) -> None:
    spec = _spec(keep_home)
    running = Running(keep_home, spec, _loaded(FakeHandle()))
    reply = running.decide({"questions": {"q": {"type": "choice", "criteria": {"b": None}}}})
    assert reply["ok"] is True
    assert reply["response"]["model"] == "a"
    assert reply["keep"]["requests"] == 1 and reply["keep"]["pid"] == os.getpid()
    assert reply["keep"]["key_digest"] == spec.digest
    # the ledger says what is resident, where, and for how long
    record = _wait_record(keep_home, lambda item: item.requests == 1)
    assert record.state == "ready" and record.digest == spec.digest
    assert record.last_used is not None
    # the socket is private (0600) and the directory too (0700)
    assert stat.S_IMODE(os.stat(spec.socket_path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(state.keep_dir(keep_home)).st_mode) == 0o700
    running.call({"schema": host_module.REQUEST_SCHEMA, "op": "stop", "key": spec.digest})
    running.join()


def test_the_host_exits_by_itself_once_the_idle_window_passes(keep_home: pathlib.Path) -> None:
    """RED pin: `--keep-alive` is a promise about the *process*, not a report field."""
    spec = _spec(keep_home, keep_alive=0.3)
    handle = FakeHandle()
    running = Running(keep_home, spec, _loaded(handle))
    loaded_at = state.read_record(keep_home).loaded_at
    assert running.join(timeout=10.0) == 0
    # it really waited out the window (measured from the load, not from the test's own poll)…
    assert time.time() - loaded_at >= 0.3 - 0.05
    assert time.time() - loaded_at < 10.0
    assert handle.closed is True                      # the model was freed, not leaked
    assert not pathlib.Path(spec.socket_path).exists()
    assert state.read_record(keep_home) is None
    assert not pathlib.Path(spec.spec_path).exists()


def test_a_request_moves_the_idle_deadline(keep_home: pathlib.Path) -> None:
    spec = _spec(keep_home, keep_alive=1.0)
    running = Running(keep_home, spec, _loaded(FakeHandle()))
    time.sleep(0.6)
    running.decide()
    time.sleep(0.6)                                   # 1.2 s after the load, 0.6 s after the use
    assert running.thread.is_alive(), "the countdown must restart on every request"
    running.call({"schema": host_module.REQUEST_SCHEMA, "op": "stop", "key": spec.digest})
    running.join()


def test_requests_are_serialized_inside_the_host(keep_home: pathlib.Path) -> None:
    """SPEC 2.12: one model, one decision at a time — a queue, not a race."""
    inside = threading.Semaphore(0)
    overlap: list[int] = []

    def decide(payload: dict) -> dict:
        overlap.append(1)
        assert len(overlap) == 1, "two decisions ran at once inside one host"
        time.sleep(0.2)
        overlap.pop()
        return {"model": "a", "engine": {}, "answers": {}, "usage": {}, "timings": {},
                "warnings": []}

    spec = _spec(keep_home)
    running = Running(keep_home, spec, _loaded(FakeHandle(), decide=decide))
    results: list[dict] = []

    def client() -> None:
        results.append(running.decide({"questions": {}}))

    threads = [threading.Thread(target=client) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 3 and all(reply["ok"] for reply in results)
    assert [reply["keep"]["requests"] for reply in results] == [1, 2, 3]
    del inside
    running.call({"schema": host_module.REQUEST_SCHEMA, "op": "stop", "key": spec.digest})
    running.join()


def test_a_typed_error_travels_as_a_typed_error(keep_home: pathlib.Path) -> None:
    """A bad decision is reported, not swallowed — and the host survives to answer the next one."""
    calls: list[int] = []

    def decide(payload: dict) -> dict:
        calls.append(1)
        if len(calls) == 1:
            raise PrefillFailedError("E_PREFILL_FAILED: the prefix was refused (test)")
        return {"model": "a", "engine": {}, "answers": {}, "usage": {}, "timings": {},
                "warnings": []}

    spec = _spec(keep_home)
    running = Running(keep_home, spec, _loaded(FakeHandle(), decide=decide))
    failed = running.decide({"questions": {}})
    assert failed["ok"] is False
    assert failed["error"]["code"] == "E_PREFILL_FAILED"
    assert failed["error"]["exit_code"] == 3 and "E_PREFILL_FAILED" in failed["error"]["message"]
    again = running.decide({"questions": {}})
    assert again["ok"] is True
    running.call({"schema": host_module.REQUEST_SCHEMA, "op": "stop", "key": spec.digest})
    running.join()


def test_a_client_that_disappears_mid_request_does_not_wedge_the_host(
        keep_home: pathlib.Path) -> None:
    """The CLI can be killed at any moment: the answer is dropped, the host is not."""
    started = threading.Event()

    def decide(payload: dict) -> dict:
        started.set()
        time.sleep(0.3)
        return {"model": "a", "engine": {}, "answers": {}, "usage": {}, "timings": {},
                "warnings": []}

    spec = _spec(keep_home)
    running = Running(keep_home, spec, _loaded(FakeHandle(), decide=decide))
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(spec.socket_path)
        message = {"schema": host_module.REQUEST_SCHEMA, "op": "decide", "key": spec.digest,
                   "payload": {"questions": {}}}
        client.sendall(json.dumps(message).encode() + b"\n")
        assert started.wait(5.0)
    # the socket is gone: the host must notice, keep going, and answer the next request
    assert running.decide({"questions": {}})["ok"] is True
    running.call({"schema": host_module.REQUEST_SCHEMA, "op": "stop", "key": spec.digest})
    running.join()


def test_the_stop_op_ends_the_host(keep_home: pathlib.Path) -> None:
    spec = _spec(keep_home)
    running = Running(keep_home, spec, _loaded(FakeHandle()))
    reply = running.call({"schema": host_module.REQUEST_SCHEMA, "op": "stop", "key": spec.digest})
    assert reply["ok"] is True and reply["stopping"] is True
    assert running.join() == 0
    assert state.read_record(keep_home) is None


def test_a_stale_socket_file_is_replaced_instead_of_fought_over(keep_home: pathlib.Path) -> None:
    """RED pin (card t_7e24cea4): a crash must not leave a socket nobody can rebind."""
    spec = _spec(keep_home)
    state.ensure_dir(keep_home)
    socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).close()      # a socket with no listener
    pathlib.Path(spec.socket_path).write_text("", encoding="utf-8")
    running = Running(keep_home, spec, _loaded(FakeHandle()))
    assert running.decide({"questions": {}})["ok"] is True
    running.call({"schema": host_module.REQUEST_SCHEMA, "op": "stop", "key": spec.digest})
    running.join()


def test_the_host_reports_the_placement_and_devices_it_proved(keep_home: pathlib.Path) -> None:
    spec = _spec(keep_home)
    loaded = _loaded(FakeHandle(), placement={"note": "fit plan: 32 layer(s) offloaded",
                                              "n_gpu_layers": 32},
                     devices={"devices": ["Vulkan0"], "compute_buffers": {"Vulkan0": 9},
                              "effective_backend": "vulkan"},
                     model_load_ms=1234.5)
    running = Running(keep_home, spec, loaded)
    reply = running.decide({"questions": {}})["keep"]
    assert reply["placement"]["n_gpu_layers"] == 32
    assert reply["devices"]["effective_backend"] == "vulkan"
    assert reply["model_load_ms"] == 1234.5 and reply["key"] == spec.key.to_dict()
    record = state.read_record(keep_home)
    assert record is not None and record.placement == loaded.placement
    idle = state.host_status(record, now=record.last_used)["idle_left_s"]
    assert abs(idle - spec.keep_alive) < 1.0, "the countdown is the host's own window, ± a poll"
    running.call({"schema": host_module.REQUEST_SCHEMA, "op": "stop", "key": spec.digest})
    running.join()


def test_a_request_for_another_key_is_refused(keep_home: pathlib.Path) -> None:
    """One host = one identity: a mismatched request is a bug in the caller, and it says so."""
    spec = _spec(keep_home)
    running = Running(keep_home, spec, _loaded(FakeHandle()))
    reply = running.call({"schema": host_module.REQUEST_SCHEMA, "op": "decide", "key": "otherkey",
                          "payload": {"questions": {"q": {"type": "noul"}}}})
    assert reply["ok"] is False and reply["error"]["code"] == "E_KEEP_KEY_MISMATCH"
    assert spec.digest in reply["error"]["message"]
    running.call({"schema": host_module.REQUEST_SCHEMA, "op": "stop", "key": spec.digest})
    running.join()


def test_an_unreadable_request_is_an_error_not_a_crash(keep_home: pathlib.Path) -> None:
    spec = _spec(keep_home)
    running = Running(keep_home, spec, _loaded(FakeHandle()))
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(5.0)
        client.connect(spec.socket_path)
        client.sendall(b"{not json}\n")
        reply = host_module.read_reply(client)
    assert reply["ok"] is False and reply["error"]["code"] == "E_UNKNOWN_KEY"
    assert running.decide({"questions": {}})["ok"] is True
    running.call({"schema": host_module.REQUEST_SCHEMA, "op": "stop", "key": spec.digest})
    running.join()


def test_the_spec_round_trips_through_its_file(keep_home: pathlib.Path) -> None:
    spec = _spec(keep_home)
    written = spec.save()
    assert written == pathlib.Path(spec.spec_path)
    assert stat.S_IMODE(written.stat().st_mode) == 0o600
    assert host_module.read_spec(spec.spec_path) == spec
    assert host_module.read_spec(str(keep_home / "keep" / "nope.json")) is None


def test_a_host_whose_socket_is_taken_by_a_live_host_refuses(keep_home: pathlib.Path) -> None:
    """A second host on the same identity must not steal the first one's socket."""
    spec = _spec(keep_home)
    first = Running(keep_home, spec, _loaded(FakeHandle()))
    second = host_module.Server(spec, load=lambda: _loaded(FakeHandle()))
    second.bind()
    assert second.bind_error is not None
    assert "already" in second.bind_error or "in use" in second.bind_error
    first.call({"schema": host_module.REQUEST_SCHEMA, "op": "stop", "key": spec.digest})
    first.join()


def test_a_host_that_loses_the_socket_race_leaves_the_winners_record_alone(
        keep_home: pathlib.Path) -> None:
    """RED pin (card t_9249bb0c): the loser of the socket race writes *nothing* to the ledger.

    Two cold callers spawn two hosts on one identity; one binds, the other is refused. That loser
    used to publish its own `failed` record over the winner's `ready` one — the resident model then
    vanished from the ledger, and the loser's caller cleaned up a *live* host's socket on the next
    call (two models, SPEC 2.12). The loser's failure still travels: its own log, which the spawning
    client quotes in the fallback string. The winner's entry must read exactly as before.
    """
    spec = _spec(keep_home)
    first = Running(keep_home, spec, _loaded(FakeHandle()))
    before = state.read_record(keep_home)
    assert before is not None and before.state == "ready"

    loser = host_module.Server(spec, load=lambda: _loaded(FakeHandle()))
    assert loser.serve() == 4, "a host that cannot bind still fails, and says so by exit code"
    assert loser.bind_owned_elsewhere is True and loser.bind_error is not None
    after = state.read_record(keep_home)
    assert after is not None and after.pid == before.pid and after.state == "ready", \
        "the losing racer erased the resident host from the ledger"
    first.call({"schema": host_module.REQUEST_SCHEMA, "op": "stop", "key": spec.digest})
    first.join()


def test_a_host_that_cannot_load_leaves_a_readable_failed_record(keep_home: pathlib.Path) -> None:
    """`_fail`: the failure is readable in the ledger — code, message and the process exit code.

    A host that dies before it binds cannot answer over its socket, so the *record* is the only
    channel the spawning client has (`client.py` turns `state == "failed"` into a typed
    `KeepUnavailable`): it must carry `state=failed`, the typed code, the message and the exit code
    the process ends with. A plain crash is `E_INTERNAL` / exit 4 — never a made-up code. This
    helper held 45 mutants in the Tier-M sweep's "no tests" bucket (card t_7e24cea4).
    """
    spec = _spec(keep_home)
    refused = PrefillFailedError("E_PREFILL_FAILED: the prefix was refused (test)")

    def refuse() -> host_module.Loaded:
        raise refused

    server = host_module.Server(spec, load=refuse)
    assert server.serve() == refused.exit_code, "a typed failure keeps its own exit code"
    record = state.read_record(keep_home)
    assert record is not None, "the failure is written down for the client to read"
    assert record.state == "failed"
    assert record.error is not None
    assert record.error["code"] == "E_PREFILL_FAILED"
    assert "the prefix was refused (test)" in record.error["message"]
    assert record.error["exit_code"] == refused.exit_code
    assert not pathlib.Path(spec.socket_path).exists(), "a host that never bound leaves no socket"

    def explodes() -> host_module.Loaded:
        raise RuntimeError("the loader fell over")

    server = host_module.Server(spec, load=explodes)
    assert server.serve() == 4, "an untyped failure is E_INTERNAL, exit 4"
    record = state.read_record(keep_home)
    assert record is not None and record.error is not None
    assert record.error["code"] == "E_INTERNAL"
    assert record.error["message"] == "RuntimeError: the loader fell over"
    assert record.error["exit_code"] == 4
