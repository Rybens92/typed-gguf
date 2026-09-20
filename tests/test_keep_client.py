"""The client: reuse, swap, stale cleanup, and the inline fallback (SPEC 2.12, card t_7e24cea4).

Every test here drives a **real** host child process (`tests/fake_keep_host.py`: the production
`keep.host.Server` around a fake engine), so spawn/reuse/swap/cleanup are measured, not mocked.
The child is what makes these `needs_fork` gates: on a starved pid cgroup they skip by name
(`tests/conftest.py`) instead of blaming the product.
"""
from __future__ import annotations

import json
import os
import pathlib
import socket
import sys
import time

import pytest

from typed_gguf.errors import PrefillFailedError, TypedGgufError
from typed_gguf.keep import client as client_module
from typed_gguf.keep import identity, state

FAKE_HOST = pathlib.Path(__file__).resolve().parent / "fake_keep_host.py"
PAYLOAD = {"state": "Billing is down.", "questions": {"q": {"type": "noul", "instructions": "x"}}}


def _key(model: str = "a", **changes: object) -> identity.KeepKey:
    base = identity.KeepKey.of(None, model_path=f"/models/{model}.gguf", model_sha="deadbeef")
    return identity.KeepKey(**{**base.to_dict(), **changes})


def _inline_body(model: str = "a") -> dict:
    return {"model": model, "engine": {"backend": "cpu", "backend_source": "explicit"},
            "answers": {}, "usage": {}, "timings": {"model_load_ms": 5.0}, "warnings": []}


class Inline:
    """The cold path, as the CLI hands it to the client: a callable + a call count."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> dict:
        self.calls += 1
        return json.loads(json.dumps(_inline_body()))


@pytest.fixture
def keep_home(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("TYPED_GGUF_HOME", str(home))
    monkeypatch.delenv("TYPED_GGUF_KEEP_FAKE", raising=False)
    return home


@pytest.fixture
def make_client(keep_home: pathlib.Path):
    """A client whose "host" is the fake script — the real spawn path, a fake engine."""
    clients: list[client_module.Client] = []

    def build(*, host_command: tuple = (sys.executable, str(FAKE_HOST), "--spec"),
              **kwargs: object) -> client_module.Client:
        client = client_module.Client(home=keep_home, host_command=host_command, **kwargs)
        clients.append(client)
        return client

    yield build
    for client in clients:                             # never leave a fake host behind
        client.stop(grace=1.0)
        for _pid, proc in list(client.children.items()):
            proc.kill()


@pytest.mark.needs_fork
def test_a_call_spawns_a_host_and_the_next_one_reuses_it(make_client, keep_home) -> None:
    client = make_client()
    inline = Inline()
    key = _key()
    first = client.decide(key, PAYLOAD, keep_alive=30.0, inline=inline)
    second = client.decide(key, PAYLOAD, keep_alive=30.0, inline=inline)

    assert inline.calls == 0, "both calls must be served by the host"
    first_keep, second_keep = first["engine"]["keep"], second["engine"]["keep"]
    assert first_keep["served_by"] == "host" and second_keep["served_by"] == "host"
    assert first_keep["pid"] == second_keep["pid"]
    assert (first_keep["requests"], second_keep["requests"]) == (1, 2)
    assert second["engine"]["keep"]["fallback"] is None
    assert second["answers"] == {"q": {"type": "noul", "noul": 0.5, "probabilities": {"yes": 1.0}}}
    # one spawn, and the host is detached (its own session): a killed client cannot take it down
    spawns = [event for event in client.events if event[0] == "spawn"]
    assert spawns == [("spawn", first_keep["pid"])]
    assert os.getpgid(first_keep["pid"]) == first_keep["pid"]
    # the warm response paid no model load; the host's one-time load is named instead
    assert second["timings"]["model_load_ms"] == 0.0
    assert second_keep["model_load_ms"] == 12.5
    assert second_keep["keep_alive_s"] == 30.0 and second_keep["idle_left_s"] <= 30.0
    assert client.stop()["stopped"] is True
    assert state.read_record(keep_home) is None


@pytest.mark.needs_fork
def test_a_different_key_swaps_the_host_before_the_new_one_loads(make_client, keep_home) -> None:
    """RED pin (card t_7e24cea4): one model at a time — the old host frees the device first."""
    client = make_client()
    inline = Inline()
    first = client.decide(_key("a"), PAYLOAD, keep_alive=30.0, inline=inline)
    pid_a = first["engine"]["keep"]["pid"]
    assert state.pid_alive(pid_a)

    second = client.decide(_key("b"), PAYLOAD, keep_alive=30.0, inline=inline)
    pid_b = second["engine"]["keep"]["pid"]
    assert pid_b != pid_a and not state.pid_alive(pid_a)
    record = state.read_record(keep_home)
    assert record is not None and record.digest == _key("b").digest and record.pid == pid_b
    # the ordering is the point: stop A, *then* start B
    assert client.events.index(("stop", pid_a)) < client.events.index(("spawn", pid_b))
    client.stop()


@pytest.mark.needs_fork
def test_keep_alive_zero_stops_the_host_and_answers_inline(make_client, keep_home) -> None:
    client = make_client()
    inline = Inline()
    key = _key()
    warm = client.decide(key, PAYLOAD, keep_alive=30.0, inline=inline)
    pid = warm["engine"]["keep"]["pid"]
    cold = client.decide(key, PAYLOAD, keep_alive=0, inline=inline)
    assert inline.calls == 1, "`0` is today's behaviour: answer and unload"
    assert cold["engine"]["keep"]["served_by"] == "inline"
    assert cold["engine"]["keep"]["keep_alive_s"] == 0.0
    assert state.read_record(keep_home) is None and not state.pid_alive(pid)


@pytest.mark.needs_fork
def test_a_stale_socket_is_cleaned_up_and_the_call_still_answers(make_client, keep_home) -> None:
    """RED pin (card t_7e24cea4): a socket nobody listens on must not poison the call."""
    client = make_client()
    inline = Inline()
    key = _key()
    socket_file = state.socket_path(keep_home, key.digest)
    state.ensure_dir(keep_home)
    socket_file.write_text("", encoding="utf-8")            # dead, but on disk
    state.write_record(state.HostRecord(
        digest=key.digest, pid=os.getpid(), socket=str(socket_file), key=key.to_dict(),
        model="a", model_path=key.model_path, keep_alive=30.0, started_at=time.time(),
        loaded_at=time.time(), spec=str(state.spec_path(keep_home, key.digest)),
        log=str(state.log_path(keep_home, key.digest))), keep_home)

    body = client.decide(key, PAYLOAD, keep_alive=30.0, inline=inline)
    assert inline.calls == 1 and body["engine"]["keep"]["served_by"] == "inline"
    assert body["engine"]["keep"]["fallback"].startswith("transport")
    assert not socket_file.exists() and state.read_record(keep_home) is None
    # …and the *next* call is warm again: the cleanup was a restart, not a permanent downgrade
    again = client.decide(key, PAYLOAD, keep_alive=30.0, inline=inline)
    assert again["engine"]["keep"]["served_by"] == "host" and inline.calls == 1
    client.stop()


@pytest.mark.needs_fork
def test_a_host_that_dies_mid_decision_falls_back_inline_once(make_client, keep_home,
                                                              monkeypatch) -> None:
    """The ICD-SIGSEGV shape: the answer leaves with the host, the CLI still answers."""
    monkeypatch.setenv("TYPED_GGUF_KEEP_FAKE", "crash")
    client = make_client()
    inline = Inline()
    body = client.decide(_key(), PAYLOAD, keep_alive=30.0, inline=inline)
    assert inline.calls == 1 and body["engine"]["keep"]["served_by"] == "inline"
    assert body["engine"]["keep"]["fallback"].startswith("transport")
    assert state.read_record(keep_home) is None          # the crash is not left as a live record


@pytest.mark.needs_fork
def test_a_typed_error_from_the_host_is_reported_not_retried(make_client, keep_home,
                                                             monkeypatch) -> None:
    """A bad decision is the product's answer: report it. A dead socket is the box: retry inline."""
    monkeypatch.setenv("TYPED_GGUF_KEEP_FAKE", "error")
    client = make_client()
    inline = Inline()
    with pytest.raises(PrefillFailedError) as caught:
        client.decide(_key(), PAYLOAD, keep_alive=30.0, inline=inline)
    assert caught.value.code == "E_PREFILL_FAILED" and caught.value.exit_code == 3
    assert inline.calls == 0, "a typed answer is not re-run cold"
    record = state.read_record(keep_home)
    assert record is not None and state.pid_alive(record.pid)
    client.stop()


@pytest.mark.needs_fork
def test_status_reports_the_live_host_from_its_own_ledger(make_client, keep_home) -> None:
    client = make_client()
    client.decide(_key(), PAYLOAD, keep_alive=45.0, inline=Inline())
    status = client.status()
    assert status["state"] == "running" and status["requests"] >= 1
    assert status["keep_alive_s"] == 45.0 and 0.0 <= status["idle_left_s"] <= 45.0
    assert status["uptime_s"] >= 0.0 and status["model"] == "a"
    assert status["placement"]["note"] == "fake placement"
    assert status["devices"]["effective_backend"] == "cpu"
    assert status["key"]["model_path"] == "/models/a.gguf"
    client.stop()
    assert client.status()["state"] == "stopped"


def test_status_without_a_host_is_a_report_not_an_error(keep_home: pathlib.Path) -> None:
    client = client_module.Client(home=keep_home)
    status = client.status()
    assert status["state"] == "stopped" and status["pid"] is None
    assert status["keep_alive_s"] is None and status["idle_left_s"] is None


@pytest.mark.needs_fork
def test_status_of_a_dead_host_says_stale(keep_home: pathlib.Path) -> None:
    key = _key()
    state.ensure_dir(keep_home)
    socket_file = state.socket_path(keep_home, key.digest)
    socket_file.write_text("", encoding="utf-8")
    state.write_record(state.HostRecord(
        digest=key.digest, pid=2 ** 30, socket=str(socket_file), key=key.to_dict(), model="a",
        model_path=key.model_path, keep_alive=30.0, started_at=0.0, loaded_at=0.0,
        spec=str(state.spec_path(keep_home, key.digest)),
        log=str(state.log_path(keep_home, key.digest))), keep_home)
    status = client_module.Client(home=keep_home).status()
    assert status["state"] == "stale" and status["pid"] == 2 ** 30
    assert socket_file.exists(), "status is read-only: it reports, it does not clean up"


@pytest.mark.needs_fork
def test_stop_is_idempotent_and_cleans_a_stale_record(keep_home: pathlib.Path) -> None:
    client = client_module.Client(home=keep_home)
    assert client.stop() == {"stopped": False, "pid": None, "reason": "no host",
                             "cleaned": False}
    key = _key()
    state.write_record(state.HostRecord(
        digest=key.digest, pid=2 ** 30, socket=str(state.socket_path(keep_home, key.digest)),
        key=key.to_dict(), model="a", model_path=key.model_path, keep_alive=30.0,
        started_at=0.0, loaded_at=0.0, spec=str(state.spec_path(keep_home, key.digest)),
        log=str(state.log_path(keep_home, key.digest))), keep_home)
    report = client.stop()
    assert report["stopped"] is False and "was already gone" in report["reason"]
    assert report["cleaned"] is True and state.read_record(keep_home) is None


@pytest.mark.needs_fork
def test_the_spawn_writes_a_private_spec_and_a_log(make_client, keep_home) -> None:
    client = make_client()
    client.decide(_key(), PAYLOAD, keep_alive=30.0, inline=Inline())
    spawned = client.spawns[-1]
    spec_path = pathlib.Path(spawned["spec"])
    assert "--spec" in spawned["argv"] and spec_path.exists()
    assert oct(spec_path.stat().st_mode & 0o777) == "0o600"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec["key"]["model_path"] == "/models/a.gguf" and spec["keep_alive"] == 30.0
    assert spec["payload"]["questions"]["q"]["type"] == "noul"
    assert pathlib.Path(spawned["log"]).exists()
    assert spawned["env"]["TYPED_GGUF_HOME"] == str(keep_home)
    assert str(pathlib.Path(client_module.__file__).resolve().parents[2]) in \
        spawned["env"]["PYTHONPATH"].split(os.pathsep)
    client.stop()


def test_an_unreachable_host_is_never_waited_for_forever(make_client, keep_home,
                                                         monkeypatch) -> None:
    """A host that never becomes ready must surface as an inline answer, quickly and by name."""
    monkeypatch.setenv("TYPED_GGUF_KEEP_FAKE", "slowstart")
    client = make_client(spawn_timeout=0.5)
    inline = Inline()
    body = client.decide(_key(), PAYLOAD, keep_alive=30.0, inline=inline)
    assert inline.calls == 1 and body["engine"]["keep"]["served_by"] == "inline"
    fallback = body["engine"]["keep"]["fallback"]
    assert fallback.startswith("spawn") and "0.5" in fallback
    assert client.children == {}, "a host that never came up must not be left running"
    assert state.read_record(keep_home) is None


def test_the_client_refuses_to_guess_a_home(make_client, keep_home) -> None:
    """The ledger is per data home: a client without one would write into the user's own."""
    client = client_module.Client(home=keep_home)
    assert client.home == keep_home
    assert client_module.Client().home is None


@pytest.mark.needs_fork
def test_a_log_tail_is_quoted_when_the_host_cannot_start(make_client, keep_home,
                                                         monkeypatch) -> None:
    """A spawn that dies says *why*: the log tail travels in the fallback string."""
    client = make_client(host_command=(sys.executable, "-c",
                                       "import sys; print('E_RUNTIME_MISSING: no bundle here',"
                                       " file=sys.stderr); sys.exit(3)", "--spec"))
    inline = Inline()
    body = client.decide(_key(), PAYLOAD, keep_alive=30.0, inline=inline)
    assert inline.calls == 1 and body["engine"]["keep"]["served_by"] == "inline"
    fallback = body["engine"]["keep"]["fallback"]
    assert fallback.startswith("spawn") and "E_RUNTIME_MISSING" in fallback


def test_host_errors_are_typed_errors_again(keep_home: pathlib.Path) -> None:
    """The wire's `{code, message, exit_code}` is rebuilt into the product's own exception type."""
    error = client_module.typed_error_from_wire({"code": "E_MODEL_NOT_FOUND", "exit_code": 2,
                                                 "message": "E_MODEL_NOT_FOUND: gone"})
    assert isinstance(error, TypedGgufError) and error.exit_code == 2
    assert error.code == "E_MODEL_NOT_FOUND"
    generic = client_module.typed_error_from_wire({"code": "E_INTERNAL", "exit_code": 4,
                                                   "message": "boom"})
    assert generic.exit_code == 4 and generic.code == "E_INTERNAL"


# ------------------------------------------------------------------ who paid the load
@pytest.mark.needs_fork
def test_the_spawning_call_reports_the_load_it_waited_for(make_client, keep_home) -> None:
    """A-E4-1 (card t_7e24cea4): the cold answer carries the load the host paid for it.

    A host answers `timings.model_load_ms: 0.0` — the session *it* opened paid no load (the model
    has been resident since before this request). But the call that **spawned** that host waited
    for exactly that load, so its own timings must report it; the call that found the host already
    resident keeps reporting 0.0. That difference is the card's cold-vs-warm claim.
    """
    client = make_client()
    inline = Inline()
    key = _key()
    cold = client.decide(key, PAYLOAD, keep_alive=30.0, inline=inline)
    warm = client.decide(key, PAYLOAD, keep_alive=30.0, inline=inline)
    host_load = cold["engine"]["keep"]["model_load_ms"]
    assert host_load == 12.5, "the fake host's one-time load (tests/fake_keep_host.py)"
    assert cold["timings"]["model_load_ms"] == host_load
    assert warm["timings"]["model_load_ms"] == 0.0
    assert warm["engine"]["keep"]["model_load_ms"] == host_load, "the host still remembers it"


def test_an_inline_answer_keeps_its_own_load_number(make_client, keep_home, monkeypatch) -> None:
    """No host, no credit: an inline answer reports the load it paid itself (5.0 in the fake)."""
    monkeypatch.setenv("TYPED_GGUF_KEEP_FAKE", "slowstart")
    client = make_client(spawn_timeout=0.5)
    body = client.decide(_key(), PAYLOAD, keep_alive=30.0, inline=Inline())
    assert body["engine"]["keep"]["served_by"] == "inline"
    assert body["timings"]["model_load_ms"] == 5.0


# ------------------------------------------------------------------ the reporting verb
def test_status_of_a_host_that_cannot_answer_says_unresponsive(keep_home: pathlib.Path) -> None:
    """A host alive but not answering *now* is a state to report — never a crash.

    `keep status` is a health check, and a host serves one request at a time (SPEC 2.12): a ping
    that lands while the host is busy waits and then gives up. That give-up has to come back as
    `unresponsive` with its reason; it must not escape `status()` as `E_INTERNAL: TransportError`
    (card t_7e24cea4 — the live gates hit it right after a `kill -9` mid-request).
    """
    client = client_module.Client(home=keep_home, ping_timeout=0.2)
    key = _key()
    state.ensure_dir(keep_home)
    socket_file = state.socket_path(keep_home, key.digest)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)   # accepts, never answers
    listener.bind(str(socket_file))
    listener.listen(4)
    try:
        state.write_record(state.HostRecord(
            digest=key.digest, pid=os.getpid(), socket=str(socket_file), key=key.to_dict(),
            model="a", model_path=key.model_path, keep_alive=30.0, started_at=0.0, loaded_at=0.0,
            spec=str(state.spec_path(keep_home, key.digest)),
            log=str(state.log_path(keep_home, key.digest))), keep_home)
        status = client.status()
    finally:
        listener.close()
    assert status["state"] == "unresponsive"
    # whichever way the host failed to answer (a read that gave up reads as an empty stream),
    # the report *says so* instead of raising the transport error at the caller
    assert "TransportError" in status["detail"]
    assert status["pid"] == os.getpid(), "the report keeps the record's own half"
