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
import subprocess
import sys
import threading
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
def test_two_simultaneous_cold_callers_both_answer_and_one_host_survives(
        make_client, keep_home) -> None:
    """RED pin (card t_9249bb0c): two cold callers racing one data home — every caller answers.

    The live finding: `keep stop`, then two identical `ask`s fired at once. One pays the cold start
    and answers; the other died in ~140–215 ms with `E_INTERNAL: FileNotFoundError … .spec.json.tmp
    -> .spec.json` — a raw traceback on a designed path, no `--out`, no inline fallback, no typed
    code. Two racers write the *same* staging file for the spec (and, when they are different
    states, for the record), and the winner's `os.replace` removes it under the loser's feet.

    What the gate demands after the fix: no caller raises, every caller gets an *answer*, whoever
    is not served by the ledger's host says so with the designed fallback (`spawn:` / `transport:`),
    and exactly one host is left running — SPEC 2.12's "one host per data home" survives the race.
    """
    racers = 2
    clients = [make_client() for _ in range(racers)]
    inline = [Inline() for _ in range(racers)]
    key = _key()
    start = threading.Barrier(racers)
    answers: list[dict] = []
    failures: list[BaseException] = []

    def race(index: int) -> None:
        start.wait(30.0)
        try:
            answers.append(clients[index].decide(key, PAYLOAD, keep_alive=30.0,
                                                 inline=inline[index]))
        except BaseException as exc:                  # noqa: BLE001 - the finding, not a plan
            failures.append(exc)

    threads = [threading.Thread(target=race, args=(index,), name=f"racer-{index}")
               for index in range(racers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(180.0)
    assert failures == [], f"a cold caller died on a designed path: {[repr(e) for e in failures]}"
    assert len(answers) == racers, "every racing caller must come back with an answer"
    served = [body["engine"]["keep"]["served_by"] for body in answers]
    assert set(served) <= {"host", "inline"}, served
    for body in answers:
        assert body["answers"], "an answer, not an empty stub"
        if body["engine"]["keep"]["served_by"] == "inline":
            fallback = body["engine"]["keep"]["fallback"] or ""
            assert fallback.startswith(("spawn", "transport")), fallback
    # the ledger names the one host that is still there, and it answers on its own socket
    record = state.read_record(keep_home)
    assert record is not None and state.alive(record), "a race left no usable host"
    assert client_module._listening(record.socket), record.socket
    spawned = [spawn["pid"] for client in clients for spawn in client.spawns]
    alive = [pid for pid in spawned if state.pid_alive(pid)]
    assert alive == [record.pid], (
        f"the race left {len(alive)} host(s) for one data home ({alive}, ledger {record.pid})"
        f" — SPEC 2.12 allows one")


def test_the_race_gate_waits_a_bounded_while_for_the_loser_to_leave() -> None:
    """Pin (card t_e9fbe07f): the race gate waits out a loser — and only for a bounded while.

    The gate above used to sample `state.pid_alive` the instant both answers were in, which is a
    photo-finish rather than a check: measured on this box the losing racer left the process table
    0.2–0.3 ms *after* that sample, and one loaded box turned the same shape red 20/25 runs (the
    coordinator's: 12/25 solo, 3/3 inside the full suite). The wait carries two semantics that must
    stay apart, and both are exercised here on a synthetic predicate — no fork, no spawned process,
    no waiting on a real clock — so this pin reads the same on a starved box as on an idle one:

    * a loser that leaves the process table a few polls later is waited for, and the gate then
      reads exactly one host (the shape the flake was);
    * a loser that *never* leaves still fails, at the bound and not past it — a host that survived
      the race must never pass, or "the gate waits" would license a second resident model;
    * the bound itself stays ≤5 s in ≤50 ms steps, so a wedged loser cannot turn a red gate into a
      hung one.
    """
    winner, loser = 101, 202
    polls = {"loser": 0}

    def alive(pid: int) -> bool:
        if pid == winner:
            return True
        polls["loser"] += 1
        return polls["loser"] <= 3               # the loser is still unwinding for three polls

    assert _one_host_survives([winner, loser], winner, alive=alive, timeout=5.0,
                              step=0.001) == [winner]
    assert polls["loser"] >= 4, "the wait must poll for the loser, not assume it has left"

    started = time.monotonic()
    living = _one_host_survives([winner, loser], winner, alive=lambda _pid: True, timeout=0.15,
                                step=0.01)
    elapsed = time.monotonic() - started
    assert living == [winner, loser], "a host that never exits must still fail the gate"
    assert 0.15 <= elapsed < 1.0, f"the wait is bounded, but it ran for {elapsed:.3f}s"

    assert 0.0 < _RACE_EXIT_WAIT_S <= 5.0, "the bound is the card's: ≤5 s (t_e9fbe07f)"
    assert 0.0 < _RACE_EXIT_POLL_S <= 0.05, "25–50 ms steps"


@pytest.mark.needs_fork
def test_simultaneous_cold_callers_for_two_states_never_share_a_record(
        make_client, keep_home) -> None:
    """The same race with two *different* states: both answer *now*, not after an idle window.

    Different keys mean different specs and sockets, but the record (`keep/host.json`) is one file
    for the whole data home, so the two spawning hosts stage over each other. The card's finding
    reports the raw `E_INTERNAL` here too. The timing is part of the claim: before the fix the loser
    sat next to the winner until its own `keep_alive` ran out (30 s here — minutes in the live
    shape), because a record for another digest was simply ignored; a caller that lost the race is
    now stopped by name and answers inline instead.
    """
    racers = 2
    clients = [make_client() for _ in range(racers)]
    inline = [Inline() for _ in range(racers)]
    keys = [_key("a"), _key("b")]
    start = threading.Barrier(racers)
    answers: list[dict] = []
    failures: list[BaseException] = []

    def race(index: int) -> None:
        start.wait(30.0)
        try:
            answers.append(clients[index].decide(keys[index], PAYLOAD, keep_alive=30.0,
                                                 inline=inline[index]))
        except BaseException as exc:                  # noqa: BLE001 - the finding, not a plan
            failures.append(exc)

    threads = [threading.Thread(target=race, args=(index,), name=f"racer-{index}")
               for index in range(racers)]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(180.0)
    elapsed = time.monotonic() - started
    assert failures == [], f"a cold caller died on a designed path: {[repr(e) for e in failures]}"
    assert len(answers) == racers
    assert elapsed < 15.0, (
        f"the losers of the race waited {elapsed:.1f}s for another host's keep-alive window "
        f"instead of answering inline (keep_alive was 30 s)")
    assert set(body["engine"]["keep"]["served_by"] for body in answers) <= {"host", "inline"}
    record = state.read_record(keep_home)
    if record is not None:
        assert record.digest in {key.digest for key in keys}, record.digest
        assert state.alive(record), "the ledger kept a record of a host that is gone"


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


# ------------------------------------------------------------------ the stop path
def test_listening_probes_the_socket_without_waking_anyone(keep_home: pathlib.Path) -> None:
    """The probe that runs before every spawn: no file → False, dead file → False, up → True.

    It exists to answer "is something *there right now*", not "does the file exist": a host that
    died leaves its socket behind, and a spawn must not be skipped because of a corpse
    (card t_7e24cea4 — 14 mutants of this helper were in the sweep's "no tests" bucket). The three
    paths are real sockets in the gate home, which binds whatever `TMPDIR` the session runs under
    (card t_c3195a5c).
    """
    missing = keep_home / "nothing.sock"
    assert client_module._listening(missing) is False
    dead = keep_home / "dead.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(dead))
    listener.close()                     # the file stays behind, nothing listens on it any more
    assert client_module._listening(dead) is False
    live = keep_home / "live.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(live))
    listener.listen(4)
    try:
        assert client_module._listening(live) is True
    finally:
        listener.close()


@pytest.mark.needs_fork
def test_wait_pid_gone_tells_a_live_process_from_a_gone_one() -> None:
    """`keep stop`'s primitive: False while the pid is there, True once it is gone.

    A **zombie** counts as gone: the host exited, its parent (or init) has it to reap, and between
    those moments `os.kill(pid, 0)` would still succeed — `keep status` must not report a corpse as
    a resident model.
    """
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert state.pid_alive(child.pid) is True, "the fixture process must be alive to start"
        assert state.wait_pid_gone(child.pid, timeout=0.2) is False, "alive → not gone"
        child.kill()
        assert state.wait_pid_gone(child.pid, timeout=5.0) is True
    finally:
        child.kill()
        child.wait(timeout=10.0)
    assert state.wait_pid_gone(0, timeout=0.1) is True, "pid 0 is not a process to wait for"
    assert state.wait_pid_gone(child.pid, timeout=0.1) is True, "an exited pid is gone for good"


def _write_foreign_host(home: pathlib.Path, pid: int, key: identity.KeepKey) -> None:
    """A record that points at a process this client has no `Popen` for (another client's host)."""
    state.ensure_dir(home)
    socket_file = state.socket_path(home, key.digest)
    state.write_record(state.HostRecord(
        digest=key.digest, pid=pid, socket=str(socket_file), key=key.to_dict(), model="a",
        model_path=key.model_path, keep_alive=30.0, started_at=0.0, loaded_at=0.0,
        spec=str(state.spec_path(home, key.digest)),
        log=str(state.log_path(home, key.digest))), home)


@pytest.mark.needs_fork
def test_stop_of_a_host_this_client_did_not_spawn_goes_through_the_pid(keep_home) -> None:
    """`keep stop` on a *foreign* host: no `Popen` to wait on, so the pid itself is waited for."""
    key = _key()
    state.ensure_dir(keep_home)
    socket_file = state.socket_path(keep_home, key.digest)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_file))
    listener.listen(4)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        _write_foreign_host(keep_home, child.pid, key)
        client = client_module.Client(home=keep_home, stop_grace=5.0)
        result = client.stop()
    finally:
        listener.close()
        child.kill()
        child.wait(timeout=10.0)
    assert result == {"stopped": True, "pid": child.pid, "reason": "stopped (SIGTERM)",
                      "cleaned": True}
    assert state.read_record(keep_home) is None, "the ledger entry goes with the host"


@pytest.mark.needs_fork
def test_stop_escalates_to_sigkill_when_the_host_ignores_sigterm(keep_home) -> None:
    """A wedged host is not a host: after the grace period the pid is killed and `stop` says so.

    The record has to *look like* a host for the signal path to run at all (card t_a4ebcd36: a pid
    the record cannot prove is a host is never signalled) — a wedged host is wedged inside a
    decision, not deaf on its socket, so the gate binds a listener for it.
    """
    key = _key()
    state.ensure_dir(keep_home)
    socket_file = state.socket_path(keep_home, key.digest)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_file))
    listener.listen(4)
    # the child prints `ready` only *after* it ignores SIGTERM: without that handshake the SIGTERM
    # can land during interpreter start-up and kill it the default way (a flaky 0.4 s grace)
    child = subprocess.Popen([sys.executable, "-c",
                              "import signal, sys, time;"
                              " signal.signal(signal.SIGTERM, signal.SIG_IGN);"
                              " print('ready', flush=True); time.sleep(60)"],
                             stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout is not None and child.stdout.readline().strip() == "ready"
        _write_foreign_host(keep_home, child.pid, key)
        client = client_module.Client(home=keep_home, stop_grace=0.4)
        result = client.stop()
    finally:
        listener.close()
        child.kill()
        child.wait(timeout=10.0)
        if child.stdout is not None:
            child.stdout.close()          # an unclosed pipe is an unraisable warning (= error) here
    assert result["stopped"] is True, "the escalation is what makes `stop` a verb"
    assert result["reason"] == "killed (SIGKILL)"
    assert state.read_record(keep_home) is None


def test_stop_refuses_a_record_that_points_at_this_very_process(keep_home) -> None:
    """A corrupted record must never make the CLI signal *itself* — the reason is the answer."""
    key = _key()
    _write_foreign_host(keep_home, os.getpid(), key)
    client = client_module.Client(home=keep_home)
    result = client.stop()
    assert result["stopped"] is False
    assert result["pid"] == os.getpid()
    assert result["reason"] == "the record points at this very process; refusing to signal it"
    assert os.getpid() == result["pid"], "the process under test is obviously still here"


def _leave_debris(home: pathlib.Path, key: identity.KeepKey) -> pathlib.Path:
    """What a `kill -9`'d host leaves behind: a record, and a socket file nobody listens on.

    Bound for real and then closed, so the path *exists* (the record's `alive()` half) while a
    connect on it is refused — the shape the reviewer reproduced with a decoy process.
    """
    state.ensure_dir(home)
    socket_file = state.socket_path(home, key.digest)
    dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead.bind(str(socket_file))
    dead.close()
    return socket_file


@pytest.mark.needs_fork
def test_stop_never_signals_a_pid_the_record_cannot_prove_is_a_host(keep_home) -> None:
    """RED pin (card t_a4ebcd36): a stale record's pid may be somebody else's by now.

    The ledger's own rule is that *a record is a claim, not a fact* — `alive()` checks the pid
    **and** the socket — and `stop()` is the one verb that acts on the record. It used to SIGTERM
    on `pid_alive` alone, so a `kill -9`'d host's recycled pid went to whatever now owned it (the
    re-gate watched a decoy `sleep 600` go R -> Z). Nothing answers on this record's socket, so no
    signal may be sent, the debris must be cleaned up, and the report has to say which it was.
    """
    key = _key()
    socket_file = _leave_debris(keep_home, key)
    decoy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        _write_foreign_host(keep_home, decoy.pid, key)
        client = client_module.Client(home=keep_home, stop_grace=0.4)
        result = client.stop()
        time.sleep(0.2)                       # a signal would have landed by now
        assert state.pid_alive(decoy.pid) is True, \
            "the decoy was signalled by a record that never proved its pid was a host"
    finally:
        decoy.kill()
        decoy.wait(timeout=10.0)
    assert result["stopped"] is False and result["cleaned"] is True
    assert state.read_record(keep_home) is None and not socket_file.exists()
    assert ("listening" in result["reason"] and "nothing was signalled" in result["reason"]), \
        result["reason"]


@pytest.mark.needs_fork
def test_a_swap_over_debris_loads_the_new_host_instead_of_refusing(make_client, keep_home) -> None:
    """A record nobody answers on is *not* a resident model: a different key still gets a host.

    The swap path refuses to load a second model only when it could not stop a *host*. Refusing on
    the strength of a recycled pid (which is all a debris record carries) would turn the cleanup
    into a permanent inline downgrade — and the pid under it would have been killed to get there.
    """
    key_a = _key("a")
    socket_file = _leave_debris(keep_home, key_a)
    decoy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        _write_foreign_host(keep_home, decoy.pid, key_a)
        client = make_client()
        body = client.decide(_key("b"), PAYLOAD, keep_alive=30.0, inline=Inline())
        assert body["engine"]["keep"]["served_by"] == "host", \
            f"the swap refused to load over debris: {body['engine']['keep']}"
        assert state.pid_alive(decoy.pid) is True
        host_pid = body["engine"]["keep"]["pid"]
        assert host_pid != decoy.pid
    finally:
        decoy.kill()
        decoy.wait(timeout=10.0)
    # the debris went with the swap: the socket file is gone and the ledger entry is the new host's
    assert not socket_file.exists()
    record = state.read_record(keep_home)
    assert record is not None and record.pid == host_pid


@pytest.mark.needs_fork
def test_a_swap_refuses_when_the_record_it_cannot_stop_still_answers(
        make_client, keep_home) -> None:
    """The other half of the gate: a record whose socket *answers* is not debris (card t_a4ebcd36).

    `held` is what separates the two: the swap loads its own host over debris, and refuses in front
    of a record it could not stop *whose socket answers a probe*. Here that record points at the
    process running the gate, which `stop()` refuses to signal on purpose — so nothing was stopped
    and nothing may be started next to it: the call answers inline and names the reason instead of
    writing a second host under the same identity (the mutation that drops `held` survives the
    sweep's four gate files, so this pin is what holds the branch in place).
    """
    key_a = _key("a")
    state.ensure_dir(keep_home)
    socket_file = state.socket_path(keep_home, key_a.digest)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_file))
    listener.listen(4)                     # a host is really there: the probe answers
    try:
        _write_foreign_host(keep_home, os.getpid(), key_a)
        client = make_client()
        body = client.decide(_key("b"), PAYLOAD, keep_alive=30.0, inline=Inline())
        keep = body["engine"]["keep"]
        assert keep["served_by"] == "inline", keep
        assert "refusing to load a second model" in (keep.get("fallback") or ""), keep
        assert client.spawns == [], "a second host was started next to something that answered"
    finally:
        listener.close()
    # `stop()` cleared the ledger entry of the record it refused to signal (the pre-existing
    # teardown rule — this card's change is *what gets a signal*, not what gets cleaned).
    assert state.read_record(keep_home) is None
