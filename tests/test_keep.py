"""The keep-alive identity: durations, precedence, the host key (SPEC 2.12, card t_7e24cea4).

Offline and model-free: everything here is pure arithmetic and file bookkeeping, so it runs on
any box. The live half (a real model staying resident) is `tests/test_keep_live.py`.
"""
from __future__ import annotations

import json
import os
import pathlib
import socket

import pytest

from typed_gguf import schema
from typed_gguf.errors import UserError
from typed_gguf.keep import identity, state


# ------------------------------------------------------------------ durations
def test_the_default_keep_alive_is_ten_minutes() -> None:
    """The request's own number: a model used now stays resident for 10 minutes."""
    assert float(identity.DEFAULT_KEEP_ALIVE) == 600.0


@pytest.mark.parametrize("text,seconds", [
    ("600", 600.0), ("10m", 600.0), ("5s", 5.0), ("1h", 3600.0), ("0", 0.0), ("0s", 0.0),
    ("90m", 5400.0), ("2.5s", 2.5), ("  10m  ", 600.0),
])
def test_durations_are_read_as_seconds(text: str, seconds: float) -> None:
    assert identity.parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "  ", "ten minutes", "-5", "10x", "m", "1e9"])
def test_a_bad_duration_is_a_user_error_that_names_the_source(text: str) -> None:
    with pytest.raises(UserError) as caught:
        identity.parse_duration(text, source="$TYPED_GGUF_KEEP_ALIVE")
    assert caught.value.code == "E_UNKNOWN_KEY"
    assert "$TYPED_GGUF_KEEP_ALIVE" in str(caught.value)


def test_the_precedence_chain_is_flag_then_env_then_default() -> None:
    """RED pin (card t_7e24cea4): CLI flag wins over env; env wins over the default."""
    environ = {identity.KEEP_ALIVE_ENV: "5m"}
    assert identity.resolve_keep_alive("1m", environ=environ) == 60.0
    assert identity.resolve_keep_alive(None, environ=environ) == 300.0
    assert identity.resolve_keep_alive(None, environ={}) == float(identity.DEFAULT_KEEP_ALIVE)
    # an explicit 0 (the "no host" switch) must survive the chain in both spellings
    assert identity.resolve_keep_alive("0", environ=environ) == 0.0
    assert identity.resolve_keep_alive(0, environ=environ) == 0.0


def test_an_unparsable_env_names_the_variable() -> None:
    with pytest.raises(UserError) as caught:
        identity.resolve_keep_alive(None, environ={identity.KEEP_ALIVE_ENV: "soon"})
    assert identity.KEEP_ALIVE_ENV in str(caught.value)


def test_keep_alive_needs_unix_sockets() -> None:
    assert identity.supported() is True          # this box (Linux)
    assert identity.supported(platform="win32") is False
    if not hasattr(socket, "AF_UNIX"):           # pragma: no cover - Linux always has it
        assert identity.supported() is False


# ------------------------------------------------------------------ the host key
def _request(**options: object) -> schema.Request:
    return schema.parse_request({"state": "Billing is down.", "options": dict(options),
                                 "questions": {"q": {"type": "noul", "instructions": "page?"}}})


def test_the_key_is_the_model_plus_the_placement_options() -> None:
    key = identity.KeepKey.of(_request(n_ctx=4096, threads=4, backend="vulkan"),
                              model_path="/models/a.gguf", model_sha="deadbeef",
                              fit={"fit_enabled": True, "fit_cache": True})
    assert key.model_path == "/models/a.gguf" and key.model_sha == "deadbeef"
    assert (key.backend, key.n_ctx, key.threads) == ("vulkan", 4096, 4)
    assert key.fit is True and key.fit_cache is True
    # the digest is stable, and the key says itself what it was made of
    assert key.digest == identity.KeepKey.of(
        _request(n_ctx=4096, threads=4, backend="vulkan"), model_path="/models/a.gguf",
        model_sha="deadbeef", fit={"fit_enabled": True, "fit_cache": True}).digest
    assert len(key.digest) == 16 and key.digest.isalnum()
    assert "backend=vulkan" in key.describe() and "n_ctx=4096" in key.describe()


@pytest.mark.parametrize("changed", [
    {"n_ctx": 8192}, {"threads": 8}, {"backend": "cpu"}, {"kv_type": "q8_0"},
    {"n_seq_max": 9},
])
def test_every_placement_option_changes_the_key(changed: dict[str, object]) -> None:
    """RED pin: a second request that would load the model differently gets its own host."""
    base = identity.KeepKey.of(_request(n_ctx=4096, threads=4, kv_type="f16"),
                               model_path="/models/a.gguf", model_sha="deadbeef",
                               fit={"fit_enabled": True})
    other = identity.KeepKey.of(_request(**{"n_ctx": 4096, "threads": 4, "kv_type": "f16",
                                            **changed}),
                                model_path="/models/a.gguf", model_sha="deadbeef",
                                fit={"fit_enabled": True})
    assert base.digest != other.digest


def test_a_different_model_or_fit_call_changes_the_key() -> None:
    base = identity.KeepKey.of(_request(), model_path="/models/a.gguf", model_sha="deadbeef",
                               fit={"fit_enabled": True})
    assert base.digest != identity.KeepKey.of(
        _request(), model_path="/models/a.gguf", model_sha="cafe", fit={"fit_enabled": True}).digest
    assert base.digest != identity.KeepKey.of(
        _request(), model_path="/models/b.gguf", model_sha="deadbeef",
        fit={"fit_enabled": True}).digest
    assert base.digest != identity.KeepKey.of(
        _request(), model_path="/models/a.gguf", model_sha="deadbeef",
        fit={"fit_enabled": False}).digest
    assert base.digest != identity.KeepKey.of(
        _request(), model_path="/models/a.gguf", model_sha="deadbeef",
        fit={"fit_enabled": True, "fit_target_mb": 2048}).digest
    assert base.digest != identity.KeepKey.of(
        _request(), model_path="/models/a.gguf", model_sha="deadbeef",
        fit={"fit_enabled": True, "fit_ctx": 8192}).digest
    assert base.digest != identity.KeepKey.of(
        _request(), model_path="/models/a.gguf", model_sha="deadbeef",
        fit={"fit_enabled": True, "fit_cache": False}).digest


def test_the_key_ignores_what_cannot_change_the_placement() -> None:
    """`--format`, the questions, the readout… are per-request: they must not split a host."""
    base = identity.KeepKey.of(_request(n_ctx=4096), model_path="/models/a.gguf",
                               model_sha="deadbeef", fit={"fit_enabled": True})
    other = identity.KeepKey.of(
        _request(n_ctx=4096, readout="single_token", confidence_mode="entropy", cue="two_step",
                 state_id="other", strict=True, temperature=0.5, save_state=True,
                 coverage_floor=0.4, length_norm=0.5, max_waves=2, thinking=True),
        model_path="/models/a.gguf", model_sha="deadbeef", fit={"fit_enabled": True})
    assert base.digest == other.digest
    # the questions and the state are per-request too (the host keys the *model*, not the ticket)
    tickets = schema.parse_request({"state": "Something else entirely.",
                                    "format": "typesafe",
                                    "questions": {"other": {"type": "score",
                                                            "criteria": ["a", "b"]}},
                                    "options": {"n_ctx": 4096}})
    assert identity.KeepKey.of(tickets, model_path="/models/a.gguf", model_sha="deadbeef",
                               fit={"fit_enabled": True}).digest == base.digest


def test_the_key_round_trips_through_json() -> None:
    """The host reads its identity back out of the spec file, so the encoding must be lossless."""
    key = identity.KeepKey.of(_request(n_ctx=2048, threads=2, backend="cpu"),
                              model_path="/models/a.gguf", model_sha="",
                              fit={"fit_enabled": False, "fit_target_mb": 4096})
    assert identity.KeepKey.from_dict(key.to_dict()) == key
    assert identity.KeepKey.from_dict(json.loads(json.dumps(key.to_dict()))).digest == key.digest


# ------------------------------------------------------------------ the record
def _record(tmp_path: pathlib.Path, **overrides: object) -> state.HostRecord:
    key = identity.KeepKey.of(_request(), model_path="/models/a.gguf", model_sha="dead",
                              fit={"fit_enabled": True})
    fields: dict[str, object] = dict(
        digest=key.digest, pid=os.getpid(), socket=str(tmp_path / "keep" / f"{key.digest}.sock"),
        key=key.to_dict(), model="a", model_path="/models/a.gguf", keep_alive=600.0,
        started_at=100.0, loaded_at=100.0, spec=str(tmp_path / "keep" / "x.spec.json"),
        log=str(tmp_path / "keep" / "x.log"), version="0.1.0")
    fields.update(overrides)
    return state.HostRecord(**fields)  # type: ignore[arg-type]


def test_the_keep_dir_lives_in_the_data_home(monkeypatch: pytest.MonkeyPatch,
                                             tmp_path: pathlib.Path) -> None:
    monkeypatch.setenv("TYPED_GGUF_HOME", str(tmp_path / "home"))
    assert state.keep_dir() == tmp_path / "home" / "keep"
    assert state.keep_dir(tmp_path / "other") == tmp_path / "other" / "keep"
    made = state.ensure_dir(tmp_path / "x")
    assert made.is_dir() and (made.stat().st_mode & 0o777) == 0o700


def test_a_socket_path_that_cannot_fit_is_refused_not_truncated() -> None:
    """`sun_path` is 108 bytes: a deep home must be named, never silently mangled."""
    with pytest.raises(UserError) as caught:
        state.socket_path(pathlib.Path("/" + "x" * 120), "ab" * 8)
    assert "sun_path" in str(caught.value) or "108" in str(caught.value)


def test_the_record_round_trips_and_is_private(tmp_path: pathlib.Path) -> None:
    record = _record(tmp_path)
    written = state.write_record(record, tmp_path)
    assert written == state.state_path(tmp_path)
    assert state.read_record(tmp_path) == record
    assert (written.stat().st_mode & 0o777) == 0o600
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["schema"] == state.RECORD_SCHEMA and payload["pid"] == os.getpid()


def test_a_half_written_record_reads_as_no_host(tmp_path: pathlib.Path) -> None:
    """A killed writer must not become an exception on the next call."""
    path = state.state_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"schema": "typed_gguf.keep/v1", "pid":', encoding="utf-8")
    assert state.read_record(tmp_path) is None


def test_a_record_of_another_schema_reads_as_no_host(tmp_path: pathlib.Path) -> None:
    path = state.state_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "typed_gguf.keep/v0", "pid": 1}), encoding="utf-8")
    assert state.read_record(tmp_path) is None


def test_liveness_is_pid_and_socket_checked(tmp_path: pathlib.Path) -> None:
    record = _record(tmp_path)
    assert state.pid_alive(os.getpid()) is True
    assert state.pid_alive(0) is False
    assert state.pid_alive(2 ** 30) is False                       # no such pid
    assert state.pid_alive(-1) is False                            # never signal a process group
    # a live pid without a socket is *stale*: the host died between two calls
    assert state.alive(record) is False
    path = pathlib.Path(record.socket)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    assert state.alive(record) is True
    dead = _record(tmp_path, pid=2 ** 30)
    assert state.alive(dead) is False


def test_clear_record_removes_the_record_socket_and_spec(tmp_path: pathlib.Path) -> None:
    """RED pin (card t_7e24cea4): a stale socket must be cleaned up, not fought over."""
    record = _record(tmp_path)
    state.write_record(record, tmp_path)
    sock = pathlib.Path(record.socket)
    sock.parent.mkdir(parents=True, exist_ok=True)
    sock.write_text("", encoding="utf-8")
    pathlib.Path(record.spec).write_text("{}", encoding="utf-8")
    assert state.clear_record(tmp_path) is True
    assert state.read_record(tmp_path) is None
    assert not sock.exists() and not pathlib.Path(record.spec).exists()
    assert state.clear_record(tmp_path) is False                   # idempotent


def test_clear_record_leaves_another_hosts_record_alone(tmp_path: pathlib.Path) -> None:
    record = _record(tmp_path, digest="otherdigest")
    state.write_record(record, tmp_path)
    assert state.clear_record(tmp_path, digest="notthat") is False
    assert state.read_record(tmp_path) == record


# ------------------------------------------------------------------ the host record's own report
def test_host_status_reports_idle_left_from_its_own_clock(tmp_path: pathlib.Path) -> None:
    record = _record(tmp_path, keep_alive=60.0)
    pathlib.Path(record.socket).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(record.socket).write_text("", encoding="utf-8")     # a live host's socket
    report = state.host_status(record, now=130.0, placement={"note": "cpu"})
    assert report["state"] == "running" and report["pid"] == os.getpid()
    assert report["uptime_s"] == pytest.approx(30.0)
    assert report["idle_left_s"] == pytest.approx(30.0)
    assert report["keep_alive_s"] == 60.0 and report["placement"] == {"note": "cpu"}
    # a request moves the deadline: the timer counts from the last use, not from the load
    used = state.host_status(_record(tmp_path, keep_alive=60.0, last_used=125.0), now=130.0)
    assert used["idle_left_s"] == pytest.approx(55.0)


def test_host_status_counts_a_stale_host_as_stale(tmp_path: pathlib.Path) -> None:
    record = _record(tmp_path, pid=2 ** 30, last_used=5.0)
    assert state.host_status(record)["state"] == "stale"


def test_host_records_carry_the_session_evidence(tmp_path: pathlib.Path) -> None:
    """The placement the engine log proves, not the flags that were passed."""
    record = _record(tmp_path, requests=3, last_used=7.0,
                     placement={"note": "fit plan: 32 layer(s) offloaded", "n_gpu_layers": 32},
                     devices={"devices": ["Vulkan0"], "compute_buffers": {"Vulkan0": 9},
                              "effective_backend": "vulkan"})
    payload = record.to_dict()
    assert payload["requests"] == 3 and payload["last_used"] == 7.0
    assert payload["devices"]["effective_backend"] == "vulkan"
    assert state.HostRecord.from_dict(json.loads(json.dumps(payload))) == record
