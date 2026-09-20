"""The CLI's half of the warm host: `keep status|stop`, `--keep-alive`, and the two halves of the
decision path (SPEC 2.12, card t_7e24cea4).

Offline: the host process itself is exercised in `tests/test_keep_client.py`, the server in
`tests/test_keep_host.py`; here it is the *surface* — flags, precedence, the split of
`decide_payload` into "prepare once" and "decide on a loaded handle", and what a Windows box does.
"""
from __future__ import annotations

import contextlib
import json
import pathlib
import sys
from collections.abc import Mapping
from typing import Any

import pytest

from typed_gguf import cli
from typed_gguf.keep import identity, state

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fake_engine import FakeSession, biased_row  # noqa: E402


def _payload(*, model: str | None = None, **options: object) -> dict:
    body: dict = {"state": "Billing is down.",
                  "questions": {"q": {"type": "choice",
                                      "criteria": {"billing": None, "tech": None}}},
                  "options": dict(options)}
    if model is not None:                              # `model` is a *top-level* request field
        body["model"] = model
    return body


def _model_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """A file that `_resolve_model` accepts as a path (the model itself is faked out)."""
    from tests.test_calibration import _model_file as builder

    return builder(tmp_path)


# ------------------------------------------------------------------ the surface
def test_keep_is_a_command_with_status_and_stop(capsys) -> None:
    assert "keep" in cli.COMMANDS
    assert cli.MILESTONES["keep"] == "E4"
    code = cli.main(["keep", "--help"])
    out = capsys.readouterr().out
    assert code == 0 and "usage: typed-gguf keep" in out and "milestone: E4" in out
    assert "status" in out and "stop" in out
    # an unknown subcommand is the user's typo, not a silent no-op
    code = cli.main(["keep", "warm"])
    assert code == 2 and "E_UNKNOWN_KEY" in capsys.readouterr().err


def test_keep_alive_is_a_run_and_ask_flag() -> None:
    assert any(flag.startswith("--keep-alive") for flag in cli.COMMAND_HELP["run"])
    # `ask` documents the run flags by reference (it accepts every one of them)
    assert any("every `run` flag" in flag for flag in cli.COMMAND_HELP["ask"])
    value_flags = cli.ENGINE_VALUE_FLAGS + cli.FIT_VALUE_FLAGS
    positionals, options = cli._parse_args(["--keep-alive", "10m", "--state", "s"],
                                           value_flags=value_flags,
                                           bool_flags=cli.ENGINE_BOOL_FLAGS)
    assert positionals == [] and options["keep_alive"] == "10m"


def test_a_bad_keep_alive_is_a_user_error(keep_home: pathlib.Path, capsys) -> None:
    code = cli.main(["ask", "--state", "S", "--noul", "q=page?", "--keep-alive", "soon"])
    assert code == 2 and "E_UNKNOWN_KEY" in capsys.readouterr().err


# ------------------------------------------------------------------ status / stop
def test_keep_status_without_a_host_is_a_report(keep_home: pathlib.Path, capsys) -> None:
    assert cli.main(["keep", "status", "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["schema"] == "typed_gguf.keep.status/v1" and status["state"] == "stopped"
    assert status["pid"] is None and status["idle_left_s"] is None
    assert cli.main(["keep", "status"]) == 0
    assert "no warm host" in capsys.readouterr().out


def test_keep_stop_cleans_a_stale_record_and_reports_it(keep_home: pathlib.Path,
                                                        capsys) -> None:
    key = identity.KeepKey.of(None, model_path="/models/a.gguf")
    state.ensure_dir(keep_home)
    socket_file = state.socket_path(keep_home, key.digest)
    socket_file.write_text("", encoding="utf-8")
    state.write_record(state.HostRecord(
        digest=key.digest, pid=2 ** 30, socket=str(socket_file), key=key.to_dict(), model="a",
        model_path=key.model_path, keep_alive=600.0, started_at=0.0, loaded_at=0.0,
        spec=str(state.spec_path(keep_home, key.digest)),
        log=str(state.log_path(keep_home, key.digest))), keep_home)
    assert cli.main(["keep", "stop", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["stopped"] is False and report["cleaned"] is True
    assert "was already gone" in report["reason"]
    assert state.read_record(keep_home) is None and not socket_file.exists()
    # a second stop is still a report, still exit 0
    assert cli.main(["keep", "stop"]) == 0
    assert "no host" in capsys.readouterr().out


# ------------------------------------------------------------------ the warm path's wiring
class _Recorder:
    """A stand-in for `keep.client.Client`: records the call, answers with a host-shaped body."""

    calls: list[dict] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.home = kwargs.get("home")

    def decide(self, key, payload, *, keep_alive, inline, fit=None, model=""):
        type(self).calls.append({"key": key, "keep_alive": float(keep_alive),
                                 "fit": dict(fit or {}),
                                 "model": model, "home": self.kwargs.get("home")})
        return {"model": model, "engine": {"keep": {"served_by": "host", "pid": 1234}},
                "answers": {}, "usage": {}, "timings": {}, "warnings": []}

    def stop(self, **_kwargs: object) -> dict:
        return {"stopped": False, "pid": None, "reason": "no host", "cleaned": False}


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> type[_Recorder]:
    _Recorder.calls = []
    monkeypatch.setattr(cli.keep_client, "Client", _Recorder)
    monkeypatch.setattr(cli, "keep_key_for", lambda *a, **k: (
        identity.KeepKey.of(None, model_path="/models/a.gguf"), "a"))
    return _Recorder


def test_the_flag_beats_the_env_beats_the_default(keep_home: pathlib.Path,
                                                  recorder, monkeypatch) -> None:
    """RED pin (card t_7e24cea4): the documented precedence chain, through the real entry point."""
    monkeypatch.setenv(identity.KEEP_ALIVE_ENV, "5m")
    cli.decide_payload_warm(_payload(), keep_alive="1m", fit_enabled=False)
    assert recorder.calls[-1]["keep_alive"] == 60.0
    cli.decide_payload_warm(_payload(), fit_enabled=False)
    assert recorder.calls[-1]["keep_alive"] == 300.0
    monkeypatch.delenv(identity.KEEP_ALIVE_ENV)
    cli.decide_payload_warm(_payload(), fit_enabled=False)
    assert recorder.calls[-1]["keep_alive"] == float(identity.DEFAULT_KEEP_ALIVE)


def test_keep_alive_zero_answers_inline_and_spawns_nothing(keep_home: pathlib.Path,
                                                           recorder, monkeypatch) -> None:
    """`--keep-alive 0` is today's behaviour: answer and unload — no host is ever started."""
    monkeypatch.setattr(cli, "decide_payload", lambda *a, **k: (
        {"model": "a", "engine": {"backend": "cpu"}, "answers": {}, "usage": {}, "timings": {},
         "warnings": []}))
    body = cli.decide_payload_warm(_payload(), keep_alive=0, fit_enabled=False)
    assert recorder.calls == [] and body["model"] == "a"


def test_a_platform_without_unix_sockets_says_so_and_answers_inline(keep_home: pathlib.Path,
                                                                    recorder, monkeypatch,
                                                                    capsys) -> None:
    monkeypatch.setattr(cli.keep, "supported", lambda: False)
    monkeypatch.setattr(cli, "decide_payload", lambda *a, **k: (
        {"model": "a", "engine": {}, "answers": {}, "usage": {}, "timings": {}, "warnings": []}))
    cli.decide_payload_warm(_payload(), fit_enabled=False)
    err = capsys.readouterr().err
    assert "keep-alive" in err and "W_KEEP_UNAVAILABLE" in err
    assert recorder.calls == []


def test_keep_key_for_resolves_the_registry_sha_and_the_placement_options(
        keep_home: pathlib.Path, monkeypatch, tmp_path: pathlib.Path) -> None:
    from typed_gguf.registry import store

    model = _model_file(tmp_path)
    registry = store.Registry(aliases={}, current="a")
    store.add_entry(registry, store.Entry(alias="a", path=str(model), sha256="abc123"),
                    alias="a")
    store.save_registry(registry, store.registry_path(keep_home))
    key, alias = cli.keep_key_for(_payload(n_ctx=4096, threads=3, backend="cpu",
                                           kv_type="q8_0", n_seq_max=7),
                                  home=keep_home, fit={"fit_enabled": True,
                                                       "fit_target_mb": 2048})
    assert alias == "a" and key.model_path == str(model) and key.model_sha == "abc123"
    assert (key.n_ctx, key.threads, key.backend, key.kv_type, key.n_seq_max) == \
        (4096, 3, "cpu", "q8_0", 7)
    assert key.fit_target_mb == 2048
    # a *bare* path (nothing registered under it) is keyed by the file's own identity: a replaced
    # file must not keep a host loaded from the old weights
    other = tmp_path / "other.gguf"
    other.write_bytes(b"GGUF" + b"\0" * 16)
    path_key, alias = cli.keep_key_for(_payload(model=str(other)), home=keep_home,
                                       fit={"fit_enabled": True})
    assert path_key.model_sha.startswith("stat:")
    assert str(other.stat().st_size) in path_key.model_sha
    assert alias == "other"


# ------------------------------------------------------------------ the two halves of a decision
def _settled(body: Mapping[str, Any]) -> dict:
    """A response body without the numbers that can only be wall-clock (timings, noise)."""
    return {key: value for key, value in body.items() if key not in ("timings", "warnings")}


def _fake_serving_path(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> FakeSession:
    """Patch the model out of the decision path (the pattern `tests/test_calibration.py` uses)."""
    plan = type("Plan", (), {"n_ctx": 4096, "n_seq_max": 4, "kv_type": "auto",
                             "model_sha256": "", "host_fingerprint": "",
                             "notes": (), "insufficient": False,
                             "to_dict": lambda self: {}})()
    monkeypatch.setattr(cli, "fit_plan_for", lambda *a, **k: plan)
    session = FakeSession(n_vocab=8192)
    session.row_fn = lambda ctx, session=session: biased_row(
        session.n_vocab, {session.tokenize("billing")[0]: 30.0})
    # a real handle records how it was placed; the warm host republishes exactly this
    session.placement = {"n_gpu_layers": 0, "note": "cpu", "fit": "off"}
    session.load_ms = 7.5

    @contextlib.contextmanager
    def fake_open_model(*args, **kwargs):
        yield session

    @contextlib.contextmanager
    def fake_session(handle, plan_, *, load_ms=None, **kwargs):
        if load_ms is not None:                 # the real `ModelSession` records this as its own
            handle.load_ms = float(load_ms)
        yield handle

    monkeypatch.setattr(cli.session_module, "open_model", fake_open_model)
    monkeypatch.setattr(cli.session_module, "ModelSession", fake_session)
    monkeypatch.setattr(cli.session_module, "runtime_backend", lambda home=None: "cpu")
    return session


def test_decide_payload_is_prepare_plus_decide_on_a_loaded_handle(
        keep_home: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """The split the host reuses: same body, one load, and a warm half that pays no model load."""
    _fake_serving_path(monkeypatch, tmp_path)
    model = _model_file(tmp_path)
    payload = _payload(model=str(model))

    cold = cli.decide_payload(payload, home=keep_home, fit_enabled=False)
    prepared = cli.prepare_decision(payload, home=keep_home, fit_enabled=False)
    with cli.session_module.open_model(prepared.model_path) as handle:
        warm = cli.decide_on_handle(prepared, payload, handle)
    assert _settled(warm) == _settled(cold)                 # timings are wall-clock, not content
    assert cold["answers"]["q"]["choice"] == "billing"

    # the host's own half reports the load it paid once and *not* the load this call did not do
    with cli.session_module.open_model(prepared.model_path) as handle:
        hosted = cli.decide_on_handle(prepared, payload, handle, session_load_ms=0.0)
    assert hosted["timings"]["model_load_ms"] == 0.0
    assert cold["timings"]["model_load_ms"] == warm["timings"]["model_load_ms"] != 0.0
    assert hosted["answers"] == cold["answers"]


def test_a_warm_session_reports_the_load_this_call_paid(monkeypatch) -> None:
    """RED pin (card t_7e24cea4): `model_load_ms` is what *this* call paid, not the handle's.

    The host's `Loaded.model_load_ms` publishes the one load the host paid; a request answered
    three hours later must not claim it loaded the model again.
    """
    from types import SimpleNamespace

    from typed_gguf.engine import session as session_module

    llama = SimpleNamespace(llama_get_memory=lambda ctx: "mem")
    handle = SimpleNamespace(load_ms=12345.0, path="m.gguf", n_vocab=8, load_log=(),
                             runtime=SimpleNamespace(llama=llama))
    plan = SimpleNamespace(kv_type="auto", n_ctx=64, n_seq_max=1, threads=1)
    monkeypatch.setattr(session_module, "_init_context", lambda *a, **k: ("ctx", None, "auto"))
    monkeypatch.setattr(session_module, "_runtime_name", lambda handle: "fake")
    assert session_module.ModelSession(handle, plan, backend="cpu").load_ms == 12345.0
    warm = session_module.ModelSession(handle, plan, backend="cpu", load_ms=0.0)
    assert warm.load_ms == 0.0


def test_prepare_decision_reports_the_same_facts_as_the_cold_path(
        keep_home: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    _fake_serving_path(monkeypatch, tmp_path)
    model = _model_file(tmp_path)
    payload = _payload(model=str(model))
    prepared = cli.prepare_decision(payload, home=keep_home, fit_enabled=True)
    assert prepared.alias == model.name.removesuffix(".gguf")
    assert prepared.model_path == str(model)
    assert prepared.plan is not None and prepared.n_ctx_cap == prepared.plan.n_ctx
    assert prepared.fit_enabled is True and prepared.home == keep_home
    assert prepared.route_plan is None
    # a second request must not have to re-resolve the model: the prepared half is reusable
    assert cli.prepare_decision(payload, home=keep_home, fit_enabled=True).model_path == \
        prepared.model_path


# ------------------------------------------------------------------ the host entry point
def test_the_internal_host_subcommand_refuses_a_spec_it_cannot_read(keep_home: pathlib.Path,
                                                                    capsys) -> None:
    assert cli.main(["keep", "_host", "--spec", str(keep_home / "nope.json")]) == 2
    assert "E_UNKNOWN_KEY" in capsys.readouterr().err


def test_the_internal_host_subcommand_is_not_advertised() -> None:
    """`_host` is the client's own spawn target, not a user-facing verb."""
    assert not any("_host" in flag for flag in cli.COMMAND_HELP["keep"])
    assert cli.KEEP_SUBCOMMANDS == ("status", "stop")


def test_a_host_spec_round_trips_into_a_loaded_handle(keep_home: pathlib.Path,
                                                      monkeypatch: pytest.MonkeyPatch,
                                                      tmp_path: pathlib.Path) -> None:
    """The loader the host runs: the spec's own payload prepares the plan, once."""
    session = _fake_serving_path(monkeypatch, tmp_path)
    model = _model_file(tmp_path)
    key = identity.KeepKey.of(None, model_path=str(model))
    from typed_gguf.keep import host as host_module

    spec = host_module.HostSpec(
        key=key, digest=key.digest, model="a", model_path=str(model), keep_alive=30.0,
        spec_path=str(state.spec_path(keep_home, key.digest)),
        socket_path=str(state.socket_path(keep_home, key.digest)),
        log_path=str(state.log_path(keep_home, key.digest)),
        payload=_payload(model=str(model)), fit={"fit_enabled": False}, home=str(keep_home))
    loaded = cli.keep_host_loaded(spec)
    assert loaded.model_path == str(model) and loaded.model == "a"
    assert loaded.placement == session.placement, "the handle's own placement, republished"
    assert loaded.model_load_ms == 7.5, "the load the host paid, once"
    body = loaded.decide(_payload())                       # every request carries its own state
    assert body["answers"]["q"]["choice"] == "billing"
    assert body["timings"]["model_load_ms"] == 0.0          # a warm decision pays no load
    # the evidence the host publishes is the engine's, not the flags it was handed
    assert "devices" in loaded.devices and "effective_backend" in loaded.devices
    assert session.batches, "the decision really ran on the fake engine"
