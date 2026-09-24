"""A-E5-1..5: the `serve` HTTP surface (SPEC 2.9) — wire, mapping, errors, same engine, cold/warm.

Everything here is offline. The wire gates drive the **in-process** app (`typed_gguf.api.http.App`)
with the SDK's own request bytes and validate every served key set against
`tests/fixtures/typesafe_sdk_0_7_1_fields.json` — the field lists captured from the installed
`typesafe-sdk` 0.7.1 by `tools/typesafe_fields_capture.py`, so "no invented fields" is a test.

The HTTP layer itself (request line, headers, `Content-Length`, keep-alive) is exercised through a
socketpair: a real `http.server` handler on a real socket, with no `AF_INET` anywhere — the CI's
net-off flag (`TYPED_GGUF_TEST_BLOCK_NET=1`) forbids exactly the network families. The one gate that
needs a loopback bind skips *by name* when that flag is on.

`A-E5-4` (same engine) is pinned through the **production** warm path (`cli.decide_payload_warm`)
with its keep host replaced by `tests/fake_keep_host.py`: the request a client sends over HTTP and
the request `run` builds for the same state + questions are the *same payload*, and the served
answer is `schema.render_response(native, format="typesafe")` of the very same engine body.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import pathlib
import socket
import subprocess
import sys
import threading
import time
from typing import Any

import pytest

from tests.fake_engine import FakeSession, biased_row
from typed_gguf import cli, schema
from typed_gguf.api import http as serve
from typed_gguf.engine import decide as decide_module
from typed_gguf.errors import PrefillFailedError, RuntimeError_, TypedGgufError, UserError
from typed_gguf.keep import client as keep_client
from typed_gguf.keep import identity
from typed_gguf.keep import state as keep_state
from typed_gguf.registry import store

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "typesafe_sdk_0_7_1_fields.json"
FIELDS = json.loads(FIXTURE.read_text(encoding="utf-8"))
SDK_CONSTANTS = {**FIELDS["constants"]["typesafe_sdk._core.constants"],
                 **FIELDS["constants"]["typesafe_sdk.constants"]}
WIRE = FIELDS["wire_models"]
ANSWER_FIELDS = FIELDS["answers"]
FAKE_HOST = pathlib.Path(__file__).resolve().parent / "fake_keep_host.py"
NET_BLOCK_ENV = "TYPED_GGUF_TEST_BLOCK_NET"
ALIAS = "spark-x2.5-4b-q8_0"
#: The body the SDK builds for `client.system_one(state, questions)`
#: (`endpoints.prepare_system_one` serializes exactly `{state, model, questions}`; `Noul`/`Score`
#: omit their unset optionals and a choice's `criteria` is the mapping the caller passed).
SDK_MIXED_BODY = {
    "state": "The dashboard is blank for every account since this morning.",
    "model": "jev-latest",
    "questions": {
        "urgency": {"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["Can wait", "This week", "Today"]},
        "area": {"type": "choice", "instructions": "Which area is this about?",
                 "criteria": {"billing": "Payments, invoicing, refunds",
                              "technical": "API, uptime, errors",
                              "sales": "Pricing, plans, contracts"}},
        "spam": {"type": "noul", "instructions": "Is this unsolicited advertising?",
                 "criteria": {"true": "Unsolicited advertising", "false": "A real report"}},
    },
}


def _net_blocked() -> bool:
    """Does this run forbid the socket family a loopback server needs? (`tests/conftest.py`)"""
    return os.environ.get(NET_BLOCK_ENV) in ("1", "true", "yes")


def _no_loading() -> dict[str, Any]:
    raise AssertionError("this route must never load a model")


# --------------------------------------------------------------------- the fixture is the oracle
def test_the_fixture_pins_the_sdk_release_the_spec_is_written_against() -> None:
    assert FIELDS["sdk"]["version"] == "0.7.1", (
        "SPEC 2.9 is written against typesafe-sdk 0.7.1: regenerate the fixture deliberately")
    assert FIELDS["_provenance"]["fixture_pin"] == FIELDS["sdk"]["version"]
    assert FIELDS["sdk"]["captured_by"] == "tools/typesafe_fields_capture.py"


def test_our_paths_and_the_compat_name_are_the_sdk_s_own() -> None:
    assert serve.SYSTEM_ONE_PATH == SDK_CONSTANTS["SYSTEM_ONE_PATH"] == "/v1/systemone"
    assert serve.MODELS_PATH == SDK_CONSTANTS["MODELS_PATH"] == "/v1/models"
    assert serve.COMPAT_MODEL == SDK_CONSTANTS["DEFAULT_MODEL"] == "jev-latest"
    assert tuple(WIRE["SystemOneRequest"]) == serve.REQUEST_KEYS, (
        "the served request keys are the SDK's SystemOneRequest, in its own order")
    assert serve.SERVICE == "typed-gguf"


def test_the_answer_key_sets_are_the_documented_ones() -> None:
    """SPEC 2.9: `type` + value key + `probabilities` (+ `confidence` for choice/score, + the
    `legend` for score). The SDK's own answer models are the same sets 1:1 for choice/score;
    `noul` differs only by the `probabilities` the SPEC documents and the SDK ignores."""
    assert serve.ANSWER_KEYS["choice"] == ("type", "choice", "confidence", "probabilities")
    assert serve.ANSWER_KEYS["score"] == ("type", "score", "confidence", "legend", "probabilities")
    assert serve.ANSWER_KEYS["noul"] == ("type", "noul", "probabilities")
    assert set(ANSWER_FIELDS["choice"]) == set(serve.ANSWER_KEYS["choice"])
    assert set(ANSWER_FIELDS["score"]) == set(serve.ANSWER_KEYS["score"])
    assert set(ANSWER_FIELDS["noul"]) | {"probabilities"} == set(serve.ANSWER_KEYS["noul"])
    assert tuple(WIRE["ModelMetadata"]) == serve.MODELS_KEYS
    assert tuple(WIRE["Usage"]) == serve.USAGE_KEYS
    assert tuple(WIRE["SystemOneResponse"]) == serve.SERVICE_KEYS
    assert serve.ANSWER_TYPES == ("noul", "choice", "score")
    assert [name.casefold().removesuffix("answer")
            for name in FIELDS["answer_discriminator"]["members"]] == list(serve.ANSWER_TYPES)


# ------------------------------------------------------------------------------ the test doubles
def mixed_engine_body(payload: dict[str, Any]) -> dict[str, Any]:
    """A native body from the *real* engine over the deterministic fake session (test_cli's trade).

    Same validation, same readout, same rendering as a model load — only the weights are fake, so
    the mapping gates see real probability vectors, real confidence and real legends. The `keep`
    block the warm CLI path always writes is added too (`served_by: host`), so the log and the
    `served_by` this app reports are the ones the real path produces.
    """
    request = schema.parse_request(payload)
    session = FakeSession(n_vocab=512)
    session.model_alias = payload.get("model") or session.model_alias
    preferred = session.tokenize("billing")[0]
    session.row_fn = lambda ctx, s=session: biased_row(s.n_vocab, {preferred: 6.0})
    result = decide_module.decide_request(request, session)
    body = schema.render_response(result.payload(), format=request.format)
    if isinstance(body.get("engine"), dict):
        body["engine"]["keep"] = {"served_by": "host", "pid": 4242, "requests": 1}
    return body


class Engine:
    """A `decide` callable that records every native payload it is handed."""

    def __init__(self, body: Any = mixed_engine_body, error: Exception | None = None,
                 delay: float = 0.0) -> None:
        self.calls: list[dict[str, Any]] = []
        self.active = 0
        self.max_active = 0
        self.order: list[str] = []
        self._body = body
        self._error = error
        self._delay = delay
        self._lock = threading.Lock()

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.calls.append(json.loads(json.dumps(payload)))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.order.append(payload.get("state", ""))
        try:
            if self._delay:
                time.sleep(self._delay)
            if self._error is not None:
                raise self._error
            return self._body(payload)
        finally:
            with self._lock:
                self.active -= 1


class Log:
    """The app's one-line-per-request sink (`App(log=…)`)."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, line: str) -> None:
        self.lines.append(line)


def body_of(response: serve.Response) -> dict[str, Any]:
    return json.loads(response.body.decode("utf-8"))


def post(app: serve.App, payload: Any, path: str = serve.SYSTEM_ONE_PATH, *,
         headers: dict[str, str] | None = None) -> serve.Response:
    raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    return app.handle("POST", path, headers or {}, raw)


def _registry_with_models(home: pathlib.Path, *aliases: str) -> None:
    """A data home whose registry lists `aliases`, each pointing at a real (tiny) file."""
    home.mkdir(parents=True, exist_ok=True)
    entries: dict[str, store.Entry] = {}
    for index, alias in enumerate(aliases):
        model = home / "models" / f"{alias}.gguf"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"GGUF" + bytes(index))
        os.utime(model, (1_700_000_000 + index, 1_700_000_000 + index))
        entries[alias] = store.Entry(alias=alias, path=str(model), arch="qwen35", quant="Q8_0",
                                     size=4_000_000_000 + index)
    store.save_registry(store.Registry(aliases=entries, current=aliases[0]),
                        store.registry_path(home))


@pytest.fixture
def home(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """A data home with one registry alias whose file exists.

    `TYPED_GGUF_HOME` points at it too, so every path that resolves the home for itself — `run`,
    `ask`, the keep ledger — lands in the same place `serve` was given: `home=None` means *this*
    home, which is what the same-host claim rests on.
    """
    data_home = tmp_path / "home"
    _registry_with_models(data_home, ALIAS)
    monkeypatch.setenv("TYPED_GGUF_HOME", str(data_home))
    return data_home


# ---------------------------------------------------------------------------- 1. /health
def test_health_is_exactly_the_five_keys_and_never_touches_the_engine(tmp_path) -> None:
    app = serve.App(decide=_no_loading, home=tmp_path / "nothing", log=Log())
    response = app.handle("GET", serve.HEALTH_PATH, {}, b"")
    assert response.status == 200
    payload = body_of(response)
    assert set(payload) == {"status", "service", "version", "model", "warm"}
    assert payload["status"] == "ok" and payload["service"] == serve.SERVICE
    assert payload["version"] == cli.__version__
    assert payload["model"] is None and payload["warm"] is False


def test_health_reads_the_resident_host_from_the_ledger(tmp_path) -> None:
    data_home = tmp_path / "home"
    sock = data_home / "keep" / "host.sock"
    sock.parent.mkdir(parents=True)
    sock.write_text("", encoding="utf-8")
    keep_state.write_record(keep_state.HostRecord(
        digest="d", pid=os.getpid(), socket=str(sock), key={}, model=ALIAS,
        model_path=f"/models/{ALIAS}.gguf", keep_alive=600.0, started_at=time.time(),
        loaded_at=time.time(), spec=str(data_home / "keep" / "spec.json"),
        log=str(data_home / "keep" / "host.log")), data_home)
    app = serve.App(decide=_no_loading, home=data_home, log=Log())
    payload = body_of(app.handle("GET", serve.HEALTH_PATH, {}, b""))
    assert payload["warm"] is True and payload["model"] == ALIAS


def test_health_reports_a_dead_host_as_cold(tmp_path) -> None:
    data_home = tmp_path / "home"
    data_home.mkdir()
    keep_state.write_record(keep_state.HostRecord(
        digest="d", pid=999_999_999, socket=str(data_home / "keep" / "gone.sock"), key={},
        model="m", model_path="/models/m.gguf", keep_alive=1.0, started_at=0.0, loaded_at=0.0,
        spec="s", log="l"), data_home)
    app = serve.App(decide=_no_loading, home=data_home, log=Log())
    payload = body_of(app.handle("GET", serve.HEALTH_PATH, {}, b""))
    assert payload["warm"] is False and payload["model"] is None


# ---------------------------------------------------------------------------- 2. /v1/models
def test_models_lists_the_registry_sorted_plus_the_compat_name(tmp_path) -> None:
    data_home = tmp_path / "home"
    _registry_with_models(data_home, ALIAS, "aaa-small")
    app = serve.App(decide=_no_loading, home=data_home, log=Log())
    payload = body_of(app.handle("GET", serve.MODELS_PATH, {}, b""))
    assert set(payload) == set(WIRE["ModelMetadataList"]) == {"models"}
    assert [model["name"] for model in payload["models"]] == [
        "aaa-small", ALIAS, serve.COMPAT_MODEL], "aliases sorted by name, then the compat name"
    for model in payload["models"]:
        assert set(model) == set(WIRE["ModelMetadata"]) == set(serve.MODELS_KEYS)
        assert model["release_date"] == "2023-11-14", "the model file's mtime, as a UTC date"
    described = {model["name"]: model["description"] for model in payload["models"]}
    assert described["aaa-small"].startswith("qwen35 Q8_0 GGUF ("), described
    assert described["aaa-small"].endswith("local; typed-gguf's own engine")
    assert described[serve.COMPAT_MODEL] == described[ALIAS], (
        "the compat name answers for the registry's `current` alias")


def test_models_is_empty_without_a_registry_and_the_compat_name_is_gone(tmp_path) -> None:
    app = serve.App(decide=_no_loading, home=tmp_path / "empty-home", log=Log())
    assert body_of(app.handle("GET", serve.MODELS_PATH, {}, b"")) == {"models": []}


def test_an_empty_registry_says_pull_something(tmp_path) -> None:
    home = tmp_path / "empty-home"
    home.mkdir()
    app = serve.App(decide=Engine(), home=home, log=Log())
    response = post(app, SDK_MIXED_BODY)
    assert response.status == 422
    detail = body_of(response)["detail"][0]
    assert detail["type"] == "value_error" and "models pull" in detail["msg"]


# --------------------------------------------------------- 3. /v1/systemone: mapping + wire
def test_the_served_call_answers_the_sdk_s_mixed_body(home) -> None:
    engine = Engine()
    app = serve.App(decide=engine, home=home, log=Log())
    response = post(app, SDK_MIXED_BODY, headers={"Authorization": "Bearer local-key"})
    assert response.status == 200
    payload = body_of(response)
    assert set(payload) == set(WIRE["SystemOneResponse"]) == set(serve.SERVICE_KEYS)
    assert payload["model"] == ALIAS, "the resolved alias, not the raw `jev-latest`"
    assert set(payload["answers"]) == set(SDK_MIXED_BODY["questions"])
    for qid, answer in payload["answers"].items():
        assert set(answer) == set(serve.ANSWER_KEYS[answer["type"]]), (qid, answer)
        assert answer["type"] in serve.ANSWER_TYPES
    assert set(payload["usage"]) == set(WIRE["Usage"]) == set(serve.USAGE_KEYS)
    assert response.served_by == "host", "the fake engine stands in for a warm host"
    # the engine saw the *native* payload: the resolved alias, the SDK's questions verbatim
    assert engine.calls == [{"state": SDK_MIXED_BODY["state"], "model": ALIAS,
                             "questions": SDK_MIXED_BODY["questions"]}]


def test_every_answer_type_keeps_its_documented_keys(home) -> None:
    app = serve.App(decide=Engine(), home=home, log=Log())
    answers = body_of(post(app, SDK_MIXED_BODY))["answers"]
    assert answers["spam"]["type"] == "noul" and "confidence" not in answers["spam"]
    assert set(answers["spam"]) == {"type", "noul", "probabilities"}
    assert answers["urgency"]["legend"] == {"0": "Can wait", "1": "This week", "2": "Today"}
    assert set(answers["urgency"]["legend"]) == set(answers["urgency"]["probabilities"])
    assert "legend" not in answers["area"], "the native legend is score-only on this wire"


def test_the_served_answer_is_schema_render_response_of_the_native_body(home) -> None:
    """A-E5-2: no serve-local projection, no renormalization — byte-equal to the adapter."""
    native = {"model": ALIAS,
              "engine": {"runtime": "llama.cpp b11026", "backend": "vulkan"},
              "answers": {"area": {"type": "choice", "choice": "billing",
                                   "probabilities": {"billing": 0.61, "technical": 0.33,
                                                     "sales": 0.06000000004},
                                   "confidence": 0.41500000001, "coverage": 0.93,
                                   "reliability": "ok", "decode_steps": 3},
                          "urgency": {"type": "score", "score": 1.3, "legend": {"0": "Can wait"},
                                      "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8},
                                      "confidence": 0.55},
                          "spam": {"type": "noul", "noul": 0.93,
                                   "probabilities": {"yes": 0.93, "no": 0.07}}},
              "usage": {"input_tokens": 812, "output_tokens": 12, "questions": 3, "forks": 9,
                        "prefill_tokens": 0, "decode_steps": 12, "waves": 1},
              "timings": {"model_load_ms": 0.0, "prefill_ms": 430.0, "questions_ms": 41.0,
                          "total_ms": 471.0},
              "warnings": []}
    app = serve.App(decide=Engine(body=lambda payload: json.loads(json.dumps(native))),
                    home=home, log=Log())
    served = body_of(post(app, SDK_MIXED_BODY))
    assert served == schema.render_response(native, format="typesafe")
    assert served["usage"] == {"input_tokens": 812, "output_tokens": 12}
    # the 6-significant-decimal rounding is the engine's (SPEC 2.5): the served numbers are its own
    assert served["answers"]["area"]["probabilities"]["sales"] == 0.06
    assert served["answers"]["area"]["confidence"] == 0.415


def test_a_plain_path_or_repo_reference_is_refused_on_the_typesafe_route(home) -> None:
    app = serve.App(decide=_no_loading, home=home, log=Log())
    for ref in (f"{home}/models/{ALIAS}.gguf", "someone/model:Q4_K_M", "nope",
                f"{ALIAS}.gguf"):
        response = post(app, {**SDK_MIXED_BODY, "model": ref})
        assert response.status == 422, ref
        detail = body_of(response)["detail"][0]
        assert detail["type"] == "value_error"
        assert detail["msg"].startswith("E_MODEL_NOT_FOUND: "), detail["msg"]
        assert detail["loc"] == ["body", "model"]


# ------------------------------------------------------------------- 4. errors are FastAPI-shaped
def test_a_malformed_body_is_a_json_invalid_detail(tmp_path) -> None:
    app = serve.App(decide=_no_loading, home=tmp_path / "nothing", log=Log())
    for raw in (b"{not json", b"", b"[1, 2, 3]", b'"a string"'):
        response = post(app, raw)
        assert response.status == 422, raw
        detail = body_of(response)["detail"]
        assert isinstance(detail, list) and len(detail) == 1
        assert detail[0]["type"] == "json_invalid" and detail[0]["loc"] == ["body"]
        assert isinstance(detail[0]["msg"], str) and detail[0]["msg"]


def test_a_missing_required_key_names_the_field(tmp_path) -> None:
    app = serve.App(decide=_no_loading, home=tmp_path / "nothing", log=Log())
    for key in serve.REQUEST_KEYS:
        body = {name: value for name, value in SDK_MIXED_BODY.items() if name != key}
        response = post(app, body)
        assert response.status == 422, key
        detail = body_of(response)["detail"][0]
        assert detail["type"] == "missing" and detail["loc"] == ["body", key]
        assert detail["msg"] == "Field required"


def test_an_unknown_top_level_key_is_extra_forbidden(tmp_path) -> None:
    app = serve.App(decide=_no_loading, home=tmp_path / "nothing", log=Log())
    response = post(app, {**SDK_MIXED_BODY, "extra_body": {"temperature": 1.0}})
    assert response.status == 422
    detail = body_of(response)["detail"][0]
    assert detail["type"] == "extra_forbidden" and detail["loc"] == ["body", "extra_body"]
    assert set(detail) <= set(WIRE["ValidationError"])


def test_every_validation_detail_is_a_sdk_validation_error(home) -> None:
    """The shape, not just the status: `ValidationError`'s fields inside `HTTPValidationError`."""
    app = serve.App(decide=_no_loading, home=home, log=Log())
    cases = [b"{not json", json.dumps({"state": "x", "model": ALIAS}).encode(),
             json.dumps({**SDK_MIXED_BODY, "nope": 1}).encode(),
             json.dumps({**SDK_MIXED_BODY, "model": "nope"}).encode()]
    for raw in cases:
        payload = body_of(post(app, raw))
        assert set(payload) == set(WIRE["HTTPValidationError"]) == {"detail"}
        assert isinstance(payload["detail"], list) and payload["detail"]
        for detail in payload["detail"]:
            assert set(detail) <= set(WIRE["ValidationError"])
            assert isinstance(detail["loc"], list) and detail["loc"][0] == "body"
            assert isinstance(detail["msg"], str) and isinstance(detail["type"], str)


def test_an_engine_validation_code_is_a_422_value_error_with_its_code(home) -> None:
    engine = Engine(error=UserError(
        "question 'urgency': score criteria must be an ordered array of 2..10 levels",
        code="E_SCORE_LEVELS"))
    app = serve.App(decide=engine, home=home, log=Log())
    body = json.loads(json.dumps(SDK_MIXED_BODY))
    body["questions"]["urgency"]["criteria"] = ["only one level"]
    response = post(app, body)
    assert response.status == 422
    detail = body_of(response)["detail"][0]
    assert detail["type"] == "value_error"
    assert detail["msg"].startswith("E_SCORE_LEVELS: "), detail["msg"]
    assert detail["loc"] == ["body", "questions", "urgency", "criteria"]
    assert detail["input"] == ["only one level"], "the value that failed, when we can point at it"


def test_too_many_choice_options_are_refused_by_our_own_engine(home) -> None:
    """SPEC 2.9: the vendor's looser limit never buys a wrong answer — we answer a typed 422."""
    app = serve.App(decide=Engine(), home=home, log=Log())
    body = json.loads(json.dumps(SDK_MIXED_BODY))
    body["questions"] = {"big": {"type": "choice", "instructions": "x",
                                 "criteria": {f"o{index}": None for index in range(256)}}}
    response = post(app, body)
    assert response.status == 422
    detail = body_of(response)["detail"][0]
    assert detail["msg"].startswith("E_CHOICE_TOO_MANY: "), detail["msg"]
    assert detail["loc"] == ["body", "questions", "big", "criteria"]


def test_a_runtime_failure_is_a_500_with_the_code_in_the_detail(home) -> None:
    app = serve.App(decide=Engine(error=PrefillFailedError(
        "E_PREFILL_FAILED: the prefix did not fit")), home=home, log=Log())
    response = post(app, SDK_MIXED_BODY)
    assert response.status == 500
    payload = body_of(response)
    assert set(payload) == {"detail"} and isinstance(payload["detail"], str)
    assert payload["detail"].startswith("E_PREFILL_FAILED: "), payload["detail"]


def test_an_unknown_route_is_a_plain_404(tmp_path) -> None:
    app = serve.App(decide=_no_loading, home=tmp_path / "nothing", log=Log())
    for method, path in (("GET", "/nope"), ("POST", "/v1/nothing"), ("GET", serve.DECIDE_PATH),
                         ("GET", "/")):
        response = app.handle(method, path, {}, b"")
        assert response.status == 404, (method, path)
        assert body_of(response) == {"detail": "Not Found"}


def test_a_query_string_does_not_change_the_route(tmp_path) -> None:
    app = serve.App(decide=_no_loading, home=tmp_path / "nothing", log=Log())
    assert body_of(app.handle("GET", serve.HEALTH_PATH + "?probe=1", {}, b""))["status"] == "ok"


# --------------------------------------------------------------------- 5. auth + request logging
def test_a_bearer_token_or_no_token_at_all_is_accepted_and_never_logged(home) -> None:
    log = Log()
    app = serve.App(decide=Engine(), home=home, log=log)
    sentinel = "sk-sentinel-must-never-be-logged"
    for headers in ({"Authorization": f"Bearer {sentinel}"}, {}, {"Authorization": sentinel}):
        assert post(app, SDK_MIXED_BODY, headers=headers).status == 200
    assert len(log.lines) == 3, "one line per request"
    assert sentinel not in "\n".join(log.lines)
    assert "Bearer" not in "\n".join(log.lines)


def test_every_request_leaves_one_line_with_route_status_served_by_and_ms(home) -> None:
    log = Log()
    app = serve.App(decide=Engine(), home=home, log=log)
    post(app, SDK_MIXED_BODY)
    app.handle("GET", serve.HEALTH_PATH, {}, b"")
    assert len(log.lines) == 2, log.lines
    assert log.lines[0].startswith("POST /v1/systemone 200 "), log.lines[0]
    assert "served_by=host" in log.lines[0] and "ms" in log.lines[0]
    assert log.lines[1].startswith("GET /health 200 ") and "ms" in log.lines[1]


# ------------------------------------------------------- 6. /v1/decide is the native wire
def test_decide_is_the_native_wire_and_is_never_projected(home) -> None:
    engine = Engine()
    app = serve.App(decide=engine, home=home, log=Log(), default_format="native")
    native = {"state": "x", "model": ALIAS,
              "questions": {"q": {"type": "noul", "instructions": "is it spam"}}}
    response = post(app, native, path=serve.DECIDE_PATH)
    assert response.status == 200
    payload = body_of(response)
    assert {"model", "engine", "answers", "usage", "timings", "warnings"} <= set(payload)
    assert engine.calls[-1]["format"] == "native", "`--format` is the default for the request"


def test_decide_takes_the_bodys_own_format_over_the_servers(home) -> None:
    engine = Engine()
    native = {"state": "x", "model": ALIAS, "format": "native",
              "questions": {"q": {"type": "noul", "instructions": "x"}}}
    app = serve.App(decide=engine, home=home, log=Log(), default_format="typesafe")
    assert post(app, native, path=serve.DECIDE_PATH).status == 200
    assert engine.calls[-1]["format"] == "native", "the body's own format wins"
    del native["format"]
    assert post(app, native, path=serve.DECIDE_PATH).status == 200
    assert engine.calls[-1]["format"] == "typesafe", "no format in the body: the flag decides"


def test_decide_maps_exit_codes_to_400_503_and_500(home) -> None:
    native = {"state": "x", "model": ALIAS,
              "questions": {"q": {"type": "noul", "instructions": "x"}}}
    for error, status in ((UserError("bad question", code="E_Q_TYPE_UNKNOWN"), 400),
                          (RuntimeError_("the backend died", code="E_PREFILL_FAILED"), 503),
                          (TypedGgufError("boom", code="E_INTERNAL"), 500)):
        app = serve.App(decide=Engine(error=error), home=home, log=Log())
        response = post(app, native, path=serve.DECIDE_PATH)
        assert response.status == status, (error, error.exit_code)
        payload = body_of(response)
        assert set(payload) == {"error"} and set(payload["error"]) == {"code", "message"}
        assert payload["error"]["code"] == error.code
    app = serve.App(decide=_no_loading, home=home, log=Log())
    response = post(app, b"{not json", path=serve.DECIDE_PATH)
    assert response.status == 400
    assert body_of(response)["error"]["code"] == "E_UNKNOWN_KEY"


# --------------------------------------------------- 7. the real HTTP layer (socketpair, no net)
class _DuckServer:
    """What `BaseHTTPRequestHandler` needs of its server here: the app."""

    def __init__(self, app: serve.App) -> None:
        self.app = app


def _read_response(conn: socket.socket) -> tuple[bytes, dict[bytes, bytes], bytes]:
    """One HTTP/1.1 response off `conn` (status line + headers + exactly Content-Length bytes)."""
    head = b""
    while b"\r\n\r\n" not in head:
        chunk = conn.recv(1)
        if not chunk:
            raise AssertionError(f"connection closed mid-headers: {head!r}")
        head += chunk
    head = head[:-4]
    lines = head.split(b"\r\n")
    headers = dict(line.split(b": ", 1) for line in lines[1:] if b": " in line)
    body = b""
    remaining = int(headers.get(b"Content-Length", b"0"))
    while len(body) < remaining:
        body += conn.recv(remaining - len(body))
    return lines[0], headers, body


#: The reason phrase in a status line is the *interpreter's* text, not ours: `Handler.send_response`
#: takes it from `http.HTTPStatus`, and CPython rewrites those strings between versions. 3.13
#: refreshed them per RFC 9110, so the body cap reads `Content Too Large` there and `Request Entity
#: Too Large` on 3.11/3.12 (card `t_b5872762`: CI ran 3.11/3.12, the certifying host 3.13, so only
#: the host went red). RFC 9110 §15 also tells clients to ignore the phrase, so what this file pins
#: off the wire is the *code* — via `_code_of`, never a spelled-out phrase — plus the body shape,
#: which is the part the spec actually fixes.
def _code_of(status: bytes) -> int:
    """`b"HTTP/1.1 413 Content Too Large"` -> `413`, whatever phrase the interpreter chose."""
    version, _space, rest = status.partition(b" ")
    assert version == b"HTTP/1.1", status
    code, _space, _phrase = rest.partition(b" ")
    assert code.isdigit() and len(code) == 3, status
    return int(code)


@pytest.fixture
def handler_in_a_thread():
    """Run the real handler on one end of a socketpair and hand back the client end."""
    started: list[tuple[socket.socket, socket.socket, threading.Thread]] = []

    def start(app: serve.App) -> socket.socket:
        ours, theirs = socket.socketpair()
        thread = threading.Thread(target=serve.Handler, args=(ours, ("127.0.0.1", 0),
                                                              _DuckServer(app)), daemon=True)
        thread.start()
        started.append((ours, theirs, thread))
        return theirs

    yield start
    for ours, theirs, thread in started:
        theirs.close()
        thread.join(timeout=5.0)
        ours.close()


def test_the_http_layer_speaks_http_1_1_with_a_content_length(home, handler_in_a_thread) -> None:
    raw = json.dumps(SDK_MIXED_BODY).encode("utf-8")
    conn = handler_in_a_thread(serve.App(decide=Engine(), home=home, log=Log()))
    conn.sendall(b"POST /v1/systemone HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                 b"Authorization: Bearer local\r\nContent-Type: application/json\r\n"
                 b"Accept: application/json\r\nX-TypeSafe-SDK: typesafe-sdk/0.7.1\r\n"
                 b"Connection: close\r\nContent-Length: " + str(len(raw)).encode() + b"\r\n\r\n"
                 + raw)
    status, headers, body = _read_response(conn)
    assert _code_of(status) == 200, status
    assert headers[b"Content-Type"] == b"application/json"
    assert b"x-typesafe-request-id" in {name.lower() for name in headers}, (
        "the SDK reads this header for its own logs")
    assert int(headers[b"Content-Length"]) == len(body)
    assert set(json.loads(body)) == set(serve.SERVICE_KEYS)


def test_the_http_layer_answers_a_get_health(tmp_path, handler_in_a_thread) -> None:
    conn = handler_in_a_thread(serve.App(decide=_no_loading, home=tmp_path / "nothing", log=Log()))
    conn.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
    status, _headers, body = _read_response(conn)
    assert _code_of(status) == 200, status
    assert json.loads(body)["status"] == "ok"


def test_the_http_layer_keeps_the_connection_alive_for_a_second_request(
        tmp_path, handler_in_a_thread) -> None:
    conn = handler_in_a_thread(serve.App(decide=_no_loading, home=tmp_path / "nothing", log=Log()))
    conn.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
    first = _read_response(conn)
    assert _code_of(first[0]) == 200, first[0]
    assert json.loads(first[2])["status"] == "ok"
    conn.sendall(b"GET /models-typo HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
    second = _read_response(conn)
    assert _code_of(second[0]) == 404, second[0]
    assert json.loads(second[2]) == {"detail": "Not Found"}


# ------------------------ the request-body cap, the idle timeout, the RST (card t_ba767a2b, M3)
# `413`'s reason phrase is the *interpreter's* text (see `_code_of`): the cap is asserted as a code,
# never as `Request Entity Too Large` / `Content Too Large` (card t_b5872762).
def test_a_body_over_the_cap_is_refused_before_a_byte_of_it_is_read(
        home, handler_in_a_thread) -> None:
    """`http.server` reads whatever `Content-Length` claims, so the cap is answered from the header
    alone: this client sends *no* body at all and still gets its `413` (a server that tried to read
    first would sit on the socket until the handler timeout, and the read below would fail). The
    status is read as a code — the phrase that follows it is the interpreter's."""
    log = Log()
    conn = handler_in_a_thread(serve.App(decide=_no_loading, home=home, log=log))
    conn.settimeout(5.0)
    over = serve.MAX_BODY_BYTES + 1
    conn.sendall(b"POST /v1/systemone HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                 b"Content-Type: application/json\r\nConnection: close\r\n"
                 b"Content-Length: " + str(over).encode() + b"\r\n\r\n")
    status, headers, body = _read_response(conn)
    assert _code_of(status) == 413, status
    assert headers[b"Connection"] == b"close", "the rest of that body must not be parsed as one"
    detail = json.loads(body)["detail"][0]
    assert detail["type"] == "too_large" and detail["loc"] == ["body"]
    assert str(over) in detail["msg"] and str(serve.MAX_BODY_BYTES) in detail["msg"]
    assert len(log.lines) == 1 and " 413 " in log.lines[0], "a refusal is still a request"
    assert headers[serve.REQUEST_ID_HEADER.encode()], "the refusal carries the request id like any"


def test_the_native_route_speaks_its_own_shape_for_the_cap(home, handler_in_a_thread) -> None:
    """§2.9's two shapes survive the cap: `/v1/decide` keeps `{"error": {code, message}}`."""
    conn = handler_in_a_thread(serve.App(decide=_no_loading, home=home, log=Log()))
    conn.settimeout(5.0)
    conn.sendall(b"POST /v1/decide HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n"
                 b"Content-Length: " + str(serve.MAX_BODY_BYTES * 4).encode() + b"\r\n\r\n")
    status, _headers, body = _read_response(conn)
    assert _code_of(status) == 413, status
    error = json.loads(body)["error"]
    assert error["code"] == serve.BODY_TOO_LARGE
    assert str(serve.MAX_BODY_BYTES * 4) in error["message"], "the refusal names the size announced"


def test_a_body_at_the_cap_is_still_read_and_routed(home, handler_in_a_thread) -> None:
    """The cap is a ceiling, not a policy: a body of exactly `MAX_BODY_BYTES` reaches the app."""
    seen: list[int] = []

    def decide(payload: dict[str, Any]) -> dict[str, Any]:
        seen.append(len(payload["state"]))
        return {"model": ALIAS, "answers": {}, "usage": {"input_tokens": 1, "output_tokens": 1}}

    app = serve.App(decide=decide, home=home, log=Log())
    conn = handler_in_a_thread(app)
    conn.settimeout(10.0)
    overhead = len(json.dumps({"state": "", "model": ALIAS, "questions": {}}).encode("utf-8"))
    state = "x" * (serve.MAX_BODY_BYTES - overhead)
    raw = json.dumps({"state": state, "model": ALIAS, "questions": {}}).encode("utf-8")
    assert len(raw) == serve.MAX_BODY_BYTES, "the cap itself is legal; the byte above it is not"
    conn.sendall(b"POST /v1/systemone HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                 b"Content-Type: application/json\r\nConnection: close\r\n"
                 b"Content-Length: " + str(len(raw)).encode() + b"\r\n\r\n" + raw)
    status, _headers, _body = _read_response(conn)
    assert _code_of(status) == 200, "a legal body is not a refusal"
    assert seen == [len(state)], "the body arrived whole"


def test_the_handler_drops_a_client_that_sends_nothing(home,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """One thread per connection: a client that opens one and says nothing must not hold it
    forever (`http.server`'s default is no timeout at all). The production window is asserted as a
    bound; the behaviour is measured with the same shape at 0.3 s.

    The handler is driven on a bare socketpair here rather than through the fixture: the fixture
    keeps *both* ends open, so an EOF is not observable from the client even after the handler gave
    up — the thread finishing is the fact under test.
    """
    assert serve.Handler.timeout == serve.HANDLER_TIMEOUT
    assert 0.0 < serve.HANDLER_TIMEOUT <= 120.0
    monkeypatch.setattr(serve.Handler, "timeout", 0.3)
    ours, theirs = socket.socketpair()
    thread = threading.Thread(
        target=serve.Handler,
        args=(ours, ("127.0.0.1", 0), _DuckServer(serve.App(decide=_no_loading, home=home,
                                                           log=Log()))),
        daemon=True)
    try:
        thread.start()                                  # nothing is ever sent on `theirs`
        thread.join(timeout=5.0)
        assert not thread.is_alive(), "an idle connection kept the handler thread"
    finally:
        theirs.close()
        ours.close()
        thread.join(timeout=5.0)


def test_a_client_that_left_mid_request_is_not_a_stdlib_traceback(home, capsys) -> None:
    """An `RST` on the wire is the client's business; a bug of ours must still be visible.

    `BaseServer.handle_error` prints a full traceback for every `BrokenPipeError` a vanished client
    causes — and `--host 0.0.0.0` is explicitly supported (S-8).
    """
    server = serve.Server.__new__(serve.Server)          # no bind: `handle_error` needs no state
    try:
        raise ConnectionResetError("the client left")
    except ConnectionResetError:
        server.handle_error(None, ("127.0.0.1", 43210))
    assert capsys.readouterr().err == ""
    try:
        raise RuntimeError("our own bug")
    except RuntimeError:
        server.handle_error(None, ("127.0.0.1", 43210))
    printed = capsys.readouterr().err
    assert "RuntimeError" in printed
    assert "43210" in printed, "the report names the client it came from"


@pytest.mark.skipif(_net_blocked(), reason=(
    "TYPED_GGUF_TEST_BLOCK_NET forbids AF_INET, the family a loopback server binds"))
def test_the_server_binds_loopback_and_answers_over_tcp(home) -> None:
    app = serve.App(decide=Engine(), home=home, log=Log())
    server = serve.make_server(app, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = str(server.server_address[0]), int(server.server_address[1])
        assert host == "127.0.0.1", "the default bind is loopback"
        with socket.create_connection((host, port), timeout=5.0) as conn:
            conn.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            status, _headers, body = _read_response(conn)
        assert _code_of(status) == 200, status
        assert json.loads(body)["service"] == serve.SERVICE
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


# ------------------------------------------------ 8. same engine: the served call IS the warm CLI
def test_the_decision_callable_is_the_warm_cli_path(home, monkeypatch) -> None:
    """`serve` has no engine of its own (SPEC 2.12): every decision is `decide_payload_warm`."""
    seen: list[dict[str, Any]] = []

    def fake_warm(payload: dict, *, home=None, keep_alive=None, **kwargs):
        seen.append({"payload": payload, "home": home, "keep_alive": keep_alive,
                     "fit": kwargs})
        return mixed_engine_body(payload)

    monkeypatch.setattr(cli, "decide_payload_warm", fake_warm)
    app = cli.serve_app(home=home, keep_alive=42.0, default_format="native")
    response = post(app, SDK_MIXED_BODY)
    assert response.status == 200
    assert seen == [{"payload": {"state": SDK_MIXED_BODY["state"], "model": ALIAS,
                                 "questions": SDK_MIXED_BODY["questions"]},
                     "home": home, "keep_alive": 42.0, "fit": {}}], seen
    assert body_of(response) == schema.render_response(
        mixed_engine_body(seen[0]["payload"]), format="typesafe")


def test_the_request_a_client_sends_is_the_payload_run_builds_for_the_same_questions(
        home, monkeypatch, capsys) -> None:
    """A-E5-4 offline half: one state + questions -> one payload, whether they arrive by HTTP or
    through `run`'s request file — and the same home + keep-alive window, so the warm host that
    answers is the same host (same ledger, same placement, same calibration)."""
    calls: list[dict[str, Any]] = []

    def fake_warm(payload: dict, *, home=None, keep_alive=None, **kwargs):
        calls.append({"payload": payload, "home": home, "keep_alive": keep_alive})
        return mixed_engine_body(payload)

    monkeypatch.setattr(cli, "decide_payload_warm", fake_warm)
    app = cli.serve_app(home=home, keep_alive=600.0, default_format="native")
    served = body_of(post(app, SDK_MIXED_BODY))
    served_call = calls[-1]

    questions = home.parent / "q.json"
    questions.write_text(json.dumps({"state": SDK_MIXED_BODY["state"], "model": ALIAS,
                                     "questions": SDK_MIXED_BODY["questions"]}),
                         encoding="utf-8")
    assert cli.main(["run", "--questions", str(questions), "--format", "typesafe"]) == 0
    assert calls[-1]["payload"] == {**served_call["payload"], "format": "typesafe"}, (
        "the served payload and the CLI's are the same request, bar the format the CLI was told")
    # `run` hands `decide_payload_warm` the *unresolved* defaults (`home=None` means "the data
    # home"); `serve` hands it what the CLI resolved once at start-up. Same home, same window —
    # that is the same-host claim, and it is the payload + the ledger that make it true.
    assert calls[-1]["home"] is None and calls[-1]["keep_alive"] is None
    assert served_call["home"] == home == store.data_home()
    assert served_call["keep_alive"] == identity.resolve_keep_alive(None)
    report = json.loads(capsys.readouterr().out)
    assert report["answers"] == served["answers"], "same engine, same numbers"
    assert report["model"] == served["model"] == ALIAS


# --------------------------------------------- 9. cold/warm + serialization through the real host
@pytest.fixture
def fake_host_client(keep_home: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """`cli.decide_payload_warm` with its keep host replaced by `tests/fake_keep_host.py`.

    The whole production path stays in place — `keep_key_for`, the ledger, the unix socket, the
    detached child, the reuse/swap decision, the keep-alive window — only the engine is fake, the
    same trade `tests/test_keep_client.py` makes.

    Every client the product builds is kept alive here and stopped on the way out: the CLI drops
    its `Client` as soon as the call returns, and a dropped `Popen` for a still-running (detached,
    correctly so) host is a `ResourceWarning`, not a finding.
    """
    monkeypatch.delenv("TYPED_GGUF_KEEP_FAKE", raising=False)
    built: list[keep_client.Client] = []

    class FakeHostClient(keep_client.Client):
        def __init__(self, **kwargs: object) -> None:
            kwargs.setdefault("host_command", (sys.executable, str(FAKE_HOST), "--spec"))
            super().__init__(**kwargs)                     # type: ignore[arg-type]
            built.append(self)

    monkeypatch.setattr(cli.keep_client, "Client", FakeHostClient)
    _registry_with_models(keep_home, ALIAS)
    yield keep_home
    for client in built:
        client.stop(grace=1.0)                             # reaps the child it spawned
        for _pid, proc in list(client.children.items()):
            proc.kill()
            proc.wait(timeout=2.0)
    keep_client.Client(home=keep_home).stop(grace=1.0)


@pytest.mark.needs_fork
def test_the_first_http_request_pays_the_load_and_the_second_reuses_the_host(
        fake_host_client) -> None:
    data_home = fake_host_client
    app = cli.serve_app(home=data_home, keep_alive=30.0, default_format="native")
    first = post(app, SDK_MIXED_BODY)
    pid = keep_state.read_record(data_home).pid
    second = post(app, SDK_MIXED_BODY)
    assert (first.status, second.status) == (200, 200)
    assert (first.served_by, second.served_by) == ("host", "host")
    assert keep_state.read_record(data_home).pid == pid, "the second request reused the host"
    assert body_of(first)["answers"] == body_of(second)["answers"], (
        "the warm answer is the cold answer")


@pytest.mark.needs_fork
def test_keep_stop_still_works_after_serving_and_leaves_nothing_behind(fake_host_client) -> None:
    data_home = fake_host_client
    app = cli.serve_app(home=data_home, keep_alive=30.0, default_format="native")
    assert post(app, SDK_MIXED_BODY).status == 200
    record = keep_state.read_record(data_home)
    assert record is not None and keep_state.pid_alive(record.pid)
    report = keep_client.Client(home=data_home).stop()
    assert report["stopped"] is True
    assert keep_state.read_record(data_home) is None
    assert keep_state.wait_pid_gone(record.pid, timeout=5.0), "no leaked host process"


def test_two_concurrent_decisions_serialize_in_arrival_order(home) -> None:
    engine = Engine(delay=0.2)
    app = serve.App(decide=engine, home=home, log=Log())
    bodies = [json.loads(json.dumps(SDK_MIXED_BODY)) for _ in range(2)]
    bodies[0]["state"] = "first"
    bodies[1]["state"] = "second"
    answers: dict[int, serve.Response] = {}

    def call(index: int) -> None:
        answers[index] = post(app, bodies[index])

    first = threading.Thread(target=call, args=(0,))
    first.start()
    deadline = time.monotonic() + 5.0
    while not engine.order and time.monotonic() < deadline:    # first decision is in flight
        time.sleep(0.005)
    second = threading.Thread(target=call, args=(1,))
    second.start()
    first.join(timeout=10.0)
    second.join(timeout=10.0)
    assert engine.max_active == 1, "one decision at a time (SPEC 2.12)"
    assert engine.order == ["first", "second"], engine.order
    assert (answers[0].status, answers[1].status) == (200, 200)
    assert set(body_of(answers[0])["answers"]) == set(SDK_MIXED_BODY["questions"])


def test_health_does_not_wait_behind_a_decision(home) -> None:
    """SPEC 2.9: `/health` and `/v1/models` never wait behind a decision."""
    engine = Engine(delay=0.4)
    app = serve.App(decide=engine, home=home, log=Log())
    decision = threading.Thread(target=lambda: post(app, SDK_MIXED_BODY))
    decision.start()
    deadline = time.monotonic() + 5.0
    while not engine.calls and time.monotonic() < deadline:
        time.sleep(0.005)
    started = time.monotonic()
    assert body_of(app.handle("GET", serve.HEALTH_PATH, {}, b""))["status"] == "ok"
    assert time.monotonic() - started < 0.2, "health waited behind the decision"
    decision.join(timeout=10.0)


# ------------------------------------------------------------------ 10. the CLI flag surface
def test_serve_flags_parse_into_the_app(home, monkeypatch, capsys) -> None:
    started: list[tuple[serve.App, str, int]] = []
    monkeypatch.setattr(cli.serve, "run", lambda app, host, port: started.append(
        (app, host, port)) or 0)
    monkeypatch.setenv("TYPED_GGUF_HOME", str(home))
    assert cli.main(["serve", "--host", "127.0.0.1", "--port", "8088", "--format", "typesafe",
                     "--keep-alive", "5m"]) == 0
    assert len(started) == 1
    app, host, port = started[0]
    assert (host, port) == ("127.0.0.1", 8088)
    assert app.default_format == "typesafe" and app.keep_alive == 300.0
    assert app.version == cli.__version__ and app.home == home
    out = capsys.readouterr()
    assert "127.0.0.1:8088" in out.out + out.err, out


def test_serve_defaults_to_loopback_8088_and_the_native_format(home, monkeypatch) -> None:
    started: list[tuple[serve.App, str, int]] = []
    monkeypatch.setattr(cli.serve, "run", lambda app, host, port: started.append(
        (app, host, port)) or 0)
    monkeypatch.setenv("TYPED_GGUF_HOME", str(home))
    monkeypatch.delenv(identity.KEEP_ALIVE_ENV, raising=False)
    assert cli.main(["serve"]) == 0
    app, host, port = started[0]
    assert (host, port) == (serve.DEFAULT_HOST, serve.DEFAULT_PORT) == ("127.0.0.1", 8088)
    assert app.default_format == "native"
    assert app.keep_alive == identity.DEFAULT_KEEP_ALIVE == 600.0


def test_serve_reads_the_keep_alive_env_knob(home, monkeypatch) -> None:
    """Precedence flag > env > default (SPEC 2.12), on the server's own window."""
    started: list[serve.App] = []
    monkeypatch.setattr(cli.serve, "run", lambda app, host, port: started.append(app) or 0)
    monkeypatch.setenv("TYPED_GGUF_HOME", str(home))
    monkeypatch.setenv(identity.KEEP_ALIVE_ENV, "2m")
    assert cli.main(["serve"]) == 0
    assert started[0].keep_alive == 120.0
    assert cli.main(["serve", "--keep-alive", "0"]) == 0
    assert started[-1].keep_alive == 0.0, "`0` means every request pays the load"


def test_serve_refuses_a_bad_format_port_and_keep_alive(home, monkeypatch, capsys) -> None:
    monkeypatch.setenv("TYPED_GGUF_HOME", str(home))
    for flags in (["--format", "nope"], ["--port", "not-a-port"], ["--port", "70000"],
                  ["--port", "-1"], ["--keep-alive", "soon"]):
        assert cli.main(["serve", *flags]) == 2, flags
        assert "E_UNKNOWN_KEY" in capsys.readouterr().err, flags


def test_serve_warns_when_it_binds_all_interfaces(home, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli.serve, "run", lambda app, host, port: 0)
    monkeypatch.setenv("TYPED_GGUF_HOME", str(home))
    assert cli.main(["serve", "--host", "0.0.0.0"]) == 0
    err = capsys.readouterr().err
    assert "0.0.0.0" in err and "warning" in err.lower(), err


def test_serve_help_and_the_command_list_agree_that_serve_ships(capsys) -> None:
    assert cli.main(["serve", "--help"]) == 0              # never starts a server
    help_text = capsys.readouterr().out
    assert "serve a decision API for TypeSafe clients" in help_text, help_text
    assert cli.COMMAND_DESCRIPTIONS["serve"] != cli.PLANNED_NOTE
    assert cli.NOT_IMPLEMENTED == ("mcp",), "mcp keeps its wording; serve ships"
    assert "--keep-alive" in " ".join(cli.COMMAND_HELP["serve"])


# ------------------------------------------------- 10. the host gate (acceptance 6, rehearsed)
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
GATE_SCRIPT = ROOT_DIR / "tools" / "host_gate_serve.sh"
GATE_DRIVER = ROOT_DIR / "tools" / "host_gate_serve_client.py"
REHEARSAL = ROOT_DIR / "tools" / "rehearse_serve_sdk_offline.py"
#: the qids of SDK_MIXED_BODY, in the driver's own vocabulary
MIXED_KINDS = {"urgency": "score", "area": "choice", "spam": "noul"}


def _gate_driver():
    """The host gate's client driver, imported from `tools/` (the `test_probe_isolation` trade)."""
    spec = importlib.util.spec_from_file_location("host_gate_serve_client", GATE_DRIVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["host_gate_serve_client"] = module
    spec.loader.exec_module(module)
    return module


def _served_typesafe_body() -> dict[str, Any]:
    """What `/v1/systemone` answers for SDK_MIXED_BODY — the real mapping, then the projection."""
    native = mixed_engine_body(SDK_MIXED_BODY)
    return json.loads(json.dumps(schema.render_response(native, format="typesafe")))


def test_the_host_gate_driver_accepts_the_body_our_mapping_serves() -> None:
    """The gate's own judgements must pass on the body this product produces, or the host run is
    guaranteed red for a reason that has nothing to do with the host."""
    driver = _gate_driver()
    body = _served_typesafe_body()
    assert set(body["answers"]) == set(MIXED_KINDS)
    assert driver.verify(body, questions=MIXED_KINDS) == []
    # the annotated view is the shape the driver actually judges: the server sends bits (JSON) and
    # the SDK hands `model_dump()` a decoded document, so the served body has to pass as it is
    assert driver.verify(json.loads(json.dumps(body)), questions=MIXED_KINDS) == []


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda body: body["answers"].pop("area"), "answers are"),
        (lambda body: body["answers"]["area"].pop("confidence"), "choice) is missing"),
        (lambda body: body["answers"]["area"].update({"confidence": 1.5}), "outside [0, 1]"),
        (lambda body: body["answers"]["area"].update({"choice": "sales"}),
         "is not the argmax"),
        (lambda body: body["answers"]["urgency"].update({"score": 9.0}), "outside [0, 2]"),
        (lambda body: body["answers"]["urgency"].update({"legend": {"7": "no"}}), "legend keys"),
        (lambda body: body["answers"]["spam"].update({"confidence": 0.5}),
         "noul carries no confidence"),
        (lambda body: body["answers"]["spam"].update({"probabilities": {"yes": 0.4, "no": 0.4}}),
         "sum to"),
        (lambda body: body.pop("usage"), "usage keys are"),
        (lambda body: body["usage"].update({"input_tokens": 0}), "input_tokens is 0"),
        (lambda body: body.update({"model": ""}), "must echo the resolved name"),
        (lambda body: body.update({"engine": {}}), "top-level keys are"),
    ],
)
def test_the_host_gate_driver_names_each_mismatch_it_can_find(mutation, expected) -> None:
    driver = _gate_driver()
    body = _served_typesafe_body()
    mutation(body)
    problems = driver.verify(body, questions=MIXED_KINDS)
    assert problems, "this mutation must be caught"
    assert any(expected in problem for problem in problems), (expected, problems)


def test_the_host_gate_driver_refuses_a_missing_or_different_sdk(tmp_path) -> None:
    """A gate that measures one wire must not run against another: exit 3, loudly, before any call.

    With the pinned SDK installed somewhere else this checks the *version* branch; with no SDK in
    this interpreter it checks the *missing* branch — the same refusal, measured either way.
    """
    if importlib.util.find_spec("typesafe_sdk") is None:
        argv = [sys.executable, str(GATE_DRIVER), "--base-url", "http://127.0.0.1:1"]
        expected = "no typesafe-sdk"
    else:                                                    # pragma: no cover - env dependent
        argv = [sys.executable, str(GATE_DRIVER), "--base-url", "http://127.0.0.1:1",
                "--expect-sdk", "0.0.0"]
        expected = "REFUSING TO RUN"
    completed = subprocess.run(argv, cwd=ROOT_DIR, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 3, completed
    assert expected in completed.stderr, completed.stderr
    assert completed.stdout == "", "a refusal reaches no socket and prints no answers"


def test_the_host_gate_script_pins_the_sdk_and_walks_the_documented_steps() -> None:
    script = GATE_SCRIPT.read_text(encoding="utf-8")
    assert "typesafe-sdk==0.7.1" in script
    assert os.access(GATE_SCRIPT, os.X_OK), "the host gate is run as a script"
    for needle in ("TYPESAFE_BASE_URL", "TYPESAFE_API_KEY", "host_gate_serve_client.py",
                   "0.7.1", "/health", "/v1/models", "served_by=host", "keep stop",
                   "leaked host", "TYPED_GGUF_GATE_MODEL"):
        assert needle in script, f"the host gate no longer does {needle!r}"
    # the rehearsal and the driver are the same tools the container can run: no drift between them
    rehearsal = REHEARSAL.read_text(encoding="utf-8")
    assert '"host_gate_serve_client.py"' in rehearsal, "the rehearsal drives the gate's own client"
    assert "tools" in rehearsal and "verify(" in GATE_DRIVER.read_text(encoding="utf-8")


def test_the_gate_model_default_is_the_pinned_4b_or_an_override() -> None:
    """The host gate names its model once, and the env override is the documented one."""
    script = GATE_SCRIPT.read_text(encoding="utf-8")
    lines = [line for line in script.splitlines() if line.startswith("MODEL=")]
    assert len(lines) == 1, lines
    assert "TYPED_GGUF_GATE_MODEL" in lines[0] and "gguf" in lines[0], lines[0]


# -------------------------------- 11. the helpers the model row is built from (mutation round 2)
def test_the_model_row_helpers_are_exact() -> None:
    """`/v1/models` only sampled these: an exact row is what the row's own wording claims.

    (Mutation round 1, card t_f5d8b6c7: the `_human_size` / `_release_date` / `_description`
    families survived a gate that asserted a prefix — 63 mutants on paths whose whole job is the
    one line a caller reads.)
    """
    assert serve._human_size(None) == "size unknown"
    assert serve._human_size(0) == "size unknown"
    assert serve._human_size(512) == "512 B"
    assert serve._human_size(1024) == "1.0 KiB"
    assert serve._human_size(4_000_000_000) == "3.7 GiB"
    assert serve._human_size(2 * 1024 ** 4) == "2.0 TiB"
    assert serve._human_size(1024 ** 5) == "1024.0 TiB", "TiB is the last unit the loop offers"
    entry = store.Entry(alias="a", path="/nonexistent.gguf", arch="qwen35", quant="Q8_0",
                        size=4_000_000_000)
    assert serve._description(entry) == ("qwen35 Q8_0 GGUF (3.7 GiB), local; typed-gguf's own "
                                        "engine")
    bare = dataclasses.replace(entry, arch=None, quant=None, size=None)
    assert serve._description(bare) == ("unknown arch unknown quant GGUF (size unknown), local; "
                                        "typed-gguf's own engine")


def test_the_release_date_is_this_copy_s_mtime_and_never_a_guess(tmp_path) -> None:
    weights = tmp_path / "w.gguf"
    weights.write_bytes(b"GGUF")
    os.utime(weights, (1_700_000_000, 1_700_000_000))
    dated = store.Entry(alias="a", path=str(weights), arch="qwen35", quant="Q8_0", size=4,
                        added_at="2024-05-06T07:08:09Z")
    assert serve._release_date(str(weights), dated) == "2023-11-14", "the file's own mtime, in UTC"
    missing = str(tmp_path / "gone.gguf")
    assert serve._release_date(missing, dated) == "2024-05-06", "no file: the entry's own date"
    undated = dataclasses.replace(dated, added_at="")
    assert serve._release_date(missing, undated) == "", "no file and no date: nothing invented"
    assert serve._release_date(missing, dataclasses.replace(dated, added_at="sometime")) == ""


def test_the_typed_error_location_follows_the_code_and_the_question(tmp_path) -> None:
    payload = {"questions": {"q1": {"type": "noul"}, "o'brien": {"type": "score"}}}
    assert serve._location_for(UserError("x", code="E_STATE_EMPTY"), payload) == ("body", "state")
    assert serve._location_for(UserError("x", code="E_MODEL_NOT_FOUND"), payload) == ("body",
                                                                                     "model")
    # a question-level code lands on the question it names, when the payload has it
    assert serve._location_for(UserError("question 'q1': x", code="E_SCORE_LEVELS"),
                               payload) == ("body", "questions", "q1", "criteria")
    # …a quoted id (the message escapes it) and one the payload does not carry
    assert serve._location_for(UserError("question 'o\\'brien': x", code="E_Q_TYPE_UNKNOWN"),
                               payload) == ("body", "questions", "o'brien", "type")
    assert serve._location_for(UserError("question 'gone': x", code="E_CHOICE_CRITERIA"),
                               payload) == ("body", "questions")
    assert serve._location_for(UserError("no id at all", code="E_CHOICE_CRITERIA"),
                               payload) == ("body", "questions")
    assert serve._location_for(UserError("x", code="E_UNKNOWN_KEY"), {"questions": "no"}) == \
        ("body",), "a code the map does not know locates the body and stops there"


def test_run_owns_the_server_lifecycle(monkeypatch) -> None:
    """`run` is the blocking half the CLI calls: it must bind, serve, and close — once."""
    events: list[Any] = []

    class FakeServer:
        def serve_forever(self) -> None:
            events.append("serve_forever")

        def server_close(self) -> None:
            events.append("server_close")

    fake = FakeServer()

    def make_server(app, host, port):
        events.append(("make_server", app, host, port))
        return fake

    monkeypatch.setattr(serve, "make_server", make_server)
    app = serve.App(decide=_no_loading, home=None, log=Log())
    assert serve.run(app, "127.0.0.1", 8123) == 0
    assert events == [("make_server", app, "127.0.0.1", 8123), "serve_forever", "server_close"]

    class Interrupted(FakeServer):
        def serve_forever(self) -> None:
            raise KeyboardInterrupt

    monkeypatch.setattr(serve, "make_server", lambda *a, **k: Interrupted())
    assert serve.run(app) == 0, "a Ctrl-C is a clean exit, and the socket still closes"
