"""stdlib HTTP server: `/health`, `/v1/models`, `/v1/systemone`, `/v1/decide` (SPEC 2.9).

`typed-gguf serve` is a `http.server` server with **no engine of its own** (SPEC 2.12): every
decision goes through the `decide` callable it is built with, which the CLI hands it as the warm
CLI path (`cli.decide_payload_warm` → the resident keep host). Same model, same fit plan, same
calibration, same numbers as `ask`/`run` — the served numbers *are* the engine's numbers.

Two surface families are mounted at every `--format`:

* `/health` — liveness and what is resident, read from the keep ledger (never loads, never waits).
* `/v1/models` + `/v1/systemone` — the TypeSafe wire, exactly as `typesafe-sdk` 0.7.1 speaks it
  (`tests/fixtures/typesafe_sdk_0_7_1_fields.json`, captured from the SDK by
  `tools/typesafe_fields_capture.py`). `/v1/systemone` always answers the §2.6 projection: a client
  that only sets `TYPESAFE_BASE_URL` must not depend on a server-side flag.
* `/v1/decide` — the native §2.5 wire verbatim; `--format` is only its default response format.

`App.handle` is a pure function of (method, target, headers, body bytes) → `Response`: the tests
drive it directly, with no socket in the way. `Handler`/`make_server` are the thin HTTP layer over
it — real request parsing and `Content-Length`, nothing else.

Errors are FastAPI-shaped on the TypeSafe routes `{"detail": [...]}` — the shape the SDK's own
`HTTPValidationError`/`ValidationError` models describe and `extract_message` reads — and native
(`{"error": {"code", "message"}}`) on `/v1/decide`.

Logging: one line per request on the app's `log` sink (stderr by default). Never the
`Authorization` value, never a request body.
"""
from __future__ import annotations

import dataclasses
import datetime
import http.server
import json
import pathlib
import re
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from typed_gguf import __version__, schema
from typed_gguf.errors import TypedGgufError
from typed_gguf.keep import state as keep_state
from typed_gguf.registry import store

#: The `service` field of `/health` (and the `Server`'s server-version string).
SERVICE = "typed-gguf"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8088
#: The compat name every request may use, and what it resolves to: the registry's `current`
#: (`typesafe_sdk.constants.DEFAULT_MODEL`).
COMPAT_MODEL = "jev-latest"

HEALTH_PATH = "/health"
MODELS_PATH = "/v1/models"
SYSTEM_ONE_PATH = "/v1/systemone"
DECIDE_PATH = "/v1/decide"
#: Every route this server answers. Anything else is the §2.9 `404` row.
ROUTES: tuple[str, ...] = (HEALTH_PATH, MODELS_PATH, SYSTEM_ONE_PATH, DECIDE_PATH)
#: `/v1/systemone`'s body keys, in the SDK's own order (`SystemOneRequest`): these three *only*.
REQUEST_KEYS: tuple[str, ...] = ("state", "model", "questions")
#: `/v1/systemone`'s top-level keys (`SystemOneResponse`).
SERVICE_KEYS: tuple[str, ...] = ("model", "answers", "usage")
#: `usage`'s keys (`Usage`): the native counters, verbatim.
USAGE_KEYS: tuple[str, ...] = ("input_tokens", "output_tokens")
#: `/v1/models`'s item keys (`ModelMetadata`).
MODELS_KEYS: tuple[str, ...] = ("name", "description", "release_date")
#: The answer key sets of §2.9/§2.6 — `type` + value key + `probabilities`, plus `confidence` for
#: choice/score and `legend` for score. `noul` carries no `confidence` (the SDK's own answer model
#: has no such field).
ANSWER_KEYS: dict[str, tuple[str, ...]] = {
    "noul": ("type", "noul", "probabilities"),
    "choice": ("type", "choice", "confidence", "probabilities"),
    "score": ("type", "score", "confidence", "legend", "probabilities"),
}
#: The discriminator of the SDK's answer union, in its declaration order.
ANSWER_TYPES: tuple[str, ...] = ("noul", "choice", "score")
#: The answer key that carries the decision itself (SPEC 2.6's value key).
VALUE_KEYS = {"noul": "noul", "choice": "choice", "score": "score"}
#: Where a `_*`/`E_*` code belongs in the request, for the FastAPI-shaped `loc` (§2.9). The codes
#: that name a question carry its id in the message (`question 'x': …`), which is parsed out.
CODE_LOCATION: dict[str, tuple[str, ...]] = {
    "E_STATE_EMPTY": ("state",),
    "E_MODEL_NOT_FOUND": ("model",),
}
QUESTION_KEYWORD: dict[str, str] = {
    "E_Q_TYPE_UNKNOWN": "type",
    "E_CHOICE_CRITERIA": "criteria",
    "E_CHOICE_TOO_MANY": "criteria",
    "E_SCORE_LEVELS": "criteria",
    "E_NOUL_CRITERIA": "criteria",
}
#: `schema`'s question-level messages all start with `question <repr>: …`.
QUESTION_ID = re.compile(r"^question '((?:[^'\\]|\\.)*)'")
#: The header the SDK reads back for its own logs (`_core.constants.REQUEST_ID_HEADER`).
REQUEST_ID_HEADER = "x-typesafe-request-id"

Decide = Callable[[dict[str, Any]], dict[str, Any]]


@dataclasses.dataclass(frozen=True)
class Response:
    """One answered request: the status, the JSON body, and what the server did with it."""

    status: int
    body: bytes
    headers: dict[str, str] = dataclasses.field(default_factory=dict)
    #: `"host"`/`"inline"` from the decision's `engine.keep`, or None when no engine ran. A local
    #: fact about the answer, deliberately *not* a wire field (§2.9 lists the served keys).
    served_by: str | None = None

    def payload(self) -> Any:
        """The body as parsed JSON (the tests' own reader)."""
        return json.loads(self.body.decode("utf-8"))


def _json(headers: Mapping[str, str] | None = None) -> dict[str, str]:
    """The JSON content type plus whatever else this response carries."""
    merged = {"Content-Type": "application/json"}
    merged.update(headers or {})
    return merged


def _problem(status: int, detail: Any, headers: Mapping[str, str] | None = None) -> Response:
    """A response whose body is `detail` — the shape is the caller's business."""
    return Response(status, json.dumps(detail).encode("utf-8"), _json(headers))


def _not_found() -> Response:
    """SPEC 2.9: an unknown route (or a method a route does not answer) is the 404 row."""
    return _problem(404, {"detail": "Not Found"})


def _validation(details: list[dict[str, Any]],
                headers: Mapping[str, str] | None = None) -> Response:
    """`422` in the SDK's own `HTTPValidationError` shape."""
    return _problem(422, {"detail": details}, headers)


def _with_code(exc: TypedGgufError) -> str:
    """`E_X: <message>` — the typed code always rides in the message (§2.9)."""
    text = str(exc).strip()
    prefix = f"{exc.code}: "
    return text if text.startswith(prefix) else f"{prefix}{text}"


def _human_size(size: int | None) -> str:
    """A byte count as the one unit a human reads (`4.0 GiB`), or `size unknown`."""
    if not size:
        return "size unknown"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TiB"                        # pragma: no cover - unreachable loop tail


def _release_date(path: str, entry: store.Entry) -> str:
    """The model file's mtime as a UTC date; the entry's own `added_at` date as the fallback.

    SPEC 2.9: when *this copy* was written on this box. We do not know an upstream release date and
    do not invent one; a file we cannot stat and an entry we cannot date answer `""`.
    """
    try:
        stamp = pathlib.Path(path).stat().st_mtime
    except OSError:
        added = str(entry.added_at or "")
        return added[:10] if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", added) else ""
    return datetime.datetime.fromtimestamp(stamp, datetime.UTC).strftime("%Y-%m-%d")


def _description(entry: store.Entry) -> str:
    """One honest auto line, no marketing (SPEC 2.9)."""
    arch = entry.arch or "unknown arch"
    quant = entry.quant or "unknown quant"
    return f"{arch} {quant} GGUF ({_human_size(entry.size)}), local; typed-gguf's own engine"


class App:
    """The routes, as a pure function of the request — no socket, no thread, no globals.

    `decide` is the one engine seam: it takes the *native* §2.5 payload and returns the native §2.5
    response. `serve` never reaches into a model itself; the CLI passes the warm keep-host path, so
    a served answer and an `ask` answer are the same code answering the same request (§2.12).
    """

    def __init__(self, *, decide: Decide, home: pathlib.Path | None = None,
                 version: str = __version__, keep_alive: float | None = None,
                 default_format: str = "native", log: Callable[[str], None] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if default_format not in schema.FORMATS:
            raise ValueError(f"default_format must be one of {', '.join(schema.FORMATS)}")
        self.decide = decide
        self.home = home
        self.version = version
        self.keep_alive = keep_alive
        self.default_format = default_format
        self.log = log if log is not None else _stderr_log
        self.clock = clock
        self.requests = 0
        #: Decisions serialize (SPEC 2.12): one model, one host, one at a time, arrival order.
        #: `/health` and `/v1/models` never take this lock — they must not wait behind a decision.
        self.decision_lock = threading.Lock()

    # ------------------------------------------------------------------ the one entry point
    def handle(self, method: str, target: str, headers: Mapping[str, str],
               body: bytes, *, request_id: str | None = None) -> Response:
        """Answer one request. `target` is the raw request target (a query string is ignored)."""
        started = self.clock()
        path = target.split("?", 1)[0]
        self.requests += 1
        rid = request_id or uuid.uuid4().hex[:16]
        response = self._route(method, path, headers, body, rid)
        elapsed_ms = (self.clock() - started) * 1000.0
        self.log(f"{method} {path} {response.status} served_by={response.served_by or '-'} "
                 f"{elapsed_ms:.1f}ms req={rid}")
        return response

    def _route(self, method: str, path: str, headers: Mapping[str, str], body: bytes,
               rid: str) -> Response:
        extra = {REQUEST_ID_HEADER: rid}
        if method == "GET" and path == HEALTH_PATH:
            return _problem(200, self._health(), extra)
        if method == "GET" and path == MODELS_PATH:
            return _problem(200, self._models(), extra)
        if method == "POST" and path == SYSTEM_ONE_PATH:
            return self._system_one(body, extra)
        if method == "POST" and path == DECIDE_PATH:
            return self._decide(body, extra)
        return _not_found()

    # ------------------------------------------------------------------ /health, /v1/models
    def _health(self) -> dict[str, Any]:
        """The ledger's own answer: no ping, no load, no wait (SPEC 2.9)."""
        record = keep_state.read_record(self.home)
        warm = bool(record is not None and keep_state.alive(record, home=self.home))
        return {"status": "ok", "service": SERVICE, "version": self.version,
                "model": record.model if warm and record is not None else None, "warm": warm}

    def _models(self) -> dict[str, Any]:
        """The registry's aliases (sorted) plus the compat name, which resolves to `current`."""
        registry, _warnings = store.load_registry(store.registry_path(self.home))
        listed = [self._model(name, entry)
                  for name, entry in sorted(registry.aliases.items())]
        current = store.resolve(registry, None, use_current=True)
        if current is not None:
            listed.append(self._model(COMPAT_MODEL, current))
        return {"models": listed}

    def _model(self, name: str, entry: store.Entry) -> dict[str, Any]:
        return {"name": name, "description": _description(entry),
                "release_date": _release_date(entry.path, entry)}

    # ------------------------------------------------------------------ /v1/systemone
    def _system_one(self, body: bytes, extra: Mapping[str, str]) -> Response:
        """The SDK's own body in, the §2.6 projection out."""
        decoded = self._decode(body, extra)
        if isinstance(decoded, Response):
            return decoded
        payload = decoded
        missing = [key for key in REQUEST_KEYS if key not in payload]
        if missing:
            return _validation([_missing(key) for key in missing], extra)
        unknown = sorted(str(key) for key in payload if key not in REQUEST_KEYS)
        if unknown:
            return _validation([_extra_forbidden(key) for key in unknown], extra)
        alias, refusal = self._resolve(payload["model"], payload, extra)
        if refusal is not None:
            return refusal
        native = {"state": payload["state"], "model": alias, "questions": payload["questions"]}
        try:
            answer = self._decide_locked(native)
        except TypedGgufError as exc:
            return _typesafe_error(exc, payload, extra)
        except Exception as exc:                       # noqa: BLE001 - never a traceback over HTTP
            return _problem(500, {"detail": f"E_INTERNAL: {exc.__class__.__name__}: {exc}"}, extra)
        projected = schema.render_response(answer, format="typesafe")
        return Response(200, json.dumps(projected).encode("utf-8"), _json(extra),
                        served_by=_served_by(answer))

    def _resolve(self, model: Any, payload: Mapping[str, Any],
                 extra: Mapping[str, str]) -> tuple[str, Response | None]:
        """`alias | jev-latest` → the registry's alias. A path or `repo:quant` is refused (§2.9)."""
        registry, _warnings = store.load_registry(store.registry_path(self.home))
        entry = None
        if isinstance(model, str) and model != COMPAT_MODEL:
            entry = registry.aliases.get(model)
        elif model == COMPAT_MODEL:                     # the compat name: the registry's `current`
            entry = store.resolve(registry, None, use_current=True)
        if entry is None:
            hint = ("no model is in the registry — run `typed-gguf models pull` (or "
                    "`typed-gguf models use`) first" if not registry.aliases else
                    f"known aliases: {', '.join(sorted(registry.aliases))} (use "
                    f"`typed-gguf models pull` or `typed-gguf models use`)")
            # a remote client must not point the server at files: no path/repo resolution here
            return "", _typesafe_detail(
                "value_error", ("model",), f"E_MODEL_NOT_FOUND: {model!r} is not a registry alias "
                                           f"or `{COMPAT_MODEL}`; {hint}", model, extra)
        return entry.alias, None

    def _decode(self, body: bytes, extra: Mapping[str, str]) -> dict[str, Any] | Response:
        """The body as a JSON object, or the `json_invalid` 422 the SDK expects (SPEC 2.9)."""
        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return _validation([_json_invalid(f"the body is not valid JSON: {exc}")], extra)
        if not isinstance(decoded, dict):
            return _validation([_json_invalid("the body must be a JSON object")], extra)
        return decoded

    # ------------------------------------------------------------------ /v1/decide
    def _decide(self, body: bytes, extra: Mapping[str, str]) -> Response:
        """The native §2.5 wire verbatim: the body's own `format`, then the flag's default."""
        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return _native_error("E_UNKNOWN_KEY", f"the body is not valid JSON: {exc}", 400, extra)
        if not isinstance(decoded, dict):
            return _native_error("E_UNKNOWN_KEY", "the body must be a JSON object", 400, extra)
        payload = dict(decoded)
        payload.setdefault("format", self.default_format)
        try:
            answer = self._decide_locked(payload)
        except TypedGgufError as exc:
            status = {2: 400, 3: 503}.get(exc.exit_code, 500)
            return _native_error(exc.code, str(exc), status, extra)
        except Exception as exc:                       # noqa: BLE001 - never a traceback over HTTP
            return _native_error("E_INTERNAL", f"{exc.__class__.__name__}: {exc}", 500, extra)
        return Response(200, json.dumps(answer).encode("utf-8"), _json(extra),
                        served_by=_served_by(answer))

    # ------------------------------------------------------------------ the engine seam
    def _decide_locked(self, payload: dict[str, Any]) -> dict[str, Any]:
        """One decision, serialized with every other decision this server is running."""
        with self.decision_lock:
            return self.decide(payload)


def _served_by(answer: Mapping[str, Any]) -> str | None:
    """Who answered: the warm host, or the inline fallback the client names (SPEC 2.12)."""
    engine = answer.get("engine")
    keep = engine.get("keep") if isinstance(engine, Mapping) else None
    served_by = keep.get("served_by") if isinstance(keep, Mapping) else None
    return str(served_by) if served_by else None


# --------------------------------------------------------------------- error bodies (§2.9 table)
def _json_invalid(message: str) -> dict[str, Any]:
    return {"type": "json_invalid", "loc": ["body"], "msg": message}


def _missing(key: str) -> dict[str, Any]:
    return {"type": "missing", "loc": ["body", key], "msg": "Field required"}


def _extra_forbidden(key: str) -> dict[str, Any]:
    return {"type": "extra_forbidden", "loc": ["body", key],
            "msg": f"Extra inputs are not permitted (the request takes {', '.join(REQUEST_KEYS)} "
                   f"only)"}


def _typesafe_detail(kind: str, loc: tuple[str, ...], message: str, given: Any = None,
                     headers: Mapping[str, str] | None = None) -> Response:
    detail: dict[str, Any] = {"type": kind, "loc": ["body", *loc], "msg": message}
    if given is not None:
        detail["input"] = given
    return _validation([detail], headers)


def _typesafe_error(exc: TypedGgufError, payload: Mapping[str, Any],
                    extra: Mapping[str, str]) -> Response:
    """One engine failure, as the §2.9 table's row for its exit status."""
    if exc.exit_code != 2:
        return _problem(500, {"detail": _with_code(exc)}, extra)
    loc = _location_for(exc, payload)
    detail: dict[str, Any] = {"type": "value_error", "loc": list(loc),
                              "msg": _with_code(exc)}
    if len(loc) > 1:
        given = _value_at(payload, loc[1:])
        if given is not None:
            detail["input"] = given
    return _validation([detail], extra)


def _location_for(exc: TypedGgufError, payload: Mapping[str, Any]) -> tuple[str, ...]:
    """The request path a code belongs to, FastAPI-style (`["body","questions","x","criteria"]`)."""
    fixed = CODE_LOCATION.get(exc.code)
    if fixed is not None:
        return ("body", *fixed)
    keyword = QUESTION_KEYWORD.get(exc.code)
    if keyword is not None:
        match = QUESTION_ID.match(str(exc))
        qid = match.group(1).replace("\\'", "'") if match else None
        if qid is not None and isinstance(payload.get("questions"), Mapping) \
                and qid in payload["questions"]:
            return ("body", "questions", qid, keyword)
        return ("body", "questions")
    return ("body",)


def _value_at(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    """The value at a `loc` path inside the request, when it is there (for `input`)."""
    current: Any = payload
    for step in path:
        if isinstance(current, Mapping) and step in current:
            current = current[step]
        else:
            return None
    return current


def _native_error(code: str, message: str, status: int,
                  headers: Mapping[str, str]) -> Response:
    """The native §2.5 error: `{"error": {"code", "message"}}` (the keep host's own shape)."""
    return _problem(status, {"error": {"code": code, "message": message}}, headers)


def _stderr_log(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- the HTTP layer
class Handler(http.server.BaseHTTPRequestHandler):
    """The thin HTTP half: parse the request, hand the bytes to the app, write the answer.

    `protocol_version = "HTTP/1.1"` with the stdlib's keep-alive loop (`BaseHTTPRequestHandler
    .handle`) and an always-present `Content-Length`, which is what an SDK's HTTP client expects.
    """

    protocol_version = "HTTP/1.1"
    server_version = f"{SERVICE}/{__version__}"
    sys_version = ""

    def do_GET(self) -> None:                            # noqa: N802 - the stdlib's own names
        self._serve("GET")

    def do_POST(self) -> None:                           # noqa: N802 - the stdlib's own names
        self._serve("POST")

    def _serve(self, method: str) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length > 0 else b""
        except (ValueError, OSError):                    # a broken length: a bad request, not a 500
            length, body = 0, b""
        app = self.server.app                              # type: ignore[attr-defined]
        response = app.handle(method, self.path, dict(self.headers.items()), body)
        self.send_response(response.status)
        for name, value in response.headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(response.body)))
        self.end_headers()
        self.wfile.write(response.body)

    def log_message(self, format: str, *args: Any) -> None:   # noqa: A002 - the stdlib's signature
        """Silent: the app's own log line is the request record (and never a header value)."""


class Server(http.server.ThreadingHTTPServer):
    """One thread per connection, one decision at a time (`App.decision_lock`).

    Threading on purpose: `/health` and `/v1/models` must never wait behind a decision (SPEC 2.9),
    which a single-threaded server cannot promise.
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], app: App) -> None:
        self.app = app
        super().__init__(address, Handler)


def make_server(app: App, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> Server:
    """Bind the server (exposed so a caller — the CLI, a test — can own the lifecycle)."""
    return Server((host, int(port)), app)


def run(app: App, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> int:
    """Serve until interrupted. Returns the exit code (0 for a clean Ctrl-C)."""
    server = make_server(app, host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0
