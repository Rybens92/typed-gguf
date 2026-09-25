"""`tools/matrix_serve_probe.py` — the serve assertions the matrix jobs make (t_f96fed7f, AC2/AC3).

Two jobs grow a serve story in this card, and they must make *opposite* claims about the same
product:

* **macOS (AC3)** — the warm host exists, so the decision *after* the first one must be answered by
  it: `engine.keep.served_by == "host"` and `timings.model_load_ms == 0.0` (SPEC 2.12: "a warm
  answer reports `0.0`"), with the server's own log line saying `served_by=host`.
* **Windows (AC2)** — `keep.supported()` is false for `win*` (`typed_gguf.keep.identity`), so
  `serve`'s decisions answer **inline on that same call**, naming the reason: the
  `W_KEEP_UNAVAILABLE` warning on the server's own output (SPEC 2.12 "Platforms"), and a served
  body with **no** `engine.keep` block at all — the pre-E4 shape, because the unsupported-platform
  branch returns `decide_payload`'s body untouched (`cli.decide_payload_warm`).

The tool is both the readiness poll (no fixed sleeps: `/health` decides) and the judge. `judge` is a
pure function over collected facts, so the RED directions — a Windows run that claims a host, a
macOS second call that paid a load, a missing warning — are gates rather than prose. The HTTP half
is exercised over a real loopback `serve.App` (skipped, by name, when the net-off flag forbids
`AF_INET`); the rest of the suite drives `judge` directly.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import threading
from typing import Any

import pytest

from typed_gguf.api import http as serve

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "matrix_serve_probe.py"
#: The flag `.github/workflows/ci.yml` sets on the whole offline suite: it forbids `AF_INET`,
#: which is the family a loopback probe binds (`tests/conftest.py`).
NET_BLOCK_ENV = "TYPED_GGUF_TEST_BLOCK_NET"
ALIAS = "smoke-0.5b"
#: The native §2.5 body the probe sends: one choice and one score, the SDK's own question shapes.
#: The *content* is pinned here on purpose — the served decision must be about a real question set,
#: not an empty shell that answers vacuously.
QUESTIONS = {
    "area": {"type": "choice", "instructions": "Which area is this about?",
             "criteria": {"billing": "Payments, invoicing, refunds",
                          "technical": "API, uptime, errors"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["Can wait", "This week", "Today"]},
}


def _net_blocked() -> bool:
    return os.environ.get(NET_BLOCK_ENV) in ("1", "true", "yes")


def load_tool():
    if not TOOL.exists():
        raise AssertionError(
            f"{TOOL.relative_to(ROOT)} is missing: the macOS and Windows serve steps call it, and "
            "without it those jobs would assert nothing about who answered (card t_f96fed7f)")
    spec = importlib.util.spec_from_file_location("matrix_serve_probe", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------- the bodies the engine hands out
def answer_body(load_ms: float, *, host: bool) -> dict[str, Any]:
    """A native §2.5 decision, as `serve` renders it (plus the keep block the warm path writes)."""
    body: dict[str, Any] = {
        "model": ALIAS,
        "answers": {
            "area": {"type": "choice", "choice": "billing", "confidence": 0.6,
                     "probabilities": {"billing": 0.7, "technical": 0.3}},
            "urgency": {"type": "score", "score": 1.3, "legend": {"0": "Can wait"},
                        "confidence": 0.5, "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2}},
        },
        "usage": {"input_tokens": 128, "output_tokens": 4},
        "timings": {"model_load_ms": load_ms, "prefill_ms": 400.0, "questions_ms": 30.0},
        "engine": {"backend": "cpu", "runtime": "llama.cpp b11026"},
    }
    if host:
        body["engine"]["keep"] = {"served_by": "host", "pid": 4242, "requests": 1,
                                  "model_load_ms": 1200.0, "keep_alive_s": 600.0}
    return body


WARM = answer_body(0.0, host=True)
COLD = answer_body(1200.0, host=True)
#: What an unsupported-platform decision looks like: `decide_payload`'s body, no keep block at all.
INLINE = answer_body(1200.0, host=False)
#: The warning the CLI prints on a platform without unix sockets (SPEC 2.12), on the server's
#: own stderr — the workflow points both streams at the file the probe reads.
WARNING = ("warning: W_KEEP_UNAVAILABLE: keep-alive needs unix sockets, which this platform does "
           "not have — answering inline instead of keeping a host alive (SPEC 2.12)")
HOST_LOG = "POST /v1/decide 200 served_by=host 4.1ms req=abc\n"
INLINE_LOG = f"GET /health 200 served_by=- 0.1ms req=aaa\nPOST /v1/decide 200 served_by=- 900.0ms req=bbb\n"


def facts(decisions: list[dict[str, Any]], log: str = HOST_LOG) -> Any:
    """The probe's collected facts, the way `probe()` hands them to `judge`."""
    module = load_tool()
    return module.Facts(base_url="http://127.0.0.1:8088", health={"status": "ok"}, decisions=decisions,
                        log=log)


# ------------------------------------------------------------------------- the warm-host story
def test_a_warm_second_decision_is_the_macos_claim() -> None:
    module = load_tool()
    assert module.judge(facts([COLD, WARM]), expect="host") == []


def test_a_second_decision_that_paid_a_load_is_refused() -> None:
    """Both bodies are `served_by=host`, so only `model_load_ms` can tell warm from cold."""
    module = load_tool()
    problems = module.judge(facts([COLD, COLD]), expect="host")
    assert problems and any("model_load_ms" in problem for problem in problems), problems


def test_a_second_decision_the_host_did_not_answer_is_refused() -> None:
    module = load_tool()
    body = json.loads(json.dumps(WARM))
    body["engine"]["keep"]["served_by"] = "inline"
    problems = module.judge(facts([COLD, body]), expect="host")
    assert problems and any("served_by" in problem for problem in problems), problems


def test_a_decision_with_no_keep_block_at_all_is_refused_in_host_mode() -> None:
    module = load_tool()
    problems = module.judge(facts([COLD, INLINE]), expect="host")
    assert problems and any("engine.keep" in problem for problem in problems), problems


def test_a_server_log_that_never_names_the_host_is_refused() -> None:
    module = load_tool()
    problems = module.judge(facts([COLD, WARM], log=INLINE_LOG), expect="host")
    assert problems and any("served_by=host" in problem for problem in problems), problems


def test_a_decision_without_answers_is_not_an_answer() -> None:
    """HTTP 200 is not the claim: the served body has to carry the typed decisions."""
    module = load_tool()
    body = json.loads(json.dumps(WARM))
    body["answers"] = {}
    problems = module.judge(facts([COLD, body]), expect="host")
    assert problems and any("answers" in problem for problem in problems), problems


# ---------------------------------------------------------------------- the Windows fallback story
def test_an_inline_decision_with_the_named_warning_is_the_windows_claim() -> None:
    module = load_tool()
    log = INLINE_LOG + WARNING + "\n"
    assert module.judge(facts([INLINE], log=log), expect="inline") == []


def test_a_windows_style_decision_that_claims_a_host_is_refused() -> None:
    """`keep.supported()` is false for `win*`: a host claim there means the platform rule broke."""
    module = load_tool()
    problems = module.judge(facts([WARM], log=HOST_LOG), expect="inline")
    assert problems and any("win" in problem for problem in problems), problems


def test_an_inline_decision_without_the_named_warning_is_refused() -> None:
    """The warning is the contract (SPEC 2.12): a silent inline answer is not the documented one."""
    module = load_tool()
    problems = module.judge(facts([INLINE], log=INLINE_LOG), expect="inline")
    assert problems and any("W_KEEP_UNAVAILABLE" in problem for problem in problems), problems


def test_an_inline_decision_that_shows_up_as_a_host_in_the_log_is_refused() -> None:
    module = load_tool()
    problems = module.judge(facts([INLINE], log=INLINE_LOG + WARNING + "\n" + HOST_LOG),
                            expect="inline")
    assert problems and any("served_by=host" in problem for problem in problems), problems


def test_an_expectation_this_tool_does_not_know_is_refused() -> None:
    module = load_tool()
    problems = module.judge(facts([WARM]), expect="warmish")
    assert problems and any("warmish" in problem for problem in problems), problems


# ------------------------------------------------------------------------------- the HTTP half
class Logged:
    """A `serve.App` log sink that also stands in for the process's stderr (one file, 2>&1)."""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        with open(self.path, "w", encoding="utf-8"):
            pass

    def __call__(self, line: str) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def warning(self) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(WARNING + "\n")


def _serve_in_thread(app: serve.App) -> tuple[serve.Server, threading.Thread]:
    server = serve.make_server(app, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


@pytest.mark.skipif(_net_blocked(), reason=(
    "TYPED_GGUF_TEST_BLOCK_NET forbids AF_INET, the family a loopback probe binds"))
def test_the_probe_accepts_a_real_loopback_server_and_the_warm_host(tmp_path: pathlib.Path) -> None:
    """End to end over a socket: readiness poll, two native decisions, the server's own log."""
    module = load_tool()
    log = Logged(tmp_path / "serve.log")
    calls: list[int] = []

    def decide(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(len(calls))
        return json.loads(json.dumps(COLD if len(calls) == 1 else WARM))

    app = serve.App(decide=decide, home=tmp_path / "home", log=log)
    server, thread = _serve_in_thread(app)
    try:
        port = int(server.server_address[1])
        code = module.main(["--base-url", f"http://127.0.0.1:{port}", "--model", ALIAS,
                            "--server-log", str(log.path), "--expect", "host",
                            "--ready-budget", "30"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)
    assert code == 0, log.path.read_text(encoding="utf-8")
    assert len(calls) == 2, "the probe decides twice: cold, then the warm one it judges"


@pytest.mark.skipif(_net_blocked(), reason=(
    "TYPED_GGUF_TEST_BLOCK_NET forbids AF_INET, the family a loopback probe binds"))
def test_the_probe_accepts_a_real_loopback_server_and_the_inline_fallback(
        tmp_path: pathlib.Path) -> None:
    module = load_tool()
    log = Logged(tmp_path / "serve.log")

    def decide(payload: dict[str, Any]) -> dict[str, Any]:
        log.warning()                       # the CLI's warning lives on the same stream (2>&1)
        return json.loads(json.dumps(INLINE))

    app = serve.App(decide=decide, home=tmp_path / "home", log=log)
    server, thread = _serve_in_thread(app)
    try:
        port = int(server.server_address[1])
        code = module.main(["--base-url", f"http://127.0.0.1:{port}", "--model", ALIAS,
                            "--server-log", str(log.path), "--expect", "inline",
                            "--ready-budget", "30"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)
    assert code == 0, log.path.read_text(encoding="utf-8")


def test_the_probe_refuses_a_base_url_that_never_answers(tmp_path: pathlib.Path) -> None:
    """A readiness poll, not a fixed sleep: a server that never comes up is a named failure."""
    module = load_tool()
    code = module.main(["--base-url", "http://127.0.0.1:1", "--model", ALIAS, "--expect", "host",
                        "--server-log", str(tmp_path / "absent.log"), "--ready-budget", "1"])
    assert code == 1


def test_the_probe_writes_its_verdict_as_json(tmp_path: pathlib.Path) -> None:
    module = load_tool()
    out = tmp_path / "probe.json"
    log = tmp_path / "serve.log"
    log.write_text(HOST_LOG, encoding="utf-8")
    code = module.main(["--base-url", "http://127.0.0.1:1", "--model", ALIAS, "--expect", "host",
                        "--server-log", str(log), "--ready-budget", "1", "--json", str(out)])
    assert code == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["expect"] == "host" and payload["ok"] is False
    assert payload["problems"], "a refused probe records why"


def test_the_probe_sends_the_questions_the_engine_expects() -> None:
    """The native payload is the §2.5 one: state, model, questions — nothing invented."""
    module = load_tool()
    payload = module.native_payload(ALIAS)
    assert set(payload) == {"state", "model", "questions"}, sorted(payload)
    assert payload["model"] == ALIAS and payload["state"].strip()
    assert set(payload["questions"]) == set(QUESTIONS)
    for qid, question in payload["questions"].items():
        assert question["type"] == QUESTIONS[qid]["type"]
        assert question["criteria"] == QUESTIONS[qid]["criteria"], qid
        assert question["instructions"] == QUESTIONS[qid]["instructions"], qid
