#!/usr/bin/env python3
"""Probe a running `typed-gguf serve` and judge *who answered* (card t_f96fed7f, AC2/AC3).

Two matrix jobs grow a serve story in this card, and they make opposite claims about the same
product:

**macOS (AC3)** — the warm host exists, so the decision *after* the first one must be answered by
it: `engine.keep.served_by == "host"` and `timings.model_load_ms == 0.0` (SPEC 2.12: "a warm answer
reports `0.0`"), with the server's own log line saying `served_by=host`. `engine.keep` is a
deliberately *local* fact — the TypeSafe projection drops it and `serve`'s own log line is where it
survives — so the log is part of the assertion, not decoration.

**Windows (AC2)** — `keep.supported()` is false for `win*` (`typed_gguf.keep.identity`), so every
decision answers **inline on that same call** and says why: the `W_KEEP_UNAVAILABLE` warning on the
server's own output (SPEC 2.12, "Platforms"), and a served body with **no** `engine.keep` block —
the unsupported-platform branch returns `decide_payload`'s pre-E4 body untouched
(`cli.decide_payload_warm`). A Windows run that *claims* a host is the failure this mode exists for.

The tool is the readiness poll as well (no fixed sleeps: `/health` decides) and the judge. `judge`
is a pure function over collected facts, so every RED direction — a Windows run that claims a host,
a macOS second call that paid a load, a missing warning — is a gate the suite drives.

    tools/matrix_serve_probe.py --base-url http://127.0.0.1:8088 --model smoke-0.5b \
        --server-log /tmp/serve.log --expect host --json /tmp/serve_probe.json

Exit 0 = the platform's claim holds. 1 = it does not (each problem on stderr, and in `--json`).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

#: The native wire the probe speaks (`/v1/decide`, SPEC 2.5) — no SDK, no projection.
DECIDE_PATH = "/v1/decide"
HEALTH_PATH = "/health"
#: The two claims this tool can judge. `host` = the warm host answered the later decision;
#: `inline` = this platform has no unix sockets, so it answers inline and names the reason.
EXPECTATIONS = ("host", "inline")
#: The named warning SPEC 2.12 pins for a platform without `AF_UNIX` (`cli.decide_payload_warm`).
WARNING = "W_KEEP_UNAVAILABLE"
#: The state the probe sends: a real decision, short enough for a 0.5B smoke model's context.
STATE = ("The nightly export job has failed for three days; the finance team is reconciling "
         "invoices by hand and the account has asked twice for an explanation. Support has one "
         "open ticket, no owner, and the last release touched the export path.")
#: The questions: one choice and one score, built the way the SDK/native wire spells them.
QUESTIONS: dict[str, Any] = {
    "area": {"type": "choice", "instructions": "Which area is this about?",
             "criteria": {"billing": "Payments, invoicing, refunds",
                          "technical": "API, uptime, errors"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["Can wait", "This week", "Today"]},
}
#: How long the readiness poll waits for `/health` before the server is called broken.
DEFAULT_READY_BUDGET = 120.0
#: One decision can be a cold model load: generous, and only a *client-side* ceiling (the server
#: has its own). The SDK's own 10 s default is documented as too short for a cold box (SPEC 2.9).
DEFAULT_TIMEOUT = 300.0


class ProbeError(RuntimeError):
    """The probe could not get a fact — the server never came up, or a request failed."""


@dataclasses.dataclass(frozen=True)
class Facts:
    """What the probe collected: the health body, the decisions, and the server's own stream."""

    base_url: str
    health: dict[str, Any]
    decisions: list[dict[str, Any]]
    log: str

    @property
    def last(self) -> dict[str, Any] | None:
        return self.decisions[-1] if self.decisions else None


def native_payload(model: str) -> dict[str, Any]:
    """The §2.5 native request: `state`, `model`, `questions` — nothing invented."""
    return {"state": STATE, "model": model, "questions": json.loads(json.dumps(QUESTIONS))}


def http_opener(base_url: str, timeout: float) -> Callable[..., tuple[int, Any]]:
    """`(method, path, body) -> (status, parsed JSON)`, over the real wire."""
    root = base_url.rstrip("/")

    def opener(method: str, path: str, body: bytes | None = None) -> tuple[int, Any]:
        request = urllib.request.Request(                      # noqa: S310 - an explicit http url
            root + path, data=body, method=method,
            headers={"Content-Type": "application/json"} if body is not None else {})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                raw = response.read()
                return int(response.status), json.loads(raw.decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                return int(exc.code), json.loads(raw.decode("utf-8") or "{}")
            except (UnicodeDecodeError, json.JSONDecodeError):
                return int(exc.code), {"detail": raw.decode("utf-8", "replace")}

    return opener


def wait_ready(base_url: str, budget: float, opener: Callable[..., Any], *,
               poll: float = 0.5, sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Poll `/health` until the server answers `status: ok`; no fixed sleep anywhere.

    Every failure — a refused connection, a 5xx, a body that is not the health object — is the
    "not yet" answer: the readiness question is the server's own, and the last one travels in the
    error so a never-ready server reads as a reason instead of a timeout.
    """
    deadline = time.monotonic() + float(budget)
    last = "never tried"
    while True:
        try:
            status, payload = opener("GET", HEALTH_PATH, None)
            if status == 200 and isinstance(payload, dict) and payload.get("status") == "ok":
                return payload
            last = f"HTTP {status}: {json.dumps(payload)[:200]}"
        except Exception as exc:                        # noqa: BLE001 - any failure is "not ready"
            last = f"{exc.__class__.__name__}: {exc}"
        if time.monotonic() >= deadline:
            raise ProbeError(
                f"{base_url}{HEALTH_PATH} never answered `status: ok` within {budget:g}s "
                f"(last: {last}) — `typed-gguf serve` is not up, so nothing was measured")
        sleep(poll)


def read_logs(paths: list[pathlib.Path]) -> str:
    """The server's own stream, as the workflow redirects it (`> serve.log 2>&1`)."""
    chunks: list[str] = []
    for path in paths:
        if path.exists():
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def describe(decision: dict[str, Any]) -> str:
    """One line of receipt per decision: who answered it, and what it cost."""
    engine = decision.get("engine") if isinstance(decision.get("engine"), dict) else {}
    keep = engine.get("keep") if isinstance(engine.get("keep"), dict) else {}
    timings = decision.get("timings") if isinstance(decision.get("timings"), dict) else {}
    return (f"served_by={keep.get('served_by') or '-'} "
            f"model_load_ms={timings.get('model_load_ms')!r} "
            f"answers={len(decision.get('answers') or {})}")


def judge(facts: Facts, *, expect: str) -> list[str]:
    """Every problem with what the server did, in the order they are found. `[]` = the claim holds."""
    if expect not in EXPECTATIONS:
        return [f"`--expect` must be one of {', '.join(EXPECTATIONS)}, got {expect!r}"]
    problems: list[str] = []
    log = facts.log or ""
    if not log.strip():
        problems.append(
            "the server's own stream is empty: `engine.keep` is a local fact (the TypeSafe "
            "projection drops it), so `served_by=` on that stream is where who-answered survives "
            "(SPEC 2.12) — pass --server-log for the file the job redirects serve into")
    last = facts.last
    if last is None:
        return [*problems, "no decision reached the server: nothing was measured"]
    answers = last.get("answers")
    if not isinstance(answers, dict) or not answers:
        problems.append("the served body carries no `answers`: the server must answer, not just "
                        "return 200")
    keep = last.get("engine", {}).get("keep") if isinstance(last.get("engine"), dict) else None
    keep = keep if isinstance(keep, dict) else {}
    if expect == "host":
        if not keep:
            problems.append(
                "the last decision carries no `engine.keep`: on this platform a warm host must "
                "have answered it (SPEC 2.12) — the block is what says who did")
        elif keep.get("served_by") != "host":
            problems.append(
                f"the last decision's engine.keep.served_by is {keep.get('served_by')!r}, not "
                f"'host': no host answered it")
        load = (last.get("timings") or {}).get("model_load_ms")
        if load != 0.0:
            problems.append(
                f"the last decision reports timings.model_load_ms={load!r}: a warm answer reports "
                f"0.0 (SPEC 2.12, A-E4-1) — the model was loaded again instead of reused")
        if "served_by=host" not in log:
            problems.append(
                "the server's own stream never says `served_by=host`: the warm decision did not go "
                "through the resident host (SPEC 2.12)")
        if WARNING in log:
            problems.append(
                f"the server printed {WARNING} on a platform that must have a warm host: the "
                f"keep path fell back where it should have run (SPEC 2.12)")
    else:
        if keep:
            problems.append(
                f"the decision carries engine.keep={keep!r}: keep.supported() is false for win* "
                f"(typed_gguf.keep.identity), and SPEC 2.12's unsupported-platform branch answers "
                f"the pre-E4 body — no host block, no daemon attempted")
        if WARNING not in log:
            problems.append(
                f"the server never named {WARNING}: SPEC 2.12 pins that warning as the reason an "
                f"inline answer was taken, and a silent fallback is not the documented behaviour")
        if "served_by=host" in log:
            problems.append(
                "the server's own stream says `served_by=host`, which this platform cannot have "
                "(no AF_UNIX): the platform rule in typed_gguf.keep.identity broke")
    return problems


def probe(base_url: str, model: str, expect: str, *, opener: Any = None, timeout: float,
          ready_budget: float, logs: list[pathlib.Path], sleep: Any = time.sleep) -> Facts:
    """Poll for readiness, send the decisions this claim needs, read the server's own stream."""
    opener = opener or http_opener(base_url, timeout)
    health = wait_ready(base_url, ready_budget, opener, sleep=sleep)
    payload = json.dumps(native_payload(model)).encode("utf-8")
    calls = 2 if expect == "host" else 1
    decisions: list[dict[str, Any]] = []
    for index in range(calls):
        status, body = opener("POST", DECIDE_PATH, payload)
        if status != 200:
            raise ProbeError(f"{DECIDE_PATH} answered HTTP {status}: {json.dumps(body)[:300]}")
        if not isinstance(body, dict):
            raise ProbeError(f"{DECIDE_PATH} answered a non-object body: {type(body).__name__}")
        decisions.append(body)
        print(f"  decision {index + 1}: {describe(body)}", flush=True)
    return Facts(base_url=base_url, health=health, decisions=decisions, log=read_logs(logs))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assert that `typed-gguf serve` answered through the warm host (host) or "
                    "inline with the named warning (inline) — card t_f96fed7f.")
    parser.add_argument("--base-url", required=True, help="the served root, e.g. http://127.0.0.1:8088")
    parser.add_argument("--model", required=True,
                        help="the registry alias the decision names (the served routes resolve "
                             "aliases, never paths)")
    parser.add_argument("--expect", required=True, choices=EXPECTATIONS)
    parser.add_argument("--server-log", action="append", default=[],
                        help="the file the job redirects serve's stdout/stderr into (repeatable)")
    parser.add_argument("--ready-budget", type=float, default=DEFAULT_READY_BUDGET)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--json", default=None, help="write the verdict here")
    opts = parser.parse_args(argv)

    logs = [pathlib.Path(path) for path in opts.server_log]
    problems: list[str] = []
    facts: Facts | None = None
    try:
        facts = probe(opts.base_url, opts.model, opts.expect, timeout=opts.timeout,
                      ready_budget=opts.ready_budget, logs=logs)
        problems = judge(facts, expect=opts.expect)
    except ProbeError as exc:
        problems = [str(exc)]

    if facts is not None:
        print(f"health: {json.dumps(facts.health)}", flush=True)
    for problem in problems:
        print(f"FAIL {problem}", file=sys.stderr)
    verdict = {
        "schema": "typed_gguf.matrix.serve_probe/v1",
        "base_url": opts.base_url,
        "model": opts.model,
        "expect": opts.expect,
        "health": facts.health if facts else None,
        "decisions": [describe(decision) for decision in (facts.decisions if facts else [])],
        "problems": problems,
        "ok": not problems,
    }
    if opts.json:
        pathlib.Path(opts.json).write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    if problems:
        print(f"serve probe RED ({opts.expect}): {len(problems)} problem(s)", file=sys.stderr)
        return 1
    print(f"serve probe OK ({opts.expect}): the served decisions are this platform's documented "
          f"behaviour (SPEC 2.12)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
