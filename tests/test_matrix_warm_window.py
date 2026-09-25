"""`tools/matrix_warm_window.py` — the keep-alive window, proven end to end (t_f96fed7f, AC4).

SPEC 2.12's window is the mechanism behind the 600 s default: the host that a call leaves behind
exits by itself when `keep_alive` lapses, so a long-lived model does not pin a device forever. The
Linux job proves it with a **short** window (the card: 15-30 s, "never burn 10 minutes of runner"):

1. `ask … --keep-alive 20` -> the host spawns and `keep status` shows it, with *this call's* window
   (the window belongs to the call that pays the host);
2. a second `ask` inside the window -> the model is already resident: `timings.model_load_ms == 0.0`
   and `engine.keep.served_by == "host"`, the *same* pid (a warm answer, not a respawn);
3. sleep past the window -> `keep status` says stopped, the pid is gone, the socket is unlinked.

Nothing calls `keep stop` in between: "the host exits by itself" is the claim, and a `keep stop`
would make the third leg a tautology. The tool owns the whole drive (`ask`, `keep status`, the
sleep, the liveness readings) and `judge` is a pure function over what it collected, so each RED
direction — a second call that paid a load, a window the host ignored, a socket left behind — is a
gate the suite drives rather than prose in a workflow comment.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import shlex
import subprocess
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "matrix_warm_window.py"
MODEL = "/tmp/smoke.gguf"
CLI = "uv run typed-gguf"
WINDOW = 20.0
PID = 4242
SOCKET = "/tmp/warm-window/keep/host.sock"


def load_tool():
    if not TOOL.exists():
        raise AssertionError(
            f"{TOOL.relative_to(ROOT)} is missing: the Linux job's warm-window step runs it, and "
            "without it SPEC 2.12's idle-unload has no CI proof (card t_f96fed7f)")
    spec = importlib.util.spec_from_file_location("matrix_warm_window", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def keeper(served_by: str = "host", *, load_ms: float = 0.0, pid: int = PID,
           keep_alive: float = WINDOW) -> dict[str, Any]:
    """The `engine.keep` block a decision carries (SPEC 2.12)."""
    return {"served_by": served_by, "pid": pid, "keep_alive_s": keep_alive, "requests": 1,
            "model_load_ms": 1200.0, "idle_left_s": keep_alive}


def decision(*, served_by: str = "host", load_ms: float = 0.0, pid: int = PID) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "smoke-0.5b",
        "answers": {"area": {"type": "choice", "choice": "billing"}},
        "usage": {"input_tokens": 120, "output_tokens": 3},
        "timings": {"model_load_ms": load_ms, "prefill_ms": 300.0},
    }
    if served_by is not None:
        body["engine"] = {"keep": keeper(served_by, load_ms=1200.0, pid=pid)}
    return body


def status(state: str = "running", *, pid: int = PID, keep_alive: float = WINDOW,
           socket: str = SOCKET) -> dict[str, Any]:
    if state == "stopped":
        return {"state": "stopped", "pid": None, "keep_alive_s": None, "socket": None}
    return {"state": state, "pid": pid, "socket": socket, "keep_alive_s": keep_alive,
            "model": "smoke-0.5b", "model_path": MODEL, "idle_left_s": keep_alive,
            "requests": 2, "key_digest": "abc"}


def window(module: Any = None, **overrides: Any) -> Any:
    module = module or load_tool()
    fields: dict[str, Any] = {
        "keep_alive": WINDOW,
        "first": decision(load_ms=1200.0),
        "second": decision(),
        "before": status(),
        "after": status("stopped"),
        "pid_alive": False,
        "socket_present": False,
        "waited_s": WINDOW + 10.0,
    }
    fields.update(overrides)
    return module.Window(**fields)


# ------------------------------------------------------------------- the happy path + the shape
def test_a_window_that_opens_and_closes_by_itself_is_the_claim() -> None:
    module = load_tool()
    assert module.judge(window(module)) == []


def test_the_step_asks_with_the_short_window_and_reads_the_answer_from_a_file() -> None:
    """Pins the CLI shape: `ask` has no `--json` (its JSON goes to `--out`), and the window is
    passed on the call (`--keep-alive`), because SPEC 2.12 ties it to the call that spawns."""
    module = load_tool()
    argv = module.ask_argv(shlex.split(CLI), MODEL, WINDOW, pathlib.Path("/tmp/first.json"))
    assert argv[:2] == ["uv", "run"], argv
    assert "--json" not in argv, "--json is `keep status`'s flag; `ask` writes JSON to --out"
    assert argv[-2:] == ["--out", "/tmp/first.json"], argv
    assert "--keep-alive" in argv and argv[argv.index("--keep-alive") + 1] == "20"
    assert argv[argv.index("--model") + 1] == MODEL
    assert argv[argv.index("--state") + 1].strip()
    assert any(part.startswith("area=") for part in argv), "one choice question"
    assert any(part.startswith("urgency=") for part in argv), "one score question"


# ------------------------------------------------------------------- the RED directions
def test_a_second_call_that_paid_a_load_is_refused() -> None:
    module = load_tool()
    problems = module.judge(window(module, second=decision(load_ms=640.0)))
    assert problems and any("model_load_ms" in problem for problem in problems), problems


def test_a_second_call_answered_by_another_host_is_refused() -> None:
    """A respawn means the host did *not* stay warm: the pid is the proof."""
    module = load_tool()
    problems = module.judge(window(module, second=decision(pid=PID + 1)))
    assert problems and any("pid" in problem for problem in problems), problems


def test_a_first_call_that_never_spawned_a_host_is_refused() -> None:
    module = load_tool()
    problems = module.judge(window(module, first=decision(served_by="inline")))
    assert problems and any("host" in problem for problem in problems), problems


def test_a_before_status_that_does_not_show_the_host_is_refused() -> None:
    module = load_tool()
    problems = module.judge(window(module, before=status("stopped")))
    assert problems and any("running" in problem for problem in problems), problems


def test_a_host_holding_another_window_than_the_call_asked_for_is_refused() -> None:
    """SPEC 2.12: the window travels with the request that pays for the load."""
    module = load_tool()
    problems = module.judge(window(module, before=status(keep_alive=600.0)))
    assert problems and any("keep_alive" in problem for problem in problems), problems


def test_a_host_still_resident_after_the_window_is_refused() -> None:
    module = load_tool()
    problems = module.judge(window(module, after=status()))
    assert problems and any("still" in problem for problem in problems), problems


def test_a_pid_that_outlived_the_window_is_refused() -> None:
    module = load_tool()
    problems = module.judge(window(module, pid_alive=True))
    assert problems and any("pid" in problem for problem in problems), problems


def test_a_socket_left_behind_is_refused() -> None:
    module = load_tool()
    problems = module.judge(window(module, socket_present=True))
    assert problems and any("socket" in problem for problem in problems), problems


def test_a_missing_decision_is_refused() -> None:
    module = load_tool()
    problems = module.judge(window(module, first=None))
    assert problems, "with no first decision there is nothing to judge"


# --------------------------------------------------------------------------- the drive itself
class Runner:
    """A scripted `typed-gguf`: it answers `ask` (writing `--out`) and `keep status --json`.

    The states it walks are the product's own sequence — cold host, warm host, gone — so the test
    covers the *drive* (order, the sleep, the readings) without a model or a runtime.
    """

    def __init__(self, tmp_path: pathlib.Path) -> None:
        self.tmp_path = tmp_path
        self.asks = 0
        self.calls: list[list[str]] = []
        self.states = iter([status(), status("stopped")])

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        if "ask" in argv:
            self.asks += 1
            out = pathlib.Path(argv[argv.index("--out") + 1])
            out.write_text(json.dumps(decision(load_ms=1200.0 if self.asks == 1 else 0.0)),
                           encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, "", "")
        if "status" in argv:
            return subprocess.CompletedProcess(argv, 0, json.dumps(next(self.states)), "")
        raise AssertionError(f"unexpected argv: {argv}")


def test_the_drive_asks_twice_reads_status_twice_and_waits_out_the_window(
        tmp_path: pathlib.Path) -> None:
    module = load_tool()
    runner = Runner(tmp_path)
    slept: list[float] = []
    facts = module.drive(shlex.split(CLI), MODEL, WINDOW, runner=runner,
                         workdir=tmp_path, sleep=slept.append, margin=10.0,
                         pid_alive_fn=lambda pid: False, socket_fn=lambda path: False)
    assert runner.asks == 2, "one call to spawn, one inside the window"
    verbs = ["ask" if "ask" in call else "keep" for call in runner.calls]
    assert verbs == ["ask", "keep", "ask", "keep"], runner.calls
    assert slept == [WINDOW + 10.0], "the wait is `keep_alive` plus the margin, once"
    assert module.judge(facts) == []
    assert facts.waited_s == WINDOW + 10.0, "the drive records how long it waited"
