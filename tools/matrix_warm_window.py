#!/usr/bin/env python3
"""The keep-alive window, driven and judged end to end (card t_f96fed7f, AC4).

SPEC 2.12's window is the mechanism behind the 600 s default: the host a call leaves behind exits
by itself when `keep_alive` lapses, so a long-lived model cannot pin a device forever. This tool
proves it on the Linux job with a **short** window (the card's 15-30 s: "never burn 10 minutes of
runner"), in one step:

1. `ask … --keep-alive 20` -> the host spawns; `keep status` shows it, holding *this call's* window
   (SPEC 2.12: "the window belongs to the call that starts the host, so it travels with the request
   that pays for it");
2. a second `ask` inside the window -> the model is already resident: `timings.model_load_ms == 0.0`
   and `engine.keep.served_by == "host"`, with the **same pid** (a warm answer, not a respawn);
3. sleep past the window -> `keep status` says stopped, the pid is gone, the socket is unlinked.

**Nothing calls `keep stop` in between.** "The host exits by itself" is the claim; a `keep stop`
would make the third leg a tautology, which is why the step that runs this tool must not call it
(the teardown leg of the macOS serve job is a different step, and it runs after its own `keep stop`
on purpose).

    tools/matrix_warm_window.py --cli "uv run typed-gguf" --model /tmp/smoke.gguf \
        --keep-alive 20 --json /tmp/warm_window.json

Exit 0 = the window opened, stayed warm, and closed by itself. 1 = it did not, with each problem on
stderr and in `--json`.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

from typed_gguf.keep import state as keep_state

#: A state real enough that the decision is about something (SPEC 2.8 needs a non-empty one).
STATE = ("The nightly export job has failed for three days; finance reconciles invoices by hand "
         "and the account has asked twice for an explanation. One open ticket, no owner.")
#: One choice + one score: `id=instructions:labels` (SPEC 2.8).
CHOICE = "area=Which area is this about?:billing|technical"
SCORE = "urgency=How urgent is this?:Can wait|This week|Today"
#: The margin the drive waits *past* the window before asking the ledger again. The host counts from
#: its last request, so the wait happens after the second call; 10 s is a scheduling-jitter budget,
#: not a second window.
DEFAULT_MARGIN = 10.0
#: What the tool was asked to wait for, so the message can name it without re-deriving it.
STATUS_VERB = ("keep", "status", "--json")


@dataclasses.dataclass(frozen=True)
class Window:
    """Everything the drive collected: the two decisions, the two ledgers, and the liveness."""

    keep_alive: float
    first: dict[str, Any] | None
    second: dict[str, Any] | None
    before: dict[str, Any] | None
    after: dict[str, Any] | None
    pid_alive: bool
    socket_present: bool
    waited_s: float

    @property
    def pid(self) -> int | None:
        return _keep_pid(self.first)


def ask_argv(prefix: Sequence[str], model: str, keep_alive: float,
             out: pathlib.Path) -> list[str]:
    """`ask` with the short window. `--json` is *not* an `ask` flag (its JSON goes to `--out`)."""
    return [*prefix, "ask", "--state", STATE, "--model", model, "--choice", CHOICE,
            "--score", SCORE, "--keep-alive", f"{keep_alive:g}", "--out", str(out)]


def keep_pid(decision: dict[str, Any] | None) -> int | None:
    """`engine.keep.pid`: which process answered (SPEC 2.12)."""
    return _keep_pid(decision)


def _keep_pid(decision: dict[str, Any] | None) -> int | None:
    if not isinstance(decision, dict):
        return None
    engine = decision.get("engine") if isinstance(decision.get("engine"), dict) else {}
    keep = engine.get("keep") if isinstance(engine.get("keep"), dict) else {}
    pid = keep.get("pid")
    return int(pid) if isinstance(pid, int) else None


def served_by(decision: dict[str, Any] | None) -> str | None:
    """Who answered a decision: `"host"`, `"inline"`, or `None` (no keep block at all)."""
    if not isinstance(decision, dict):
        return None
    engine = decision.get("engine") if isinstance(decision.get("engine"), dict) else {}
    keep = engine.get("keep") if isinstance(engine.get("keep"), dict) else {}
    value = keep.get("served_by")
    return str(value) if value else None


def judge(window: Window) -> list[str]:
    """Every problem with the window story, in the order they are found. `[]` = it held."""
    problems: list[str] = []
    pid = window.pid
    if window.first is None:
        problems.append("the first `ask` produced no decision — nothing was measured")
    elif served_by(window.first) != "host":
        problems.append(
            f"the first `ask` was served_by={served_by(window.first)!r}, not 'host': with "
            f"--keep-alive {window.keep_alive:g} the call must leave a warm host behind "
            f"(SPEC 2.12)")
    if window.second is None:
        problems.append("the second `ask` produced no decision — the warm leg was not measured")
    else:
        if served_by(window.second) != "host":
            problems.append(
                f"the second `ask` was served_by={served_by(window.second)!r}, not 'host': "
                f"the host the first call left behind did not answer it")
        load = (window.second.get("timings") or {}).get("model_load_ms")
        if load != 0.0:
            problems.append(
                f"the second call reports timings.model_load_ms={load!r}: it paid a model load, so "
                f"the host was not warm (SPEC 2.12: a warm answer reports 0.0)")
        other = keep_pid(window.second)
        if pid is not None and other is not None and pid != other:
            problems.append(
                f"the second call was answered by pid {other}, not by the host the first call left "
                f"behind (pid {pid}): the host did not stay resident (a swap or a respawn)")
    if window.before is None:
        problems.append("`keep status` produced no report right after the first call")
    else:
        state = window.before.get("state")
        if state != "running":
            problems.append(
                f"`keep status` right after the first call says state={state!r}, not 'running': no "
                f"warm host was left behind (SPEC 2.12)")
        held = window.before.get("keep_alive_s")
        if not isinstance(held, (int, float)) or abs(float(held) - window.keep_alive) > 0.5:
            problems.append(
                f"`keep status` reports keep_alive_s={held!r} but the call asked for "
                f"{window.keep_alive:g}s: the window belongs to the call that pays the host "
                f"(SPEC 2.12, flag > env > default)")
        ledger_pid = window.before.get("pid")
        if pid is not None and ledger_pid is not None and int(ledger_pid) != pid:
            problems.append(
                f"`keep status` reports pid {ledger_pid} while the decision was answered by pid "
                f"{pid}: the ledger and the answer name different hosts")
    if window.after is None:
        problems.append("`keep status` produced no report after the window lapsed")
    else:
        state = window.after.get("state")
        if state != "stopped":
            problems.append(
                f"`keep status` {window.waited_s:g}s after a {window.keep_alive:g}s window still "
                f"reports state={state!r}: the host did not exit by itself (SPEC 2.12's idle "
                f"unload) — the ledger entry is still there")
    if window.pid_alive and pid is not None:
        problems.append(
            f"pid {pid} is still alive {window.waited_s:g}s after a {window.keep_alive:g}s window: "
            f"the host process outlived its idle window (a leaked host)")
    if window.socket_present:
        problems.append(
            "the host's socket is still on disk: the host unlinks it on the way out (SPEC 2.12), "
            "so a socket with no host behind it is debris the next call has to clean up")
    return problems


def _checked(result: Any, what: str) -> None:
    code = getattr(result, "returncode", 0)
    if code:
        tail = (getattr(result, "stderr", "") or getattr(result, "stdout", "") or "").strip()
        raise RuntimeError(f"{what} exited {code}: {tail[-600:]}")


def _status(runner: Callable[[list[str]], Any], prefix: Sequence[str]) -> dict[str, Any]:
    result = runner([*prefix, *STATUS_VERB])
    _checked(result, "`keep status`")
    try:
        returned = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"`keep status --json` did not print JSON ({exc}): {result.stdout!r}") from exc
    return returned if isinstance(returned, dict) else {"state": returned}


def _read_decision(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def drive(prefix: Sequence[str], model: str, keep_alive: float, *, workdir: pathlib.Path,
          margin: float = DEFAULT_MARGIN, runner: Callable[[list[str]], Any] | None = None,
          sleep: Callable[[float], None] = time.sleep,
          pid_alive_fn: Callable[[int], bool] | None = None,
          socket_fn: Callable[[str], bool] | None = None) -> Window:
    """Ask, ask again, wait out the window, and read the ledger and the process table around it."""
    run = runner or (lambda argv: subprocess.run(argv, capture_output=True, text=True,  # noqa: S603
                                                 cwd=str(workdir)))
    first_out = pathlib.Path(workdir) / "warm_first.json"
    second_out = pathlib.Path(workdir) / "warm_second.json"
    for stale in (first_out, second_out):
        stale.unlink(missing_ok=True)
    print(f"== call 1: ask --keep-alive {keep_alive:g} (spawns the host)", flush=True)
    _checked(run(ask_argv(prefix, model, keep_alive, first_out)), "the first `ask`")
    before = _status(run, prefix)
    print(f"   keep status: {json.dumps(before)}", flush=True)
    print("== call 2: ask inside the window (must be warm)", flush=True)
    _checked(run(ask_argv(prefix, model, keep_alive, second_out)), "the second `ask`")
    first = _read_decision(first_out)
    second = _read_decision(second_out)
    pid = keep_pid(first)
    wait = float(keep_alive) + float(margin)
    print(f"== waiting {wait:g}s (window plus a {margin:g}s margin; no `keep stop`)",
          flush=True)
    sleep(wait)
    after = _status(run, prefix)
    print(f"   keep status: {json.dumps(after)}", flush=True)
    socket = (before or {}).get("socket")
    alive = bool(pid is not None and (pid_alive_fn or keep_state.pid_alive)(pid))
    present = bool(socket and (socket_fn or _socket_present)(str(socket)))
    return Window(keep_alive=float(keep_alive), first=first, second=second, before=before,
                  after=after, pid_alive=alive, socket_present=present, waited_s=wait)


def _socket_present(path: str) -> bool:
    return pathlib.Path(path).exists()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prove SPEC 2.12's keep-alive window end to end: the host spawns, a second "
                    "call inside the window is warm, and the host exits by itself afterwards "
                    "(card t_f96fed7f). This tool never calls `keep stop`.")
    parser.add_argument("--cli", default="uv run typed-gguf",
                        help="the command prefix that runs the product "
                             "(default: `uv run typed-gguf`)")
    parser.add_argument("--model", required=True, help="the model path or alias to ask about")
    parser.add_argument("--keep-alive", type=float, default=20.0,
                        help="the window, in seconds: short on purpose (15-30 s on a runner)")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                        help="how long past the window the drive waits before reading the ledger")
    parser.add_argument("--workdir", default=None, help="where the two decisions are written")
    parser.add_argument("--json", default=None, help="write the verdict here")
    opts = parser.parse_args(argv)

    prefix = shlex.split(opts.cli)
    workdir = pathlib.Path(opts.workdir or ".").resolve()
    window: Window | None = None
    problems: list[str] = []
    try:
        window = drive(prefix, opts.model, opts.keep_alive, workdir=workdir, margin=opts.margin)
        problems = judge(window)
    except (RuntimeError, OSError) as exc:
        problems = [str(exc)]

    for problem in problems:
        print(f"FAIL {problem}", file=sys.stderr)
    verdict = {
        "schema": "typed_gguf.matrix.warm_window/v1",
        "cli": opts.cli,
        "model": opts.model,
        "keep_alive_s": opts.keep_alive,
        "margin_s": opts.margin,
        "waited_s": window.waited_s if window else None,
        "first": window.first if window else None,
        "second": window.second if window else None,
        "status_before": window.before if window else None,
        "status_after": window.after if window else None,
        "pid_alive_after": window.pid_alive if window else None,
        "socket_present_after": window.socket_present if window else None,
        "problems": problems,
        "ok": not problems,
    }
    if opts.json:
        pathlib.Path(opts.json).write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    if problems:
        print(f"warm window RED: {len(problems)} problem(s)", file=sys.stderr)
        return 1
    print(f"warm window OK: the host spawned, answered the second call warm "
          f"(model_load_ms=0.0), and exited by itself after {window.keep_alive:g}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
