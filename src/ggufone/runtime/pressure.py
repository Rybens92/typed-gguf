"""Pid-cgroup pressure: name the box, never mislabel it (card t_a696ce02)

The worker container runs under a small, *shared* pid cgroup (`pids.max = 256` here, with the
sibling cards' campaigns inside it). Every ggufone probe is a child process by design
(`ggufone.runtime.isolated`: a C library must never be able to kill the command), so the probe
path is exactly the thing that dies first when the box is at the cap: `fork`/`posix_spawn`
answers `EAGAIN` and `subprocess.run` raises `BlockingIOError: [Errno 11] Resource temporarily
unavailable`.

Measured (card t_a696ce02, `tests/test_runtime_{fallback,install,contract}.py` under
pytest-randomly): green while `pids.current` is under ~200, red once it reaches 244-256 —
`['probe_failed', 'no_asset'] == ['loader_error', 'no_asset']`, i.e. the install probe's
*reason string* turning into another string. The tests were right; the box was answering.

Two things live here, because they are one concern:

* `read_pid_headroom` / `PidHeadroom` — the live reading, so a failure can name the box
  (`pids.current=254/256 (2 free)`) instead of blaming a bundle;
* `spawn` / `SpawnBlocked` — `subprocess.run` with a *bounded* retry budget for the transient
  spawn errnos, and a named `E_PID_PRESSURE` error when the budget is out.

The tests' own gate over this reading (skip the fork-dependent gates, and refuse to look green
while they are skipped) lives in `tests/conftest.py`.
"""
from __future__ import annotations

import errno
import os
import pathlib
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

#: cgroup v2 files (the sandbox mounts them read-only, but readable: that is all this needs).
PID_CGROUP_ROOT = pathlib.Path("/sys/fs/cgroup")
PIDS_CURRENT = "pids.current"
PIDS_MAX = "pids.max"

#: pids a fork-dependent gate needs free before it can trust itself. Below this the suite skips
#: the gates that spawn a real child (and says so loudly) instead of reporting the box's
#: exhaustion as a product failure.
PID_HEADROOM_FLOOR = 16

#: Machine-readable name for "could not fork, the box is out of pids" — the one string that
#: separates "this host cannot load the backend" (`loader_error`) from "this host cannot fork
#: right now". Callers key on it; it is never a loader story.
E_PID_PRESSURE = "E_PID_PRESSURE"

#: Spawn retries: 1 try + 3 retries, 0.2 s + 0.4 s + 0.8 s of waiting (1.4 s worst case).
#: Enough to ride out the dips a shared box shows; a *sustained* cap is named, not waited on.
SPAWN_ATTEMPTS = 4
SPAWN_BACKOFF = 0.2

#: Errnos that mean "the box, not the command": a momentarily full pid/fd table, or a signal
#: interrupting the spawn. Anything else (ENOENT, EACCES, ENOEXEC) is the command's own fault
#: and is re-raised untouched.
SPAWN_ERRNOS = frozenset({errno.EAGAIN, errno.EINTR, errno.ENOMEM, errno.EMFILE, errno.ENFILE})


@dataclass(frozen=True)
class PidHeadroom:
    """One reading of the pid cgroup: how many pids are live, and what the cap is."""

    current: int
    maximum: int | None = None      # None: `pids.max` is "max" — no pid limit (a plain host)

    @property
    def free(self) -> int | None:
        """Pids left under the cap, or `None` when the cgroup has no cap."""
        if self.maximum is None:
            return None
        return max(self.maximum - self.current, 0)

    def starved(self, floor: int = PID_HEADROOM_FLOOR) -> bool:
        """Is there less than `floor` pids of headroom? (A cgroup with no cap never starves.)"""
        free = self.free
        return free is not None and free < floor

    def describe(self) -> str:
        """`pids.current=254/256 (2 free)` — the phrase every pressure failure carries."""
        if self.maximum is None:
            return f"pids.current={self.current} (no pid limit)"
        return f"pids.current={self.current}/{self.maximum} ({self.free} free)"


def _read_int(path: pathlib.Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    if text == "max":                    # `pids.max` may be unlimited
        return None
    try:
        return int(text)
    except ValueError:
        return None


def read_pid_headroom(root: str | os.PathLike[str] = PID_CGROUP_ROOT) -> PidHeadroom | None:
    """The live pid-cgroup reading, or `None` when this host has no readable pid cgroup.

    `None` is a real answer (a plain host, or a cgroup v1 box): callers then keep the generic
    message instead of inventing numbers. `pids.max == "max"` is reported as `maximum=None`.
    """
    root = pathlib.Path(root)
    current = _read_int(root / PIDS_CURRENT)
    if current is None:
        return None
    maximum = _read_int(root / PIDS_MAX)
    return PidHeadroom(current=current, maximum=maximum)


def is_spawn_pressure(exc: BaseException) -> bool:
    """Is this OSError the box saying "no pids" rather than the command saying "no"?"""
    return isinstance(exc, OSError) and exc.errno in SPAWN_ERRNOS


def pressure_note(root: str | os.PathLike[str] = PID_CGROUP_ROOT) -> str:
    """`; pid cgroup: pids.current=254/256 (2 free)` — or `''` when there is no cgroup."""
    headroom = read_pid_headroom(root)
    return f"; pid cgroup: {headroom.describe()}" if headroom is not None else ""


class SpawnBlocked(OSError):
    """A child could not be started: the box is out of fork headroom (`E_PID_PRESSURE`).

    An `OSError` subclass on purpose: every existing `except OSError` around a spawn keeps
    working, while the message now names the box instead of the command. `.errno` is the
    kernel's own answer (`EAGAIN` here), so a caller can still branch on it.
    """


def spawn(command: Sequence[str], *, timeout: float | None = None,
          **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """`subprocess.run`, but a momentarily exhausted pid table is retried and named.

    Retries only `SPAWN_ERRNOS` (a transient EAGAIN/ENOMEM/EINTR from the kernel), never a
    non-zero exit or a timeout: a child that ran and answered is *data* about the bundle, a
    child the kernel refused to create says nothing about it. After `SPAWN_ATTEMPTS` the last
    errno is re-raised as `SpawnBlocked` carrying `E_PID_PRESSURE` and the live cgroup reading.
    """
    argv = [str(part) for part in command]
    last: OSError | None = None
    for attempt in range(1, SPAWN_ATTEMPTS + 1):
        try:
            return subprocess.run(argv, timeout=timeout, **kwargs)   # noqa: S603
        except subprocess.TimeoutExpired:
            raise
        except OSError as exc:
            if not is_spawn_pressure(exc):
                raise
            last = exc
            if attempt < SPAWN_ATTEMPTS:
                time.sleep(SPAWN_BACKOFF * 2 ** (attempt - 1))
    assert last is not None
    raise SpawnBlocked(
        last.errno,
        f"{E_PID_PRESSURE}: could not start {argv[0]!r} after {SPAWN_ATTEMPTS} attempts "
        f"({' '.join(argv)}): {last.__class__.__name__}: {last}{pressure_note()} — the box has "
        f"no fork headroom right now (a shared pid cgroup at its cap, not a bundle problem); "
        f"retry when pids.current is lower, or give the container more pids"
    ) from last
