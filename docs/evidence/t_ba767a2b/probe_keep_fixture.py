"""M1 mechanism probe (card t_ba767a2b) — the fixture's shape, with the luck taken out.

`tests/test_keep_client.py`'s `make_client` teardown did, in this order:

    for client in clients:                    # the list is dropped when the fixture generator goes
        client.stop(grace=1.0)
        for _pid, proc in list(client.children.items()):
            proc.kill()                       # <- no wait(): the handle is dropped right after

A `Popen` handle dropped with `returncode is None` emits `ResourceWarning: subprocess N is still
running` whenever the GC reaches it *before the SIGKILL has landed*. Whether it does is a race, so
on this box it is 0/17 four-way-concurrent runs of the whole committed sub-gate (the reviewer's box
saw 2/32). This probe removes the luck and keeps the shape: the holder is cleared *immediately*
after the kill loop — as the fixture's `clients` list is — and the collection happens in that same
breath, which is the window the fixture left open.

`_WAIT` is the one-line fix under test: `kill()` then `wait(timeout=5.0)`.

    # the CI's error: 2 passed, 1 error -- PytestUnraisableExceptionWarning: Exception ignored
    # in: <function Popen.__del__ ...>   (output: m1-probe-red.txt)
    .venv/bin/python -m pytest -q docs/evidence/t_ba767a2b/probe_keep_fixture.py

    # green with `_WAIT = True`            (output: m1-probe-green.txt)
    sed 's/^_WAIT = False/_WAIT = True/' docs/evidence/t_ba767a2b/probe_keep_fixture.py \
        > /tmp/probe_fixed.py && .venv/bin/python -m pytest -q /tmp/probe_fixed.py

Both runs above carry the repo's own `filterwarnings = ["error"]` plus pytest's unraisable hook,
which is what turns the warning into the CI's `1 error`.
"""
from __future__ import annotations

import gc
import subprocess
import sys

import pytest

_WAIT = False          # the fixture's line, before the fix: kill() with no wait()


class _Holder:
    """The fixture's `clients` list: the only strong reference to the spawned children."""

    def __init__(self) -> None:
        self.procs: list[subprocess.Popen] = []


HELD = _Holder()


@pytest.fixture
def child():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    HELD.procs.append(proc)
    pid = str(proc.pid)
    del proc                       # the handle lives in the holder only, as it does in `children`
    yield pid
    for held in list(HELD.procs):
        held.kill()
        if _WAIT:
            held.wait(timeout=5.0)
    HELD.procs.clear()             # the `clients` list going away...
    gc.collect()                   # ...and the collection that turns `returncode is None` into it


def test_the_child_ran(child) -> None:
    assert isinstance(child, str)


def test_a_later_phase_like_the_ci_s() -> None:
    assert True
