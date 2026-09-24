"""Probe 2: an unreaped Popen that becomes unreachable during the *next* test (probe only)."""
from __future__ import annotations

import gc
import subprocess
import sys

import pytest

HELD: list[subprocess.Popen] = []


@pytest.fixture
def child():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    yield proc
    proc.kill()
    if _WAIT:
        proc.wait(timeout=2.0)
    HELD.append(proc)          # keep it alive past the fixture teardown, like the `clients` list


_WAIT = True


def test_one_spawns_and_kills(child) -> None:
    assert child.pid > 0


def test_two_drops_the_handle_and_collects() -> None:
    HELD.clear()               # the fixture's `clients` list going away
    gc.collect()
    assert True


def test_three_still_runs() -> None:
    assert True
