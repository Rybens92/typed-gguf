"""Temporary measurement plugin (card t_a696ce02): pids.current per test.

Loaded with `PYTHONPATH=/work/t_a696ce02 .venv/bin/python -m pytest -p pidtrace`.
Not part of the repo: it measures whether the *suite itself* accumulates pid-cgroup
pressure between tests (leaked children) or whether the pressure is external.
"""
from __future__ import annotations

import pathlib

LOG = pathlib.Path("/work/t_a696ce02/out/pidtrace.log")


def _pids() -> str:
    try:
        cur = pathlib.Path("/sys/fs/cgroup/pids.current").read_text().strip()
        mx = pathlib.Path("/sys/fs/cgroup/pids.max").read_text().strip()
        return f"{cur}/{mx}"
    except OSError:
        return "?"


def _write(line: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def pytest_sessionstart(session) -> None:  # noqa: ANN001
    _write(f"=== session start pids={_pids()}")


def pytest_runtest_setup(item) -> None:  # noqa: ANN001
    _write(f"setup    {item.nodeid} pids={_pids()}")


def pytest_runtest_teardown(item, nextitem) -> None:  # noqa: ANN001
    nxt = nextitem.nodeid if nextitem is not None else "<end>"
    _write(f"teardown {item.nodeid} pids={_pids()} next={nxt}")


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ANN001
    _write(f"=== session end pids={_pids()} exit={exitstatus}")
