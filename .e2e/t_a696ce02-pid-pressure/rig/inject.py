"""Measurement rig (card t_a696ce02): deny child spawns, like the pid cgroup does at its cap.

Loaded with `-p inject` (PYTHONPATH=/work/t_a696ce02). `GGUFONE_TEST_DENY_SPAWN=N` denies the
first N `subprocess.run` calls, `always` denies every one of them. The product sees exactly what
the kernel produces at the cap: `OSError(EAGAIN, 'Resource temporarily unavailable')`.

Also pins `pressure.SPAWN_BACKOFF = 0` so the *bounded* attempt count is what is measured, not
the wait budget (that one is pinned separately in tests/test_probe_pressure.py).
"""
from __future__ import annotations

import errno
import os
import pathlib
import subprocess

REAL_RUN = subprocess.run
LOG = pathlib.Path("/work/t_a696ce02/out/inject.log")


def pytest_configure(config) -> None:  # noqa: ANN001
    mode = os.environ.get("GGUFONE_TEST_DENY_SPAWN", "").strip()
    if not mode:
        return
    from ggufone.runtime import pressure
    pressure.SPAWN_BACKOFF = 0.0
    denied = {"n": 0}
    limit = None if mode == "always" else int(mode)

    def refuse(*args: object, **kwargs: object) -> object:
        denied["n"] += 1
        if limit is None or denied["n"] <= limit:
            raise OSError(errno.EAGAIN, "Resource temporarily unavailable")
        return REAL_RUN(*args, **kwargs)      # noqa: S603

    subprocess.run = refuse                     # noqa: S603
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(f"mode={mode}\n")


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ANN001
    with LOG.open("a") as fh:
        fh.write(f"exit={exitstatus} pids={pathlib.Path('/sys/fs/cgroup/pids.current').read_text().strip()}\n")
