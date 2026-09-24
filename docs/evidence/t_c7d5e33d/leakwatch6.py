"""Leakwatch 6: which Client holds an unreaped child at each test's teardown (probe, /tmp)."""
from __future__ import annotations

import gc
import subprocess

CURRENT = "?"
SPAWNS: list[tuple[int, str]] = []
_FIRST = {"flag": False}
REPORT = "/tmp/lw6-report.txt"


def _note(line: str) -> None:
    with open(REPORT, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def pytest_configure(config):  # noqa: ANN001
    if _FIRST["flag"]:
        return
    _FIRST["flag"] = True

    from typed_gguf.keep import client as client_module

    original_spawn = client_module._spawn_detached

    def spawn(argv, *, env, log):
        proc = original_spawn(argv, env=env, log=log)
        SPAWNS.append((proc.pid, CURRENT))
        _note(f"SPAWN pid={proc.pid} created_in={CURRENT}")
        return proc

    client_module._spawn_detached = spawn

    original_init = subprocess.Popen.__init__
    original_del = subprocess.Popen.__del__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._probe_created_in = CURRENT

    def delete(self):
        if getattr(self, "returncode", None) is None:
            _note(f"DEL pid={self.pid} returncode={self.returncode} created_in="
                  f"{getattr(self, '_probe_created_in', '?')} collected_during={CURRENT}")
        original_del(self)

    subprocess.Popen.__init__ = init
    subprocess.Popen.__del__ = delete


def _living_clients():
    from typed_gguf.keep import client as client_module

    return [obj for obj in gc.get_objects() if isinstance(obj, client_module.Client)]


def _report(phase: str, item) -> None:  # noqa: ANN001
    for client in _living_clients():
        leftovers = {pid: proc.returncode for pid, proc in client.children.items()}
        if leftovers:
            _note(f"CHILDREN {phase} {item.nodeid} children={leftovers}")


def pytest_runtest_setup(item):
    global CURRENT
    CURRENT = f"setup:{item.nodeid}"
    _report("at-setup", item)


def pytest_runtest_call(item):
    global CURRENT
    CURRENT = f"call:{item.nodeid}"


def pytest_runtest_teardown(item, nextitem):
    global CURRENT
    CURRENT = f"teardown:{item.nodeid}"
    _report("at-teardown", item)


def pytest_runtest_makereport(item, call):
    """Runs after each phase; `teardown` is post-fixture-finalizer."""
    if call.when == "teardown":
        _report("after-teardown", item)
