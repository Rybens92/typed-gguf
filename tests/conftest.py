"""Test-suite gates: the offline unit gate must never touch the network or a real model.

`uv run pytest -q` runs everything except `@pytest.mark.network` / `@pytest.mark.model`
tests (SPEC A7); those are opt-in via `--run-network` and print as SKIPPED otherwise.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from ggufone.runtime import fit


@pytest.fixture
def in_process_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive the deep probe in-process — the implementation the probe child runs.

    `probe_runtime(deep=True)` asks a child process by default (E1a FIX t_eae35404: a command
    must never dlopen a bundle itself). A test that patches `load_backend_library` /
    `probe_symbols` is testing that implementation directly, so it has to say so.
    """
    from ggufone.runtime import capability
    monkeypatch.setattr(capability, "DEFAULT_SCAN", capability.scan_in_process)


@pytest.fixture
def pin_host_facts(
        monkeypatch: pytest.MonkeyPatch) -> Callable[[fit.HostFacts], fit.HostFacts]:
    """Freeze the ONE host reader a fit plan goes through (card t_e29734e6).

    `fit.host_facts` is the single function that reads the machine's RAM/device memory (the E1a
    rule, SPEC 2.2), so replacing that one name hands the whole call a *named* world — including
    the `fit`/`decide` CLI paths, which have no injection point of their own. A fit test that
    asserts a warning list without this pin is asserting the box's mood: the same
    `ggufone fit --json` legitimately warns `W_FIT_DOWNGRADE`/`W_KV_TYPE_DOWNGRADE` on a busy
    desktop (1631 MiB free of 8192 MiB — the operator's measured failure) and warns only
    `W_FIT_ESTIMATED` on a quiet one.

    The pinned callable answers calls *without* explicit facts only: a test that both pins a world
    and passes `meminfo_path=`/`device_probe=` has two worlds in flight, which is a bug in the
    test, not a preference. (`tests/test_fit_free_vram.py` and `tests/test_fit_oom_recovery.py`
    pin the same seam inline with `monkeypatch.setattr(fit, "host_facts", ...)`; this is that pin,
    named once so the next CLI-level gate does not have to reinvent it.)
    """
    from ggufone.runtime import fit

    def install(host: fit.HostFacts) -> fit.HostFacts:
        def pinned(**kwargs: object) -> fit.HostFacts:
            if kwargs:
                raise AssertionError(
                    f"the pinned host world owns the answer; {sorted(kwargs)} was passed instead "
                    f"of reading it")
            return host

        monkeypatch.setattr(fit, "host_facts", pinned)
        return host

    return install


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-network", action="store_true", default=False,
                     help="run @pytest.mark.network / @pytest.mark.model tests (needs network "
                          "and/or a real GGUF on disk)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-network"):
        return
    for item in items:
        if item.get_closest_marker("network") or item.get_closest_marker("model"):
            item.add_marker(pytest.mark.skip(reason="live test: pass --run-network to run"))


@pytest.fixture(autouse=True)
def _network_disabled_for_this_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A-E1c-10: with `GGUFONE_TEST_BLOCK_NET=1` every socket call raises.

    Used by `tools/e1c_offline_gate.py` to run the whole E1c surface (offline *and* the live
    model/runtime tests) with the network switched off at the Python level: no decision path may
    touch it. The flag is off by default so the `network`-marked download tests still work.
    """
    import os
    import socket
    if os.environ.get("GGUFONE_TEST_BLOCK_NET") not in ("1", "true", "yes"):
        return

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "network disabled for this run (GGUFONE_TEST_BLOCK_NET=1): a decision path must "
            "never need it")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
