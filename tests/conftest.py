"""Test-suite gates: the offline unit gate must never touch the network or a real model.

`uv run pytest -q` runs everything except `@pytest.mark.network` / `@pytest.mark.model`
tests (SPEC A7); those are opt-in via `--run-network` and print as SKIPPED otherwise.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def in_process_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive the deep probe in-process — the implementation the probe child runs.

    `probe_runtime(deep=True)` asks a child process by default (E1a FIX t_eae35404: a command
    must never dlopen a bundle itself). A test that patches `load_backend_library` /
    `probe_symbols` is testing that implementation directly, so it has to say so.
    """
    from ggufone.runtime import capability
    monkeypatch.setattr(capability, "DEFAULT_SCAN", capability.scan_in_process)


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
