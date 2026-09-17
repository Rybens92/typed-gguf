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
