"""Test-suite gates: the offline unit gate must never touch the network or a real model.

`uv run pytest -q` runs everything except `@pytest.mark.network` / `@pytest.mark.model`
tests (SPEC A7); those are opt-in via `--run-network` and print as SKIPPED otherwise.

Second gate (card t_a696ce02): this container's pid cgroup is small and *shared* (256 pids,
sibling cards inside it), so `fork` answers EAGAIN exactly when the box is busy — which used to
be reported as product behaviour ("cuda does not load on this host" turning into
"the isolated probe could not be started"). A test that spawns a real child now says so
(`@pytest.mark.needs_fork`), the header carries the live reading, and when the cgroup is starved
those gates skip *loudly* while the run refuses to exit 0 — a starved box must never read as a
green suite. The reading lives in `typed_gguf.runtime.pressure` (the same one the probe path uses
to name its own failures).
"""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from typed_gguf.runtime import pressure

if TYPE_CHECKING:
    from typed_gguf.runtime import fit

#: A test that spawns a real child process cannot be measured on a box whose pid cgroup is at
#: its cap; the mark makes that requirement explicit and machine-readable.
FORK_GATE_MARKER = "needs_fork"
#: Prefix of the skip reason the gate writes (the terminal summary counts these).
PID_PRESSURE_SKIP_PREFIX = "pid cgroup has no fork headroom"


def parse_headroom(text: str) -> pressure.PidHeadroom | None:
    """`"250/256"` / `"17/max"` / `"none"` -> a `PidHeadroom` (or `None` for "no cgroup")."""
    text = text.strip()
    if text in ("", "none"):
        return None
    current, _, maximum = text.partition("/")
    return pressure.PidHeadroom(current=int(current),
                                maximum=None if maximum in ("", "max") else int(maximum))


def pid_headroom() -> pressure.PidHeadroom | None:
    """The live pid-cgroup reading, or the `TYPED_GGUF_TEST_PID_HEADROOM` override.

    The override is a *test* knob (same pattern as `TYPED_GGUF_TEST_BLOCK_NET`) so the gate itself
    can be pinned on a quiet box: `TYPED_GGUF_TEST_PID_HEADROOM=250/256` makes the gate see a
    starved cgroup, `none` makes it see a host without one. Unset -> the real cgroup.
    """
    override = os.environ.get("TYPED_GGUF_TEST_PID_HEADROOM")
    if override is not None:
        return parse_headroom(override)
    return pressure.read_pid_headroom()


@pytest.fixture
def in_process_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive the deep probe in-process — the implementation the probe child runs.

    `probe_runtime(deep=True)` asks a child process by default (E1a FIX t_eae35404: a command
    must never dlopen a bundle itself). A test that patches `load_backend_library` /
    `probe_symbols` is testing that implementation directly, so it has to say so.
    """
    from typed_gguf.runtime import capability
    monkeypatch.setattr(capability, "DEFAULT_SCAN", capability.scan_in_process)


@pytest.fixture
def pin_host_facts(
        monkeypatch: pytest.MonkeyPatch) -> Callable[[fit.HostFacts], fit.HostFacts]:
    """Freeze the ONE host reader a fit plan goes through (card t_e29734e6).

    `fit.host_facts` is the single function that reads the machine's RAM/device memory (the E1a
    rule, SPEC 2.2), so replacing that one name hands the whole call a *named* world — including
    the `fit`/`decide` CLI paths, which have no injection point of their own. A fit test that
    asserts a warning list without this pin is asserting the box's mood: the same
    `typed-gguf fit --json` legitimately warns `W_FIT_DOWNGRADE`/`W_KV_TYPE_DOWNGRADE` on a busy
    desktop (1631 MiB free of 8192 MiB — the operator's measured failure) and warns only
    `W_FIT_ESTIMATED` on a quiet one.

    The pinned callable answers calls *without* explicit facts only: a test that both pins a world
    and passes `meminfo_path=`/`device_probe=` has two worlds in flight, which is a bug in the
    test, not a preference. (`tests/test_fit_free_vram.py` and `tests/test_fit_oom_recovery.py`
    pin the same seam inline with `monkeypatch.setattr(fit, "host_facts", ...)`; this is that pin,
    named once so the next CLI-level gate does not have to reinvent it.)
    """
    from typed_gguf.runtime import fit

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


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{FORK_GATE_MARKER}: spawns a real child process (isolated probe child, llama-cli, the "
        "oracle); skipped — loudly — when the pid cgroup has no fork headroom, because a starved "
        "box is not a product finding")


def pytest_report_header(config: pytest.Config) -> list[str] | None:
    """Every run says which box it measured on: green only means something with the reading."""
    headroom = pid_headroom()
    if headroom is None:
        return ["pid cgroup: no readable /sys/fs/cgroup/pids.current (not a container)"]
    state = "STARVED — fork gates will skip" if headroom.starved() else "ok"
    return [f"pid cgroup: {headroom.describe()} ({state}; the fork gates need "
            f"{pressure.PID_HEADROOM_FLOOR} free)"]


@pytest.fixture(autouse=True)
def _fork_gate_headroom(request: pytest.FixtureRequest) -> None:
    """A fork-dependent gate on a starved box skips *by name*; it never reports the box as product.

    The check is per test, not per session: a shared cgroup can go from roomy to full in the
    middle of a run (measured: `pids.current` 213 -> 256 inside one 3.3 s run), and the gate the
    box cannot feed is exactly the test that would mislabel it.
    """
    if request.node.get_closest_marker(FORK_GATE_MARKER) is None:
        return
    headroom = pid_headroom()
    if headroom is None or not headroom.starved():
        return
    session = request.session
    session.__dict__["_gg_pid_pressure_skips"] = \
        int(session.__dict__.get("_gg_pid_pressure_skips", 0)) + 1
    pytest.skip(f"{PID_PRESSURE_SKIP_PREFIX} ({headroom.describe()}): this gate spawns a real "
                f"child process and the box is at its pid cap — the box, not the product; "
                f"re-run when pids.current < pids.max - {pressure.PID_HEADROOM_FLOOR}")


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter,
                            exitstatus: int | pytest.ExitCode,
                            config: pytest.Config) -> None:
    """A run that had to skip fork gates says so in full, next to its own summary line."""
    skipped = [report for report in terminalreporter.stats.get("skipped", [])
               if PID_PRESSURE_SKIP_PREFIX in str(report.longrepr)]
    if not skipped:
        return
    terminalreporter.write_sep(
        "=", f"PID PRESSURE: {len(skipped)} fork gate(s) skipped — this run could not measure them")
    for report in skipped:
        terminalreporter.write_line(f"  {report.nodeid}: {report.longrepr}")
    terminalreporter.write_line("  the box, not the product: the exit status is forced non-zero "
                                "so a starved cgroup can never be read as a green suite")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int | pytest.ExitCode) -> None:
    """A starved run must not exit 0 (the skipped gates are the reason green would be a lie)."""
    if int(exitstatus) == 0 and int(session.__dict__.get("_gg_pid_pressure_skips", 0)) > 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-network"):
        return
    for item in items:
        if item.get_closest_marker("network") or item.get_closest_marker("model"):
            item.add_marker(pytest.mark.skip(reason="live test: pass --run-network to run"))


@pytest.fixture(autouse=True)
def _network_disabled_for_this_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A-E1c-10: with `TYPED_GGUF_TEST_BLOCK_NET=1` every socket call raises.

    Used by `tools/e1c_offline_gate.py` to run the whole E1c surface (offline *and* the live
    model/runtime tests) with the network switched off at the Python level: no decision path may
    touch it. The flag is off by default so the `network`-marked download tests still work.
    """
    import os
    import socket
    if os.environ.get("TYPED_GGUF_TEST_BLOCK_NET") not in ("1", "true", "yes"):
        return

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "network disabled for this run (TYPED_GGUF_TEST_BLOCK_NET=1): a decision path must "
            "never need it")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
