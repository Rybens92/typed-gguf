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

import contextlib
import os
import pathlib
import shutil
import socket
import tempfile
from collections.abc import Callable, Iterator
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
#: The mode knob `tests/fake_keep_host.py` reads: a keep gate must not inherit a session's mode.
FAKE_HOST_MODE_ENV = "TYPED_GGUF_KEEP_FAKE"

# --------------------------------------------------------------------- the keep gates' home
#: A keep home binds a unix socket under `<home>/keep/`, and that path has to fit in `sun_path`
#: (108 bytes including the NUL — `typed_gguf.keep.state.SUN_PATH_MAX`). pytest's `tmp_path`
#: inherits the *ambient* `TMPDIR`, and agent/session environments point that at a long
#: profile-scratch path: a home built from it overflows the limit and every keep gate goes red on a
#: box where the product is right (card t_c3195a5c measured 35 red of 93 — all of them the
#: product's own `E_UNKNOWN_KEY`). The gates therefore build their home under a short base; a path
#: whose length does not matter still uses `tmp_path`.
SHORT_SOCKET_BASES: tuple[pathlib.Path, ...] = (pathlib.Path("/tmp"),)
KEEP_HOME_PREFIX = "tg-keep-"
#: The skip a keep gate raises when no short base is writable *and* the ambient one is too long:
#: the box cannot host the gate, and saying so by name is the honest answer.
SHORT_BASE_SKIP_REASON = (
    "no short writable base for the keep socket (tried {tried}) and the ambient base {ambient} "
    "cannot carry one either — `sun_path` is {limit} bytes including the NUL: run with a shorter "
    "TMPDIR or a writable /tmp")


def socket_fits(home: pathlib.Path) -> bool:
    """Would the product accept a socket path under this `home`? Its own rule is the oracle."""
    from typed_gguf.errors import UserError
    from typed_gguf.keep import identity, state

    digest = identity.KeepKey.of(None, model_path="/models/socket-fit-check.gguf").digest
    try:
        state.socket_path(home, digest)
    except UserError:
        return False
    return True


def short_socket_base() -> pathlib.Path | None:
    """The first writable base of `SHORT_SOCKET_BASES`, or None when none of them is."""
    for candidate in SHORT_SOCKET_BASES:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = tempfile.mkdtemp(prefix=KEEP_HOME_PREFIX, dir=str(candidate))
        except OSError:                      # not writable, not a dir, or no such base at all
            continue
        with contextlib.suppress(OSError):
            os.rmdir(probe)                  # a probe: the home itself is the caller's to make
        return candidate
    return None


@contextlib.contextmanager
def keep_home_that_binds(ambient_base: pathlib.Path,
                         *, name: str = "home") -> Iterator[pathlib.Path]:
    """A throwaway data home whose `keep/` socket fits in `sun_path`.

    The home goes under `short_socket_base()` — never under `ambient_base`, whose length is the
    ambient `TMPDIR`'s business and not the gate's. With no short base to pin it to, the ambient
    base is the fallback: a short one hosts the gate exactly as before, and one that genuinely
    cannot carry the socket skips **by name** (`SHORT_BASE_SKIP_REASON`) instead of failing — a box
    that cannot host a gate must never read as a product failure.
    """
    from typed_gguf.keep import state

    base = short_socket_base()
    if base is None:
        home = ambient_base / name
        if not socket_fits(home):
            pytest.skip(SHORT_BASE_SKIP_REASON.format(
                tried=", ".join(str(candidate) for candidate in SHORT_SOCKET_BASES),
                ambient=ambient_base, limit=state.SUN_PATH_MAX))
        home.mkdir(parents=True, exist_ok=True)
        yield home
        return
    root = pathlib.Path(tempfile.mkdtemp(prefix=KEEP_HOME_PREFIX, dir=str(base)))
    home = root / name
    home.mkdir()
    try:
        yield home
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def keep_home(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> Iterator[pathlib.Path]:
    """The data home every keep gate runs against, with the two knobs it must not inherit.

    `TYPED_GGUF_HOME` points at it (the product and the fake host child both read that home), the
    keep-alive env is cleared (the precedence chain is exercised by name in
    `tests/test_keep_cli.py`) and so is the fake host's mode. One fixture, because four gate files
    ask the same three questions of their home — and the home is *short enough to bind its socket*
    whatever `TMPDIR` the session was started with (`keep_home_that_binds`).
    """
    from typed_gguf.keep import identity

    with keep_home_that_binds(tmp_path) as home:
        monkeypatch.setenv("TYPED_GGUF_HOME", str(home))
        monkeypatch.delenv(identity.KEEP_ALIVE_ENV, raising=False)
        monkeypatch.delenv(FAKE_HOST_MODE_ENV, raising=False)
        yield home


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


# --------------------------------------------------------------------- the net-off flag
#: The knob `tools/e1c_offline_gate.py` and `.github/workflows/ci.yml` set on a whole run.
NET_BLOCK_ENV = "TYPED_GGUF_TEST_BLOCK_NET"
#: The families the flag forbids: the ones that can leave the box. `AF_UNIX` is a *file*, not a
#: network — the E4 keep host binds one in-process and the CI runs this whole suite with the flag
#: on, so forbidding every family turned the first push of that tree red (card t_a4ebcd36: 24
#: failed / 11 errors, all of them the hook's own `AssertionError`).
NETWORK_FAMILIES: tuple[int, ...] = (socket.AF_INET, socket.AF_INET6)
#: The real constructor, captured at import time: installing the block twice (a gate that pins the
#: helper *inside* a run the flag already governs) must keep one base class instead of nesting.
_REAL_SOCKET: type = socket.socket


def _network_forbidden(*_args: object, **_kwargs: object) -> None:
    """The flag's own raiser: the message is what a failing decision path reads."""
    raise AssertionError(
        f"network disabled for this run ({NET_BLOCK_ENV}=1): a decision path must never need it")


class _NoNetworkSocket(_REAL_SOCKET):
    """`socket.socket`, minus the families that could reach a network.

    A *subclass*, not a stand-in raiser: everything about it is the stdlib's own class, so
    `isinstance`, `socketpair`, timeouts and every method keep working — only building a socket in
    `NETWORK_FAMILIES` (or in the default family, which *is* `AF_INET`) raises. `AF_UNIX` and the
    other local families are deliberately allowed: the flag's subject is the network.
    """

    def __init__(self, family: int = -1, type: int = -1, proto: int = -1,
                 fileno: int | None = None) -> None:
        if fileno is None and family == -1:
            family = socket.AF_INET          # `socket.py`'s own default, spelled out here
        if family in NETWORK_FAMILIES:
            _network_forbidden()
        super().__init__(family, type, proto, fileno)


def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the net-off block a run under `TYPED_GGUF_TEST_BLOCK_NET=1` gets.

    One place owns what the flag means, so the gates can pin it directly instead of only through a
    whole session (`tests/test_net_block_scope.py`): the network families cannot be built,
    `create_connection`/`getaddrinfo` raise, and local IPC is untouched.
    """
    monkeypatch.setattr(socket, "socket", _NoNetworkSocket)
    monkeypatch.setattr(socket, "create_connection", _network_forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", _network_forbidden)


@pytest.fixture(autouse=True)
def _network_disabled_for_this_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A-E1c-10: with `TYPED_GGUF_TEST_BLOCK_NET=1` no *network* socket call can be built.

    Used by `tools/e1c_offline_gate.py` to run the whole E1c surface (offline *and* the live
    model/runtime tests) with the network switched off at the Python level, and by the CI's
    offline-suite step: no decision path may touch it. The block is scoped to the network families
    (`block_network`) — the keep host's `AF_UNIX` socket is local IPC and stays available, or the
    CI step would fail the feature as well as the download path it was written to guard. The flag
    is off by default so the `network`-marked download tests still work.
    """
    if os.environ.get(NET_BLOCK_ENV) not in ("1", "true", "yes"):
        return
    block_network(monkeypatch)

