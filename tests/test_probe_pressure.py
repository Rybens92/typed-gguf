"""A starved pid table is *named*, never mislabelled as a bundle failure (card t_a696ce02).

Found while landing `t_80f1a4c6`: the repo's default gate (`pytest -q`, pytest-randomly) is not
reliably green in the worker container. The runtime-probe files fail a variable number of
assertions per run — `['probe_failed', 'no_asset'] == ['loader_error', 'no_asset']` — because
`tests/test_runtime_{fallback,install,contract}.py` spawn *real* children (the isolated probe
child, `llama-cli --version`, the oracle) and the container's shared pid cgroup (256 pids, with
the sibling cards inside it) answers `fork` with `EAGAIN` once it is at the cap. Measured: green
while `pids.current` is under ~200, red at 244-256 — the box, reported as the product.

These gates pin the two halves of the fix:

* the probe path retries a *transient* spawn failure for a bounded budget and then names it
  (`E_PID_PRESSURE` + the live cgroup reading) — the reason strings never turn into a loader
  story (`tests/test_probe_pressure.py::test_the_install_chain_...`);
* the suite's own gate (`tests/conftest.py`) skips the fork-dependent gates under a starved
  cgroup *loudly* and refuses to look green while they are skipped.
"""
from __future__ import annotations

import errno
import json
import os
import pathlib
import subprocess
import sys

import pytest

from ggufone.runtime import install, isolated, pins, pressure
from tests.conftest import pid_headroom
from tests.test_runtime_fallback import GPU_HOST, bundle_cache, multi_lock

ROOT = pathlib.Path(__file__).resolve().parents[1]

EAGAIN = OSError(errno.EAGAIN, "Resource temporarily unavailable")


def count_spawns(monkeypatch: pytest.MonkeyPatch, *, fail_first: int = 0, always: bool = False,
                 error: OSError | None = None) -> dict[str, int]:
    """Wrap `subprocess.run`: deny the first `fail_first` spawns (or all) with `error`/EAGAIN."""
    real = subprocess.run
    seen = {"attempts": 0}

    def flaky(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        seen["attempts"] += 1
        if always or seen["attempts"] <= fail_first:
            raise error if error is not None else EAGAIN
        return real(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", flaky)
    return seen


def empty_runtime(tmp_path: pathlib.Path) -> pathlib.Path:
    """A runtime dir the probe child can start on (it answers "libggml-base.so is missing")."""
    rt = tmp_path / "runtime"
    rt.mkdir(parents=True, exist_ok=True)
    return rt


# --------------------------------------------------------------------- the reading
def test_the_reading_names_the_numbers(tmp_path: pathlib.Path) -> None:
    """The phrase every pressure failure carries: `pids.current=250/256 (6 free)`."""
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "pids.current").write_text("250\n")
    (root / "pids.max").write_text("256\n")

    head = pressure.read_pid_headroom(root)

    assert head == pressure.PidHeadroom(current=250, maximum=256)
    assert head.describe() == "pids.current=250/256 (6 free)"
    assert head.free == 6
    assert head.starved() is True                      # 6 free < the 16-pid floor
    assert head.starved(floor=6) is False              # exactly at the floor is not starved
    assert pressure.PidHeadroom(current=1, maximum=256).starved() is False
    # a cgroup that is over its cap must not report negative headroom
    assert pressure.PidHeadroom(current=260, maximum=256).free == 0


def test_an_unlimited_or_absent_cgroup_is_not_pressure(tmp_path: pathlib.Path) -> None:
    """`pids.max == "max"` (a plain host) and a box without the files are both real answers."""
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "pids.current").write_text("17\n")
    (root / "pids.max").write_text("max\n")

    head = pressure.read_pid_headroom(root)

    assert head is not None and head.maximum is None
    assert head.free is None and head.starved() is False
    assert head.describe() == "pids.current=17 (no pid limit)"
    assert pressure.read_pid_headroom(tmp_path / "nowhere") is None


def test_an_unreadable_reading_is_none_not_a_guess(tmp_path: pathlib.Path) -> None:
    """Garbage in the files (or no `pids.max`) must never become invented numbers."""
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "pids.current").write_text("not-a-number\n")
    (root / "pids.max").write_text("256\n")
    assert pressure.read_pid_headroom(root) is None       # no reading at all

    (root / "pids.current").write_text("13\n")
    (root / "pids.max").write_text("weird\n")
    head = pressure.read_pid_headroom(root)
    assert head == pressure.PidHeadroom(current=13, maximum=None)
    assert head.starved() is False and head.describe() == "pids.current=13 (no pid limit)"


def test_only_pressure_errnos_are_treated_as_pressure() -> None:
    assert pressure.is_spawn_pressure(EAGAIN) is True
    assert pressure.is_spawn_pressure(OSError(errno.ENOMEM, "Cannot allocate memory")) is True
    assert pressure.is_spawn_pressure(FileNotFoundError(errno.ENOENT, "No such file")) is False
    assert pressure.is_spawn_pressure(subprocess.TimeoutExpired("x", 1)) is False


# --------------------------------------------------------------------- the retry
@pytest.mark.needs_fork
def test_a_transient_fork_failure_is_retried_until_the_probe_answers(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Two EAGAINs then the box frees up: the probe must still answer (that is the flake)."""
    seen = count_spawns(monkeypatch, fail_first=2)

    scan = isolated.scan_bundle(empty_runtime(tmp_path))

    assert scan.child_error is None, scan.child_error      # the child really ran
    assert scan.error is not None and "libggml-base.so" in scan.error
    assert seen["attempts"] == 3                           # 2 denied + 1 that got through


def test_a_sustained_fork_block_is_named_pid_pressure_not_a_bundle(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """The box at its cap: bounded retries, then one *named* reason carrying the cgroup reading."""
    monkeypatch.setattr(pressure, "SPAWN_BACKOFF", 0)   # the wait budget is pinned separately
    seen = count_spawns(monkeypatch, always=True)

    scan = isolated.scan_bundle(empty_runtime(tmp_path))

    assert scan.child_error is not None
    assert pressure.E_PID_PRESSURE in scan.child_error
    assert "fork headroom" in scan.child_error
    assert seen["attempts"] == pressure.SPAWN_ATTEMPTS          # bounded, not endless
    head = pressure.read_pid_headroom()
    if head is not None:              # the box's own numbers, when there are any
        assert head.describe() in scan.child_error
    for lie in ("does not load on this host", "cannot open shared object", "carries no"):
        assert lie not in scan.child_error, scan.child_error


@pytest.mark.needs_fork
def test_a_child_that_ran_is_data_and_is_never_retried(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """A probe that started and died is a finding about the bundle/box — retrying it would lie."""
    seen = count_spawns(monkeypatch)
    monkeypatch.setattr(isolated, "child_command",
                        lambda: [sys.executable, "-c", "raise SystemExit(9)"])

    scan = isolated.scan_bundle(empty_runtime(tmp_path))

    assert seen["attempts"] == 1
    assert scan.child_error is not None and "exited 9" in scan.child_error


def test_a_non_pressure_oserror_is_raised_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """ENOENT is the command's own fault: propagate it as-is, after exactly one attempt."""
    seen = count_spawns(monkeypatch, always=True, error=FileNotFoundError(errno.ENOENT, "nope"))

    with pytest.raises(FileNotFoundError):
        pressure.spawn(["nowhere-at-all"])

    assert seen["attempts"] == 1                     # not retried: nothing about the box
    assert issubclass(pressure.SpawnBlocked, OSError)   # existing OSError handlers keep working


def test_the_retry_budget_is_bounded_and_documented() -> None:
    assert pressure.SPAWN_ATTEMPTS >= 2
    assert pressure.SPAWN_BACKOFF > 0
    total = sum(pressure.SPAWN_BACKOFF * 2 ** (n - 1) for n in range(1, pressure.SPAWN_ATTEMPTS))
    assert total <= 10, f"the wait budget is too long to sit through: {total}s"


def test_the_backoff_schedule_is_the_documented_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wait between attempts is exponential (0.2/0.4/0.8 s), one sleep per retry, no more.

    The schedule *is* the policy (bounded, spread out, short enough to sit through), so it is
    pinned rather than left to the formula's luck: three retries sleep three times, and the last
    attempt does not sleep at all.
    """
    count_spawns(monkeypatch, always=True)
    slept: list[float] = []
    monkeypatch.setattr(pressure.time, "sleep", slept.append)

    with pytest.raises(pressure.SpawnBlocked):
        pressure.spawn(["never-reached"])

    assert slept == [pressure.SPAWN_BACKOFF * 2 ** n
                     for n in range(pressure.SPAWN_ATTEMPTS - 1)], slept


def test_a_sustained_block_names_the_underlying_errno(monkeypatch: pytest.MonkeyPatch) -> None:
    """The message carries the kernel's own words, not just the box's numbers: a reader can see
    *why* the child did not start (`[Errno 11] Resource temporarily unavailable`)."""
    monkeypatch.setattr(pressure, "SPAWN_BACKOFF", 0)
    count_spawns(monkeypatch, always=True)

    with pytest.raises(pressure.SpawnBlocked) as caught:
        pressure.spawn(["never-reached"])

    text = str(caught.value)
    assert "BlockingIOError" in text and "Errno 11" in text
    assert "Resource temporarily unavailable" in text
    assert pressure.E_PID_PRESSURE in text
    assert caught.value.errno == errno.EAGAIN      # the causal errno survives the wrapping


def test_the_pressure_note_is_empty_when_there_is_no_cgroup(tmp_path: pathlib.Path) -> None:
    """`pressure_note` is appended to messages: with nothing to read it must add nothing at all
    (never a placeholder, never an `AttributeError` on a `None` reading)."""
    assert pressure.pressure_note(tmp_path / "nowhere") == ""

    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "pids.current").write_text("250\n")
    (root / "pids.max").write_text("256\n")
    assert pressure.pressure_note(root) == "; pid cgroup: pids.current=250/256 (6 free)"


# --------------------------------------------------------------------- the contract
def test_the_install_chain_names_the_pressure_never_the_loader(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """The reason strings other cards assert on: a refuted dlopen is `loader_error`, a box that
    cannot fork is `probe_failed` *with the pressure named* — never one dressed as the other."""
    cache = bundle_cache(tmp_path, ("cuda", "vulkan", "cpu"))
    lock = pins.load_lock(multi_lock(cache, ("cuda", "vulkan", "cpu")))
    monkeypatch.setattr(pressure, "SPAWN_BACKOFF", 0)   # the wait budget is pinned separately
    count_spawns(monkeypatch, always=True)

    result = install.install("auto", home=tmp_path / "home", lock=lock, offline_cache=cache,
                             free_bytes=1 << 40, probes=GPU_HOST)

    # the chain still lands on the tier that cannot fail to fork: cpu
    assert result["variant"] == "linux-x64-cpu"
    assert result["working_backend"] == "cpu"
    codes = [attempt["code"] for attempt in result["fallback_attempts"]]
    assert codes == [install.REASON_PROBE_FAILED, install.REASON_PROBE_FAILED]
    assert [attempt["backend"] for attempt in result["fallback_attempts"]] == ["cuda", "vulkan"]
    for attempt in result["fallback_attempts"]:
        assert pressure.E_PID_PRESSURE in attempt["reason"], attempt
        assert "isolated probe" in attempt["reason"], attempt
        assert "does not load on this host" not in attempt["reason"], attempt
        assert "carries no" not in attempt["reason"], attempt
    record = json.loads((tmp_path / "home" / "runtime.json").read_text())
    assert record["fallback_reason_code"] == install.REASON_PROBE_FAILED
    assert pressure.E_PID_PRESSURE in record["fallback_reason"]


# --------------------------------------------------------------------- the suite gate
@pytest.mark.needs_fork
def test_a_fork_gate_probe() -> None:
    """The stand-in the nested run below collects: a gate that needs a real child process."""
    assert True


def run_nested(*selection: str, headroom: str) -> subprocess.CompletedProcess[str]:
    """A real pytest run with a *simulated* cgroup reading (the gate's own end-to-end pin)."""
    env = {**os.environ, "GGUFONE_TEST_PID_HEADROOM": headroom, "PYTEST_ADDOPTS": ""}
    return subprocess.run(                                    # noqa: S603
        [sys.executable, "-m", "pytest", "-q", "-p", "no:randomly", "-rs", "--tb=line",
         *selection],
        capture_output=True, text=True, cwd=str(ROOT), env=env, timeout=300, check=False)


@pytest.mark.needs_fork
def test_the_gate_skips_fork_gates_under_a_starved_cgroup_and_refuses_to_look_green() -> None:
    """`GGUFONE_TEST_PID_HEADROOM=250/256`: the fork gate skips *loudly*, the headroom gate fails,
    and the run exits non-zero — a starved box can never be read as "the product is fine"."""
    result = run_nested(
        "tests/test_probe_pressure.py::test_a_fork_gate_probe",
        "tests/test_probe_pressure.py::test_the_pid_cgroup_has_fork_headroom_for_the_probe_gates",
        headroom="250/256")

    assert result.returncode != 0, result.stdout[-3000:]
    assert "1 failed" in result.stdout and "1 skipped" in result.stdout, result.stdout[-3000:]
    assert "pid cgroup has no fork headroom (pids.current=250/256 (6 free))" in result.stdout, \
        result.stdout[-3000:]
    assert "PID PRESSURE" in result.stdout, result.stdout[-3000:]


@pytest.mark.needs_fork
def test_the_gate_runs_the_fork_gates_on_a_healthy_box() -> None:
    """Same selection, no simulated starvation: the fork gate runs and the headroom gate passes."""
    result = run_nested(
        "tests/test_probe_pressure.py::test_a_fork_gate_probe",
        "tests/test_probe_pressure.py::test_the_pid_cgroup_has_fork_headroom_for_the_probe_gates",
        headroom="0/256")

    assert result.returncode == 0, result.stdout[-3000:]
    assert "2 passed" in result.stdout, result.stdout[-3000:]
    assert "PID PRESSURE" not in result.stdout


def test_the_pid_cgroup_has_fork_headroom_for_the_probe_gates() -> None:
    """The gate itself: a starved box fails *here*, by name, instead of mislabelling a bundle.

    It is one loud, named failure — not 36 phantom product failures. Re-run when the box has
    headroom; the number is the container's, not the code's.
    """
    head = pid_headroom()
    if head is None:
        pytest.skip("no readable pid cgroup (not a container): the fork gates cannot be gated here")
    assert not head.starved(), (
        f"the pid cgroup is exhausted: {head.describe()} — the runtime-probe gates need to fork "
        f"(isolated probe children, `llama-cli --version`, the oracle) and cannot run reliably "
        f"under {pressure.PID_HEADROOM_FLOOR} free pids. This run cannot measure them, so it must "
        f"not look green: re-run when pids.current < pids.max - {pressure.PID_HEADROOM_FLOOR} "
        f"(sibling work shares this cgroup)."
    )
