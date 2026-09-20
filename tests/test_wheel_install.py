"""An out-of-tree install finds its own pins (card t_eff926f9).

Everything else in the suite runs from the checkout, where `runtime.lock` sits above
`src/typed_gguf/` — i.e. from the one directory where this bug cannot show. The user asked about
the `uvx` one-liner, which is the opposite shape: an installed wheel has no repository above it,
so `pins.default_lock_path()` walked up, found nothing, and every command answered

    E_RUNTIME_MISSING: /tmp/runtime.lock not found; run from the repository root

This gate builds the real wheel (`uv build`), installs it into a fresh tool env in a temp dir
(`uv tool install` — the same install the `uvx --from …` one-liner performs) and runs `version`,
`init --dry-run` and `doctor --json` from a *neutral* cwd. That cwd also carries a decoy lock with
a different tag and different asset sizes, so a pass cannot come from the old cwd fallback either:
the answer has to come from the copy inside the wheel.

Offline by design (SPEC A7): the wheel has no dependencies, `--offline` reads the local uv cache
for the build backend, and every command reads only its own package. A box without `uv`, or
without that backend in its cache, skips *loudly* — it cannot measure this contract.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
from dataclasses import dataclass

import pytest

from tests.conftest import PID_PRESSURE_SKIP_PREFIX, pid_headroom
from typed_gguf.runtime import pressure

ROOT = pathlib.Path(__file__).resolve().parents[1]
UV = shutil.which("uv")
TOOL = "typed-gguf"
REPO_LOCK = json.loads((ROOT / "runtime.lock").read_text(encoding="utf-8"))
TAG = REPO_LOCK["llama_cpp"]["tag"]
#: a lock that is *not* the pinned one: if a command reads it, the tag/size assertions below see it
DECOY_TAG = "b00042"
#: what `uv build --offline` answers when the build backend is not in the local cache. Fetching it
#: is exactly what SPEC A7 forbids the offline gate, so that box skips instead of guessing.
NO_BACKEND_TOKENS = ("network was disabled", "not found in the cache",
                     "no solution found when resolving")
#: what a full pid cgroup answers instead: `uv` panics on the thread it cannot spawn. Measured on
#: this box (card t_eff926f9) — the fixture, not just the tests, has to recognise it.
STARVATION_TOKENS = ("resource temporarily unavailable", "os error 11", "cannot allocate memory")

#: the artifact gates start real children: `uv build`, `uv tool install`, the installed CLI. The
#: two harness pins at the end of this file spawn nothing and stay measurable on a starved box.
FORK = pytest.mark.needs_fork


def pressure_skip(session: pytest.Session, detail: str) -> None:
    """Skip the way the conftest's fork gate does: named, counted, and non-zero at the end.

    `_gg_pid_pressure_skips` is what `pytest_sessionfinish` reads to refuse a green status, and the
    message carries `PID_PRESSURE_SKIP_PREFIX` so `pytest_terminal_summary` prints it in full.
    """
    session.__dict__["_gg_pid_pressure_skips"] = \
        int(session.__dict__.get("_gg_pid_pressure_skips", 0)) + 1
    pytest.skip(f"{PID_PRESSURE_SKIP_PREFIX} ({detail}): this gate spawns real children "
                f"(`uv build`, `uv tool install`, the installed CLI) and the box is at its pid "
                f"cap — the box, not the product; re-run when pids.current < "
                f"pids.max - {pressure.PID_HEADROOM_FLOOR}")


def starved_detail() -> str | None:
    """The live reading when the cgroup has no fork headroom, else None (honours the test knob)."""
    headroom = pid_headroom()
    if headroom is None or not headroom.starved():
        return None
    return f"starved before it started: {headroom.describe()}"


def starvation(blob: str) -> str | None:
    """The token that says a `uv` failure was the box's pid cgroup rather than the artifact.

    The cgroup can fill *during* a build the box was roomy enough to start (`uv` panics on the
    thread it cannot spawn: "OS can't spawn worker thread: Resource temporarily unavailable
    (os error 11)"), so the failure has to be classified, not asserted away.
    """
    lowered = blob.lower()
    return next((token for token in STARVATION_TOKENS if token in lowered), None)


def decoy_lock() -> str:
    """A valid lock with a different pin, dropped into the cwd a wheel user stands in."""
    payload = json.loads((ROOT / "runtime.lock").read_text(encoding="utf-8"))
    payload["llama_cpp"]["tag"] = DECOY_TAG
    for spec in payload["llama_cpp"]["assets"].values():
        spec["size"] = int(spec["size"]) + 1
    return json.dumps(payload)


@dataclass(frozen=True)
class Installed:
    """The tool env a `uvx --from … typed-gguf` user ends up with."""

    bin: pathlib.Path
    site_packages: pathlib.Path
    home: pathlib.Path

    def env(self, **extra: str) -> dict[str, str]:
        """A command environment: an empty data home, no lock override, this box's own PATH."""
        env = {**os.environ, "TYPED_GGUF_HOME": str(self.home)}
        env.pop("TYPED_GGUF_LOCK", None)   # an override would answer for the package
        env.update(extra)
        return env

    def run(self, *argv: str, cwd: pathlib.Path,
            **extra: str) -> subprocess.CompletedProcess[str]:
        return pressure.spawn([str(self.bin), *argv], cwd=str(cwd), capture_output=True,
                              text=True, check=False, env=self.env(**extra))


@pytest.fixture(scope="module")
def wheel(request: pytest.FixtureRequest,
          tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """The distribution this tree builds — the artifact the `uvx` one-liner installs."""
    if UV is None:
        pytest.skip("uv is not on PATH: this gate installs the wheel the way `uvx` does")
    if (detail := starved_detail()) is not None:
        # the conftest's autouse fork gate cannot see this: a fixture is not a test, and `uv build`
        # answers a full cgroup by panicking — which would read as a product failure below
        pressure_skip(request.session, detail)
    dist = tmp_path_factory.mktemp("dist")
    result = pressure.spawn([UV, "build", "--wheel", "--offline", "-o", str(dist), str(ROOT)],
                            capture_output=True, text=True, check=False, timeout=600)
    blob = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        if any(token in blob.lower() for token in NO_BACKEND_TOKENS):
            pytest.skip("the build backend is not in the local uv cache and this gate builds "
                        f"offline (SPEC A7): {blob.strip()[-300:]}")
        if starvation(blob):
            pressure_skip(request.session, f"`uv build` hit a full cgroup: {blob.strip()[-200:]}")
        pytest.fail(f"`uv build` failed:\n{blob[-2000:]}")
    built = sorted(dist.glob("typed_gguf-*.whl"))
    assert len(built) == 1, f"expected exactly one wheel, found {[p.name for p in built]}"
    return built[0]


@pytest.fixture()
def installed(wheel: pathlib.Path, request: pytest.FixtureRequest,
              tmp_path: pathlib.Path) -> Installed:
    """A fresh `uv tool install` of the built wheel, in its own temp dir."""
    root = tmp_path / "tool"
    env = {**os.environ, "UV_TOOL_DIR": str(root / "tools"), "UV_TOOL_BIN_DIR": str(root / "bin")}
    if (detail := starved_detail()) is not None:
        pressure_skip(request.session, detail)
    result = pressure.spawn([UV, "tool", "install", "--offline", "--from", str(wheel), TOOL],
                            capture_output=True, text=True, check=False, timeout=600, env=env)
    blob = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        if starvation(blob):
            pressure_skip(request.session,
                          f"`uv tool install` hit a full cgroup: {blob.strip()[-200:]}")
        raise AssertionError(f"`uv tool install` failed:\n{blob[-2000:]}")
    site_packages = next((root / "tools" / TOOL).glob("lib/python*/site-packages"))
    return Installed(bin=root / "bin" / TOOL, site_packages=site_packages, home=root / "home")


@pytest.fixture()
def neutral(tmp_path: pathlib.Path) -> pathlib.Path:
    """The cwd of a `uvx` user: nothing to do with the checkout, and a lock-shaped decoy in it."""
    cwd = tmp_path / "neutral"
    cwd.mkdir()
    (cwd / "runtime.lock").write_text(decoy_lock(), encoding="utf-8")
    return cwd


@FORK
def test_the_wheel_carries_the_root_lock_verbatim(installed: Installed) -> None:
    """Requirement 2, measured on the artifact: one source (the repo root) and one copy inside the
    distribution — byte-identical, so neither can drift away from the other."""
    packaged = installed.site_packages / "typed_gguf" / "data" / "runtime.lock"
    assert packaged.exists(), (
        "the wheel ships no runtime.lock: an out-of-tree install has nothing to read and every "
        "command answers E_RUNTIME_MISSING")
    assert packaged.read_bytes() == (ROOT / "runtime.lock").read_bytes()


@FORK
def test_version_reads_the_packaged_pin_from_a_neutral_cwd(installed: Installed,
                                                           neutral: pathlib.Path) -> None:
    result = installed.run("version", "--json", cwd=neutral)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["lock"]["tag"] == TAG
    assert payload["lock"]["tag"] != DECOY_TAG, (
        "the answer came from the decoy in the cwd, not from the lock inside the wheel")


@FORK
def test_init_dry_run_plans_from_the_packaged_pin(installed: Installed,
                                                  neutral: pathlib.Path) -> None:
    result = installed.run("init", "--dry-run", "--json", cwd=neutral)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "E_RUNTIME_MISSING" not in result.stdout + result.stderr
    payload = json.loads(result.stdout)
    spec = REPO_LOCK["llama_cpp"]["assets"][payload["variant"]]
    assert (payload["asset"], payload["size"]) == (spec["asset"], spec["size"])
    assert payload["sha256"] == spec.get("sha256")
    assert payload["dry_run"] is True
    assert not pathlib.Path(payload["destination"]).exists(), "a dry run writes nothing"


@FORK
def test_doctor_reports_an_empty_home_instead_of_a_missing_lock(installed: Installed,
                                                               neutral: pathlib.Path) -> None:
    result = installed.run("doctor", "--json", cwd=neutral)
    report = json.loads(result.stdout)
    # the documented reading: 1 = broken (nothing installed yet), 2 = warnings, 0 = ready
    assert result.returncode == report["exit_code"] == 1
    assert report["status"] == "failures"
    assert report["runtime"]["pinned_tag"] == TAG
    checks = " | ".join(f"{check['id']}: {check['detail']}" for check in report["checks"])
    assert "E_RUNTIME_MISSING" not in checks, checks
    assert "run `typed-gguf init`" in checks, checks   # the honest message for an empty home


@FORK
def test_a_broken_install_names_every_path_it_searched(installed: Installed,
                                                       tmp_path: pathlib.Path) -> None:
    """Requirement 3, on the artifact: with no lock left to read, the error lists what it searched
    instead of telling a wheel user to go and be a checkout."""
    packaged = installed.site_packages / "typed_gguf" / "data" / "runtime.lock"
    packaged.unlink()
    elsewhere = tmp_path / "no-lock-anywhere"
    elsewhere.mkdir()
    result = installed.run("version", cwd=elsewhere)
    assert result.returncode == 3, result.stdout + result.stderr   # the documented code
    assert "E_RUNTIME_MISSING" in result.stderr
    assert str(packaged) in result.stderr, "the copy that is missing is not named"
    assert str(elsewhere / "runtime.lock") in result.stderr, "the last resort is not named"
    assert "run from the repository root" not in result.stderr


@FORK
def test_a_missing_override_is_reported_not_skipped(installed: Installed,
                                                    tmp_path: pathlib.Path) -> None:
    """`$TYPED_GGUF_LOCK` stays authoritative for an installed tool too: a typo is an error, never
    a quiet fallback to the packaged copy (which would leave the caller's pin looking in use)."""
    elsewhere = tmp_path / "neutral-2"
    elsewhere.mkdir()
    typo = elsewhere / "typo.lock"
    result = installed.run("version", cwd=elsewhere, TYPED_GGUF_LOCK=str(typo))
    assert result.returncode == 3, result.stdout + result.stderr
    assert "TYPED_GGUF_LOCK" in result.stderr and str(typo) in result.stderr


#: the exact panic this box produced while a sibling held the cgroup (card t_eff926f9) — the
#: reason the fixtures classify their `uv` failures instead of asserting them away
CGROUP_PANIC = ("thread 'main2' (233252) panicked at crates/uv-fs/src/locked_file.rs:205:34:\n"
                "OS can't spawn worker thread: Resource temporarily unavailable (os error 11)")


def test_a_full_cgroup_is_never_read_as_a_product_failure() -> None:
    """The fixture's classification, pinned against the real thing: `uv` panics with the cgroup's
    error, a build failure does not — and only the second one may fail this gate."""
    assert starvation(CGROUP_PANIC) == "resource temporarily unavailable"
    assert starvation("error: failed to build wheel: no such file or directory") is None
    assert starvation("") is None


def test_the_pressure_skip_is_the_one_the_conftest_counts(request: pytest.FixtureRequest) -> None:
    """A skip raised from a *fixture* has to reach the conftest's summary and its exit-status gate,
    or a starved box would look like a green suite with this whole file silently unmeasured."""
    session = request.session
    before = int(session.__dict__.get("_gg_pid_pressure_skips", 0))
    try:
        with pytest.raises(pytest.skip.Exception) as excinfo:
            pressure_skip(session, "unit probe")
        assert PID_PRESSURE_SKIP_PREFIX in str(excinfo.value)
        assert int(session.__dict__["_gg_pid_pressure_skips"]) == before + 1
    finally:
        # the probe measured the counter, it did not starve the box
        session.__dict__["_gg_pid_pressure_skips"] = before
