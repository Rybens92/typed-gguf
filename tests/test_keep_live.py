"""A-E4-1..7 live: the real model stays resident, ages out, and gives way (card t_7e24cea4).

`model`-marked and gated by `--run-network`, like every live gate in this repo (nothing here touches
the network — the flag means "needs real assets"). Each gate drives the **real CLI** in a child
process against its own throwaway data home, so the pids it asserts are real pids and the operator's
`$TYPED_GGUF_HOME` is never touched.

Run (this box's recipe; the paths are whatever the shell already exports)::

    VK_DRIVER_FILES=<vulkan icd json> \
    TYPED_GGUF_RUNTIME_DIR=<$TYPED_GGUF_HOME>/runtime/b11026-linux-x64-vulkan \
      uv run --frozen pytest -q --run-network tests/test_keep_live.py -s

The 4B (Spark-X2.5-4B-Q8_0) is the model the card's numbers are about; `TYPED_GGUF_KEEP_MODEL` and
`TYPED_GGUF_KEEP_MODEL_B` override the pair (the second one is what the swap gate loads instead).
A box without a bundle or without the models is a skip with the reason, never a silent pass.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping
from typing import Any

import pytest

import typed_gguf
from typed_gguf.runtime import finder, pressure

#: The 4B of the card's measurements, and the second model the swap gate loads instead.
SPARK = pathlib.Path("/var/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf")
SECOND = pathlib.Path("/var/home/rybens/.hermes/models/Ling-3.0-tiny-Q5_K_M.gguf")
#: How long a gate waits for a host the card says is gone (the idle window, the swap's stop).
GONE_TIMEOUT = 60.0
#: How long one CLI call may take (a cold 4B load on Vulkan is tens of seconds, not minutes).
CALL_TIMEOUT = 900.0
#: What a *child* of this box prints when the pid cgroup refused it: the sandbox's pid cap is also
#: its thread cap (`libgomp` cannot spawn a worker), so a starved box shows up inside the loader
#: with exit 1. `tests/test_wheel_install.py` classifies the same tokens for `uv`; a starved box
#: is not a product finding, so these gates skip by name instead of failing.
STARVATION_TOKENS = ("thread creation failed", "resource temporarily unavailable", "os error 11",
                     "cannot allocate memory")

pytestmark = [pytest.mark.network, pytest.mark.model, pytest.mark.needs_fork]


def _starvation(blob: str) -> str | None:
    """The cgroup token in a child's streams, or None when the child failed for its own reasons."""
    lowered = blob.lower()
    return next((token for token in STARVATION_TOKENS if token in lowered), None)


def _child_ok(proc: subprocess.CompletedProcess, what: str) -> None:
    """Assert a CLI child answered — unless the pid cgroup, not the product, refused it."""
    if proc.returncode == 0:
        return
    blob = (proc.stderr or "") + (proc.stdout or "")
    if (token := _starvation(blob)) is not None:
        pytest.skip(f"the pid cgroup refused this child ({token}{pressure.pressure_note()}): "
                    f"{what} — the box, not the product; re-run with a lower pids.current")
    raise AssertionError(f"{what} failed (exit {proc.returncode}): {blob[-2000:]}")


# ------------------------------------------------------------------ the box
def _runtime_dir() -> pathlib.Path:
    """The bundle this box would load through (env first: the box carries more than one)."""
    env = os.environ.get("TYPED_GGUF_RUNTIME_DIR")
    if env and (pathlib.Path(env) / "libllama.so").exists():
        return pathlib.Path(env)
    found = finder.find_runtime()
    if found:
        return pathlib.Path(found)
    pytest.skip("no llama.cpp runtime on this box (set TYPED_GGUF_RUNTIME_DIR)")


def _model(which: str = "a") -> pathlib.Path:
    """The model a gate runs: the 4B by default, the second model for the swap."""
    variable = "TYPED_GGUF_KEEP_MODEL" if which == "a" else "TYPED_GGUF_KEEP_MODEL_B"
    env = os.environ.get(variable)
    fallback = SPARK if which == "a" else SECOND
    for candidate in ([pathlib.Path(env)] if env else []) + [fallback]:
        if candidate.is_file():
            return candidate
    pytest.skip(f"no {variable} on this box (set {variable} at a GGUF file)")


def _device_free_mib() -> int | None:
    """Free device memory in MiB, best effort: the driver number the card asks for."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    proc = subprocess.run([smi, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True)
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    with contextlib.suppress(ValueError):
        return int(proc.stdout.splitlines()[0].strip())
    return None


# ------------------------------------------------------------------ the child world
def _child_env(home: pathlib.Path, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The CLI child's env: this checkout's `src/`, the gate's own home, the box's bundle."""
    env = dict(os.environ)
    env["TYPED_GGUF_HOME"] = str(home)
    env["TYPED_GGUF_RUNTIME_DIR"] = str(_runtime_dir())
    env.pop("TYPED_GGUF_KEEP_ALIVE", None)      # the chain is exercised by name, never inherited
    src_root = str(pathlib.Path(typed_gguf.__file__).resolve().parents[1])
    parts = [part for part in env.get("PYTHONPATH", "").split(os.pathsep) if part]
    if src_root not in parts:
        env["PYTHONPATH"] = os.pathsep.join([src_root, *parts])
    env.update(extra or {})
    return env


def _cli(home: pathlib.Path, *args: str, env: Mapping[str, str] | None = None,
         timeout: float = CALL_TIMEOUT) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "typed_gguf", *args], env=_child_env(home, env),
                          capture_output=True, text=True, timeout=timeout)


def _questions(model: pathlib.Path, *, threads: int | None = None) -> dict[str, Any]:
    options: dict[str, Any] = {} if threads is None else {"threads": threads}
    return {
        "state": "The billing dashboard is blank for every user after login since 09:12.",
        "model": str(model),
        "questions": {"area": {"type": "choice", "instructions": "Which team owns this?",
                               "criteria": {"billing": "payments, invoices, refunds",
                                            "technical": "api and infrastructure",
                                            "sales": "contracts and pricing"}}},
        "options": options,
    }


@dataclasses.dataclass
class Run:
    """One `run` call: the response, the wall clock it cost, and the child it was."""

    body: dict[str, Any]
    wall_s: float
    proc: subprocess.CompletedProcess

    @property
    def keep(self) -> dict[str, Any]:
        """`engine.keep`: who answered this call, and the host's own numbers when one did."""
        return dict(((self.body.get("engine") or {}).get("keep")) or {})

    @property
    def timings(self) -> dict[str, Any]:
        return dict(self.body.get("timings") or {})

    @property
    def load_ms(self) -> float:
        """The load *this call paid* (`timings.model_load_ms`), not the host's one-time load."""
        return float(self.timings.get("model_load_ms") or 0.0)


def _run(home: pathlib.Path, model: pathlib.Path, *, tag: str, keep_alive: str | None = "10m",
         threads: int | None = None, env: Mapping[str, str] | None = None) -> Run:
    """One `run --questions … --keep-alive …`, timed, with its response read back."""
    root = home.parent / "q"
    root.mkdir(parents=True, exist_ok=True)
    questions = root / f"{tag}.json"
    out = root / f"{tag}.out.json"
    questions.write_text(json.dumps(_questions(model, threads=threads)), encoding="utf-8")
    argv = ["run", "--questions", str(questions), "--out", str(out)]
    if keep_alive is not None:
        argv += ["--keep-alive", str(keep_alive)]
    started = time.monotonic()
    proc = _cli(home, *argv, env=env)
    wall = time.monotonic() - started
    _child_ok(proc, f"`run` for {tag}")
    return Run(json.loads(out.read_text(encoding="utf-8")), wall, proc)


def _status(home: pathlib.Path) -> dict[str, Any]:
    """`keep status --json`: the ledger's own claim about the host."""
    proc = _cli(home, "keep", "status", "--json", timeout=120.0)
    _child_ok(proc, "`keep status`")
    return json.loads(proc.stdout)


def _stop(home: pathlib.Path) -> dict[str, Any]:
    """`keep stop --json`, tolerated when it has to kill something this test killed first."""
    proc = _cli(home, "keep", "stop", "--json", timeout=120.0)
    _child_ok(proc, "`keep stop`")
    return json.loads(proc.stdout)


def _pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def _wait_pid_gone(pid: Any, *, timeout: float = GONE_TIMEOUT) -> float:
    """Seconds it took for the pid to go away; raises when it outlives the budget."""
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if not _pid_alive(pid):
            return time.monotonic() - started
        time.sleep(0.2)
    raise AssertionError(f"pid {pid} is still alive after {timeout}s")


def _keep_dir(home: pathlib.Path) -> pathlib.Path:
    return home / "keep"


@pytest.fixture()
def keep_home(tmp_path: pathlib.Path) -> Iterator[pathlib.Path]:
    """A throwaway data home, and no resident host left behind whatever a gate asserted."""
    home = tmp_path / "h"
    home.mkdir(parents=True, exist_ok=True)
    try:
        yield home
    finally:
        with contextlib.suppress(Exception):
            _stop(home)
    assert _status(home)["state"] == "stopped", "a gate left a host resident"


# ------------------------------------------------------------------ A-E4-1
def test_the_second_call_skips_the_model_load(keep_home: pathlib.Path) -> None:
    """Cold pays the load, warm does not: same host, `timings.model_load_ms` 0 the second time."""
    model = _model()
    cold = _run(keep_home, model, tag="cold", keep_alive="10m")
    warm = _run(keep_home, model, tag="warm", keep_alive="10m")
    assert cold.keep["served_by"] == "host" and warm.keep["served_by"] == "host"
    assert warm.keep["pid"] == cold.keep["pid"], "the second call must reuse the resident host"
    assert cold.load_ms > 100.0, f"the cold call reported no load ({cold.load_ms} ms)"
    assert warm.load_ms == 0.0, f"the warm call paid {warm.load_ms} ms of load"
    # the host remembers the one load it paid, and says so on the warm answer
    assert warm.keep["model_load_ms"] == cold.keep["model_load_ms"] > 0
    assert warm.wall_s < cold.wall_s, (f"warm {warm.wall_s:.2f}s vs cold {cold.wall_s:.2f}s")
    saved = cold.wall_s - warm.wall_s
    assert saved > 0.3 * cold.load_ms / 1000.0, (
        f"the wall-clock saving ({saved:.2f}s) is not the load ({cold.load_ms:.0f} ms)")
    assert warm.keep["requests"] >= 2
    print(f"\nA-E4-1 cold {cold.wall_s:.2f}s (load {cold.load_ms:.0f} ms) → "
          f"warm {warm.wall_s:.2f}s (load {warm.load_ms:.0f} ms), pid {warm.keep['pid']}")


# ------------------------------------------------------------------ A-E4-2
def test_idle_unload_frees_the_host_and_the_next_call_is_cold(keep_home: pathlib.Path) -> None:
    """`--keep-alive 5s`: the host exits itself, the device comes back, the next call is cold."""
    model = _model()
    free_before = _device_free_mib()
    first = _run(keep_home, model, tag="idle-cold", keep_alive="5s")
    pid = first.keep["pid"]
    report = _status(keep_home)
    assert report["state"] == "running" and report["pid"] == pid
    assert 0 < report["idle_left_s"] <= 5.0
    loaded_free = _device_free_mib()
    gone_after = _wait_pid_gone(pid)
    assert gone_after >= 2.0, f"the host exited after {gone_after:.1f}s, before its window"
    assert not _keep_dir(keep_home).joinpath(f"{report['key_digest']}.sock").exists()
    time.sleep(1.0)
    free_after = _device_free_mib()
    second = _run(keep_home, model, tag="idle-next", keep_alive="5s")
    assert second.keep["pid"] != pid, "the aged-out host answered a new call"
    assert second.load_ms > 0.0, "the call after an idle unload must pay a cold load"
    print(f"\nA-E4-2 idled out {gone_after:.1f}s after the window; device free {free_before} → "
          f"{loaded_free} (loaded) → {free_after} MiB (unloaded)")


# ------------------------------------------------------------------ A-E4-3
def test_a_model_switch_stops_the_old_host_before_the_new_one_loads(
        keep_home: pathlib.Path) -> None:
    """A → B → A: one host at a time, the old pid gone before the new model loads."""
    first_model, second_model = _model("a"), _model("b")
    a1 = _run(keep_home, first_model, tag="a1")
    assert _pid_alive(a1.keep["pid"])
    b1 = _run(keep_home, second_model, tag="b1")
    assert b1.keep["pid"] != a1.keep["pid"]
    assert not _pid_alive(a1.keep["pid"]), "B loaded while A was still resident"
    assert b1.load_ms > 0.0
    report = _status(keep_home)
    assert report["state"] == "running" and report["pid"] == b1.keep["pid"]
    assert report["model_path"] == str(second_model)
    assert len(list(_keep_dir(keep_home).glob("*.sock"))) == 1, "two hosts left two sockets"
    a2 = _run(keep_home, first_model, tag="a2")
    assert not _pid_alive(b1.keep["pid"]), "A reloaded while B was still resident"
    assert a2.keep["pid"] not in (a1.keep["pid"], b1.keep["pid"])
    assert a2.body["model"] == a1.body["model"]
    print(f"\nA-E4-3 pids: A {a1.keep['pid']} → B {b1.keep['pid']} → A {a2.keep['pid']}")


# ------------------------------------------------------------------ A-E4-5
def test_a_placement_change_swaps_the_host(keep_home: pathlib.Path) -> None:
    """The key includes the placement options: same model, different `--threads` ⇒ a swap."""
    model = _model()
    four = _run(keep_home, model, tag="t4", threads=4)
    eight = _run(keep_home, model, tag="t8", threads=8)
    assert four.keep["key"]["threads"] == 4 and eight.keep["key"]["threads"] == 8
    assert eight.keep["key_digest"] != four.keep["key_digest"]
    assert eight.keep["pid"] != four.keep["pid"]
    assert not _pid_alive(four.keep["pid"])
    assert _status(keep_home)["pid"] == eight.keep["pid"]


# ------------------------------------------------------------------ A-E4-4
def test_a_killed_client_leaves_a_host_the_next_call_reuses(keep_home: pathlib.Path) -> None:
    """A `kill -9` mid-request: the host survives (that is the feature), the ledger stays honest."""
    model = _model()
    seed = _run(keep_home, model, tag="seed")
    pid = seed.keep["pid"]
    root = keep_home.parent / "q"
    questions = root / "victim.json"
    questions.write_text(json.dumps(_questions(model)), encoding="utf-8")
    victim = subprocess.Popen([sys.executable, "-m", "typed_gguf", "run", "--questions",
                               str(questions), "--keep-alive", "10m"],
                              env=_child_env(keep_home), stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    time.sleep(1.0)                     # let it reach the host and start deciding
    victim.send_signal(signal.SIGKILL)
    victim.wait(timeout=30.0)
    assert _pid_alive(pid), "the host must outlive the client it answered (that is the feature)"
    report = _status(keep_home)
    # the host may still be finishing the abandoned request, and that is `unresponsive`, not broken
    assert report["state"] in ("running", "unresponsive"), report
    assert report["pid"] == pid, "the ledger still points at the host, not at debris"
    after = _run(keep_home, model, tag="after-kill")
    assert after.keep["pid"] == pid and after.load_ms == 0.0
    assert after.keep["fallback"] is None
    assert _status(keep_home)["state"] == "running"


def test_a_stale_socket_is_cleaned_up_on_the_next_call(keep_home: pathlib.Path) -> None:
    """A host killed *hard* leaves a socket and a record: the ledger says `stale`, the next call
    cleans both up and still answers (cold)."""
    model = _model()
    seed = _run(keep_home, model, tag="seed-hard")
    pid = seed.keep["pid"]
    os.kill(int(pid), signal.SIGKILL)
    _wait_pid_gone(pid)
    report = _status(keep_home)
    assert report["state"] == "stale", report
    socket = pathlib.Path(report["socket"])
    assert socket.exists(), "the gate means to test debris, and a kill -9 leaves its socket"
    recovered = _run(keep_home, model, tag="recovered")
    assert recovered.keep["pid"] != pid
    assert recovered.load_ms > 0.0, "a fresh host means a fresh load — and an answer"
    assert recovered.keep["fallback"] is None
    assert _status(keep_home)["state"] == "running"


# ------------------------------------------------------------------ A-E4-6
def test_keep_alive_zero_answers_inline_and_leaves_nothing(keep_home: pathlib.Path) -> None:
    """`--keep-alive 0` — the flag and the env spelling — is the pre-E4 call: no host at all.

    The response is the old one verbatim, `engine.keep` included in its absence: turning the host
    off means the keep path never ran, and a reader of that response has nothing to read into.
    """
    model = _model()
    off = _run(keep_home, model, tag="off", keep_alive="0")
    assert off.keep == {}, "keep-alive 0 reproduces the pre-E4 response verbatim"
    assert off.load_ms > 0.0, "the inline call pays its own load"
    assert _status(keep_home)["state"] == "stopped"
    assert not list(_keep_dir(keep_home).glob("*.sock"))
    env_off = _run(keep_home, model, tag="env-off", keep_alive=None,
                   env={"TYPED_GGUF_KEEP_ALIVE": "0"})
    assert env_off.keep == {}
    assert _status(keep_home)["state"] == "stopped"
    # and the env is not merely ignored: 5s in the environment does start a host
    env_on = _run(keep_home, model, tag="env-on", keep_alive=None,
                  env={"TYPED_GGUF_KEEP_ALIVE": "5s"})
    assert env_on.keep["served_by"] == "host" and env_on.keep["keep_alive_s"] == 5.0
