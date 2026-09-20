"""The net-off flag's subject is the *network*; local IPC is not the network (card t_a4ebcd36).

`TYPED_GGUF_TEST_BLOCK_NET=1` is the shape `.github/workflows/ci.yml` runs the whole offline suite
in (and what `tools/e1c_offline_gate.py` was written for): no decision path may reach the wire. The
hook used to replace `socket.socket` *wholesale* — every address family — while the E4 keep host is
built on `AF_UNIX`, so the commit that added the keep feature turned the repository's own CI step
red: the two files that bind a unix socket in-process answered **24 failed / 69 passed / 11 errors**
on the head before this fix, every one of them the hook's `AssertionError`.

The flag now forbids the families that can leave the box (`AF_INET`/`AF_INET6`, and the default
family, which *is* `AF_INET`) and leaves local IPC alone. Both halves are pinned here: the keep
gates pass with the flag on, and the download paths the flag exists for still cannot reach a
socket with it on.

The last pin drives the shape the CI step runs (a child pytest session with the flag set), because
"the keep gates are green under the flag" is a claim about a *run*, not about a fixture.
"""
from __future__ import annotations

import os
import pathlib
import re
import socket
import subprocess
import sys

import pytest

from tests import conftest as suite_environment
from typed_gguf.errors import DownloadError
from typed_gguf.registry import hf

#: The flag under test — the one `.github/workflows/ci.yml` sets on the whole offline suite.
FLAG = "TYPED_GGUF_TEST_BLOCK_NET"
#: The four files the CI's offline step failed on at the head (the B1 reproduction).
KEEP_GATES = ("tests/test_keep.py", "tests/test_keep_host.py", "tests/test_keep_client.py",
              "tests/test_keep_cli.py")


def test_the_block_forbids_the_network_families_and_their_helpers(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Every way a decision path could reach the wire raises, and the message names the flag."""
    suite_environment.block_network(monkeypatch)
    for family in (socket.AF_INET, socket.AF_INET6):
        with pytest.raises(AssertionError) as caught:
            socket.socket(family, socket.SOCK_STREAM)       # noqa: S603 - it must not be built
        assert FLAG in str(caught.value)
    # `socket.socket()` defaults to `AF_INET` (`socket.py`: `if family == -1: family = AF_INET`),
    # so refusing the default is what keeps the gate from having a one-argument hole
    with pytest.raises(AssertionError):
        socket.socket()                                     # noqa: S603
    with pytest.raises(AssertionError):
        socket.create_connection(("huggingface.co", 443))
    with pytest.raises(AssertionError):
        socket.getaddrinfo("huggingface.co", 443)


def test_the_block_leaves_local_ipc_alone(keep_home: pathlib.Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """`AF_UNIX` is a file, not a network: the block must not touch it (B1's whole subject)."""
    suite_environment.block_network(monkeypatch)
    path = keep_home / "probe.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(path))
        listener.listen(4)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(2.0)
            client.connect(str(path))
            peer, _ = listener.accept()
            with peer:
                client.sendall(b"local ipc\n")
                assert peer.recv(64) == b"local ipc\n"
    finally:
        listener.close()


def test_a_pull_probe_still_fails_with_the_block_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """The flag must keep meaning what it was written for: the download path cannot reach a socket.

    `hf.model_info(offline=False)` is the product's own pull path (tree API -> `urlopen` ->
    `socket.create_connection`). With the block installed it has to die at the socket layer and come
    back as the product's `E_DOWNLOAD_FAILED`, *before* anything is asked of the network: the probe
    never resolves a name, because the connection factory raises first.
    """
    suite_environment.block_network(monkeypatch)
    with pytest.raises(DownloadError) as caught:
        hf.model_info(hf.DEFAULT_REPO, offline=False)
    assert "network disabled" in str(caught.value)
    assert hf.HF_HOST in str(caught.value) or hf.DEFAULT_REPO in str(caught.value)


@pytest.mark.needs_fork
def test_the_keep_gates_run_green_with_the_net_block_on() -> None:
    """The CI shape, re-run: the four keep files under the flag must come back green.

    Red-first at the head before this fix: `24 failed, 69 passed, 11 errors` (the reproduction the
    re-gate recorded). The child session is the *committed workflow's* shape — same flag, same
    files, cwd, and interpreter — so a regression here is a red CI step, not a hypothetical.
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH",
                                                                         "TYPED_GGUF_HOME")}
    env[FLAG] = "1"
    command = [sys.executable, "-m", "pytest", "-q", *KEEP_GATES]
    done = subprocess.run(command, cwd=root, env=env, capture_output=True,  # noqa: S603
                          encoding="utf-8", errors="replace", timeout=900, check=False)
    blob = f"{done.stdout}\n{done.stderr}"
    tail = done.stdout.strip().splitlines()[-1] if done.stdout.strip() else ""
    if suite_environment.PID_PRESSURE_SKIP_PREFIX in blob:
        pytest.skip(f"the box is at its pid cap, so the child session could not measure the keep "
                    f"gates: {tail} — the box, not the product")
    assert done.returncode == 0, (
        f"`{FLAG}=1 pytest -q {' '.join(KEEP_GATES)}` must be green (the committed CI step runs "
        f"exactly this): exit {done.returncode}, {tail}\n{blob[-3000:]}")
    assert re.search(r"\d+ passed", tail), tail
    assert not re.search(r"\d+ (failed|error)", tail), tail
