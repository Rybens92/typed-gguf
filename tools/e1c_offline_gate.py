#!/usr/bin/env python3
"""A-E1c-10: run the whole E1c surface with the network switched off.

Every decision path (chain resolution, template rendering, fit planning, the engine) must work
with no network at all. This gate runs the E1c tests — offline *and* the live model/runtime
tests — with `TYPED_GGUF_TEST_BLOCK_NET=1`, which makes `tests/conftest.py` build `AF_INET`/
`AF_INET6` sockets through a class that raises and replace `socket.create_connection` /
`socket.getaddrinfo` with a function that raises. Local IPC (`AF_UNIX`) is not the network and
stays available: the warm keep host answers on one.

Usage::

    TYPED_GGUF_RUNTIME_DIR=<bundle> uv run python tools/e1c_offline_gate.py

Exit code 0 only when every collected test passed (skips are allowed and reported).
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUITE = ("tests/test_templates.py", "tests/test_fit.py", "tests/test_fit_live.py",
         "tests/test_cli_e1c.py", "tests/test_e1c_mutation_pins.py",
         "tests/test_engine_fork.py", "tests/test_ctypes_binding.py",
         "tests/test_cli.py", "tests/test_no_finetune.py")
SUMMARY = re.compile(r"(\d+) (passed|failed|skipped|error)")


def main(argv: list[str]) -> int:
    environment = {**os.environ, "TYPED_GGUF_TEST_BLOCK_NET": "1",
                   "PYTHONPATH": str(ROOT / "src")}
    command = [sys.executable, "-m", "pytest", "-q", *SUITE, *argv]
    print("$ TYPED_GGUF_TEST_BLOCK_NET=1", " ".join(command), flush=True)
    done = subprocess.run(command, cwd=ROOT, env=environment,  # noqa: S603
                          capture_output=True, check=False,
                          # Live model output is *byte* text: a detokenized piece can be an
                          # invalid UTF-8 continuation, so decode leniently instead of letting
                          # `text=True` raise after the child already exited.
                          encoding="utf-8", errors="replace")
    tail = done.stdout.strip().splitlines()[-1] if done.stdout.strip() else ""
    print(done.stdout[-2000:])
    if done.stderr.strip():
        print(done.stderr[-2000:], file=sys.stderr)
    counts = {kind: int(number) for number, kind in SUMMARY.findall(tail)}
    print(f"\nnetwork-disabled run: {counts} (exit {done.returncode})")
    if done.returncode != 0 or counts.get("failed") or counts.get("error"):
        return 1
    if not counts.get("passed"):
        print("no test ran — the gate would be vacuous", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
