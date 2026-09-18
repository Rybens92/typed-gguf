"""Fidelity check for the offline proxy used by `tests/test_cli_teardown.py` (card t_97f1bc93).

The gate's `BOMB` is an `atexit` callback that raises SIGSEGV — the same place and the same signal
as the measured ICD handler. This script shows the proxy behaves like the real thing *on the
parent tree*: `cli.main(["--help"])` returns 0, the interpreter's own shutdown then runs the
callback, and the process dies with 139 instead of 0.

    python .e2e/t_97f1bc93-vulkan-teardown/proxy_fidelity.py    # exit 139 on the parent tree
"""
from __future__ import annotations

import atexit
import os
import pathlib
import signal
import sys

SOURCE_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SOURCE_ROOT))


def third_party_teardown() -> None:
    sys.stderr.write("THIRD-PARTY-TEARDOWN-RAN\n")
    sys.stderr.flush()
    os.kill(os.getpid(), signal.SIGSEGV)


atexit.register(third_party_teardown)

from ggufone import cli  # noqa: E402 - after the handler, like a dlopened driver

code = cli.main(["--help"])
sys.stderr.write(f"MAIN-RETURNED-{code}\n")
sys.stderr.flush()
raise SystemExit(code)
