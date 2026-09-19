"""Card t_57cc0179 — deterministic **fault injection** for the teardown-crash shape.

The natural defect (a Vulkan child that SIGSEGVs *after* writing a complete report) is flaky on this
box — the sibling card `t_97f1bc93` measures ~1 crash in 4 identical runs — and it needs a
memory-starved device, so a live retry cannot be produced on demand. This module makes the *shape*
deterministic without touching the code under test:

* every Python process that has this directory on `PYTHONPATH` imports it (stdlib `sitecustomize`);
* it arms an `atexit` handler that raises **SIGSEGV in this process**, and only when
  `GGUFONE_T57_FAULT` is set, the process is the documented isolated child
  (`-m ggufone bench … --out <path>` with `--backend vulkan`), and — in `retry-ok` mode — the `--out`
  path is the *first* attempt's report (no `-retry` in it);
* `atexit` runs after the CLI's `main()` returned and after the report was written, so the parent
  sees exactly the shape under test: **exit -11 with a complete `ok: true` report** on disk.

Modes: `retry-ok` = attempt 1 dies, the retry answers (the row is recovered); `always` = both
attempts die (the row is withheld and carries `W_BACKEND_CRASHED_AT_TEARDOWN`). The parent process
(`bench --backend all`) has no `--out` in its argv, so it is never faulted.
"""

from __future__ import annotations

import atexit
import os
import signal
import sys

_MODE = os.environ.get("GGUFONE_T57_FAULT", "")
_ARGS = list(sys.argv)


def _is_isolated_vulkan_child() -> bool:
    return ("--out" in _ARGS and "vulkan" in _ARGS
            and "-m" in _ARGS and "bench" in _ARGS)


def _is_first_attempt() -> bool:
    try:
        out = _ARGS[_ARGS.index("--out") + 1]
    except (ValueError, IndexError):
        return False
    return "-retry" not in os.path.basename(out)


def _raise_sigsegv() -> None:
    os.kill(os.getpid(), signal.SIGSEGV)


if _MODE in ("retry-ok", "always") and _is_isolated_vulkan_child() \
        and (_MODE == "always" or _is_first_attempt()):
    atexit.register(_raise_sigsegv)
