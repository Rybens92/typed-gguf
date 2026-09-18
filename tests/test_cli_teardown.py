"""The CLI's exit status belongs to the CLI, not to a third-party destructor (card t_97f1bc93).

Measured on the operator's box: a **single** Vulkan bundle prints its whole report and then dies
with SIGSEGV — `exit 139`, the row withheld by the isolation layer because the code and the report
contradict each other. The fault is not in ggufone's frames and not in the bundle's: it is the
NVIDIA ICD's own exit handler (`libnvidia-eglcore` → `libnvidia-glvkspirv`, fault address `0x18`)
running from libc's `__run_exit_handlers` — i.e. *other people's destructors at interpreter exit*,
the same class `runtime.isolated` already refuses to trust for probes ("it dies alone").
Raw material: `.e2e/t_97f1bc93-vulkan-teardown/` and
`docs/evidence/e2_fix_t_97f1bc93_vulkan_teardown.md`.

This file is the offline half: a child process installs a *faithful proxy* for that handler — an
`atexit` callback that raises SIGSEGV, exactly what the ICD's does — and then runs the CLI. The
process must end with the CLI's code, with its streams already flushed, and the proxy must never
run; a process that never loaded a bundle keeps the interpreter's normal shutdown.
"""
from __future__ import annotations

import importlib
import pathlib
import subprocess
import sys

import ggufone
from ggufone import cli

ROOT = pathlib.Path(__file__).resolve().parents[1]
#: the checkout's `src/`: the child processes import the working tree, not an installed copy
SOURCE_ROOT = str(pathlib.Path(ggufone.__file__).resolve().parents[1])

#: What the child counts as "this process dlopened a bundle": the documented probe, stubbed, so the
#: offline gate needs no bundle, no device and no driver.
LOADED_BUNDLE = (
    "from ggufone.runtime import ctypes_binding\n"
    "ctypes_binding.loaded_runtimes = lambda: ('/fake/bundle',)\n"
)

#: The proxy for the measured crash: a third-party exit handler that raises SIGSEGV.
BOMB = """
import atexit, os, signal

def _third_party_teardown():
    sys.stderr.write("THIRD-PARTY-TEARDOWN-RAN\\n")
    sys.stderr.flush()
    os.kill(os.getpid(), signal.SIGSEGV)
atexit.register(_third_party_teardown)
"""

PREAMBLE = f"""
import sys
sys.path.insert(0, {SOURCE_ROOT!r})
{BOMB}
"""
#: The same child without the proxy: for the branch that *should* keep the normal shutdown, a
#: registered bomb would only prove that the interpreter's exit path runs (which is the point of
#: the other tests) — not what `run` does.
PLAIN_PREAMBLE = f"""
import sys
sys.path.insert(0, {SOURCE_ROOT!r})
"""


def run_child(source: str) -> subprocess.CompletedProcess[str]:
    """Run one child process with this checkout on `sys.path`."""
    return subprocess.run([sys.executable, "-c", source], capture_output=True, text=True,
                          timeout=120)


def test_the_entry_point_ends_the_process_before_third_party_teardown() -> None:
    """`cli.run` never returns: the entry point's code is the process's, the handler never runs."""
    child = PREAMBLE + LOADED_BUNDLE + """
from ggufone import cli
cli.run(["--help"])
sys.stderr.write("ENTRY-POINT-RETURNED\\n")
sys.stderr.flush()
"""
    completed = run_child(child)
    assert completed.returncode == 0, (
        f"the entry point exited {completed.returncode}, not the CLI's own code\n"
        f"{completed.stderr[-600:]}")
    assert "ENTRY-POINT-RETURNED" not in completed.stderr, (
        "the entry point returned instead of ending the process")
    assert "THIRD-PARTY-TEARDOWN-RAN" not in completed.stderr, (
        "a third-party exit handler ran: the process's code is not the CLI's")


def test_the_entry_point_hands_the_cli_code_to_the_shell() -> None:
    """The shell sees the code `main` produced, not a signal and not a silent 0."""
    child = PREAMBLE + LOADED_BUNDLE + """
from ggufone import cli
cli.run(["definitely-not-a-command"])
"""
    completed = run_child(child)
    assert completed.returncode == 2, (
        f"expected the CLI's own 2 (unknown command), got {completed.returncode}\n"
        f"{completed.stderr[-600:]}")


def test_the_streams_are_flushed_before_the_process_ends() -> None:
    """A piped stdout is block-buffered: the usage text survives only if it was flushed."""
    child = PREAMBLE + LOADED_BUNDLE + """
from ggufone import cli
cli.run(["--help"])
"""
    completed = run_child(child)
    assert completed.returncode == 0
    assert "usage: ggufone <command> [options]" in completed.stdout, (
        "the entry point ended without flushing stdout: the answer is not readable\n"
        f"stdout: {completed.stdout!r}")


def test_a_process_that_never_loaded_a_bundle_keeps_the_normal_shutdown() -> None:
    """Only the process that dlopened a bundle ends deliberately; everything else is untouched."""
    child = PLAIN_PREAMBLE + """
from ggufone import cli
try:
    cli.run(["--help"])
except SystemExit as exc:
    sys.stderr.write(f"SYSTEM-EXIT-{exc.code}\\n")
    sys.stderr.flush()
"""
    completed = run_child(child)
    assert completed.returncode == 0, completed.stderr[-600:]
    assert "SYSTEM-EXIT-0" in completed.stderr, (
        "without a loaded bundle the entry point must raise SystemExit like any other CLI\n"
        f"{completed.stderr[-600:]}")
    assert "usage: ggufone <command> [options]" in completed.stdout


def test_the_module_entry_point_runs_the_process_entry_point() -> None:
    """`python -m ggufone` — the command every live gate and the isolation child uses."""
    child = PREAMBLE + LOADED_BUNDLE + """
import runpy, sys
sys.argv = ["ggufone", "--help"]
runpy.run_module("ggufone", run_name="__main__", alter_sys=True)
sys.stderr.write("ENTRY-POINT-RETURNED\\n")
sys.stderr.flush()
"""
    completed = run_child(child)
    assert completed.returncode == 0, (
        f"`python -m ggufone` exited {completed.returncode}\n{completed.stderr[-600:]}")
    assert "ENTRY-POINT-RETURNED" not in completed.stderr
    assert "THIRD-PARTY-TEARDOWN-RAN" not in completed.stderr


def test_the_console_script_points_at_the_process_entry_point() -> None:
    """`ggufone …` in a shell gets the same story as `python -m ggufone …`."""
    import tomllib

    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    target = payload["project"]["scripts"]["ggufone"]
    assert target == "ggufone.cli:run", (
        f"the console script points at {target}: it must use the entry point that ends the "
        f"process itself (ggufone.cli:run)")
    module_name, _, attribute = target.partition(":")
    assert callable(getattr(importlib.import_module(module_name), attribute))


def test_main_still_returns_the_code_in_process() -> None:
    """Every existing caller (`tests/`, the API) keeps a pure function — never a process exit."""
    assert cli.main(["--help"]) == 0
    assert cli.main(["definitely-not-a-command"]) == 2
