"""Card t_97f1bc93, live half: one Vulkan bundle, a busy device, an exit code that is readable.

A **single** Vulkan bundle used to print its whole report and then die with SIGSEGV (`exit 139`)
on a device that is nearly full — the row's own process, no second bundle anywhere. The fault is
not ggufone's and not the bundle's: it is the NVIDIA ICD's own exit handler
(`libnvidia-eglcore` → `libnvidia-glvkspirv`, fault address `0x18`) running from libc's
`__run_exit_handlers`, i.e. *other people's destructors at interpreter exit* (backtrace and raws in
`.e2e/t_97f1bc93-vulkan-teardown/`). The remedy is `ggufone.cli.run`: a process that has a bundle
loaded ends itself, with the command's code and its streams already flushed.

This gate is the executable form of the card's requirement 3: the operator's command, the pinned
Vulkan bundle, a real GGUF, and device memory deliberately held by a pressure child — the exit code
must be the report's (0 or 1), never a signal, and the report must be parseable on stdout.

    VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json \\
    GGUFONE_RUNTIME_DIR=/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan \\
    GGUFONE_BENCH_MODEL=/var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf \\
      uv run --frozen pytest -q --run-network tests/test_bench_vulkan_teardown_live.py -s

It skips, with the reason, on a box without a Vulkan bundle or without a benchmarkable GGUF: this
gate is about a host shape, and a missing host shape is a skip, never a silent pass. The crash it
guards against is intermittent (roughly a third of the runs on the box that found it — the .e2e
directory carries the count), so this gate asserts the *guarantee*, not the crash: whatever the
driver does at teardown, the command's exit status is its report's.
"""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator

import pytest

import ggufone
from ggufone.bench import harness

#: The operator's model for this crash (the same one the card's raws use).
CARD_MODEL = pathlib.Path("/var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf")
QUICK_MODEL = pathlib.Path("/var/home/rybens/.cache/llama.cpp/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
SPARK = pathlib.Path("/var/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf")
#: how long the pressure child holds its device memory (the target run's budget is ~2-5 minutes)
PRESSURE_SECONDS = 900.0
#: how long the pressure child may take to get its device memory (a load, on a busy box)
PRESSURE_READY_TIMEOUT = 300.0


def vulkan_bundle() -> str:
    """The accelerator bundle this box would run `--backend vulkan` through."""
    runtimes = harness.backend_runtimes()
    for backend in ("vulkan", "cuda", "metal"):
        if backend in runtimes:
            return str(runtimes[backend])
    pytest.skip("no accelerator llama.cpp bundle on this box "
                "(set GGUFONE_RUNTIME_DIR at an extracted bundle)")


def benchmarkable_model() -> pathlib.Path:
    explicit = os.environ.get("GGUFONE_BENCH_MODEL")
    candidates = ([pathlib.Path(explicit)] if explicit else []) + [CARD_MODEL, QUICK_MODEL, SPARK]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    pytest.skip("no benchmarkable GGUF on this box (set GGUFONE_BENCH_MODEL)")


def child_env() -> dict[str, str]:
    """The CLI child's env: the caller's, plus this checkout's `src/` on `PYTHONPATH`."""
    env = dict(os.environ)
    src_root = str(pathlib.Path(ggufone.__file__).resolve().parents[1])
    parts = [part for part in env.get("PYTHONPATH", "").split(os.pathsep) if part]
    if src_root not in parts:
        env["PYTHONPATH"] = os.pathsep.join([src_root, *parts])
    return env


def device_free_mib() -> int | None:
    """Free device memory in MiB, when this box can tell (the pressure evidence, best effort)."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        out = subprocess.run([smi, "--query-gpu=memory.free", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return int(out.split()[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


PRESSURE_SOURCE = '''
"""Hold device memory on the same bundle while the measured run happens (the card's shape)."""
import pathlib, sys, time

sys.path.insert(0, {src!r})

from ggufone.engine import session as session_module


class ExactPlacement:
    """The minimal placement `fit.coerce_plan` normalizes (`harness.Placement`'s own shape)."""

    n_gpu_layers = {layers}


handle = session_module.open_model({model!r}, runtime_dir={bundle!r},
                                   fit_plan=ExactPlacement(), degrade=False)
print("PRESSURE-READY", flush=True)
time.sleep({seconds!r})
handle.close()
'''


@contextlib.contextmanager
def device_pressure(tmp_path: pathlib.Path, *, model: pathlib.Path, bundle: str,
                    layers: int = 12) -> Iterator[bool]:
    """Hold some device memory for the measured run's duration, best effort.

    A box whose device is already busy cannot always give the child the memory it asks for; that is
    not this gate's failure (the target still has to answer readably), so the child is started,
    given a bounded moment to report readiness, and killed in any case. Its log goes to a file
    rather than a pipe: the gate must not add an unclosed file object to a suite that turns
    resource warnings into errors.
    """
    script = tmp_path / "pressure.py"
    layers = int(os.environ.get("GGUFONE_LIVE_PRESSURE_LAYERS", layers))
    script.write_text(PRESSURE_SOURCE.format(src=str(pathlib.Path(ggufone.__file__).resolve()
                                                     .parents[1]),
                                             model=str(model), bundle=bundle, layers=int(layers),
                                             seconds=PRESSURE_SECONDS), encoding="utf-8")
    log_path = tmp_path / "pressure.log"
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen([sys.executable, str(script)], stdout=log,
                                 stderr=subprocess.STDOUT, text=True, env=child_env())
        try:
            deadline = time.monotonic() + PRESSURE_READY_TIMEOUT
            ready = False
            while time.monotonic() < deadline:
                ready = "PRESSURE-READY" in log_path.read_text(encoding="utf-8")
                if ready or child.poll() is not None:
                    break
                time.sleep(0.5)
            print(f"pressure ready={ready} layers={layers} free_mib={device_free_mib()}\n"
                  f"{log_path.read_text(encoding='utf-8')[-800:]}", file=sys.stderr)
            yield ready
        finally:
            child.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                child.wait(timeout=60)


@pytest.mark.model
def test_the_single_bundle_command_ends_with_its_report_on_a_busy_device(
        tmp_path: pathlib.Path) -> None:
    """The card's gate: no SIGSEGV, no silent 0 — the exit code is the report's own."""
    bundle = vulkan_bundle()
    model = benchmarkable_model()
    report_path = tmp_path / "throughput-vulkan.json"
    command = [sys.executable, "-m", "ggufone", "bench", "--suite", "throughput",
               "--model", str(model), "--backend", "vulkan", "--runs", "1", "--threads", "4",
               "--sizes", "64", "--out", str(report_path), "--json"]
    # the same command the card measured, on a device that is deliberately short of memory
    with device_pressure(tmp_path, model=model, bundle=bundle):
        completed = subprocess.run(command, capture_output=True, text=True, env=child_env(),
                                   timeout=1800)
    tail = (completed.stderr or "")[-1500:]
    print(f"exit={completed.returncode} free_mib_after={device_free_mib()}", file=sys.stderr)
    # requirement: the process ends with the CLI's own codes, never with the kernel's signal
    assert completed.returncode in (0, 1), (
        f"`--backend vulkan` exited {completed.returncode} on {bundle}\n{tail}")
    assert report_path.is_file(), f"the report was never written\n{tail}"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == harness.SCHEMA
    # the exit code *is* the report's (SPEC 2.5): never 139, never a silent 0 for a flagged report
    assert completed.returncode == (0 if report["ok"] else 1), (
        f"exit {completed.returncode} contradicts report ok={report['ok']}\n{tail}")
    # `--json` stays parseable: the report the operator reads on stdout is the file's
    assert json.loads(completed.stdout) == report
    rows = {row["backend"]: row for row in report["backends"]}
    assert "vulkan" in rows, f"no row for the bundle that ran: {sorted(rows)}\n{tail}"
    row = rows["vulkan"]
    if row.get("measured"):
        # a measured row is a row this process produced: the bundle's own device must be in it
        assert row["runtime_dir"] == bundle, row
        assert row["devices"], f"a measured row without its device evidence: {row}"
    else:
        # the pressure was real: the fit ladder refused the placement with a typed reason
        assert row.get("reason"), f"an unmeasured row must say why: {row}"
