"""F1 (card t_8dab8b3a): the fake-OOM fixture against the gate production resolves *before* the OOM.

The workflow's step `Engine smoke — the placement retry answers a typed row, never E_INTERNAL`
builds `tools/fixtures/fit_oom_bundle.c` and runs `bench` on it. On the first live matrix run
(36159785190, job 108153215436) that step never reached the OOM path: production resolves the
bundle's CPU device before the ladder for every `cpu` row (card t_55de5779), the fixture had no
`ggml_backend_dev_by_name`, and the row was refused with

    E_RUNTIME_SYMBOLS: the bundle at /tmp/fake-bundle cannot name its CPU device
    (ggml_backend_dev_by_name('CPU') is missing); a CPU-pinned load cannot be honoured

These gates are the unit-level replay of that: the fixture, compiled here with `cc`, through
production's own dlopen path (`ctypes_binding.load_libraries` + `cpu_device`) and then through the
whole world the CI step drives (`tools/fit_oom_probe.py`, run as a subprocess because it spawns
one). The *negative* control compiles the same source with `-DTYPED_GGUF_FIXTURE_NO_CPU_DEVICE`
(the pre-fix shape) so the refusal stays reproducible.

No compiler on this box -> skipped loudly (the live step is the acceptance either way).
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from typed_gguf.runtime import ctypes_binding

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tools" / "fixtures" / "fit_oom_bundle.c"
PROBE = ROOT / "tools" / "fit_oom_probe.py"
CC = shutil.which("cc")
pytestmark = pytest.mark.skipif(
    CC is None, reason="no `cc` in this environment: the C fixture cannot be built (the live "
                       "matrix step is where this is proven)")


def build_bundle(tmp_path: pathlib.Path, *, cpu_device: bool = True) -> pathlib.Path:
    """The two shared libraries the workflow's step compiles from the fixture."""
    rt = tmp_path / "fake-bundle"
    rt.mkdir(parents=True, exist_ok=True)
    flags = [] if cpu_device else ["-DTYPED_GGUF_FIXTURE_NO_CPU_DEVICE"]
    for name in ("libllama.so", "libggml.so"):
        subprocess.run([str(CC), "-shared", "-fPIC", "-O1", *flags,
                        "-o", str(rt / name), str(FIXTURE)],  # noqa: S603
                       check=True, capture_output=True)
    return rt


def test_the_fixture_can_name_its_cpu_device(tmp_path: pathlib.Path) -> None:
    """The gate the first live run tripped: a `cpu`-pinned load resolves this before the ladder."""
    rt = build_bundle(tmp_path)
    runtime = ctypes_binding.load_libraries(rt)
    assert ctypes_binding.cpu_device(runtime) is not None, (
        "the fixture cannot name its CPU device: production refuses it with E_RUNTIME_SYMBOLS "
        "before the OOM ladder, which is exactly what job 108153215436 proved")


def test_the_pre_fix_shape_is_the_red_control(tmp_path: pathlib.Path) -> None:
    """Same source, one symbol removed: the refusal is what the gate is for, not a story."""
    rt = build_bundle(tmp_path / "red", cpu_device=False)
    runtime = ctypes_binding.load_libraries(rt)
    assert ctypes_binding.cpu_device(runtime) is None
    assert not hasattr(runtime.ggml, "ggml_backend_dev_by_name")


@pytest.mark.needs_fork
def test_the_oom_world_answers_the_ladder_the_step_asserts(tmp_path: pathlib.Path) -> None:
    """The probe world of the same step: `E_BACKEND_OOM` over the three-rung ladder."""
    rt = build_bundle(tmp_path)
    gguf = tmp_path / "synthetic.gguf"
    receipt_path = tmp_path / "oom-probe.json"
    subprocess.run([sys.executable, str(PROBE), "--make-gguf", str(gguf)],  # noqa: S603
                   check=True, cwd=str(ROOT), capture_output=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("TYPED_GGUF_")}
    env["TYPED_GGUF_FAKE_OOM_ALL"] = "1"
    env["TYPED_GGUF_HOME"] = str(tmp_path / "home")
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(PROBE), "--model", str(gguf), "--runtime", str(rt),
         "--json", str(receipt_path)],
        cwd=str(ROOT), capture_output=True, env=env, timeout=300)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")[-2000:]
    assert receipt["error"]["code"] == "E_BACKEND_OOM", receipt["error"]
    assert "tried 3 placement(s) down to CPU-only" in receipt["error"]["message"], receipt["error"]
