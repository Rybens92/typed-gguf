"""Live runtime tests: the real bundle and the real model (A-E1a-3, A-E1a-12).

Marked `model`: they need an installed runtime (and, for the warm-up, the pinned model), so
they are skipped unless `--run-network` is passed.

Each probe runs in a **child process** (`tools/live_probe.py`). Reason: with several model
load/free cycles in one long-lived interpreter, the shared library's teardown can abort at
exit (`free(): invalid pointer`) long after every call returned correctly. Every real ggufone
command is its own process, so this is a test-harness concern, not a product one — and a C
library must never be able to kill the test runner.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

from ggufone.runtime import finder

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE = ROOT / "tools" / "live_probe.py"
HOME = pathlib.Path.home()
PINNED_MODEL = HOME / ".hermes" / "models" / "Spark-X2.5-4B-Q8_0.gguf"


def run_probe(*args: str, timeout: int = 300) -> dict:
    if finder.find_runtime() is None:
        pytest.skip("no runtime installed (run `ggufone init` or set GGUFONE_RUNTIME_DIR)")
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(PROBE), *args], capture_output=True, text=True,
        errors="replace",  # llama.cpp dumps tokenizer pieces: not always valid UTF-8
        cwd=str(ROOT), timeout=timeout, check=False)
    assert result.returncode == 0, (
        f"live_probe {args} exited {result.returncode}\n"
        f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.model
def test_bindings_load_in_a_child_process() -> None:
    payload = run_probe("bindings")
    assert payload["bindings"] >= 28
    assert payload["batch_n_tokens"] == 0          # llama_batch_init/free round-tripped
    assert payload["cached"]                        # the runtime is cached per process


@pytest.mark.model
def test_deep_probe_in_a_child_process() -> None:
    payload = run_probe("probe")
    assert payload["ok"] is True
    assert payload["missing_symbols"] == 0
    assert payload["build"] == 11026
    assert payload["fit_params_help_exit"] == 0


@pytest.mark.model
@pytest.mark.skipif(not PINNED_MODEL.exists(), reason="pinned Spark GGUF not present")
def test_warmup_decodes_the_pinned_model() -> None:
    payload = run_probe("warmup")
    assert payload["warmup_ms"] > 0


@pytest.mark.model
@pytest.mark.skipif(not PINNED_MODEL.exists(), reason="pinned Spark GGUF not present")
def test_init_records_the_warmup_number(tmp_path: pathlib.Path) -> None:
    """`init` writes warmup_ms into runtime.json when a model is present (R2 / A13)."""
    payload = run_probe("install", "--home", str(tmp_path / "home"))
    assert payload["warmup_ms"] and payload["warmup_ms"] > 0
    assert payload["warmup_error"] is None
    assert payload["build"] == 11026
    record = json.loads(pathlib.Path(payload["record_path"]).read_text())
    assert record["warmup_ms"] > 0
    assert record["libllama_sha256"] == payload["libllama_sha256"]
