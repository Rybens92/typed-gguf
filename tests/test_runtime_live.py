"""Live runtime tests: the real bundle and the real model (A-E1a-3, A-E1a-12).

Marked `model`: they need an installed runtime (and, for the warm-up, the pinned model), so
they are skipped unless `--run-network` is passed. They cover the code paths that only exist
with real artifacts loaded — the ctypes binding table, `llama_batch_init/free`, and the
warm-up decode that `init` performs when a model is already present.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from ggufone.registry import store
from ggufone.runtime import capability, ctypes_binding, finder, install

HOME = pathlib.Path.home()
PINNED_MODEL = HOME / ".hermes" / "models" / "Spark-X2.5-4B-Q8_0.gguf"


def installed_runtime() -> pathlib.Path:
    found = finder.find_runtime()
    if found is None:
        pytest.skip("no runtime installed (run `ggufone init` or set GGUFONE_RUNTIME_DIR)")
    return found


@pytest.mark.model
def test_load_libraries_binds_the_pinned_abi_and_round_trips_a_batch() -> None:
    runtime = ctypes_binding.load_libraries(installed_runtime())
    assert "llama_decode" in runtime.bindings
    assert "llama_memory_seq_cp" in runtime.bindings
    assert len(runtime.bindings) >= 28
    # llama_batch_init/free is the smallest call that proves the binding really works
    batch = runtime.llama.llama_batch_init(4, 0, 1)
    assert batch.n_tokens == 0
    runtime.llama.llama_batch_free(batch)
    assert ctypes_binding.loaded_runtimes() == (str(runtime.directory.resolve()),)


@pytest.mark.model
def test_backend_loader_ran_before_model_load() -> None:
    """PoC pitfall 1: without ggml_backend_load_all_from_path the model load crashes."""
    runtime = ctypes_binding.load_libraries(installed_runtime())
    assert hasattr(runtime.ggml, "ggml_backend_load_all_from_path")
    assert runtime.llama.llama_backend_init is not None  # bound and callable
    assert runtime.directory.name.startswith("b11026-")


@pytest.mark.model
@pytest.mark.skipif(not PINNED_MODEL.exists(), reason="pinned Spark GGUF not present")
def test_warmup_decodes_and_reports_milliseconds() -> None:
    runtime_dir = installed_runtime()
    capability.require_arch(runtime_dir, "spark2_5")
    ms = install.warmup(runtime_dir, PINNED_MODEL, n_ctx=128)
    assert ms > 0
    assert ms < 120_000


@pytest.mark.model
@pytest.mark.skipif(not PINNED_MODEL.exists(), reason="pinned Spark GGUF not present")
def test_warmup_through_init_records_the_number(tmp_path: pathlib.Path) -> None:
    """`init` with `warmup_model` writes warmup_ms into runtime.json (R2 / A13)."""
    lock = install.pins.load_lock()
    result = install.install("cpu", home=tmp_path / "home", lock=lock,
                             warmup_model=PINNED_MODEL, free_bytes=1 << 40)
    record = result["record"]
    assert record["warmup_ms"] and record["warmup_ms"] > 0
    assert record["warmup_error"] is None
    assert record["warmup_model"] == str(PINNED_MODEL)
    assert (tmp_path / "home" / "runtime.json").exists()
    assert json.loads((tmp_path / "home" / "runtime.json").read_text())["warmup_ms"] > 0
