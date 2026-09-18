"""A-E2 live gates: the bench suites against a real model and a real runtime.

Marked `model` (skipped unless `--run-network` is passed) so the offline gate stays model-free.
Nothing here touches the network — the marker is this repository's switch for "needs real
assets on disk" (SPEC A7).

    GGUFONE_RUNTIME_DIR=<bundle> uv run pytest -q --run-network tests/test_bench_live.py -s

The published numbers live in `docs/BENCHMARKS.md`; these tests only prove the *path* works on a
real GGUF and that the dev-set budget holds on a real vocabulary.
"""
from __future__ import annotations

import os
import pathlib

import pytest

from ggufone.bench import devset, harness, suites
from ggufone.engine import session as session_module
from ggufone.runtime import finder

SPARK = pathlib.Path.home() / ".hermes" / "models" / "Spark-X2.5-4B-Q8_0.gguf"
QWEN = pathlib.Path.home() / ".cache" / "llama.cpp" / "Qwen3.5-0.8B-UD-Q4_K_XL.gguf"
MAX_ITEM_TOKENS = 200


def _runtime() -> pathlib.Path:
    env = os.environ.get("GGUFONE_RUNTIME_DIR")
    if env and (pathlib.Path(env) / "libllama.so").exists():
        return pathlib.Path(env)
    found = finder.find_runtime()
    if found:
        return found
    pytest.skip("no llama.cpp runtime on this box (set GGUFONE_RUNTIME_DIR)")


def _model() -> pathlib.Path:
    """A benchmarkable GGUF: `GGUFONE_BENCH_MODEL` first, then the two known local models.

    The env override is deliberate — a live run must not depend on `$HOME` matching the box that
    holds the models (this container runs with a scratch HOME).
    """
    explicit = os.environ.get("GGUFONE_BENCH_MODEL")
    if explicit and pathlib.Path(explicit).exists():
        return pathlib.Path(explicit)
    for candidate in (QWEN, SPARK):
        if candidate.exists():
            return candidate
    pytest.skip(f"no benchmarkable GGUF on this box ({QWEN} / {SPARK}); "
                f"set GGUFONE_BENCH_MODEL")


@pytest.mark.model
def test_the_dev_set_fits_the_token_budget_on_a_real_vocabulary():
    """A-E2-3's "≤ 200 tokens" measured with the real tokenizer, not the word proxy."""
    path = _model()
    handle = session_module.open_model(path, runtime_dir=_runtime())
    try:
        worst = 0
        for item in devset.load():
            tokens = handle.tokenize(item.state)
            worst = max(worst, len(tokens))
            assert len(tokens) <= MAX_ITEM_TOKENS, f"{item.id} tokenizes to {len(tokens)} tokens"
    finally:
        handle.close()
    assert worst > 0


@pytest.mark.model
def test_the_latency_suite_runs_against_a_real_model():
    report = suites.run_suite(
        harness.BenchConfig(suite="latency", model_path=str(_model()), runs=1,
                            prefill_sizes=(128,), candidate_counts=(2,), wave_scaling=(1, 2),
                            threads=2),
        factory=suites.live_factory)
    assert report["model_load"]["n"] == 1
    assert report["model_load"]["p50"] > 0
    assert report["prefill"][0]["tokens"] == 128
    assert report["prefill"][0]["tok_per_s"]["p50"] > 0
    assert report["per_question"][0]["prefill_reused"] is True
    assert report["warm_cache"]["prefill_reused"] is True
    assert report["model"]["arch"]


@pytest.mark.model
def test_determinism_holds_on_this_box():
    report = suites.run_suite(
        harness.BenchConfig(suite="determinism", model_path=str(_model()), backend="cpu",
                            threads=1),
        factory=suites.live_factory)
    assert report["ok"] is True, report["notes"]
    row = report["backends"][0]
    assert row["identical"] is True
    assert len(set(row["digests"])) == 1


@pytest.mark.model
def test_the_quality_suite_answers_a_few_real_dev_items():
    report = suites.run_suite(
        harness.BenchConfig(suite="quality", model_path=str(_model()), runs=1, items=6,
                            threads=2),
        factory=suites.live_factory)
    assert report["devset"]["items"] == 6
    assert report["overall"]["n"] == 6
    assert 0.0 <= report["overall"]["agreement"] <= 1.0
    for row in report["items"]:
        assert row["got"] in row["probabilities"]
        assert abs(sum(row["probabilities"].values()) - 1.0) < 1e-6
