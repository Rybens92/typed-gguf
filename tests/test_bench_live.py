"""A-E2 live gates: the bench suites against a real model and a real runtime.

Marked `model` (skipped unless `--run-network` is passed) so the offline gate stays model-free.
Nothing here touches the network — the marker is this repository's switch for "needs real
assets on disk" (SPEC A7).

    GGUFONE_RUNTIME_DIR=<bundle> uv run pytest -q --run-network tests/test_bench_live.py -s

The published numbers live in `docs/BENCHMARKS.md`; these tests only prove the *path* works on a
real GGUF and that the dev-set budget holds on a real vocabulary.

Card t_f46cec41 adds two `--quick` gates: the wall-clock budget of the whole preset on a CPU-only
box (`harness.QUICK_TARGET_SECONDS`, printed per suite) and the same preset on a *bigger* local
model — the operator named Occamy 1.0 / Tiel-Coder-35B-A3B, both 35B-A3B MoE files. A worker
container that cannot see the file, or whose memory cgroup cannot hold it, skips with that reason
instead of quietly dropping the gate.
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
#: the operator's "bigger model" for quick evidence (2026-09-18): local, never downloaded
OCCAMY = pathlib.Path.home() / ".hermes" / "models" / "Accio-Lab_occamy-1.0-Q4_K_L.gguf"
TIEL = pathlib.Path.home() / ".hermes" / "models" / "Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf"
BIG_MODELS = (OCCAMY, TIEL)
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


def _big_model() -> pathlib.Path:
    """A local model *bigger* than the CI smoke one, for the quick preset's real-weight shape.

    Occamy 1.0 (23 GiB, `qwen35moe`) and Tiel-Coder-35B-A3B (21 GiB) are the operator's local
    files; `GGUFONE_BENCH_MODEL_BIG` points at any other. The skip names which of the two reasons
    it was — absent, or larger than this box's memory cgroup — so a missing measurement is never
    a silent one (the host run measures the big model; see
    `docs/evidence/e2_t_f46cec41_bench_quick.md`).
    """
    explicit = os.environ.get("GGUFONE_BENCH_MODEL_BIG")
    candidates = ([pathlib.Path(explicit)] if explicit else []) + list(BIG_MODELS)
    for candidate in candidates:
        if not candidate.is_file():
            continue
        ceiling = harness.host_facts().get("cgroup_memory_bytes")
        size = candidate.stat().st_size
        if ceiling is not None and size > ceiling:
            pytest.skip(f"{candidate} is {size / 1024 ** 3:.1f} GiB and this box's memory cgroup "
                        f"caps it at {ceiling / 1024 ** 3:.1f} GiB — measure it on the host")
        return candidate
    pytest.skip("no bigger local GGUF on this box ("
                + ", ".join(str(path) for path in BIG_MODELS)
                + "); set GGUFONE_BENCH_MODEL_BIG")


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


# --------------------------------------------------------- the quick preset (card t_f46cec41)
#: the model class the card's ≤ 3 min target is calibrated to (the CI smoke GGUF); a bigger local
#: model scales the budget with its file size — the preset's cost is per-token weight traffic, so
#: the gate stays meaningful on a 4B (and conservative on a 35B MoE, whose active slice is a
#: fraction of the file) while the printed wall time stays the measurement
QUICK_TARGET_BYTES = 1 << 30
#: one `ggufone bench --quick` run is the unit the target is written for (one suite, end to end);
#: a five-suite campaign on a *shared* box (siblings run builds and mutation sweeps on the same 2
#: CPU-seconds/s quota) is allowed this multiple — the measured factor against this box's quiet
#: numbers is ~1.6×, and the campaign total is printed either way
QUICK_CAMPAIGN_FACTOR = 2.0


def quick_budget_seconds(model: pathlib.Path) -> float:
    """The card's target for *this* model: `QUICK_TARGET_SECONDS` up to `QUICK_TARGET_BYTES`."""
    return harness.QUICK_TARGET_SECONDS * max(1.0, model.stat().st_size / QUICK_TARGET_BYTES)


def quick_report(suite: str, model: pathlib.Path, **kwargs) -> dict:
    config = harness.quick_config(harness.BenchConfig(
        suite=suite, model_path=str(model), backend=harness.CPU_BACKEND, threads=2, **kwargs))
    return suites.run_suite(config, factory=suites.live_factory)


@pytest.mark.model
def test_the_quick_preset_finishes_inside_its_wall_clock_budget():
    """The card's target: `--quick` ≤ ~3 min end to end on the CPU-only container.

    Every suite runs through the same code path the CLI uses, on the same local model;
    `report["wall_ms"]` is the clock the renderer prints, and the printed per-suite numbers are
    what `docs/evidence/e2_t_f46cec41_bench_quick.md` quotes. The assertion is the card's own
    budget (`harness.QUICK_TARGET_SECONDS`) per suite, plus a documented multiple for the whole
    five-suite campaign on this shared box — never a number this file gets to lower.
    """
    model = _model()
    budget_s = quick_budget_seconds(model)
    reports: list[tuple[str, dict]] = []
    total_ms = 0.0
    for suite in harness.SUITES:
        report = quick_report(suite, model)
        assert report["quick"] is True
        assert report["truncated"] is False
        assert report["wall_ms"] > 0
        reports.append((suite, report))
        total_ms += float(report["wall_ms"])
    print(f"\nquick preset on {model.name} "
          f"(cgroup cpu.max {harness.host_facts().get('cgroup_cpu_max')}, "
          f"budget {budget_s:.0f} s/suite):")
    for suite, report in reports:
        print(f"  {suite:<11s} {report['wall_ms'] / 1000.0:7.1f} s")
    print(f"  {'total':<11s} {total_ms / 1000.0:7.1f} s "
          f"(<= {budget_s * QUICK_CAMPAIGN_FACTOR:.0f} s)")
    for suite, report in reports:
        assert float(report["wall_ms"]) <= budget_s * 1000.0, (
            f"{suite} took {float(report['wall_ms']) / 1000.0:.1f} s on {model.name}; the "
            f"preset targets {budget_s:.0f} s per suite")
    assert total_ms <= budget_s * QUICK_CAMPAIGN_FACTOR * 1000.0, (
        f"the quick campaign took {total_ms / 1000.0:.1f} s on {model.name}; "
        f"{budget_s * QUICK_CAMPAIGN_FACTOR:.0f} s is the documented bound")


@pytest.mark.model
def test_the_quick_preset_measures_a_bigger_local_model_too():
    """`--quick` on the operator's bigger MoE: shape and completion, not the 3-minute budget.

    A 35B-A3B (Occamy 1.0 / Tiel-Coder) on two CPU-seconds per second is a minutes-scale *load*,
    so this gate asserts the preset's row shapes and prints the wall time for the evidence doc
    instead of pretending the container budget applies to a model the preset cannot fit.
    """
    model = _big_model()
    report = quick_report("latency", model)
    assert report["quick"] is True
    assert report["truncated"] is False
    assert [row["tokens"] for row in report["prefill"]] == list(harness.QUICK_PREFILL_SIZES)
    assert [row["candidates"] for row in report["per_question"]] == \
        list(harness.QUICK_CANDIDATE_COUNTS)
    assert report["model_load"]["n"] == 1
    assert report["model"]["arch"], report["model"]
    print(f"\nquick latency on {model.name} ({report['model'].get('arch')}, "
          f"{report['model'].get('bytes', 0) / 1024 ** 3:.1f} GiB): "
          f"{report['wall_ms'] / 1000.0:.1f} s")
