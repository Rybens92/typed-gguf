"""E2 FIX (card t_31b3943a): the benchmark's placement must survive the loader's degrade ladder.

`ggufone bench` names its placement explicitly (``--gpu-layers``, the minimal ``harness.Placement``)
instead of consuming a fit plan, and ``session.open_model`` builds the degradation ladder *before*
its first load attempt. On the parent tree that combination died with

    AttributeError: 'Placement' object has no attribute 'kv_type'      ->  E_INTERNAL, exit 4

on a Vulkan host and on the pinned CPU bundle alike (the ladder is built before any device is
touched, so no GPU is needed to reach it — see
``docs/evidence/e2_fix_t_31b3943a_bench_placement.md``).

What this file pins:

* the ladder accepts a *placement-like* object and normalizes it (`fit.coerce_plan`): a minimal
  placement gains the honest defaults instead of an `AttributeError`;
* a negative layer count means "all layers on the device" (llama.cpp's own reading), so the ladder
  can still walk *down* from it — that is the bench default on a GPU box;
* the bench load seam walks fewer layers -> CPU-only and succeeds;
* a placement that fits nowhere is a typed `E_BACKEND_OOM`, never `E_INTERNAL`.

The vehicle is the fake runtime of `tests/test_fit_oom_recovery.py`: a real bundle directory, the
operator's own OOM tail through the real `llama_log_set` ABI. Model-free and GPU-free.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import pytest

from ggufone import cli
from ggufone.bench import harness, suites
from ggufone.engine import session as session_module
from ggufone.errors import BackendOomError
from ggufone.runtime import fit
from tests.test_fit import GIB, MIB, write_gguf
from tests.test_fit_oom_recovery import FakeBackend, fake_runtime

LAYERS = 4


def model_file(tmp_path: pathlib.Path, *, n_layer: int = LAYERS) -> pathlib.Path:
    return write_gguf(tmp_path / "model.gguf", n_layer=n_layer)


def bench_spec(path: pathlib.Path, backend: FakeBackend, *,
               layers: int = LAYERS) -> harness.ModelSpec:
    """The spec `suites.live_factory` builds for one bench row (the loader-facing input)."""
    return harness.ModelSpec(path=str(path), backend="cpu",
                             runtime_dir=str(backend.directory), threads=1,
                             n_gpu_layers=layers)


def roomy_gpu_host() -> fit.HostFacts:
    return fit.HostFacts(backend="vulkan", ram_bytes=31 * GIB, vram_bytes=8 * GIB,
                         n_cpu=8, fingerprint="vulkan:roomy")


# ------------------------------------------------------------------ the ladder accepts a placement
def test_the_ladder_normalizes_a_placement_and_never_returns_an_unknown_kv_type(
        tmp_path: pathlib.Path) -> None:
    """`harness.Placement` carries what a benchmark can know: the layer count. Nothing else."""
    facts = fit.ModelFacts.read(model_file(tmp_path), want_sha256=False)

    steps = fit.degrade_ladder(harness.Placement(LAYERS), facts)

    assert [(step.n_gpu_layers, step.kv_type) for step in steps] == [
        (LAYERS // 2, "f16"), (0, "f16"), (0, "q8_0"), (0, "q4_0")]
    assert all(step.kv_type in fit.KV_DOWNGRADE_ORDER for step in steps)
    assert all("W_FIT_DOWNGRADE" in step.warnings for step in steps)
    assert "W_KV_TYPE_DOWNGRADE" not in steps[0].warnings     # auto -> f16 is not a downgrade

    plan = fit.coerce_plan(harness.Placement(LAYERS))
    assert isinstance(plan, fit.FitPlan)
    assert plan.n_gpu_layers == LAYERS
    assert plan.kv_type == "auto"          # nothing was pinned: the context init resolves it
    assert plan.n_ctx == 0                 # a load sizes no cache
    assert all(step.kv_type != "auto" for step in steps)


def test_a_real_plan_keeps_its_identity_through_the_normalization(tmp_path: pathlib.Path) -> None:
    """A caller that already speaks `FitPlan` must not be copied, rewritten or degraded."""
    model = fit.ModelFacts.read(model_file(tmp_path), want_sha256=False)
    plan = fit.estimate_plan(model, roomy_gpu_host(), n_ctx=4096)
    assert plan.n_gpu_layers == LAYERS
    assert fit.coerce_plan(plan) is plan
    assert fit.degrade_ladder(plan, model)[0].kv_type == plan.kv_type


def test_a_negative_placement_means_all_layers_and_still_walks_down(
        tmp_path: pathlib.Path) -> None:
    """`--gpu-layers -1` is the bench default on a GPU box: offload everything, reduce from there.

    Reading it as "0 or fewer layers, nothing to reduce" (the parent tree did) makes the ladder
    empty on the one host class this card is about.
    """
    n_layer = 8
    model_path = model_file(tmp_path, n_layer=n_layer)
    facts = fit.ModelFacts.read(model_path, want_sha256=False)
    all_layers = dataclasses.replace(
        fit.estimate_plan(facts, roomy_gpu_host(), n_ctx=4096), n_gpu_layers=-1)

    steps = fit.degrade_ladder(all_layers, facts)

    assert [(step.n_gpu_layers, step.kv_type) for step in steps] == [
        (n_layer // 2, "f16"), (0, "f16"), (0, "q8_0"), (0, "q4_0")]
    assert fit.plan_device_bytes(all_layers, facts) == (
        all_layers.est_weights_bytes + all_layers.est_kv_bytes)


# ------------------------------------------------------------------- the bench seam, end to end
def test_the_bench_load_seam_walks_the_ladder_instead_of_raising(tmp_path: pathlib.Path) -> None:
    """RED on the parent commit: the first `model.load()` of a bench run raised `AttributeError`.

    `harness.LiveModel.load()` is the only call the suites make to reach the loader, so this is
    the bench path itself — with a backend that fails while anything is offloaded, exactly like
    the operator's busy desktop.
    """
    backend = FakeBackend(n_layer=LAYERS)                     # fails while n_gpu_layers > 0
    model_path = model_file(tmp_path)

    with fake_runtime(tmp_path, backend):
        model = harness.LiveModel(bench_spec(model_path, backend))
        try:
            load_ms = float(model.load())
            handle = model.handle
            assert backend.load_calls == [LAYERS, LAYERS // 2, 0]
            assert load_ms == pytest.approx(handle.load_ms)
            assert handle.n_gpu_layers == 0
            assert handle.placement.degraded is True
            assert set(handle.warnings) == {"W_BACKEND_OOM", "W_FIT_DOWNGRADE"}
            assert handle.placement.attempts == (f"n_gpu_layers={LAYERS} -> oom",
                                                 f"n_gpu_layers={LAYERS // 2} -> oom")
            assert "degraded after a backend allocation failure" in handle.placement.note
            assert handle.fit_plan.kv_type in fit.KV_DOWNGRADE_ORDER
            # the row must carry what the loader did, not only what the flags asked for
            assert model.placement["degraded"] is True
            assert model.placement["n_gpu_layers"] == 0
            assert harness.placement_of(model, model.spec) == {
                "requested": f"n_gpu_layers={LAYERS}",
                "used": {"note": handle.placement.note, "n_gpu_layers": 0, "kv_type": "f16",
                         "degraded": True,
                         "attempts": [f"n_gpu_layers={LAYERS} -> oom",
                                      f"n_gpu_layers={LAYERS // 2} -> oom"],
                         "warnings": ["W_BACKEND_OOM", "W_FIT_DOWNGRADE"]}}
        finally:
            model.close()


def test_a_bench_placement_the_device_holds_loads_with_the_requested_layers(
        tmp_path: pathlib.Path) -> None:
    """The happy half of the same seam: no degradation, and the placement says what it did."""
    backend = FakeBackend(n_layer=LAYERS, fail=lambda ngl, call: False)
    model_path = model_file(tmp_path)

    with fake_runtime(tmp_path, backend):
        model = harness.LiveModel(bench_spec(model_path, backend))
        try:
            model.load()
            handle = model.handle
            assert backend.load_calls == [LAYERS]
            assert handle.n_gpu_layers == LAYERS
            assert handle.placement.degraded is False
            assert handle.placement.kv_type == "auto"        # the load never pinned a rung
            assert handle.warnings == ()
            assert f"{LAYERS} layer(s) offloaded" in handle.placement.note
            assert harness.placement_of(model, model.spec)["used"]["n_gpu_layers"] == LAYERS
            assert harness.placement_of(model, model.spec)["used"]["degraded"] is False
        finally:
            model.close()


def test_the_row_carries_the_request_and_the_usage_for_a_model_free_seam() -> None:
    """The model-free seam never saw a loader: the report says so instead of inventing a usage."""
    spec = harness.ModelSpec(path="/tmp/none.gguf", backend="cpu", n_gpu_layers=8)

    class Seamless:
        pass

    assert harness.placement_of(Seamless(), spec) == {"requested": "n_gpu_layers=8", "used": None}


def test_the_rendered_table_prints_the_placement_the_row_used(tmp_path: pathlib.Path) -> None:
    """The published table must name both the request and what the loader did with it."""

    def rendered(backend: FakeBackend) -> str:
        model_path = model_file(tmp_path)
        with fake_runtime(tmp_path, backend):
            model = harness.LiveModel(bench_spec(model_path, backend))
            try:
                model.load()
                report = {"suite": "latency", "generated_at": "2026-01-01T00:00:00Z",
                          "host": {}, "config": {}, "commands": {},
                          "model": {"name": "model.gguf"},
                          "placement": harness.placement_of(model, model.spec)}
                return harness.render_report(report)
            finally:
                model.close()

    fits = rendered(FakeBackend(n_layer=LAYERS, fail=lambda ngl, call: False))
    assert f"- placement: requested n_gpu_layers={LAYERS}, used " \
           f"n_gpu_layers={LAYERS} kv_type=auto" in fits
    assert "(degraded)" not in fits

    degraded = rendered(FakeBackend(n_layer=LAYERS))          # fails while anything is offloaded
    assert "- placement: requested " \
           f"n_gpu_layers={LAYERS}, used n_gpu_layers=0 kv_type=f16 (degraded)" in degraded


def test_a_negative_placement_says_it_asked_for_every_layer(tmp_path: pathlib.Path) -> None:
    """`-1` is "every layer", not "nothing offloaded": the note the operator reads must say so."""
    backend = FakeBackend(n_layer=LAYERS, fail=lambda ngl, call: False)
    model_path = model_file(tmp_path)

    with fake_runtime(tmp_path, backend):
        model = harness.LiveModel(bench_spec(model_path, backend, layers=-1))
        try:
            model.load()
            handle = model.handle
            assert backend.load_calls == [-1]                  # passed to llama.cpp unchanged
            assert handle.n_gpu_layers == -1
            assert handle.placement.note == "all layers requested: n_gpu_layers=-1 (kv_type=auto)"
            assert handle.placement.degraded is False
        finally:
            model.close()


def test_a_placement_that_fits_nowhere_is_a_typed_oom_error(tmp_path: pathlib.Path) -> None:
    """Every rung fails: `E_BACKEND_OOM` (exit 3) — never an `AttributeError` in `E_INTERNAL`."""
    backend = FakeBackend(n_layer=LAYERS, fail=lambda ngl, call: True)
    model_path = model_file(tmp_path)

    with fake_runtime(tmp_path, backend), pytest.raises(BackendOomError) as excinfo:
        session_module.open_model(model_path, runtime_dir=backend.directory,
                                  fit_plan=harness.Placement(LAYERS), free_probe=lambda: 1112 * MIB)
    assert excinfo.value.code == "E_BACKEND_OOM"
    assert backend.load_calls == [LAYERS, LAYERS // 2, 0]


# ------------------------------------------- which bundle `--backend auto` really measured
def test_an_auto_run_records_the_bundle_it_selected_and_the_one_it_passed_over(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A box with a CPU *and* an accelerator bundle: the report must say which one ran.

    Reported from the operator's Vulkan host (card t_31b3943a, coordinator note): `bench --suite
    latency --backend auto` without `--gpu-layers` measured CPU only, and neither the report nor
    the rendered table said a Vulkan bundle had been left unused. `auto` resolves to the locally
    installed bundles in `DEFAULT_BACKENDS` order (`cpu` first) and one run measures the first of
    them, so a reader must be able to tell "no accelerator here" from "not selected".
    """
    from tests.fake_engine import BenchModel

    bundles = {"cpu": tmp_path / "b11026-linux-x64-cpu",
               "vulkan": tmp_path / "b11026-linux-x64-vulkan"}
    monkeypatch.setattr(harness, "backend_runtimes", lambda **kwargs: dict(bundles))
    config = harness.BenchConfig(suite="latency", model_path="/tmp/fake.gguf", runs=1,
                                prefill_sizes=(64,))

    report = suites.run_suite(config, factory=lambda spec: BenchModel(spec))

    selection = report["backend_selection"]
    assert selection["requested"] == "auto"
    assert selection["selected"] == "cpu"                       # DEFAULT_BACKENDS order
    assert selection["available"] == ["cpu", "vulkan"]
    assert selection["missing"] == {}                           # nothing was unavailable here
    hint = [note for note in report["notes"] if "--backend vulkan" in note]
    assert hint, report["notes"]
    assert "cpu" in hint[0]
    assert "- backend selection: cpu of the local bundles (cpu, vulkan)" in \
        harness.render_report(report)


def test_a_forced_backend_is_reported_as_such_not_as_a_choice(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`--backend vulkan` is not a selection among bundles: the record must not blame `auto`."""
    from tests.fake_engine import BenchModel

    bundles = {"cpu": tmp_path / "b11026-linux-x64-cpu",
               "vulkan": tmp_path / "b11026-linux-x64-vulkan"}
    monkeypatch.setattr(harness, "backend_runtimes", lambda **kwargs: dict(bundles))
    config = harness.BenchConfig(suite="quality", model_path="/tmp/fake.gguf", backend="vulkan",
                                items=2)

    report = suites.run_suite(config, factory=lambda spec: BenchModel(spec))

    selection = report["backend_selection"]
    assert selection["requested"] == "vulkan"
    assert selection["selected"] == "vulkan"
    assert selection["available"] == ["vulkan"]                 # only what the run asked for
    assert not [note for note in report["notes"] if "--backend auto" in note]
    assert "- backend selection" not in harness.render_report(report)   # a forced run is no choice


def test_the_default_gpu_layers_follow_the_backend_and_the_flag_always_wins() -> None:
    """The default-side contract the operator note is about, pinned next to the ladder it feeds.

    `spec_for` turns `--backend` into a layer count: a CPU bundle asks for no offload, an
    accelerator bundle for *all* layers (`-1`), and `--gpu-layers` overrides both.
    """
    runtimes = {"cpu": "/tmp/cpu", "vulkan": "/tmp/vulkan"}
    auto = harness.BenchConfig(suite="throughput", model_path="/tmp/m.gguf")
    assert harness.spec_for(auto, "cpu", runtimes=runtimes).n_gpu_layers == 0
    assert harness.spec_for(auto, "vulkan", runtimes=runtimes).n_gpu_layers == -1
    forced = harness.BenchConfig(suite="throughput", model_path="/tmp/m.gguf", gpu_layers=36)
    assert harness.spec_for(forced, "vulkan", runtimes=runtimes).n_gpu_layers == 36
    assert harness.spec_for(forced, "cpu", runtimes=runtimes).n_gpu_layers == 36


def test_the_bench_cli_reports_the_typed_reason_never_an_attribute_error(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """The surface the operator runs: a bench row that cannot load names the typed code."""
    backend = FakeBackend(n_layer=LAYERS, fail=lambda ngl, call: True)
    model_path = model_file(tmp_path)

    with fake_runtime(tmp_path, backend):
        # the bench discovers bundles itself: point it at the fake one for this run
        monkeypatch.setattr(harness, "backend_runtimes",
                            lambda **kwargs: {"cpu": backend.directory})
        code = cli.main(["bench", "--suite", "throughput", "--model", str(model_path),
                         "--gpu-layers", str(LAYERS), "--runs", "1", "--json"])
    report: dict[str, Any] = json.loads(capsys.readouterr().out)
    row = report["backends"][0]
    assert code == 1                                   # nothing was measured: a reported row
    assert row["measured"] is False
    assert "E_BACKEND_OOM" in row["reason"]
    assert "AttributeError" not in row["reason"]
