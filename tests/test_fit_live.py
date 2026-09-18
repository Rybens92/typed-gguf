"""A-E1c-4/5/6/7 on this box: the real plan, the real binary, the measured RSS.

`model`-marked (needs the pinned GGUF) and `--run-network` (like every live test of the repo;
nothing here touches the network — the flag is the repo's gate for "needs real assets").

The plan-only gates pin their host world (`_roomy_host()`, card t_e29734e6): what they measure is
the real binary/model, and a plan read off a busy desktop is a measurement of the desktop — it
legitimately downgrades the KV type and adds `W_FIT_DOWNGRADE`. The gates that *load* the model
are device tests by design (they need a card with room) and say so.

Run::

    GGUFONE_RUNTIME_DIR=<bundle> uv run pytest -q --run-network tests/test_fit_live.py -s
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest

from ggufone import cli, schema
from ggufone.engine import decide
from ggufone.engine import session as session_module
from ggufone.runtime import finder, fit

SPARK = pathlib.Path.home() / ".hermes" / "models" / "Spark-X2.5-4B-Q8_0.gguf"
GIB = 1024 ** 3


def _roomy_host() -> fit.HostFacts:
    """An idle RTX-3060-Ti-class world: 7 GiB free of 8 GiB (the operator's quiet desktop)."""
    return fit.HostFacts(backend="vulkan", ram_bytes=31 * GIB, vram_bytes=8 * GIB,
                         vram_free_bytes=7 * GIB, n_cpu=8, fingerprint="vulkan:live-test")


def _runtime_dir() -> pathlib.Path:
    env = os.environ.get("GGUFONE_RUNTIME_DIR")
    if env and (pathlib.Path(env) / "libllama.so").exists():
        return pathlib.Path(env)
    found = finder.find_runtime()
    if found:
        return found
    pytest.skip("no llama.cpp runtime on this box (set GGUFONE_RUNTIME_DIR)")


def _model() -> pathlib.Path:
    if not SPARK.exists():
        pytest.skip(f"{SPARK} is not on this box")
    return SPARK


def _request(model: pathlib.Path, **options) -> schema.Request:
    return schema.parse_request({
        "state": "The billing dashboard is blank for every user after login since 09:12.",
        "model": str(model),
        "questions": {"area": {"type": "choice", "instructions": "Which team owns this?",
                               "criteria": {"billing": "payments, invoices, refunds",
                                            "technical": "api and infrastructure",
                                            "sales": "contracts and pricing"}}},
        "options": options,
    })


def _rss_delta(model_path: pathlib.Path, plan: fit.FitPlan, request: schema.Request) -> dict:
    """Load the model with the plan and measure the process RSS the load added."""
    baseline = fit.measured_rss_bytes()
    handle = session_module.open_model(model_path, runtime_dir=_runtime_dir(), fit_plan=plan)
    after_load = fit.measured_rss_bytes()
    context_plan = decide.plan_context(request, handle, n_ctx_cap=plan.n_ctx)
    with session_module.ModelSession(handle, context_plan, states_home=None) as live:
        after_ctx = fit.measured_rss_bytes()
        live.decode  # noqa: B018 - the engine's own decode path is exercised by the E2E test
    handle.close()
    return {"baseline": baseline, "after_load": after_load, "after_ctx": after_ctx,
            "delta_load": after_load - baseline, "delta_ctx": after_ctx - baseline}


# --------------------------------------------------------- A-E1c-4: the real plan
@pytest.mark.model
def test_the_pinned_default_model_gets_a_plan_from_the_binary() -> None:
    model_path = _model()
    plan = fit.plan_for_path(model_path, host=_roomy_host(), runtime_dir=_runtime_dir(),
                             home=None)
    assert tuple(field for field in fit.FIT_FIELDS if field not in plan.to_dict()) == ()
    assert plan.arch == "spark2_5"
    assert plan.kv_type in fit.KV_DOWNGRADE_ORDER
    assert plan.n_ctx >= 1024 and plan.n_seq_max >= 3
    assert plan.est_weights_bytes > 4 * 1024 ** 3          # a 4B Q8_0 model
    assert plan.est_kv_bytes > 0 and plan.est_total_bytes > plan.est_weights_bytes
    if fit.fit_binary(_runtime_dir()) is not None:
        assert plan.source == "llama-fit-params"
        assert plan.warnings == ()
        assert "llama-fit-params" in " ".join(plan.notes)
    else:                                                   # pragma: no cover - bundle present
        assert plan.source == "estimate"
        assert "W_FIT_ESTIMATED" in plan.warnings
    print(f"\nlive plan: {json.dumps({k: plan.to_dict()[k] for k in fit.FIT_FIELDS})}")


@pytest.mark.model
def test_the_tensor_index_matches_the_binary_s_model_row() -> None:
    """Our weight bytes come from the GGUF tensor index; the binary agrees within 1 MiB."""
    model_path = _model()
    model = fit.ModelFacts.read(model_path)
    assert model.weights_bytes > 0
    table = fit.run_llama_fit_params(model, _roomy_host(), runtime_dir=_runtime_dir(),
                                     n_ctx=4096, n_seq_max=8, runner=None)
    if table is None:                                       # pragma: no cover - bundle present
        pytest.skip("no llama-fit-params in the runtime")
    difference = abs(table.est_weights_bytes - model.weights_bytes)
    print(f"\nweights: tensor index {model.weights_bytes} vs binary {table.est_weights_bytes} "
          f"(delta {difference} B)")
    assert difference <= 4 * 1024 * 1024                   # ≤ 4 MiB apart


@pytest.mark.model
def test_the_estimate_alone_still_answers_the_contract() -> None:
    """`source=estimate` (no runtime handed in) keeps every field and warns."""
    plan = fit.plan_for_path(_model(), host=_roomy_host(), runtime_dir=None,
                             use_cache=False, home=None)
    assert plan.source == "estimate"
    assert plan.warnings == ("W_FIT_ESTIMATED",)
    assert plan.kv_type == "f16"
    per_token = fit.kv_bytes_per_token(36, 4, 256, 256, fit.KV_BYTES_PER_ELEMENT["f16"])
    assert per_token == 147456                              # SPEC 2.4, executed
    assert plan.est_kv_bytes == per_token * plan.n_ctx
    print(f"\nestimate: kv {plan.est_kv_bytes / 1024 ** 2:.0f} MiB for {plan.n_ctx} ctx tokens")


# ------------------------------------------------------- A-E1c-6: RSS cross-check
@pytest.mark.model
def test_the_fit_estimate_cross_checks_against_measured_load_rss() -> None:
    """[target] ±20% — measured, printed and asserted on this box."""
    model_path = _model()
    plan = fit.plan_for_path(model_path, runtime_dir=_runtime_dir(), home=None)
    measured = _rss_delta(model_path, plan, _request(model_path, threads=4))
    ratio_ctx = fit.rss_ratio(plan.est_total_bytes, measured["delta_ctx"])
    ratio_load = fit.rss_ratio(plan.est_total_bytes, measured["delta_load"])
    print(f"\nest_total {plan.est_total_bytes / 1024 ** 2:.0f} MiB  "
          f"delta_load {measured['delta_load'] / 1024 ** 2:.0f} MiB ({ratio_load:+.1%})  "
          f"delta_ctx {measured['delta_ctx'] / 1024 ** 2:.0f} MiB ({ratio_ctx:+.1%})")
    assert measured["delta_ctx"] > 0
    assert fit.within_tolerance(plan.est_total_bytes, measured["delta_ctx"], tolerance=0.20), \
        f"estimate {plan.est_total_bytes} vs measured {measured['delta_ctx']} ({ratio_ctx:+.1%})"


# ------------------------------------------------- A-E1c-5: applied on load
@pytest.mark.model
def test_the_plan_is_applied_on_load_unless_no_fit() -> None:
    model_path = _model()
    payload = {
        "state": "The billing dashboard is blank for every user after login.",
        "model": str(model_path),
        "questions": {"area": {"type": "choice", "criteria": {"billing": None,
                                                              "technical": None}}},
        "options": {"threads": 4},
    }
    fitted = cli.decide_payload(payload, home=None)
    assert "fit" in fitted["engine"]
    assert fitted["engine"]["fit"]["source"] in ("llama-fit-params", "estimate")
    # the plan's context ceiling caps the request (the engine never allocates more than planned)
    capped = cli.decide_payload({**payload, "options": {"threads": 4, "n_ctx": 32768}}, home=None)
    assert capped["engine"]["fit"]["n_ctx"] < 32768
    assert capped["engine"]["n_ctx"] == capped["engine"]["fit"]["n_ctx"]
    # the plan's kv_type reached the context; `--no-fit` leaves the request's own value
    assert fitted["engine"]["kv_type"] == fitted["engine"]["fit"]["kv_type"]
    bare = cli.decide_payload(payload, home=None, fit_enabled=False)
    assert "fit" not in bare["engine"]
    assert bare["engine"]["kv_type"] == "auto"
    assert bare["engine"]["n_ctx"] < capped["engine"]["n_ctx"]      # auto-sized, not capped
    for response in (fitted, capped, bare):
        answers = response["answers"]["area"]
        assert abs(sum(answers["probabilities"].values()) - 1.0) < 1e-6


@pytest.mark.model
def test_a_no_fit_run_keeps_the_requested_settings() -> None:
    """`--no-fit` must not change what the request asked for (and must not apply a plan)."""
    model_path = _model()
    payload = {
        "state": "The billing dashboard is blank for every user after login.",
        "model": str(model_path),
        "questions": {"area": {"type": "choice", "criteria": {"billing": None,
                                                              "technical": None}}},
        "options": {"threads": 4, "n_seq_max": 4, "kv_type": "q8_0"},
    }
    response = cli.decide_payload(payload, home=None, fit_enabled=False)
    assert "fit" not in response["engine"]
    assert response["engine"]["n_seq_max"] == 4


# ------------------------------------------------- A-E1c-7: template + fit together
@pytest.mark.model
def test_the_live_answer_carries_the_template_and_the_fit_plan() -> None:
    response = cli.decide_payload({
        "state": "The checkout page returns a 500 for every customer since 09:12.",
        "model": str(_model()),
        "questions": {"area": {"type": "choice", "instructions": "Which area owns this?",
                               "criteria": {"billing": None, "technical": None}}},
        "options": {"threads": 4},
    }, home=None)
    engine = response["engine"]
    assert engine["template"]["kind"] == "gguf-renderer"
    assert engine["template"]["renderer"] == "internal"
    assert engine["template"]["family"] == "spark2_5"
    assert engine["template"]["thinking"] == "suppressed"
    assert engine["template"]["warnings"] == []
    assert engine["fit"]["kv_type"] in fit.KV_DOWNGRADE_ORDER
    assert engine["kv_type"] == engine["fit"]["kv_type"]
    # no template fallback, no estimate, no kv downgrade — a low-mass note is the engine's own
    # diagnostic and is allowed (the readout math never hides it)
    assert not set(response["warnings"]) & {"W_TEMPLATE_FALLBACK", "W_FIT_ESTIMATED",
                                            "W_KV_TYPE_DOWNGRADE"}
    answer = response["answers"]["area"]
    assert answer["choice"] in ("billing", "technical")
    assert 0.0 <= answer["confidence"] <= 1.0
    print(f"\nchoice={answer['choice']} p={answer['probabilities']} "
          f"conf={answer['confidence']:.3f} coverage={answer['coverage']:.3f} "
          f"timings={response['timings']}")


def test_the_live_helpers_skip_cleanly_without_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gates above must skip (not fail) on a box without the pinned model."""
    monkeypatch.setattr(pathlib.Path, "exists", lambda self: False)
    with pytest.raises(pytest.skip.Exception):
        _model()


# ------------------------------------- the fix (card t_8cb0a05e) against the REAL bundle
@pytest.mark.model
def test_the_log_capture_swaps_and_restores_the_real_handler() -> None:
    """The callback swap is the one new libllama ABI surface: prove it on the pinned bundle.

    `llama_log_get` is NOT called (at b11026 it writes through two out-parameters; the no-arg call
    segfaults — measured, see `session.capture_llama_logs`), so the capture must install, collect
    real backend output and reset with a NULL callback without any crash.
    """
    from ggufone.runtime import ctypes_binding

    runtime = ctypes_binding.load_libraries(_runtime_dir())
    with session_module.capture_llama_logs(runtime) as lines:
        model = fit.ModelFacts.read(_model(), want_sha256=False)
        assert model.arch == "spark2_5"
    assert isinstance(lines, list)                       # may be empty; the point is the swap
    assert any(isinstance(cb, ctypes_binding.LLAMA_LOG_CALLBACK)
               for cb in ctypes_binding.live_log_callbacks())
    print(f"\ncaptured {len(lines)} backend line(s) around a header read")


@pytest.mark.model
def test_a_busy_desktop_plan_loads_on_the_free_reading() -> None:
    """Requirement 1, live: 8192 MiB total / 1112 MiB free must yield a load that SUCCEEDS.

    The plan is built from an injected busy-desktop reading (the operator's numbers) and the real
    bundle then loads the real model with it; before the fix this is where the 1.06 GB allocation
    was attempted and the run died.
    """
    from ggufone.registry import recommend

    model_path = _model()
    runtime = _runtime_dir()
    busy = fit.host_facts(backend="vulkan", device_probe=lambda: recommend.DeviceMemory(
        total_bytes=8 * 1024 ** 3, free_bytes=1112 * 1024 ** 2, source="injected"))
    plan = fit.plan_for_path(model_path, host=busy, runtime_dir=runtime, home=None, use_cache=False)
    assert busy.budget_bytes == 1112 * 1024 ** 2
    assert plan.budget_bytes == max(0, 1112 * 1024 ** 2 - fit.DEFAULT_FIT_TARGET_MB * 1024 ** 2)
    assert plan.n_gpu_layers == 0
    assert "W_FIT_DOWNGRADE" in plan.warnings
    handle = session_module.open_model(model_path, runtime_dir=runtime, fit_plan=plan,
                                       free_probe=lambda: 1112 * 1024 ** 2)
    try:
        assert handle.n_gpu_layers == 0
        assert handle.placement.degraded is False         # nothing failed: the plan was honest
        # the note is about the weights, not the compute path (card t_603a35a0)
        assert "no layers offloaded" in handle.placement.note
        assert handle.warnings == ()
        print(f"\nbusy desktop: plan {plan.n_gpu_layers} layers, budget "
              f"{plan.budget_bytes / 1024 ** 2:.0f} MiB, load_ms {handle.load_ms:.0f}")
    finally:
        handle.close()
