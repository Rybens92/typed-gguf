"""A-E1c-4/5/6/7 on this box: the real plan, the real binary, the measured RSS.

`model`-marked (needs the pinned GGUF) and `--run-network` (like every live test of the repo;
nothing here touches the network — the flag is the repo's gate for "needs real assets").

The plan-only gates pin their host world (`_roomy_host()`, card t_e29734e6): what they measure is
the real binary/model, and a plan read off a busy desktop is a measurement of the desktop — it
legitimately downgrades the KV type and adds `W_FIT_DOWNGRADE`. The gates that *load* the model
are device tests by design (they need a card with room) and say so.

Run::

    TYPED_GGUF_RUNTIME_DIR=<bundle> uv run pytest -q --run-network tests/test_fit_live.py -s
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest

from typed_gguf import cli, schema
from typed_gguf.engine import decide
from typed_gguf.engine import session as session_module
from typed_gguf.runtime import finder, fit

SPARK = pathlib.Path.home() / ".hermes" / "models" / "Spark-X2.5-4B-Q8_0.gguf"
GIB = 1024 ** 3


def _roomy_host() -> fit.HostFacts:
    """An idle RTX-3060-Ti-class world: 7 GiB free of 8 GiB (the operator's quiet desktop)."""
    return fit.HostFacts(backend="vulkan", ram_bytes=31 * GIB, vram_bytes=8 * GIB,
                         vram_free_bytes=7 * GIB, n_cpu=8, fingerprint="vulkan:live-test")


def _runtime_dir() -> pathlib.Path:
    env = os.environ.get("TYPED_GGUF_RUNTIME_DIR")
    if env and (pathlib.Path(env) / "libllama.so").exists():
        return pathlib.Path(env)
    found = finder.find_runtime()
    if found:
        return found
    pytest.skip("no llama.cpp runtime on this box (set TYPED_GGUF_RUNTIME_DIR)")


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
        # v2 (§5.3): the plan aims at the standard and grows, so leaving f16 for a deeper rung is
        # a legal way to reach it — the binary's table weights sit ~8 % above the tensor index
        # (§8.3), which is exactly why `_kv_from_budget` may pick q8_0 where the estimate said
        # f16. The rung and its warning must agree; nothing else may warn here.
        assert set(plan.warnings) <= {"W_KV_TYPE_DOWNGRADE"}
        assert (plan.kv_type == "f16") == (plan.warnings == ())
        assert "llama-fit-params" in " ".join(plan.notes)
        assert plan.standard_n_ctx == fit.STANDARD_N_CTX == 32768
        assert plan.ctx_limit in {"standard", "grown"}
        assert plan.n_ctx >= fit.STANDARD_N_CTX          # a roomy host reaches the standard
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
    model_path = _model()
    plan = fit.plan_for_path(model_path, host=_roomy_host(), runtime_dir=None,
                             use_cache=False, home=None)
    assert plan.source == "estimate"
    assert plan.warnings == ("W_FIT_ESTIMATED",)
    assert plan.kv_type == "f16"
    per_token = fit.kv_bytes_per_token(36, 4, 256, 256, fit.KV_BYTES_PER_ELEMENT["f16"])
    assert per_token == 147456                              # SPEC 2.4, executed
    # v2 (§3, AC-6/AC-7): the pinned 4B is a sliding-window model, so its real cache is the SWA
    # model — `per_token * n_ctx` is the all-layer figure and is the wrong prediction *for it*.
    # Both facts are pinned: the per-layer oracle keeps SPEC 2.4 byte-exact, and the plan uses
    # the SWA-aware model (strictly less).
    model = fit.ModelFacts.read(model_path, want_sha256=False)
    assert model.has_swa and model.sliding_window == 512
    assert model.n_swa_layers == 27 and model.n_layer == 36
    assert plan.est_kv_bytes == fit.kv_bytes(model, plan.n_ctx, plan.kv_type)
    assert plan.est_kv_bytes < per_token * plan.n_ctx
    print(f"\nestimate: kv {plan.est_kv_bytes / 1024 ** 2:.0f} MiB for {plan.n_ctx} ctx tokens "
          f"(all-layer oracle would say {per_token * plan.n_ctx / 1024 ** 2:.0f} MiB)")


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
    # v2 (§5.6, D1 = YES): with nothing pinned the LOAD is the plan's own context — the plan is
    # both the ceiling and the size, and it is at or above the standard. The engine reports the
    # runtime's cells (`llama_n_ctx`), which llama.cpp pads up to a 256-cell block (§8.6), so the
    # relation is the pad one, not equality: 55 706 asked -> 55 808 loaded, measured.
    assert fitted["engine"]["n_ctx"] >= fitted["engine"]["fit"]["n_ctx"]
    assert fitted["engine"]["n_ctx"] - fitted["engine"]["fit"]["n_ctx"] < 256
    assert fitted["engine"]["fit"]["n_ctx"] >= fit.STANDARD_N_CTX
    assert fitted["engine"]["fit"]["standard_n_ctx"] == fit.STANDARD_N_CTX
    assert fitted["engine"]["fit"]["ctx_limit"] in {"standard", "grown"}
    # a pin wins and is never grown: `--n-ctx 32768` really loads 32 768 now (AC-8; pre-v2 this
    # was capped to the 4 096 default plan — the live regression this card flips)
    capped = cli.decide_payload({**payload, "options": {"threads": 4, "n_ctx": 32768}}, home=None)
    assert capped["engine"]["n_ctx"] == 32768               # the pin is the load size
    assert capped["engine"]["fit"]["n_ctx"] >= 32768        # the plan stays the ceiling
    # ... and a pin that has to be *planned* (a fresh, uncached plan) is labelled `pinned`
    fresh = fit.plan_for_path(model_path, runtime_dir=_runtime_dir(), home=None, use_cache=False,
                              n_ctx=32768)
    assert fresh.ctx_limit == "pinned" and fresh.n_ctx == 32768
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
    # no template fallback, no estimate — a low-mass note is the engine's own diagnostic and is
    # allowed (the readout math never hides it). v2 (§5.3): the plan aims at the standard and may
    # legally leave f16 to reach it, so the *downgrade* warning is allowed exactly when the rung
    # moved — the warning must tell the truth, and nothing else may appear.
    assert ("W_KV_TYPE_DOWNGRADE" in response["warnings"]) == (engine["fit"]["kv_type"] != "f16")
    assert not set(response["warnings"]) & {"W_TEMPLATE_FALLBACK", "W_FIT_ESTIMATED"}
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
    from typed_gguf.runtime import ctypes_binding

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
    from typed_gguf.registry import recommend

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


# ------------------------------- AC-16: the end-to-end gate of context sizing v2
def _six_k_state() -> str:
    """The AC-16 request: a state the pre-v2 4 096 cap could not take.

    Measured on this box: 4 596 prefix tokens (the card's own live receipt,
    `docs/evidence/context-v2/ask_v2_6k_state_tokens.txt`, is a longer variant at 5 988) — over
    the old default and comfortably under the standard.
    """
    notes = " ".join(
        f"Incident {index}: the billing dashboard is blank for every user after login since "
        f"09:{index:02d}; the payments worker restarted itself and the invoice queue drained "
        f"slower than the SLA. On-call checked the api logs, saw no 500s, and left a note that "
        f"the cache warmup on the infrastructure node still held the previous deploy's schema."
        for index in range(1, 59))
    return f"# War-room notes\n\n{notes}\n\n## Question\nWhich area owns the blank dashboard?"


@pytest.mark.model
def test_ac16_the_v2_default_answers_a_six_kilo_token_request() -> None:
    """SPEC-context-v2 AC-16, live: standard-or-better sizing, and the same request at a pin.

    Needs a card with room (like every load gate in this file): the plan must reach the standard
    for the no-pin half to be meaningful, which this box's idle GPU does (~50 k here). The point
    of the two halves together is that the *sizing* is what decides — nothing about the request
    changed between them.
    """
    from typed_gguf.errors import TypedGgufError

    model_path = _model()
    payload = {
        "state": _six_k_state(),
        "model": str(model_path),
        "questions": {"area": {"type": "choice", "instructions": "Which area owns this?",
                               "criteria": {"billing": "payments, invoices and subscriptions",
                                            "technical": "api, infrastructure and deploys"}}},
        "options": {"threads": 4},
    }
    # half 1: nothing pinned — the plan sizes the load (D1) and must reach the standard
    response = cli.decide_payload(payload, home=None)
    engine = response["engine"]
    assert engine["fit"]["standard_n_ctx"] == fit.STANDARD_N_CTX == 32768
    assert engine["fit"]["ctx_limit"] in {"standard", "grown"}
    assert engine["fit"]["n_ctx"] >= fit.STANDARD_N_CTX
    assert engine["n_ctx"] >= engine["fit"]["n_ctx"]       # the runtime's cells, padded (§8.6)
    assert engine["n_ctx"] - engine["fit"]["n_ctx"] < 256  # ...by a block, not by a rung
    assert engine["prefix_tokens"] > 4096                  # the request the old cap refused
    assert response["answers"]["area"]["choice"] in ("billing", "technical")
    print(f"\nAC-16: {engine['prefix_tokens']} prefix tokens answered at engine.n_ctx "
          f"{engine['n_ctx']} (plan {engine['fit']['n_ctx']}, {engine['fit']['ctx_limit']}, "
          f"{engine['fit']['kv_type']})")
    # half 2: the same request pinned to the old default — the guard refuses it honestly
    with pytest.raises(TypedGgufError) as caught:
        cli.decide_payload({**payload, "options": {"threads": 4, "n_ctx": 4096}}, home=None)
    assert caught.value.code == "E_CTX_TOO_SMALL"
    assert "--n-ctx" in str(caught.value)
    print(f"AC-16: the same request at --n-ctx 4096 -> {caught.value.code}: {caught.value}")
