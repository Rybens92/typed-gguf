"""SPEC-context-v2 §9/§10 — context sizing v2: standard 32k, grow into the room, shrink (t_ca1d4231).

Offline half: AC-1…AC-8 and AC-11/AC-13 as unit tests on `estimate_plan` / `plan_for_model` /
`replan_for_host`, AC-9/AC-10/AC-12/AC-14 through the engine and CLI seams. Every number is the
spec's own arithmetic, reproduced on a *synthetic* 4B so the assertions do not move with the box:

* the model is §3's shape — 36 layers, 4 kv heads, 256/256, `sliding_window = 512`, 27 SWA + 9
  global — and its weight bytes are chosen so the box of §0.4/§8.2 (8 192 MiB nominal, 6 760 MiB
  free, `--fit-target 1024` -> 5 736 MiB budget) answers exactly the spec's three max-fit numbers:
  26 987 at f16, 53 511 at q8_0, 103 807 at q4_0;
* AC-6 pins the four measured KV byte counts of §3 (`ac6_check.out`) through `fit.kv_bytes`.

The live half (AC-16, the engine's own `llama_kv_cache` lines) is in `tests/test_fit_live.py`.
"""
from __future__ import annotations

from collections.abc import Callable

import pytest

from tests.fake_engine import FakeSession
from typed_gguf import cli, schema
from typed_gguf.calibration import routing
from typed_gguf.engine import decide
from typed_gguf.errors import WARNING_CODES, TypedGgufError
from typed_gguf.runtime import fit

GIB = 1024 ** 3
MIB = 1024 ** 2

#: the box of §0.4/§8.2 (measured): 8 192 MiB nominal, 6 760 MiB free -> 5 736 MiB budget
BOX = fit.HostFacts(backend="vulkan", ram_bytes=32 * GIB, vram_bytes=8 * GIB, n_cpu=24,
                    fingerprint="cafebabe12345678", vram_free_bytes=6760 * MIB)
#: a quiet box with room to spare (the growth rule must be a *policy* answer, not a box answer)
ROOMY = fit.HostFacts(backend="vulkan", ram_bytes=32 * GIB, vram_bytes=24 * GIB, n_cpu=24,
                      fingerprint="cafebabe12345679", vram_free_bytes=16 * GIB)
#: §3's executed weight sum for the 4B (tensor index). Chosen so the three rungs answer the
#: spec's own numbers on `BOX` (26 987 / 53 511 / 103 807) — see the module docstring.
WEIGHTS_4B = 4_369_634_552


def swa_model(**overrides: object) -> fit.ModelFacts:
    """§3's 4B: 36 layers, 4 kv heads, 256/256, window 512, 27 SWA of 36 (`[T,T,T,F,…]`)."""
    fields: dict[str, object] = dict(
        path="/fake/spark2_5-4b.gguf", sha256="a" * 64, arch="spark2_5", n_layer=36,
        n_kv_head=4, key_len=256, value_len=256, n_ctx_train=1_048_576,
        weights_bytes=WEIGHTS_4B, n_swa_layers=27, n_global_layers=9, sliding_window=512)
    fields.update(overrides)
    return fit.ModelFacts(**fields)  # type: ignore[arg-type]


def plain_model(**overrides: object) -> fit.ModelFacts:
    """The same shape without `attention.sliding_window` — AC-7's model (today's formula)."""
    return swa_model(n_swa_layers=0, n_global_layers=0, sliding_window=0, **overrides)


def budget(host: fit.HostFacts, fit_target_mb: int = fit.DEFAULT_FIT_TARGET_MB) -> int:
    return fit.fit_budget(host, fit_target_mb)


def _choice_payload(**options: object) -> dict:
    """A dev-set-shaped request; `options` merge over the pre-v2 cell the fork tests use."""
    merged: dict = {"cue": "shipped", "chat_format": "answer_sheet"}
    merged.update(options)
    return {
        "state": "The billing dashboard is blank for every user after login.",
        "questions": {"area": {"type": "choice", "instructions": "Which area owns this?",
                               "criteria": {"billing": "payments and invoices",
                                            "technical": "api and infrastructure"}}},
        "options": merged,
    }


# ------------------------------------------------------------------ AC-6 (🔴) SWA KV accounting
@pytest.mark.parametrize("n_ctx, kv_type, want", [
    (4096, "f16", 264241152),
    (32768, "f16", 1321205760),
    (32768, "q4_0", 371589120),
    (131072, "q4_0", 1390804992),
])
def test_ac6_the_swa_kv_formula_reproduces_the_four_measured_bytes(n_ctx: int, kv_type: str,
                                                                  want: int) -> None:
    """§3: `n_global·b·n_ctx + n_swa·b·(window + n_ubatch)`, pinned to the measured loads."""
    assert fit.kv_bytes(swa_model(), n_ctx, kv_type) == want


def test_ac6_the_swa_term_holds_window_plus_n_ubatch_cells() -> None:
    """The SWA cache is `window + n_ubatch` cells: 768 at `-ub 256` (d1 log), 1024 at 512."""
    per_layer = round(4 * 512 * (18 / 32))
    wide = fit.kv_bytes(swa_model(), 32768, "q4_0", n_ubatch=512)
    narrow = fit.kv_bytes(swa_model(), 32768, "q4_0", n_ubatch=256)
    assert wide - narrow == per_layer * 27 * 256


def test_ac7_a_model_without_sliding_window_keeps_todays_formula_byte_for_byte() -> None:
    """AC-7/SPEC 2.4: no SWA facts -> the all-layer formula, unchanged (and the old pins hold)."""
    model = plain_model()
    assert fit.kv_bytes_per_token(36, 4, 256, 256, 2.0) == 147456           # SPEC 2.4, executed
    assert fit.kv_bytes(model, 32768, "f16") == 147456 * 32768
    assert fit.kv_bytes(model, 4096, "q8_0") == int(round(36 * 4 * 512 * (34 / 32))) * 4096
    plan = fit.estimate_plan(model, BOX, n_ctx=4096)
    assert plan.est_kv_bytes == 147456 * 4096


# ---------------------------------------------------------------- AC-1 (🟡) default target
def test_ac1_a_first_rung_that_holds_the_standard_is_adopted_and_grown_into() -> None:
    model = swa_model()
    plan = fit.estimate_plan(model, ROOMY)
    assert plan.standard_n_ctx == fit.STANDARD_N_CTX == 32768
    assert plan.n_ctx == fit.max_fit_n_ctx(model, plan.kv_type, budget(ROOMY))
    assert plan.n_ctx >= fit.STANDARD_N_CTX
    assert plan.ctx_limit in {"standard", "grown"}
    assert "W_CTX_BELOW_STANDARD" not in plan.warnings


# ------------------------------------------------------------------------ AC-2 (🟡) grow
def test_ac2_the_plan_grows_to_the_measured_boxes_max_on_the_top_affordable_rung() -> None:
    model = swa_model()
    plan = fit.estimate_plan(model, BOX)
    assert plan.kv_type == "q8_0"                        # f16 cannot hold the standard here
    assert plan.n_ctx == fit.max_fit_n_ctx(model, "q8_0", budget(BOX)) == 53511
    assert plan.ctx_limit == "grown"
    assert plan.warnings == ("W_KV_TYPE_DOWNGRADE", "W_FIT_ESTIMATED")
    assert any("32768 -> 53511" in note for note in plan.notes)


# --------------------------------------------------------- AC-3 (🟡) ladder before shrink
def test_ac3_the_kv_ladder_moves_before_the_standard_is_given_up() -> None:
    plan = fit.estimate_plan(swa_model(), BOX)
    assert plan.kv_type == "q8_0"
    assert plan.n_ctx >= fit.STANDARD_N_CTX
    assert "W_KV_TYPE_DOWNGRADE" in plan.warnings
    pinned = fit.estimate_plan(swa_model(), BOX, n_ctx=32768)
    assert (pinned.n_ctx, pinned.kv_type) == (32768, "q8_0")      # the pin really loads 32 768


# --------------------------------------------------------- AC-4 (🔴) graceful shrink
def test_ac4_a_budget_below_the_standard_shrinks_gracefully_and_is_still_a_plan() -> None:
    model = swa_model()
    tight = model.weights_bytes + fit.OVERHEAD_BYTES + 300 * MIB
    plan = fit.estimate_plan(model, BOX, budget_bytes=tight)
    assert plan.n_ctx < fit.STANDARD_N_CTX
    assert plan.n_ctx >= min(fit.DEFAULT_N_CTX, model.n_ctx_train)
    assert plan.kv_type == "q4_0"
    assert plan.ctx_limit == "shrunk"
    assert "W_CTX_BELOW_STANDARD" in plan.warnings
    assert "W_KV_TYPE_DOWNGRADE" in plan.warnings
    assert plan.n_ctx == fit.max_fit_n_ctx(model, "q4_0", tight) == 27268
    assert any("32768 -> 27268" in note for note in plan.notes)
    assert any("to fit the budget (4979 MiB)" in note for note in plan.notes)


def test_ac4_a_plan_the_weights_alone_overrun_is_still_returned_with_the_note() -> None:
    """§5.4: `insufficient` fires unchanged; v2 does not turn it into an exception."""
    model = swa_model()
    plan = fit.estimate_plan(model, BOX, budget_bytes=model.weights_bytes // 2)
    assert plan.insufficient
    assert plan.n_ctx == fit.DEFAULT_N_CTX                  # never below the floor


# ------------------------------------------------------------------- AC-5 (🟡) window cap
def test_ac5_the_models_own_window_caps_the_plan_and_is_not_a_warning() -> None:
    plan = fit.estimate_plan(swa_model(n_ctx_train=8192), ROOMY)
    assert (plan.n_ctx, plan.ctx_limit) == (8192, "window")
    assert "W_CTX_BELOW_STANDARD" not in plan.warnings


def test_ac5_a_window_below_the_floor_plans_at_the_window_never_at_the_floor() -> None:
    plan = fit.estimate_plan(swa_model(n_ctx_train=2048), ROOMY)
    assert plan.n_ctx == 2048


# ------------------------------------------------------------------ AC-8 (🟡) pin semantics
def test_ac8_a_pin_wins_never_grows_and_lands_on_the_rung_that_holds_it() -> None:
    model = swa_model()
    small = fit.estimate_plan(model, BOX, n_ctx=8192)
    assert (small.n_ctx, small.kv_type, small.ctx_limit) == (8192, "f16", "pinned")
    assert "W_KV_TYPE_DOWNGRADE" not in small.warnings
    standard = fit.estimate_plan(model, BOX, n_ctx=32768)
    assert (standard.n_ctx, standard.kv_type, standard.ctx_limit) == (32768, "q8_0", "pinned")
    big = fit.estimate_plan(model, BOX, n_ctx=65536)
    assert (big.n_ctx, big.kv_type, big.ctx_limit) == (65536, "q4_0", "pinned")


def test_ac8_a_pin_above_every_rung_shrinks_to_that_rung_never_below_the_floor() -> None:
    model = swa_model()
    plan = fit.estimate_plan(model, BOX, n_ctx=1_048_576)
    assert plan.n_ctx == fit.max_fit_n_ctx(model, "q4_0", budget(BOX)) == 103807
    assert plan.ctx_limit == "shrunk"


# ------------------------------------------------------------ AC-11 (🟡) plan fields
def test_ac11_the_two_new_fields_join_the_payload_and_round_trip() -> None:
    assert "standard_n_ctx" in fit.FIT_FIELDS
    assert "ctx_limit" in fit.FIT_FIELDS
    plan = fit.estimate_plan(swa_model(), BOX)
    payload = plan.to_dict()
    assert payload["standard_n_ctx"] == 32768
    assert payload["ctx_limit"] == "grown"
    assert fit.FitPlan.from_dict(payload) == plan


def test_ac11_a_cache_written_before_v2_reads_as_zero_and_empty() -> None:
    plan = fit.estimate_plan(swa_model(), BOX)
    legacy = {key: value for key, value in plan.to_dict().items()
              if key not in ("standard_n_ctx", "ctx_limit")}
    older = fit.FitPlan.from_dict(legacy)
    assert (older.standard_n_ctx, older.ctx_limit) == (0, "")
    assert older.n_ctx == plan.n_ctx                        # nothing else moves


# ------------------------------------------------- AC-13 (🟡) cache and re-validation
def test_ac13_re_validation_never_grows_a_plan_and_keeps_the_v2_fields(tmp_path) -> None:
    model = swa_model()
    grown = fit.estimate_plan(model, ROOMY)
    shrunk = fit.replan_for_host(model, grown, BOX)
    assert shrunk.n_ctx == grown.n_ctx                      # the context is not re-sized by a re-plan
    assert shrunk.standard_n_ctx == 32768
    assert shrunk.ctx_limit == grown.ctx_limit
    assert shrunk.n_gpu_layers <= grown.n_gpu_layers
    assert "re-planned for free device memory" in " ".join(shrunk.notes)


def test_ac13_a_stored_plan_comes_back_re_validated_for_the_box_now(tmp_path) -> None:
    model = swa_model()
    fit.store_plan(fit.estimate_plan(model, ROOMY), home=tmp_path)
    live = fit.plan_for_model(model, BOX, home=tmp_path, runtime_dir=None, use_cache=True)
    assert live.n_ctx < 300_000                             # the ROOMY growth answer, not reused
    assert live.standard_n_ctx == 32768


# ----------------------------------------------------- AC-9 (🟡) the load uses the plan size
def test_ac9_the_load_uses_the_plan_context_unless_the_request_pins_one() -> None:
    session = FakeSession(n_vocab=256)
    request = schema.parse_request(_choice_payload())
    sized = decide.plan_context(request, session, n_ctx_cap=53511)
    assert sized.n_ctx == 53511                             # D1 = YES (§5.6)
    pinned = schema.parse_request(_choice_payload(n_ctx=8192))
    assert decide.plan_context(pinned, session, n_ctx_cap=53511).n_ctx == 8192
    bare = decide.plan_context(request, session)            # `--no-fit`: SPEC 2.2 literal
    assert bare.n_ctx == len(bare.prefix_tokens) + bare.max_question_tokens + decide.CONTEXT_MARGIN


# ------------------------------------------------------- AC-10 (🔴) the guard is unchanged
def test_ac10_a_request_over_the_loaded_context_names_the_reload_handle() -> None:
    session = FakeSession(n_vocab=64, n_ctx=8)
    request = schema.parse_request(_choice_payload())
    plan = decide.plan_context(request, session)
    with pytest.raises(TypedGgufError) as exc:
        decide.DecisionEngine(session).decide(request, plan=plan)
    message = str(exc.value)
    assert exc.value.code == "E_CTX_TOO_SMALL"
    assert f"context holds {session.meta.n_ctx}" in message
    assert "prefix" in message and "longest question" in message and "margin" in message
    assert "--n-ctx" in message and "reload bigger" in message


# ------------------------------------------------------------ AC-12 (🟢) honest reporting
def test_ac12_the_human_fit_output_names_the_standard_and_what_happened() -> None:
    grown = fit.estimate_plan(swa_model(), BOX)
    line = next(line for line in cli.fit_human_lines(grown) if line.startswith("n_ctx:"))
    assert str(grown.n_ctx) in line and "standard 32768" in line and "grown" in line
    tight = swa_model().weights_bytes + fit.OVERHEAD_BYTES + 300 * MIB
    shrunk = fit.estimate_plan(swa_model(), BOX, budget_bytes=tight)
    line = next(line for line in cli.fit_human_lines(shrunk) if line.startswith("n_ctx:"))
    assert str(shrunk.n_ctx) in line and "standard 32768" in line and "shrunk" in line
    pinned = fit.estimate_plan(swa_model(), BOX, n_ctx=8192)
    line = next(line for line in cli.fit_human_lines(pinned) if line.startswith("n_ctx:"))
    assert "pinned" in line


# ------------------------------------------------------- AC-14 (🟡) routing consistency
def test_ac14_routing_aims_at_the_same_standard_the_loader_uses() -> None:
    assert routing.DEFAULT_ROUTE_CTX == fit.STANDARD_N_CTX
    needs = routing.needs_for(schema.parse_request(_choice_payload()))
    assert needs.n_ctx == fit.STANDARD_N_CTX


def test_ac14_the_routing_bound_is_still_the_conservative_spec_2_4_formula() -> None:
    """§6.4: routing keeps its own (conservative) math; only the ctx it aims at moves."""
    facts = plain_model()
    assert routing._per_token(facts, "f16") == 147456        # 1 B/element bound is registry-side
    assert fit.kv_bytes_per_token(36, 4, 256, 256, 2.0) == 147456


# ------------------------------------------------------------------ fit surface (guards)
def test_the_standard_is_a_plan_default_not_a_floor() -> None:
    """§5.1: one standard constant; `DEFAULT_N_CTX` stays the shrink floor / `--fit-ctx` default."""
    assert fit.STANDARD_N_CTX == 32768
    assert fit.DEFAULT_N_CTX == 4096
    assert not hasattr(fit, "DEFAULT_N_CTX_TARGET")
    assert "W_CTX_BELOW_STANDARD" in WARNING_CODES


def test_the_estimate_plan_signature_keeps_an_explicit_int_as_a_pin() -> None:
    """An int `n_ctx` (every pre-v2 caller) still means "pin", never "target"."""
    model = swa_model()
    assert fit.estimate_plan(model, BOX, n_ctx=8192).ctx_limit == "pinned"
    assert fit.estimate_plan(model, BOX).ctx_limit == "grown"


def test_max_fit_is_the_inverse_of_kv_bytes_on_both_shapes() -> None:
    for model in (swa_model(), plain_model(n_layer=6, weights_bytes=64 * MIB)):
        for kv_type in fit.KV_DOWNGRADE_ORDER:
            room = model.weights_bytes + fit.OVERHEAD_BYTES + 512 * MIB
            cap = fit.max_fit_n_ctx(model, kv_type, room)
            assert room - fit.kv_bytes(model, cap, kv_type) >= 0
            assert fit.kv_bytes(model, cap + 1, kv_type) > room - model.weights_bytes - \
                fit.OVERHEAD_BYTES


def test_warning_codes_gain_the_shrink_code_once() -> None:
    from typed_gguf.errors import WARNING_CODES

    assert WARNING_CODES.count("W_CTX_BELOW_STANDARD") == 1


def test_a_pinned_kv_type_starts_the_ladder_and_keeps_its_place() -> None:
    model = swa_model()
    plan = fit.estimate_plan(model, BOX, kv_type="q4_0")
    assert plan.kv_type == "q4_0"
    assert "W_KV_TYPE_DOWNGRADE" not in plan.warnings
    assert plan.n_ctx == fit.max_fit_n_ctx(model, "q4_0", budget(BOX))


#: guard: the tests above must not silently drop an acceptance criterion of §9
AC_COVERED = ("AC-1", "AC-2", "AC-3", "AC-4", "AC-5", "AC-6", "AC-7", "AC-8", "AC-9", "AC-10",
              "AC-11", "AC-12", "AC-13", "AC-14")


def test_the_spec_has_an_acceptance_criterion_for_every_pin_this_file_makes() -> None:
    """§9 lists AC-1…AC-16; this file must name the offline ones it covers."""
    text = _spec_document()
    for code in AC_COVERED:
        assert f"**{code} " in text, f"{code} is not in docs/SPEC-context-v2.md §9"


def _spec_document() -> str:
    import pathlib

    return (pathlib.Path(__file__).resolve().parents[1] / "docs"
            / "SPEC-context-v2.md").read_text(encoding="utf-8")
