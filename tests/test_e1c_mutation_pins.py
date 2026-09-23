"""Mutation-driven pins for the E1c resolver + fit (Tier M hardening round).

Written from the survivor list of the first scoped mutmut pass (`mutants/*.meta`), not from a
feeling: every test here pins a *contract* value the sweep proved the E1c gates did not assert.
Three classes are the target:

* **provenance strings** — `Resolution.to_dict()` fields (`kind`/`renderer`/`source`/`thinking`/
  `explicit`/`notes`) are the response surface (`engine.template`), so a string mutant that
  changes them is observable and must die;
* **boundaries** — the fit plan's budget comparison is `<=`, the KV ladder is walked at the
  exact byte, `within_tolerance` is inclusive, `_gpu_layers` floors the per-layer split;
  these are data-integrity paths, not metric mutants;
* **lists/sets and modes** — the think-marker sets, the downgrade ladder order, the suppression
  mode strings, the cache's schema/sha guards.

Style note: the private helpers (`_kv_from_budget`, `_gpu_layers`, `_kv_int`) are called directly
on purpose — the sweep showed their branches were reachable but unasserted, and a pin is only as
strong as the specific value it claims.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import platform
import struct

import pytest

from typed_gguf.engine import template as tpl
from typed_gguf.errors import ERROR_CODES, WARNING_CODES
from typed_gguf.runtime import fit

GIB = 1024 ** 3
MIB = 1024 ** 2

SPARK_LIKE = (
    "{%- set enable_thinking = enable_thinking | default(true) %}"
    "{{- '<|System|>' }}{%- for message in messages %}"
    "{%- if message.role == 'user' %}{{- '<|User|>' + message.content + '<|End|>' }}"
    "{%- elif message.role == 'system' %}{{- '<|System|>' + message.content }}"
    "{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|Bot|>' }}{%- if enable_thinking %}{{- '<think>' }}{%- endif %}"
    "{%- if not enable_thinking %}{{- '</think>' }}{%- endif %}{%- endif %}"
)

UNSUPPORTED_LIKE = (
    "{%- if messages %}{%- include 'other.jinja' %}{%- endif %}"
    "{{- messages[0].content }}"
)

MESSAGES = [{"role": "system", "content": "framing"},
            {"role": "user", "content": "STATE:\nbroken login"}]


def facts(**kwargs) -> dict:
    base = {"arch": "spark2_5", "model_template": SPARK_LIKE, "messages": MESSAGES}
    base.update(kwargs)
    return base


def builtin_stub(text: str = "<builtin-render/>"):
    def render(template: str, messages, add_ass: bool) -> str:
        return text
    return render


# =========================================================== A-E1c-1: provenance
def test_step_one_provenance_is_pinned_field_by_field() -> None:
    """`engine.template` in every response is this dict — string mutants here are observable."""
    assert tpl.resolve(**facts()).to_dict() == {
        "kind": "gguf-renderer", "renderer": "internal",
        "source": "gguf:tokenizer.chat_template", "family": "spark2_5",
        "thinking": "suppressed", "warnings": [], "notes": [], "explicit": False,
    }


def test_step_two_provenance_carries_the_fallback_warning_and_its_reason() -> None:
    resolution = tpl.resolve(**facts(model_template=UNSUPPORTED_LIKE,
                                     builtin_renderer=builtin_stub()))
    assert resolution.to_dict() == {
        "kind": "builtin", "renderer": "builtin", "source": "llama_chat_apply_template",
        "family": "spark2_5", "thinking": "suppressed",
        "warnings": ["W_TEMPLATE_FALLBACK"],
        "notes": ["the internal renderer rejected this template; the runtime's built-in "
                  "family table rendered it"],
        "explicit": False,
    }


def test_the_plain_override_provenance_is_pinned(tmp_path: pathlib.Path) -> None:
    resolution = tpl.resolve(**facts(model_template=None, user_template="plain"))
    assert resolution.to_dict() == {
        "kind": "user", "renderer": "plain", "source": "user:plain", "family": "spark2_5",
        "thinking": "n/a", "warnings": [],
        "notes": ["--template plain keeps the model-agnostic framing of engine/prompt.py"],
        "explicit": True,
    }
    assert resolution.is_plain is True


def test_a_builtin_name_override_provenance_is_pinned() -> None:
    resolution = tpl.resolve(**facts(model_template=UNSUPPORTED_LIKE, user_template="chatml",
                                     explicit_user=True, builtin_renderer=builtin_stub()))
    assert resolution.to_dict() == {
        "kind": "user", "renderer": "builtin", "source": "user:chatml",
        "family": "spark2_5", "thinking": "suppressed", "warnings": [], "notes": [],
        "explicit": True,
    }
    assert resolution.renderer == "builtin" and resolution.is_plain is False


def test_a_file_override_provenance_carries_the_path(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "family.jinja"
    path.write_text(SPARK_LIKE, encoding="utf-8")
    resolution = tpl.resolve(**facts(model_template=None, user_template=str(path)))
    assert resolution.to_dict()["source"] == f"user:{path}"
    assert resolution.to_dict()["renderer"] == "internal"
    assert resolution.to_dict()["thinking"] == "suppressed"
    assert resolution.to_dict()["explicit"] is True


def test_an_inline_override_provenance_says_inline() -> None:
    inline = "{{ messages[0].content }}"
    resolution = tpl.resolve(**facts(model_template=None, user_template=inline))
    assert resolution.to_dict()["source"] == "user:inline"
    assert resolution.template == inline


def test_step_four_names_why_the_gguf_template_was_rejected() -> None:
    with pytest.raises(tpl.TemplateUnresolvedError) as excinfo:
        tpl.resolve(**facts(model_template=UNSUPPORTED_LIKE))
    assert "uses" in str(excinfo.value) and "include" in str(excinfo.value)


def test_step_four_says_not_present_when_the_gguf_has_no_template() -> None:
    with pytest.raises(tpl.TemplateUnresolvedError) as excinfo:
        tpl.resolve(**facts(model_template=None, arch="llama"))
    message = str(excinfo.value)
    assert "is not present" in message
    assert str(excinfo.value).startswith("E_TEMPLATE_UNRESOLVED: no template could render")


@pytest.mark.parametrize("case", ["builtin-name-without-runtime", "file-outside-subset",
                                  "not-a-name-file-or-inline"])
def test_every_user_override_failure_is_the_pinned_code_with_exit_code_three(
        case: str, tmp_path: pathlib.Path) -> None:
    if case == "builtin-name-without-runtime":
        kwargs = dict(model_template=None, user_template="chatml", explicit_user=True)
    elif case == "file-outside-subset":
        path = tmp_path / "bad.jinja"
        path.write_text(UNSUPPORTED_LIKE, encoding="utf-8")
        kwargs = dict(model_template=None, user_template=str(path))
    else:
        kwargs = dict(model_template=None, user_template="llama-three")
    with pytest.raises(tpl.TemplateUnresolvedError) as excinfo:
        tpl.resolve(**facts(**kwargs))
    error = excinfo.value
    assert error.code == "E_TEMPLATE_UNRESOLVED"
    assert error.code in ERROR_CODES
    assert error.exit_code == 3
    assert "--template" in str(error)


def test_the_builtin_name_error_names_the_missing_runtime_and_the_fix() -> None:
    with pytest.raises(tpl.TemplateUnresolvedError) as excinfo:
        tpl.resolve(**facts(model_template=None, user_template="chatml", explicit_user=True))
    message = str(excinfo.value)
    assert "llama_chat_apply_template" in message
    assert "typed-gguf init" in message and "TYPED_GGUF_RUNTIME_DIR" in message


def test_the_outside_subset_file_error_names_the_file_and_the_fix(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "bad.jinja"
    path.write_text(UNSUPPORTED_LIKE, encoding="utf-8")
    with pytest.raises(tpl.TemplateUnresolvedError) as excinfo:
        tpl.resolve(**facts(model_template=None, user_template=str(path)))
    message = str(excinfo.value)
    assert str(path) in message
    assert "--template plain" in message and "docs/TEMPLATES.md" in message


def test_the_unknown_override_error_lists_the_three_accepted_shapes() -> None:
    with pytest.raises(tpl.TemplateUnresolvedError) as excinfo:
        tpl.resolve(**facts(model_template=None, user_template="llama-three"))
    message = str(excinfo.value)
    assert "'llama-three'" in message
    assert "--template plain, --template chatml, or --template /path/to/family.jinja" in message


def test_the_unsupported_construct_error_carries_construct_and_line() -> None:
    with pytest.raises(tpl.UnsupportedTemplate) as excinfo:
        tpl.parse("{%- include 'x.jinja' %}")
    error = excinfo.value
    assert error.construct == "tag 'include'"
    assert error.line == 1
    assert str(error) == ("unsupported template construct \"tag 'include'\" at line 1 ('x.jinja')")


def test_an_explicit_override_wins_over_a_perfectly_good_gguf_template() -> None:
    resolution = tpl.resolve(**facts(explicit_user=True, user_template="plain"))
    assert resolution.kind == "user" and resolution.source == "user:plain"


# ============================================ A-E1c-2: suppression semantics
def test_suppress_thinking_returns_the_documented_mode_string() -> None:
    policy = tpl.FAMILIES["spark2_5"]
    assert tpl.suppress_thinking("a<think>", policy=policy) == ("a", "stripped")
    assert tpl.suppress_thinking("a</think>b", policy=policy) == ("a</think>b", "suppressed")
    assert tpl.suppress_thinking("a<think>", policy=policy, think_mode="on") == \
        ("a<think>", "on")
    assert tpl.suppress_thinking("a<think>", policy=None) == ("a", "stripped")


def test_a_soft_policy_keeps_the_text_and_reports_marker() -> None:
    policy = tpl.FAMILIES["k2-horizon"]
    assert tpl.suppress_thinking("a<think>", policy=policy) == ("a<think>", "marker")


def test_a_policy_that_strips_the_empty_block_reports_suppressed() -> None:
    policy = tpl.FAMILIES["qwen35"]
    text = "assistant<think>\n\n</think>\n\n"
    assert tpl.suppress_thinking(text, policy=policy) == ("assistant", "suppressed")


def test_strip_trailing_empty_think_block_removes_every_repeat() -> None:
    assert tpl.strip_trailing_empty_think_block("a<think></think><think></think>") == "a"
    assert tpl.strip_trailing_empty_think_block("a<think>\n</think>") == "a"
    assert tpl.strip_trailing_empty_think_block("a<think></think>b") == "a<think></think>b"
    # a non-empty block is *not* touched (its content is real reasoning)
    assert tpl.strip_trailing_empty_think_block("a<think>hm</think>") == "a<think>hm</think>"


def test_strip_trailing_open_think_drops_the_trailing_whitespace_with_the_opener() -> None:
    assert tpl.strip_trailing_open_think("a<think>  \n") == "a"
    assert tpl.strip_trailing_open_think("a<|think|>") == "a"
    # not at the end -> the text is returned untouched (even if it has trailing spaces)
    assert tpl.strip_trailing_open_think("a<think> b  ") == "a<think> b  "


def test_no_open_think_counts_nested_openers_and_the_pipe_markers() -> None:
    assert tpl.no_open_think("<think><think></think>") is False
    assert tpl.no_open_think("<|think|>") is False
    assert tpl.no_open_think("<|/think|>") is True
    assert tpl.no_open_think("</think><think></think>") is True


def test_the_think_marker_sets_are_the_documented_ones() -> None:
    assert tpl.THINK_OPENERS == ("<think>", "<|think|>")
    assert tpl.THINK_CLOSERS == ("</think>", "<|/think|>")


def test_the_soft_marker_is_appended_to_the_last_user_turn_only() -> None:
    messages = [{"role": "system", "content": "s"},
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "two"}]
    updated = tpl._append_soft_marker(messages, "/no_think")
    assert updated[3]["content"] == "two\n/no_think"
    assert updated[1]["content"] == "one"
    assert updated[0] is messages[0] and updated[2] is messages[2]


def test_apply_suppression_only_strips_when_an_opener_is_left_open() -> None:
    policy = tpl.FAMILIES["spark2_5"]
    assert tpl._apply_suppression("done</think>", policy) == "done</think>"
    assert tpl._apply_suppression("done<think>", policy) == "done"


def test_render_prompt_returns_empty_only_for_the_plain_renderer() -> None:
    plain = tpl.Resolution(kind="user", renderer="plain", source="user:plain", template=None,
                           family=None, thinking="n/a", explicit=True)
    assert tpl.render_prompt(MESSAGES, plain, add_generation_prompt=True) == ""
    internal = tpl.resolve(**facts())
    assert tpl.render_prompt(MESSAGES, internal, add_generation_prompt=True) != ""


def test_a_builtin_resolution_without_a_runtime_bridge_is_a_pinned_error() -> None:
    resolution = tpl.Resolution(kind="builtin", renderer="builtin", source="x", template="chatml",
                                family=None, thinking="n/a")
    with pytest.raises(tpl.TemplateUnresolvedError) as excinfo:
        tpl.render_prompt(MESSAGES, resolution, add_generation_prompt=True)
    assert "no runtime is loaded" in str(excinfo.value)


def test_a_resolution_without_template_text_is_a_pinned_error() -> None:
    resolution = tpl.Resolution(kind="gguf-renderer", renderer="internal", source="gguf",
                                template=None, family=None, thinking="suppressed")
    with pytest.raises(tpl.TemplateUnresolvedError) as excinfo:
        tpl.render_prompt(MESSAGES, resolution, add_generation_prompt=True)
    assert "carries no template text" in str(excinfo.value)


def test_resolution_call_forwards_only_when_a_builtin_bridge_exists() -> None:
    seen: list[tuple[str, int, bool]] = []

    def bridge(template: str, messages, add_ass: bool) -> str:
        seen.append((template, len(messages), add_ass))
        return "<ok/>"

    with_bridge = tpl.Resolution(kind="builtin", renderer="builtin", source="s", template="chatml",
                                 family=None, thinking="n/a", builtin=bridge)
    assert with_bridge("chatml", MESSAGES, True) == "<ok/>"
    assert seen == [("chatml", 2, True)]
    without = tpl.Resolution(kind="user", renderer="plain", source="user:plain", template=None,
                             family=None, thinking="n/a")
    assert without("chatml", MESSAGES, True) is None


# ============================================ A-E1c-9: the policy table shape
def test_the_family_policy_to_dict_is_the_documented_shape() -> None:
    assert tpl.FAMILIES["spark2_5"].to_dict() == {
        "arch": "spark2_5", "template_key": "tokenizer.chat_template", "thinking": "hard",
        "soft_marker": None, "strip_empty_think_block": False,
        "label_policy": "option name / level number as plain text right after the assistant "
                        "header",
        "notes": tpl.FAMILIES["spark2_5"].notes,
        "aliases": ["spark", "spark2.5", "spark-x2.5"],
    }
    k2 = tpl.FAMILIES["k2-horizon"].to_dict()
    assert k2["thinking"] == "soft" and k2["soft_marker"] == "/no_think"
    assert k2["aliases"] == ["kimi-k2", "kimi_k2", "kimi-k2-horizon", "k2"]
    assert tpl.FAMILIES["qwen35"].to_dict()["strip_empty_think_block"] is True


def test_the_alias_table_covers_every_row() -> None:
    for arch, policy in tpl.FAMILIES.items():
        for alias in (policy.arch, *policy.aliases):
            assert tpl.FAMILY_ALIASES[alias] == arch
            assert tpl.policy_for(alias).arch == arch


def test_policy_lookup_is_case_and_whitespace_tolerant() -> None:
    assert tpl.policy_for("  SPARK2.5 ") is not None
    assert tpl.policy_for("").__class__ is type(None)


def test_detect_family_falls_back_to_the_template_shape() -> None:
    assert tpl.detect_family(None, "{% if 1 %}<|im_start|>{% endif %}") == "chatml-unknown"
    assert tpl.detect_family(None, "{{ '<|im_start|>' }}enable_thinking") == "qwen-unknown"
    assert tpl.detect_family(None, "{{ '<|Bot|>' }}") == "spark-unknown"
    assert tpl.detect_family(None, "{{ '<｜start▁of▁sentence｜>' }}") == "spark-unknown"
    # an arch we have no policy for is *not* the end of the story: the shape still speaks
    assert tpl.detect_family("llama", "<|im_start|>") == "chatml-unknown"
    assert tpl.detect_family("llama", "plain text") is None
    assert tpl.detect_family("spark", None) == "spark2_5"       # a known alias wins over `None`


def test_the_documented_chain_labels_name_the_mechanism_of_each_step() -> None:
    assert tpl.CHAIN == ("gguf-renderer", "builtin", "user", "error")
    assert len(tpl.CHAIN_LABELS) == len(tpl.CHAIN)
    assert tpl.CHAIN_LABELS[1].endswith("W_TEMPLATE_FALLBACK)")
    assert "plain|<name>|<path>" in tpl.CHAIN_LABELS[2]


def test_the_mirrored_builtin_template_list_is_the_pinned_build_table() -> None:
    assert len(tpl.BUILTIN_TEMPLATES) == 54
    assert tpl.BUILTIN_TEMPLATES[0] == "chatml"
    assert tpl.BUILTIN_TEMPLATES[-1] == "solar-open"
    assert "kimi-k2" in tpl.BUILTIN_TEMPLATES and "gpt-oss" in tpl.BUILTIN_TEMPLATES
    assert len(set(tpl.BUILTIN_TEMPLATES)) == len(tpl.BUILTIN_TEMPLATES)   # no duplicates


# ================================================== A-E1c-4/5: the fit budget
def tiny_model() -> fit.ModelFacts:
    """36 layers × 4 kv heads × 512 → 147456 B/token at f16 (SPEC 2.4's reference number)."""
    return fit.ModelFacts(path="/fake/model.gguf", sha256="a" * 64, arch="spark2_5", n_layer=36,
                          n_kv_head=4, key_len=256, value_len=256, n_ctx_train=32768,
                          weights_bytes=4 * GIB)


def cpu_host(ram_gib: float = 31.0) -> fit.HostFacts:
    return fit.HostFacts(backend="cpu", ram_bytes=int(ram_gib * GIB), vram_bytes=0, n_cpu=24,
                         fingerprint="cpu:test")


def test_the_budget_comparison_is_inclusive_at_the_exact_byte() -> None:
    model, host = tiny_model(), cpu_host()
    per_token = fit.kv_bytes_per_token(36, 4, 256, 256, 2.0)
    assert per_token == 147456
    boundary = model.weights_bytes + fit.OVERHEAD_BYTES + per_token * 4096
    exact = fit.estimate_plan(model, host, n_ctx=4096, budget_bytes=boundary)
    assert exact.kv_type == "f16" and exact.est_total_bytes == boundary
    assert "W_KV_TYPE_DOWNGRADE" not in exact.warnings
    over = fit.estimate_plan(model, host, n_ctx=4096, budget_bytes=boundary - 1)
    assert over.kv_type == "q8_0" and "W_KV_TYPE_DOWNGRADE" in over.warnings


def test_the_min_ctx_floor_is_the_argument_and_not_the_default() -> None:
    model, host = tiny_model(), cpu_host()
    assert fit.estimate_plan(model, host, n_ctx=100, min_ctx=2048).n_ctx == 2048
    assert fit.estimate_plan(model, host, n_ctx=100).n_ctx == fit.DEFAULT_N_CTX


def test_the_ladder_is_walked_at_the_exact_byte_too() -> None:
    model, host = tiny_model(), cpu_host()
    q8 = fit.kv_bytes_per_token(36, 4, 256, 256, fit.KV_BYTES_PER_ELEMENT["q8_0"])
    boundary = model.weights_bytes + fit.OVERHEAD_BYTES + q8 * 4096
    plan = fit.estimate_plan(model, host, n_ctx=4096, budget_bytes=boundary)
    assert plan.kv_type == "q8_0" and plan.est_kv_bytes == q8 * 4096
    assert plan.est_total_bytes == boundary
    assert fit.estimate_plan(model, host, n_ctx=4096, budget_bytes=boundary - 1).kv_type == "q4_0"


def test_the_default_budget_is_the_host_budget_minus_the_fit_target() -> None:
    model = tiny_model()
    host = cpu_host(ram_gib=8.0)
    plan = fit.estimate_plan(model, host, n_ctx=4096, fit_target_mb=1024)
    assert plan.budget_bytes == host.ram_bytes - 1024 * MIB


def test_an_explicit_budget_replaces_the_host_budget_entirely() -> None:
    plan = fit.estimate_plan(tiny_model(), cpu_host(), budget_bytes=7 * GIB)
    assert plan.budget_bytes == 7 * GIB


def test_a_negative_default_budget_floors_at_zero_not_at_a_negative() -> None:
    """`max(0, host - target)`: a tiny host must not produce a negative budget."""
    host = cpu_host(ram_gib=0.5)
    plan = fit.estimate_plan(tiny_model(), host, fit_target_mb=1024)
    assert plan.budget_bytes == 0
    assert plan.insufficient is True


def test_the_kv_ladder_is_the_documented_order() -> None:
    assert fit.KV_DOWNGRADE_ORDER == ("f16", "q8_0", "q4_0")
    assert set(fit.KV_BYTES_PER_ELEMENT) == {"f16", "q8_0", "q4_0"}
    assert fit.KV_BYTES_PER_ELEMENT["f16"] == 2.0
    assert fit.KV_BYTES_PER_ELEMENT["q8_0"] == 34 / 32
    assert fit.KV_BYTES_PER_ELEMENT["q4_0"] == 18 / 32


def test_an_explicit_kv_type_starts_the_ladder_at_that_rung() -> None:
    model, host = tiny_model(), cpu_host()
    plan = fit.estimate_plan(model, host, n_ctx=4096, kv_type="q8_0", budget_bytes=1)
    assert plan.kv_type == "q4_0"                       # never upgraded back to f16
    assert "W_KV_TYPE_DOWNGRADE" in plan.warnings       # q8_0 -> q4_0 is a downgrade
    # q4_0 is already the last rung: the budget shrinks the context instead of the type,
    # and the floor is `min_ctx` (default DEFAULT_N_CTX), not MIN_CTX_FLOOR
    last = fit.estimate_plan(model, host, n_ctx=4096, kv_type="q4_0", budget_bytes=1)
    assert last.kv_type == "q4_0"
    assert last.warnings == ("W_FIT_ESTIMATED",)
    assert last.n_ctx == fit.DEFAULT_N_CTX
    explicit_floor = fit.estimate_plan(model, host, n_ctx=4096, kv_type="q4_0", budget_bytes=1,
                                       min_ctx=1024)
    assert explicit_floor.n_ctx == 1024


# ================================================== A-E1c-4: the binary path
def test_kv_from_budget_keeps_the_binarys_context_number_for_f16() -> None:
    model = tiny_model()
    per_token = fit.kv_bytes_per_token(36, 4, 256, 256, 2.0)
    budget = model.weights_bytes + fit.OVERHEAD_BYTES + per_token * 4096
    assert fit._kv_from_budget(model, 4096, budget, 512 * MIB, "auto") == ("f16", 512 * MIB)


def test_kv_from_budget_uses_our_formula_once_it_downgrades() -> None:
    model = tiny_model()
    q8 = fit.kv_bytes_per_token(36, 4, 256, 256, fit.KV_BYTES_PER_ELEMENT["q8_0"])
    budget = model.weights_bytes + fit.OVERHEAD_BYTES + q8 * 4096
    assert fit._kv_from_budget(model, 4096, budget, 512 * MIB, "auto") == ("q8_0", q8 * 4096)


def test_kv_from_budget_exhausted_ladder_returns_q4_0_with_our_number() -> None:
    model = tiny_model()
    q4 = fit.kv_bytes_per_token(36, 4, 256, 256, fit.KV_BYTES_PER_ELEMENT["q4_0"])
    assert fit._kv_from_budget(model, 4096, 1, 512 * MIB, "auto") == ("q4_0", q4 * 4096)


def test_kv_from_budget_with_an_explicit_type_never_consults_the_budget() -> None:
    model = tiny_model()
    q4 = fit.kv_bytes_per_token(36, 4, 256, 256, fit.KV_BYTES_PER_ELEMENT["q4_0"])
    assert fit._kv_from_budget(model, 4096, 1, 512 * MIB, "q4_0") == ("q4_0", q4 * 4096)


def test_the_binary_plan_sums_the_table_and_notes_where_it_came_from() -> None:
    plan = fit.plan_from_binary(tiny_model(), cpu_host(),
                                table="Host 4096 512 128\n", n_ctx=4096, n_seq_max=8,
                                runtime_dir="/rt/llama-b11026-bin-ubuntu-x64-cpu",
                                budget_bytes=64 * GIB)
    assert plan.source == "llama-fit-params"
    assert plan.est_weights_bytes == 4096 * MIB
    assert plan.est_kv_bytes == 512 * MIB                 # the binary's own context number
    assert plan.est_total_bytes == (4096 + 512 + 128) * MIB
    assert plan.warnings == ()
    # v2 (SPEC-context-v2 §5.3.3): the table note stays first, the policy's arithmetic note joins it
    assert plan.notes[0] == ("memory table from llama-b11026-bin-ubuntu-x64-cpu/llama-fit-params "
                             "(model 4096 MiB, context 512 MiB, compute 128 MiB)")
    assert plan.standard_n_ctx == fit.STANDARD_N_CTX == 32768
    assert plan.ctx_limit == "shrunk"                 # the fixture's 32768 window is not reached
    assert any(note.startswith("n_ctx shrunk 32768 -> ") for note in plan.notes[1:])
    assert plan.n_gpu_layers == 0 and plan.n_ctx == 4096 and plan.n_seq_max == 8
    assert plan.backend == "cpu" and plan.kv_type == "f16"


def test_the_binary_plan_flags_a_downgrade_when_the_ladder_moved() -> None:
    # a budget that holds our q8_0 row but not the f16 one (weights are part of the budget)
    budget = tiny_model().weights_bytes + fit.OVERHEAD_BYTES + 400 * MIB
    plan = fit.plan_from_binary(tiny_model(), cpu_host(), table="Host 4096 512 128\n",
                                n_ctx=4096, n_seq_max=8, runtime_dir=None,
                                budget_bytes=budget)
    assert plan.kv_type == "q8_0"
    assert plan.warnings == ("W_KV_TYPE_DOWNGRADE",)
    assert plan.notes[0].startswith("memory table from llama-fit-params/")
    assert plan.est_kv_bytes != 512 * MIB                 # our formula, not the binary's row
    assert plan.est_kv_bytes == fit.kv_bytes_per_token(
        36, 4, 256, 256, fit.KV_BYTES_PER_ELEMENT["q8_0"]) * 4096


def test_a_gpu_host_offloads_every_layer_in_the_binary_path() -> None:
    gpu = fit.HostFacts(backend="vulkan", ram_bytes=31 * GIB, vram_bytes=8 * GIB, n_cpu=24,
                        fingerprint="vulkan:test")
    plan = fit.plan_from_binary(tiny_model(), gpu, table="Host 1024 64 32\n", n_ctx=4096,
                                n_seq_max=8, runtime_dir=None, budget_bytes=16 * GIB)
    assert plan.n_gpu_layers == 36 and plan.backend == "vulkan"


def test_parse_fit_table_skips_short_and_non_numeric_lines() -> None:
    rows = fit.parse_fit_table("\n".join((
        "not a row",                       # three fields -> skipped
        "Host 4096 512 128",
        "bundle 1 2",                      # too few numbers -> skipped
        "blk 12 300 500 42",               # the last three fields are the numbers
        "  ",
        "Host 4096 512 x",                 # non-numeric tail -> skipped
    )))
    assert [row.name for row in rows] == ["Host", "blk 12"]
    assert rows[1].model_bytes == 300 * MIB
    assert rows[1].context_bytes == 500 * MIB
    assert rows[1].compute_bytes == 42 * MIB
    assert rows[1].total_bytes == (300 + 500 + 42) * MIB


def test_an_empty_table_is_not_a_plan() -> None:
    assert fit.run_llama_fit_params(tiny_model(), cpu_host(), runtime_dir=None, n_ctx=4096,
                                    n_seq_max=8,
                                    runner=lambda argv: "no rows here at all\n") is None


def test_the_fit_binary_argv_is_the_spec_invocation() -> None:
    seen: list[list[str]] = []

    def runner(argv: list[str]) -> str:
        seen.append(list(argv))
        return "Host 1024 64 32\n"

    plan = fit.run_llama_fit_params(tiny_model(), cpu_host(), runtime_dir=None, n_ctx=8192,
                                    n_seq_max=4, fit_target_mb=2048, min_ctx=1024,
                                    runner=runner)
    assert plan is not None
    argv = seen[0]
    assert argv[1:3] == ["-m", "/fake/model.gguf"]
    assert argv[3:9] == ["--fit", "on", "--fit-target", "2048", "--fit-ctx", "1024"]
    assert argv[9:11] == ["--fit-print", "on"]
    assert argv[11:13] == ["-c", "8192"]
    assert argv[13:15] == ["-b", "8192"]                  # batch = max(512, n_ctx)
    assert argv[15:17] == ["-ub", "512"]                  # ub = min(512, batch)
    assert argv[17:19] == ["-ngl", "0"]                   # no VRAM on this host


# ================================================== A-E1c-4: cache + facts
def test_host_facts_fingerprint_is_the_pinned_sixteen_hex_digest(tmp_path: pathlib.Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       32768000 kB\nMemFree:  1 kB\n", encoding="utf-8")
    host = fit.host_facts(meminfo_path=meminfo, vram_probe=lambda: 0, backend="cpu", n_cpu=7)
    payload = (f"backend=cpu|ram={32768000 * 1024}|vram=0|cpus=7"
               f"|arch={platform.machine()}|os={platform.system().lower()}")
    assert host.fingerprint == hashlib.sha256(payload.encode()).hexdigest()[:16]
    assert len(host.fingerprint) == 16
    assert host.ram_bytes == 32768000 * 1024 and host.vram_bytes == 0 and host.n_cpu == 7


def test_the_host_budget_prefers_vram_when_the_probe_reports_it(tmp_path: pathlib.Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       1024 kB\n", encoding="utf-8")
    gpu = fit.host_facts(meminfo_path=meminfo, vram_probe=lambda: 6 * GIB, backend="vulkan",
                         n_cpu=2)
    assert gpu.budget_bytes == 6 * GIB
    cpu = fit.host_facts(meminfo_path=meminfo, vram_probe=lambda: 0, backend="cpu", n_cpu=2)
    assert cpu.budget_bytes == 1024 * 1024


def test_gpu_layers_is_the_floored_per_layer_split() -> None:
    model = tiny_model()
    gpu = fit.HostFacts(backend="vulkan", ram_bytes=31 * GIB, vram_bytes=8 * GIB, n_cpu=24,
                        fingerprint="vulkan:test")
    assert fit._gpu_layers(model, gpu, kv_bytes=0, budget=2 * GIB, overhead_bytes=0) == 18
    assert fit._gpu_layers(model, gpu, kv_bytes=0, budget=64 * GIB, overhead_bytes=0) == 36
    assert fit._gpu_layers(model, gpu, kv_bytes=0, budget=0, overhead_bytes=0) == 0
    assert fit._gpu_layers(model, gpu, kv_bytes=8 * GIB, budget=8 * GIB, overhead_bytes=0) == 0
    assert fit._gpu_layers(model, cpu_host(), kv_bytes=0, budget=64 * GIB, overhead_bytes=0) == 0
    empty = fit.ModelFacts(path="/f.g", sha256="", arch=None, n_layer=0, n_kv_head=1, key_len=1,
                           value_len=1, n_ctx_train=0, weights_bytes=0)
    assert fit._gpu_layers(empty, gpu, kv_bytes=0, budget=64 * GIB, overhead_bytes=0) == 0


def test_kv_bytes_per_token_rounds_the_real_block_ratio() -> None:
    assert fit.kv_bytes_per_token(36, 4, 256, 256, 2.0) == 147456          # SPEC 2.4, f16
    assert fit.kv_bytes_per_token(36, 4, 256, 256, 34 / 32) == 78336       # q8_0 block cost
    assert fit.kv_bytes_per_token(36, 4, 256, 256, 18 / 32) == 41472       # q4_0 block cost
    assert fit.kv_bytes_per_token(0, 4, 256, 256, 2.0) == 0
    assert fit.kv_bytes_per_token(36, 4, 0, 0, 2.0) == 0


def test_size_of_tensor_pads_to_the_block_boundary() -> None:
    assert fit.size_of_tensor([3], 8) == 34               # q8_0: one padded block
    assert fit.size_of_tensor([64], 8) == 2 * 34
    assert fit.size_of_tensor([5], 1) == 10               # f16: 1 element per block
    assert fit.size_of_tensor([64, 32], 8) == (2048 // 32) * 34


def test_an_unknown_ggml_type_names_the_known_range() -> None:
    with pytest.raises(fit.UnknownTensorType) as excinfo:
        fit.size_of_tensor([1], 5)                        # 5 = removed Q4_2/Q4_3
    message = str(excinfo.value)
    assert "unknown ggml tensor type 5" in message
    assert "known types: 0..39" in message


def test_kv_int_prefers_the_arch_key_then_general() -> None:
    assert fit._kv_int({"spark2_5.block_count": 3, "general.block_count": 5},
                       "spark2_5", "block_count") == 3
    assert fit._kv_int({"general.block_count": 5}, "spark2_5", "block_count") == 5
    assert fit._kv_int({"general.block_count": 5}, None, "block_count") == 5
    assert fit._kv_int({"spark2_5.block_count": "3"}, "spark2_5", "block_count") is None
    assert fit._kv_int({}, "spark2_5", "block_count") is None


def test_fit_plan_round_trips_through_its_own_json() -> None:
    plan = fit.estimate_plan(tiny_model(), cpu_host(), n_ctx=2048, n_seq_max=4)
    payload = json.loads(json.dumps(plan.to_dict()))
    assert payload["schema"] == fit.FIT_SCHEMA == "typed_gguf.fit/v1"
    restored = fit.FitPlan.from_dict(payload)
    assert restored.to_dict() == payload
    assert isinstance(restored.warnings, tuple) and "W_FIT_ESTIMATED" in restored.warnings


def test_fit_fields_are_exactly_the_documented_names() -> None:
    """AC-11 (SPEC-context-v2 §5.1): the nine original names, then the two v2 policy fields."""
    assert tuple(fit.FIT_FIELDS) == ("n_gpu_layers", "n_ctx", "kv_type", "n_seq_max",
                                     "est_weights_bytes", "est_kv_bytes", "est_total_bytes",
                                     "backend", "source", "standard_n_ctx", "ctx_limit")


def test_load_cached_rejects_a_foreign_schema_version(tmp_path: pathlib.Path) -> None:
    path = fit.cache_path("a" * 64, "fp", home=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "typed_gguf.fit/v2", "n_ctx": 1}), encoding="utf-8")
    assert fit.load_cached("a" * 64, "fp", home=tmp_path) is None


def test_load_cached_rejects_a_truncated_or_missing_payload(tmp_path: pathlib.Path) -> None:
    path = fit.cache_path("b" * 64, "fp", home=tmp_path)
    assert fit.load_cached("b" * 64, "fp", home=tmp_path) is None      # nothing written yet
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert fit.load_cached("b" * 64, "fp", home=tmp_path) is None
    path.write_text(json.dumps({"schema": fit.FIT_SCHEMA}), encoding="utf-8")
    assert fit.load_cached("b" * 64, "fp", home=tmp_path) is None      # required key missing


def test_store_plan_needs_the_sha_and_leaves_no_temp_file(tmp_path: pathlib.Path) -> None:
    plan = fit.estimate_plan(tiny_model(), cpu_host())
    with pytest.raises(ValueError) as excinfo:
        fit.store_plan(dataclasses.replace(plan, model_sha256=""), home=tmp_path)
    assert "needs the model sha256" in str(excinfo.value)
    written = fit.store_plan(plan, home=tmp_path)
    assert written == fit.cache_path(plan.model_sha256, plan.host_fingerprint, home=tmp_path)
    assert written.exists()
    assert [entry.name for entry in written.parent.iterdir()] == [written.name]


def test_the_cache_key_is_the_model_sha_and_the_host_fingerprint(tmp_path: pathlib.Path) -> None:
    assert fit.cache_path("abc", "h1", home=tmp_path) == tmp_path / "fit" / "abc.h1.json"
    assert fit.cache_path("abc", "h2", home=tmp_path) != fit.cache_path("abc", "h1",
                                                                        home=tmp_path)


# ================================================== A-E1c-6: RSS + overrides
def test_rss_ratio_pins_the_formula_and_the_zero_guard() -> None:
    assert fit.rss_ratio(1000, 1100) == pytest.approx(0.1)
    assert fit.rss_ratio(1000, 900) == pytest.approx(-0.1)
    assert fit.rss_ratio(0, 900) == float("inf")


def test_within_tolerance_is_inclusive_at_the_boundary() -> None:
    assert fit.within_tolerance(1000, 1200) is True
    assert fit.within_tolerance(1000, 800) is True
    assert fit.within_tolerance(1000, 1201) is False
    assert fit.within_tolerance(1000, 1000) is True
    assert fit.within_tolerance(1000, 1000, tolerance=0.0) is True


def test_measured_rss_reads_this_process_and_degrades_to_none() -> None:
    mine = fit.measured_rss_bytes()
    assert mine is not None and mine > 0
    later = fit.measured_rss_bytes(os.getpid())
    assert later is not None and later > 0
    # the same process read twice: RSS moves while the suite allocates, so bound the delta
    assert abs(later - mine) < 256 * MIB
    assert fit.measured_rss_bytes(pid=999_999_999) is None


def test_session_overrides_let_the_request_win_every_field() -> None:
    plan = fit.estimate_plan(tiny_model(), cpu_host(), n_ctx=2048, n_seq_max=4)
    assert fit.session_overrides(plan, n_ctx=None, kv_type="auto", n_seq_max=None) == {
        "n_ctx": plan.n_ctx, "kv_type": plan.kv_type, "n_seq_max": plan.n_seq_max,
        "n_gpu_layers": plan.n_gpu_layers}
    assert fit.session_overrides(plan, n_ctx=32768, kv_type="q4_0", n_seq_max=1) == {
        "n_ctx": 32768, "kv_type": "q4_0", "n_seq_max": 1,
        "n_gpu_layers": plan.n_gpu_layers}


def test_a_plan_without_a_budget_note_is_not_insufficient() -> None:
    plan = fit.estimate_plan(tiny_model(), cpu_host())
    assert plan.insufficient is False
    loud = dataclasses.replace(plan, notes=("insufficient device memory: ...",))
    assert loud.insufficient is True


# ============================================== GGUF tensor index (fit's own reader)
def gstr(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def kv(key: str, type_id: int, payload: bytes) -> bytes:
    return gstr(key) + struct.pack("<I", type_id) + payload


def u32(value: int) -> bytes:
    return struct.pack("<I", value)


def array_payload(item_type: int, items: list[bytes]) -> bytes:
    return struct.pack("<I", item_type) + struct.pack("<Q", len(items)) + b"".join(items)


def write_gguf_with_arrays(path: pathlib.Path) -> pathlib.Path:
    """A GGUF whose KV block holds string/int arrays and a nested array — real models do."""
    nested = array_payload(4, [u32(1), u32(2)])
    kvs = [
        kv("general.architecture", 8, gstr("spark2_5")),
        kv("general.name", 8, gstr("synthetic")),
        kv("tokenizer.ggml.tokens", 9, array_payload(8, [gstr("a"), gstr("bb")])),
        kv("tokenizer.ggml.token_type", 9, array_payload(4, [u32(1), u32(2)])),
        kv("general.attention.order", 9, array_payload(9, [nested, nested])),
        kv("spark2_5.block_count", 4, u32(2)),
        kv("spark2_5.attention.head_count_kv", 4, u32(4)),
        kv("spark2_5.attention.key_length", 4, u32(256)),
        kv("spark2_5.attention.value_length", 4, u32(256)),
        kv("spark2_5.context_length", 4, u32(4096)),
        kv("general.file_type", 4, u32(7)),
        kv("spark2_5.rope.freq_base", 6, struct.pack("<f", 10000.0)),
        kv("general.quantization_version", 4, u32(2)),
    ]
    tensors = [("token_embd.weight", [64, 32], 8), ("blk.0.attn_norm.weight", [64], 0)]
    blob = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors))
            + struct.pack("<Q", len(kvs)) + b"".join(kvs))
    index = b""
    for name, dims, ttype in tensors:
        index += gstr(name) + u32(len(dims))
        for dim in dims:
            index += struct.pack("<Q", dim)
        index += u32(ttype) + struct.pack("<Q", 0)
    path.write_bytes(blob + index + b"\0" * 64)
    return path


def test_read_tensor_index_walks_arrays_to_reach_the_index(tmp_path: pathlib.Path) -> None:
    path = write_gguf_with_arrays(tmp_path / "arr.gguf")
    rows = fit.read_tensor_index(path)
    assert rows == [("token_embd.weight", [64, 32], 8, 0), ("blk.0.attn_norm.weight", [64], 0, 0)]
    model = fit.ModelFacts.read(path, want_sha256=False)
    assert model.arch == "spark2_5"
    assert model.n_layer == 2 and model.n_kv_head == 4 and model.key_len == 256
    assert model.n_ctx_train == 4096
    assert model.weights_bytes == fit.size_of_tensor([64, 32], 8) + fit.size_of_tensor([64], 0)


def test_a_non_gguf_file_is_reported_not_guessed(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "not.gguf"
    path.write_bytes(b"NOPE" + b"\0" * 32)
    with pytest.raises(Exception) as excinfo:
        fit.read_tensor_index(path)
    assert "E_GGUF_CORRUPT" in str(excinfo.value)


def test_a_missing_model_path_is_the_pinned_code(tmp_path: pathlib.Path) -> None:
    with pytest.raises(Exception) as excinfo:
        fit.ModelFacts.read(tmp_path / "absent.gguf")
    assert "E_MODEL_NOT_FOUND" in str(excinfo.value)


def test_read_tensor_index_rejects_an_unknown_metadata_value_type(
        tmp_path: pathlib.Path) -> None:
    """A KV whose value type this reader does not know must stop the parse, not mis-skip it."""
    path = tmp_path / "weird.gguf"
    kvs = [kv("general.architecture", 8, gstr("spark2_5")), kv("odd", 99, u32(1))]
    blob = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
            + struct.pack("<Q", len(kvs)) + b"".join(kvs))
    path.write_bytes(blob)
    with pytest.raises(Exception) as excinfo:
        fit.read_tensor_index(path)
    assert "unknown metadata value type 99" in str(excinfo.value)


def test_the_plan_warnings_are_all_in_the_frozen_catalog() -> None:
    for plan in (fit.estimate_plan(tiny_model(), cpu_host(), budget_bytes=1),
                 fit.plan_from_binary(tiny_model(), cpu_host(), table="Host 4096 512 128\n",
                                      n_ctx=512, n_seq_max=1, runtime_dir=None, budget_bytes=1)):
        assert plan.warnings
        for code in plan.warnings:
            assert code in WARNING_CODES, code
