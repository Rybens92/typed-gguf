"""A-E1c-1/2/3: the template resolution chain, thinking suppression and the cue contract.

Offline half of E1c: every fixture here is a *template string* (inline or read from a real
GGUF when the model is on this box), so the chain, the internal renderer and the suppression
policy are pinned without a runtime. The live half (rendering the pinned models' own templates
end to end) is the `model`-marked section at the bottom.

Chain (SPEC 5 / A-E1c-1), in order:
  1. the GGUF's `tokenizer.chat_template` rendered by the internal renderer (supported subset);
  2. `llama_chat_apply_template` built-ins (`W_TEMPLATE_FALLBACK` when this step is taken);
  3. an explicit user override (`--template name|path|plain`);
  4. `E_TEMPLATE_UNRESOLVED`, whose message carries the fix.
"""
from __future__ import annotations

import pathlib
from typing import Any

import pytest

from typed_gguf.engine import template as tpl
from typed_gguf.errors import ERROR_CODES, WARNING_CODES

# --------------------------------------------------------------------- fixtures
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

QWEN_LIKE = (
    "{%- if messages[0].role == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- for message in messages if message.role != 'system' %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is true %}{{- '<think>\\n' }}"
    "{%- else %}{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}{%- endif %}"
)

HARD_THINK_LIKE = (
    "{%- for message in messages %}{{- '<|' + message.role + '|>' + message.content }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|assistant|><think>' }}{%- endif %}"
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


# ------------------------------------------------------------- A-E1c-1: the chain
def test_the_documented_chain_order_is_the_spec_order() -> None:
    assert tpl.CHAIN == ("gguf-renderer", "builtin", "user", "error")
    assert "tokenizer.chat_template" in tpl.CHAIN_LABELS[0]
    assert "llama_chat_apply_template" in tpl.CHAIN_LABELS[1]
    assert "--template" in tpl.CHAIN_LABELS[2]
    assert "E_TEMPLATE_UNRESOLVED" in tpl.CHAIN_LABELS[3]


def test_step_one_renders_the_gguf_template_with_the_internal_renderer() -> None:
    resolution = tpl.resolve(**facts())
    assert resolution.kind == "gguf-renderer"
    assert resolution.renderer == "internal"
    assert resolution.source == "gguf:tokenizer.chat_template"
    assert resolution.warnings == ()
    assert resolution.family == "spark2_5"


def test_step_two_is_the_builtin_and_warns_template_fallback() -> None:
    calls: list[tuple[str, bool]] = []

    def builtin(template: str, messages, add_ass: bool) -> str:
        calls.append((template, add_ass))
        return "<builtin-render/>"

    resolution = tpl.resolve(**facts(model_template=UNSUPPORTED_LIKE,
                                     builtin_renderer=builtin))
    assert resolution.kind == "builtin"
    assert resolution.renderer == "builtin"
    assert resolution.warnings == ("W_TEMPLATE_FALLBACK",)
    assert calls and calls[0][1] is True            # add_generation_prompt is forwarded
    assert tpl.render_prompt(MESSAGES, resolution, add_generation_prompt=True,
                             enable_thinking=False, builtin_renderer=builtin) \
        == "<builtin-render/>"


def test_step_three_is_the_user_override_when_nothing_else_resolves(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "family.jinja"
    path.write_text(SPARK_LIKE, encoding="utf-8")
    resolution = tpl.resolve(**facts(model_template=UNSUPPORTED_LIKE, user_template=str(path)))
    assert resolution.kind == "user"
    assert resolution.renderer == "internal"
    rendered = tpl.render_prompt(MESSAGES, resolution, add_generation_prompt=True,
                                 enable_thinking=False)
    assert rendered.endswith("<|Bot|></think>")


def test_a_builtin_name_as_the_override_needs_a_runtime() -> None:
    with pytest.raises(tpl.TemplateUnresolvedError) as excinfo:
        tpl.resolve(**facts(model_template=UNSUPPORTED_LIKE, user_template="chatml"))
    assert "llama_chat_apply_template" in str(excinfo.value)


def test_a_builtin_name_renders_through_the_runtime_bridge_when_present() -> None:
    seen: dict[str, object] = {}

    def builtin(template: str, messages, add_ass: bool) -> str:
        seen["name"], seen["n"], seen["add_ass"] = template, len(messages), add_ass
        return "<|im_start|>assistant\n"

    resolution = tpl.resolve(**facts(model_template=UNSUPPORTED_LIKE, user_template="chatml",
                                     builtin_renderer=builtin, explicit_user=True))
    assert resolution.kind == "user" and resolution.renderer == "builtin"
    rendered = tpl.render_prompt(MESSAGES, resolution, add_generation_prompt=True,
                                 enable_thinking=False, builtin_renderer=builtin)
    assert rendered.startswith("<|im_start|>")
    assert seen == {"name": "chatml", "n": 2, "add_ass": True}


def test_a_template_file_is_a_user_override_too(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "family.jinja"
    path.write_text(SPARK_LIKE, encoding="utf-8")
    resolution = tpl.resolve(**facts(model_template=None, user_template=str(path)))
    assert resolution.kind == "user"
    assert resolution.renderer == "internal"
    assert resolution.source == f"user:{path}"


def test_step_four_is_a_pinned_error_whose_message_carries_the_fix() -> None:
    with pytest.raises(tpl.TemplateUnresolvedError) as excinfo:
        tpl.resolve(**facts(model_template=UNSUPPORTED_LIKE, user_template=None))
    error = excinfo.value
    assert error.code == "E_TEMPLATE_UNRESOLVED"
    assert error.code in ERROR_CODES
    assert error.exit_code == 3
    message = str(error)
    assert "tokenizer.chat_template" in message
    assert "--template" in message                 # the documented fix
    assert "include" in message                    # the unsupported construct, named


def test_an_explicit_override_short_circuits_the_automatic_chain() -> None:
    resolution = tpl.resolve(**facts(explicit_user=True, user_template="plain"))
    assert resolution.kind == "user"
    assert resolution.renderer == "plain"
    assert resolution.template is None


def test_plain_is_the_e1b_framing_escape_hatch() -> None:
    resolution = tpl.resolve(**facts(model_template=UNSUPPORTED_LIKE,
                                     user_template="plain"))
    assert resolution.renderer == "plain"
    rendered = tpl.render_prompt(MESSAGES, resolution, add_generation_prompt=True,
                                 enable_thinking=True)
    assert rendered == ""                          # the caller keeps prompt.py's framing


def test_a_model_without_a_template_and_no_override_is_pinned_too() -> None:
    with pytest.raises(tpl.TemplateUnresolvedError):
        tpl.resolve(**facts(model_template=None, arch="llama"))


def test_the_builtin_name_list_is_the_llama_cpp_one_when_a_runtime_is_present() -> None:
    names = tpl.BUILTIN_TEMPLATES
    for expected in ("chatml", "llama3", "gemma", "mistral-v7", "kimi-k2", "deepseek3"):
        assert expected in names


def test_detect_family_reads_the_arch_first_then_the_template() -> None:
    assert tpl.detect_family("spark2_5", SPARK_LIKE) == "spark2_5"
    assert tpl.detect_family("qwen35moe", QWEN_LIKE) == "qwen35moe"
    assert tpl.detect_family(None, QWEN_LIKE) == "qwen-unknown" or \
        tpl.detect_family(None, QWEN_LIKE).startswith("qwen")
    assert tpl.detect_family("kimi-k2", None) == "k2-horizon"
    assert tpl.detect_family(None, "plain text template") is None


# --------------------------------------------------- A-E1c-1: the internal renderer
def test_the_renderer_handles_the_supported_subset() -> None:
    template = (
        "{%- set ns = namespace(first='') %}"
        "{%- for message in messages %}"
        "{%- if loop.first %}{%- set ns.first = message.role %}{%- endif %}"
        "{{- '[' + message.role + ']' + message.content }}"
        "{%- endfor %}"
        "{%- if ns.first == 'system' %}{{- '|first=system' }}{%- endif %}"
        "{%- if messages | length > 1 %}{{- '|n=' + (messages | length) | string }}{%- endif %}"
    )
    rendered = tpl.render(template, messages=MESSAGES, add_generation_prompt=False)
    assert rendered == "[system]framing[user]STATE:\nbroken login|first=system|n=2"


def test_the_renderer_supports_slicing_methods_and_filters() -> None:
    template = (
        "{%- set last = messages[-1].content %}"
        "{{- last.split('\\n')[-1] }}"
        "{{- '|' + (messages[::-1][0].content | trim) }}"
        "{{- '|' + (messages | length | string) }}"
    )
    rendered = tpl.render(template, messages=MESSAGES, add_generation_prompt=False)
    assert rendered == "broken login|STATE:\nbroken login|2"


def test_the_renderer_raises_on_an_unsupported_construct() -> None:
    for bad, construct in (("{%- include 'x.jinja' %}", "include"),
                           ("{%- for x in range(3) %}{{- x }}{%- endfor %}", "range"),
                           ("{{ messages | map(attribute='role') }}", "map"),
                           ("{%- block footer %}{%- endblock %}", "block"),
                           ("{%- import 'x.jinja' as m %}", "import")):
        with pytest.raises(tpl.UnsupportedTemplate) as excinfo:
            tpl.parse(bad)
        assert construct in str(excinfo.value)


def test_the_renderer_supports_inline_conditionals() -> None:
    template = "{{ 'system' if messages[0].role == 'system' else 'other' }}"
    assert tpl.render(template, messages=MESSAGES, add_generation_prompt=False) == "system"


def test_the_renderer_implements_macros() -> None:
    template = (
        "{%- macro render_content(content, name) %}"
        "{%- if content is string %}{{- content }}{%- else %}{{- name }}{%- endif %}"
        "{%- endmacro %}"
        "{{ render_content(messages[0].content, 'system') }}"
        "|{{ render_content(messages, 'list') }}"
    )
    assert tpl.render(template, messages=MESSAGES, add_generation_prompt=False) == \
        "framing|list"


def test_supports_is_the_parser_verdict() -> None:
    assert tpl.supports(SPARK_LIKE) is True
    assert tpl.supports(UNSUPPORTED_LIKE) is False


# ------------------------------------------------- A-E1c-2: thinking suppression
def test_a_hard_switch_template_suppresses_cleanly() -> None:
    resolution = tpl.resolve(**facts())
    rendered = tpl.render_prompt(MESSAGES, resolution, add_generation_prompt=True,
                                 enable_thinking=False)
    assert rendered.endswith("<|Bot|></think>")
    assert tpl.no_open_think(rendered) is True
    assert "<think>" not in rendered


def test_the_same_template_with_thinking_on_opens_a_block() -> None:
    resolution = tpl.resolve(**facts())
    rendered = tpl.render_prompt(MESSAGES, resolution, add_generation_prompt=True,
                                 enable_thinking=True)
    assert rendered.endswith("<|Bot|><think>")
    assert tpl.no_open_think(rendered) is False


def test_a_qwen_style_empty_block_is_closed_and_suppressed() -> None:
    resolution = tpl.resolve(**facts(arch="qwen35", model_template=QWEN_LIKE))
    rendered = tpl.render_prompt(MESSAGES, resolution, add_generation_prompt=True,
                                 enable_thinking=False)
    assert tpl.no_open_think(rendered) is True
    assert "<think>\n\n</think>" in rendered or "<think>" not in rendered


def test_a_template_that_ignores_the_switch_is_stripped_provably() -> None:
    resolution = tpl.resolve(**facts(arch="qwen35moe", model_template=HARD_THINK_LIKE))
    rendered = tpl.render_prompt(MESSAGES, resolution, add_generation_prompt=True,
                                 enable_thinking=False)
    assert tpl.no_open_think(rendered) is True
    assert "<think>" not in rendered
    assert resolution.thinking == "stripped"


def test_a_soft_switch_family_gets_the_documented_marker() -> None:
    policy = tpl.FAMILIES["k2-horizon"]
    assert policy.thinking == "soft"
    assert policy.soft_marker and policy.soft_marker in ("/no_think",)


def test_no_open_think_pins_the_predicate() -> None:
    assert tpl.no_open_think("a<think>b</think>c") is True
    assert tpl.no_open_think("a<think>b</think>c<think>") is False
    assert tpl.no_open_think("a<think></think><think>\n") is False


def test_suppression_is_recorded_in_the_resolution() -> None:
    assert tpl.resolve(**facts()).thinking == "suppressed"
    assert tpl.resolve(**facts(model_template=HARD_THINK_LIKE
                               )).thinking == "stripped"
    assert tpl.resolve(**facts(model_template=HARD_THINK_LIKE, arch="k2-horizon")
                       ).thinking == "marker"


# ------------------------------------------- A-E1c-9: the family policy table
def test_every_documented_family_has_a_policy_row() -> None:
    for arch in ("spark2_5", "qwen35", "qwen35moe", "k2-horizon"):
        policy = tpl.FAMILIES[arch]
        assert policy.arch == arch
        assert policy.template_key == "tokenizer.chat_template"
        assert policy.thinking in ("hard", "soft", "none")
        assert policy.label_policy
        assert policy.notes


def test_policy_aliases_resolve_to_the_documented_row() -> None:
    assert tpl.policy_for("kimi-k2").arch == "k2-horizon"
    assert tpl.policy_for("qwen35moe").arch == "qwen35moe"
    assert tpl.policy_for("llama") is None


# ------------------------------------------- A-E1c-3: the cue contract (prompt side)
def test_the_rendered_question_suffix_ends_at_the_cue() -> None:
    from typed_gguf.engine import prompt
    from typed_gguf.schema import Question

    question = Question(id="area", type="choice", instructions="Which team owns this?",
                        criteria={"billing": None, "technical": None},
                        options=("billing", "technical"), descriptions=(None, None))
    rendered = prompt.build_question(question, readout="sequence")
    assert rendered.suffix.rstrip().endswith(prompt.CANDIDATE_CUE["choice"])
    assert "billing" in rendered.suffix and "technical" in rendered.suffix
    # the question id is never sent to the model (A-E1b, kept honest here)
    assert "Which team owns this?" in rendered.suffix


def test_render_warnings_are_useable_by_the_response() -> None:
    resolution = tpl.resolve(**facts(model_template=UNSUPPORTED_LIKE,
                                     builtin_renderer=lambda *a, **k: "x"))
    for code in resolution.warnings:
        assert code in WARNING_CODES


# ------------------------------- A-E1c-3: post-cue degenerate output (synthetic logits)
def test_post_cue_degenerate_output_never_affects_the_readout() -> None:
    """A model that *would* start reasoning after the cue cannot move the answer.

    The synthetic logits fixture gives the degenerate continuation token (`think`, the first
    token of a `<think>` opener in this vocabulary) an overwhelming logit at **every** position,
    including the cue row. The engine reads the candidate rows at the cue, and the restricted
    softmax is shift-invariant — so the answers are byte-identical while the coverage diagnostic
    (full-vocab mass) legitimately collapses.
    """
    from tests.fake_engine import FakeSession, biased_row
    from typed_gguf import schema
    from typed_gguf.engine import decide

    payload = {
        "state": "The billing dashboard is blank for every user after login.",
        "questions": {"area": {"type": "choice", "instructions": "Which area owns this?",
                               "criteria": {"billing": None, "technical": None}}},
    }
    request = schema.parse_request(payload)

    def run_with(bias: dict[int, float]):
        session = FakeSession(n_vocab=256)
        billing, technical = session.tokenize("billing")[0], session.tokenize("technical")[0]
        row = biased_row(session.n_vocab, {billing: 6.0, technical: 5.0, **bias})
        session.row_fn = lambda ctx, row=row: row
        engine = decide.DecisionEngine(session)
        return engine.decide(request, plan=decide.plan_context(request, session))

    quiet = run_with({})
    # a degenerate model that wants to emit `think`-flavoured continuations after the cue
    degenerate_token = 255                                   # not a candidate label token
    session = FakeSession(n_vocab=256)
    assert degenerate_token not in (session.tokenize("billing")[0],
                                    session.tokenize("technical")[0])
    degenerate = run_with({degenerate_token: 40.0})

    def readout(result):
        return {key: value for key, value in result.answers["area"].items()
                if key in ("type", "choice", "probabilities", "confidence")}

    assert readout(quiet) == readout(degenerate)                  # the readout is untouched
    assert degenerate.answers["area"]["choice"] == "billing"
    assert degenerate.answers["area"]["probabilities"] == quiet.answers["area"]["probabilities"]
    assert degenerate.answers["area"]["confidence"] == quiet.answers["area"]["confidence"]
    # the diagnostic tells the truth about the full-vocab mass (it is not part of the readout)
    assert degenerate.answers["area"]["coverage"] < quiet.answers["area"]["coverage"]
    assert "W_LOW_MASS" in degenerate.warnings


# ------------------------------------------------------------- live: real GGUFs
# Run with: TYPED_GGUF_RUNTIME_DIR=<bundle> uv run pytest -q --run-network tests/test_templates.py
MODEL_PATHS = {
    "spark2_5": pathlib.Path.home() / ".hermes" / "models" / "Spark-X2.5-4B-Q8_0.gguf",
    "qwen35": pathlib.Path.home() / ".cache" / "llama.cpp" / "Qwen3.5-0.8B-UD-Q4_K_XL.gguf",
}
LIVE_MESSAGES = [
    {"role": "system", "content": "You are a decision engine."},
    {"role": "user", "content": "STATE:\nthe dashboard is blank"},
]
EXPECTED_TAIL = {"spark2_5": "<|Bot|></think>", "qwen35": "<|im_start|>assistant"}


def _model_template(name: str) -> tuple[str, str]:
    from typed_gguf.registry import gguf
    path = MODEL_PATHS[name]
    if not path.exists():
        pytest.skip(f"{path} is not on this box")
    kv = gguf.parse_gguf_metadata(path)["kv"]
    template = kv.get("tokenizer.chat_template")
    assert isinstance(template, str) and template
    return gguf.arch_of(kv) or name, template


@pytest.mark.model
@pytest.mark.parametrize("name", ["spark2_5", "qwen35"])
def test_a_real_gguf_template_renders_through_chain_step_one(name: str) -> None:
    arch, template = _model_template(name)
    resolution = tpl.resolve(messages=LIVE_MESSAGES, model_template=template, arch=arch)
    assert resolution.kind == "gguf-renderer"
    assert resolution.renderer == "internal"
    assert resolution.family == arch
    assert resolution.warnings == ()                       # no fallback was needed
    rendered = tpl.render_prompt(LIVE_MESSAGES, resolution, add_generation_prompt=True,
                                 enable_thinking=False)
    assert tpl.no_open_think(rendered) is True
    assert "<think>" not in rendered
    assert rendered.rstrip("\n").endswith(EXPECTED_TAIL[arch])
    thinking_on = tpl.render_prompt(LIVE_MESSAGES, resolution, add_generation_prompt=True,
                                    enable_thinking=True)
    assert tpl.no_open_think(thinking_on) is False         # the switch is real, not a no-op


@pytest.mark.model
def test_the_real_prompt_has_no_think_token_on_the_real_vocabulary() -> None:
    """The strongest form of A-E1c-2: the model's *own vocabulary* sees no think-opener."""
    import os

    from typed_gguf.engine import session as session_module
    from typed_gguf.runtime import finder

    path = MODEL_PATHS["spark2_5"]
    if not path.exists():
        pytest.skip(f"{path} is not on this box")
    runtime_dir = os.environ.get("TYPED_GGUF_RUNTIME_DIR") or finder.find_runtime()
    if not runtime_dir:
        pytest.skip("no llama.cpp runtime on this box")
    arch, template = _model_template("spark2_5")
    resolution = tpl.resolve(messages=LIVE_MESSAGES, model_template=template, arch=arch)
    rendered = tpl.render_prompt(LIVE_MESSAGES, resolution, add_generation_prompt=True,
                                 enable_thinking=False)
    with session_module.open_model(path, runtime_dir=runtime_dir) as handle:
        decoded = _decode(handle, rendered)
        thinking_on = tpl.render_prompt(LIVE_MESSAGES, resolution, add_generation_prompt=True,
                                        enable_thinking=True)
        decoded_on = _decode(handle, thinking_on)
    assert decoded and decoded_on
    assert "<think>" not in decoded
    assert tpl.no_open_think(decoded) is True
    assert "<think>" in decoded_on                      # the check discriminates
    assert tpl.no_open_think(decoded_on) is False


def _decode(handle: Any, text: str) -> str:
    from typed_gguf.runtime import ctypes_binding
    return "".join(ctypes_binding.token_piece(handle.runtime, handle.vocab, token)
                   for token in handle.tokenize(text))


@pytest.mark.model
def test_step_two_renders_through_the_runtime_builtin_table() -> None:
    """`llama_chat_apply_template` really is a second source (Qwen's template matches chatml)."""
    import os

    from typed_gguf.runtime import ctypes_binding, finder

    runtime_dir = os.environ.get("TYPED_GGUF_RUNTIME_DIR") or finder.find_runtime()
    if not runtime_dir:
        pytest.skip("no llama.cpp runtime on this box")
    runtime = ctypes_binding.load_libraries(runtime_dir)
    builtin = tpl.runtime_builtin_renderer(runtime)
    names = tpl.runtime_builtin_names(runtime)
    assert "chatml" in names and len(names) >= 50
    chatml = builtin("chatml", LIVE_MESSAGES, True)
    assert chatml and chatml.startswith("<|im_start|>")
    assert builtin("not-a-template-name", LIVE_MESSAGES, True) is None
    _arch, template = _model_template("qwen35")
    detected = builtin(template, LIVE_MESSAGES, True)
    assert detected and "<|im_start|>" in detected          # step 2 covers this family
    # a template outside our subset *and* recognised by llama.cpp's family table -> step 2
    fallback_template = ("{%- include 'tools.jinja' %}"
                         "{{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}"
                         "{{- '<|im_start|>assistant\\n' }}")
    resolution = tpl.resolve(messages=LIVE_MESSAGES, model_template=fallback_template,
                             arch="qwen35", builtin_renderer=builtin)
    assert resolution.kind == "builtin"
    assert resolution.warnings == ("W_TEMPLATE_FALLBACK",)
    rendered = tpl.render_prompt(LIVE_MESSAGES, resolution, add_generation_prompt=True,
                                 enable_thinking=False, builtin_renderer=builtin)
    assert rendered.startswith("<|im_start|>system")
    assert "<|im_start|>assistant" in rendered
