"""E3e: the two prompt levers behind frozen defaults — `json_instructed` and `role_split`.

Card t_4c48f40a, from the E3c/E3d measurements (`docs/evidence/e3c_cue_shapes_4b.md`,
`docs/evidence/e3d_cue_decision_4b.md`):

* the 4B's dominant failure at the cue is a **refusal** — its top token is a turn-closer, so the
  label mass at the readout row can never reach the floor. The fix a refusal asks for is a prompt
  shape, not a threshold;
* E3d moved the readout one token in (`two_step`) and opened the field (`json_field`) on the
  *same* question text. E3e adds the two levers that were still missing:

  * **`--cue json_instructed`** — the question says the answer is a JSON object
    (`JSON_CONTRACT`) and the assistant turn is prefilled with the opened field, so the readout
    sits at the **value row** the contract names. A row whose top token is the single-token `"`
    is classified, never left as an anonymous low-mass row: `empty_value` when the object closes
    with it, `wrong_field` when the model closes this key and fills another (`W_JSON_*`);
  * **`--chat-format role_split`** — the question block stops being prefilled into the assistant
    turn and becomes the question's **own user turn**: the context is rendered in two halves (the
    shared prefix + the question's tail), every family's template has to render that conversation,
    and a template that merges, drops or opens a thinking block is refused by name
    (`E_ROLE_SPLIT_UNSUPPORTED`) with the fallback named in the message.

Both are off by default and the *answer-sheet* bytes every published table measured may not move:
the tests below pin the frozen bytes, the new bytes, the verdicts, the placement the response
publishes, and the per-family acceptance. The engine gates fail on the pre-E3e tree (`options`
has no `chat_format`, `prompt` has no `question_block`/`role_split_render`, `cue.value_verdict`
does not exist, `W_JSON_*` are not warning codes).

The fake session is `tests/fake_engine.FakeSession` plus a JSON-punctuation vocabulary
(`JsonSession`): `engine/cue.py`'s marker map is a *vocabulary fact* (`value_markers` matches only
markers the tokenizer encodes as one token), so the fake has to encode them the way a real
vocabulary does or the verdicts it produces would not be the verdicts a real model produces.
"""
from __future__ import annotations

import pathlib

import pytest

from ggufone import cli, errors, schema
from ggufone.bench import harness, suites
from ggufone.engine import cue as cue_module
from ggufone.engine import decide, prompt
from ggufone.engine import template as template_module
from tests.fake_engine import FakeSession, biased_row

#: the bias that puts ~1.0 of a 512-slot row's mass on one token
TOP = 11.0
#: the ids this fake vocabulary reserves for the closers / JSON punctuation it encodes as ONE token
IM_END = 400
EOS = 401
END_OF_TEXT = 402
QUOTE = 403
CLOSE = 404

STATE = "The billing dashboard is blank for every user after login."
#: the per-type JSON opener (`prompt.json_opener`) the instructed shape ends its suffix with
OPENER = '{"choice": "'

#: A ChatML-shaped template, exactly the structure the pinned families use: a turn per message,
#: the generation prompt appended only when asked for. `{{-`'s whitespace control is not needed
#: here; the E1c renderer handles both.
CHATML = ("{% for message in messages %}"
          "{{ '<|im_start|>' + message['role'] + '\\n' + message['content']"
          " + '<|im_end|>' + '\\n' }}"
          "{% endfor %}"
          "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")
#: A template that renders only the *last* message: the state turn does not exist for it, so the
#: conversation the role split promises cannot be built (a real family that dropped `messages[1]`).
LAST_ONLY = "{{ messages[-1]['content'] }}"


class JsonSession(FakeSession):
    """`FakeSession` with a real special-token vocabulary: closers **and** JSON punctuation."""

    SPECIAL = {"<|im_end|>": IM_END, "</s>": EOS, "<|endoftext|>": END_OF_TEXT,
               '"': QUOTE, "}": CLOSE, ",": 405, "{": 406}

    def tokenize(self, text: str) -> list[int]:
        if text in self.SPECIAL:
            return [self.SPECIAL[text]]
        return super().tokenize(text)


def request_for(cue: str | None = None, chat_format: str | None = None, *,
                json_contract: str | None = None, qtype: str = "choice",
                instructions: str = "Which area owns this?") -> dict:
    """One dev-set-shaped request; these options are left out unless a test names one."""
    bodies = {
        "choice": ("area", {"type": "choice", "instructions": instructions,
                            "criteria": {"billing": "payments and invoices",
                                         "technical": "api and infrastructure"}}),
        "score": ("sev", {"type": "score", "instructions": "How bad?",
                          "criteria": ["cosmetic", "annoying", "blocking"]}),
        "noul": ("page", {"type": "noul", "instructions": "Should we page?",
                          "criteria": {"true": "page now", "false": "wait"}}),
    }
    qid, body = bodies[qtype]
    payload: dict = {"state": STATE, "questions": {qid: body}}
    options: dict = {}
    if cue is not None:
        options["cue"] = cue
    if chat_format is not None:
        options["chat_format"] = chat_format
    if json_contract is not None:
        options["json_contract"] = json_contract
    if options:
        payload["options"] = options
    return payload


def parsed(**kwargs) -> schema.Request:
    return schema.parse_request(request_for(**kwargs))


def chat_plan(request: schema.Request, session: FakeSession,
              *, template: str = CHATML) -> decide.ContextPlan:
    """The plan of a request rendered through a chat template (the family path, offline)."""
    resolution = template_module.Resolution(
        kind="gguf-renderer", renderer="internal", source="gguf", template=template,
        family="chatml-test", thinking="suppressed")
    return decide.plan_context(request, session, template=resolution, resolve=False)


# ==================================================================== the frozen default
def test_the_default_chat_format_is_the_answer_sheet() -> None:
    assert schema.OPTION_DEFAULTS["chat_format"] == "answer_sheet"
    assert schema.Options().chat_format == "answer_sheet"
    assert schema.CHAT_FORMATS == ("answer_sheet", "role_split")
    # the bench's own literals (harness never imports `schema`) must be the schema's values
    assert harness.DEFAULT_CHAT_FORMAT == schema.ANSWER_SHEET
    assert schema.CUE_SHAPES[0] == harness.DEFAULT_CUE
    assert harness.BenchConfig(suite="quality").chat_format == "answer_sheet"
    assert cli.BENCH_DEFAULTS["chat-format"] == "answer_sheet"


def test_an_unknown_chat_format_is_a_named_option_error() -> None:
    with pytest.raises(errors.UserError) as caught:
        schema.parse_request(request_for(chat_format="letters"))
    assert caught.value.code == "E_UNKNOWN_KEY"
    assert "chat_format" in str(caught.value)


def test_an_explicit_chat_format_survives_parsing() -> None:
    assert parsed(chat_format="role_split").options.chat_format == "role_split"


def test_the_default_cue_line_is_still_the_shipped_one_for_every_type() -> None:
    """`shipped` and `two_step` ask for a bare label — the question text may not grow a byte."""
    request = parsed()
    question = request.questions[0]
    assert prompt.question_block(question) == (
        "QUESTION:\nWhich area owns this?\nCandidates:\n- billing: payments and invoices\n"
        "- technical: api and infrastructure\nAnswer with exactly one candidate name:\n")
    assert prompt.question_block(question, cue="two_step") == prompt.question_block(question)


def test_json_field_keeps_the_e3d_bytes() -> None:
    """E3d's `json_field` is the shipped cue line plus the opener — E3e may not restyle it."""
    request = parsed()
    question = request.questions[0]
    block = prompt.question_block(question, cue="json_field")
    assert block == prompt.question_block(question)      # the cue line is the shipped one
    assert prompt.build_question(question, cue="json_field").suffix == block + '{"choice": "'


def test_json_instructed_replaces_the_cue_line_with_the_contract() -> None:
    request = parsed()
    choice, score, noul = (schema.parse_request(request_for(qtype=qtype)).questions[0]
                           for qtype in ("choice", "score", "noul"))
    assert prompt.JSON_CUES == ("json_field", "json_instructed")
    assert prompt.question_block(choice, cue="json_instructed").endswith(
        'Answer with JSON: {"choice": "<exactly one candidate name>"}\n')
    assert prompt.question_block(score, cue="json_instructed").endswith(
        'Answer with JSON: {"severity": "<exactly one level number>"}\n')
    assert prompt.question_block(noul, cue="json_instructed").endswith(
        'Answer with JSON: {"answer": "<yes or no>"}\n')
    # the shipped line is gone from the instructed shape (that is the whole point of the cue)
    assert "Answer with exactly one" not in prompt.question_block(request.questions[0],
                                                                 cue="json_instructed")


def test_the_json_framing_asks_for_the_object_the_cue_instructs() -> None:
    assert prompt.framing_for("json_instructed") == prompt.JSON_FRAMING
    for cue in ("shipped", "two_step", "json_field"):
        assert prompt.framing_for(cue) == prompt.SYSTEM_FRAMING
    assert "single JSON object" in prompt.JSON_FRAMING
    assert "JSON" not in prompt.SYSTEM_FRAMING


def test_the_answer_sheet_question_suffix_carries_the_opener_at_its_end() -> None:
    request = parsed(cue="json_instructed")
    suffix = prompt.build_question(request.questions[0], cue="json_instructed").suffix
    assert suffix.endswith(f"\n{OPENER}")
    assert suffix.count("Answer with JSON:") == 1
    assert suffix.count(OPENER) == 2          # once in the contract, once as the opened field


# ==================================================================== the role split render
def test_the_plain_role_split_prefix_is_the_answer_sheet_prefix() -> None:
    """The escape hatch (`--template plain`) adds a user turn without moving the state bytes."""
    request = parsed(chat_format="role_split")
    role = prompt.role_split_render(request.state, request.questions)
    assert role.prefix == prompt.build_prefix(request.state)
    assert role.prefix == f"{prompt.SYSTEM_FRAMING}{prompt.STATE_HEADER}{STATE}\n"
    assert role.tails[0].startswith(f"{prompt.PLAIN_USER_HEADER}QUESTION:\n")
    assert role.tails[0].endswith(f"{prompt.PLAIN_ASSISTANT_HEADER}")
    assert role.dropped == ""
    assert role.prefix == prompt.build_prefix(request.state, chat_format="role_split",
                                              questions=request.questions)


def test_role_split_renders_the_question_as_its_own_user_turn() -> None:
    request = parsed(chat_format="role_split")
    question = request.questions[0]
    role = chat_plan(request, JsonSession(n_vocab=512))
    block = prompt.question_block(question)
    full = template_module.render_prompt(
        [{"role": "system", "content": prompt.SYSTEM_FRAMING.strip()},
         {"role": "user", "content": f"{prompt.STATE_HEADER}{STATE}"},
         {"role": "user", "content": block}],
        template_module.Resolution(kind="gguf-renderer", renderer="internal", source="gguf",
                                   template=CHATML, family="chatml-test",
                                   thinking="suppressed"),
        add_generation_prompt=True, enable_thinking=False)
    # the two halves are the render, byte for byte — that is what makes the prefill legal
    assert role.role.prefix + role.role.tails[0] == full
    # the prefix is the state's own turn and does NOT carry the question
    assert role.role.prefix == (f"<|im_start|>system\n{prompt.SYSTEM_FRAMING.strip()}"
                                f"<|im_end|>\n<|im_start|>user\n{prompt.STATE_HEADER}{STATE}"
                                f"<|im_end|>\n")
    assert "QUESTION:" not in role.role.prefix
    # the tail is the question's own user turn followed by the generation prompt
    assert role.role.tails[0] == f"<|im_start|>user\n{block}<|im_end|>\n<|im_start|>assistant\n"
    assert role.role.dropped == ""


def test_role_split_carries_the_json_opener_after_the_generation_prompt() -> None:
    request = parsed(cue="json_instructed", chat_format="role_split")
    role = prompt.role_split_render(request.state, request.questions, cue="json_instructed",
                                    resolution=template_module.Resolution(
                                        kind="gguf-renderer", renderer="internal", source="gguf",
                                        template=CHATML, family="chatml-test",
                                        thinking="suppressed"))
    assert role.tails[0].endswith(f"<|im_start|>assistant\n{OPENER}")
    assert template_module.no_open_think(role.prefix + role.tails[0])


def test_the_prefix_is_shared_by_every_question_and_each_tail_is_its_own() -> None:
    request = schema.parse_request({
        "state": STATE,
        "options": {"chat_format": "role_split"},
        "questions": {
            "area": {"type": "choice", "instructions": "Which area owns this?",
                     "criteria": {"billing": "payments", "technical": "api"}},
            "sev": {"type": "score", "instructions": "How bad?",
                    "criteria": ["cosmetic", "blocking"]},
        }})
    role = prompt.role_split_render(request.state, request.questions,
                                    resolution=template_module.Resolution(
                                        kind="gguf-renderer", renderer="internal", source="gguf",
                                        template=CHATML, family="chatml-test",
                                        thinking="suppressed"))
    assert len(role.tails) == 2
    assert role.tails[0] != role.tails[1]
    for question, tail in zip(request.questions, role.tails, strict=True):
        assert prompt.question_block(question) in tail
    assert "sev" not in role.prefix and "area" not in role.prefix


def test_the_role_split_context_is_none_unless_asked_for() -> None:
    """The one place that decides *whether* a request is role-split (plan and questions agree)."""
    assert prompt.role_split_context(parsed()) is None
    assert prompt.role_split_context(parsed(cue="json_instructed")) is None
    role = prompt.role_split_context(parsed(chat_format="role_split"))
    assert role is not None and role.prefix.startswith(prompt.SYSTEM_FRAMING)


def test_a_merged_or_state_dropping_template_is_refused_by_name() -> None:
    request = parsed(chat_format="role_split")
    with pytest.raises(errors.UserError) as caught:
        prompt.role_split_render(
            request.state, request.questions,
            resolution=template_module.Resolution(
                kind="gguf-renderer", renderer="internal", source="gguf", template=LAST_ONLY,
                family="chatml-test", thinking="suppressed"))
    assert caught.value.code == "E_ROLE_SPLIT_UNSUPPORTED"
    assert "E_ROLE_SPLIT_UNSUPPORTED" in str(caught.value)
    assert "--chat-format answer_sheet" in str(caught.value)   # the fallback is named


def test_a_template_without_role_markers_is_still_accepted() -> None:
    """The acceptance is about *turns*, not markers: a bare concatenation carries both turns."""
    template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    request = parsed(chat_format="role_split")
    role = prompt.role_split_render(
        request.state, request.questions,
        resolution=template_module.Resolution(
            kind="gguf-renderer", renderer="internal", source="gguf", template=template,
            family="chatml-test", thinking="suppressed"))
    assert f"{prompt.STATE_HEADER}{STATE}" in role.prefix
    assert prompt.question_block(request.questions[0]) in role.tails[0]


def test_a_prompt_left_inside_a_thinking_block_is_refused_too(monkeypatch) -> None:
    """The guard the role split adds on top of the renderer's own guarantee.

    `template_module._render_raw` drops generation-prompt markers when thinking is off, so on the
    internal renderer no open block can survive — but the *contract* the readout depends on
    (`no_open_think`) is checked here rather than assumed, because a fallback path that skipped
    suppression would otherwise put the label row inside a thinking block, silently.
    """
    def fake_render(messages, resolution, *, add_generation_prompt, enable_thinking):
        body = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return body + ("<assistant><think>" if add_generation_prompt else "")

    monkeypatch.setattr(template_module, "render_prompt", fake_render)
    request = parsed(chat_format="role_split")
    with pytest.raises(errors.UserError) as caught:
        prompt.role_split_render(request.state, request.questions,
                                 resolution=template_module.Resolution(
                                     kind="builtin", renderer="builtin", source="bridge",
                                     template="x", family="chatml-test", thinking="n/a"))
    assert caught.value.code == "E_ROLE_SPLIT_UNSUPPORTED"
    assert "thinking block" in str(caught.value)
    # asking for thinking is what makes it legal: the readout is then *supposed* to be in the block
    role = prompt.role_split_render(request.state, request.questions, enable_thinking=True,
                                    resolution=template_module.Resolution(
                                        kind="builtin", renderer="builtin", source="bridge",
                                        template="x", family="chatml-test", thinking="on"))
    assert role.tails[0].endswith("<assistant><think>")


def test_the_engine_refuses_a_role_split_question_without_its_render() -> None:
    request = parsed(chat_format="role_split")
    with pytest.raises(errors.UserError) as caught:
        prompt.build_question(request.questions[0], chat_format="role_split")
    assert caught.value.code == "E_ROLE_SPLIT_UNSUPPORTED"


def test_the_tail_lookup_is_bounds_checked() -> None:
    role = prompt.RoleSplitRender(prefix="p", tails=("t",))
    assert role.tail_for(0) == "t"
    with pytest.raises(errors.UserError) as caught:
        role.tail_for(1)
    assert caught.value.code == "E_ROLE_SPLIT_UNSUPPORTED"


def test_the_plan_carries_the_role_split_it_rendered() -> None:
    """The executed questions come from the plan's own render (the t_6de5fc53 seam, one step on)."""
    request = parsed(chat_format="role_split", cue="json_instructed")
    plan = chat_plan(request, JsonSession(n_vocab=512))
    assert plan.chat_format == "role_split"
    assert plan.role is not None
    assert plan.template is not None and plan.template.template == CHATML
    assert len(plan.prefix_tokens) > 0                            # token ids, not bytes
    # prefix + tail bytes are what the engine will decode: the question never enters the prefix
    assert "QUESTION:" not in plan.role.prefix


# ==================================================================== the value row's verdicts
def test_the_marker_map_is_a_vocabulary_fact() -> None:
    markers = cue_module.value_markers(JsonSession(n_vocab=512).tokenize)
    assert markers[QUOTE] == "quote"
    assert markers[CLOSE] == "close"
    assert cue_module.value_markers(FakeSession(n_vocab=512).tokenize) == {}   # word-level fake
    assert cue_module.JSON_VERDICTS == ("answered", "refused", "empty_value", "wrong_field")


def test_a_candidate_token_at_the_value_row_is_an_answered_row() -> None:
    session = JsonSession(n_vocab=512)
    row = biased_row(512, {session.tokenize("billing")[0]: TOP})
    verdict = cue_module.value_verdict(row, _scale(row), {}, cue_module.value_markers(
        session.tokenize))
    assert verdict["verdict"] == "answered"
    assert verdict["refused"] is False and verdict["next"] is None


def test_a_closed_empty_value_is_named_not_low_mass() -> None:
    session = JsonSession(n_vocab=512)
    row = biased_row(512, {QUOTE: TOP})
    closer_row = biased_row(512, {CLOSE: TOP})
    verdict = cue_module.value_verdict(row, _scale(row), {}, cue_module.value_markers(
        session.tokenize), next_row=closer_row, next_scale=_scale(closer_row))
    assert verdict["verdict"] == "empty_value"
    assert verdict["token"] == QUOTE and verdict["refused"] is False
    assert verdict["next"]["marker"] == "close"


def test_a_closed_value_that_leads_to_another_key_is_wrong_field() -> None:
    session = JsonSession(n_vocab=512)
    row = biased_row(512, {QUOTE: TOP})
    other = biased_row(512, {session.tokenize("severity")[0]: TOP})
    verdict = cue_module.value_verdict(row, _scale(row), {}, cue_module.value_markers(
        session.tokenize), next_row=other, next_scale=_scale(other))
    assert verdict["verdict"] == "wrong_field"
    assert verdict["next"]["marker"] is None
    assert verdict["next"]["token"] == session.tokenize("severity")[0]


def test_a_closed_value_without_the_next_row_reads_empty() -> None:
    """`empty_value` is what the object holds up to the closing quote (the walk refines it)."""
    session = JsonSession(n_vocab=512)
    row = biased_row(512, {QUOTE: TOP})
    verdict = cue_module.value_verdict(row, _scale(row), {}, cue_module.value_markers(
        session.tokenize))
    assert verdict["verdict"] == "empty_value" and verdict["next"] is None


def test_a_turn_closer_at_the_value_row_is_a_refusal_with_its_own_hint() -> None:
    session = JsonSession(n_vocab=512)
    row = biased_row(512, {IM_END: TOP})
    verdict = cue_module.value_verdict(row, _scale(row), {IM_END: "<|im_end|>"},
                                       cue_module.value_markers(session.tokenize))
    assert verdict["verdict"] == "refused"
    assert verdict["refused"] is True and verdict["closer"] == "<|im_end|>"
    assert verdict["hint"] == cue_module.VALUE_REFUSED_HINT
    assert "value row" in verdict["hint"]


def _scale(row: list[float]) -> float:
    from ggufone.engine import readout
    return readout.logsumexp(row)


# ==================================================================== the engine, scripted rows
def scripted(request: schema.Request, *, plan: decide.ContextPlan, session: JsonSession,
             value: dict[int, float], after: dict[int, float] | None = None,
             choice: str = "billing"):
    """A fake row function for the value-row shape: the row the suffix ends on is biased `value`,
    the row after it (only decoded when the value closes) is biased `after`, and every row a step
    further on prefers `choice` (so the candidate sequences are scoreable)."""
    n_suffix = len(session.tokenize(
        prompt.build_question(request.questions[0], cue=request.options.cue,
                              chat_format=request.options.chat_format,
                              role=plan.role, index=0).suffix))
    value_position = len(plan.prefix_tokens) + n_suffix - 1
    label_token = session.tokenize(choice)[0]

    def row_fn(context) -> list[float]:
        if context.position == value_position:
            return biased_row(session.n_vocab, value)
        if after is not None and context.position == value_position + 1:
            return biased_row(session.n_vocab, after)
        return biased_row(session.n_vocab, {label_token: TOP})

    session.row_fn = row_fn
    return session


def test_the_role_split_engine_reads_the_question_in_a_user_turn() -> None:
    request = parsed(chat_format="role_split")
    session = JsonSession(n_vocab=512)
    plan = chat_plan(request, session)
    scripted(request, plan=plan, session=session, value={session.tokenize("billing")[0]: TOP})
    result = decide.DecisionEngine(session).decide(request, plan=plan)
    answer = result.answers["area"]
    assert answer["choice"] == "billing"
    assert result.engine["chat_format"]["kind"] == "role_split"
    assert result.engine["chat_format"]["question_turn"] == "user"
    assert result.engine["chat_format"]["prefix_chars"] == len(plan.role.prefix)
    assert result.engine["chat_format"]["dropped"] == plan.role.dropped
    # the shipped cue block is untouched by the placement (no value verdict on a bare-label shape)
    assert answer["cue"]["refused"] is False and "verdict" not in answer["cue"]


def test_the_answer_sheet_default_publishes_the_assistant_turn() -> None:
    request = parsed()
    session = JsonSession(n_vocab=512)
    plan = chat_plan(request, session)
    scripted(request, plan=plan, session=session, value={session.tokenize("billing")[0]: TOP})
    result = decide.DecisionEngine(session).decide(request, plan=plan)
    assert result.engine["chat_format"] == {"kind": "answer_sheet", "question_turn": "assistant",
                                            "contract": None}


def test_the_value_row_verdict_is_published_on_the_answer() -> None:
    request = parsed(cue="json_instructed", chat_format="role_split")
    session = JsonSession(n_vocab=512)
    plan = chat_plan(request, session)
    scripted(request, plan=plan, session=session, value={session.tokenize("billing")[0]: TOP})
    result = decide.DecisionEngine(session).decide(request, plan=plan)
    answer = result.answers["area"]
    assert answer["cue"]["verdict"] == "answered"
    assert answer["cue"]["refused"] is False
    assert answer["choice"] == "billing"
    # no advance token: the readout did not move past anything (the field row *is* the readout)
    assert "advance" not in answer
    assert "W_JSON_EMPTY_VALUE" not in result.warnings


def test_an_empty_value_is_named_on_the_answer_and_in_the_warnings() -> None:
    request = parsed(cue="json_instructed", chat_format="role_split")
    session = JsonSession(n_vocab=512)
    plan = chat_plan(request, session)
    # the value row closes the value, the row after it closes the object
    scripted(request, plan=plan, session=session, value={QUOTE: TOP}, after={CLOSE: TOP})
    result = decide.DecisionEngine(session).decide(request, plan=plan)
    answer = result.answers["area"]
    assert answer["cue"]["verdict"] == "empty_value"
    assert answer["cue"]["next"]["marker"] == "close"
    assert "W_JSON_EMPTY_VALUE" in result.warnings
    assert "W_LOW_MASS" in result.warnings                 # the mass is low *and* named
    assert answer["reliability"] == "low_mass"
    # `decode_steps` counts the readout's scored tokens (2 candidates × 1) — the walk is *not* one
    # of them (it is the model's own continuation, like the two-step advance), but it is decoded:
    assert answer["decode_steps"] == 2
    assert any(len(batch.tokens) == 1 and batch.tokens[0] == QUOTE for batch in session.batches)


def test_a_wrong_field_is_named_too() -> None:
    request = parsed(cue="json_instructed", chat_format="role_split")
    session = JsonSession(n_vocab=512)
    plan = chat_plan(request, session)
    other = session.tokenize("severity")[0]
    scripted(request, plan=plan, session=session, value={QUOTE: TOP}, after={other: TOP})
    result = decide.DecisionEngine(session).decide(request, plan=plan)
    answer = result.answers["area"]
    assert answer["cue"]["verdict"] == "wrong_field"
    assert answer["cue"]["next"]["marker"] is None
    assert "W_JSON_WRONG_FIELD" in result.warnings
    assert "W_JSON_EMPTY_VALUE" not in result.warnings


def test_a_refused_value_row_keeps_the_e3c_refusal() -> None:
    request = parsed(cue="json_instructed", chat_format="role_split")
    session = JsonSession(n_vocab=512)
    plan = chat_plan(request, session)
    scripted(request, plan=plan, session=session, value={IM_END: TOP})
    result = decide.DecisionEngine(session).decide(request, plan=plan)
    answer = result.answers["area"]
    assert answer["cue"]["verdict"] == "refused"
    assert answer["cue"]["closer"] == "<|im_end|>"
    assert "W_CUE_REFUSED" in result.warnings
    assert "W_JSON_EMPTY_VALUE" not in result.warnings
    assert "W_JSON_WRONG_FIELD" not in result.warnings
    # a refusal never walks: no token was decoded past the value row
    assert not any(len(batch.tokens) == 1 and batch.tokens[0] == QUOTE
                   for batch in session.batches)


def test_the_shipped_shape_gets_no_verdict_key() -> None:
    """Adding the value verdict may not change any other shape's `cue` block."""
    request = parsed()
    session = JsonSession(n_vocab=512)
    plan = chat_plan(request, session)
    scripted(request, plan=plan, session=session, value={session.tokenize("billing")[0]: TOP})
    result = decide.DecisionEngine(session).decide(request, plan=plan)
    assert "verdict" not in result.answers["area"]["cue"]
    assert "W_JSON_EMPTY_VALUE" not in result.warnings
    assert "W_JSON_WRONG_FIELD" not in result.warnings


# ==================================================================== the surfaces (CLI/bench)
def test_the_cli_forwards_the_chat_format_to_the_engine_options() -> None:
    options = cli._engine_options({"chat_format": "role_split", "cue": "json_instructed"})
    assert options == {"chat_format": "role_split", "cue": "json_instructed"}
    assert "chat-format" in cli.ENGINE_VALUE_FLAGS and "chat-format" in cli.BENCH_VALUE_FLAGS


def test_the_bench_validates_the_placement_like_the_cue() -> None:
    assert cli._bench_chat_format(None) == "answer_sheet"
    assert cli._bench_chat_format("role_split") == "role_split"
    with pytest.raises(errors.UserError) as caught:
        cli._bench_chat_format("letters")
    assert caught.value.code == "E_BENCH_USAGE"
    assert "answer_sheet|role_split" in str(caught.value)
    assert cli._bench_json_contract(None) == "question"
    assert cli._bench_json_contract("system") == "system"
    with pytest.raises(errors.UserError) as caught:
        cli._bench_json_contract("everywhere")
    assert caught.value.code == "E_BENCH_USAGE"
    assert "question|system" in str(caught.value)
    assert "json-contract" in cli.ENGINE_VALUE_FLAGS and "json-contract" in cli.BENCH_VALUE_FLAGS


def test_the_contract_is_the_amendments_second_variant() -> None:
    """Same words, heard in the other place: the framing states it, the question asks."""
    inline = parsed(cue="json_instructed")
    system = parsed(cue="json_instructed", json_contract="system")
    assert prompt.framing_for("json_instructed", "question") == prompt.JSON_FRAMING
    assert prompt.framing_for("json_instructed", "system") == prompt.JSON_SYSTEM_FRAMING
    for key in ("choice", "severity", "answer"):
        assert f'"{key}"' in prompt.JSON_SYSTEM_FRAMING      # all three contracts up front
    system_block = prompt.question_block(system.questions[0], cue="json_instructed",
                                         contract="system")
    assert system_block.endswith(prompt.CANDIDATE_CUE["choice"] + "\n")
    assert prompt.JSON_CONTRACT["choice"] not in system_block
    # the inline variant is byte-frozen: the framing says JSON, the question names the key
    assert prompt.question_block(inline.questions[0], cue="json_instructed").endswith(
        prompt.JSON_CONTRACT["choice"] + "\n")
    # and the location is a no-op for every other cue
    for cue in ("shipped", "two_step", "json_field"):
        assert prompt.framing_for(cue, "system") == prompt.framing_for(cue, "question")
        assert prompt.question_block(inline.questions[0], cue=cue, contract="system") == \
            prompt.question_block(inline.questions[0], cue=cue, contract="question")


def test_the_contract_location_reaches_the_plan_and_the_answer_surface() -> None:
    request = parsed(cue="json_instructed", chat_format="role_split", json_contract="system")
    plan = decide.plan_context(request, FakeSession())
    assert plan.json_contract == "system"
    assert plan.role is not None and plan.role.prefix.startswith(prompt.JSON_SYSTEM_FRAMING)
    view = prompt.build_question(request.questions[0], cue="json_instructed",
                                 role=plan.role, chat_format="role_split", contract="system")
    assert view.suffix.endswith(OPENER)                     # the field is still opened
    assert prompt.JSON_CONTRACT["choice"] not in view.suffix  # ... and the key is not re-named
    assert plan.role.tails[0].startswith(prompt.PLAIN_USER_HEADER) or \
        prompt.CANDIDATE_CUE["choice"] in plan.role.tails[0]


def test_the_default_contract_is_the_inline_one_and_the_enumeration_is_pinned() -> None:
    assert schema.JSON_CONTRACT == "question"
    assert schema.JSON_CONTRACTS == ("question", "system")
    assert schema.OPTION_DEFAULTS["json_contract"] == "question"
    assert harness.DEFAULT_JSON_CONTRACT == schema.JSON_CONTRACT
    with pytest.raises(errors.UserError) as caught:
        schema.parse_request({"state": "s", "model": "m",
                              "questions": {"q1": {"type": "choice",
                                                   "criteria": {"billing": ["b"]}}},
                              "options": {"json_contract": "everywhere"}})
    assert caught.value.code == "E_UNKNOWN_KEY"



def test_the_bench_row_asks_for_the_placement() -> None:
    item = suites.devset_module.DevItem(id="c01", type="choice", instructions="Which area?",
                                        criteria={"billing": "payments", "technical": "api"},
                                        state=STATE, gold="billing")
    payload = suites.devset_module.request_for(item, model="bench", threads=4,
                                               cue="json_instructed", chat_format="role_split")
    request = schema.parse_request(payload)
    assert request.options.chat_format == "role_split"
    assert request.options.cue == "json_instructed"


def test_the_reproduce_line_names_a_non_default_policy() -> None:
    base = harness.BenchConfig(suite="quality", model_path="/m.gguf", backend="vulkan", runs=1,
                               threads=4, items=60)
    plain = harness.reproduce_command(base)
    assert "--cue" not in plain and "--chat-format" not in plain
    shaped = harness.reproduce_command(
        harness.BenchConfig(suite="quality", model_path="/m.gguf", backend="vulkan", runs=1,
                            threads=4, items=60, cue="json_instructed",
                            chat_format="role_split"))
    assert "--cue json_instructed" in shaped
    assert "--chat-format role_split" in shaped


def test_the_report_says_which_policy_measured_the_rows() -> None:
    report = {"suite": "quality", "generated_at": "2026-09-19T00:00:00Z",
              "config": {"backend": "vulkan", "runs": 1, "threads": 4, "cue": "json_instructed",
                         "chat_format": "role_split"},
              "model": {"path": "/m.gguf"}, "overall": {}, "per_type": {},
              "commands": {"reproduce": "uv run ggufone bench --suite quality"}}
    text = harness.render_report(report)
    assert "- prompt policy: cue=json_instructed · chat_format=role_split" in text
    report["config"].update({"cue": "shipped", "chat_format": "answer_sheet"})
    assert "- prompt policy:" not in harness.render_report(report)


def test_the_cue_verdict_table_shows_the_named_json_verdicts() -> None:
    report = {"items": [
        {"id": "c01", "type": "choice", "cue": {"token": QUOTE, "mass": 0.9, "refused": False,
                                                "verdict": "empty_value", "closer": None}},
        {"id": "c02", "type": "choice", "cue": {"token": 7, "mass": 0.8, "refused": False,
                                                "verdict": "answered", "closer": None}},
        {"id": "c03", "type": "choice", "cue": {"token": IM_END, "mass": 0.9, "refused": True,
                                                "closer": "<|im_end|>"}},
    ]}
    rows = harness.cue_verdict_rows(report)
    assert "| W_JSON_EMPTY_VALUE |" in rows[0]
    assert rows[1].endswith("| ok |")
    assert rows[2].endswith("| W_CUE_REFUSED |")


def test_the_new_codes_are_registered() -> None:
    from ggufone.errors import ERROR_CODES, WARNING_CODES
    assert "E_ROLE_SPLIT_UNSUPPORTED" in ERROR_CODES
    assert "W_JSON_EMPTY_VALUE" in WARNING_CODES and "W_JSON_WRONG_FIELD" in WARNING_CODES


def test_the_two_step_readout_is_untouched_by_e3e() -> None:
    """E3d's advance rule is byte-frozen: E3e only added shapes beside it."""
    request = parsed(cue="two_step")
    session = JsonSession(n_vocab=512)
    plan = chat_plan(request, session)
    label = session.tokenize("billing")[0]
    suffix = prompt.build_question(request.questions[0], cue="two_step",
                                   chat_format="answer_sheet", role=None, index=0).suffix
    n_suffix = len(session.tokenize(suffix))
    cue_position = len(plan.prefix_tokens) + n_suffix - 1

    def row_fn(context) -> list[float]:
        if context.position == cue_position:
            return biased_row(session.n_vocab, {session.tokenize("Team")[0]: TOP})
        return biased_row(session.n_vocab, {label: TOP})

    session.row_fn = row_fn
    result = decide.DecisionEngine(session).decide(request, plan=plan)
    answer = result.answers["area"]
    assert answer["advance"]["rule"] == cue_module.ADVANCE_RULE
    assert answer["advance"]["token"] == session.tokenize("Team")[0]
    assert answer["choice"] == "billing"
    assert "verdict" not in answer["cue"]


# ==================================================================== live: real family templates
# Run with: GGUFONE_RUNTIME_DIR=<bundle> uv run pytest -q --run-network tests/test_e3e_roles.py
MODEL_PATHS = {
    "spark2_5": pathlib.Path.home() / ".hermes" / "models" / "Spark-X2.5-4B-Q8_0.gguf",
    "qwen35": pathlib.Path.home() / ".hermes" / "models" / "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
    "qwen35moe": pathlib.Path.home() / ".hermes" / "models" / "Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf",
}
#: the bytes the generation prompt ends with before the JSON opener, per family (card t_4c48f40a:
#: "pin the rendered bytes per family"). Spark closes a think block inside its own generation
#: prompt (`<|Bot|></think>`) *and* its template emits a trailing newline after the last turn, so
#: the opener follows that newline; Qwen3.5's prompt opens a think block, which the E1c
#: suppression strips, and its template trims the content it wraps.
EXPECTED_GENERATION_TAIL = {
    "spark2_5": "<|Bot|></think>\n",
    "qwen35": "<|im_start|>assistant\n",
    "qwen35moe": "<|im_start|>assistant\n",
}


def _family_template(name: str) -> tuple[str, str]:
    from ggufone.registry import gguf
    path = MODEL_PATHS[name]
    if not path.exists():
        pytest.skip(f"{path} is not on this box")
    kv = gguf.parse_gguf_metadata(path)["kv"]
    template = kv.get(template_module.TEMPLATE_KEY)
    assert isinstance(template, str) and template
    return gguf.arch_of(kv) or name, template


@pytest.mark.model
@pytest.mark.parametrize("name", sorted(MODEL_PATHS))
def test_each_family_renders_the_two_user_turn_conversation(name: str) -> None:
    """The card's acceptance, per family: the template renders two user turns, bytes pinned.

    A family whose GGUF template is outside the internal renderer's subset is *skipped with the
    reason* here and recorded as `not-renderable` by `tools/e3e_role_render.py` — the fallback
    (`--template plain`, or the live builtin bridge) is a fact about that family, not a silent
    substitution.
    """
    arch, text = _family_template(name)
    request = parsed(cue="json_instructed", chat_format="role_split")
    messages = prompt.chat_messages(request.state, framing=prompt.framing_for("json_instructed"))
    try:
        resolution = template_module.resolve(messages=messages, model_template=text, arch=arch)
    except template_module.TemplateUnresolvedError as exc:
        pytest.skip(f"{name}: the GGUF template is outside the internal renderer's subset "
                    f"({str(exc).splitlines()[0]}); the live path resolves it through the builtin "
                    f"bridge or --template plain, and the family is recorded as not-renderable")
    assert resolution.renderer == "internal"                # no builtin fallback was needed
    role = prompt.role_split_render(request.state, request.questions, resolution=resolution,
                                    cue="json_instructed")
    block = prompt.question_block(request.questions[0], cue="json_instructed")
    full = template_module.render_prompt(
        [*messages, {"role": "user", "content": block}], resolution,
        add_generation_prompt=True, enable_thinking=False)
    assert role.prefix + role.tails[0] == full + OPENER     # the render + the opened field
    assert f"{prompt.STATE_HEADER}{STATE}" in role.prefix   # the state is in the shared prefix
    assert block.strip() in role.tails[0]                   # the question is its own user turn
    assert role.tails[0].endswith(f"{EXPECTED_GENERATION_TAIL[name]}{OPENER}")
    assert template_module.no_open_think(role.prefix + role.tails[0])
    assert "QUESTION:" not in role.prefix
