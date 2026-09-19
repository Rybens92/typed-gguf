"""Prompt assembly (SPEC 2.3.1).

The prefix (`system framing + state`) is byte-identical for every question of a request — that
is what makes the single prefill + fork legal. Everything that differs per question (the
instructions, the rendered criteria, the answer cue) lives in the question suffix.

Candidate labels are rendered from the request, never invented: option names for `choice`, the
level numbers for `score` ("0".."K-1"), and `yes`/`no` for `noul` — exactly the keys the
response reports back, so nothing inside `answers` has to be renamed (SPEC 2.6).

Two assemblies (E1c):

* **plain** — the model-agnostic framing of E1b, kept as the documented escape hatch
  (`--template plain`, and the default for a session that has no model handle);
* **chat template** — the model's own template, resolved by `engine/template.py` (A-E1c-1) and
  rendered with thinking suppressed (A-E1c-2). The prefix ends at the template's generation
  prompt; the question suffix is the continuation of that assistant turn, so the candidate label
  is read exactly where the family expects it.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ggufone import schema
from ggufone.engine import template as template_module
from ggufone.errors import UserError
from ggufone.schema import Question

SYSTEM_FRAMING = (
    "You are a decision engine. Read the state, then answer every question by choosing exactly "
    "one of the candidate labels. Reply with the label alone — no explanation, no punctuation, "
    "no extra words.\n"
)
#: E3e (card t_4c48f40a): the framing the `json_instructed` cue asks for. The shipped framing says
#: "reply with the label alone", which contradicts a JSON answer; a shape that *asks* for JSON has
#: to *say* so (the old `json_field` cue opened a structure the model was never told about).
JSON_FRAMING = (
    "You are a decision engine. Read the state, then answer every question by choosing exactly "
    "one of the candidate labels. Reply with a single JSON object and nothing else — no "
    "explanation, no code fence, no extra keys.\n"
)
STATE_HEADER = "STATE:\n"
QUESTION_HEADER = "QUESTION:"
CANDIDATE_CUE = {
    "choice": "Answer with exactly one candidate name:",
    "score": "Answer with exactly one level number:",
    "noul": "Answer with exactly one of yes or no:",
}
#: The JSON contract `json_instructed` puts in the question block, per question type. It names the
#: field the response will be read from, so the opener the assistant turn carries is the *natural
#: continuation* of an instructed format rather than a structure bolted onto a label question.
JSON_CONTRACT = {
    "choice": 'Answer with JSON: {"choice": "<exactly one candidate name>"}',
    "score": 'Answer with JSON: {"severity": "<exactly one level number>"}',
    "noul": 'Answer with JSON: {"answer": "<yes or no>"}',
}
#: E3e, the amendment's second variant: the *system framing* states every contract, so the model is
#: told the format once and each question keeps its own ask line (`{"<key>": "<one of the candidate
#: names>"}`, key per question type from `JSON_FIELDS`). `options.json_contract` picks where the
#: contract is stated — `question` (inline, default) or `system` — and both are measured, because
#: "where the model hears the format" is exactly the knob the card asks about.
JSON_SYSTEM_FRAMING = (
    "You are a decision engine. Read the state, then answer every question with a single JSON "
    "object and nothing else — no explanation, no code fence, no extra keys. The object's key "
    "names the question type: {\"choice\": \"<exactly one candidate name>\"} for a choice, "
    "{\"severity\": \"<exactly one level number>\"} for a level, {\"answer\": \"<yes or no>\"} "
    "for a yes/no question.\n"
)
#: where the JSON contract is stated (`options.json_contract`); `question` is the default (the
#: question block names its own key next to the candidates it is about)
JSON_CONTRACTS: tuple[str, ...] = ("question", "system")
JSON_CONTRACT_DEFAULT = JSON_CONTRACTS[0]
#: the cue shapes whose suffix is a JSON object: the instruction is a JSON contract and the
#: assistant turn is prefilled with the opened field (`json_field` keeps the *shipped* cue line —
#: every E3d row was measured on those bytes).
JSON_CUES = ("json_field", "json_instructed")

#: The plain fallback's role markers (E3e): a session without a chat template still gets an
#: explicit user turn and an assistant turn, so `role_split` means the same thing everywhere.
PLAIN_USER_HEADER = "USER:\n"
PLAIN_ASSISTANT_HEADER = "ASSISTANT:\n"


def framing_for(cue: str, contract: str = JSON_CONTRACT_DEFAULT) -> str:
    """The system framing a cue shape asks for: the JSON one when the answer is a JSON object.

    `contract` decides which JSON framing: `question` says only that the answer is a JSON object
    (the question block names the key), `system` states all three contracts up front. Anything
    that is not `json_instructed` gets the shipped framing — the other cues ask for a bare label.
    """
    if cue != "json_instructed":
        return SYSTEM_FRAMING
    return JSON_SYSTEM_FRAMING if contract == "system" else JSON_FRAMING


@dataclass(frozen=True, slots=True)
class RenderedQuestion:
    """One question as it is sent to the model."""

    id: str
    type: str
    suffix: str                    # the per-question framing (the prefix is shared)
    options: tuple[str, ...]       # candidate keys, in wire order
    texts: tuple[str, ...]         # the rendered candidate label for each key


def render_value(value: Any) -> str:
    """Render a `string | object | array` (state, instructions) deterministically."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def render_state(state: Any) -> str:
    return render_value(state)


def render_instructions(instructions: Any) -> str:
    if isinstance(instructions, (list, tuple)) and all(isinstance(x, str) for x in instructions):
        return "\n".join(instructions)
    return render_value(instructions)


def chat_messages(state: Any, *, framing: str = SYSTEM_FRAMING) -> list[dict[str, str]]:
    """The default turn structure every family template renders (system framing + state)."""
    return [{"role": "system", "content": framing.strip()},
            {"role": "user", "content": f"{STATE_HEADER}{render_state(state)}"}]


@dataclass(frozen=True, slots=True)
class RoleSplitRender:
    """The `role_split` assembly of one request (E3e, card t_4c48f40a).

    `prefix` is the bytes every question of this request shares (the state turn plus whatever the
    template emits with it); `tails` is one tail per question, in request order — the rendered
    **user** turn that carries the question, the template's generation prompt, and the JSON opener
    when the cue asks for one. `prefix + tail` is the *byte-exact* render of
    `[system, state-as-user, question-as-user]` with the generation prompt (plus the opener for the
    JSON cues), which is what makes the prefill/fork legal: the model sees exactly the conversation
    its own template defines, and the question never becomes part of the assistant turn.

    `dropped` is what the state-only render carries that the shared prefix cannot: a template may
    emit its own trailing text (Spark‑X2.5's template ends with a newline) *after* the last
    message, and that text belongs at the end of a render, not in the middle of one. It is
    recorded, never silently lost.
    """

    prefix: str
    tails: tuple[str, ...]
    dropped: str = ""

    def tail_for(self, index: int) -> str:
        """The tail of the question at `index` (the request's own order)."""
        if not 0 <= index < len(self.tails):
            raise UserError(f"role_split render carries {len(self.tails)} question tail(s); "
                            f"asked for index {index}", code="E_ROLE_SPLIT_UNSUPPORTED")
        return self.tails[index]


def question_block(question: Question, *, cue: str = "shipped",
                   contract: str = JSON_CONTRACT_DEFAULT) -> str:
    """The question's own bytes: instructions, rendered candidates, and the ask line (E3e).

    The ask line is the *whole* difference between the cue shapes' question text: the bare label
    (`shipped`, `two_step`), the shipped line plus an opened field (`json_field`, byte-frozen since
    E3d), or the JSON contract the assistant turn then continues (`json_instructed`). With
    `contract="system"` the contract is stated in the framing instead, and the question keeps the
    shipped ask line — the same words, heard in the other place.
    """
    lines = [QUESTION_HEADER]
    instructions = render_instructions(question.instructions)
    if instructions:
        lines.append(instructions)
    if question.type == "choice":
        lines.append("Candidates:")
        for name, description in zip(question.options, question.descriptions, strict=True):
            lines.append(f"- {name}" + (f": {description}" if description else ""))
    elif question.type == "score":
        lines.append("Levels (0 is the lowest):")
        for number, description in zip(question.options, question.descriptions, strict=True):
            lines.append(f"{number}: {description}")
    else:  # noul
        truth, falsity = question.descriptions
        lines.append(f"yes: {truth}" if truth else "yes")
        lines.append(f"no: {falsity}" if falsity else "no")
    lines.append(JSON_CONTRACT[question.type]
                 if cue == "json_instructed" and contract == "question"
                 else CANDIDATE_CUE[question.type])
    return "\n".join(lines) + "\n"


def _question_messages(state: Any, block: str, *, framing: str) -> list[dict[str, str]]:
    return [*chat_messages(state, framing=framing), {"role": "user", "content": block}]


def _common_prefix(left: str, right: str) -> str:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return left[:index]


def normalised(text: str) -> str:
    """Whitespace-normalised bytes — what a template that trims or re-wraps a message still carries.

    The acceptance below is about *content*, not layout: a family template may trim a message or
    join its newlines (Ternary-Bonsai's template renders `{{ content | trim }}`, so a question
    block's final newline never reaches the prompt) and the model still reads the question's own
    words. A template that drops or rewrites them fails this, which is the case it exists for.
    """
    return " ".join(text.split())


def role_split_render(state: Any, questions: Sequence[Question], *,
                      resolution: template_module.Resolution | None = None,
                      enable_thinking: bool = False, cue: str = "shipped",
                      contract: str = JSON_CONTRACT_DEFAULT
                      ) -> RoleSplitRender:
    """Render the `role_split` shape of a whole request: `(shared prefix, one tail per question)`.

    With a chat-template `resolution` the question is a real **user** message rendered by the
    model's own template; the shared prefix is the render of the state messages where the template
    stops sharing bytes with them (`_common_prefix` — Spark's template emits its trailing newline
    only at the end of a render, so the state-only render is one byte longer than the shared part).

    Two acceptance rules are *measured*, never assumed (card t_4c48f40a, amendment 4):

    * the shared prefix must carry the state turn's own bytes and every question's render must
      extend it — a template that merges the turns, reorders them, or drops the state is
      `E_ROLE_SPLIT_UNSUPPORTED`, with the fallback named in the message;
    * every question's own block must survive into its own render — a template that drops or
      rewrites the last user message is refused the same way.
    """
    framing = framing_for(cue, contract)
    messages = chat_messages(state, framing=framing)
    state_text = str(messages[-1]["content"])
    blocks = [question_block(question, cue=cue, contract=contract) for question in questions]
    openters = [json_opener(question.type) if cue in JSON_CUES else "" for question in questions]
    if resolution is None or resolution.is_plain:
        prefix = f"{framing}{STATE_HEADER}{render_state(state)}\n"
        tails = tuple(f"{PLAIN_USER_HEADER}{block}{PLAIN_ASSISTANT_HEADER}{opener}"
                      for block, opener in zip(blocks, openters, strict=True))
        return RoleSplitRender(prefix=prefix, tails=tails)
    rendered = [template_module.render_prompt(_question_messages(state, block, framing=framing),
                                              resolution, add_generation_prompt=True,
                                              enable_thinking=enable_thinking)
                for block in blocks]
    state_only = template_module.render_prompt(messages, resolution,
                                               add_generation_prompt=False,
                                               enable_thinking=enable_thinking)
    prefix = _common_prefix(state_only, rendered[0]) if rendered else state_only
    if normalised(state_text) not in normalised(prefix):
        raise UserError(
            "E_ROLE_SPLIT_UNSUPPORTED: the template does not render this conversation's state as "
            "its own user turn before the question (the shared prefix does not carry the state's "
            "own words, so the question would not be preceded by the state the model must read). "
            "Fix: --chat-format answer_sheet (the published shape) or --template plain",
            code="E_ROLE_SPLIT_UNSUPPORTED")
    for question, block, full in zip(questions, blocks, rendered, strict=True):
        if not full.startswith(prefix):
            raise UserError(
                f"E_ROLE_SPLIT_UNSUPPORTED: question {question.id!r}: the template does not render "
                f"this conversation's question as its own user turn after the state turn (the "
                f"render does not extend the shared prefix). Fix: --chat-format answer_sheet (the "
                f"published shape) or --template plain",
                code="E_ROLE_SPLIT_UNSUPPORTED")
        if normalised(block) not in normalised(full):
            raise UserError(
                f"E_ROLE_SPLIT_UNSUPPORTED: question {question.id!r}: the template dropped or "
                f"rewrote the question's user turn (its words are not in the render). Fix: "
                f"--chat-format answer_sheet or --template plain",
                code="E_ROLE_SPLIT_UNSUPPORTED")
        if not enable_thinking and not template_module.no_open_think(prefix + full[len(prefix):]):
            raise UserError(
                f"E_ROLE_SPLIT_UNSUPPORTED: question {question.id!r}: the rendered prompt ends "
                f"inside a thinking block; the readout would sit inside it. Fix: --chat-format "
                f"answer_sheet, --template plain, or render with thinking on",
                code="E_ROLE_SPLIT_UNSUPPORTED")
    tails = tuple(full[len(prefix):] + opener
                  for full, opener in zip(rendered, openters, strict=True))
    return RoleSplitRender(prefix=prefix, tails=tails, dropped=state_only[len(prefix):])


def role_split_context(request: Any,
                       resolution: template_module.Resolution | None = None
                       ) -> RoleSplitRender | None:
    """The `RoleSplitRender` of a request — `None` for the answer-sheet default.

    The one place that decides *whether* a request is role-split, so `plan_context` (the prefix)
    and `question_requirements` (the tails) can never disagree about it.
    """
    options = request.options
    if options.chat_format != schema.ROLE_SPLIT:
        return None
    return role_split_render(request.state, request.questions, resolution=resolution,
                             enable_thinking=options.thinking, cue=options.cue,
                             contract=options.json_contract)


def build_prefix(state: Any, *, resolution: template_module.Resolution | None = None,
                 enable_thinking: bool = False, chat_format: str = schema.ANSWER_SHEET,
                 cue: str = "shipped", questions: Sequence[Question] = (),
                 role: RoleSplitRender | None = None,
                 contract: str = JSON_CONTRACT_DEFAULT) -> str:
    """`system framing + state` — the bytes every question of this request shares.

    With a chat-template `resolution` the bytes are that template's rendering of
    `[system, user]` plus its generation prompt (A-E1c-1/2); without one (or with
    `--template plain`) they are the model-agnostic framing E1b shipped. With
    `chat_format="role_split"` the shared bytes stop where the question's own user turn starts
    (`role_split_render`), which is what keeps the question out of the assistant turn; a caller
    that already holds that render passes it as `role` instead of paying for it twice.
    """
    if resolution is None or resolution.is_plain:
        return f"{framing_for(cue, contract)}{STATE_HEADER}{render_state(state)}\n"
    if chat_format == schema.ROLE_SPLIT:
        render = role if role is not None else role_split_render(
            state, questions, resolution=resolution, enable_thinking=enable_thinking, cue=cue,
            contract=contract)
        return render.prefix
    return template_module.render_prompt(
        chat_messages(state, framing=framing_for(cue, contract)), resolution,
        add_generation_prompt=True, enable_thinking=enable_thinking)


def build_question(question: Question, *, readout: str = "sequence",
                   cue: str = schema.CUE_SHAPES[0], chat_format: str = schema.ANSWER_SHEET,
                   role: RoleSplitRender | None = None, index: int = 0,
                   contract: str = JSON_CONTRACT_DEFAULT) -> RenderedQuestion:
    """Render one question's suffix and its candidate labels (independent of `readout`).

    `cue` names *where* the label will be read (E3d, card t_d90404ac) and nothing else: with
    `two_step` the suffix is **byte-identical** to the shipped one — the readout moves one token
    in, the prompt does not — and with `json_field` the shipped cue line is followed by the
    per-type JSON opener (`{"choice": "`), so the field row is the prompt's own last row.

    `json_instructed` (E3e) says the answer is a JSON object instead of a bare label: its question
    block carries `JSON_CONTRACT` and the suffix ends on the opened field. `chat_format` decides
    where that block *lives*: `answer_sheet` prefills it inside the assistant turn (the shape every
    published table measured), `role_split` renders it as the question's own user turn — in which
    case `role` (from `role_split_render`) supplies the already-rendered tail and `index` names the
    question's position in the request.
    """
    if cue not in schema.CUE_SHAPES:
        raise UserError(f"options.cue must be one of {', '.join(schema.CUE_SHAPES)} (got {cue!r})",
                        code="E_UNKNOWN_KEY")
    if chat_format == schema.ROLE_SPLIT:
        if role is None:
            raise UserError(
                "chat_format=role_split needs the request's rendered role split "
                "(prompt.role_split_context); the question cannot be rendered as a user turn "
                "without it", code="E_ROLE_SPLIT_UNSUPPORTED")
        suffix = role.tail_for(index)
    else:
        suffix = question_block(question, cue=cue, contract=contract)
        if cue in JSON_CUES:
            suffix += json_opener(question.type)
    return RenderedQuestion(id=question.id, type=question.type, suffix=suffix,
                            options=question.options, texts=question.options)


#: The field name the `json_field` cue opens per question type (the E3c probe's mapping, kept
#: verbatim so the engine can reproduce the measured row; card t_d90404ac).
JSON_FIELDS = {"choice": "choice", "score": "severity", "noul": "answer"}


def json_opener(question_type: str) -> str:
    """The `json_field` opener: the shipped cue line, then an open JSON field."""
    return '{{"{field}": "'.format(field=JSON_FIELDS.get(question_type, "answer"))


def empty_candidate_code(question_type: str) -> str:
    """The schema code to raise when a rendered label tokenizes to nothing."""
    return {"choice": "E_CHOICE_CRITERIA", "score": "E_SCORE_LEVELS",
            "noul": "E_NOUL_CRITERIA"}[question_type]
