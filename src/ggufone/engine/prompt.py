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
STATE_HEADER = "STATE:\n"
QUESTION_HEADER = "QUESTION:"
CANDIDATE_CUE = {
    "choice": "Answer with exactly one candidate name:",
    "score": "Answer with exactly one level number:",
    "noul": "Answer with exactly one of yes or no:",
}


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


def chat_messages(state: Any) -> list[dict[str, str]]:
    """The default turn structure every family template renders (system framing + state)."""
    return [{"role": "system", "content": SYSTEM_FRAMING.strip()},
            {"role": "user", "content": f"{STATE_HEADER}{render_state(state)}"}]


def build_prefix(state: Any, *, resolution: template_module.Resolution | None = None,
                 enable_thinking: bool = False) -> str:
    """`system framing + state` — the bytes every question of this request shares.

    With a chat-template `resolution` the bytes are that template's rendering of
    `[system, user]` plus its generation prompt (A-E1c-1/2); without one (or with
    `--template plain`) they are the model-agnostic framing E1b shipped.
    """
    if resolution is None or resolution.is_plain:
        return f"{SYSTEM_FRAMING}{STATE_HEADER}{render_state(state)}\n"
    return template_module.render_prompt(chat_messages(state), resolution,
                                        add_generation_prompt=True,
                                        enable_thinking=enable_thinking)


def build_question(question: Question, *, readout: str = "sequence",
                   cue: str = schema.CUE_SHAPES[0]) -> RenderedQuestion:
    """Render one question's suffix and its candidate labels (independent of `readout`).

    `cue` names *where* the label will be read (E3d, card t_d90404ac) and nothing else: with
    `two_step` the suffix is **byte-identical** to the shipped one — the readout moves one token
    in, the prompt does not — and with `json_field` the shipped cue line is followed by the
    per-type JSON opener (`{"choice": "`), so the field row is the prompt's own last row.
    """
    if cue not in schema.CUE_SHAPES:
        raise UserError(f"options.cue must be one of {', '.join(schema.CUE_SHAPES)} (got {cue!r})",
                        code="E_UNKNOWN_KEY")
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
    lines.append(CANDIDATE_CUE[question.type])
    suffix = "\n".join(lines) + "\n"
    if cue == "json_field":
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
