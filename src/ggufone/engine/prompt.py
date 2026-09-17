"""Prompt assembly (SPEC 2.3.1).

The prefix (`system framing + state`) is byte-identical for every question of a request — that
is what makes the single prefill + fork legal. Everything that differs per question (the
instructions, the rendered criteria, the answer cue) lives in the question suffix.

Candidate labels are rendered from the request, never invented: option names for `choice`, the
level numbers for `score` ("0".."K-1"), and `yes`/`no` for `noul` — exactly the keys the
response reports back, so nothing inside `answers` has to be renamed (SPEC 2.6).

Readout policy note: E1b scores the rendered label itself (`sequence` = every token of the
label, `single_token` = its first token, same prompt either way). The per-family label policy
(bracket letters, leading space, chat template) is E1c/§5 — this module keeps the seam: pass
`readout=` and the *same* suffix is produced for both modes (A-E1b-6).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

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


def build_prefix(state: Any) -> str:
    """`system framing + state` — the bytes every question of this request shares."""
    return f"{SYSTEM_FRAMING}{STATE_HEADER}{render_state(state)}\n"


def build_question(question: Question, *, readout: str = "sequence") -> RenderedQuestion:
    """Render one question's suffix and its candidate labels (independent of `readout`)."""
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
    return RenderedQuestion(id=question.id, type=question.type, suffix=suffix,
                            options=question.options, texts=question.options)


def empty_candidate_code(question_type: str) -> str:
    """The schema code to raise when a rendered label tokenizes to nothing."""
    return {"choice": "E_CHOICE_CRITERIA", "score": "E_SCORE_LEVELS",
            "noul": "E_NOUL_CRITERIA"}[question_type]
