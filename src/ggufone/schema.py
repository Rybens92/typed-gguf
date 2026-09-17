"""Request/response schema, validation and the typesafe adapter (SPEC 2.5, 2.6).

Milestone: E1b.

Owns `parse_request()`, `render_response(format=...)` and every `E_*` schema code.

Validation rules that matter (SPEC 2.5):
  * unknown keys at the TOP level are always `E_UNKNOWN_KEY`;
  * unknown keys inside `options` produce `W_UNKNOWN_OPTION` unless `options.strict` is true;
  * `state`, `questions` and the per-type `criteria` shapes are validated here, before the
    engine ever sees them — A-E1b-13 pins the codes (`E_STATE_EMPTY`, `E_QID_INVALID`,
    `E_Q_TYPE_UNKNOWN`, `E_CHOICE_CRITERIA`, `E_CHOICE_TOO_MANY`, `E_SCORE_LEVELS`,
    `E_NOUL_CRITERIA`).

Values that are not in the frozen enumerations (e.g. `readout: "letters"`, `temperature: 0`)
are reported as `E_UNKNOWN_KEY`: the catalog has no `E_BAD_VALUE`, and what the caller needs to
know is that the request named something the engine does not implement — exit code 2.

Numbers are rounded to 6 significant decimals on the way out (SPEC 2.5) so that two runs are
byte-comparable; `score_weighted_mean` already returns that precision (see engine/readout.py).

The typesafe adapter is a pure projection of the native response (SPEC 2.6): it passes
`state`/`questions` through untouched, maps a non-registry `model` reference to the configured
default alias, and drops every native-only key — nothing inside `answers` is renamed.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ggufone.engine.readout import CONFIDENCE_MODES, round_sig
from ggufone.errors import UserError

REQUEST_KEYS = ("state", "model", "questions", "format", "options")
QUESTION_KEYS = ("type", "instructions", "criteria")
FORMATS = ("native", "typesafe")
QUESTION_TYPES = ("choice", "score", "noul")
CHOICE_LIMIT = 255
SCORE_LEVELS = (2, 10)
KV_TYPES = ("auto", "f16", "q8_0", "q4_0")
BACKENDS = ("auto", "cpu", "vulkan", "cuda", "metal")
NOUL_KEYS = ("true", "false")

# SPEC 2.5 defaults. Keep this table the single source of truth for the wire defaults.
OPTION_DEFAULTS: dict[str, Any] = {
    "temperature": 1.0,
    "length_norm": 1.0,
    "readout": "sequence",
    "confidence_mode": "normalized_peak",
    "n_ctx": None,
    "n_seq_max": None,
    "kv_type": "auto",
    "coverage_floor": 0.10,
    "seed": 0,
    "threads": None,
    "backend": "auto",
    "state_id": None,
    "state_cache": True,
    "save_state": False,
    "max_waves": None,
    "strict": False,
}


def _fail(message: str, code: str) -> UserError:
    return UserError(message, code=code)


@dataclass(frozen=True, slots=True)
class Question:
    """One typed question after validation.

    `options` are the candidate keys in wire order: option names for `choice`, the level numbers
    as strings for `score` ("0".."K-1"), and ("yes", "no") for `noul`. The question id is never
    sent to the model — it only names the answer in the response.
    """

    id: str
    type: str
    instructions: Any
    criteria: Any
    options: tuple[str, ...]
    descriptions: tuple[str | None, ...]


@dataclass(frozen=True, slots=True)
class Options:
    """Engine options (SPEC 2.5 `options`). Every field has a frozen default."""

    temperature: float = 1.0
    length_norm: float = 1.0
    readout: str = "sequence"
    confidence_mode: str = "normalized_peak"
    n_ctx: int | None = None
    n_seq_max: int | None = None
    kv_type: str = "auto"
    coverage_floor: float = 0.10
    seed: int = 0
    threads: int | None = None
    backend: str = "auto"
    state_id: str | None = None
    state_cache: bool = True
    save_state: bool = False
    max_waves: int | None = None
    strict: bool = False


@dataclass(frozen=True, slots=True)
class Request:
    state: Any
    model: str | None
    questions: tuple[Question, ...]
    options: Options = field(default_factory=Options)
    format: str = "native"
    warnings: tuple[str, ...] = ()


# --------------------------------------------------------------------- parsing
def parse_request(payload: Any) -> Request:
    """Validate a native request (SPEC 2.5) into a `Request`. Raises `UserError` (exit 2)."""
    if not isinstance(payload, Mapping):
        raise _fail("the request must be a JSON object", "E_UNKNOWN_KEY")
    unknown = [key for key in payload if key not in REQUEST_KEYS]
    if unknown:
        raise _fail(f"unknown top-level key(s): {', '.join(sorted(map(str, unknown)))}; "
                    f"allowed: {', '.join(REQUEST_KEYS)}", "E_UNKNOWN_KEY")

    state = payload.get("state")
    _require_state(state)

    model = payload.get("model")
    if model is not None and not isinstance(model, str):
        raise _fail("model must be a string (alias | path | repo[:quant])", "E_UNKNOWN_KEY")

    fmt = payload.get("format", "native")
    if fmt not in FORMATS:
        raise _fail(f"format must be one of {', '.join(FORMATS)} (got {fmt!r})", "E_UNKNOWN_KEY")

    options, warnings = _parse_options(payload.get("options"))
    questions = _parse_questions(payload.get("questions"))
    return Request(state=state, model=model, questions=questions, options=options,
                   format=fmt, warnings=tuple(warnings))


def _require_state(state: Any) -> None:
    if state is None:
        raise _fail("state is required (string | object | array)", "E_STATE_EMPTY")
    if isinstance(state, str):
        if not state.strip():
            raise _fail("state is empty", "E_STATE_EMPTY")
        return
    if isinstance(state, (Mapping, Sequence)):
        if not state:
            raise _fail("state is empty", "E_STATE_EMPTY")
        return
    raise _fail(f"state must be a string, object or array (got {type(state).__name__})",
                "E_STATE_EMPTY")


def _parse_questions(raw: Any) -> tuple[Question, ...]:
    if not isinstance(raw, Mapping) or not raw:
        raise _fail("questions must be a non-empty object of "
                    "{id: {type, instructions, criteria}}", "E_QID_INVALID")
    questions: list[Question] = []
    for qid, body in raw.items():
        if not isinstance(qid, str) or not qid.strip():
            raise _fail(f"question id must be a non-empty string (got {qid!r})", "E_QID_INVALID")
        questions.append(_parse_question(qid, body))
    return tuple(questions)


def _parse_question(qid: str, body: Any) -> Question:
    if not isinstance(body, Mapping):
        raise _fail(f"question {qid!r} must be an object", "E_Q_TYPE_UNKNOWN")
    unknown = [key for key in body if key not in QUESTION_KEYS]
    if unknown:
        raise _fail(f"question {qid!r} has unknown key(s): {', '.join(sorted(map(str, unknown)))}",
                    "E_UNKNOWN_KEY")
    qtype = body.get("type")
    if not isinstance(qtype, str) or qtype not in QUESTION_TYPES:
        raise _fail(f"question {qid!r}: type must be one of {', '.join(QUESTION_TYPES)} "
                    f"(got {qtype!r})", "E_Q_TYPE_UNKNOWN")
    criteria = body.get("criteria")
    instructions = body.get("instructions")
    if qtype == "choice":
        options, descriptions = _choice_criteria(qid, criteria)
    elif qtype == "score":
        options, descriptions = _score_criteria(qid, criteria)
    else:
        options, descriptions = _noul_criteria(qid, criteria)
    return Question(id=qid, type=qtype, instructions=instructions, criteria=criteria,
                    options=options, descriptions=descriptions)


def _choice_criteria(qid: str, criteria: Any) -> tuple[tuple[str, ...], tuple[str | None, ...]]:
    if not isinstance(criteria, Mapping) or not criteria:
        raise _fail(f"question {qid!r}: choice criteria must be a non-empty "
                    f"object of {{option: description|null}}", "E_CHOICE_CRITERIA")
    for key in criteria:
        if not isinstance(key, str) or not key.strip():
            raise _fail(f"question {qid!r}: option names must be non-empty strings "
                        f"(got {key!r})", "E_CHOICE_CRITERIA")
    if len(criteria) > CHOICE_LIMIT:
        raise _fail(f"question {qid!r}: {len(criteria)} options exceed the documented limit of "
                    f"{CHOICE_LIMIT}", "E_CHOICE_TOO_MANY")
    options = tuple(criteria.keys())
    descriptions = tuple(value if isinstance(value, str) else None for value in criteria.values())
    return options, descriptions


def _score_criteria(qid: str, criteria: Any) -> tuple[tuple[str, ...], tuple[str | None, ...]]:
    if not isinstance(criteria, list) or not SCORE_LEVELS[0] <= len(criteria) <= SCORE_LEVELS[1]:
        raise _fail(f"question {qid!r}: score criteria must be an ordered array of "
                    f"{SCORE_LEVELS[0]}..{SCORE_LEVELS[1]} levels", "E_SCORE_LEVELS")
    for level in criteria:
        if not isinstance(level, str) or not level.strip():
            raise _fail(f"question {qid!r}: every level needs a non-empty description "
                        f"(got {level!r})", "E_SCORE_LEVELS")
    options = tuple(str(index) for index in range(len(criteria)))
    return options, tuple(criteria)


def _noul_criteria(qid: str, criteria: Any) -> tuple[tuple[str, ...], tuple[str | None, ...]]:
    """`{true: str, false: str}` — both sides or neither (the prompt renders both labels)."""
    descriptions: dict[str, str | None] = {"true": None, "false": None}
    if criteria is not None:
        if not isinstance(criteria, Mapping) or not criteria:
            raise _fail(f"question {qid!r}: noul criteria must be an object "
                        f"{{true: str, false: str}}", "E_NOUL_CRITERIA")
        for key in criteria:
            if key not in NOUL_KEYS:
                raise _fail(f"question {qid!r}: noul criteria accepts only "
                            f"{', '.join(NOUL_KEYS)} (got {key!r})", "E_NOUL_CRITERIA")
        for key in NOUL_KEYS:
            value = criteria.get(key)
            if not isinstance(value, str) or not value.strip():
                raise _fail(f"question {qid!r}: noul criterion {key!r} must be non-empty text "
                            f"(got {value!r})", "E_NOUL_CRITERIA")
            descriptions[key] = value
    return ("yes", "no"), (descriptions["true"], descriptions["false"])


def _parse_options(raw: Any) -> tuple[Options, list[str]]:
    if raw is None:
        return Options(), []
    if not isinstance(raw, Mapping):
        raise _fail("options must be an object", "E_UNKNOWN_KEY")
    strict = raw.get("strict", False)
    if not isinstance(strict, bool):
        raise _fail("options.strict must be a boolean", "E_UNKNOWN_KEY")
    unknown = [key for key in raw if key not in OPTION_DEFAULTS]
    if unknown and strict:
        raise _fail(f"unknown option(s) under strict=true: {', '.join(sorted(map(str, unknown)))}",
                    "E_UNKNOWN_KEY")
    warnings = ["W_UNKNOWN_OPTION"] if unknown else []

    values: dict[str, Any] = dict(OPTION_DEFAULTS)
    for key, value in raw.items():
        if key in OPTION_DEFAULTS:
            values[key] = value
    options = Options(
        temperature=_number("temperature", values["temperature"], low=0.0, inclusive=False),
        length_norm=_number("length_norm", values["length_norm"], low=0.0, inclusive=True),
        readout=_choice("readout", values["readout"], ("sequence", "single_token")),
        confidence_mode=_choice("confidence_mode", values["confidence_mode"],
                                tuple(CONFIDENCE_MODES)),
        n_ctx=_optional_int("n_ctx", values["n_ctx"], low=1),
        n_seq_max=_optional_int("n_seq_max", values["n_seq_max"], low=3),
        kv_type=_choice("kv_type", values["kv_type"], KV_TYPES),
        coverage_floor=_number("coverage_floor", values["coverage_floor"], low=0.0,
                               inclusive=True, high=1.0),
        seed=_int("seed", values["seed"]),
        threads=_optional_int("threads", values["threads"], low=1),
        backend=_choice("backend", values["backend"], BACKENDS),
        state_id=_optional_str("state_id", values["state_id"]),
        state_cache=_bool("state_cache", values["state_cache"]),
        save_state=_bool("save_state", values["save_state"]),
        max_waves=_optional_int("max_waves", values["max_waves"], low=1),
        strict=strict,
    )
    return options, warnings


def _number(name: str, value: Any, *, low: float, inclusive: bool,
            high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(f"options.{name} must be a number (got {value!r})", "E_UNKNOWN_KEY")
    number = float(value)
    if (number < low) or (number == low and not inclusive):
        raise _fail(f"options.{name} must be {'>=' if inclusive else '>'} {low} (got {number})",
                    "E_UNKNOWN_KEY")
    if high is not None and number > high:
        raise _fail(f"options.{name} must be <= {high} (got {number})", "E_UNKNOWN_KEY")
    return number


def _int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(f"options.{name} must be an integer (got {value!r})", "E_UNKNOWN_KEY")
    return value


def _optional_int(name: str, value: Any, *, low: int) -> int | None:
    if value is None:
        return None
    parsed = _int(name, value)
    if parsed < low:
        raise _fail(f"options.{name} must be >= {low} (got {parsed})", "E_UNKNOWN_KEY")
    return parsed


def _optional_str(name: str, value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _fail(f"options.{name} must be a non-empty string (got {value!r})", "E_UNKNOWN_KEY")
    return value


def _bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise _fail(f"options.{name} must be a boolean (got {value!r})", "E_UNKNOWN_KEY")
    return value


def _choice(name: str, value: Any, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise _fail(f"options.{name} must be one of {', '.join(allowed)} (got {value!r})",
                    "E_UNKNOWN_KEY")
    return str(value)


# -------------------------------------------------------------------- rendering
VALUE_KEYS = {"choice": "choice", "score": "score", "noul": "noul"}


def render_response(result: Mapping[str, Any], *, format: str = "native") -> dict[str, Any]:
    """Round every number to the wire precision and project to `format` (SPEC 2.5/2.6)."""
    if format not in FORMATS:
        raise _fail(f"format must be one of {', '.join(FORMATS)} (got {format!r})",
                    "E_UNKNOWN_KEY")
    rounded = _round_tree(dict(result))
    if format == "native":
        return rounded
    answers: dict[str, Any] = {}
    for qid, answer in rounded["answers"].items():
        projected: dict[str, Any] = {"type": answer["type"]}
        value_key = VALUE_KEYS[answer["type"]]
        projected[value_key] = answer[value_key]
        projected["probabilities"] = answer["probabilities"]
        if "confidence" in answer:
            projected["confidence"] = answer["confidence"]
        # §2.6: the legend key is emitted for `score` only (level number -> description)
        if "legend" in answer and answer["type"] == "score":
            projected["legend"] = answer["legend"]
        answers[qid] = projected
    usage = rounded.get("usage", {})
    return {
        "model": rounded["model"],
        "answers": answers,
        "usage": {"input_tokens": usage.get("input_tokens", 0),
                  "output_tokens": usage.get("output_tokens", 0)},
    }


def _round_tree(value: Any) -> Any:
    if isinstance(value, float):
        return round_sig(value)
    if isinstance(value, Mapping):
        return {key: _round_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_tree(item) for item in value]
    return value


def adapter_model_ref(ref: str | None, *, known_aliases: Sequence[str] = (),
                      default_alias: str | None = None) -> str | None:
    """SPEC 2.6: a reference the registry does not know (`jev-latest`, …) -> the default alias.

    Paths (contain a separator) and `repo[:quant]` references pass through untouched: they are
    resolved by the registry layer, not by the adapter. Only a bare, unknown alias is remapped —
    that is the documented Typesafe behaviour (`model: "jev-latest"` must not 404).
    """
    if ref is None:
        return default_alias
    if ref in known_aliases:
        return ref
    if "/" in ref or ref.endswith(".gguf") or ":" in ref:
        return ref
    return default_alias or ref
