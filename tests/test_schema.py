"""A-E1b-13 (+ A-E1b-11): request validation, pinned E_* codes and response rendering.

Pure schema work: no model, no runtime, no decode. Every error here is a *user* error
(exit code 2) and must name one of the frozen codes from SPEC 2.5.
"""
from __future__ import annotations

import copy

import pytest

from ggufone import schema
from ggufone.errors import UserError


def request(**overrides) -> dict:
    payload = {
        "state": "The production dashboard is blank for every user.",
        "model": "spark-x2.5-4b-q8_0",
        "questions": {
            "severity": {"type": "choice",
                         "criteria": {"outage": "all users down", "cosmetic": "a typo"}},
        },
    }
    payload.update(overrides)
    return payload


def code_of(exc: pytest.ExceptionInfo[UserError]) -> str:
    return exc.value.code


# --------------------------------------------------------------- happy path
def test_choice_request_parses_with_the_documented_defaults() -> None:
    parsed = schema.parse_request(request())
    assert parsed.state.startswith("The production dashboard")
    assert parsed.format == "native"
    assert [q.id for q in parsed.questions] == ["severity"]
    question = parsed.questions[0]
    assert question.type == "choice"
    assert question.options == ("outage", "cosmetic")
    assert question.descriptions == ("all users down", "a typo")
    options = parsed.options
    assert (options.temperature, options.length_norm, options.readout) == (1.0, 1.0, "sequence")
    assert options.confidence_mode == "normalized_peak"
    assert options.coverage_floor == 0.10
    assert options.kv_type == "auto" and options.n_ctx is None and options.n_seq_max is None
    assert options.state_cache is True and options.save_state is False and options.strict is False
    assert options.seed == 0 and options.threads is None and options.backend == "auto"


def test_score_and_noul_questions_parse() -> None:
    parsed = schema.parse_request(request(questions={
        "sev": {"type": "score", "criteria": ["cosmetic", "annoying", "critical"]},
        "page": {"type": "noul", "criteria": {"true": "yes, page now", "false": "no"}},
    }))
    score, noul = parsed.questions
    assert score.options == ("0", "1", "2")            # level numbers, as documented
    assert score.descriptions == ("cosmetic", "annoying", "critical")
    assert noul.options == ("yes", "no")
    assert noul.descriptions == ("yes, page now", "no")


def test_question_ids_are_never_sent_to_the_model_but_survive() -> None:
    parsed = schema.parse_request(request())
    assert parsed.questions[0].id == "severity"


# --------------------------------------------------------------- state / qids
@pytest.mark.parametrize("state", [None, "", {}, []])
def test_empty_state_is_a_pinned_user_error(state) -> None:
    payload = request()
    if state is None:
        payload.pop("state")
    else:
        payload["state"] = state
    with pytest.raises(UserError) as exc:
        schema.parse_request(payload)
    assert code_of(exc) == "E_STATE_EMPTY"
    assert exc.value.exit_code == 2


@pytest.mark.parametrize("questions", [None, {}, [], "severity"])
def test_missing_or_empty_questions_is_a_qid_error(questions) -> None:
    payload = request()
    if questions is None:
        payload.pop("questions")
    else:
        payload["questions"] = questions
    with pytest.raises(UserError) as exc:
        schema.parse_request(payload)
    assert code_of(exc) == "E_QID_INVALID"


@pytest.mark.parametrize("qid", ["", 3, None])
def test_a_question_id_must_be_a_non_empty_string(qid) -> None:
    with pytest.raises(UserError) as exc:
        schema.parse_request(request(questions={qid: {"type": "noul"}}))
    assert code_of(exc) == "E_QID_INVALID"


# --------------------------------------------------------------- keys / options
def test_unknown_top_level_keys_are_always_an_error() -> None:
    with pytest.raises(UserError) as exc:
        schema.parse_request(request(readout="sequence"))
    assert code_of(exc) == "E_UNKNOWN_KEY"


def test_unknown_keys_inside_options_warn_and_pass_but_strict_rejects() -> None:
    payload = request(options={"temperature": 2.0, "verbose": True})
    parsed = schema.parse_request(payload)
    assert parsed.options.temperature == 2.0
    assert parsed.warnings == ("W_UNKNOWN_OPTION",)
    with pytest.raises(UserError) as exc:
        schema.parse_request(request(options={"strict": True, "verbose": True}))
    assert code_of(exc) == "E_UNKNOWN_KEY"


@pytest.mark.parametrize("bad", [0, -1.0, "hot"])
def test_a_non_positive_temperature_is_rejected(bad) -> None:
    with pytest.raises(UserError) as exc:
        schema.parse_request(request(options={"temperature": bad}))
    assert code_of(exc) == "E_UNKNOWN_KEY"


@pytest.mark.parametrize("fmt", ["json", "", 3])
def test_only_native_and_typesafe_formats_are_accepted(fmt) -> None:
    with pytest.raises(UserError) as exc:
        schema.parse_request(request(format=fmt))
    assert code_of(exc) == "E_UNKNOWN_KEY"


@pytest.mark.parametrize("readout,good", [("sequence", True), ("single_token", True),
                                          ("letters", False), (None, False), (7, False)])
def test_the_readout_mode_is_one_of_the_two_implemented(readout, good) -> None:
    payload = request(options={"readout": readout})
    if good:
        assert schema.parse_request(payload).options.readout == readout
    else:
        with pytest.raises(UserError) as exc:
            schema.parse_request(payload)
        assert code_of(exc) == "E_UNKNOWN_KEY"


# --------------------------------------------------------------- question body
def test_unknown_question_type_is_a_pinned_error() -> None:
    with pytest.raises(UserError) as exc:
        schema.parse_request(request(questions={"q": {"type": "ranking", "criteria": []}}))
    assert code_of(exc) == "E_Q_TYPE_UNKNOWN"


@pytest.mark.parametrize("body", [{}, {"criteria": {"a": None}}, "choice", 7, None])
def test_a_question_without_a_type_is_a_pinned_error(body) -> None:
    with pytest.raises(UserError) as exc:
        schema.parse_request(request(questions={"q": body}))
    assert code_of(exc) == "E_Q_TYPE_UNKNOWN"


# --------------------------------------------------------------- choice criteria
@pytest.mark.parametrize("criteria", [None, [], "a|b", {"a": "desc", 3: None}, {}])
def test_choice_criteria_shape_errors(criteria) -> None:
    with pytest.raises(UserError) as exc:
        schema.parse_request(request(questions={"q": {"type": "choice", "criteria": criteria}}))
    assert code_of(exc) == "E_CHOICE_CRITERIA"


def test_choice_accepts_up_to_255_options_and_rejects_256() -> None:
    ok = {f"option{i}": None for i in range(255)}
    parsed = schema.parse_request(request(questions={"q": {"type": "choice", "criteria": ok}}))
    assert len(parsed.questions[0].options) == 255
    too_many = {f"option{i}": None for i in range(256)}
    with pytest.raises(UserError) as exc:
        schema.parse_request(request(questions={"q": {"type": "choice", "criteria": too_many}}))
    assert code_of(exc) == "E_CHOICE_TOO_MANY"


def test_choice_descriptions_may_be_null_or_text() -> None:
    parsed = schema.parse_request(request(questions={
        "q": {"type": "choice", "criteria": {"a": None, "b": "some text", "c": 3}}}))
    assert parsed.questions[0].descriptions == (None, "some text", None)
    assert isinstance(parsed.questions[0].criteria["c"], int)   # structured guidance passes through


# --------------------------------------------------------------- score / noul
@pytest.mark.parametrize("levels", [[], ["only"], [f"l{i}" for i in range(11)], "not-a-list"])
def test_score_levels_outside_2_to_10_are_rejected(levels) -> None:
    with pytest.raises(UserError) as exc:
        schema.parse_request(request(questions={"q": {"type": "score", "criteria": levels}}))
    assert code_of(exc) == "E_SCORE_LEVELS"


@pytest.mark.parametrize("levels", [["a", "b"], [f"l{i}" for i in range(10)]])
def test_score_levels_2_to_10_are_accepted(levels) -> None:
    parsed = schema.parse_request(request(questions={"q": {"type": "score", "criteria": levels}}))
    assert parsed.questions[0].options == tuple(str(i) for i in range(len(levels)))


@pytest.mark.parametrize("criteria", ["yes", [1, 2], {"true": "y", "maybe": "?"}, {"false": "n"},
                                      {"true": None}, {"true": "y", "false": 1}])
def test_noul_criteria_shape_errors(criteria) -> None:
    with pytest.raises(UserError) as exc:
        schema.parse_request(request(questions={"q": {"type": "noul", "criteria": criteria}}))
    assert code_of(exc) == "E_NOUL_CRITERIA"


def test_noul_criteria_is_optional() -> None:
    parsed = schema.parse_request(request(questions={"q": {"type": "noul"}}))
    assert parsed.questions[0].options == ("yes", "no")
    assert parsed.questions[0].descriptions == (None, None)


# --------------------------------------------------------------- rendering
RESULT = {
    "model": "spark-x2.5-4b-q8_0",
    "engine": {"runtime": "llama.cpp b11026", "backend": "cpu", "readout": "sequence",
               "kv_unified": True, "n_ctx": 4096, "n_seq_max": 5, "prefix_tokens": 812,
               "state_id": "sha256:abc", "prefill_reused": False},
    "answers": {
        "severity": {"type": "choice", "choice": "outage",
                     "probabilities": {"outage": 0.6123456789, "cosmetic": 0.3876543211},
                     "confidence": 0.2246913578, "legend": {"outage": "all users down",
                                                            "cosmetic": "a typo"},
                     "coverage": 0.931234567, "reliability": "ok", "decode_steps": 1},
        "sev": {"type": "score", "score": 1.2999999999999998,
                "probabilities": {"0": 0.0, "1": 0.7, "2": 0.30000000000000004},
                "confidence": 0.55, "legend": {"0": "a", "1": "b", "2": "c"},
                "coverage": 0.9, "reliability": "ok", "decode_steps": 1},
        "page": {"type": "noul", "noul": 0.8123456,
                 "probabilities": {"yes": 0.8123456, "no": 0.1876544},
                 "coverage": 0.88, "reliability": "ok", "decode_steps": 1},
    },
    "usage": {"input_tokens": 900, "output_tokens": 3, "questions": 3, "forks": 6,
              "prefill_tokens": 812, "decode_steps": 9, "waves": 4},
    "timings": {"model_load_ms": 120.0, "prefill_ms": 430.0, "questions_ms": 41.0,
                "total_ms": 471.123456},
    "warnings": ["W_LOW_MASS"],
}


def test_native_rendering_keeps_the_frozen_key_set_and_rounds_to_six_significant_digits() -> None:
    out = schema.render_response(RESULT, format="native")
    assert out["engine"]["readout"] == "sequence"
    assert out["timings"]["total_ms"] == 471.123
    severity = out["answers"]["severity"]
    assert severity["probabilities"] == {"outage": 0.612346, "cosmetic": 0.387654}
    assert severity["confidence"] == 0.224691
    assert severity["coverage"] == 0.931235
    assert out["usage"]["prefill_tokens"] == 812
    assert out["warnings"] == ["W_LOW_MASS"]


def test_typesafe_rendering_drops_every_native_only_key() -> None:
    out = schema.render_response(RESULT, format="typesafe")
    assert set(out) == {"model", "answers", "usage"}
    assert out["usage"] == {"input_tokens": 900, "output_tokens": 3}
    choice = out["answers"]["severity"]
    assert set(choice) == {"type", "choice", "probabilities", "confidence"}
    score = out["answers"]["sev"]
    assert set(score) == {"type", "score", "probabilities", "confidence", "legend"}
    noul = out["answers"]["page"]
    assert set(noul) == {"type", "noul", "probabilities"}          # noul has no confidence
    assert noul["noul"] == 0.812346


def test_native_and_typesafe_carry_byte_comparable_numbers() -> None:
    native = schema.render_response(RESULT, format="native")
    typesafe = schema.render_response(RESULT, format="typesafe")
    for qid, answer in typesafe["answers"].items():
        source = native["answers"][qid]
        assert answer["probabilities"] == source["probabilities"]
        if "confidence" in answer:
            assert answer["confidence"] == source["confidence"]
        if "legend" in answer:
            assert answer["legend"] == source["legend"]


def test_score_is_rendered_at_wire_precision() -> None:
    out = schema.render_response(RESULT, format="typesafe")
    assert out["answers"]["sev"]["score"] == 1.3
    assert out["answers"]["sev"]["probabilities"] == {"0": 0.0, "1": 0.7, "2": 0.3}


def test_render_response_does_not_mutate_its_input() -> None:
    before = copy.deepcopy(RESULT)
    schema.render_response(RESULT, format="typesafe")
    assert before == RESULT


def test_an_unknown_format_is_rejected_at_render_time() -> None:
    with pytest.raises(UserError) as exc:
        schema.render_response(RESULT, format="yaml")
    assert code_of(exc) == "E_UNKNOWN_KEY"


def test_typesafe_model_mapping_follows_spec_2_6() -> None:
    """`jev-latest` (a bare, non-registry alias) -> the configured default alias."""
    known = ("spark-x2.5-4b-q8_0",)
    assert schema.adapter_model_ref("jev-latest", known_aliases=known,
                                    default_alias="spark-x2.5-4b-q8_0") == "spark-x2.5-4b-q8_0"
    assert schema.adapter_model_ref("spark-x2.5-4b-q8_0", known_aliases=known,
                                    default_alias="x") == "spark-x2.5-4b-q8_0"
    assert schema.adapter_model_ref(None, known_aliases=known, default_alias="x") == "x"
    # paths and repo[:quant] references are not aliases: the registry resolves them
    assert schema.adapter_model_ref("/m/models/a.gguf", known_aliases=known,
                                    default_alias="x") == "/m/models/a.gguf"
    assert schema.adapter_model_ref("XHToken/Spark-X2.5-4B-GGUF:Q8_0", known_aliases=known,
                                    default_alias="x") == "XHToken/Spark-X2.5-4B-GGUF:Q8_0"


# --------------------------------------------------------------- mutation-driven pins
# Written to kill the surviving mutants of this module (Tier M run — see the evidence report).
def test_validation_rejects_empty_strings_where_text_is_required() -> None:
    with pytest.raises(UserError) as exc:
        schema.parse_request(request(questions={"q": {"type": "score",
                                                      "criteria": ["ok level", "   "]}}))
    assert code_of(exc) == "E_SCORE_LEVELS"
    with pytest.raises(UserError) as exc:
        schema.parse_request(request(questions={"q": {"type": "choice",
                                                      "criteria": {"   ": None, "b": None}}}))
    assert code_of(exc) == "E_CHOICE_CRITERIA"
    with pytest.raises(UserError) as exc:
        schema.parse_request(request(questions={
            "q": {"type": "noul", "criteria": {"true": "yes", "false": "   "}}}))
    assert code_of(exc) == "E_NOUL_CRITERIA"


def test_option_value_edges_are_inclusive_where_documented() -> None:
    assert schema.parse_request(
        request(options={"coverage_floor": 0.0})).options.coverage_floor == 0.0
    assert schema.parse_request(
        request(options={"coverage_floor": 1.0})).options.coverage_floor == 1.0
    with pytest.raises(UserError) as exc:
        schema.parse_request(request(options={"coverage_floor": 1.5}))
    assert code_of(exc) == "E_UNKNOWN_KEY"
    assert schema.parse_request(request(options={"length_norm": 0})).options.length_norm == 0.0
    with pytest.raises(UserError) as exc:
        schema.parse_request(request(options={"length_norm": -0.1}))
    assert code_of(exc) == "E_UNKNOWN_KEY"
    with pytest.raises(UserError) as exc:
        schema.parse_request(request(options={"state_id": "   "}))
    assert code_of(exc) == "E_UNKNOWN_KEY"
    # booleans are not integers for the numeric options
    for name in ("threads", "n_seq_max", "seed", "max_waves"):
        with pytest.raises(UserError) as exc:
            schema.parse_request(request(options={name: True}))
        assert code_of(exc) == "E_UNKNOWN_KEY", name
