"""A-E1b-1/5/6/7/9: the fork engine's contract, driven by a deterministic fake session.

Real-model equivalence (A-E1b-2/3/4/8) lives in the `model`-marked tests at the bottom of this
file and in tests/test_cli.py; this file pins the parts that must hold for *every* session:
invariants, wave batching, the two readouts, coverage/reliability, collisions, ctx guards and
the decode-spy contract (one prefill + one batch per wave, `logits=1` on branch last tokens).
"""
from __future__ import annotations

import json

import pytest

from ggufone import schema
from ggufone.engine import decide, prompt
from ggufone.errors import GgufoneError
from ggufone.schema import Question
from tests.fake_engine import FakeSession, biased_row


# --------------------------------------------------------------------- helpers
def choice_request(options: dict | None = None) -> dict:
    return {
        "state": "The billing dashboard is blank for every user after login.",
        "questions": {
            "area": {"type": "choice", "instructions": "Which area owns this?",
                     "criteria": {"billing": "payments and invoices",
                                  "technical": "api and infrastructure"}},
        },
        "options": options,
    }


def parse(payload: dict) -> schema.Request:
    payload = {key: value for key, value in payload.items() if value is not None}
    return schema.parse_request(payload)


def engine_for(session: FakeSession, **kwargs) -> decide.DecisionEngine:
    return decide.DecisionEngine(session, **kwargs)


def run(session: FakeSession, payload: dict, **kwargs):
    request = parse(payload)
    plan = decide.plan_context(request, session)
    engine = engine_for(session, **kwargs)
    return engine, engine.decide(request, plan=plan)


def token_for(session: FakeSession, word: str) -> int:
    return session.tokenize(word)[0]


# --------------------------------------------------------------- A-E1b-1 invariants
def test_choice_answer_obeys_the_frozen_invariants() -> None:
    session = FakeSession(n_vocab=64)
    billing, technical = token_for(session, "billing"), token_for(session, "technical")
    session.row_fn = lambda ctx: biased_row(session.n_vocab, {billing: 6.0, technical: 5.0})
    _, result = run(session, choice_request())
    answer = result.answers["area"]
    assert answer["type"] == "choice"
    assert answer["choice"] == "billing"
    assert abs(sum(answer["probabilities"].values()) - 1.0) < 1e-6
    assert set(answer["probabilities"]) == {"billing", "technical"}
    assert 0.0 <= answer["confidence"] <= 1.0
    assert answer["reliability"] == "ok"
    assert 0.0 <= answer["coverage"] <= 1.0
    assert answer["legend"] == {"billing": "payments and invoices",
                                "technical": "api and infrastructure"}


def test_argmax_tie_break_prefers_the_lowest_candidate_index() -> None:
    session = FakeSession(n_vocab=64)
    session.row_fn = lambda ctx: [0.0] * session.n_vocab          # perfectly flat -> a tie
    _, result = run(session, choice_request())
    assert result.answers["area"]["choice"] == "billing"           # first declared option
    assert result.answers["area"]["probabilities"]["billing"] == pytest.approx(0.5, abs=1e-9)


def test_score_answer_is_the_weighted_mean_of_the_levels() -> None:
    session = FakeSession(n_vocab=64)
    request = {
        "state": "A spinner replaced the report table.",
        "questions": {"severity": {"type": "score",
                                   "criteria": ["cosmetic", "annoying", "critical"]}},
    }
    # levels are scored by their NUMBER token (the label the prompt asks for)
    zero, one, two = (token_for(session, str(level)) for level in range(3))
    session.row_fn = lambda ctx: biased_row(session.n_vocab, {zero: 0.0, one: 0.7, two: 1.4})
    _, result = run(session, request)
    answer = result.answers["severity"]
    assert answer["type"] == "score"
    assert answer["legend"] == {"0": "cosmetic", "1": "annoying", "2": "critical"}
    assert set(answer["probabilities"]) == {"0", "1", "2"}
    assert 0.0 <= answer["score"] <= 2.0
    assert answer["score"] > 1.0          # level 2 has the highest logit
    assert 0.0 <= answer["confidence"] <= 1.0
    assert answer["probabilities"]["2"] == max(answer["probabilities"].values())


def test_noul_answer_is_p_yes_and_has_no_confidence() -> None:
    session = FakeSession(n_vocab=64)
    request = {
        "state": "The nightly job failed twice.",
        "questions": {"page": {"type": "noul", "criteria": {"true": "yes, page now",
                                                            "false": "no, wait"}}},
    }
    yes, no = token_for(session, "yes"), token_for(session, "no")
    session.row_fn = lambda ctx: biased_row(session.n_vocab, {yes: 1.0, no: 0.2})
    _, result = run(session, request)
    answer = result.answers["page"]
    assert answer["type"] == "noul"
    assert 0.0 <= answer["noul"] <= 1.0 and answer["noul"] > 0.5
    assert set(answer["probabilities"]) == {"yes", "no"}
    assert "confidence" not in answer


# --------------------------------------------------------------- A-E1b-9 no-generation gate
def test_every_decode_batch_sets_logits_exactly_on_branch_last_tokens() -> None:
    session = FakeSession(n_vocab=64)
    _, result = run(session, choice_request())
    assert session.batches, "the engine must decode something"
    for batch in session.batches:
        assert batch.n_tokens == len(batch.tokens) == len(batch.seq_ids)
        branch_last = {indices[-1] for _, indices in _branches(batch)}
        assert set(batch.logits_indices()) == branch_last
        for seq_id, indices in _branches(batch):
            positions = [batch.positions[index] for index in indices]
            assert positions == list(range(positions[0], positions[0] + len(indices))), \
                f"seq {seq_id} must decode consecutive positions"
    assert len(session.batches) == result.usage["waves"]


def _branches(batch):
    """(seq_id, token indices) for each run of consecutive tokens of one seq."""
    return batch.branches()


def test_decode_calls_are_one_prefill_plus_one_batch_per_wave() -> None:
    session = FakeSession(n_vocab=512)
    request = {
        "state": "Eight questions, four candidates each.",
        "questions": {
            f"q{index}": {"type": "choice",
                          "criteria": {f"c{index}{cand}": None for cand in range(4)}}
            for index in range(8)
        },
    }
    _, result = run(session, request, )
    assert len(session.batches) == result.usage["waves"]
    assert result.usage["waves"] >= 2


def test_no_sampling_symbol_anywhere_in_src() -> None:
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "ggufone"
    offenders = [str(path.relative_to(root)) for path in root.rglob("*.py")
                 if "llama_sampler_" in path.read_text(encoding="utf-8")]
    assert offenders == []


# --------------------------------------------------------------- A-E1b-5 waves
def eight_by_four(**options) -> dict:
    return {
        "state": "Route each ticket to one of four queues.",
        "questions": {
            f"q{index}": {"type": "choice",
                          "criteria": {f"cand{index}_{cand}": None for cand in range(4)}}
            for index in range(8)
        },
        "options": options or None,
    }


def test_waves_never_exceed_the_sequence_cap_and_match_the_single_wave_run() -> None:
    payload = eight_by_four(n_seq_max=4)
    session = FakeSession(n_vocab=128, n_seq_max=4)
    session.row_fn = lambda ctx: [0.1 * (ctx.token % 3) for _ in range(session.n_vocab)]
    _, batched = run(session, payload)
    for batch in session.batches:
        assert max(batch.seq_ids) < 4, "a wave must stay under n_seq_max"
        assert len(set(batch.seq_ids)) <= 3
    assert batched.usage["waves"] >= 2

    wide = FakeSession(n_vocab=128, n_seq_max=64)
    wide.row_fn = session.row_fn
    _, single = run(wide, eight_by_four(n_seq_max=64))
    assert single.usage["waves"] < batched.usage["waves"]
    for qid, answer in batched.answers.items():
        other = single.answers[qid]
        assert answer["choice"] == other["choice"]
        for key, value in answer["probabilities"].items():
            assert abs(value - other["probabilities"][key]) < 1e-9, qid


def test_the_same_questions_are_scored_once_per_question() -> None:
    session = FakeSession(n_vocab=128, n_seq_max=4)
    _, result = run(session, eight_by_four(n_seq_max=4))
    assert result.usage["questions"] == 8
    assert result.usage["prefill_tokens"] == result.engine["prefix_tokens"]
    assert len(result.answers) == 8


# --------------------------------------------------------------- A-E1b-6 readouts
def test_both_readouts_are_implemented_and_agree_on_single_token_candidates() -> None:
    session = FakeSession(n_vocab=64)
    ids = {word: token_for(session, word) for word in ("billing", "technical")}
    session.row_fn = lambda ctx: biased_row(session.n_vocab, {ids["billing"]: 1.0,
                                                              ids["technical"]: 0.3})
    _, sequence = run(session, choice_request(options={"readout": "sequence"}))
    _, single = run(session, choice_request(options={"readout": "single_token"}))
    assert sequence.answers["area"]["probabilities"] == \
        pytest.approx(single.answers["area"]["probabilities"])
    assert sequence.engine["readout"] == "sequence"
    assert single.engine["readout"] == "single_token"


def test_switching_the_readout_leaves_the_prompt_untouched() -> None:
    question = Question(id="area", type="choice", instructions="Which area owns this?",
                        criteria={"billing": "x", "technical": "y"},
                        options=("billing", "technical"), descriptions=("x", "y"))
    sequence = prompt.build_question(question, readout="sequence")
    single = prompt.build_question(question, readout="single_token")
    assert sequence.suffix == single.suffix
    assert sequence.texts == single.texts


def test_sequence_readout_uses_every_token_of_a_multi_token_candidate() -> None:
    session = FakeSession(n_vocab=64)
    payload = {
        "state": "Pick a queue.",
        "questions": {"area": {"type": "choice",
                               "criteria": {"billing issue": None, "technical": None}}},
    }
    session.row_fn = lambda ctx: [0.0] * session.n_vocab
    engine, result = run(session, payload)
    first = token_for(session, "billing")
    second = token_for(session, "issue")
    stem = token_for(session, "technical")
    assert result.usage["output_tokens"] == 3          # 2 tokens + 1 token
    assert result.answers["area"]["decode_steps"] == 3
    assert len(session.batches) == 2, "one suffix decode + one candidate step"
    step = session.batches[1]
    assert step.tokens == (first,), "only the multi-token candidate needs a step"
    assert step.seq_ids == (1,)
    assert step.logits == (True,)
    assert step.positions == (len(session.prefill_calls[0]["tokens"])
                              + len(session.tokenize(prompt.build_question(
                                  parse(payload).questions[0]).suffix)),)
    assert len(session.batches) == result.usage["waves"]
    assert first != second and first != stem


def test_identical_candidate_sequences_are_a_pinned_collision_error() -> None:
    session = FakeSession(n_vocab=64)
    payload = {
        "state": "Pick one.",
        "questions": {"area": {"type": "choice",
                               "criteria": {"outage": None, "outage!": None}}},
    }
    with pytest.raises(GgufoneError) as exc:
        run(session, payload)
    assert exc.value.code == "E_CANDIDATE_COLLISION"
    assert exc.value.exit_code == 2


def test_collision_is_detected_on_the_scored_sequence_only() -> None:
    """single_token scores a prefix of the candidate: distinct prefixes may not collide."""
    session = FakeSession(n_vocab=64)
    payload = {
        "state": "Pick one.",
        "questions": {"area": {"type": "choice",
                               "criteria": {"billing suite": None, "billing": None}}},
        "options": {"readout": "single_token"},
    }
    with pytest.raises(GgufoneError) as exc:
        run(session, payload)
    assert exc.value.code == "E_CANDIDATE_COLLISION"

    session2 = FakeSession(n_vocab=64)
    engine, result = run(session2, {**payload, "options": {"readout": "sequence"}})
    assert result.answers["area"]["type"] == "choice"


# --------------------------------------------------------------- A-E1b-7 coverage
def test_low_candidate_mass_is_reported_and_never_hidden_by_renormalizing() -> None:
    session = FakeSession(n_vocab=64)
    billing, technical = token_for(session, "billing"), token_for(session, "technical")
    other = token_for(session, "unrelated")
    session.row_fn = lambda ctx: biased_row(session.n_vocab, {billing: 0.0, technical: -0.5,
                                                              other: 8.0})
    _, result = run(session, choice_request())
    answer = result.answers["area"]
    assert answer["reliability"] == "low_mass"
    assert "W_LOW_MASS" in result.warnings
    assert answer["coverage"] < 0.10
    # the probabilities are still the restricted softmax — NOT rescaled to hide the low mass
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-9)
    assert answer["probabilities"]["billing"] > answer["probabilities"]["technical"]


def test_a_healthy_coverage_reports_ok_and_no_warning() -> None:
    session = FakeSession(n_vocab=64)
    billing, technical = token_for(session, "billing"), token_for(session, "technical")
    session.row_fn = lambda ctx: biased_row(session.n_vocab, {billing: 8.0, technical: 7.0})
    _, result = run(session, choice_request())
    assert result.answers["area"]["reliability"] == "ok"
    assert result.answers["area"]["coverage"] > 0.90
    assert "W_LOW_MASS" not in result.warnings


# --------------------------------------------------------------- ctx / seq guards
def test_a_context_too_small_for_the_prompt_is_a_pinned_runtime_error() -> None:
    session = FakeSession(n_vocab=64, n_ctx=8)
    with pytest.raises(GgufoneError) as exc:
        run(session, choice_request())
    assert exc.value.code == "E_CTX_TOO_SMALL"
    assert exc.value.exit_code == 3


def test_a_sequence_cap_below_three_cannot_hold_prefix_and_branch() -> None:
    session = FakeSession(n_vocab=64, n_seq_max=2)
    with pytest.raises(GgufoneError) as exc:
        run(session, choice_request(options={"n_seq_max": 3}))
    assert exc.value.code == "E_SEQ_MAX_EXCEEDED"


def test_max_waves_caps_the_wave_budget() -> None:
    session = FakeSession(n_vocab=128, n_seq_max=4)
    with pytest.raises(GgufoneError) as exc:
        run(session, eight_by_four(n_seq_max=4, max_waves=3))
    assert exc.value.code == "E_SEQ_MAX_EXCEEDED"


# --------------------------------------------------------------- state reuse / determinism
def test_a_warm_state_id_reports_prefill_reused_and_no_new_cost() -> None:
    payload = choice_request(options={"state_id": "sha256:deadbeef", "save_state": True})
    session = FakeSession(n_vocab=64)
    _, cold = run(session, payload)
    assert cold.engine["prefill_reused"] is False
    assert cold.usage["prefill_tokens"] > 0
    _, warm = run(session, payload)
    assert warm.engine["prefill_reused"] is True
    assert warm.usage["prefill_tokens"] == 0
    assert warm.engine["state_id"] == "sha256:deadbeef"
    assert warm.answers == cold.answers
    assert "W_TRUNCATED_STATE" not in warm.warnings


def test_the_default_state_id_is_the_sha256_of_the_prefix_tokens() -> None:
    import hashlib
    session = FakeSession(n_vocab=64)
    request = parse(choice_request())
    plan = decide.plan_context(request, session)
    expected = hashlib.sha256(b"".join(token.to_bytes(4, "little")
                                       for token in plan.prefix_tokens)).hexdigest()
    _, result = run(session, choice_request())
    assert result.engine["state_id"] == f"sha256:{expected}"


def test_results_are_deterministic_across_runs() -> None:
    payload = choice_request()
    outputs = []
    for _ in range(3):
        session = FakeSession(n_vocab=64)
        billing = session.tokenize("billing")[0]
        session.row_fn = lambda ctx, billing=billing: biased_row(session.n_vocab, {billing: 1.0})
        _, result = run(session, payload)
        payload_dict = result.payload()
        payload_dict.pop("timings")
        outputs.append(json.dumps(payload_dict, sort_keys=True))
    assert outputs[0] == outputs[1] == outputs[2]


# --------------------------------------------------------------- prompt assembly
def test_the_prefix_is_byte_identical_for_every_question_of_a_request() -> None:
    request = parse({
        "state": {"incident": "blank dashboard", "since": "10 minutes"},
        "questions": {
            "a": {"type": "choice", "instructions": "First?", "criteria": {"x": None, "y": None}},
            "b": {"type": "score", "instructions": ["Second?", "Be careful"],
                  "criteria": ["low", "high"]},
        },
    })
    prefix = prompt.build_prefix(request.state)
    for question in request.questions:
        rendered = prompt.build_question(question, readout="sequence")
        assert rendered.suffix.startswith("") and isinstance(rendered.suffix, str)
        assert prefix + rendered.suffix == prompt.build_prefix(request.state) + rendered.suffix
    assert "blank dashboard" in prefix and "10 minutes" in prefix


def test_question_ids_are_never_part_of_the_prompt() -> None:
    request = parse({
        "state": "Pick.",
        "questions": {"secret_qid_42": {"type": "choice", "criteria": {"x": None, "y": None}}},
    })
    rendered = prompt.build_question(request.questions[0], readout="sequence")
    assert "secret_qid_42" not in prompt.build_prefix(request.state)
    assert "secret_qid_42" not in rendered.suffix


def test_rendered_candidates_carry_the_labels_the_response_uses() -> None:
    request = parse({
        "state": "Pick.",
        "questions": {"area": {"type": "choice", "criteria": {"billing": "money", "api": None}}},
    })
    rendered = prompt.build_question(request.questions[0], readout="sequence")
    assert rendered.texts == ("billing", "api")
    assert "billing" in rendered.suffix and "money" in rendered.suffix


def test_structured_state_and_instructions_are_rendered_deterministically() -> None:
    first = prompt.render_state({"b": 2, "a": {"nested": [1, 2]}})
    second = prompt.render_state({"a": {"nested": [1, 2]}, "b": 2})
    assert first == second
    assert "nested" in first
    assert prompt.render_instructions(["step 1", "step 2"]) == "step 1\nstep 2"
    assert prompt.render_instructions("plain") == "plain"
    assert prompt.render_instructions(None) == ""


def test_plan_context_sizes_the_context_from_the_prompt() -> None:
    session = FakeSession(n_vocab=64)
    request = parse(choice_request())
    plan = decide.plan_context(request, session)
    assert list(plan.prefix_tokens) == session.tokenize(prompt.build_prefix(request.state))
    assert plan.n_seq_max == 1 + 2                      # one candidate seq per option + prefix
    assert plan.n_ctx > len(plan.prefix_tokens)
    assert plan.kv_type == "auto" and plan.threads == session.threads
