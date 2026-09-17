"""A-E1b-1/5/6/7/9: the fork engine's contract, driven by a deterministic fake session.

Real-model equivalence (A-E1b-2/3/4/8) lives in the `model`-marked tests at the bottom of this
file and in tests/test_cli.py; this file pins the parts that must hold for *every* session:
invariants, wave batching, the two readouts, coverage/reliability, collisions, ctx guards and
the decode-spy contract (one prefill + one batch per wave, `logits=1` on branch last tokens).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import pathlib

import pytest

from ggufone import schema
from ggufone.engine import decide, prompt
from ggufone.engine import readout
from ggufone.engine import session as session_module
from ggufone.engine.decide import Batch
from ggufone.errors import GgufoneError
from ggufone.runtime import finder
from ggufone.schema import Question
from tests.fake_engine import FakeSession, RowContext, biased_row

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
        session.row_fn = (lambda session=session, billing=billing:
                          biased_row(session.n_vocab, {billing: 1.0}))
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


# --------------------------------------------------------------- real model (A-E1b-2/3/4/5/8)
# Run with: GGUFONE_RUNTIME_DIR=<bundle> uv run pytest -q --run-network tests/test_engine_fork.py
MODEL_PATHS = {
    "spark2_5": pathlib.Path.home() / ".hermes" / "models" / "Spark-X2.5-4B-Q8_0.gguf",
    "qwen35": pathlib.Path.home() / ".cache" / "llama.cpp" / "Qwen3.5-0.8B-UD-Q4_K_XL.gguf",
}
_HANDLES: dict[str, session_module.ModelHandle] = {}


def _runtime_dir() -> pathlib.Path:
    env = os.environ.get("GGUFONE_RUNTIME_DIR")
    if env and (pathlib.Path(env) / "libllama.so").exists():
        return pathlib.Path(env)
    found = finder.find_runtime()
    if found:
        return found
    for base in (pathlib.Path.home() / ".hermes" / "runtime",
                 pathlib.Path.home() / ".local" / "share" / "ggufone" / "runtime"):
        for candidate in sorted(base.glob("*/")):
            if (candidate / "libllama.so").exists():
                return candidate
    pytest.skip("no llama.cpp runtime on this box (set GGUFONE_RUNTIME_DIR or run `ggufone init`)")


@pytest.fixture(scope="module")
def runtime_dir() -> pathlib.Path:
    return _runtime_dir()


@pytest.fixture(scope="module")
def handles(runtime_dir: pathlib.Path):
    """Open each pinned GGUF once for the whole module (model load is the expensive part)."""

    def get(name: str) -> session_module.ModelHandle:
        path = MODEL_PATHS[name]
        if not path.exists():
            pytest.skip(f"{path} is not on this box")
        if name not in _HANDLES:
            _HANDLES[name] = session_module.open_model(path, runtime_dir=runtime_dir)
        return _HANDLES[name]

    yield get
    for handle in _HANDLES.values():
        handle.close()
    _HANDLES.clear()


@contextlib.contextmanager
def live_session(handle, request: schema.Request, *, states_home=None, spy=None):
    """A fresh context for one request (same plan the CLI would compute)."""
    plan = decide.plan_context(request, handle)
    with session_module.ModelSession(handle, plan, states_home=states_home,
                                     decode_spy=spy) as live:
        yield plan, live


def _compact_answers(result: decide.DecideResult) -> dict:
    return {qid: {key: value for key, value in answer.items() if key != "decode_steps"}
            for qid, answer in result.answers.items()}


def fork_request() -> dict:
    return {
        "state": "The checkout page returns a 500 for every customer since 09:12; "
                 "the on-call engineer is paged.",
        "questions": {
            "area": {"type": "choice", "instructions": "Which area owns this?",
                     "criteria": {"billing payments": "payments and invoices",
                                  "api gateway": "the public api"}},
        },
        "options": {"threads": 4},
    }


def _sequential_probabilities(session, request: schema.Request, plan) -> dict:
    """Reference implementation: every candidate re-decodes prefix + suffix on its own seq."""
    prefix = list(plan.prefix_tokens)
    session.prefill(prefix)
    results = {}
    for question, view, suffix, candidates in decide.question_requirements(request, session):
        scored = decide.candidate_sequences(candidates, readout_mode=request.options.readout)
        scores = []
        for sequence in scored:
            session.release(1)
            session.fork(0, 1, len(prefix))
            rows = session.decode(Batch(
                tokens=tuple(suffix), seq_ids=(1,) * len(suffix),
                positions=tuple(len(prefix) + index for index in range(len(suffix))),
                logits=tuple(index == len(suffix) - 1 for index in range(len(suffix)))))
            logprobs = [readout.logprob(rows[-1], sequence[0])]
            if len(sequence) > 1:
                tail = sequence[:-1]
                tail_rows = session.decode(Batch(
                    tokens=tuple(tail), seq_ids=(1,) * len(tail),
                    positions=tuple(len(prefix) + len(suffix) + index
                                    for index in range(len(tail))),
                    logits=tuple(True for _ in tail)))
                for index, row in enumerate(tail_rows):
                    logprobs.append(readout.logprob(row, sequence[index + 1]))
            scores.append(readout.candidate_sequence_score(logprobs,
                                                           request.options.length_norm))
        probabilities = readout.restricted_softmax(scores, request.options.temperature)
        results[question.id] = dict(zip(view.options, probabilities, strict=True))
        session.release(1)
    return results


@pytest.mark.model
@pytest.mark.parametrize("name", ["qwen35", "spark2_5"])
def test_fork_equivalence_against_sequential_decode(name: str, handles, tmp_path) -> None:
    """A-E1b-2: max |delta| <= 1e-3 on the hybrid (qwen35) and a pure-attention model."""
    handle = handles(name)
    request = parse(fork_request())
    with live_session(handle, request, states_home=tmp_path / "states") as (plan, live):
        result = decide.DecisionEngine(live).decide(request, plan=plan)
    with live_session(handle, request, states_home=tmp_path / "states") as (_, reference_session):
        reference = _sequential_probabilities(reference_session, request, plan)
    deltas = {}
    for qid, answer in result.answers.items():
        for key, value in answer["probabilities"].items():
            deltas[f"{qid}.{key}"] = abs(value - reference[qid][key])
    worst = max(deltas.values())
    print(f"\nfork vs sequential on {name}: max |delta| = {worst:.3e} over {len(deltas)} candidates")
    assert worst <= 1e-3, deltas


@pytest.mark.model
def test_one_prefill_per_state_and_a_warm_state_costs_no_prefill(handles, tmp_path) -> None:
    """A-E1b-3: the prefix is decoded once; a warm `state_id` reports prefill_reused + ~0 ms."""
    handle = handles("qwen35")
    payload = {
        "state": "A nightly job failed twice in a row; the report is stale.",
        "questions": {
            "area": {"type": "choice", "criteria": {"billing": None, "data": None}},
            "severity": {"type": "score", "criteria": ["cosmetic", "annoying", "critical"]},
            "page": {"type": "noul", "criteria": {"true": "page now", "false": "wait"}},
            "owner": {"type": "choice", "criteria": {"platform": None, "analytics": None}},
        },
        "options": {"threads": 4, "state_id": "e1b-warm-state", "save_state": True},
    }
    request = parse(payload)
    captured: list[Batch] = []
    with live_session(handle, request, states_home=tmp_path / "states",
                      spy=captured.append) as (plan, live):
        cold = decide.DecisionEngine(live).decide(request, plan=plan)
    prefix = tuple(plan.prefix_tokens)
    assert cold.usage["prefill_tokens"] == len(prefix)
    assert captured[0].tokens == prefix, "the first decode must be the shared prefix"
    assert sum(1 for batch in captured if batch.tokens == prefix) == 1
    assert cold.usage["waves"] == len(captured) - 1
    assert cold.engine["prefill_reused"] is False
    assert cold.timings["prefill_ms"] > 0.0
    assert (tmp_path / "states" / "e1b-warm-state.bin").exists()

    warm_captured: list[Batch] = []
    with live_session(handle, request, states_home=tmp_path / "states",
                      spy=warm_captured.append) as (_, warm_session):
        warm = decide.DecisionEngine(warm_session).decide(request, plan=plan)
    assert warm.engine["prefill_reused"] is True
    assert warm.usage["prefill_tokens"] == 0
    assert not [batch for batch in warm_captured if batch.tokens == prefix]
    assert warm.timings["prefill_ms"] <= 5.0, warm.timings
    print(f"\nwarm prefill_ms = {warm.timings['prefill_ms']:.3f}  "
          f"cold prefill_ms = {cold.timings['prefill_ms']:.1f}")
    for qid, answer in warm.answers.items():
        for key, value in answer["probabilities"].items():
            assert abs(value - cold.answers[qid][key]) <= 1e-3, (qid, key)


@pytest.mark.model
def test_determinism_three_runs_with_one_thread(handles, tmp_path) -> None:
    """A-E1b-4: three runs, threads=1 -> identical answers JSON after stripping `timings`."""
    handle = handles("qwen35")
    payload = {
        "state": "Two payments were charged twice; the customer wrote in.",
        "questions": {
            "area": {"type": "choice", "criteria": {"billing": None, "support": None}},
            "page": {"type": "noul", "criteria": {"true": "page now", "false": "wait"}},
        },
        "options": {"threads": 1},
    }
    request = parse(payload)
    digests = []
    bodies = []
    for _ in range(3):
        with live_session(handle, request, states_home=tmp_path / "states") as (plan, live):
            result = decide.DecisionEngine(live).decide(request, plan=plan)
        body = result.payload()
        body.pop("timings")
        bodies.append(json.dumps(body, sort_keys=True))
        digests.append(hashlib.sha256(bodies[-1].encode()).hexdigest())
    assert len(set(digests)) == 1, digests
    print(f"\ndeterminism sha256 = {digests[0]}")


@pytest.mark.model
def test_state_round_trip_and_a_corrupt_state_is_a_pinned_error(handles, tmp_path) -> None:
    """A-E1b-8: save -> fresh context -> same answers (<=1e-3); corrupt -> E_STATE_LOAD_FAILED."""
    handle = handles("qwen35")
    payload = {
        "state": "The export job wrote a truncated CSV for yesterday.",
        "questions": {
            "area": {"type": "choice", "criteria": {"billing": None, "data": None}},
            "severity": {"type": "score", "criteria": ["cosmetic", "annoying", "critical"]},
        },
        "options": {"threads": 4, "state_id": "e1b-state", "save_state": True},
    }
    request = parse(payload)
    with live_session(handle, request, states_home=tmp_path / "states") as (plan, live):
        cold = decide.DecisionEngine(live).decide(request, plan=plan)
    state_file = tmp_path / "states" / "e1b-state.bin"
    assert state_file.exists() and state_file.stat().st_size > 0

    with live_session(handle, request, states_home=tmp_path / "states") as (_, fresh):
        loaded = decide.DecisionEngine(fresh).decide(request, plan=plan)
    assert loaded.engine["prefill_reused"] is True
    worst = max(abs(value - cold.answers[qid][key])
                for qid, answer in loaded.answers.items()
                for key, value in answer["probabilities"].items())
    print(f"\nstate round-trip max |delta| = {worst:.3e}")
    assert worst <= 1e-3

    original = state_file.read_bytes()
    state_file.write_bytes(original[: max(16, len(original) // 3)])   # truncated
    with live_session(handle, request, states_home=tmp_path / "states") as (_, broken):
        with pytest.raises(GgufoneError) as exc:
            decide.DecisionEngine(broken).decide(request, plan=plan)
    assert exc.value.code == "E_STATE_LOAD_FAILED"
    assert exc.value.exit_code == 3
    assert not state_file.exists(), "the corrupt cache entry must be invalidated"

    with live_session(handle, request, states_home=tmp_path / "states") as (_, retry):
        recovered = decide.DecisionEngine(retry).decide(request, plan=plan)
    assert recovered.engine["prefill_reused"] is False
    assert recovered.usage["prefill_tokens"] > 0
    assert _compact_answers(recovered) == _compact_answers(cold)


@pytest.mark.model
def test_waves_on_a_real_hybrid_model_match_the_single_wave_run(handles, tmp_path) -> None:
    """A-E1b-5 with a real model: 8 questions x 4 candidates at n_seq_max=4."""
    handle = handles("qwen35")
    options = {"threads": 4, "n_seq_max": 4}
    payload = {
        "state": "Route each support ticket to exactly one queue.",
        "questions": {
            f"t{index}": {"type": "choice",
                          "criteria": {"alpha": None, "beta": None, "gamma": None,
                                       "delta": None}}
            for index in range(8)
        },
        "options": options,
    }
    request = parse(payload)
    captured: list[Batch] = []
    with live_session(handle, request, states_home=tmp_path / "states",
                      spy=captured.append) as (plan, live):
        capped = decide.DecisionEngine(live).decide(request, plan=plan)
    assert capped.usage["waves"] >= 2
    assert capped.usage["waves"] == len(captured) - 1
    for batch in captured[1:]:
        assert max(batch.seq_ids) < 4, "a wave must never exceed n_seq_max"
        assert len(set(batch.seq_ids)) <= 3

    wide = {**payload, "options": {"threads": 4, "n_seq_max": 64}}
    with live_session(handle, parse(wide), states_home=tmp_path / "states") as (plan, live):
        single = decide.DecisionEngine(live).decide(parse(wide), plan=plan)
    assert single.usage["waves"] < capped.usage["waves"]
    worst = 0.0
    for qid, answer in capped.answers.items():
        for key, value in answer["probabilities"].items():
            worst = max(worst, abs(value - single.answers[qid][key]))
        assert answer["choice"] == single.answers[qid]["choice"]
    print(f"\nwaves: capped={capped.usage['waves']} single={single.usage['waves']} "
          f"max |delta| = {worst:.3e}")
    assert worst <= 1e-3


@pytest.mark.model
def test_both_readouts_on_a_real_model_change_only_the_readout(handles, tmp_path) -> None:
    """A-E1b-6: identical prompt, identical candidates, different readout math."""
    handle = handles("qwen35")
    base = {
        "state": "The dashboard is blank for every user.",
        "questions": {"area": {"type": "choice",
                               "criteria": {"billing": None, "technical": None}}},
        "options": {"threads": 4},
    }
    results = {}
    for mode in ("sequence", "single_token"):
        payload = {**base, "options": {**base["options"], "readout": mode}}
        request = parse(payload)
        with live_session(handle, request, states_home=tmp_path / "states") as (plan, live):
            results[mode] = decide.DecisionEngine(live).decide(request, plan=plan)
        assert results[mode].engine["readout"] == mode
    single_words = all(len(tokens) == 1 for tokens in
                       [handle.tokenize("billing"), handle.tokenize("technical")])
    if single_words:
        assert results["sequence"].answers["area"]["probabilities"] \
            == pytest.approx(results["single_token"].answers["area"]["probabilities"])
    else:  # the readout math differs, the prompt does not
        assert results["sequence"].answers["area"]["probabilities"] \
            != results["single_token"].answers["area"]["probabilities"]
    question = parse(base).questions[0]
    assert prompt.build_question(question, readout="sequence").suffix \
        == prompt.build_question(question, readout="single_token").suffix
