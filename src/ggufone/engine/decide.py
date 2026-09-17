"""Top-level decide(): request -> validated plan -> answers envelope (SPEC 2.3).

Milestone: E1b.

The engine talks to a narrow session seam (`Tokenize + prefill/fork/release/decode`) so the
whole decision contract — invariants, waves, both readouts, coverage, collisions, the decode
spy — is testable without a model, while `engine/session.py` owns every libllama call.

Mechanics implemented here (SPEC 2.3 steps 1-10):

  prefill     the shared prefix is decoded once on seq 0 (SPEC 2.3.2)
  fork        per question group: seq 1 takes the prefix (`seq_cp(0 -> 1, 0, prefix_len)`),
              the question suffix is decoded there, and every further candidate seq is a copy
              of `prefix + suffix` — so candidate 0 shares the suffix decode instead of
              repeating it (peak seqs = 1 + candidates, SPEC 2.2)
  branches    one branch = one (question, candidate) pair; the branch's last token is the
              position whose row the readout needs, and it is the ONLY position flagged
              `logits=1` (A-E1b-9)
  waves       branches are packed into waves of `n_seq_max - 1` candidates; a question that
              does not fit is split into several waves, and the answer must not depend on the
              split (A-E1b-5)
  scoring     z_c = sum(log p(tok_i | prefix, suffix, tok_<i)) / L ** length_norm, then the
              restricted softmax over the question's candidates (SPEC 2.3.4/5)
  coverage    from the full-vocab row at the decision position, before any renormalization
              (`readout.coverage_from_row`); below the floor -> low_mass + W_LOW_MASS
  no sampling no sampler, no token loop: `decode_calls == 1 + waves`, and no code path here
              can generate text (A-E1b-9)
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ggufone import schema
from ggufone.engine import prompt, readout
from ggufone.errors import (
    CandidateCollisionError,
    ContextTooSmallError,
    SeqMaxExceededError,
    UserError,
)

CONTEXT_MARGIN = 32          # SPEC 2.2: "prefix tokens + longest suffix + margin (32)"
MIN_SEQ_MAX = 3              # prefix + question + one candidate


@dataclass(frozen=True, slots=True)
class SessionMeta:
    """What the session reports about itself (surfaced in `response.engine`)."""

    runtime: str
    backend: str
    n_ctx: int
    n_seq_max: int
    kv_unified: bool
    threads: int
    n_vocab: int
    model_path: str = ""
    model_alias: str | None = None
    load_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class PrefillInfo:
    prefill_tokens: int
    prefill_ms: float
    prefill_reused: bool
    state_id: str | None = None
    state_path: str | None = None


@dataclass(frozen=True, slots=True)
class Batch:
    """One `llama_decode` call: parallel per-token arrays, in batch order."""

    tokens: tuple[int, ...]
    seq_ids: tuple[int, ...]
    positions: tuple[int, ...]
    logits: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.tokens:
            raise ValueError("a batch must carry at least one token")
        lengths = {len(self.tokens), len(self.seq_ids), len(self.positions), len(self.logits)}
        if len(lengths) != 1:
            raise ValueError("batch arrays must have the same length")

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    def logits_indices(self) -> tuple[int, ...]:
        return tuple(index for index, flag in enumerate(self.logits) if flag)

    def branches(self) -> list[tuple[int, tuple[int, ...]]]:
        """Runs of consecutive tokens belonging to one seq: `(seq_id, indices)`."""
        out: list[tuple[int, tuple[int, ...]]] = []
        start = 0
        for index in range(1, len(self.seq_ids) + 1):
            if index == len(self.seq_ids) or self.seq_ids[index] != self.seq_ids[start]:
                out.append((self.seq_ids[start], tuple(range(start, index))))
                start = index
        return out


@dataclass(frozen=True, slots=True)
class ContextPlan:
    """What the request needs from a context (SPEC 2.2 sizing rules)."""

    prefix_tokens: tuple[int, ...]
    n_ctx: int
    n_seq_max: int
    threads: int
    kv_type: str
    # per-question requirements, for the ctx guard
    max_question_tokens: int = 0

    @property
    def n_prefix(self) -> int:
        return len(self.prefix_tokens)


class Tokenizer(Protocol):
    def tokenize(self, text: str) -> list[int]: ...


class Session(Tokenizer, Protocol):
    meta: SessionMeta

    def prefill(self, tokens: Sequence[int], *, state_id: str | None = None,
                state_cache: bool = True, save_state: bool = False) -> PrefillInfo: ...

    def fork(self, src: int, dst: int, upto: int) -> None: ...

    def release(self, seq: int) -> None: ...

    def decode(self, batch: Batch) -> list[list[float]]: ...


@dataclass
class DecideResult:
    """The native response body (SPEC 2.5), before rounding by `schema.render_response`."""

    model: str | None
    engine: dict[str, Any]
    answers: dict[str, Any]
    usage: dict[str, Any]
    timings: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def payload(self) -> dict[str, Any]:
        return {"model": self.model, "engine": self.engine, "answers": self.answers,
                "usage": self.usage, "timings": self.timings, "warnings": list(self.warnings)}


# ----------------------------------------------------------------------- planning
def question_requirements(request: schema.Request, tokenizer: Tokenizer,
                          ) -> list[tuple[schema.Question, prompt.RenderedQuestion,
                                          list[int], list[list[int]]]]:
    """Render + tokenize every question: the one place that decides what a candidate costs."""
    rendered: list[tuple[schema.Question, prompt.RenderedQuestion, list[int], list[list[int]]]] = []
    for question in request.questions:
        view = prompt.build_question(question, readout=request.options.readout)
        suffix_tokens = tokenizer.tokenize(view.suffix)
        candidates = [tokenizer.tokenize(text) for text in view.texts]
        for text, tokens in zip(view.texts, candidates, strict=True):
            if not tokens:
                raise UserError(
                    f"question {question.id!r}: candidate {text!r} tokenizes to zero tokens",
                    code=prompt.empty_candidate_code(question.type))
        rendered.append((question, view, suffix_tokens, candidates))
    return rendered


def candidate_sequences(candidates: Sequence[Sequence[int]], *,
                        readout_mode: str) -> list[tuple[int, ...]]:
    """The token sequence the readout actually scores for each candidate (collision domain)."""
    return [tuple(tokens[:1]) if readout_mode == "single_token" else tuple(tokens)
            for tokens in candidates]


def plan_context(request: schema.Request, tokenizer: Tokenizer) -> ContextPlan:
    """Size the context from the request itself (SPEC 2.2)."""
    options = request.options
    prefix_tokens = tokenizer.tokenize(prompt.build_prefix(request.state))
    requirements = question_requirements(request, tokenizer)
    per_question = [len(suffix) + max(len(tokens) for tokens in candidates)
                    for _, _, suffix, candidates in requirements]
    max_question = max(per_question, default=0)
    max_candidates = max((len(candidates) for _, _, _, candidates in requirements), default=0)
    meta = getattr(tokenizer, "meta", None)
    meta_threads = getattr(meta, "threads", 0)
    threads = options.threads or meta_threads or _default_threads()
    n_ctx = options.n_ctx or (len(prefix_tokens) + max_question + CONTEXT_MARGIN)
    n_seq_max = options.n_seq_max or max(MIN_SEQ_MAX, 1 + max_candidates)
    return ContextPlan(prefix_tokens=tuple(prefix_tokens), n_ctx=int(n_ctx),
                       n_seq_max=int(n_seq_max), threads=int(threads),
                       kv_type=options.kv_type, max_question_tokens=max_question)


def _default_threads() -> int:
    import os
    return os.cpu_count() or 1


def state_id_for(prefix_tokens: Sequence[int]) -> str:
    """SPEC 2.3.10: default `state_id` = SHA-256 of the prefix tokens."""
    digest = hashlib.sha256()
    for token in prefix_tokens:
        digest.update(int(token).to_bytes(4, "little", signed=True))
    return f"sha256:{digest.hexdigest()}"


# ------------------------------------------------------------------------- engine
class DecisionEngine:
    """The decision core: a session + the frozen readout options."""

    def __init__(self, session: Session, *, coverage_floor: float | None = None,
                 confidence_floor: float | None = None,
                 context_margin: int = CONTEXT_MARGIN) -> None:
        self.session = session
        self.coverage_floor = coverage_floor
        self.confidence_floor = confidence_floor
        self.context_margin = context_margin
        self.batches: list[Batch] = []      # every decode issued for this engine (the spy)
        self.forks = 0

    # ---- decode seam (everything the engine decodes goes through here)
    def _decode(self, batch: Batch) -> list[list[float]]:
        self.batches.append(batch)
        return self.session.decode(batch)

    def _fork(self, src: int, dst: int, upto: int) -> None:
        self.forks += 1
        self.session.fork(src, dst, upto)

    def decide(self, request: schema.Request, *, plan: ContextPlan | None = None,
               model_alias: str | None = None) -> DecideResult:
        started = time.perf_counter()
        session = self.session
        meta = session.meta
        plan = plan or plan_context(request, session)
        requirements = question_requirements(request, session)
        self._guard_context(request, plan, meta, requirements)
        options = request.options
        coverage_floor = options.coverage_floor if self.coverage_floor is None \
            else self.coverage_floor

        state_id = options.state_id or state_id_for(plan.prefix_tokens)
        prefill = session.prefill(list(plan.prefix_tokens), state_id=state_id,
                                 state_cache=options.state_cache, save_state=options.save_state)
        self.batches.clear()
        self.forks = 0

        question_started = time.perf_counter()
        answers: dict[str, Any] = {}
        warnings: list[str] = list(request.warnings)
        input_tokens = len(plan.prefix_tokens)
        output_tokens = 0
        for question, view, suffix_tokens, candidates in requirements:
            input_tokens += len(suffix_tokens)
            answer, scored = self._answer_question(question, view, suffix_tokens, candidates,
                                                   plan, options, coverage_floor, warnings)
            answers[question.id] = answer
            output_tokens += scored
        questions_ms = (time.perf_counter() - question_started) * 1000.0
        loops = len(self.batches)
        return DecideResult(
            model=model_alias or meta.model_alias,
            engine={
                "runtime": meta.runtime,
                "backend": meta.backend,
                "readout": options.readout,
                "kv_unified": bool(meta.kv_unified),
                "n_ctx": meta.n_ctx,
                "n_seq_max": meta.n_seq_max,
                "prefix_tokens": len(plan.prefix_tokens),
                "state_id": state_id,
                "prefill_reused": bool(prefill.prefill_reused),
            },
            answers=answers,
            usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "questions": len(request.questions),
                "forks": self.forks,
                "prefill_tokens": prefill.prefill_tokens,
                "decode_steps": output_tokens,
                "waves": loops,
            },
            timings={
                "model_load_ms": meta.load_ms,
                "prefill_ms": prefill.prefill_ms,
                "questions_ms": questions_ms,
                "total_ms": (time.perf_counter() - started) * 1000.0,
            },
            warnings=warnings,
        )

    # ---- guards
    def _guard_context(self, request: schema.Request, plan: ContextPlan, meta: SessionMeta,
                       requirements: Sequence[tuple[schema.Question, prompt.RenderedQuestion,
                                                     list[int], list[list[int]]]]) -> None:
        options = request.options
        if plan.n_seq_max < MIN_SEQ_MAX:
            raise SeqMaxExceededError(
                "E_SEQ_MAX_EXCEEDED: n_seq_max must be >= "
                f"{MIN_SEQ_MAX} (prefix + question + one candidate); got {plan.n_seq_max}")
        if plan.n_seq_max > meta.n_seq_max:
            raise SeqMaxExceededError(
                f"E_SEQ_MAX_EXCEEDED: this request needs {plan.n_seq_max} sequences "
                f"(prefix + candidates) but the context was created with n_seq_max="
                f"{meta.n_seq_max}; raise options.n_seq_max or lower the option count")
        needed = len(plan.prefix_tokens) + plan.max_question_tokens + self.context_margin
        if needed > meta.n_ctx:
            raise ContextTooSmallError(
                f"E_CTX_TOO_SMALL: the prompt needs {needed} tokens (prefix "
                f"{len(plan.prefix_tokens)} + longest question {plan.max_question_tokens} + "
                f"margin {self.context_margin}) but the context holds {meta.n_ctx}; "
                f"raise options.n_ctx")
        waves = planned_waves(requirements, plan.n_seq_max, readout_mode=options.readout)
        if options.max_waves is not None and waves > options.max_waves:
            raise SeqMaxExceededError(
                f"E_SEQ_MAX_EXCEEDED: this request needs {waves} waves but max_waves is "
                f"{options.max_waves}; raise options.max_waves or options.n_seq_max")

    # ---- one question -> one answer
    def _answer_question(self, question: schema.Question, view: prompt.RenderedQuestion,
                         suffix_tokens: list[int], candidates: list[list[int]],
                         plan: ContextPlan, options: schema.Options,
                         coverage_floor: float, warnings: list[str],
                         ) -> tuple[dict[str, Any], int]:
        scored = candidate_sequences(candidates, readout_mode=options.readout)
        _check_collisions(question, view, scored)
        per_wave = max(1, plan.n_seq_max - 1)
        logprobs: list[list[float]] = [[] for _ in candidates]
        coverage_pairs: list[tuple[list[float], int]] = []
        for group in _chunks(list(range(len(candidates))), per_wave):
            rows = self._score_group(plan, suffix_tokens,
                                     [candidates[index] for index in group],
                                     [scored[index] for index in group])
            decision_row = rows["decision_row"]
            for offset, index in enumerate(group):
                logprobs[index] = rows["logprobs"][offset]
                coverage_pairs.append((decision_row, scored[index][0]))

        z = [readout.candidate_sequence_score(values, options.length_norm)
             for values in logprobs]
        probabilities = readout.restricted_softmax(z, options.temperature)
        coverage = 0.0
        for row, token_id in coverage_pairs:
            coverage += readout.coverage_from_row(row, [token_id])
        coverage = min(1.0, coverage)
        confidence_value = readout.confidence(probabilities, options.confidence_mode)
        reliability = readout.reliability(coverage, coverage_floor=coverage_floor,
                                          confidence_floor=self.confidence_floor,
                                          confidence_value=confidence_value)
        if reliability == "low_mass":
            _add_warning(warnings, "W_LOW_MASS")
        elif reliability == "low_confidence":
            _add_warning(warnings, "W_LOW_CONFIDENCE")

        keys = list(question.options)
        probability_map = {key: probability for key, probability in zip(keys, probabilities,
                                                                       strict=True)}
        legend = {key: description for key, description in zip(keys, question.descriptions,
                                                              strict=True)
                  if description is not None}
        decode_steps = sum(len(tokens) for tokens in scored)
        if question.type == "choice":
            answer: dict[str, Any] = {
                "type": "choice",
                "choice": keys[readout.argmax_first(probabilities)],
                "probabilities": probability_map,
                "confidence": confidence_value,
                "coverage": coverage,
                "reliability": reliability,
                "decode_steps": decode_steps,
            }
            if legend:
                answer["legend"] = legend
        elif question.type == "score":
            answer = {
                "type": "score",
                "score": readout.score_weighted_mean(probabilities),
                "probabilities": probability_map,
                "confidence": confidence_value,
                "legend": legend,
                "coverage": coverage,
                "reliability": reliability,
                "decode_steps": decode_steps,
            }
        else:
            answer = {
                "type": "noul",
                "noul": probability_map["yes"],
                "probabilities": probability_map,
                "coverage": coverage,
                "reliability": reliability,
                "decode_steps": decode_steps,
            }
        return answer, decode_steps

    # ---- one wave group: fork the suffix once, then score every candidate in it
    def _score_group(self, plan: ContextPlan, suffix_tokens: list[int],
                     candidates: list[list[int]], scored: list[tuple[int, ...]],
                     ) -> dict[str, Any]:
        head_seq = 1
        n_prefix = plan.n_prefix
        n_suffix = len(suffix_tokens)
        self.session.release(head_seq)
        self._fork(0, head_seq, n_prefix)
        rows = self._decode(Batch(
            tokens=tuple(suffix_tokens),
            seq_ids=(head_seq,) * n_suffix,
            positions=tuple(n_prefix + index for index in range(n_suffix)),
            logits=tuple(index == n_suffix - 1 for index in range(n_suffix)),
        ))
        decision_row = rows[-1]
        logprobs = [[readout.logprob(decision_row, tokens[0])] for tokens in scored]
        for offset in range(1, len(candidates)):
            seq = head_seq + offset
            self.session.release(seq)
            self._fork(head_seq, seq, n_prefix + n_suffix)
        max_length = max(len(tokens) for tokens in scored)
        for step in range(1, max_length):
            tokens: list[int] = []
            seqs: list[int] = []
            positions: list[int] = []
            targets: list[int] = []
            for offset, sequence in enumerate(scored):
                if len(sequence) <= step:
                    continue
                tokens.append(sequence[step - 1])
                seqs.append(head_seq + offset)
                positions.append(n_prefix + n_suffix + step - 1)
                targets.append(offset)
            if not tokens:
                continue
            step_rows = self._decode(Batch(tokens=tuple(tokens), seq_ids=tuple(seqs),
                                           positions=tuple(positions),
                                           logits=tuple(True for _ in tokens)))
            for row, offset in zip(step_rows, targets, strict=True):
                logprobs[offset].append(readout.logprob(row, scored[offset][step]))
        for offset in range(len(candidates)):
            self.session.release(head_seq + offset)
        return {"decision_row": decision_row, "logprobs": logprobs}


def _chunks(indices: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(indices), size):
        yield indices[start:start + size]


def planned_waves(requirements: Sequence[tuple[schema.Question, prompt.RenderedQuestion,
                                                list[int], list[list[int]]]],
                  n_seq_max: int, *, readout_mode: str) -> int:
    """How many decode batches (waves) this request will take, given the sequence cap.

    One wave per question group (the suffix decode that produces the decision row) plus one
    wave per candidate step for candidates longer than a single token. `decode_calls` is then
    `1 (prefill) + waves` — the A-E1b-9 contract.
    """
    per_wave = max(1, n_seq_max - 1)
    total = 0
    for _, _, _, candidates in requirements:
        scored = candidate_sequences(candidates, readout_mode=readout_mode)
        for group in _chunks(list(range(len(scored))), per_wave):
            lengths = [len(scored[index]) for index in group]
            total += 1 + max(0, max(lengths) - 1)
    return total


def _check_collisions(question: schema.Question, view: prompt.RenderedQuestion,
                      scored: Sequence[tuple[int, ...]]) -> None:
    seen: dict[tuple[int, ...], str] = {}
    for key, sequence in zip(view.options, scored, strict=True):
        if sequence in seen:
            raise CandidateCollisionError(
                f"E_CANDIDATE_COLLISION: question {question.id!r} candidates "
                f"{seen[sequence]!r} and {key!r} share the same token sequence "
                f"{list(sequence)}; make the labels distinct")
        seen[sequence] = key


def _add_warning(warnings: list[str], code: str) -> None:
    if code not in warnings:
        warnings.append(code)


def decide_request(request: schema.Request, session: Session,
                   **kwargs: Any) -> DecideResult:
    """Convenience wrapper: plan from the request, then decide."""
    plan = plan_context(request, session)
    return DecisionEngine(session, **kwargs).decide(request, plan=plan)
