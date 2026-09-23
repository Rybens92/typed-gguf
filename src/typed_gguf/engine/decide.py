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
  cue         what the row's top token IS (cards t_6c119626, t_635124bf): a turn-closer
              (`<|im_end|>`, `</s>`, …) or any other control/user-defined token the vocabulary
              carries (`</think>`) means the cue shape — not the label rendering — is what
              suppressed the answer -> W_CUE_REFUSED + the answer's `cue` block (`engine/cue.py`)
  no sampling no sampler, no token loop: `decode_calls == 1 + waves`, and no code path here
              can generate text (A-E1b-9)
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from typed_gguf import schema
from typed_gguf.engine import cue as cue_module
from typed_gguf.engine import prompt, readout
from typed_gguf.engine import template as template_module
from typed_gguf.errors import (
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
    kv_type: str = "auto"            # E1c: what the context was created with (fit plan or request)
    n_gpu_layers: int = 0            # E1c: from the fit plan (0 = CPU placement)
    #: E1c FIX (card t_8cb0a05e): WHY the model ended up where it did, and every load attempt.
    placement: Any | None = None
    #: Codes from the placement itself (W_BACKEND_OOM / W_FIT_DOWNGRADE) — merged into `warnings`.
    placement_warnings: tuple[str, ...] = ()
    #: card t_80f1a4c6: where `backend` came from (request | bundle | record | default |
    #: explicit). A label with no source is exactly the silent claim the E3 fix removes.
    backend_source: str = ""


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
    # E1c: the template that produced `prefix_tokens` (None = the plain E1b framing)
    template: template_module.Resolution | None = None
    enable_thinking: bool = False
    #: E3e (card t_4c48f40a): where the question block lives, and — for `role_split` — the render
    #: that produced the prefix and the per-question tails. Carried on the plan (not recomputed)
    #: so the executed context and the executed questions can never be rendered from two
    #: different assemblies (the seam card t_6de5fc53 closed for the template chain).
    chat_format: str = schema.ANSWER_SHEET
    #: E3e: where the `json_instructed` contract is stated (a no-op for the other cues) — part of
    #: the plan for the same reason `chat_format` is: the prompt bytes are a measurement decision
    json_contract: str = schema.JSON_CONTRACT
    role: prompt.RoleSplitRender | None = None

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
    #: E2.5 (A-E2p5-1): whether a stored calibration was applied, and where it came from
    calibrated: bool = False
    calibration: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {"model": self.model, "engine": self.engine, "answers": self.answers,
                "usage": self.usage, "timings": self.timings, "warnings": list(self.warnings),
                "calibrated": bool(self.calibrated), "calibration": dict(self.calibration)}


# ----------------------------------------------------------------------- planning
def question_requirements(request: schema.Request, tokenizer: Tokenizer,
                          *, resolution: template_module.Resolution | None = None,
                          role: prompt.RoleSplitRender | None = None,
                          ) -> list[tuple[schema.Question, prompt.RenderedQuestion,
                                          list[int], list[list[int]]]]:
    """Render + tokenize every question: the one place that decides what a candidate costs.

    `resolution`/`role` are the plan's own template and role-split render (E3e): a request whose
    question block lives in a user turn must be rendered from the *same* assembly the executed
    prefix was built from, or the two halves would describe different conversations.
    """
    options = request.options
    if role is None and options.chat_format == schema.ROLE_SPLIT:
        role = prompt.role_split_render(request.state, request.questions, resolution=resolution,
                                        enable_thinking=options.thinking, cue=options.cue,
                                        contract=options.json_contract)
    rendered: list[tuple[schema.Question, prompt.RenderedQuestion, list[int], list[list[int]]]] = []
    for index, question in enumerate(request.questions):
        view = prompt.build_question(question, readout=options.readout, cue=options.cue,
                                     chat_format=options.chat_format, role=role, index=index,
                                     contract=options.json_contract)
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


def resolve_template(request: schema.Request, tokenizer: Any) -> template_module.Resolution | None:
    """A-E1c-1 for a live handle: the chain over the model's own template.

    Returns `None` when the session carries no model (the deterministic fake sessions of the
    engine tests, and any caller that wants the plain E1b framing): plain is the documented
    fallback of `--template plain`, not a silent default for a real model.
    """
    if getattr(tokenizer, "model", None) is None or getattr(tokenizer, "runtime", None) is None:
        return None
    options = request.options
    user_template = options.template if options.template not in (None, "auto") else None
    return template_module.resolve_for_handle(
        tokenizer, messages=prompt.chat_messages(
            request.state, framing=prompt.framing_for(options.cue, options.json_contract)),
        user_template=user_template,
        explicit_user=user_template is not None,
        think_mode="on" if options.thinking else "auto")


def plan_context(request: schema.Request, tokenizer: Tokenizer, *,
                 template: template_module.Resolution | None = None,
                 resolve: bool = True, n_ctx_cap: int | None = None) -> ContextPlan:
    """Size the context from the request and the fit plan (SPEC 2.2 + SPEC-context-v2 §5.6).

    `n_ctx_cap` is the fit plan's own `n_ctx` (A-E1c-5). v2 (§5.6, D1 = YES) made it the *load
    size* too: with a plan applied and the request pinning nothing, the context really allocated
    is the plan's (standard 32 768, grown when the box holds more, shrunk when it does not) — not
    `prefix + question + margin`. A request that pins `options.n_ctx` keeps today's behaviour
    (`min(pin, plan.n_ctx)`), and without a plan the SPEC 2.2 formula applies literally. The
    request-fit guard is unchanged either way: a request whose needs exceed the loaded context
    fails with `E_CTX_TOO_SMALL` instead of being truncated.
    """
    options = request.options
    resolution = template
    if resolution is None and resolve:
        resolution = resolve_template(request, tokenizer)
    role = prompt.role_split_context(request, resolution=resolution)
    prefix_tokens = tokenizer.tokenize(prompt.build_prefix(
        request.state, resolution=resolution, enable_thinking=options.thinking,
        chat_format=options.chat_format, cue=options.cue, role=role,
        contract=options.json_contract))
    requirements = question_requirements(request, tokenizer, resolution=resolution, role=role)
    per_question = [len(suffix) + max(len(tokens) for tokens in candidates)
                    for _, _, suffix, candidates in requirements]
    max_question = max(per_question, default=0)
    max_candidates = max((len(candidates) for _, _, _, candidates in requirements), default=0)
    meta = getattr(tokenizer, "meta", None)
    meta_threads = getattr(meta, "threads", 0)
    threads = options.threads or meta_threads or _default_threads()
    if options.n_ctx:
        # a pin: the request's own number, never above the plan (A-E1c-5)
        n_ctx = int(options.n_ctx)
        if n_ctx_cap is not None:
            n_ctx = min(n_ctx, int(n_ctx_cap))
    elif n_ctx_cap is not None:
        n_ctx = int(n_ctx_cap)               # v2 §5.6: the plan sizes the load
    else:
        n_ctx = len(prefix_tokens) + max_question + CONTEXT_MARGIN    # --no-fit: SPEC 2.2 literal
    n_seq_max = options.n_seq_max or max(MIN_SEQ_MAX, 1 + max_candidates)
    return ContextPlan(prefix_tokens=tuple(prefix_tokens), n_ctx=int(n_ctx),
                       n_seq_max=int(n_seq_max), threads=int(threads),
                       kv_type=options.kv_type, max_question_tokens=max_question,
                       template=resolution, enable_thinking=options.thinking,
                       chat_format=options.chat_format, json_contract=options.json_contract,
                       role=role)


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
                 context_margin: int = CONTEXT_MARGIN,
                 calibration: Any | None = None) -> None:
        self.session = session
        self.coverage_floor = coverage_floor
        self.confidence_floor = confidence_floor
        self.context_margin = context_margin
        #: E2.5 (SPEC 2.10): a `calibration.calibrate.Table` (duck-typed: `apply()` is the whole
        #: contract) or None. Imported structurally so the engine keeps its zero-dependency
        #: surface — the table arrives from the CLI, already resolved for this model.
        self.calibration = calibration
        #: question type -> the confidence statistic this response actually reported for it
        self.calibrated_types: dict[str, str] = {}
        self.batches: list[Batch] = []      # every decode issued for this engine (the spy)
        self.forks = 0
        #: card t_6c119626: `{token id: closer}` for this session's vocabulary, resolved once
        #: (`engine/cue.py`). Card t_635124bf widened it to every CONTROL / USER_DEFINED token the
        #: vocabulary carries. `None` = not resolved yet.
        self._closers: dict[int, str] | None = None
        #: E3e (card t_4c48f40a): `{token id: marker}` for the JSON punctuation this vocabulary
        #: encodes as one token (`engine/cue.py:value_markers`), resolved once per session.
        self._value_marker_map: dict[int, str] | None = None

    # ---- decode seam (everything the engine decodes goes through here)
    def _decode(self, batch: Batch) -> list[list[float]]:
        self.batches.append(batch)
        return self.session.decode(batch)

    def _fork(self, src: int, dst: int, upto: int) -> None:
        self.forks += 1
        self.session.fork(src, dst, upto)

    def _closer_map(self) -> dict[int, str]:
        """`{token id: name}` for this session's vocabulary (`engine/cue.py`).

        The mapping depends only on the model's tokenizer and its vocabulary's own token
        attributes, neither of which can change while the engine holds the session, so it is
        resolved once. The vocabulary decides which catalogue entries count: a closer this
        tokenizer splits into several tokens is not one token and is never reported as a refusal
        — and since card t_635124bf it also contributes the tokens it marks CONTROL or
        USER_DEFINED (`ModelHandle.special_tokens`), the class no string catalogue can name.
        """
        if self._closers is None:
            self._closers = cue_module.closer_map(self.session)
        return self._closers

    def decide(self, request: schema.Request, *, plan: ContextPlan | None = None,
               model_alias: str | None = None) -> DecideResult:
        started = time.perf_counter()
        session = self.session
        meta = session.meta
        # card t_80f1a4c6: the device evidence the session's own log carries, read once here and
        # published next to the label — `meta.backend` is a *claim*, this is the measurement.
        evidence = device_evidence(session, meta.backend)
        plan = plan or plan_context(request, session)
        requirements = question_requirements(request, session, resolution=plan.template,
                                             role=plan.role)
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
        self.calibrated_types.clear()
        if plan.template is not None:
            for code in plan.template.warnings:
                _add_warning(warnings, code)
        for code in meta.placement_warnings:
            # E1c FIX: a degraded placement (allocation failure survived by reducing the plan) is
            # part of the answer's provenance, not a detail of the load.
            _add_warning(warnings, code)
        for code in evidence["warnings"]:
            # E3 FIX (card t_80f1a4c6): a backend label the engine's own log refutes is named,
            # never published silently (the serving path's `engine.backend` read `cpu` while the
            # process' stderr showed `Vulkan0 compute buffer size`).
            _add_warning(warnings, code)
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
                # E3 FIX (card t_80f1a4c6): the label above is a claim — say where it came from
                # and what the engine's own log proves (`effective_backend` is `null` when the
                # log carries no compute-buffer line: unverified, never claimed).
                "backend_source": meta.backend_source,
                "devices": evidence["devices"],
                "device_buffers": evidence["device_buffers"],
                "effective_backend": evidence["effective_backend"],
                "readout": options.readout,
                "cue": options.cue,
                # E3e (card t_4c48f40a): where the question block lives — a knob that does not
                # appear in the response is a silent knob (`engine.cue`, E3d).
                "chat_format": self._chat_format_surface(plan, options),
                "kv_unified": bool(meta.kv_unified),
                "n_ctx": meta.n_ctx,
                "n_seq_max": meta.n_seq_max,
                "prefix_tokens": len(plan.prefix_tokens),
                "state_id": state_id,
                "prefill_reused": bool(prefill.prefill_reused),
                "template": self._template_surface(plan),
                "kv_type": meta.kv_type,
                "n_gpu_layers": meta.n_gpu_layers,
                # E1c FIX (card t_8cb0a05e): say WHERE the model ran and why — `n_gpu_layers: 0`
                # on its own could mean "--no-fit", a busy desktop or a degraded retry.
                "placement": (meta.placement.to_dict()
                              if hasattr(meta.placement, "to_dict") else meta.placement),
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
            calibrated=bool(self.calibrated_types),
            calibration=self._calibration_surface(),
        )

    def _calibration_surface(self) -> dict[str, Any]:
        """The response's `calibrated` / `calibration` block (A-E2p5-1), never guessed."""
        if self.calibration is None:
            return {"source": "", "applied": False, "model": None, "params_hash": "",
                    "temperatures": {}, "confidence_modes": {}, "accepted_types": []}
        return self.calibration.response_fields(self.calibrated_types)

    # ---- guards
    def _confidence_mode(self, qtype: str, options: schema.Options) -> str:
        """The statistic the answer's `confidence` is computed with (E2.5 / A-E2p5-3).

        An explicit `options.confidence_mode` always wins; otherwise a stored calibration that
        promoted a mode for this question type supplies it; otherwise the documented default.
        """
        if options.confidence_mode:
            return options.confidence_mode
        if self.calibration is not None:
            promoted = self.calibration.mode_for(qtype)
            if promoted:
                return promoted
        return readout.DEFAULT_CONFIDENCE_MODE

    def _calibrate(self, probabilities: list[float], qtype: str, mode: str
                   ) -> tuple[list[float], float]:
        """E2.5 (SPEC 2.10): scale one question's probabilities, never its ranking.

        The table is duck-typed: `apply()` returns the (possibly) scaled vector, the confidence
        computed from it with the *requested* mode, and whether a parameter was applied at all.
        A table that was never accepted — or a question type it has no accepted parameter for —
        changes nothing, so `calibrated` in the response is only ever true when a stored
        parameter really rescaled these numbers.
        """
        if self.calibration is None:
            return probabilities, readout.confidence(probabilities, mode)
        applied = self.calibration.apply(probabilities, qtype, mode=mode)
        if not applied["calibrated"]:
            return probabilities, readout.confidence(probabilities, mode)
        self.calibrated_types[qtype] = str(applied.get("mode") or mode)
        return list(applied["probabilities"]), float(applied["confidence"])

    @staticmethod
    def _template_surface(plan: ContextPlan) -> dict[str, Any]:
        """`engine.template` — which template produced the prompt (A-E1c-1/2, evidence-ready)."""
        if plan.template is None:
            return {"kind": "plain", "renderer": "plain", "source": "prompt.py framing",
                    "family": None, "thinking": "n/a"}
        return plan.template.to_dict()

    @staticmethod
    def _chat_format_surface(plan: ContextPlan, options: schema.Options) -> dict[str, Any]:
        """`engine.chat_format` — where the question block lives, with the render's own receipts.

        E3e (card t_4c48f40a): `question_turn` names the role the question block is prefilled into
        (`assistant` = the answer-sheet shape, `user` = the role split). A role split publishes the
        bytes the state-only render carries that the shared prefix cannot (`dropped` — Spark's
        template ends a render with a newline, which cannot sit in the middle of one), so the
        acceptance the tool measured per family is visible on every response. `contract` names where
        the `json_instructed` contract was stated (the amendment's two variants); it is a no-op for
        the other cues and says so.
        """
        role = plan.role
        surface: dict[str, Any] = {
            "kind": options.chat_format,
            "question_turn": "user" if role is not None else "assistant",
            "contract": options.json_contract if options.cue == "json_instructed" else None,
        }
        if role is not None:
            surface["prefix_chars"] = len(role.prefix)
            surface["dropped"] = role.dropped
        return surface

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
                f"this host is loaded with n_ctx={meta.n_ctx}; pass --n-ctx {needed} to reload "
                f"bigger")
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
        coverage = 0.0
        # card t_6c119626: the verdict on the row the coverage is read from. Every wave re-decodes
        # the same suffix from the same prefix, so the row — and its verdict — is wave-independent,
        # and the answer publishes the last wave's.
        cue_verdict: dict[str, Any] = {}
        cue_row_verdict: dict[str, Any] = {}
        advance_record: dict[str, Any] | None = None
        for group in _chunks(list(range(len(candidates))), per_wave):
            rows = self._score_group(plan, suffix_tokens,
                                     [candidates[index] for index in group],
                                     [scored[index] for index in group],
                                     options.cue, floor=coverage_floor)
            for offset, index in enumerate(group):
                logprobs[index] = rows["logprobs"][offset]
            coverage += readout.coverage_from_scale(rows["decision_row"], rows["coverage_ids"],
                                                    rows["decision_scale"])
            # E3e: `json_instructed` reads the label *at the opened field*, so its decision row is
            # a value row and its verdict is the value block (`answered`/`refused`/`empty_value`/
            # `wrong_field`). Every other shape keeps E3d's row verdict, byte-identical.
            cue_verdict = rows["value"] or cue_module.cue_verdict(
                rows["decision_row"], rows["decision_scale"], self._closer_map(),
                floor=coverage_floor)
            # E3d: the row the *refusal* lives on is the cue row — for `shipped` that is the
            # decision row, for `two_step` it is the row the readout moved past.
            cue_row_verdict = cue_module.cue_verdict(rows["cue_row"],
                                                     readout.logsumexp(rows["cue_row"]),
                                                     self._closer_map(), floor=coverage_floor)
            if rows["advance"] is not None:
                advance_record = {"token": int(rows["advance"]), "rule": cue_module.ADVANCE_RULE,
                                  "cue": cue_row_verdict}

        z = [readout.candidate_sequence_score(values, options.length_norm)
             for values in logprobs]
        probabilities = readout.restricted_softmax(z, options.temperature)
        coverage = min(1.0, coverage)
        mode = self._confidence_mode(question.type, options)
        probabilities, confidence_value = self._calibrate(probabilities, question.type, mode)
        reliability = readout.reliability(coverage, coverage_floor=coverage_floor,
                                          confidence_floor=self.confidence_floor,
                                          confidence_value=confidence_value)
        if reliability == "low_mass":
            _add_warning(warnings, "W_LOW_MASS")
        elif reliability == "low_confidence":
            _add_warning(warnings, "W_LOW_CONFIDENCE")
        if cue_row_verdict.get("refused"):
            # why a `low_mass` row is low: the model closes the assistant turn at the cue instead
            # of answering. Named on its own, because the fix is the prompt shape, not the labels.
            # E3d: the verdict is read on the *cue* row, so a `two_step` request that never
            # advanced (a refusal) is reported exactly like the shipped shape it fell back to.
            _add_warning(warnings, "W_CUE_REFUSED")
        # E3e (card t_4c48f40a): the two `json_instructed` verdicts that are neither a label nor a
        # refusal. `low_mass` is true for both — the row really is low on label mass — but it is
        # also true of a model that simply did not choose a candidate, and the reader's fix is not
        # the same, so each gets its own named code next to it.
        if cue_verdict.get("verdict") == "empty_value":
            _add_warning(warnings, "W_JSON_EMPTY_VALUE")
        elif cue_verdict.get("verdict") == "wrong_field":
            _add_warning(warnings, "W_JSON_WRONG_FIELD")

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
                "cue": cue_verdict,
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
                "cue": cue_verdict,
                "decode_steps": decode_steps,
            }
        else:
            answer = {
                "type": "noul",
                "noul": probability_map["yes"],
                "probabilities": probability_map,
                "coverage": coverage,
                "reliability": reliability,
                "cue": cue_verdict,
                "decode_steps": decode_steps,
            }
        if advance_record is not None:
            # E3d: only a shape that moved the readout carries this — a `shipped` answer's key set
            # is byte-frozen (every published row was measured on it).
            answer["advance"] = advance_record
        return answer, decode_steps

    # ---- one wave group: fork the suffix once, then score every candidate in it
    def _score_group(self, plan: ContextPlan, suffix_tokens: list[int],
                     candidates: list[list[int]], scored: list[tuple[int, ...]],
                     cue: str = schema.CUE_SHAPES[0], *,
                     floor: float = cue_module.REFUSAL_FLOOR,
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
        cue_row = rows[-1]
        # E3d (card t_d90404ac): `two_step` decodes one more token on the head sequence — the
        # model's own first *content* token — and reads the label at the row after it. A cue the
        # model closes never advances (`_advance_token`), so the refusal verdict E3c publishes is
        # read from a row that is byte-identical to the shipped shape's.
        advance = self._advance_token(cue_row, cue)
        if advance is None:
            decision_row = cue_row
        else:
            decision_row = self._decode(Batch(tokens=(advance,), seq_ids=(head_seq,),
                                              positions=(n_prefix + n_suffix,),
                                              logits=(True,)))[-1]
        base = n_prefix + n_suffix + (0 if advance is None else 1)
        decision_scale = readout.logsumexp(decision_row)
        logprobs = [[readout.logprob_from_scale(decision_row, tokens[0], decision_scale)]
                    for tokens in scored]
        coverage_ids = [tokens[0] for tokens in scored]
        for offset in range(1, len(candidates)):
            seq = head_seq + offset
            self.session.release(seq)
            self._fork(head_seq, seq, base)
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
                positions.append(base + step - 1)
                targets.append(offset)
            if not tokens:
                continue
            step_rows = self._decode(Batch(tokens=tuple(tokens), seq_ids=tuple(seqs),
                                           positions=tuple(positions),
                                           logits=tuple(True for _ in tokens)))
            for row, offset in zip(step_rows, targets, strict=True):
                scale = readout.logsumexp(row)          # one full pass per row, not per token
                logprobs[offset].append(
                    readout.logprob_from_scale(row, scored[offset][step], scale))
        for offset in range(len(candidates)):
            self.session.release(head_seq + offset)
        return {"decision_row": decision_row, "decision_scale": decision_scale,
                "coverage_ids": coverage_ids, "logprobs": logprobs,
                "cue_row": cue_row, "advance": advance,
                "value": self._value_row_verdict(cue, head_seq, base, decision_row,
                                                 decision_scale, floor)}

    def _value_row_verdict(self, cue: str, head_seq: int, base: int,
                           decision_row: Sequence[float], decision_scale: float,
                           floor: float) -> dict[str, Any] | None:
        """`json_instructed` only: classify the opened field's value row (E3e, card t_4c48f40a).

        Runs *after* the candidate sequences forked off the head at `base`, so the one token it
        decodes when the model closes the value cannot leak into what the candidates score: the
        head is read one step further, exactly as the model itself would continue. One `_decode`
        call at most, and only when the value row's argmax is the vocabulary's single-token `"`
        (`value_markers`) — otherwise the row is a plain answer and nothing extra is decoded.
        """
        if cue != "json_instructed":
            return None
        markers = self._value_markers()
        winner = int(readout.argmax_first(decision_row))
        next_row = None
        next_scale = None
        if markers.get(winner) == "quote":
            next_row = self._decode(Batch(tokens=(winner,), seq_ids=(head_seq,),
                                          positions=(base,), logits=(True,)))[-1]
            next_scale = readout.logsumexp(next_row)
        return cue_module.value_verdict(decision_row, decision_scale, self._closer_map(), markers,
                                        next_row=next_row, next_scale=next_scale, floor=floor)

    def _value_markers(self) -> dict[int, str]:
        """The session's own single-token JSON punctuation — a vocabulary fact, not a guess."""
        if self._value_marker_map is None:
            self._value_marker_map = cue_module.value_markers(self.session.tokenize)
        return self._value_marker_map

    def _advance_token(self, row: Sequence[float], cue: str) -> int | None:
        """The one token `two_step` decodes before it reads — `None` when the cue refuses.

        The rule is `cue.ADVANCE_RULE` ("content"): the row's argmax among tokens that are **not**
        turn-closers ("if the model cannot close the turn, what does it start to say?"). The
        engine's form is conditional on purpose: a row whose *argmax* is a turn-closer is a
        refusal, and a refusal never advances — the label is read where the shipped shape reads
        it, so `W_CUE_REFUSED` and the `low_mass` it explains stay exactly as published for the
        families that refuse the cue (card t_d90404ac, constraint 4). For a row whose argmax is
        content the rule and the argmax coincide, which is what the 60-item 4B run measured.
        """
        if cue != "two_step":
            return None
        winner = readout.argmax_first(row)
        if winner in self._closer_map():
            return None
        return int(winner)


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


def _device_module() -> Any:
    """The device-evidence parser, imported in-function so the engine path stays light."""
    from typed_gguf.runtime import devices as devices_module
    return devices_module


def device_evidence(session: Any, claimed: str) -> dict[str, Any]:
    """What one serving session's own log proves about the device that computed.

    `devices` is every device name the engine touched, `device_buffers` its **compute** buffers
    per device, `effective_backend` the compute path read from them (`None` when the log carries
    no compute-buffer line — unverified, never claimed), and `warnings` holds
    `W_BACKEND_MISMATCH` when `claimed` is refuted by that evidence, or cannot be corroborated
    by it (an accelerator claim with no device line at all). This mirrors `harness.device_usage`,
    which fixed the same class of lie for the bench tables (card t_603a35a0); here it is the
    serving path — `run`/`ask` — whose `engine.backend` is the field a reader trusts (card
    t_80f1a4c6).
    """
    text = getattr(session, "device_log", "") or ""
    usage = _device_module().parse_device_usage(
        text if isinstance(text, str) else "\n".join(text))
    warning = "W_BACKEND_MISMATCH" if _device_module().contradicts(claimed, usage) else None
    return {"devices": list(usage.devices),
            "device_buffers": dict(usage.compute_buffers),
            "effective_backend": usage.effective,
            "warnings": [warning] if warning else []}


def decide_request(request: schema.Request, session: Session,
                   **kwargs: Any) -> DecideResult:
    """Convenience wrapper: plan from the request, then decide."""
    plan = plan_context(request, session)
    return DecisionEngine(session, **kwargs).decide(request, plan=plan)
