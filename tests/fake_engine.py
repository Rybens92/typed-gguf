"""A deterministic, model-free session used by the engine tests (and the CLI tests).

It implements exactly the seam `typed_gguf.engine.decide` talks to:

    meta, tokenize, prefill, fork, release, decode, close

`decode()` returns full-vocab logit rows produced by `row_fn(RowContext)` — tests shape the
model's opinion by biasing token ids, without a runtime, a GGUF file or a GPU.

The tokenizer is word level (`re.findall(r"[a-z0-9]+", text.lower())`), which is enough to
exercise prompt assembly, candidate rendering, collisions and multi-token scoring: two labels
that differ only in punctuation/case collapse onto the same candidate sequence.
"""
from __future__ import annotations

import contextlib
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from typed_gguf.engine.decide import Batch, PrefillInfo, SessionMeta

WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class RowContext:
    """What the fake knows when it is asked for a row."""

    seq_id: int
    position: int
    token: int
    tokens_on_seq: tuple[int, ...]      # tokens decoded on this seq so far, this one last
    batch: Batch
    session: FakeSession


RowFn = Callable[[RowContext], list[float]]


def biased_row(n_vocab: int, bias: dict[int, float]) -> list[float]:
    row = [0.0] * n_vocab
    for token_id, value in bias.items():
        row[token_id] = value
    return row


@dataclass
class FakeSession:
    n_ctx: int = 4096
    n_seq_max: int = 5
    n_vocab: int = 256
    threads: int = 1
    runtime: str = "llama.cpp b11026 (fake)"
    backend: str = "cpu"
    #: the engine's own log lines (card t_80f1a4c6): a live session fills this from its load +
    #: context captures, a test scripts the operator's own lines verbatim.
    device_log: str = ""
    #: where `backend` came from (request | bundle | record | default | explicit)
    backend_source: str = ""
    model_alias: str | None = "fake-model"
    model_path: str = "/fake/model.gguf"
    load_ms: float = 12.5
    prefill_ms: float = 0.5
    step_ms: float = 0.05
    row_fn: RowFn | None = None
    # bench seam (E2): a shared vocabulary + a shared state cache make several sessions look
    # like one model with a file-backed prefix cache, and a per-token prefill cost makes the
    # latency numbers proportional to the work the suite asked for.
    words: dict[str, int] | None = None
    loaded_state_ids: set[str] | None = None
    prefill_ms_per_token: float = 0.0
    # bookkeeping the tests assert on
    batches: list[Batch] = field(default_factory=list)
    forks: list[tuple[int, int, int]] = field(default_factory=list)
    released: list[int] = field(default_factory=list)
    prefill_calls: list[dict[str, Any]] = field(default_factory=list)
    decoded: list[tuple[int, int]] = field(default_factory=list)   # (seq_id, token)
    loaded_states: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._words: dict[str, int] = self.words if self.words is not None else {}
        self._seq_tokens: dict[int, list[int]] = {}
        self._loaded: set[str] = (self.loaded_state_ids if self.loaded_state_ids is not None
                                  else set())

    # ---- tokenizer
    def token_id(self, word: str) -> int:
        key = word.lower()
        if key not in self._words:
            if len(self._words) + 1 >= self.n_vocab:
                raise AssertionError("fake vocabulary exhausted; raise FakeSession.n_vocab")
            self._words[key] = len(self._words) + 1
        return self._words[key]

    def tokenize(self, text: str) -> list[int]:
        return [self.token_id(word) for word in WORD_RE.findall(text)]

    # ---- session surface
    @property
    def meta(self) -> SessionMeta:
        return SessionMeta(runtime=self.runtime, backend=self.backend,
                           backend_source=self.backend_source, n_ctx=self.n_ctx,
                           n_seq_max=self.n_seq_max, kv_unified=True, threads=self.threads,
                           n_vocab=self.n_vocab, model_path=self.model_path,
                           model_alias=self.model_alias, load_ms=self.load_ms)

    def prefill(self, tokens, *, state_id=None, state_cache=True, save_state=False) -> PrefillInfo:
        self.prefill_calls.append({"tokens": list(tokens), "state_id": state_id,
                                   "state_cache": state_cache, "save_state": save_state})
        reused = bool(state_cache and state_id and state_id in self._loaded)
        if not reused:
            self._seq_tokens[0] = list(tokens)
            self.decoded.extend((0, token) for token in tokens)
            if save_state and state_id:
                self._loaded.add(state_id)
        else:
            self.loaded_states.append(state_id)
            self._seq_tokens.setdefault(0, list(tokens))
        cost = self.prefill_ms if self.prefill_ms_per_token == 0.0 \
            else len(tokens) * self.prefill_ms_per_token
        return PrefillInfo(prefill_tokens=0 if reused else len(tokens),
                           prefill_ms=0.0 if reused else cost,
                           prefill_reused=reused, state_id=state_id,
                           state_path=f"/states/{state_id}.bin" if state_id else None)

    def fork(self, src: int, dst: int, upto: int) -> None:
        self.forks.append((src, dst, upto))
        self._seq_tokens[dst] = list(self._seq_tokens.get(src, []))[:upto]

    def release(self, seq: int) -> None:
        self.released.append(seq)
        self._seq_tokens.pop(seq, None)

    def decode(self, batch: Batch) -> list[list[float]]:
        self.batches.append(batch)
        tokens = self._seq_tokens.setdefault(batch.seq_ids[0] if batch.seq_ids else 0, [])
        rows: list[list[float]] = []
        for index in batch.logits_indices():
            seq_id = batch.seq_ids[index]
            tokens_on_seq = tuple(self._seq_tokens.get(seq_id, []))
            context = RowContext(seq_id=seq_id, position=batch.positions[index],
                                 token=batch.tokens[index], tokens_on_seq=tokens_on_seq,
                                 batch=batch, session=self)
            rows.append(self._row(context))
        for token, seq_id in zip(batch.tokens, batch.seq_ids, strict=True):
            self._seq_tokens.setdefault(seq_id, []).append(token)
            self.decoded.append((seq_id, token))
        del tokens
        return rows

    def _row(self, context: RowContext) -> list[float]:
        if self.row_fn is None:
            return [0.0] * self.n_vocab
        return list(self.row_fn(context))

    def close(self) -> None:  # pragma: no cover - nothing to release in the fake
        pass


class BenchModel:
    """The bench seam (`bench.harness.ModelLike`) without a runtime, a GGUF or a GPU.

    The suites only ever see this surface::

        spec, load_ms, load(), tokenize(), session(n_ctx=…, n_seq_max=…, threads=…),
        decide(request), close()

    `script` maps a request's state text to the candidate *label* this model "prefers"; the row
    at the decision position is biased (+30) on that label's first token, so the real
    `DecisionEngine` readout picks it. Everything else — waves, forks, decode steps, coverage,
    warnings, timings — is produced by production code over the word-level `FakeSession`, which
    is what makes the suite tests meaningful without a model.
    """

    def __init__(self, spec: object, *, script: dict[str, str] | None = None, n_vocab: int = 8192,
                 load_ms: float = 12.5, prefill_ms: float = 0.5, prefill_ms_per_token: float = 0.0,
                 threads: int = 1, nondeterministic: bool = False, step: float = 30.0,
                 device_log: str = "", template: dict[str, object] | None = None,
                 prefix_tokens: int | None = None) -> None:
        self.spec = spec
        self.script = dict(script or {})
        self.n_vocab = n_vocab
        self.load_ms = float(load_ms)
        self.prefill_ms = float(prefill_ms)
        self.prefill_ms_per_token = float(prefill_ms_per_token)
        self.threads = int(threads)
        self.nondeterministic = bool(nondeterministic)
        self.step = float(step)
        #: the engine's own log lines (card t_603a35a0): the live seam fills this from
        #: `llama_log_set` captures, a test scripts the operator's own lines verbatim.
        self.device_log = str(device_log)
        #: card t_6de5fc53: the template surface / prefix token count this fake's *responses*
        #: report. A `FakeSession` carries no model handle, so the plans a fake builds are plain
        #: by construction; these two let a test declare what the response says it measured (the
        #: row/report plumbing reads the response, and that is what the offline gate pins).
        self.template = dict(template) if template is not None else None
        self.prefix_tokens = prefix_tokens
        self.loads = 0
        self.sessions = 0
        self._words: dict[str, int] = {}
        self._states: set[str] = set()
        self._tick = 0

    # ---- the seam
    def load(self) -> float:
        self.loads += 1
        return self.load_ms

    def tokenize(self, text: str) -> list[int]:
        session = self._session(n_ctx=1, n_seq_max=1, threads=self.threads)
        return session.tokenize(text)

    @contextlib.contextmanager
    def session(self, *, n_ctx: int, n_seq_max: int, threads: int | None = None,
                row_fn: object = None) -> Iterator[FakeSession]:
        self.sessions += 1
        yield self._session(n_ctx=n_ctx, n_seq_max=n_seq_max,
                            threads=self.threads if threads is None else threads, row_fn=row_fn)

    def decide(self, request: object, *, n_ctx: int | None = None,
               n_seq_max: int | None = None, threads: int | None = None) -> object:
        from typed_gguf.engine import decide as decide_module

        label = self.script.get(_state_text(request.state))
        session = self._session(n_ctx=n_ctx or 65536,
                                n_seq_max=n_seq_max or max(3, 1 + _max_candidates(request)),
                                threads=self.threads if threads is None else threads)
        if label:
            token = session.tokenize(label)[0]
            boost = self.step
            if self.nondeterministic:
                self._tick += 1
                boost += float(self._tick)
            row = biased_row(self.n_vocab, {token: boost})

            def _biased(_context: object, row: list[float] = row) -> list[float]:
                return list(row)

            session.row_fn = _biased
        plan = decide_module.plan_context(request, session, resolve=False)
        result = decide_module.DecisionEngine(session).decide(
            request, plan=plan, model_alias=f"bench-{getattr(self.spec, 'backend', 'cpu')}")
        if self.template is not None:                      # card t_6de5fc53: scripted response
            result.engine["template"] = dict(self.template)
        if self.prefix_tokens is not None:
            result.engine["prefix_tokens"] = int(self.prefix_tokens)
        return result

    def close(self) -> None:
        pass

    # ---- internals
    def _session(self, *, n_ctx: int, n_seq_max: int, threads: int,
                 row_fn: object = None) -> FakeSession:
        return FakeSession(n_ctx=int(n_ctx), n_seq_max=int(n_seq_max), threads=int(threads),
                           n_vocab=self.n_vocab, model_path=getattr(self.spec, "path", "fake.gguf"),
                           model_alias=f"bench-{getattr(self.spec, 'backend', 'cpu')}",
                           load_ms=self.load_ms, prefill_ms=self.prefill_ms,
                           prefill_ms_per_token=self.prefill_ms_per_token,
                           words=self._words, loaded_state_ids=self._states,
                           row_fn=row_fn if row_fn is not None else None)


def make_bench_model(spec: object, *, script: dict[str, str] | None = None, **kwargs: object):
    """`factory(spec)` for the tests (a live run passes `bench.suites.live_factory`)."""
    return BenchModel(spec, script=script, **kwargs)


def _state_text(state: object) -> str:
    return state if isinstance(state, str) else repr(state)


def _max_candidates(request: object) -> int:
    return max((len(question.options) for question in request.questions), default=1)
