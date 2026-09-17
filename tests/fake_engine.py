"""A deterministic, model-free session used by the engine tests (and the CLI tests).

It implements exactly the seam `ggufone.engine.decide` talks to:

    meta, tokenize, prefill, fork, release, decode, close

`decode()` returns full-vocab logit rows produced by `row_fn(RowContext)` — tests shape the
model's opinion by biasing token ids, without a runtime, a GGUF file or a GPU.

The tokenizer is word level (`re.findall(r"[a-z0-9]+", text.lower())`), which is enough to
exercise prompt assembly, candidate rendering, collisions and multi-token scoring: two labels
that differ only in punctuation/case collapse onto the same candidate sequence.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ggufone.engine.decide import Batch, PrefillInfo, SessionMeta

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
    model_alias: str | None = "fake-model"
    model_path: str = "/fake/model.gguf"
    load_ms: float = 12.5
    prefill_ms: float = 0.5
    step_ms: float = 0.05
    row_fn: RowFn | None = None
    # bookkeeping the tests assert on
    batches: list[Batch] = field(default_factory=list)
    forks: list[tuple[int, int, int]] = field(default_factory=list)
    released: list[int] = field(default_factory=list)
    prefill_calls: list[dict[str, Any]] = field(default_factory=list)
    decoded: list[tuple[int, int]] = field(default_factory=list)   # (seq_id, token)
    loaded_states: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._words: dict[str, int] = {}
        self._seq_tokens: dict[int, list[int]] = {}
        self._loaded: set[str] = set()

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
        return SessionMeta(runtime=self.runtime, backend=self.backend, n_ctx=self.n_ctx,
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
        return PrefillInfo(prefill_tokens=0 if reused else len(tokens),
                           prefill_ms=0.0 if reused else self.prefill_ms,
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
