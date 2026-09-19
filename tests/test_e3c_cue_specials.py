"""E3c FIX (card `t_635124bf`): the *second* silent refusal — a special token the catalogue
cannot name.

The E3c verdict (`engine/cue.py`, card `t_6c119626`) matches a **catalogue** of documented
turn-closer strings through the session's own tokenizer. That is enough for `<|im_end|>` and `</s>`
and it is not enough for Tiel-Coder: on the serving-shaped batch the cue row's argmax is the
*user-defined special* `</think>` (token 248069) on 19/20 items. The string is not in the
catalogue, so the row published `refused: false` and the reader got `low_mass` with no reason —
the same silence E3c removed for Occamy, one vocabulary class over.

The generalised rule this file pins:

* the classifier is the **vocabulary's own token attributes** (`llama_token_get_attr`): a token the
  model marks `CONTROL` or `USER_DEFINED` is a turn-shaping/special token, never content. The
  catalogue stays as the human-readable name for the families ggufone documents; the attribute
  table is the fallback that names a special token the catalogue never heard of;
* the token must **dominate** the row: it is the argmax *and* holds at least the engine's own
  coverage floor (`OPTION_DEFAULTS["coverage_floor"]`, 0.10) of the row's mass — the same
  threshold `reliability` uses. A control token that wins a flat row by a hair is not a refusal;
* the payload keeps the E3c block shape `{refused, token, closer, mass[, hint]}` — `closer` is the
  vocabulary's own text for the token (`</think>`), the catalogue string when the catalogue names
  it, and `<special <id>>` when the vocabulary carries no text for it.

Every gate here fails on the pre-fix tree: `single_token_closers` has no `special` argument and
`cue_verdict` has no dominance floor, so `</think>` reads `refused: false`, `closer: null`.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import pathlib
import sys
from typing import Any

import pytest

from ggufone.engine import cue as cue_module
from ggufone.engine import decide
from ggufone.runtime import ctypes_binding
from ggufone.schema import OPTION_DEFAULTS, parse_request
from tests.fake_engine import BenchModel, FakeSession, biased_row

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: the bias that puts ~0.99 of one token's mass in a 512-slot vocabulary
TOP = 11.0
#: the bias whose mass lands *below* the 0.10 floor: e^0.5 / (e^0.5 + 511) ≈ 0.0032
BELOW_FLOOR = 0.5
#: the fake slot for `</think>`; Tiel's own id is 248069 (measured,
#: `docs/evidence/t635124bf_cue_specials.md`)
THINK_END = 413
#: what the fake vocabulary's tokenizer encodes as one special token
THINK_END_TEXT = "</think>"
#: a special token the *catalogue* has never heard of and whose string the fake tokenizer splits
UNKNOWN_SPECIAL = 414
UNKNOWN_SPECIAL_TEXT = "<|turn_off|>"
#: a special token the vocabulary carries without text (llama.cpp can report an empty piece)
UNNAMED_SPECIAL = 415
#: the id the fake vocabulary reserves for the catalogue closer `<|im_end|>`
IM_END = 400
#: the id of the plain word the "content token" gate biases (`words={"unrelated": 7}`)
WORD = 7


@dataclasses.dataclass
class SpecialSession(FakeSession):
    """`FakeSession` with a vocabulary that carries control/special tokens.

    A live GGUF records each token's type (`tokenizer.ggml.token_type`): `CONTROL` for
    `<|im_end|>`/`<|endoftext|>`, `USER_DEFINED` for the chat-template specials (`</think>`,
    `<tool_call>`). `ModelHandle.special_tokens()` reads that table through llama.cpp's own
    attribute API and the session hands it to `engine/cue.py`; this fake hands over the same shape
    — `[(token id, the vocabulary's own text)]`.

    `catalogue` is what this vocabulary encodes as **one** token, the way llama.cpp's
    `parse_special=True` encodes `<|im_end|>`. The specials are a separate class: a token can be
    special without the catalogue naming its string, which is exactly the Tiel fixture.
    """

    specials: dict[int, str] = dataclasses.field(default_factory=dict)
    catalogue: dict[str, int] = dataclasses.field(default_factory=dict)

    def special_tokens(self) -> list[tuple[int, str]]:
        return sorted(self.specials.items())

    def tokenize(self, text: str) -> list[int]:
        if text in self.catalogue:
            return [self.catalogue[text]]
        return super().tokenize(text)


def session_with(specials: dict[int, str] | None = None,
                 catalogue: dict[str, int] | None = None, **kwargs: Any) -> SpecialSession:
    """A 512-slot fake vocabulary with the given specials and single-token catalogue strings."""
    return SpecialSession(n_vocab=512, specials=dict(specials or {}),
                          catalogue=dict(catalogue or {}), **kwargs)


def choice_request() -> dict:
    return {
        "state": "The billing dashboard is blank for every user after login.",
        "questions": {
            "area": {"type": "choice", "instructions": "Which area owns this?",
                     "criteria": {"billing": "payments and invoices",
                                  "technical": "api and infrastructure"}},
        },
    }


def run(session: FakeSession, payload: dict | None = None):
    request = parse_request(payload if payload is not None else choice_request())
    plan = decide.plan_context(request, session)
    return decide.DecisionEngine(session).decide(request, plan=plan)


def run_row(session: FakeSession, bias: dict[int, float], payload: dict | None = None):
    """The engine's answer for a session whose every row carries `bias`."""
    return run_result(session, bias, payload).answers["area"]


def run_result(session: FakeSession, bias: dict[int, float], payload: dict | None = None):
    """The whole `DecideResult` for a session whose every row carries `bias`."""
    session.row_fn = lambda ctx: biased_row(session.n_vocab, bias)
    return run(session, payload)


# --------------------------------------------------------------------- the detector (fixtures a-c)
def test_a_dominating_special_token_outside_the_catalogue_is_a_refusal():
    """Fixture (b) — the Tiel shape: `</think>` dominates, the catalogue cannot name it."""
    assert THINK_END_TEXT not in cue_module.TURN_CLOSERS          # the premise of the fixture
    session = session_with({THINK_END: THINK_END_TEXT})
    assert cue_module.single_token_closers(session.tokenize).get(THINK_END) is None
    result = run_result(session, {THINK_END: TOP})
    answer = result.answers["area"]
    assert "W_CUE_REFUSED" in result.warnings
    assert answer["cue"]["refused"] is True
    assert answer["cue"]["closer"] == THINK_END_TEXT
    assert answer["cue"]["token"] == THINK_END
    assert answer["cue"]["mass"] == pytest.approx(0.99, abs=0.02)
    assert answer["cue"]["hint"].startswith("docs/TEMPLATES.md")


def test_the_dominating_catalogue_closer_still_fires_exactly_as_before():
    """Fixture (a) — the E3c gate, re-pinned here because the rule is now shared."""
    session = session_with({}, {"<|im_end|>": IM_END})
    answer = run_row(session, {IM_END: TOP})
    assert answer["cue"]["refused"] is True
    assert answer["cue"]["closer"] == "<|im_end|>"
    assert answer["cue"]["token"] == IM_END
    assert answer["cue"]["mass"] == pytest.approx(0.99, abs=0.02)


def test_a_content_token_dominating_the_cue_is_not_a_refusal():
    """Fixture (c) — a plain word at 0.99 must never be read as a turn-shape problem."""
    session = session_with({THINK_END: THINK_END_TEXT}, words={"unrelated": WORD})
    answer = run_row(session, {WORD: TOP})
    assert answer["cue"]["refused"] is False
    assert answer["cue"]["closer"] is None
    assert answer["cue"]["token"] == WORD
    assert answer["cue"]["mass"] == pytest.approx(0.99, abs=0.02)
    assert "hint" not in answer["cue"]


def test_an_unknown_special_token_is_named_by_the_vocabulary_not_by_its_id():
    """The fallback: the catalogue never heard of it; the vocabulary's attribute table did."""
    session = session_with({UNKNOWN_SPECIAL: UNKNOWN_SPECIAL_TEXT})
    assert session.tokenize(UNKNOWN_SPECIAL_TEXT) != [UNKNOWN_SPECIAL]   # not a catalogue string
    assert cue_module.single_token_closers(session.tokenize).get(UNKNOWN_SPECIAL) is None
    answer = run_row(session, {UNKNOWN_SPECIAL: TOP})
    assert answer["cue"]["refused"] is True
    assert answer["cue"]["closer"] == UNKNOWN_SPECIAL_TEXT
    assert answer["cue"]["token"] == UNKNOWN_SPECIAL


def test_a_special_token_the_vocabulary_does_not_name_is_still_reported():
    """A vocabulary can carry a special token with no text: name it by its id, never `None`."""
    answer = run_row(session_with({UNNAMED_SPECIAL: ""}), {UNNAMED_SPECIAL: TOP})
    assert answer["cue"]["refused"] is True
    assert answer["cue"]["closer"] == f"<special {UNNAMED_SPECIAL}>"


# --------------------------------------------------------------------- the dominance rule
def test_the_dominance_floor_is_the_engine_s_own_coverage_floor():
    """`dominating` is defined by the same threshold `reliability` uses — one number, not two."""
    assert pytest.approx(OPTION_DEFAULTS["coverage_floor"]) == cue_module.REFUSAL_FLOOR


def test_a_special_token_below_the_floor_is_not_called_dominating():
    answer = run_row(session_with({THINK_END: THINK_END_TEXT}), {THINK_END: BELOW_FLOOR})
    assert answer["cue"]["mass"] < cue_module.REFUSAL_FLOOR          # the premise of the fixture
    assert answer["cue"]["refused"] is False
    assert answer["cue"]["closer"] is None
    assert "hint" not in answer["cue"]


def test_a_catalogue_closer_below_the_floor_is_not_called_dominating():
    """The rule is uniform: `dominating` is about the row, not about which class matched."""
    session = session_with({}, {"<|im_end|>": IM_END})
    answer = run_row(session, {IM_END: BELOW_FLOOR})
    assert answer["cue"]["mass"] < cue_module.REFUSAL_FLOOR          # the premise of the fixture
    assert answer["cue"]["refused"] is False


# --------------------------------------------------------------------- the closer map
def test_the_closer_map_merges_the_catalogue_with_the_vocabulary_specials():
    session = session_with({THINK_END: THINK_END_TEXT, UNKNOWN_SPECIAL: UNKNOWN_SPECIAL_TEXT},
                           {"<|im_end|>": IM_END})
    mapping = cue_module.closer_map(session)
    assert mapping[THINK_END] == THINK_END_TEXT
    assert mapping[UNKNOWN_SPECIAL] == UNKNOWN_SPECIAL_TEXT
    assert mapping[IM_END] == "<|im_end|>"


def test_the_closer_map_falls_back_to_the_catalogue_without_the_attribute_table():
    """A session that cannot enumerate its vocabulary keeps the pre-fix behaviour exactly."""
    session = FakeSession(n_vocab=512)                       # no `special_tokens()` at all
    assert not hasattr(session, "special_tokens")
    assert cue_module.closer_map(session) == cue_module.single_token_closers(session.tokenize)


def test_the_catalogue_wins_when_a_special_token_shares_its_id():
    """`</s>` is a CONTROL token in most vocabularies: the documented name must survive."""
    mapping = cue_module.closer_map(session_with({IM_END: THINK_END_TEXT},
                                                 {"<|im_end|>": IM_END}))
    assert mapping[IM_END] == "<|im_end|>"


# --------------------------------------------------------------------- the vocabulary scan
class FakeLlama:
    """The vocabulary calls `ModelHandle` makes, with a scripted attribute/text table."""

    def __init__(self, attrs: dict[int, int], texts: dict[int, bytes], n_vocab: int) -> None:
        self.attrs = dict(attrs)
        self.texts = dict(texts)
        self.n_vocab = int(n_vocab)
        self.attr_calls = 0

    def llama_model_get_vocab(self, model: object) -> object:
        return "vocab"

    def llama_vocab_n_tokens(self, vocab: object) -> int:
        return self.n_vocab

    def llama_model_n_layer(self, model: object) -> int:
        return 4

    def llama_token_get_attr(self, vocab: object, token: int) -> int:
        self.attr_calls += 1
        return self.attrs.get(int(token), ctypes_binding.TOKEN_ATTR_NORMAL)

    def llama_token_get_text(self, vocab: object, token: int) -> bytes:
        return self.texts.get(int(token), b"plain")


def fake_handle(attrs: dict[int, int], texts: dict[int, bytes], n_vocab: int):
    from ggufone.engine import session as session_module

    llama = FakeLlama(attrs, texts, n_vocab)
    runtime = type("FakeRuntime", (), {"llama": llama})()
    handle = session_module.ModelHandle(runtime, model=object(), path="/fake/model.gguf",
                                        arch="qwen35moe", load_ms=1.0)
    return handle, llama


def test_the_scan_reads_the_vocabulary_attributes_control_and_user_defined_only():
    from ggufone.runtime import ctypes_binding as binding

    attrs = {
        10: binding.TOKEN_ATTR_NORMAL,
        11: binding.TOKEN_ATTR_CONTROL,
        12: binding.TOKEN_ATTR_USER_DEFINED,
        13: binding.TOKEN_ATTR_UNUSED,
        14: binding.TOKEN_ATTR_BYTE,
        15: binding.TOKEN_ATTR_UNKNOWN,
    }
    texts = {11: b"<|im_end|>", 12: b"</think>", 10: b"billing"}
    handle, llama = fake_handle(attrs, texts, n_vocab=16)
    assert handle.special_tokens() == [(11, "<|im_end|>"), (12, "</think>")]
    assert llama.attr_calls == 16                                 # every id asked exactly once
    assert handle.special_tokens() == [(11, "<|im_end|>"), (12, "</think>")]
    assert llama.attr_calls == 16                                 # ... and cached after that


def test_the_scan_degrades_to_empty_when_the_bundle_has_no_attribute_api():
    from ggufone.engine import session as session_module

    class Bare:
        llama_model_get_vocab = staticmethod(lambda model: "vocab")
        llama_vocab_n_tokens = staticmethod(lambda vocab: 16)
        llama_model_n_layer = staticmethod(lambda model: 4)

    runtime = type("FakeRuntime", (), {"llama": Bare})()
    handle = session_module.ModelHandle(runtime, model=object(), path="/fake/model.gguf",
                                        arch="qwen35moe", load_ms=1.0)
    assert ctypes_binding.token_attr(runtime, "vocab", 11) == 0
    assert handle.special_tokens() == []


def test_the_special_attribute_mask_is_control_plus_user_defined():
    from ggufone.runtime import ctypes_binding as binding

    assert binding.SPECIAL_TOKEN_ATTRS == (binding.TOKEN_ATTR_CONTROL
                                           | binding.TOKEN_ATTR_USER_DEFINED)
    for name in ("TOKEN_ATTR_NORMAL", "TOKEN_ATTR_UNUSED", "TOKEN_ATTR_BYTE",
                 "TOKEN_ATTR_UNKNOWN"):
        assert getattr(binding, name) & binding.SPECIAL_TOKEN_ATTRS == 0


# --------------------------------------------------------------------- the rendered table
class SpecialBenchModel(BenchModel):
    """`BenchModel` whose cue row always puts its mass on `</think>` (Tiel's serving shape)."""

    def decide(self, request, *, n_ctx=None, n_seq_max=None, threads=None):
        candidates = max((len(question.options) for question in request.questions), default=1)
        session = SpecialSession(n_ctx=n_ctx or 65536,
                                 n_seq_max=n_seq_max or max(3, 1 + candidates),
                                 threads=threads or 1, n_vocab=self.n_vocab,
                                 model_path="/fake/tiel.gguf", model_alias="bench-vulkan",
                                 load_ms=self.load_ms, prefill_ms=self.prefill_ms,
                                 words=self._words, specials={THINK_END: THINK_END_TEXT},
                                 row_fn=lambda ctx: biased_row(self.n_vocab, {THINK_END: TOP}))
        plan = decide.plan_context(request, session, resolve=False)
        return decide.DecisionEngine(session).decide(request, plan=plan, model_alias="bench-vulkan")


def test_the_quality_report_renders_the_special_token_where_it_used_to_read_ok(monkeypatch,
                                                                               tmp_path):
    from ggufone.bench import harness, suites

    monkeypatch.setattr(harness, "backend_runtimes", lambda **kwargs: {"cpu": tmp_path / "bundle"})
    report = suites.run_suite(harness.BenchConfig(suite="quality", model_path="/tmp/fake.gguf",
                                                  runs=1, items=6),
                              factory=lambda spec: SpecialBenchModel(spec))
    assert report["items"]
    assert all(row["cue"]["refused"] for row in report["items"])
    assert all(row["cue"]["closer"] == THINK_END_TEXT for row in report["items"])
    markdown = harness.render_report(report)
    assert THINK_END_TEXT in markdown
    assert "W_CUE_REFUSED" in markdown


# --------------------------------------------------------------------- the live instrument
def load_probe():
    spec = importlib.util.spec_from_file_location("e3c_cue_shapes_specials",
                                                  ROOT / "tools" / "e3c_cue_shapes.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["e3c_cue_shapes_specials"] = module
    spec.loader.exec_module(module)
    return module


def test_the_shape_probe_reads_its_verdict_map_from_the_vocabulary():
    """`tools/e3c_cue_shapes.py` is the instrument for this verdict: it must see the same map."""
    probe = load_probe()

    class Handle:
        SPECIALS = {THINK_END: THINK_END_TEXT}

        def tokenize(self, text: str) -> list[int]:
            return [IM_END] if text == "<|im_end|>" else [WORD]

        def special_tokens(self):
            return sorted(self.SPECIALS.items())

    mapping = probe.verdict_map(Handle())
    assert mapping[THINK_END] == THINK_END_TEXT
    assert mapping[IM_END] == "<|im_end|>"
