"""E3c: the cue verdict — `W_CUE_REFUSED` and the table that shows it (card t_6c119626).

E3b (`t_6952f0dd`) measured the label-policy question into a clean negative: 0/6 items clear the
engine's 0.10 coverage floor on every one of the 15 (cue × label) combinations, because the *cue
row* itself is dominated by `<|im_end|>` (p = 0.9976…1.00000) — the model closes the assistant
turn instead of answering. The engine reported only `low_mass`, which hides *why*: a caller sees
"not enough candidate mass" where the honest verdict is "this family refuses this prompt shape".

These gates pin the named warning, the payload the flat `warnings` list cannot carry (the closer,
its mass, the doc pointer), the quality report's rendered verdict table, and the document rows the
measurements made false. Every one of them fails on the pre-E3c tree:

* `W_CUE_REFUSED` does not exist in the catalog and `engine/cue.py` does not exist;
* the engine publishes `coverage`/`reliability` and nothing about the row it read them from;
* `harness.render_report` has no cue verdict table at all;
* `docs/TEMPLATES.md` still marks `qwen35moe` **[UNVERIFIED]** ("no MoE GGUF on this box").

The closer is matched through the *session's own tokenizer*, never by a hard-coded token id: a
closer counts only when the model's vocabulary encodes that string as exactly one token and that
token is the row's argmax. The fake below emulates a real special-token vocabulary (each closer
one id); a closer the vocabulary splits (`<|eot_id|>` in a word-level fake) must never fire.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

from ggufone import errors, schema
from ggufone.bench import harness, suites
from ggufone.engine import cue as cue_module
from ggufone.engine import decide
from tests.fake_engine import BenchModel, FakeSession, biased_row

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "docs" / "TEMPLATES.md"

#: the bias that puts ~0.99 of one token's mass in a 512-slot vocabulary
TOP = 11.0
#: the ids this fake vocabulary reserves for the closers it encodes as ONE token
IM_END = 400
EOS = 401
#: the id of the plain word the "content token" gate biases (`words={"unrelated": 7}`)
WORD = 7


class CloserSession(FakeSession):
    """`FakeSession` with a real special-token vocabulary for four closers.

    A live model's tokenizer parses `<|im_end|>` into its single special id (llama.cpp runs
    `llama_tokenize` with `parse_special=True`); the word-level fake would split it into `im` and
    `end`, which is exactly the case a closer must *not* match on (the split-closer gate).
    """

    SPECIAL = {"<|im_end|>": IM_END, "</s>": EOS, "<|endoftext|>": EOS + 1, "eos": EOS + 2}

    def tokenize(self, text: str) -> list[int]:
        if text in self.SPECIAL:
            return [self.SPECIAL[text]]
        return super().tokenize(text)


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


def run(session: FakeSession, payload: dict):
    request = schema.parse_request({key: value for key, value in payload.items()
                                    if value is not None})
    plan = decide.plan_context(request, session)
    return decide.DecisionEngine(session).decide(request, plan=plan)


def refusing_session(row_bias: dict[int, float], **kwargs) -> CloserSession:
    """A session whose every row puts `TOP` on the biased token ids."""
    session = CloserSession(n_vocab=512, **kwargs)
    session.row_fn = lambda ctx: biased_row(session.n_vocab, row_bias)
    return session


# --------------------------------------------------------------------- the catalog
def test_the_warning_catalog_carries_the_named_verdict():
    assert "W_CUE_REFUSED" in errors.WARNING_CODES
    assert len(set(errors.WARNING_CODES)) == len(errors.WARNING_CODES)


def test_the_closer_catalog_names_the_documented_families():
    """One entry per documented turn-closer; this card's measurement is `<|im_end|>`."""
    assert "<|im_end|>" in cue_module.TURN_CLOSERS
    assert "</s>" in cue_module.TURN_CLOSERS
    assert "<|endoftext|>" in cue_module.TURN_CLOSERS
    assert "eos" in cue_module.TURN_CLOSERS
    assert all(isinstance(text, str) and text for text in cue_module.TURN_CLOSERS)


# --------------------------------------------------------------------- the verdict
def test_a_closer_at_the_cue_raises_w_cue_refused_and_carries_its_mass():
    result = run(refusing_session({IM_END: TOP}), choice_request())
    answer = result.answers["area"]
    assert "W_CUE_REFUSED" in result.warnings
    assert answer["cue"]["refused"] is True
    assert answer["cue"]["closer"] == "<|im_end|>"
    assert answer["cue"]["token"] == IM_END
    assert answer["cue"]["mass"] == pytest.approx(0.99, abs=0.02)
    assert answer["reliability"] == "low_mass"          # the refusal is *why* the mass is low
    assert "W_LOW_MASS" in result.warnings


def test_a_content_token_at_the_cue_is_not_a_refusal():
    result = run(refusing_session({WORD: TOP}, words={"unrelated": WORD}), choice_request())
    answer = result.answers["area"]
    assert "W_CUE_REFUSED" not in result.warnings
    assert answer["cue"]["refused"] is False
    assert answer["cue"]["closer"] is None
    assert answer["cue"]["token"] == WORD
    assert answer["cue"]["mass"] > 0.99
    assert "hint" not in answer["cue"]                  # the pointer is for a refusal, not a row


def test_the_verdict_is_reported_for_every_question_type():
    payload = {
        "state": "The checkout page returns HTTP 500 for every customer.",
        "questions": {
            "area": {"type": "choice", "criteria": {"billing": "payments", "technical": "api"}},
            "severity": {"type": "score", "criteria": ["cosmetic", "critical"]},
            "page": {"type": "noul", "criteria": {"true": "yes, page now", "false": "no, wait"}},
        },
    }
    result = run(refusing_session({IM_END: TOP}), payload)
    for qid in ("area", "severity", "page"):
        assert result.answers[qid]["cue"]["refused"] is True, qid
    assert result.warnings.count("W_CUE_REFUSED") == 1        # one code, not one per question


def test_a_closer_the_vocabulary_splits_into_several_tokens_never_fires():
    """`<|eot_id|>` is not one token in the fake vocabulary: a prefix of it is not a closer."""
    session = CloserSession(n_vocab=512, words={"eot": 300})
    split = session.tokenize("<|eot_id|>")
    assert len(split) > 1                                     # the premise of the gate
    session.row_fn = lambda ctx: biased_row(session.n_vocab, {split[0]: TOP})
    result = run(session, choice_request())
    assert "W_CUE_REFUSED" not in result.warnings
    assert result.answers["area"]["cue"]["refused"] is False


def test_the_hint_points_at_the_documented_section_and_no_imaginary_flag():
    answer = run(refusing_session({IM_END: TOP}), choice_request()).answers["area"]
    hint = answer["cue"]["hint"]
    assert hint.startswith("docs/TEMPLATES.md")
    assert "§4" in hint
    assert "--cue" not in hint                                # the CLI has no such flag yet


def test_the_cue_block_survives_native_rendering_at_wire_precision():
    result = run(refusing_session({IM_END: TOP}), choice_request())
    rendered = schema.render_response(result.payload(), format="native")
    block = rendered["answers"]["area"]["cue"]
    assert block["closer"] == "<|im_end|>"
    assert block["mass"] == pytest.approx(result.answers["area"]["cue"]["mass"], abs=1e-6)
    assert block["refused"] is True
    # the typesafe projection drops it with the other native-only keys (SPEC 2.6)
    typesafe = schema.render_response(result.payload(), format="typesafe")
    assert "cue" not in typesafe["answers"]["area"]


def test_a_healthy_answer_reports_no_refusal_and_keeps_its_cue_block():
    session = CloserSession(n_vocab=512, words={"billing": 5})
    session.row_fn = lambda ctx: biased_row(session.n_vocab, {5: TOP})
    answer = run(session, choice_request()).answers["area"]
    assert answer["reliability"] == "ok"
    assert answer["cue"]["refused"] is False
    assert answer["cue"]["mass"] > 0.99


# --------------------------------------------------------------------- the rendered table
class CloserBenchModel(BenchModel):
    """`BenchModel` whose cue row always closes the turn (the E3b shape, without a model)."""

    def decide(self, request, *, n_ctx=None, n_seq_max=None, threads=None):
        candidates = max((len(question.options) for question in request.questions), default=1)
        session = CloserSession(n_ctx=n_ctx or 65536,
                                n_seq_max=n_seq_max or max(3, 1 + candidates),
                                threads=threads or 1, n_vocab=self.n_vocab,
                                model_path="/fake/refusing.gguf", model_alias="bench-cpu",
                                load_ms=self.load_ms, prefill_ms=self.prefill_ms,
                                words=self._words,
                                row_fn=lambda ctx: biased_row(self.n_vocab, {IM_END: TOP}))
        plan = decide.plan_context(request, session, resolve=False)
        return decide.DecisionEngine(session).decide(request, plan=plan, model_alias="bench-cpu")


def quality_report(factory, monkeypatch, tmp_path):
    monkeypatch.setattr(harness, "backend_runtimes", lambda **kwargs: {"cpu": tmp_path / "bundle"})
    return suites.run_suite(harness.BenchConfig(suite="quality", model_path="/tmp/fake.gguf",
                                                runs=1, items=6), factory=factory)


def test_the_quality_row_carries_the_verdict_and_the_table_renders_it(monkeypatch, tmp_path):
    report = quality_report(lambda spec: CloserBenchModel(spec), monkeypatch, tmp_path)
    assert report["items"], "the fixture measured no row"
    assert all(row["cue"]["refused"] for row in report["items"])
    markdown = harness.render_report(report)
    assert "<|im_end|>" in markdown
    assert "W_CUE_REFUSED" in markdown
    # the mass is the row's own number, rendered — not implied by the verdict word
    assert f"{report['items'][0]['cue']['mass']:.4f}" in markdown


def test_the_table_renders_a_content_row_without_the_refusal_warning(monkeypatch, tmp_path):
    """The card's second fixture row: the top token is content, so nothing is refused."""
    report = quality_report(lambda spec: BenchModel(spec, script={}, n_vocab=512),
                            monkeypatch, tmp_path)
    assert report["items"] and all(row["cue"]["refused"] is False for row in report["items"])
    markdown = harness.render_report(report)
    assert "cue verdicts" in markdown.lower()
    assert "W_CUE_REFUSED" not in markdown
    assert "| ok |" in markdown


def test_a_report_written_before_e3c_still_renders(monkeypatch, tmp_path):
    """Rows without a cue block (E2/E3 reports on disk) must not change the rendered document."""
    report = quality_report(lambda spec: CloserBenchModel(spec), monkeypatch, tmp_path)
    for row in report["items"]:
        row.pop("cue", None)
    markdown = harness.render_report(report)
    assert "cue verdict" not in markdown.lower()
    assert markdown.startswith("### ")


# --------------------------------------------------------------------- the documents
def templates_text() -> str:
    return TEMPLATES.read_text(encoding="utf-8")


def family_row(family: str) -> str:
    """The one row of §4's family table that documents `family`.

    Scoped to §4: §3 names the same families in its `suppressed` row (a fact about thinking
    suppression, not about the cue), so a whole-file scan would find two rows.
    """
    body = templates_text().partition("## 4. The family table")[2].partition("### ")[0]
    rows = [line for line in body.splitlines()
            if line.startswith("|") and f"`{family}`" in line]
    assert len(rows) == 1, f"§4 has {len(rows)} rows for {family}"
    return rows[0]


def section(title: str) -> str:
    text = templates_text()
    assert title in text, f"docs/TEMPLATES.md has no {title!r} section"
    return text.partition(title)[2].split("\n## ")[0]


def test_the_qwen35moe_row_is_no_longer_unverified():
    row = family_row("qwen35moe")
    assert "[UNVERIFIED]" not in row
    assert "[executed]" in row
    assert "cue" in row.lower()


def test_the_known_limits_no_longer_claim_there_is_no_moe_gguf_here():
    assert "no MoE GGUF on this box" not in templates_text()


def test_templates_records_the_measured_refusal_and_the_numbers_that_back_it():
    text = templates_text()
    assert "E3b" in text and "Occamy" in text
    assert "low_mass" in text
    assert "0.9976" in text or "1.00000" in text


def test_templates_publishes_the_cue_guidance_from_the_probe():
    """Every shape the probe measures is named in the guidance — from the probe, not re-typed."""
    body = section("### Cue shapes that put the readout mid-answer")
    for shape in probe_shape_names():
        assert f"`{shape}`" in body, shape


def probe_shape_names() -> tuple[str, ...]:
    """The shape catalogue of `tools/e3c_cue_shapes.py` (the doc's source of truth)."""
    import sys

    spec = importlib.util.spec_from_file_location("e3c_cue_shapes_doc_gate",
                                                  ROOT / "tools" / "e3c_cue_shapes.py")
    module = importlib.util.module_from_spec(spec)
    # a `dataclass(slots=True)` needs its module registered before the class body runs
    sys.modules["e3c_cue_shapes_doc_gate"] = module
    spec.loader.exec_module(module)
    return tuple(module.SHAPE_NAMES)
