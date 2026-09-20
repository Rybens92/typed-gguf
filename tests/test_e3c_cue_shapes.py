"""E3c: the mid-answer cue-shape probe, offline (card t_6c119626 §1).

`tools/e3c_cue_shapes.py` measures the live model; these gates pin the parts that must not move
between runs, without a model:

* every shape is the **shipped suffix plus an opener** — the instructions, the criteria and the
  question can never differ between shapes (the strongest form of "this measures the cue, not the
  question");
* the `two_step_*` shapes read at a **second** decision point and never at a closer: their advance
  rule excludes the turn-closer catalogue, `greedy` is the engine's own argmax with the frozen
  tie-break;
* the rendered tables carry the verdict — the card's fixture rows (a closer at 0.99 ⇒
  `W_CUE_REFUSED`, a content token at 0.99 ⇒ `ok`) and the EOT-vs-candidate masses;
* the probe's own ranking is the candidate with the largest first-token mass, and it refuses to
  rank when the label variant cannot separate the candidates.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

from typed_gguf.bench import labels
from typed_gguf.engine import readout
from typed_gguf.schema import parse_request

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_probe():
    """Import `tools/e3c_cue_shapes.py` the way the other gates import a tool."""
    spec = importlib.util.spec_from_file_location("e3c_cue_shapes",
                                                  ROOT / "tools" / "e3c_cue_shapes.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["e3c_cue_shapes"] = module
    spec.loader.exec_module(module)
    return module


probe = load_probe()

CHOICE = {"type": "choice", "instructions": "Which team owns this?",
          "criteria": {"billing": "payments", "technical": "api", "support": None}}
SCORE = {"type": "score", "instructions": "How severe?", "criteria": ["cosmetic", "critical"]}
NOUL = {"type": "noul", "instructions": "Page the engineer?", "criteria": {"true": "yes",
                                                                           "false": "no"}}


def question_of(body: dict):
    request = parse_request({"state": "The checkout page returns HTTP 500.", "model": "test",
                             "questions": {"q": body}})
    return request.questions[0]


# --------------------------------------------------------------------- the shape catalogue
def test_the_shipped_shape_is_the_engine_s_own_suffix_byte_for_byte():
    """The control: `shipped` must render exactly what E3/E3b measured `low_mass` on."""
    shipped = "instructions\nAnswer with exactly one candidate name:\n"
    question = question_of(CHOICE)
    assert probe.shape_by_name("shipped").suffix(shipped, question) == shipped


@pytest.mark.parametrize("name", [name for name in probe.SHAPE_NAMES if name != "shipped"])
def test_every_shape_keeps_every_byte_before_its_opener(name):
    shipped = "instructions\nAnswer with exactly one candidate name:\n"
    question = question_of(CHOICE)
    shape = probe.shape_by_name(name)
    suffix = shape.suffix(shipped, question)
    assert suffix.startswith(shipped)
    # a shape either adds an opener or moves the readout (the two-step shapes' opener is "")
    assert suffix != shipped or shape.advance > 0


def test_the_two_step_shape_on_the_shipped_cue_is_byte_identical_to_it():
    """It moves the *readout*, not the prompt: same bytes, second decision point."""
    shipped = "instructions\nAnswer with exactly one candidate name:\n"
    shape = probe.shape_by_name("two_step_shipped")
    assert shape.suffix(shipped, question_of(CHOICE)) == shipped
    assert shape.advance == 1


def test_the_json_shape_opens_the_field_for_the_question_type():
    shipped = "cue:\n"
    assert probe.shape_by_name("json_field").suffix(shipped, question_of(CHOICE)) == \
        shipped + '{"choice": "'
    assert probe.shape_by_name("json_field").suffix(shipped, question_of(SCORE)) == \
        shipped + '{"severity": "'
    assert probe.shape_by_name("json_field").suffix(shipped, question_of(NOUL)) == \
        shipped + '{"answer": "'


def test_the_catalogue_is_frozen_and_names_the_languages_it_tests():
    assert probe.SHAPE_NAMES == ("shipped", "answer_is", "answer_colon", "wybieram_pl",
                                 "json_field", "two_step_shipped", "two_step_answer_is")
    assert "Zgodnie" in probe.shape_by_name("wybieram_pl").opener      # Polish, not English
    assert probe.shape_by_name("answer_is").opener == "The answer is "
    for name in ("two_step_shipped", "two_step_answer_is"):
        shape = probe.shape_by_name(name)
        assert shape.advance == 1 and shape.rule == probe.CONTENT


def test_an_unknown_shape_names_the_catalogue():
    with pytest.raises(SystemExit) as info:
        probe.shape_by_name("cue_vibes")
    assert "shipped" in str(info.value)


# --------------------------------------------------------------------- the advance rules
def test_the_content_rule_never_advances_onto_a_turn_closer():
    closet = {2: "<|im_end|>"}
    row = [0.0, 5.0, 9.0]                                # a closer at 9.0, content at 5.0
    assert probe.advance_token(row, closet, probe.CONTENT) == 1
    assert probe.advance_token(row, closet, probe.GREEDY) == 2


def test_the_greedy_rule_keeps_the_engine_s_lowest_index_tie_break():
    row = [3.0, 3.0, 0.0]
    assert probe.advance_token(row, {}, probe.GREEDY) == 0
    assert readout.argmax_first(row) == probe.advance_token(row, {}, probe.GREEDY)


def test_a_row_that_is_all_closers_is_a_named_error():
    """A vocabulary with exactly one token, and that token is a closer: nothing to advance to."""
    with pytest.raises(SystemExit):
        probe.advance_token([1.0], {0: "<|im_end|>"}, probe.CONTENT)


def test_the_closer_mass_block_is_the_engine_s_own_coverage_function():
    row = [0.0, 4.0, 0.0]
    scale = readout.logsumexp(row)
    masses = probe.closer_mass(row, scale, {1: "<|im_end|>", 2: "</s>"})
    assert set(masses) == {"<|im_end|>", "</s>"}
    assert masses["<|im_end|>"] == readout.coverage_from_scale(row, (1,), scale)
    assert masses["<|im_end|>"] > masses["</s>"]


# --------------------------------------------------------------------- the probe's ranking
def test_the_probe_s_ranking_is_the_largest_first_token_mass():
    block = {"bare": {"first_mass": [0.01, 0.4, 0.2]}}
    assert probe.top_coverage(block, "bare", ("billing", "technical", "support")) == "technical"


def test_the_probe_refuses_to_rank_when_the_variant_cannot_separate_the_candidates():
    """A caller that passes the wrong option list gets no ranking, never a guess."""
    assert probe.top_coverage({"bare": {"first_mass": [0.1]}}, "bare", ("a", "b")) is None
    assert probe.top_coverage({}, "bare", ("a",)) is None


# --------------------------------------------------------------------- the rendered tables
def readout_block(*, mass: float, coverage: float, closer: str | None) -> dict:
    return {
        "row": "cue", "position": 120,
        "top_tokens": [{"token": 7, "piece": closer or "billing", "p_full": mass}],
        "closer_mass": ({} if closer is None else {closer: mass}),
        "cue": {"refused": closer is not None, "token": 7, "closer": closer, "mass": mass},
        "labels": {"bare": {"texts": ["billing", "technical"], "first_tokens": [7, 8],
                            "coverage": coverage, "reliability": labels.reliability_of(coverage),
                            "first_mass": [coverage / 2, coverage / 2]}},
    }


def record(*, closer: str | None, mass: float = 0.99, coverage: float = 0.001,
           ranked: bool = False) -> dict:
    shapes = ["shipped", "two_step_shipped"]
    item = {"id": "c01", "type": "choice", "expected": "technical",
            "shapes": {name: {"suffix_tokens": 12, "opener": "", "readout": readout_block(
                mass=mass, coverage=coverage, closer=closer)} for name in shapes},
            "ranked": ({"shipped=bare": {
                "shape": "shipped", "label": "bare", "got": "billing", "correct": False,
                "confidence": 0.6, "coverage": coverage, "reliability": "low_mass",
                "top_coverage": "technical", "agrees": False, "probabilities": {}}} if ranked
                else {})}
    return {"schema": probe.SCHEMA, "generated_at": "2026-09-19T00:00:00Z",
            "model": {"name": "Occamy", "arch": "qwen35moe", "bytes": 24113674848},
            "model_sha256": "633ae57f", "runtime": "/runtime/vulkan", "threads": 4,
            "gpu_layers": 7, "placement": {"n_gpu_layers": 3}, "load_ms": 22000.0,
            "shapes": shapes, "label_variants": ["bare"], "mass_floor": labels.MASS_FLOOR,
            "ranked_keys": (["shipped=bare"] if ranked else []),
            "counts": {"choice": 1}, "devset": "devset.jsonl", "wall_s": 100.0,
            "items": [item]}


def test_the_verdict_table_marks_a_closer_row_and_leaves_a_content_row_alone():
    refused = probe.render_report(record(closer="<|im_end|>"))
    assert "| W_CUE_REFUSED |" in refused
    assert refuted_table_row(refused, "shipped").endswith("W_CUE_REFUSED |")
    content = probe.render_report(record(closer=None))
    section_one = content.partition("## 1.")[2].partition("## 2.")[0]
    assert "| W_CUE_REFUSED |" not in section_one            # the cell, not the legend below it
    assert refuted_table_row(content, "shipped").endswith("| ok |")


def refuted_table_row(markdown: str, shape: str) -> str:
    """The verdict-table row of `shape` in section 1 (the document's first table)."""
    body = markdown.partition("## 1.")[2].partition("## 2.")[0]
    rows = [line for line in body.splitlines()
            if line.startswith("| c01 ") and f"| `{shape}` |" in line]
    assert len(rows) == 1, f"{len(rows)} verdict rows for {shape}"
    return rows[0]


def test_the_verdict_table_carries_the_mass_the_engine_measured():
    row = refuted_table_row(probe.render_report(record(closer="<|im_end|>", mass=0.9876)),
                            "shipped")
    assert "0.9876" in row
    assert "<|im_end|>" in row


def test_the_eot_table_compares_the_closer_s_mass_with_the_candidate_s():
    markdown = probe.render_report(record(closer="<|im_end|>", coverage=0.001))
    body = markdown.partition("## 2.")[2].partition("## 3.")[0]
    assert "<|im_end|>" in body and "1.000e-03" in body


def test_the_coverage_table_marks_every_value_below_the_floor():
    markdown = probe.render_report(record(closer="<|im_end|>", coverage=0.001))
    body = markdown.partition("## 3.")[2].partition("## 4.")[0]
    assert "0.0010*" in body                                  # the star is the floor verdict
    assert "| 0/1 |" in body


def test_the_summary_counts_the_refused_items_per_shape():
    markdown = probe.render_report(record(closer="<|im_end|>"))
    body = markdown.partition("## 4.")[2].partition("## 5.")[0]
    assert "| 1/1 |" in body                                  # one item, one refusal


def test_a_run_without_ranking_says_so_and_one_with_ranking_tabulates_it():
    assert "No shape was ranked" in probe.render_report(record(closer=None))
    markdown = probe.render_report(record(closer=None, ranked=True))
    body = markdown.partition("## 5.")[2]
    assert "`shipped=bare`" in body
    assert "0/1" in body                                      # ranked winner != coverage winner
