"""E3b: the label-policy variants (card t_6952f0dd) — offline gates.

The published comparison put every Occamy answer below the engine's mass floor (`low_mass`,
20/20), while the 4B is mostly `measured`. The card's hypothesis is that the *rendering* of the
candidate labels is part of that: `coverage` is the full-vocabulary mass of the label's first
token at the cue, so a family that opens its answer with a newline (or with a space-prefixed
word, as BPE vocabularies usually encode it) shows a small mass on a bare label even when it is
deciding exactly as asked.

These gates pin the variants themselves — the strings a probe can ask for — without a model:

* `bare` must be what the engine renders today (`prompt.build_question(...).texts`), because
  that is the only rendering any published number was measured with;
* every other variant must be a deterministic, documented transformation of it;
* the cue variants must be rewrites of the rendered suffix that never touch a byte before the
  cue line (`shipped` is the identity — the strongest form of "this is a no-op").
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any

import pytest

from ggufone.bench import labels
from ggufone.engine import prompt, readout
from ggufone.schema import parse_request

ROOT = pathlib.Path(__file__).resolve().parents[1]

CHOICE = {"choice": {"type": "choice", "instructions": "Which team owns this?",
                     "criteria": {"billing": "payments, invoices and refunds",
                                  "technical": "api, infrastructure and deploys",
                                  "support": None}}}
SCORE = {"severity": {"type": "score", "instructions": "How severe?",
                      "criteria": ["cosmetic", "annoying", "catastrophic"]}}
NOUL = {"page": {"type": "noul", "instructions": "Page the on-call engineer?",
                 "criteria": {"true": "yes, page immediately", "false": "no, wait"}}}


def question_payload(questions: dict) -> dict:
    return {"state": "The checkout page returns HTTP 500 for every customer.",
            "model": "test", "questions": questions}


def rendered(questions: dict) -> tuple:
    """The parsed question plus what `prompt.build_question` renders for it."""
    request = parse_request(question_payload(questions))
    question = request.questions[0]
    return question, prompt.build_question(question)


# ------------------------------------------------------------------ label variants
@pytest.mark.parametrize("questions", [CHOICE, SCORE, NOUL])
def test_the_bare_variant_is_exactly_what_the_engine_renders(questions):
    question, view = rendered(questions)
    texts = labels.label_texts(question.type, question.options, question.descriptions, "bare")
    assert texts == view.texts


@pytest.mark.parametrize("questions", [CHOICE, SCORE, NOUL])
@pytest.mark.parametrize("variant", labels.LABEL_VARIANTS)
def test_every_variant_renders_one_distinct_label_per_candidate(questions, variant):
    question, _ = rendered(questions)
    texts = labels.label_texts(question.type, question.options, question.descriptions, variant)
    assert len(texts) == len(question.options)
    assert len(set(texts)) == len(texts)
    assert all(text for text in texts)


def test_the_leading_space_variant_prefixes_every_label():
    question, _ = rendered(CHOICE)
    texts = labels.label_texts(question.type, question.options, question.descriptions, "space")
    assert texts == (" billing", " technical", " support")


def test_the_caps_variant_upper_cases_the_first_character_only():
    question, _ = rendered(CHOICE)
    assert labels.label_texts(question.type, question.options, question.descriptions, "caps") == \
        ("Billing", "Technical", "Support")
    score, _ = rendered(SCORE)
    # a level *number* has no case to change: the variant must be the identity for it
    assert labels.label_texts(score.type, score.options, score.descriptions, "caps") == \
        ("0", "1", "2")


def test_the_newline_variant_is_the_two_step_readout():
    """The model's own blank line, then the label — `TEMPLATES.md` §4's two-step readout."""
    question, _ = rendered(NOUL)
    texts = labels.label_texts(question.type, question.options, question.descriptions, "newline")
    assert texts == ("\nyes", "\nno")


def test_the_long_variant_uses_the_description_and_falls_back_to_the_bare_label():
    question, _ = rendered(CHOICE)
    texts = labels.label_texts(question.type, question.options, question.descriptions, "long")
    assert texts == ("billing: payments, invoices and refunds",
                     "technical: api, infrastructure and deploys",
                     "support")          # no description -> the bare label, never "support: None"
    noul, _ = rendered(NOUL)
    assert labels.label_texts(noul.type, noul.options, noul.descriptions, "long") == \
        ("yes, page immediately", "no, wait")
    score, _ = rendered(SCORE)
    assert labels.label_texts(score.type, score.options, score.descriptions, "long") == \
        ("cosmetic", "annoying", "catastrophic")


def test_an_unknown_label_variant_names_the_accepted_ones():
    question, _ = rendered(CHOICE)
    with pytest.raises(labels.LabelsError) as info:
        labels.label_texts(question.type, question.options, question.descriptions, "shouty")
    assert "E_LABEL_VARIANT" in str(info.value)
    assert "bare" in str(info.value) and "newline" in str(info.value)


def test_a_description_only_long_variant_is_still_allowed_for_choice():
    """`long` on a choice question whose descriptions are all absent is the bare rendering."""
    payload = {"q": {"type": "choice", "criteria": {"a1": None, "b2": None}}}
    question, _ = rendered(payload)
    assert labels.label_texts(question.type, question.options, question.descriptions, "long") == \
        ("a1", "b2")


# ------------------------------------------------------------------ cue variants
@pytest.mark.parametrize("questions", [CHOICE, SCORE, NOUL])
def test_the_shipped_cue_variant_is_byte_identical_to_the_rendered_suffix(questions):
    question, view = rendered(questions)
    assert labels.suffix_with_cue(view.suffix, question.type, question.options, "shipped") == \
        view.suffix


def test_the_blank_cue_variant_adds_exactly_one_empty_line():
    question, view = rendered(CHOICE)
    shipped = view.suffix
    cue = prompt.CANDIDATE_CUE["choice"] + "\n"
    assert shipped.endswith(cue)
    blank = labels.suffix_with_cue(shipped, question.type, question.options, "blank")
    assert blank == shipped + "\n"                 # cue + "\n\n": one extra newline, nothing else
    assert blank.startswith(shipped[: -len(cue)])  # every byte before the cue is untouched


def test_the_explicit_cue_variant_names_every_label():
    question, view = rendered(CHOICE)
    explicit = labels.suffix_with_cue(view.suffix, question.type, question.options, "explicit")
    tail = explicit[len(view.suffix) - len(prompt.CANDIDATE_CUE["choice"] + "\n"):]
    for name in question.options:
        assert name in tail
    assert tail.endswith("\n") and not tail.endswith("\n\n")
    assert explicit.startswith(view.suffix[: -len(prompt.CANDIDATE_CUE["choice"] + "\n")])


def test_the_explicit_cue_of_a_score_question_names_the_level_numbers():
    question, view = rendered(SCORE)
    explicit = labels.suffix_with_cue(view.suffix, question.type, question.options, "explicit")
    assert "0, 1, 2" in explicit


def test_a_suffix_that_does_not_end_at_the_cue_is_rejected():
    question, _ = rendered(CHOICE)
    with pytest.raises(labels.LabelsError) as info:
        labels.suffix_with_cue("QUESTION:\nno cue here\n", question.type, question.options, "blank")
    assert "E_LABEL_CUE" in str(info.value)


def test_an_unknown_cue_variant_names_the_accepted_ones():
    question, view = rendered(CHOICE)
    with pytest.raises(labels.LabelsError) as info:
        labels.suffix_with_cue(view.suffix, question.type, question.options, "loud")
    assert "E_LABEL_CUE" in str(info.value)
    assert "shipped" in str(info.value)


# ------------------------------------------------------------------ row math
class FakeTokenizer:
    """One token per label, so the arithmetic of the readout is visible without a model."""

    def __init__(self, mapping: dict[str, list[int]]) -> None:
        self.mapping = mapping

    def tokenize(self, text: str) -> list[int]:
        return list(self.mapping[text])


def test_label_coverage_is_the_engine_coverage_math():
    """The probe must not invent its own coverage: same row, same scale, same function."""
    tokenizer = FakeTokenizer({" billing": [7, 8], " technical": [9]})
    row = [0.0] * 12
    row[7], row[8], row[9], row[3] = 2.0, 1.5, 0.5, 1.0
    scale = readout.logsumexp(row)
    got = labels.label_coverage((" billing", " technical"), tokenizer.tokenize, row, scale)
    assert got == readout.coverage_from_scale(row, [7, 9], scale)
    assert 0.0 < got < 1.0


def test_label_coverage_counts_a_shared_first_token_once_per_candidate():
    """Exactly like the engine: the sum is over candidates, and it is capped at 1.0."""
    tokenizer = FakeTokenizer({"\nyes": [4], "\nno": [4]})
    row = [0.0] * 8
    row[4] = 10.0
    scale = readout.logsumexp(row)
    assert labels.label_coverage(("\nyes", "\nno"), tokenizer.tokenize, row, scale) == 1.0


def test_shared_first_tokens_names_the_candidates_a_variant_cannot_tell_apart():
    tokenizer = FakeTokenizer({"very low": [1, 2], "very high": [1, 3], "low": [5]})
    assert labels.shared_first_tokens(("very low", "very high"), tokenizer.tokenize) == \
        [("very low", "very high")]
    assert labels.shared_first_tokens(("very low", "low"), tokenizer.tokenize) == []


def test_the_dev_set_labels_are_what_the_card_measured():
    """The card's own framing: names for `choice`, numbers for `score`, yes/no for `noul`."""
    assert labels.label_texts("choice", ("billing", "technical"), (None, None), "bare") == \
        ("billing", "technical")
    assert labels.label_texts("score", ("0", "1"), ("cosmetic", "annoying"), "bare") == ("0", "1")
    assert labels.label_texts("noul", ("yes", "no"), ("y", "n"), "bare") == ("yes", "no")


# ------------------------------------------------------------------ the report's number format
def load_probe():
    """Import `tools/e3b_label_policy.py` — the report formatters live there, not in `labels.py`."""
    spec = importlib.util.spec_from_file_location("e3b_label_policy_probe",
                                                  ROOT / "tools" / "e3b_label_policy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["e3b_label_policy_probe"] = module
    spec.loader.exec_module(module)
    return module


def report_record(*, coverage: float = 1e-08, floor: float = 0.10) -> dict:
    """The smallest record the report formatters accept: one item, one cue, all five labels."""
    labels_block = {
        name: {"texts": ["a"], "first_tokens": [1], "pieces": ["a"], "coverage": coverage,
               "reliability": "low_mass", "shared_first_tokens": []}
        for name in labels.LABEL_VARIANTS}
    piece = {"id": "c01", "type": "choice", "expected": "a", "cues": {
        "shipped": {"scale": 1.0, "labels": dict(labels_block),
                    "top_tokens": [{"piece": "<|im_end|>", "p_full": 0.9999}]}},
        "ranked": {}}
    return {"cues": ["shipped"], "label_variants": list(labels.LABEL_VARIANTS), "mass_floor": floor,
            "items": [{"id": "c01", "type": "choice", "expected": "a",
                       "prefixes": {"shipped": piece}}]}


def test_the_coverage_tables_print_small_masses_in_scientific_notation():
    """The whole card is about masses 1e-8 … 1e-6: a table of `0.0000` publishes nothing.

    The rule is keyed off the engine's own floor: when the floor itself is smaller than 1e-3 the
    table switches to 3-decimal scientific notation, and the star (below floor) marker is appended
    outside the number.
    """
    probe = load_probe()
    lines = probe.coverage_table(report_record(coverage=8.0684e-08))
    row = next(line for line in lines if line.startswith("| `shipped` | `bare` |"))
    assert "8.068e-08*" in row
    assert "0.0000" not in row

    summary = "\n".join(probe.variant_summary(report_record(coverage=4.2487e-07)))
    assert "4.249e-07" in summary, "the mean is printed the same way as the per-item cells"


def test_the_tables_stay_fixed_point_when_the_floor_is_readable():
    """A floor at 1e-3 or above keeps the familiar 4-decimal form — no gratuitous notation."""
    probe = load_probe()
    lines = probe.coverage_table(report_record(coverage=0.25, floor=0.4))
    row = next(line for line in lines if line.startswith("| `shipped` | `bare` |"))
    assert "0.2500*" in row
    assert "e-0" not in row


# ------------------------------------------------------------------ the ranked readout's trie
class FakeRowSession:
    """A session seam: canned rows **by position**, plus a record of every fork and decode.

    The ranked readout's whole contract is *which* token is decoded at *which* position off
    *which* sequence — that is what these gates pin, with rows whose logprobs are exact.
    """

    def __init__(self, rows_by_position: dict[int, list[float]]) -> None:
        self.rows_by_position = rows_by_position
        self.forks: list[tuple[int, int, int]] = []
        self.batches: list[Any] = []

    def fork(self, src: int, dst: int, upto: int) -> None:
        self.forks.append((int(src), int(dst), int(upto)))

    def decode(self, batch: Any) -> list[list[float]]:
        self.batches.append(batch)
        return [self.rows_by_position[int(position)] for position in batch.positions]


def exact_row(mass: dict[int, float], *, width: int = 4) -> list[float]:
    """A logit row whose softmax is exactly `mass` on the given tokens (rest zero logits)."""
    import math
    row = [0.0] * width
    for token, value in mass.items():
        row[token] = math.log(value)
    return row


def test_the_trie_levels_are_the_prefixes_the_readout_must_decode():
    # a single-token path needs no level: its logprob is the decision row's
    assert labels.trie_levels([(2,)]) == []
    # two paths sharing their first token decode it once, at level 1
    assert labels.trie_levels([(2, 3), (2, 4)]) == [[(2,)]]
    # level k holds every prefix of length k that a *longer* path passes through: (2, 4) is a
    # leaf (its token 4 comes from the row after (2,)) and is never decoded
    assert labels.trie_levels([(2, 3, 9), (2, 3, 7), (2, 4)]) == [[(2,)], [(2, 3)]]
    assert labels.trie_nodes([(2, 3, 9), (2, 3, 7), (2, 4)]) == 2


def test_score_paths_decodes_the_shared_prefix_once_at_the_engines_positions():
    """The bug this gate is for: the parent of a level-`k` prefix is the prefix of length `k-1`."""
    row_a = exact_row({1: 3.0}, width=10)           # p(1) = 3/7
    row_b = exact_row({3: 1.0}, width=10)           # p(3) = 1/7
    session = FakeRowSession({10: row_a, 11: row_b})
    got = labels.score_paths(session, head_seq=1, base=10, pool=[7, 8, 9],
                             paths=[(2, 1, 3), (2, 1, 0), (2, 3, 9), (2, 3)])
    # level 1: one child prefix (2,) — decoded once, forked from the head sequence
    # level 2: (2, 1) and (2, 3) — both forked from the (2,) sequence, in one batch
    assert session.forks == [(1, 7, 10), (7, 8, 11), (7, 9, 11)]
    assert [(batch.tokens, batch.positions) for batch in session.batches] == [
        ((2,), (10,)), ((1, 3), (11, 11))]
    scale_a, scale_b = readout.logsumexp(row_a), readout.logsumexp(row_b)
    assert got[(2, 1, 3)] == [readout.logprob_from_scale(row_a, 1, scale_a),
                              readout.logprob_from_scale(row_b, 3, scale_b)]
    assert got[(2, 1, 0)] == [readout.logprob_from_scale(row_a, 1, scale_a),
                              readout.logprob_from_scale(row_b, 0, scale_b)]
    assert got[(2, 3, 9)] == [readout.logprob_from_scale(row_a, 3, scale_a),
                              readout.logprob_from_scale(row_b, 9, scale_b)]
    # a two-token path is a leaf one level up: its last token is scored by the level-1 row
    assert got[(2, 3)] == [readout.logprob_from_scale(row_a, 3, scale_a)]
    assert list(got) == [(2, 1, 3), (2, 1, 0), (2, 3, 9), (2, 3)]      # keyed by tuple(path)


def test_score_paths_needs_no_decode_for_single_token_labels():
    session = FakeRowSession({})
    assert labels.score_paths(session, head_seq=1, base=10, pool=[], paths=[(2,), (3,)]) == \
        {(2,): [], (3,): []}
    assert session.forks == [] and session.batches == []


def test_an_exhausted_pool_is_an_error_not_a_silent_reuse():
    """A reused sequence would read another candidate's state — the pool is the caller's job."""
    session = FakeRowSession({10: exact_row({1: 1.0})})
    with pytest.raises(labels.LabelsError) as info:
        labels.score_paths(session, head_seq=1, base=10, pool=[], paths=[(2, 1), (2, 3)])
    assert "E_LABEL_POOL" in str(info.value)
    assert session.forks == []
