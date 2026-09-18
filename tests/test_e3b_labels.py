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

import pytest

from ggufone.bench import labels
from ggufone.engine import prompt, readout
from ggufone.schema import parse_request

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
