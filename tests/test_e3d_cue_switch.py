"""E3d: the `--cue` switch — three cue shapes behind one frozen default (card t_d90404ac).

E3c (`t_6c119626`) found seven cue shapes on 6 dev items; E3d re-measured three of them on the
full 60-item dev set (`tools/e3d_cue_decision.py`, `docs/evidence/e3d_cue_decision_4b.md`). These
gates pin the *mechanism* that measurement asks for, not the numbers:

* **`shipped`** (the frozen default): the label is read at the row the suffix ends on. The bytes,
  the payload and the warnings of a `shipped` request may not move at all — every published table
  was measured with this shape.
* **`two_step`**: *the same prompt bytes* plus one decoded token. The label is read one row later,
  at the model's own first content token — and **only** when that token is not a turn-closer, so a
  model that refuses the cue keeps the refusal verdict E3c built (`W_CUE_REFUSED`), and a refused
  request never advances. This is the shape the 60-item run measures as the honest one: 44/60
  `low_mass` at the cue drops to 2/60.
* **`json_field`**: the shipped cue line plus the per-type JSON opener (`{"choice": "`), read at
  the field row. It is the agreement winner of the 60-item run and it is *not* the default: the
  opener is part of the prompt, so the model can no longer refuse the cue where E3c reads it.

The gates below fail on the pre-E3d tree: `options.cue` does not exist, `build_question` has no
`cue` argument, `DecisionEngine` reads every label at the cue row, and the bench cannot ask for a
shape at all.

The fake session is `tests/fake_engine.FakeSession` — a word-level tokenizer that emulates a real
special-token vocabulary for the three closers, so `engine/cue.py`'s catalogue matches exactly the
tokens a real model's vocabulary encodes as one token.
"""
from __future__ import annotations

import pytest

from tests.fake_engine import BenchModel, FakeSession, biased_row
from typed_gguf import cli, errors, schema
from typed_gguf.bench import harness, suites
from typed_gguf.engine import cue as cue_module
from typed_gguf.engine import decide, prompt

#: the bias that puts ~1.0 of a 512-slot row's mass on one token
TOP = 11.0
#: the ids this fake vocabulary reserves for the closers it encodes as ONE token
IM_END = 400
EOS = 401


class CloserSession(FakeSession):
    """`FakeSession` with a real special-token vocabulary (the E3c refusal gates' fake)."""

    SPECIAL = {"<|im_end|>": IM_END, "</s>": EOS, "<|endoftext|>": EOS + 1}

    def tokenize(self, text: str) -> list[int]:
        if text in self.SPECIAL:
            return [self.SPECIAL[text]]
        return super().tokenize(text)


def choice_request(cue: str | None = None, *, qtype: str = "choice",
                   chat_format: str | None = None) -> dict:
    """One dev-set-shaped request; `cue` is left out entirely unless a test names one.

    After policy v2 (card t_5b754458) a request that names nothing is the instructed/role-split
    cell; this file's byte and readout pins are the *shipped* ones, so they name `cue` (and, where
    the placement matters, `chat_format=answer_sheet`).
    """
    bodies = {
        "choice": ("area", {"type": "choice", "instructions": "Which area owns this?",
                            "criteria": {"billing": "payments and invoices",
                                         "technical": "api and infrastructure"}}),
        "score": ("sev", {"type": "score", "instructions": "How bad?",
                          "criteria": ["cosmetic", "annoying", "blocking"]}),
        "noul": ("page", {"type": "noul", "instructions": "Should we page?",
                          "criteria": {"true": "page now", "false": "wait"}}),
    }
    qid, body = bodies[qtype]
    payload: dict = {"state": "The billing dashboard is blank for every user after login.",
                     "questions": {qid: body}}
    options: dict = {}
    if cue is not None:
        options["cue"] = cue
    if chat_format is not None:
        options["chat_format"] = chat_format
    if options:
        payload["options"] = options
    return payload


def question_of(request: schema.Request) -> schema.Question:
    """The single question of a one-question request (the id differs per type)."""
    return request.questions[0]


def scripted(request: schema.Request, *, cue: str, cue_text: str | None = "unrelated",
             label: str = "billing"):
    """A fake row function: the *cue row* prefers `cue_text`, every later row prefers `label`.

    `cue_text=None` means "no cue-row special case": every row prefers the label, which is the
    shape a `json_field` prompt sees (the opener is part of the suffix, so its field row *is* the
    prompt's last row).
    """
    session = CloserSession(n_vocab=512)
    plan = decide.plan_context(request, session, resolve=False)
    view = prompt.build_question(request.questions[0], cue=cue)
    n_suffix = len(session.tokenize(view.suffix))
    cue_position = len(plan.prefix_tokens) + n_suffix - 1
    label_token = session.tokenize(label)[0]
    cue_token = None if cue_text is None else session.tokenize(cue_text)[0]

    def row_fn(context) -> list[float]:
        if cue_token is not None and context.position == cue_position:
            return biased_row(session.n_vocab, {cue_token: TOP})
        return biased_row(session.n_vocab, {label_token: TOP})

    session.row_fn = row_fn
    return session, plan, {"n_suffix": n_suffix, "cue_position": cue_position,
                           "label_token": label_token, "cue_token": cue_token}


# ------------------------------------------------------------------ the frozen enumeration
def test_the_default_cue_is_the_instructed_one() -> None:
    """Policy v2 (card t_5b754458) moved the default; the enumeration itself did not move."""
    assert schema.DEFAULT_CUE == "json_instructed"
    assert schema.OPTION_DEFAULTS["cue"] == schema.DEFAULT_CUE
    assert schema.Options().cue == schema.DEFAULT_CUE
    # E3e (card t_4c48f40a) grew the enumeration by one: `json_instructed` says the answer is a
    # JSON object instead of a bare label, which needed the question *and* the assistant turn to
    # change (see tests/test_e3e_roles.py). E3d's `shipped` — the shape this file's bytes were
    # measured on — is `CUE_SHAPES[0]` and stays reachable by `--cue shipped`.
    assert schema.CUE_SHAPES == ("shipped", "two_step", "json_field", "json_instructed")


def test_an_unknown_cue_is_a_named_option_error() -> None:
    with pytest.raises(errors.UserError) as caught:
        schema.parse_request(choice_request("letters"))
    assert caught.value.code == "E_UNKNOWN_KEY"
    assert "cue" in str(caught.value)


def test_an_explicit_cue_survives_parsing() -> None:
    request = schema.parse_request(choice_request("two_step"))
    assert request.options.cue == "two_step"


def test_build_question_refuses_an_unknown_cue() -> None:
    request = schema.parse_request(choice_request())
    with pytest.raises(errors.UserError) as caught:
        prompt.build_question(request.questions[0], cue="letters")
    assert caught.value.code == "E_UNKNOWN_KEY"


# ------------------------------------------------------------------ the prompt bytes
def test_two_step_changes_no_prompt_byte() -> None:
    """The whole point of the two-step shape: the prompt is the shipped prompt."""
    request = schema.parse_request(choice_request())
    question = request.questions[0]
    shipped = prompt.build_question(question)
    two_step = prompt.build_question(question, cue="two_step")
    assert two_step.suffix == shipped.suffix
    assert two_step.texts == shipped.texts


def test_every_cue_reads_the_same_labels() -> None:
    request = schema.parse_request(choice_request())
    question = request.questions[0]
    labels = [prompt.build_question(question, cue=cue).texts for cue in schema.CUE_SHAPES]
    assert labels == [question.options] * len(schema.CUE_SHAPES)


@pytest.mark.parametrize(("qtype", "opener"), [("choice", '{"choice": "'),
                                               ("score", '{"severity": "'),
                                               ("noul", '{"answer": "')])
def test_json_field_appends_the_per_type_opener(qtype: str, opener: str) -> None:
    request = schema.parse_request(choice_request(qtype=qtype))
    question = request.questions[0]
    shipped = prompt.build_question(question).suffix
    field = prompt.build_question(question, cue="json_field").suffix
    assert field == shipped + opener


# ------------------------------------------------------------------ the engine readout
def test_two_step_reads_the_label_at_the_advanced_row() -> None:
    request = schema.parse_request(choice_request("two_step"))
    session, plan, ids = scripted(request, cue="two_step")
    result = decide.DecisionEngine(session).decide(request, plan=plan)
    answer = result.answers["area"]
    assert answer["choice"] == "billing"
    # the cue row carries the model's own first content token, the row after it the label
    assert answer["coverage"] > 0.9
    assert answer["reliability"] == "ok"
    assert answer["advance"]["token"] == ids["cue_token"]
    assert answer["advance"]["rule"] == cue_module.ADVANCE_RULE
    assert answer["advance"]["cue"]["refused"] is False
    assert result.engine["cue"] == "two_step"


def test_the_shipped_cue_reads_the_cue_row_itself() -> None:
    """The control: the same script, one shape over — and the pre-E3d payload, unchanged."""
    request = schema.parse_request(choice_request("shipped"))
    session, plan, _ = scripted(request, cue="shipped")
    result = decide.DecisionEngine(session).decide(request, plan=plan)
    answer = result.answers["area"]
    assert answer["coverage"] < 0.1
    assert answer["reliability"] == "low_mass"
    assert "advance" not in answer
    assert "W_LOW_MASS" in result.warnings
    assert result.engine["cue"] == "shipped"


def test_a_two_step_request_never_advances_past_a_refusal() -> None:
    """Occamy's verdict survives: a cue the model closes stays a refusal (card constraint 4)."""
    request = schema.parse_request(choice_request("two_step"))
    session, plan, _ = scripted(request, cue="two_step", cue_text="<|im_end|>")
    result = decide.DecisionEngine(session).decide(request, plan=plan)
    answer = result.answers["area"]
    assert "advance" not in answer
    assert answer["cue"]["refused"] is True
    assert answer["cue"]["closer"] == "<|im_end|>"
    assert "W_CUE_REFUSED" in result.warnings


def test_json_field_reads_the_field_row_without_advancing() -> None:
    request = schema.parse_request(choice_request("json_field"))
    session, plan, ids = scripted(request, cue="json_field", cue_text=None)
    result = decide.DecisionEngine(session).decide(request, plan=plan)
    answer = result.answers["area"]
    view = prompt.build_question(request.questions[0], cue="json_field")
    assert view.suffix.endswith('{"choice": "')
    assert answer["choice"] == "billing"
    assert answer["coverage"] > 0.9
    assert "advance" not in answer
    assert result.engine["cue"] == "json_field"


def test_the_pre_v2_request_publishes_no_advance_key() -> None:
    """The `shipped`/`answer_sheet` cell's payload is the published one, key for key.

    It is the *pre-v2* cell now (policy v2, card t_5b754458), so the request spells it out; a
    request that names nothing is the instructed/role-split cell (tests/test_policy_v2.py).
    """
    request = schema.parse_request(choice_request("shipped", chat_format="answer_sheet"))
    session, plan, _ = scripted(request, cue="shipped", cue_text=None)
    result = decide.DecisionEngine(session).decide(request, plan=plan)
    assert set(result.answers["area"]) == {
        "type", "choice", "probabilities", "confidence", "coverage", "reliability", "cue",
        "decode_steps", "legend"}


# ------------------------------------------------------------------ the bench and the CLI
def test_the_bench_carries_the_cue_into_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    from typed_gguf.bench import devset as devset_module

    seen: dict = {}
    original = devset_module.request_for

    def spy(item, *, model: str, **options):
        seen.update(options)
        return original(item, model=model, **options)

    monkeypatch.setattr(suites.devset_module, "request_for", spy)
    items = devset_module.load(str(suites.devset_module.devset_path()))
    config = harness.BenchConfig(suite="quality", model_path="/tmp/fake.gguf", cue="two_step",
                                 threads=1)
    model = BenchModel(harness.ModelSpec(path="/tmp/fake.gguf", backend="cpu", threads=1))
    suites._devset_row(config, model, items[0])
    assert seen.get("cue") == "two_step"


def test_an_unknown_cue_on_the_bench_cli_is_a_named_usage_error() -> None:
    with pytest.raises(errors.UserError) as caught:
        cli._bench_cue("letters")
    assert caught.value.code == "E_BENCH_USAGE"


def test_bench_config_default_is_the_instructed_cue() -> None:
    """Policy v2 (card t_5b754458): the bench's own default is the measured-good cell."""
    assert harness.BenchConfig(suite="quality").cue == schema.DEFAULT_CUE
    assert schema.CUE_SHAPES[0] == "shipped"          # the pre-v2 switch is still in the set


def test_the_cli_passes_cue_through(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """`--cue` reaches the request (the same seam `--readout` uses) and is visible in the
    `engine` block — a knob that does not appear in the response is a silent knob."""
    import pathlib

    monkeypatch.setattr(cli, "decide_payload_warm", _fake_decide)  # the E4 seam
    monkeypatch.setenv("TYPED_GGUF_HOME", str(tmp_path / "home"))
    pathlib.Path(str(tmp_path / "home")).mkdir(parents=True, exist_ok=True)
    code = cli.main(["ask", "--state", "S", "--cue", "two_step",
                     "--choice", "area=Which?:billing|api"])
    assert code == 0


def test_the_cli_rejects_an_unknown_cue(monkeypatch: pytest.MonkeyPatch, tmp_path,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    import pathlib

    monkeypatch.setattr(cli, "decide_payload_warm", _fake_decide)  # the E4 seam
    monkeypatch.setenv("TYPED_GGUF_HOME", str(tmp_path / "home2"))
    pathlib.Path(str(tmp_path / "home2")).mkdir(parents=True, exist_ok=True)
    code = cli.main(["ask", "--state", "S", "--cue", "letters",
                     "--choice", "area=Which?:billing|api"])
    assert code == 2
    assert "E_UNKNOWN_KEY" in capsys.readouterr().err


def _fake_decide(payload: dict, *, home=None, **kwargs) -> dict:
    """`cli.decide_payload` over the fake session: same validation, same rendering."""
    request = schema.parse_request(payload)
    session = CloserSession(n_vocab=512)
    billing = session.tokenize("billing")[0]
    session.row_fn = (lambda ctx, session=session, billing=billing:
                      biased_row(session.n_vocab, {billing: TOP}))
    result = decide.decide_request(request, session)
    return schema.render_response(result.payload(), format=request.format)
