"""Policy v2 (card t_5b754458): `json_instructed` + `role_split` are the product **defaults**.

The decision. E3e (`t_4c48f40a`, `docs/evidence/e3e_roles_decision.md`) measured eight policy arms
on the committed 60 items; its audit (`t_57bd3db2`) corrected the freeze line, and the [host]
probes (`t_9bcbecff`) asked the same question of the two 35B models the operator runs. The numbers
that decided it: the **old** defaults — `answer_sheet` + the `shipped` cue — collapse on both 35B
models (Tiel 22/60, Occamy 26/60 under the corrected instrument), while the instructed/role-split
cell reads 53/60 and 54/60 on the very same items; on the 4B that cell is 50/60, inside the noise
of the table's best (51/60). A public user only ever meets the defaults, so the measured-good
policy becomes the default (option A, user decision 2026-09-20).

What this file pins:

* the flip itself — `schema.OPTION_DEFAULTS`, `Options()`, the bench config, the CLI's bench
  defaults and `tools/e2_reproduce.py`'s flags all name the v2 cell;
* the **default-parity pin**: a request that names *nothing* renders byte-identical bytes (prefix
  + question tail + JSON opener, and the same token ids) to the request that names the v2 cell
  explicitly — offline through both assemblies (plain, and a family chat template);
* the pre-v2 cells stay reachable by exactly the flags the E3d/E3e tables were measured with, and
  their bytes did not move;
* the bench names the policy it measured whenever it is **not** the default (the E3e comparability
  rule: a row may never sit next to another from a different policy generation without a marker).

The live gate at the bottom re-measures the *default* recipe on the 4B and compares it, item for
item, with the published `json_instructed/role_split` arm (`.e3e/bench_json_instructed_role_split
.json`, `--backend vulkan`, 60 items). It is `@pytest.mark.model`: skipped unless `--run-network`.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from tests.fake_engine import FakeSession, biased_row
from typed_gguf import cli, schema
from typed_gguf.bench import harness, suites
from typed_gguf.engine import decide, prompt
from typed_gguf.engine import template as template_module

ROOT = pathlib.Path(__file__).resolve().parents[1]
#: the published v2 arm of the E3e table (`.e3e/run_arms.sh`): the cell these defaults now name
ARM = ROOT / ".e3e" / "bench_json_instructed_role_split.json"
#: the 4B every published 4B row was measured on (the live gate's model, `--run-network` only)
MODEL = pathlib.Path.home() / ".hermes" / "models" / "Spark-X2.5-4B-Q8_0.gguf"

STATE = "The billing dashboard is blank for every user after login."
OPENER = '{"choice": "'
#: the v2 cell, spelled out: what the card's probe called with flags (and what no-flags now means)
V2 = {"cue": "json_instructed", "chat_format": "role_split", "json_contract": "question"}
#: the pre-v2 cell: the defaults every published E3d/E3e row was measured under
PRE_V2 = {"cue": "shipped", "chat_format": "answer_sheet"}
#: the shipped question block, frozen since E3d (the same literal `tests/test_e3e_roles.py` pins)
SHIPPED_BLOCK = ("QUESTION:\nWhich area owns this?\nCandidates:\n- billing: payments and invoices\n"
                 "- technical: api and infrastructure\nAnswer with exactly one candidate name:\n")
#: A ChatML-shaped template: a turn per message, the generation prompt only when asked for.
CHATML = ("{% for message in messages %}"
          "{{ '<|im_start|>' + message['role'] + '\\n' + message['content']"
          " + '<|im_end|>' + '\\n' }}"
          "{% endfor %}"
          "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")
#: the ids this fake vocabulary reserves for the JSON punctuation it encodes as ONE token
QUOTE = 403
IM_END = 400
TOP = 11.0


class JsonSession(FakeSession):
    """`FakeSession` with a real special-token vocabulary: closers and the JSON quote."""

    SPECIAL = {"<|im_end|>": IM_END, "</s>": 401, "<|endoftext|>": 402, '"': QUOTE}

    def tokenize(self, text: str) -> list[int]:
        if text in self.SPECIAL:
            return [self.SPECIAL[text]]
        return super().tokenize(text)


def body(**options: str) -> dict:
    """One dev-set-shaped request; `options` is left out entirely unless a test names one."""
    payload: dict = {"state": STATE,
                     "questions": {"area": {"type": "choice",
                                            "instructions": "Which area owns this?",
                                            "criteria": {"billing": "payments and invoices",
                                                         "technical": "api and infrastructure"}}}}
    if options:
        payload["options"] = dict(options)
    return payload


def parsed(**options: str) -> schema.Request:
    return schema.parse_request(body(**options))


def resolution() -> template_module.Resolution:
    return template_module.Resolution(kind="gguf-renderer", renderer="internal", source="gguf",
                                      template=CHATML, family="chatml-test",
                                      thinking="suppressed")


def rendered(request: schema.Request, *, chat: bool = False) -> dict:
    """The executed prompt of a one-question request: the bytes and the plan's own token ids.

    `chat=True` renders through the ChatML family template (the path a real model takes);
    `chat=False` is the plain assembly. Both halves — the shared prefix and the question's own
    suffix — come from the plan's render, exactly as `decide.question_requirements` does it.
    """
    session = FakeSession(n_vocab=512)
    render = resolution() if chat else None
    plan = decide.plan_context(request, session, template=render, resolve=False)
    view = prompt.build_question(request.questions[0], readout=request.options.readout,
                                cue=request.options.cue, chat_format=request.options.chat_format,
                                role=plan.role, index=0, contract=request.options.json_contract)
    prefix = prompt.build_prefix(request.state, resolution=render,
                                 enable_thinking=request.options.thinking,
                                 chat_format=request.options.chat_format, cue=request.options.cue,
                                 questions=request.questions, role=plan.role,
                                 contract=request.options.json_contract)
    return {"prefix": prefix, "suffix": view.suffix, "tokens": tuple(plan.prefix_tokens),
            "role": plan.role, "plan": plan, "view": view}


# ============================================================ the flip (schema + the surfaces)
def test_the_default_cue_is_the_instructed_one() -> None:
    assert schema.DEFAULT_CUE == "json_instructed"
    assert schema.OPTION_DEFAULTS["cue"] == schema.DEFAULT_CUE
    assert schema.Options().cue == schema.DEFAULT_CUE
    assert parsed().options.cue == schema.DEFAULT_CUE
    # the enumeration did not move: every E3d/E3e cell is still expressible, and the pre-v2 shape
    # is still in it (`CUE_SHAPES[0]`, the cell the E3d tables were measured under)
    assert schema.CUE_SHAPES == ("shipped", "two_step", "json_field", "json_instructed")


def test_the_default_chat_format_is_the_role_split() -> None:
    assert schema.DEFAULT_CHAT_FORMAT == schema.ROLE_SPLIT
    assert schema.OPTION_DEFAULTS["chat_format"] == schema.DEFAULT_CHAT_FORMAT
    assert schema.Options().chat_format == schema.DEFAULT_CHAT_FORMAT
    assert parsed().options.chat_format == schema.DEFAULT_CHAT_FORMAT
    assert schema.CHAT_FORMATS == ("answer_sheet", "role_split")


def test_the_default_contract_is_still_the_question_one() -> None:
    """The contract was already the inline one; policy v2 keeps it and now *uses* it."""
    assert schema.JSON_CONTRACT == "question"
    assert schema.OPTION_DEFAULTS["json_contract"] == "question"
    assert parsed().options.json_contract == "question"


def test_the_bench_and_the_cli_ask_for_the_v2_cell_by_default() -> None:
    """The bench's own literals (it never imports `schema`) must be the schema's defaults."""
    assert harness.DEFAULT_CUE == schema.DEFAULT_CUE
    assert harness.DEFAULT_CHAT_FORMAT == schema.DEFAULT_CHAT_FORMAT
    assert harness.DEFAULT_JSON_CONTRACT == schema.JSON_CONTRACT
    config = harness.BenchConfig(suite="quality")
    assert (config.cue, config.chat_format, config.json_contract) == (
        schema.DEFAULT_CUE, schema.DEFAULT_CHAT_FORMAT, schema.JSON_CONTRACT)
    assert cli.BENCH_DEFAULTS["cue"] == schema.DEFAULT_CUE
    assert cli.BENCH_DEFAULTS["chat-format"] == schema.DEFAULT_CHAT_FORMAT
    assert cli.BENCH_DEFAULTS["json-contract"] == schema.JSON_CONTRACT
    # the CLI's validators answer the default when no flag is given, and the old cell when asked
    assert cli._bench_cue(None) == schema.DEFAULT_CUE
    assert cli._bench_chat_format(None) == schema.DEFAULT_CHAT_FORMAT
    assert cli._bench_json_contract(None) == schema.JSON_CONTRACT
    assert cli._bench_cue("shipped") == "shipped"
    assert cli._bench_chat_format("answer_sheet") == "answer_sheet"


def test_the_reproduce_tool_defaults_to_the_v2_cell() -> None:
    """The tool that regenerates every published row: no flags means the product's own policy."""
    import importlib.util

    path = ROOT / "tools" / "e2_reproduce.py"
    spec = importlib.util.spec_from_file_location("e2_reproduce_policy_v2", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.make_parser().parse_args(["--suite", "quality", "--model", "m.gguf"])
    assert args.cue == schema.DEFAULT_CUE
    assert args.chat_format == schema.DEFAULT_CHAT_FORMAT
    assert args.json_contract == schema.JSON_CONTRACT
    old = module.make_parser().parse_args(["--suite", "quality", "--model", "m.gguf",
                                           "--cue", "shipped", "--chat-format", "answer_sheet"])
    assert (old.cue, old.chat_format) == ("shipped", "answer_sheet")


# ================================================================== the default-parity pin
def test_a_no_flags_request_renders_the_v2_cell_byte_for_byte_plain() -> None:
    """The card's parity pin, plain assembly: no options == the explicitly flagged v2 cell."""
    default = rendered(parsed())
    flagged = rendered(parsed(**V2))
    assert default["prefix"] == flagged["prefix"]
    assert default["suffix"] == flagged["suffix"]
    assert default["tokens"] == flagged["tokens"]
    # ... and it is the *instructed* shape: the contract is in the question turn, the opener ends
    # the suffix, and the framing asks for the JSON object the contract names
    assert default["suffix"].endswith(OPENER)
    assert prompt.JSON_CONTRACT["choice"] in default["suffix"]
    assert prompt.CANDIDATE_CUE["choice"] not in default["suffix"]
    assert prompt.JSON_FRAMING in default["prefix"]
    assert "QUESTION:" not in default["prefix"]             # the question is its own user turn
    assert default["role"] is not None and default["role"].prefix == default["prefix"]
    assert default["role"].tails[0].startswith(f"{prompt.PLAIN_USER_HEADER}QUESTION:\n")
    assert default["role"].tails[0].endswith(f"{prompt.PLAIN_ASSISTANT_HEADER}{OPENER}")


def test_a_no_flags_request_renders_the_v2_cell_byte_for_byte_chat_template() -> None:
    """The same pin through a family chat template: prefix + tail + opener, and the token ids."""
    default = rendered(parsed(), chat=True)
    flagged = rendered(parsed(**V2), chat=True)
    assert default["prefix"] == flagged["prefix"]
    assert default["tokens"] == flagged["tokens"]
    assert default["role"].tails == flagged["role"].tails
    assert default["suffix"] == flagged["suffix"]
    # the tail is the question's own user turn followed by the generation prompt + the opened field
    assert default["role"].tails[0].endswith(f"<|im_start|>assistant\n{OPENER}")
    assert "QUESTION:" not in default["role"].prefix        # ... never prefilled into the prefix
    # the flagged request *is* the request the card's probe called (`--chat-format role_split
    # --cue json_instructed --json-contract question`): the two plans agree key for key
    assert default["plan"].chat_format == flagged["plan"].chat_format == schema.ROLE_SPLIT
    assert default["plan"].json_contract == flagged["plan"].json_contract == "question"


def test_the_no_flags_engine_publishes_the_v2_policy_and_the_value_row_verdict() -> None:
    """The engine surface: the response names the v2 policy and reads the instructed value row."""
    request = parsed()
    session = JsonSession(n_vocab=512)
    plan = decide.plan_context(request, session, template=resolution(), resolve=False)
    view = prompt.build_question(request.questions[0], readout=request.options.readout,
                                cue=request.options.cue, chat_format=request.options.chat_format,
                                role=plan.role, index=0, contract=request.options.json_contract)
    value_position = len(plan.prefix_tokens) + len(session.tokenize(view.suffix)) - 1
    label = session.tokenize("billing")[0]
    session.row_fn = (lambda ctx, position=value_position, label=label:
                      biased_row(session.n_vocab, {label: TOP})
                      if ctx.position == position
                      else biased_row(session.n_vocab, {label: TOP}))
    result = decide.DecisionEngine(session).decide(request, plan=plan)
    assert result.engine["cue"] == schema.DEFAULT_CUE
    assert result.engine["chat_format"]["kind"] == schema.ROLE_SPLIT
    assert result.engine["chat_format"]["question_turn"] == "user"
    assert result.engine["chat_format"]["contract"] == schema.JSON_CONTRACT   # in the question turn
    answer = result.answers["area"]
    assert answer["choice"] == "billing"
    assert answer["cue"]["verdict"] == "answered"           # the value row, not the shipped row
    assert "verdict" not in result.answers["area"].get("advance", {})


# ============================================================== the pre-v2 cells by flag
def test_the_pre_v2_cell_stays_reachable_and_its_bytes_are_frozen() -> None:
    """`--cue shipped --chat-format answer_sheet`: the E3d/E3e tables stay reproducible."""
    request = parsed(**PRE_V2)
    assert (request.options.cue, request.options.chat_format) == ("shipped", "answer_sheet")
    plain = rendered(request)
    assert plain["role"] is None                            # the answer sheet is not a role split
    assert plain["suffix"] == SHIPPED_BLOCK                 # no opener: the bare-label shape
    assert prompt.question_block(request.questions[0], cue="shipped") == SHIPPED_BLOCK
    chat = rendered(request, chat=True)
    assert chat["suffix"] == SHIPPED_BLOCK                  # the question block, no JSON anything
    assert prompt.SYSTEM_FRAMING.startswith("You are a decision engine")
    assert prompt.framing_for("shipped") == prompt.SYSTEM_FRAMING


def test_every_other_switch_still_parses_and_renders() -> None:
    """The E3d/E3e switch surface: every cell of both tables is still expressible by flags."""
    for cue in ("shipped", "two_step", "json_field", "json_instructed"):
        assert parsed(**{**PRE_V2, "cue": cue}).options.cue == cue
    for chat_format in ("answer_sheet", "role_split"):
        request = parsed(cue="json_instructed", chat_format=chat_format)
        assert request.options.chat_format == chat_format
    for contract in ("question", "system"):
        request = parsed(cue="json_instructed", chat_format="role_split", json_contract=contract)
        plan = decide.plan_context(request, FakeSession(n_vocab=512), template=resolution(),
                                   resolve=False)
        assert plan.json_contract == contract
        assert plan.role is not None
        assert (prompt.JSON_SYSTEM_FRAMING if contract == "system"
                else prompt.JSON_FRAMING).strip() in plan.role.prefix


def test_the_reproduce_line_names_the_pre_v2_cell_and_stays_silent_for_v2() -> None:
    """A row that is *not* the default must name its policy; the default row must not (E3e rule)."""
    default = harness.reproduce_command(harness.BenchConfig(suite="quality", model_path="/m.gguf",
                                                            backend="vulkan", runs=1, threads=4,
                                                            items=60))
    assert "--cue" not in default and "--chat-format" not in default
    pre_v2 = harness.reproduce_command(
        harness.BenchConfig(suite="quality", model_path="/m.gguf", backend="vulkan", runs=1,
                            threads=4, items=60, cue="shipped", chat_format="answer_sheet"))
    assert "--cue shipped" in pre_v2
    assert "--chat-format answer_sheet" in pre_v2
    system = harness.reproduce_command(
        harness.BenchConfig(suite="quality", model_path="/m.gguf", backend="vulkan", runs=1,
                            threads=4, items=60, json_contract="system"))
    assert "--json-contract system" in system


def test_the_report_marks_the_pre_v2_policy_and_not_the_default() -> None:
    report = {"suite": "quality", "generated_at": "2026-09-20T00:00:00Z",
              "config": {"backend": "vulkan", "runs": 1, "threads": 4,
                         "cue": "shipped", "chat_format": "answer_sheet"},
              "model": {"path": "/m.gguf"}, "overall": {}, "per_type": {},
              "commands": {"reproduce": "uv run typed-gguf bench --suite quality"}}
    text = harness.render_report(report)
    assert "- prompt policy: cue=shipped · chat_format=answer_sheet" in text
    report["config"].update(V2)
    assert "- prompt policy:" not in harness.render_report(report)


# ================================================================== the published documents
def test_the_documents_state_the_v2_defaults() -> None:
    """The flip is a *policy* change: the two documents a reader uses must say so."""
    benchmarks = (ROOT / "docs" / "BENCHMARKS.md").read_text(encoding="utf-8")
    templates = (ROOT / "docs" / "TEMPLATES.md").read_text(encoding="utf-8")
    for text in (benchmarks, templates):
        assert "policy v2" in text, "the policy-v2 marker is missing from a published document"
        assert "json_instructed" in text and "role_split" in text
    # §2 carries the re-measured default row; §9 the decision that flipped the defaults
    assert "2.3" in benchmarks and "defaults" in benchmarks
    section = benchmarks.split("## 9. E3e", 1)[1]
    assert "default" in section, "§9 no longer says which shape the defaults are"


# ================================================================== live: the 4B, item for item
def _runtime_dir() -> pathlib.Path:
    import os

    from typed_gguf.runtime import finder
    env = os.environ.get("TYPED_GGUF_RUNTIME_DIR")
    if env and (pathlib.Path(env) / "libllama.so").exists():
        return pathlib.Path(env)
    found = finder.find_runtime()
    if found:
        return found
    pytest.skip("no llama.cpp runtime on this box (set TYPED_GGUF_RUNTIME_DIR)")


@pytest.mark.model
def test_the_defaults_measure_the_published_v2_cell_on_the_4b(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The live half of the parity pin: the default recipe *is* the published v2 arm.

    The arm was measured by `.e3e/run_arms.sh` with `--backend vulkan --threads 4 --items 60` and
    the v2 flags spelled out; this test runs the same recipe with **no policy flags at all** and
    compares the rows with the published ones — prompt tokens, decision, and the cue verdict.
    """
    if not MODEL.is_file():
        pytest.skip(f"{MODEL} is not on this box")
    runtime = _runtime_dir()
    monkeypatch.setenv("TYPED_GGUF_RUNTIME_DIR", str(runtime))
    arm = json.loads(ARM.read_text(encoding="utf-8"))
    published = {row["id"]: row for row in arm["items"]}
    assert (arm["config"]["cue"], arm["config"]["chat_format"]) == ("json_instructed",
                                                                    "role_split")
    items = 6
    config = harness.BenchConfig(suite="quality", model_path=str(MODEL), backend="vulkan", runs=1,
                                 threads=4, items=items)
    report = suites.run_suite(config, factory=suites.live_factory)
    assert report["ok"] is True, report.get("warnings")
    assert report["config"]["cue"] == schema.DEFAULT_CUE
    assert report["config"]["chat_format"] == schema.DEFAULT_CHAT_FORMAT
    assert len(report["items"]) == items
    for row in report["items"]:
        expected = published[row["id"]]
        assert row["prefix_tokens"] == expected["prefix_tokens"], row["id"]
        assert row["got"] == expected["got"], row["id"]
        assert bool(row["correct"]) is bool(expected["correct"]), row["id"]
        assert row["cue"]["verdict"] == expected["cue"]["verdict"], row["id"]
        assert row["cue"]["refused"] is False, row["id"]
