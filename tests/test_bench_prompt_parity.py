"""Prompt parity between the instrument and the product (card t_6de5fc53).

The bench must execute the plan the *serving path* would execute for the same request and the
same model. `cli.ask`/`run` plan the context from the **model handle**, so the bench must too.

Before this card `bench/harness.py::LiveModel.decide` planned the context twice: once from the
handle (used only to size the session) and once from the live `ModelSession` — and the second
plan is the one that ran. A `ModelSession` carries no `.model`/`.runtime`, so
`decide.resolve_template` returned `None` and `prompt.build_prefix(state, None)` silently fell
back to the plain E1b framing: the bench measured a prompt the product never sends (on dev item
`c01` / Spark-X2.5-4B that is 102 prefix tokens of plain framing against 119 of chat template).

The offline gate below pins the *structure* (one plan, resolved from the handle, never re-planned
from the session); the live gate in `tests/test_bench_live.py`
(`test_the_bench_sends_the_same_prompt_as_the_serving_path`) pins the bytes on a real model.
"""
from __future__ import annotations

import pytest

from typed_gguf import schema
from typed_gguf.bench import devset as devset_module
from typed_gguf.bench import harness
from typed_gguf.engine import decide as decide_module
from typed_gguf.engine import session as session_module

HANDLE_TOKENS = (101, 102, 103)
SESSION_TOKENS = (1, 2)


class _StubSession:
    """A live session as a planner sees a `ModelSession`: no `.model`, no `.runtime`.

    That is exactly the hole the bug fell through — `resolve_template(request, session)` returns
    `None` for this object, so any plan built from it is the plain E1b framing.
    """

    def __enter__(self) -> _StubSession:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _plan(tokens: tuple[int, ...]) -> decide_module.ContextPlan:
    return decide_module.ContextPlan(prefix_tokens=tokens, n_ctx=256, n_seq_max=3, threads=1,
                                     kv_type="f16")


def test_the_executed_plan_is_resolved_from_the_handle(monkeypatch: pytest.MonkeyPatch,
                                                       tmp_path) -> None:
    """ONE plan: the executed plan is the handle's, and the session is never asked to plan."""
    handle = object()
    planned_on: list[object] = []

    def fake_plan_context(request: object, tokenizer: object, **kwargs: object
                          ) -> decide_module.ContextPlan:
        planned_on.append(tokenizer)
        return _plan(HANDLE_TOKENS) if tokenizer is handle else _plan(SESSION_TOKENS)

    executed: list[object] = []

    class StubEngine:
        def __init__(self, session: object, **kwargs: object) -> None:
            self.session = session

        def decide(self, request: object, *, plan: object = None,
                   model_alias: str | None = None) -> str:
            executed.append(plan)
            return "stub-result"

    monkeypatch.setattr(decide_module, "plan_context", fake_plan_context)
    monkeypatch.setattr(decide_module, "DecisionEngine", StubEngine)
    monkeypatch.setattr(session_module, "ModelSession", lambda *a, **kw: _StubSession())

    model = harness.LiveModel(harness.ModelSpec(path=str(tmp_path / "m.gguf"), backend="cpu"),
                              states_home=tmp_path / "states")
    model.handle = handle                                    # bypass the real load
    payload = devset_module.request_for(devset_module.load()[0], model="bench", threads=1)
    model.decide(schema.parse_request(payload))

    assert executed, "the engine never ran"
    assert executed[0] is not None, "the bench must hand the engine a plan"
    assert executed[0].prefix_tokens == HANDLE_TOKENS, (
        "the executed plan is not the one resolved from the handle: the bench planned from the "
        "session (plain framing), which is a different prompt than the serving path sends")
    assert planned_on == [handle], (
        f"plan_context ran on {len(planned_on)} tokenizer(s) {planned_on!r}; the handle is the "
        f"only planning source the serving path has")


def test_a_row_records_which_framing_it_measured(monkeypatch: pytest.MonkeyPatch,
                                                 tmp_path) -> None:
    """A published row must name the framing it measured — the fix is only visible if it does.

    The response already carries `engine.template` (the resolution that produced the prefix
    tokens); the quality row copies it, and the report summarizes it, so no table can be silent
    about which prompt its numbers describe (card t_6de5fc53, requirement 5).
    """
    from tests.fake_engine import BenchModel
    from typed_gguf.bench import suites

    def factory(spec: harness.ModelSpec) -> BenchModel:
        return BenchModel(spec, template={"kind": "gguf-renderer", "renderer": "internal",
                                          "family": "spark2_5"},
                          prefix_tokens=len(HANDLE_TOKENS))

    report = suites.run_suite(harness.BenchConfig(suite="quality", model_path=str(tmp_path / "m"),
                                                  items=1, threads=1),
                              factory=factory)
    row = report["items"][0]
    assert row["framing"]["family"] == "spark2_5"
    assert row["prefix_tokens"] == len(HANDLE_TOKENS)
    assert report["framing"]["labels"] == ["chat-template: spark2_5 / internal"]
    assert report["framing"]["prefix_tokens"] == [len(HANDLE_TOKENS)]
    assert report["framing"]["mixed"] is False
    markdown = harness.render_report(report)
    assert "- framing: chat-template: spark2_5 / internal" in markdown
