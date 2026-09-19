"""E3e: the roles-decision tool — the statistics, the rule, and the table it publishes.

The engine gates live in `tests/test_e3e_roles.py`; this file gates the *analysis* the card's
evidence rests on. A mutation here cannot be caught by the engine gates: the tool decides whether a
lever "wins", so its rule (`decide`), its direct statistics (`wilson`, `mcnemar_exact`,
`paired_difference`) and its report text are pinned against hand-computed values.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("e3e_roles_decision",
                                                  ROOT / "tools" / "e3e_roles_decision.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def report(label_items: dict[str, bool], *, cue: str = "shipped", chat_format: str | None = None,
           json_contract: str | None = None, reliability: dict[str, str] | None = None,
           verdicts: dict[str, str] | None = None, refusals: set[str] = frozenset()) -> dict:
    """A minimal quality report: item id -> correct, with the policy the tool reads."""
    reliability = reliability or {}
    verdicts = verdicts or {}
    items = []
    for index, (key, correct) in enumerate(label_items.items()):
        cue_block: dict = {"closer": None, "mass": 0.5, "refused": key in refusals,
                           "token": 100 + index}
        if key in verdicts:
            cue_block["verdict"] = verdicts[key]
        items.append({"id": key, "type": "choice", "correct": correct, "coverage": 0.9,
                      "reliability": reliability.get(key, "ok"), "cue": cue_block,
                      "expected": "a", "got": "a", "prefix_tokens": 100,
                      "framing": {"kind": "gguf-renderer", "renderer": "internal",
                                  "family": "spark2_5", "thinking": "suppressed", "warnings": []}})
    config = {"cue": cue, "backend": "vulkan", "gpu_layers": -1, "threads": 4}
    if chat_format:
        config["chat_format"] = chat_format
    if json_contract:
        config["json_contract"] = json_contract
    return {"suite": "quality", "ok": True, "config": config, "items": items, "warnings": [],
            "framing": {"labels": ["chat-template: spark2_5 / internal"], "mixed": False},
            "devset": {"path": "/x/devset.jsonl", "counts": {"choice": len(items)}},
            "model": {"name": "m.gguf"}, "wall_ms": 1000.0, "generated_at": "2026-09-19T00:00:00Z"}


# ------------------------------------------------------------------ direct statistics
def test_wilson_never_reports_certainty_for_a_small_sample() -> None:
    tool = _tool()
    low, high = tool.wilson(0, 60)
    assert low == 0.0 and 0.0 < high < 0.10
    low, high = tool.wilson(60, 60)
    assert high <= 1.0 and high > 0.99 and 0.90 < low < 1.0
    low, high = tool.wilson(51, 60)
    assert 0.70 < low < 0.76 and 0.90 < high < 0.94
    assert tool.wilson(0, 0) == (None, None)


def test_mcnemar_exact_matches_the_e3d_pairing() -> None:
    tool = _tool()
    # the E3d `shipped -> json_field` pairing: 12 vs 3 discordant, p = 0.035
    assert 0.02 < tool.mcnemar_exact(12, 3) < 0.05
    assert tool.mcnemar_exact(0, 0) == 1.0
    assert tool.mcnemar_exact(3, 3) == 1.0                     # symmetric, no evidence
    assert tool.mcnemar_exact(20, 0) < 0.001


def test_the_paired_difference_is_the_discordant_pair_delta() -> None:
    tool = _tool()
    diff = tool.paired_difference(12, 3, 60)
    assert diff["difference"] == pytest.approx(0.15)
    assert diff["ci"][0] > 0.0                                  # the E3d interval excludes zero
    assert "Wald" in diff["caveat"] and "optimistic" in diff["caveat"]
    assert tool.paired_difference(0, 0, 60)["difference"] == 0.0


# ------------------------------------------------------------------ cells and pairs
def test_the_cell_label_carries_the_policy_that_changes_the_bytes() -> None:
    tool = _tool()
    assert tool.cell_label(report({}, cue="shipped")) == "shipped/answer_sheet"
    assert tool.cell_label(report({}, cue="json_instructed",
                                  chat_format="role_split")) == "json_instructed/role_split"
    assert tool.cell_label(report({}, cue="json_instructed", chat_format="role_split",
                                  json_contract="system")) == "json_instructed/role_split/system"
    # the contract only labels the cue it belongs to
    assert tool.cell_label(report({}, cue="shipped", chat_format="role_split",
                                  json_contract="system")) == "shipped/role_split"


def test_cell_stats_read_the_report_it_is_given() -> None:
    tool = _tool()
    stats = tool.cell_stats(report({"a": True, "b": True, "c": False},
                                   verdicts={"b": "answered"}, refusals={"c"}))
    assert stats["correct"] == 2 and stats["items"] == 3
    assert stats["agreement"] == pytest.approx(2 / 3)
    assert stats["refusals"] == 1
    assert stats["verdicts"] == {"answered": 1}
    assert stats["per_type"]["choice"] == {"n": 3, "correct": 2,
                                           "agreement": pytest.approx(2 / 3),
                                           "ci": list(tool.wilson(2, 3))}
    assert stats["coverage"]["p50"] == 0.9


def test_pair_stats_count_the_discordant_items() -> None:
    tool = _tool()
    base = report({"a": True, "b": False, "c": True, "d": False})
    challenger = report({"a": True, "b": True, "c": False, "d": False})
    pair = tool.pair_stats(base, challenger)
    assert pair["both_correct"] == 1 and pair["neither_correct"] == 1
    assert pair["challenger_only"] == 1 and pair["baseline_only"] == 1
    assert pair["difference"] == 0.0
    assert pair["challenger_wins"] is False


def test_pair_stats_refuse_two_different_dev_sets() -> None:
    tool = _tool()
    with pytest.raises(tool.DecisionError) as caught:
        tool.pair_stats(report({"a": True}), report({"z": True}))
    assert "not the same dev set" in str(caught.value)


# ------------------------------------------------------------------ the rule
def test_decide_calls_a_win_only_past_the_noise() -> None:
    tool = _tool()
    # 10 items: 6 the challenger alone gets right is the smallest discordance the exact test passes
    base = dict.fromkeys("abcdefghij", False) | {"a": True, "b": True, "c": True, "d": True}
    strong = dict.fromkeys("abcdefghij", True)
    weak = dict.fromkeys("abcdefghij", True) | {"g": False, "h": False}
    base_report = report(base, cue="shipped")
    strong_report = report(strong, cue="json_instructed")
    weak_report = report(weak, cue="two_step")
    outcome = tool.decide([tool.cell_stats(base_report), tool.cell_stats(strong_report)],
                          [tool.pair_stats(base_report, strong_report)])
    assert outcome["verdicts"][0]["verdict"] == "wins"
    assert outcome["verdicts"][0]["challenger_only"] == 6
    assert outcome["verdicts"][0]["mcnemar_p"] < 0.05
    assert outcome["best"] == "json_instructed/answer_sheet"
    same = tool.decide([tool.cell_stats(base_report),
                        tool.cell_stats(report(dict(base), cue="two_step"))],
                       [tool.pair_stats(base_report, report(dict(base), cue="two_step"))])
    assert same["verdicts"][0]["verdict"] == "identical on these 60 items"
    noise = tool.decide([tool.cell_stats(base_report), tool.cell_stats(weak_report)],
                        [tool.pair_stats(base_report, weak_report)])
    assert noise["verdicts"][0]["verdict"] == "not by more than the CI noise"
    # the ranking still reports the counts — the verdict is about the *evidence*, not the order
    assert noise["best"] == "two_step/answer_sheet"
    assert noise["verdicts"][0]["challenger_only"] == 4
    assert noise["verdicts"][0]["ci"][0] > 0.0     # the interval clears zero ...
    assert noise["verdicts"][0]["mcnemar_p"] > 0.05  # ... but the exact test does not


def test_decide_refuses_two_cells_that_claim_the_same_policy() -> None:
    tool = _tool()
    with pytest.raises(tool.DecisionError) as caught:
        tool.decide([tool.cell_stats(report({"a": True})),
                     tool.cell_stats(report({"a": False}))], [])
    assert "share a label" in str(caught.value)


def test_decide_flips_the_pair_when_the_baseline_is_the_second_report() -> None:
    tool = _tool()
    first = report(dict.fromkeys("abcdefghij", False) | {"a": True, "b": True, "c": True,
                                                         "d": True}, cue="shipped")
    second = report(dict.fromkeys("abcdefghij", True), cue="json_instructed")
    cells = [tool.cell_stats(first), tool.cell_stats(second)]
    pair = tool.pair_stats(first, second)
    assert pair["challenger_only"] == 6 and pair["baseline_only"] == 0
    flipped = tool.pair_stats(second, first)              # the same two reports, reversed
    verdict = tool.decide(cells, [flipped], baseline="shipped/answer_sheet")["verdicts"][0]
    assert verdict["challenger_only"] == 6                # the counts are un-swapped by the rule
    assert verdict["baseline_only"] == 0
    assert verdict["difference"] == pytest.approx(0.6)    # ... and so is the sign
    assert verdict["verdict"] == "wins"


def test_decide_refuses_a_baseline_that_is_not_in_the_table() -> None:
    tool = _tool()
    cells = [tool.cell_stats(report({"a": True}))]
    with pytest.raises(tool.DecisionError) as caught:
        tool.decide(cells, [], baseline="nope/answer_sheet")
    assert "not in the table" in str(caught.value)


# ------------------------------------------------------------------ the report
def test_the_report_names_the_policy_the_pairs_and_the_comparability_cost() -> None:
    tool = _tool()
    base = report({"a": True, "b": False, "c": True, "d": False}, cue="shipped")
    challenger = report({"a": True, "b": True, "c": True, "d": True}, cue="json_instructed",
                        chat_format="role_split", json_contract="system")
    record = tool.analyses([base, challenger], baseline="shipped/answer_sheet")
    record.update({"schema": tool.SCHEMA, "generated_at": "2026-09-19T00:00:00Z",
                   "devset": base["devset"], "items": 4, "model": "m.gguf", "backend": "vulkan",
                   "gpu_layers": -1, "threads": 4,
                   "recommendation": "Ship the instructed JSON.",
                   "caveats": tool.build_caveats(record["cells"], record["pairs"])})
    text = tool.render(record)
    assert "`json_instructed/role_split/system`" in text
    assert "exact McNemar p" in text and "Wilson 95 %" in text
    assert "## Decision" in text and "Ship the instructed JSON." in text
    assert "Comparability cost" in text
    assert "devset.jsonl" in text                              # the dev set is named, not dumped
    assert "prompt policy" in text or "policy each cell ran under" in text


def test_load_report_refuses_a_report_that_is_not_the_quality_suite(tmp_path) -> None:
    tool = _tool()
    path = tmp_path / "bench.json"
    path.write_text(json.dumps({"suite": "prefill", "items": [{"id": "a"}]}), encoding="utf-8")
    with pytest.raises(tool.DecisionError) as caught:
        tool.load_report(path)
    assert "not 'quality'" in str(caught.value)
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"suite": "quality", "items": []}), encoding="utf-8")
    with pytest.raises(tool.DecisionError) as caught:
        tool.load_report(empty)
    assert "no items" in str(caught.value)
