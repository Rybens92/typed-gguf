"""E3d: the cue-shape decision tool, offline (card t_d90404ac).

`tools/e3d_cue_decision.py` turns an E3c probe record into the card's evidence: per shape the
agreement with a Wilson CI, the **paired** comparison (the same 60 items measured by every shape),
coverage distribution, `low_mass` share, `W_CUE_REFUSED` count and the per-type split.

These gates pin the statistics, not the model: the exact McNemar p-value on the discordant pairs,
the seeded paired bootstrap, the routing of "which row is this shape's readout" (the two-step
shapes read the *advanced* row), and the decision rule (a win is a paired CI that excludes zero —
overlapping marginal CIs are not a test).

Note on the McNemar reference values: they are the exact two-sided binomial tail values
(b=10,c=0 -> 2 * 2**-10), not a chi-square approximation — the card's user population is 60 items.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

from typed_gguf.bench import harness, labels

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_tool():
    """Import `tools/e3d_cue_decision.py` the way the other gates import a tool."""
    spec = importlib.util.spec_from_file_location("e3d_cue_decision",
                                                 ROOT / "tools" / "e3d_cue_decision.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["e3d_cue_decision"] = module
    spec.loader.exec_module(module)
    return module


tool = load_tool()


def shape_item(item_id: str, qtype: str, **per_policy) -> dict:
    """One synthetic dev item: `per_policy[policy] = (got, correct, coverage, reliability)`."""
    ranked = {}
    for policy, (got, correct, coverage, reliability) in per_policy.items():
        ranked[policy] = {"got": got, "correct": correct, "coverage": coverage,
                          "reliability": reliability, "agrees": correct}
    return {"id": item_id, "type": qtype, "expected": "a", "ranked": ranked, "shapes": {}}


def synthetic_record(n_per_type: int = 4) -> dict:
    """A tiny record with a clear winner: `two_step` beats `shipped` on 8 of 12, never loses."""
    items = []
    for index in range(n_per_type * 3):
        qtype = ("choice", "score", "noul")[index % 3]
        correct_shipped = index >= 4                            # the first 4 are misses
        correct_two_step = True
        items.append(shape_item(
            f"i{index}", qtype,
            **{"shipped=bare": ("a" if correct_shipped else "b", correct_shipped, 0.02,
                                "low_mass" if not correct_shipped else "ok"),
               "two_step_shipped=bare": ("a", correct_two_step, 0.9, "ok")}))
    return {"schema": "typed_gguf.e3c.cue-shapes/v1", "items": items,
            "shapes": ["shipped", "two_step_shipped", "json_field"],
            "label_variants": ["bare"], "mass_floor": 0.10,
            "ranked_keys": ["shipped=bare", "two_step_shipped=bare"]}


# ------------------------------------------------------------------- statistics
def test_wilson_is_the_harness_s_own_function():
    """No re-derivation: a second interval implementation is a second truth."""
    assert tool.wilson(38, 60) == harness.wilson_interval(38, 60)


def test_mcnemar_exact_pins_the_binomial_tail():
    assert tool.mcnemar_exact(0, 0) == 1.0
    assert tool.mcnemar_exact(10, 0) == pytest.approx(2 * 2 ** -10)
    assert tool.mcnemar_exact(5, 0) == pytest.approx(2 * 2 ** -5)
    assert tool.mcnemar_exact(3, 3) == pytest.approx(1.0)


def test_mcnemar_is_symmetric_and_never_leaves_the_unit_interval():
    for b, c in ((1, 4), (4, 1), (7, 0), (0, 7), (9, 9)):
        value = tool.mcnemar_exact(b, c)
        assert 0.0 <= value <= 1.0
        assert value == tool.mcnemar_exact(c, b)


def test_the_paired_bootstrap_is_deterministic_and_brackets_the_point_estimate():
    base = [True, False, False, True]
    other = [True, True, True, True]
    first = tool.paired_diff(base, other, iters=2000, seed=7)
    second = tool.paired_diff(base, other, iters=2000, seed=7)
    assert first == second
    diff = first["difference"]
    assert diff == pytest.approx(0.5)
    assert first["low"] <= diff <= first["high"]
    # base is right on items 1 and 4, the challenger on all four: two discordant pairs, all of
    # them the challenger's
    assert first["discordant"] == {"a_win": 0, "b_win": 2}


def test_identical_policies_give_a_zero_difference_and_a_degenerate_interval():
    same = [True, True, False]
    result = tool.paired_diff(same, same, iters=500, seed=1)
    assert result["difference"] == 0.0
    assert (result["low"], result["high"]) == (0.0, 0.0)


# ------------------------------------------------------------------- record reading
def test_rows_for_refuses_an_unmeasured_policy():
    record = synthetic_record()
    with pytest.raises(tool.DecisionError, match="json_field=bare"):
        tool.rows_for(record, "json_field=bare")


def test_rows_for_keeps_the_wire_order_and_the_type():
    record = synthetic_record()
    rows = tool.rows_for(record, "shipped=bare")
    assert [row["id"] for row in rows] == [item["id"] for item in record["items"]]
    assert {row["type"] for row in rows} == {"choice", "score", "noul"}
    # the fixture's first four items are misses for the baseline, the rest are hits
    assert rows[0]["correct"] is False and rows[-1]["correct"] is True


def test_coverage_summary_counts_the_floor_share_and_the_quartiles():
    summary = tool.coverage_summary([0.05, 0.2, 0.4, 0.6], floor=0.10)
    assert summary["n"] == 4
    assert summary["above_floor"] == 3
    assert summary["share_above_floor"] == pytest.approx(0.75)
    assert summary["median"] == pytest.approx(0.3)
    assert summary["min"] == 0.05 and summary["max"] == 0.6


def test_the_verdict_rule_is_the_paired_ci_not_the_marginal_overlap():
    record = synthetic_record()
    verdict = tool.decide(record, baseline="shipped=bare", challenger="two_step_shipped=bare")
    assert verdict["challenger_wins"] is True
    assert verdict["paired"]["low"] > 0.0
    # the marginal intervals of the two policies overlap in a 60-item win of 3/6 vs 6/6 -- the
    # render must say why that is not the reading
    assert "paired" in verdict["note"]


@pytest.mark.parametrize("field", ["agreement", "wilson", "low_mass", "refused", "coverage"])
def test_analyse_reports_every_column_the_card_asks_for(field):
    analysis = tool.analyse(synthetic_record())
    assert field in analysis["shapes"]["two_step_shipped=bare"]


def test_analyse_splits_every_policy_by_question_type():
    analysis = tool.analyse(synthetic_record())
    per_type = analysis["shapes"]["shipped=bare"]["per_type"]
    assert set(per_type) == {"choice", "score", "noul"}
    assert per_type["choice"]["n"] == 4


# ------------------------------------------------------------------- rendering
def test_the_render_carries_the_card_s_tables_and_the_decision():
    markdown = tool.render(tool.analyse(synthetic_record()),
                           decisions=[tool.decide(synthetic_record(), baseline="shipped=bare",
                                                  challenger="two_step_shipped=bare")])
    assert "| policy |" in markdown
    assert "paired" in markdown
    assert "McNemar" in markdown
    assert "two_step_shipped=bare" in markdown


def test_the_render_is_offline_and_reproducible_from_a_stored_record(tmp_path):
    record = synthetic_record()
    path = tmp_path / "run.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    first = tool.report(tool.load_run(path), run_path=str(path))
    second = tool.report(tool.load_run(path), run_path=str(path))
    assert first == second


def test_the_mass_floor_comes_from_the_record_not_a_constant():
    record = synthetic_record()
    record["mass_floor"] = labels.MASS_FLOOR
    analysis = tool.analyse(record)
    assert analysis["mass_floor"] == labels.MASS_FLOOR
