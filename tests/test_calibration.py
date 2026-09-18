"""A-E2p5-1/2/3/6: the calibration math, the fit and the acceptance gate (SPEC 2.10).

The whole file is offline: every case drives the pure math (or the fit over stored rows, the
shape `--suite calibration` publishes), so a fit is reproducible without a model, a runtime or a
GPU. The live numbers live in `docs/evidence/e2p5_*.json` and are reproduced by
`tools/e2p5_reproduce.py`.

Pins worth knowing before editing:
  * the binning/ECE definitions here are the SAME definitions `bench.harness` publishes — a test
    cross-checks the two implementations, because the acceptance gate (A-E2p5-2) is an ECE
    comparison and the two must never drift apart;
  * temperature 1.0 is always in the grid, so "accept only if the held-out split improves" can
    never be satisfied by a fit that is worse than doing nothing (a stronger statement than
    "we chose the best grid point");
  * calibration never changes the decision (the argmax), only the reported probabilities and
    confidence — that is asserted here and again at the engine seam.
"""
from __future__ import annotations

import json
import math
import pathlib

import pytest

from ggufone.bench import harness
from ggufone.calibration import calibrate, stats
from ggufone.engine import readout


# ------------------------------------------------------------------ bins / ECE
def test_the_calibration_bins_match_the_published_bench_definition() -> None:
    """One definition of ECE in the repo: the acceptance gate must not drift from bench."""
    confidences = [0.05, 0.15, 0.42, 0.51, 0.66, 0.71, 0.88, 0.93, 0.99, 0.5, 0.5, 0.31]
    correct = [False, False, True, True, False, True, True, True, True, False, True, False]
    mine = stats.bins(confidences, correct, n_bins=10)
    theirs = harness.reliability_bins(confidences, correct, n_bins=10)
    assert mine == theirs
    assert stats.ece(mine) == pytest.approx(harness.ece(theirs))


def test_ece_is_zero_for_a_perfectly_calibrated_series() -> None:
    confidences = [0.5] * 40
    correct = [True] * 20 + [False] * 20
    assert stats.ece(stats.bins(confidences, correct)) == pytest.approx(0.0)


def test_ece_is_the_confidence_gap_of_a_fully_overconfident_run() -> None:
    """Everything at 1.0, nothing right -> ECE = 1.0 (the worst case, not clamped away)."""
    bins = stats.bins([1.0] * 10, [False] * 10)
    assert stats.ece(bins) == pytest.approx(1.0)


def test_empty_input_is_reported_as_zero_not_as_a_crash() -> None:
    assert stats.ece(stats.bins([], [])) == 0.0
    assert stats.agreement_at([], [], 0.5)["agreement"] == 0.0


def test_bins_are_equal_width_and_keep_their_shape() -> None:
    bins = stats.bins([0.5], [True], n_bins=5)
    assert [entry["lo"] for entry in bins] == [0.0, 0.2, 0.4, 0.6, 0.8]
    assert [entry["hi"] for entry in bins] == [0.2, 0.4, 0.6, 0.8, 1.0]
    assert sum(entry["n"] for entry in bins) == 1
    assert bins[2]["n"] == 1 and bins[2]["mean_confidence"] == pytest.approx(0.5)
    assert [entry["n"] for entry in bins].count(0) == 4


def test_a_confidence_outside_the_unit_interval_is_clamped_into_the_edge_bin() -> None:
    """Candidates never produce this, but a hand-written row must not escape the table."""
    bins = stats.bins([1.5, -0.5], [True, False], n_bins=2)
    assert [entry["n"] for entry in bins] == [1, 1]


def test_an_invalid_bin_count_is_an_error() -> None:
    with pytest.raises(ValueError):
        stats.bins([0.5], [True], n_bins=0)


def test_mismatched_series_are_an_error() -> None:
    with pytest.raises(ValueError):
        stats.bins([0.5, 0.5], [True])


# ------------------------------------------------- agreement at a fixed threshold
def test_agreement_at_a_threshold_only_counts_rows_above_it() -> None:
    confidences = [0.9, 0.8, 0.2, 0.1]
    correct = [True, False, True, True]
    high = stats.agreement_at(confidences, correct, 0.5)
    assert high == {"threshold": 0.5, "n": 2, "correct": 1, "agreement": pytest.approx(0.5),
                    "ci": high["ci"]}
    assert stats.agreement_at(confidences, correct, 0.05)["n"] == 4
    assert stats.agreement_at(confidences, correct, 0.95)["n"] == 0


def test_the_threshold_agreement_carries_a_wilson_interval() -> None:
    result = stats.agreement_at([0.9] * 10, [True] * 9 + [False], 0.5)
    low, high = result["ci"]
    assert result["correct"] == 9 and result["n"] == 10
    assert 0.0 <= low < 0.9 < high <= 1.0


# ------------------------------------------------------------- temperature math
def test_power_scaling_keeps_the_ranking_and_the_unit_sum() -> None:
    probs = [0.6, 0.3, 0.1]
    for temperature in (0.25, 0.5, 1.0, 2.0, 8.0):
        scaled = stats.power_scale(probs, temperature)
        assert sum(scaled) == pytest.approx(1.0)
        assert scaled == sorted(scaled, reverse=True)
        assert scaled.index(max(scaled)) == probs.index(max(probs))


def test_power_scaling_with_temperature_one_is_the_identity() -> None:
    probs = [0.7, 0.2, 0.1]
    assert stats.power_scale(probs, 1.0) == pytest.approx(probs)


def test_a_temperature_above_one_flattens_and_below_one_sharpens() -> None:
    flat = stats.power_scale([0.8, 0.2], 4.0)
    sharp = stats.power_scale([0.8, 0.2], 0.25)
    assert flat[0] < 0.8 < sharp[0]
    assert flat[1] > 0.2 > sharp[1]


def test_power_scaling_rejects_a_non_positive_temperature() -> None:
    with pytest.raises(ValueError):
        stats.power_scale([0.5, 0.5], 0.0)


def test_power_scaling_never_loses_a_zero_probability() -> None:
    assert stats.power_scale([1.0, 0.0], 0.5) == pytest.approx([1.0, 0.0])


def test_negative_log_likelihood_reads_the_correct_slot() -> None:
    assert stats.nll([0.5, 0.25, 0.25], 0) == pytest.approx(math.log(2.0))
    assert stats.nll([0.0, 1.0], 0) == math.inf


# ------------------------------------------------------------------------ the fit
def _rows_from_softmax(temperature: float, *, n: int = 200, k: int = 4, seed: int = 7):
    """Rows drawn as if the model's own softmax temperature was `temperature`, with real errors.

    For each item a confidence `c in [0.5, 1]` is drawn, the item is correct with probability
    `c`, and the peak of the row is placed so that `normalized_peak` reports exactly `c` (K-1
    losers share the rest). A model with `temperature > 1` is then too flat for its own accuracy
    (under-confident) and one with `temperature < 1` too sharp (over-confident), so the fit's job
    is to invert that temperature: the correction is `1 / temperature`.

    Errors are the point: a row set that is never wrong can be "calibrated" by any extreme
    sharpening (accuracy 1.0, ECE 0.0), which would make the fit collapse to the grid edge and
    prove nothing.
    """
    state = seed

    def draw() -> float:
        nonlocal state
        state = (1103515245 * state + 12345) % (2 ** 31)
        return state / 2 ** 31

    rows: list[list[float]] = []
    correct: list[int] = []
    for _ in range(n):
        confidence = 0.5 + 0.5 * draw()
        hit = draw() <= confidence
        peak = 1.0 / k + (1.0 - 1.0 / k) * confidence
        rest_total = 1.0 - peak
        weights = [draw() + 0.05 for _ in range(k - 1)]
        total = sum(weights)
        probabilities = [peak] + [rest_total * weight / total for weight in weights]
        rows.append(readout.softmax([math.log(p) for p in probabilities], temperature))
        correct.append(0 if hit else 1)
    return rows, correct


def test_the_fit_grid_always_contains_temperature_one() -> None:
    """The identity is a candidate parameter: 'no calibration' is never worse than a grid point."""
    assert any(value == 1.0 for value in stats.DEFAULT_GRID)
    assert min(stats.DEFAULT_GRID) < 1.0 < max(stats.DEFAULT_GRID)


def test_the_fit_sharpens_a_soft_model() -> None:
    rows, correct = _rows_from_softmax(2.0, n=400)
    fit = stats.fit_temperature(rows, correct)
    assert fit["temperature"] == pytest.approx(0.5, abs=0.1)
    assert fit["ece"] < fit["ece_before"]
    assert fit["nll"] < fit["nll_before"]


def test_the_fit_flattens_a_sharp_model() -> None:
    rows, correct = _rows_from_softmax(0.5, n=400)
    fit = stats.fit_temperature(rows, correct)
    assert fit["temperature"] == pytest.approx(2.0, abs=0.1)
    assert fit["ece"] < fit["ece_before"]


def test_the_fit_leaves_an_already_calibrated_run_alone() -> None:
    """A model that is already honest about its accuracy gets no correction worth speaking of.

    The bound is one grid step either way: the point is that a calibrated series is not
    "corrected" by a factor of two, not that sampling noise cannot move a minimum by one step.
    """
    rows, correct = _rows_from_softmax(1.0, n=400)
    fit = stats.fit_temperature(rows, correct)
    assert abs(math.log(fit["temperature"])) <= 0.25
    assert fit["ece"] <= fit["ece_before"]


def test_the_fit_never_returns_a_grid_point_worse_than_doing_nothing() -> None:
    """The gate's honesty floor: `temperature = 1.0` is in the grid (and is the reference)."""
    for temperature in (0.2, 0.5, 1.0, 3.0, 12.0):
        rows, correct = _rows_from_softmax(temperature, n=60)
        fit = stats.fit_temperature(rows, correct)
        assert fit["ece"] <= fit["ece_before"] + 1e-12
        assert fit["nll"] <= fit["nll_before"] + 1e-12


def test_the_fit_reports_the_mode_it_optimised() -> None:
    rows, correct = _rows_from_softmax(2.0, n=400)
    fit = stats.fit_temperature(rows, correct, mode="entropy")
    assert fit["mode"] == "entropy"
    assert fit["temperature"] < 1.0


def test_the_fit_refuses_an_unknown_confidence_mode() -> None:
    rows, correct = _rows_from_softmax(2.0, n=10)
    with pytest.raises(KeyError):
        stats.fit_temperature(rows, correct, mode="vibes")


def test_the_evaluation_reports_every_documented_statistic() -> None:
    rows, correct = _rows_from_softmax(2.0, n=40)
    evaluated = stats.evaluate(rows, correct, 1.0)
    assert set(evaluated) == {"temperature", "mode", "n", "ece", "nll", "mean_confidence",
                              "agreement", "agreement_at_0.5"}
    assert evaluated["n"] == 40
    assert 0.0 <= evaluated["agreement_at_0.5"]["agreement"] <= 1.0


def test_an_empty_fit_is_an_error_rather_than_a_silent_temperature_one() -> None:
    with pytest.raises(ValueError):
        stats.fit_temperature([], [])


def test_mismatched_fit_inputs_are_an_error() -> None:
    with pytest.raises(ValueError):
        stats.fit_temperature([[0.5, 0.5]], [])


# ------------------------------------------------------------- reproducibility
def test_the_params_hash_is_stable_for_the_same_parameters() -> None:
    params = {"model": "abc", "types": {"choice": {"temperature": 1.25},
                                       "noul": {"temperature": 0.75}}}
    again = {"types": {"noul": {"temperature": 0.75}, "choice": {"temperature": 1.25}},
             "model": "abc"}
    assert stats.params_hash(params) == stats.params_hash(again)  # key order is not data


def test_the_params_hash_changes_when_a_parameter_changes() -> None:
    base = {"model": "abc", "types": {"choice": {"temperature": 1.25}}}
    bumped = {"model": "abc", "types": {"choice": {"temperature": 1.2500001}}}
    other_model = {"model": "def", "types": {"choice": {"temperature": 1.25}}}
    assert stats.params_hash(base) != stats.params_hash(bumped)
    assert stats.params_hash(base) != stats.params_hash(other_model)


def test_the_params_hash_is_a_sha256_label() -> None:
    digest = stats.params_hash({"model": "abc"})
    assert digest.startswith("sha256:")
    assert len(digest.removeprefix("sha256:")) == 64


def test_the_canonical_form_rounds_away_float_noise_but_keeps_the_value() -> None:
    """Two runs of the same fit must hash identically even if the last bit of a float moves."""
    left = {"temperature": 1.2999999999999998}
    right = {"temperature": 1.3}
    assert stats.params_hash(left) == stats.params_hash(right)
    assert stats.canonical({"b": 1, "a": 2}) == '{"a":2,"b":1}'


# ------------------------------------------- rows, the split and the gate (A-E2p5-1/2)
def _report_rows(types: tuple[str, ...] = ("choice", "noul"),
                 *, per_type: int = 12, temperature: float = 2.0) -> list[dict]:
    """Rows in the exact shape `--suite calibration` publishes (`report["items"][i]`)."""
    rows = []
    for offset, qtype in enumerate(types):
        for index in range(per_type):
            hit = index % 2 == 0
            if qtype == "choice":
                expected, got = ("billing", "billing") if hit else ("billing", "api")
                probabilities = {"billing": 0.6, "api": 0.3, "sales": 0.1} if hit else \
                    {"billing": 0.3, "api": 0.6, "sales": 0.1}
            else:
                expected, got = ("yes", "yes") if hit else ("yes", "no")
                probabilities = {"yes": 0.65, "no": 0.35} if hit else {"yes": 0.35, "no": 0.65}
            scaled = stats.power_scale(list(probabilities.values()), temperature)
            probabilities = dict(zip(probabilities, scaled, strict=True))
            rows.append({
                "id": f"{qtype[:2]}{offset}{index:02d}", "type": qtype, "expected": expected,
                "got": got, "correct": hit,
                "confidence": readout.confidence(list(probabilities.values())),
                "coverage": 0.8, "reliability": "ok", "probabilities": probabilities,
                "questions_ms": 1.0, "wall_ms": 2.0,
            })
    return rows


def _fit_rows(types: tuple[str, ...] = ("choice", "noul"), **kwargs) -> list:
    return [calibrate.Row.from_item(item) for item in _report_rows(types, **kwargs)]


def test_a_report_row_round_trips_into_a_fit_row() -> None:
    item = _report_rows(per_type=2)[0]
    row = calibrate.Row.from_item(item)
    assert row.id == item["id"] and row.type == "choice"
    assert list(row.probabilities) == [item["probabilities"][key] for key in item["probabilities"]]
    assert row.labels == tuple(item["probabilities"])
    assert row.correct is True
    assert row.correct_index == 0


def test_a_published_report_file_loads_into_rows(tmp_path) -> None:
    path = tmp_path / "e2_calibration.json"
    path.write_text(json.dumps({"items": _report_rows(per_type=3)}), encoding="utf-8")
    rows = calibrate.load_rows(path)
    assert len(rows) == 6
    assert {row.type for row in rows} == {"choice", "noul"}


def test_a_report_without_items_is_an_actionable_error(tmp_path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"n": 0}), encoding="utf-8")
    with pytest.raises(ValueError, match="items"):
        calibrate.load_rows(path)


def test_the_split_is_per_type_deterministic_and_disjoint() -> None:
    rows = _fit_rows(per_type=12)
    first = calibrate.split_rows(rows, holdout_fraction=1 / 3)
    second = calibrate.split_rows(list(reversed(rows)), holdout_fraction=1 / 3)
    assert [row.id for row in first.fit] == [row.id for row in second.fit]
    assert [row.id for row in first.holdout] == [row.id for row in second.holdout]
    assert {row.id for row in first.fit} & {row.id for row in first.holdout} == set()
    assert len(first.fit) + len(first.holdout) == len(rows)
    assert {row.type for row in first.fit} == {"choice", "noul"}
    assert {row.type for row in first.holdout} == {"choice", "noul"}
    assert len(first.holdout) == 8  # 2 types x 12 rows x 1/3


def test_a_holdout_fraction_outside_the_unit_interval_is_an_error() -> None:
    rows = _fit_rows(per_type=3)
    for fraction in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError):
            calibrate.split_rows(rows, holdout_fraction=fraction)


def test_every_type_keeps_at_least_one_row_on_both_sides() -> None:
    split = calibrate.split_rows(_fit_rows(per_type=3), holdout_fraction=0.9)
    assert split.fit and split.holdout
    for qtype in ("choice", "noul"):
        assert any(row.type == qtype for row in split.fit)
        assert any(row.type == qtype for row in split.holdout)


# ---------------------------------------------- the fit, the gate, the table (A-E2p5-1/2/3/6)
def _flat_rows(types: tuple[str, ...] = ("choice",), *, per_type: int = 12) -> list:
    """Rows that carry no information: uniform probabilities, half of them labelled right.

    Every candidate is equally likely, so *every* confidence statistic — `normalized_peak`,
    `entropy` and `margin` — reports 0 for each row at any temperature, while the accuracy is 0.5:
    the ECE is 0.5 everywhere and the fit can only choose the identity (ties break towards it).
    That is the honest "there is nothing to learn here" fixture — a type that is *already* right
    cannot be improved, and the gate must say so instead of picking up a statistic nobody asked
    about.
    """
    items = []
    for offset, qtype in enumerate(types):
        for index in range(per_type):
            hit = index % 2 == 0
            if qtype == "choice":
                probabilities = {"billing": 1 / 3, "api": 1 / 3, "sales": 1 / 3}
                expected, got = ("billing", "billing") if hit else ("api", "billing")
            else:
                probabilities = {"yes": 0.5, "no": 0.5}
                expected, got = ("yes", "yes") if hit else ("no", "yes")
            items.append({
                "id": f"flat-{qtype}-{offset}-{index:02d}", "type": qtype, "expected": expected,
                "got": got, "correct": hit, "confidence": 0.0, "coverage": 0.9,
                "reliability": "ok", "probabilities": probabilities,
            })
    return [calibrate.Row.from_item(item) for item in items]


def test_the_fit_produces_one_temperature_per_question_type() -> None:
    table = calibrate.fit_table(_fit_rows(("choice", "noul"), per_type=12),
                                model_key="sha256:test")
    assert set(table.types) == {"choice", "noul"}
    assert all(entry.temperature > 0 for entry in table.types.values())
    # the documented default: fit every mode, keep `normalized_peak` unless another wins by the
    # margin (A-E2p5-3)
    assert table.mode == calibrate.MODE_AUTO
    assert set(table.modes) == set(readout.CONFIDENCE_MODES)
    for entry in table.types.values():
        assert entry.mode == readout.DEFAULT_CONFIDENCE_MODE


def test_the_gate_accepts_a_miscalibrated_type_on_the_held_out_split() -> None:
    table = calibrate.fit_table(_fit_rows(("choice",), per_type=18, temperature=2.0),
                                model_key="sha256:test")
    entry = table.types["choice"]
    assert entry.accepted is True and table.accepted is True
    assert entry.temperature < 1.0                       # rows were too soft -> sharpen
    assert entry.holdout_after["ece"] < entry.holdout_before["ece"]
    assert entry.fit_after["ece"] < entry.fit_before["ece"]
    assert entry.reason.startswith("applied")


def test_a_type_that_is_already_calibrated_reports_no_calibration_applied() -> None:
    table = calibrate.fit_table(_flat_rows(per_type=18), model_key="sha256:test")
    entry = table.types["choice"]
    assert entry.accepted is False
    assert entry.temperature == 1.0
    assert entry.reason == ("no calibration applied: the fit chose the identity "
                            "(temperature 1.0)")
    assert table.accepted is False
    assert table.to_json()["accepted"] is False


def test_a_type_with_too_few_rows_is_never_fitted() -> None:
    table = calibrate.fit_table(_fit_rows(("choice",), per_type=6), model_key="sha256:test")
    entry = table.types["choice"]
    assert entry.accepted is False
    assert "too few" in entry.reason
    assert entry.temperature == 1.0


def _renamed(rows: list, prefix: str) -> list:
    import dataclasses
    return [dataclasses.replace(row, id=f"{prefix}-{row.id}") for row in rows]


def test_a_fit_that_does_not_survive_the_holdout_is_rejected() -> None:
    """The gate is the held-out split, not the fit split: a correction that only fits wins nothing.

    The fit half is miscalibrated (so a temperature other than 1.0 is proposed) while the held-out
    half carries no information at all — applying the fitted temperature there changes nothing for
    the better, which is exactly the case "record the score and apply nothing" exists for.
    """
    rows = (_renamed(_fit_rows(("choice",), per_type=12, temperature=2.0), "a")
            + _renamed(_flat_rows(per_type=6), "z"))
    table = calibrate.fit_table(rows, model_key="sha256:test")
    entry = table.types["choice"]
    assert entry.modes[entry.mode]["temperature"] != 1.0      # the fit did propose something
    assert entry.accepted is False
    assert entry.temperature == 1.0
    assert entry.reason == "no calibration applied: the held-out split did not improve"
    assert table.accepted is False


def test_a_question_type_that_the_dev_set_never_used_is_not_in_the_table() -> None:
    table = calibrate.fit_table(_fit_rows(("choice",), per_type=18, temperature=2.0),
                                model_key="sha256:test")
    assert "noul" not in table.types
    assert "noul" not in table.to_json()["types"]


def test_the_report_measures_all_three_confidence_modes() -> None:
    """A-E2p5-3: one calibration report carries every mode, the default stays normalized_peak."""
    table = calibrate.fit_table(_fit_rows(("choice",), per_type=18, temperature=2.0),
                                model_key="sha256:test")
    modes = table.types["choice"].modes
    assert set(modes) == set(readout.CONFIDENCE_MODES) == {"normalized_peak", "entropy", "margin"}
    for name, report in modes.items():
        assert report["mode"] == name
        assert {"ece_fit_before", "ece_fit_after", "ece_holdout_before", "ece_holdout_after",
                "agreement_holdout_before", "agreement_holdout_after", "temperature",
                "eligible", "selected", "holdout_ece_improvement"} <= set(report)
    assert table.mode == calibrate.MODE_AUTO
    assert table.to_json()["mode"] == calibrate.MODE_AUTO
    assert table.types["choice"].mode == readout.DEFAULT_CONFIDENCE_MODE


# ------------------------------------------------- the mode selection (A-E2p5-3)
def _mode_case(holdout_before: float, holdout_after: float,
               temperature: float = 1.5) -> dict:
    return {"temperature": temperature, "holdout_before": {"ece": holdout_before},
            "holdout_after": {"ece": holdout_after}}


def test_the_selection_rule_keeps_the_default_whenever_it_is_eligible() -> None:
    per_mode = {"normalized_peak": _mode_case(0.30, 0.20),
                "entropy": _mode_case(0.30, 0.05),        # better, but the default is eligible
                "margin": _mode_case(0.30, 0.29)}
    assert calibrate._select_mode(per_mode, default="normalized_peak") == ("normalized_peak", "")


def test_the_selection_rule_switches_only_when_the_margin_is_met() -> None:
    """A switch is user-visible: it has to beat the identity by more than the documented margin."""
    below = {"normalized_peak": _mode_case(0.30, 0.30),
             "entropy": _mode_case(0.30, 0.30 - 0.003),
             "margin": _mode_case(0.30, 0.30)}
    selected, reason = calibrate._select_mode(below, default="normalized_peak", margin=0.005)
    assert selected is None and "margin a mode switch needs" in reason
    above = {"normalized_peak": _mode_case(0.30, 0.31),
             "entropy": _mode_case(0.30, 0.28),           # 0.02 > 0.005
             "margin": _mode_case(0.30, 0.30)}
    assert calibrate._select_mode(above, default="normalized_peak", margin=0.005)[0] == "entropy"


def test_the_selection_rule_reports_the_identity_when_no_mode_moved() -> None:
    same = {name: _mode_case(0.3, 0.3, 1.0) for name in readout.CONFIDENCE_MODES}
    selected, reason = calibrate._select_mode(same, default="normalized_peak")
    assert selected is None and "identity" in reason


def test_the_selection_rule_reports_a_plain_holdout_rejection() -> None:
    worse = {"normalized_peak": _mode_case(0.30, 0.44, 2.2),
             "entropy": _mode_case(0.30, 0.31, 1.5),
             "margin": _mode_case(0.30, 0.31, 2.0)}
    selected, reason = calibrate._select_mode(worse, default="normalized_peak")
    assert selected is None
    assert reason == "no calibration applied: the held-out split did not improve"


def test_the_fit_accepts_a_type_only_with_an_eligible_selected_mode() -> None:
    table = calibrate.fit_table(_fit_rows(("choice", "noul"), per_type=18, temperature=2.0),
                                model_key="sha256:test")
    for name, entry in table.types.items():
        assert entry.mode in readout.CONFIDENCE_MODES
        if entry.accepted:
            assert entry.modes[entry.mode]["eligible"] is True
            assert entry.modes[entry.mode]["selected"] is True
            assert entry.temperature == entry.modes[entry.mode]["temperature"]
        else:
            assert entry.temperature == 1.0, name
            assert entry.modes[entry.mode]["selected"] is False


def test_an_explicit_mode_is_fitted_alone_and_keeps_its_name() -> None:
    table = calibrate.fit_table(_fit_rows(("choice",), per_type=18, temperature=2.0),
                                model_key="sha256:test", mode="entropy")
    assert table.mode == "entropy"
    assert set(table.types["choice"].modes) == {"entropy"}


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(KeyError):
        calibrate.fit_table(_fit_rows(("choice",), per_type=18), model_key="sha256:test",
                            mode="vibes")


# ------------------------------------- the live 0.8B rows, refit (A-E2p5-2/3 evidence pin)
EVIDENCE_ROWS = (pathlib.Path(__file__).resolve().parents[1]
                 / "docs" / "evidence" / "e2p5_rows_qwen08.json")
QWEN_KEY = "file:Qwen3.5-0.8B-UD-Q4_K_XL.gguf:558772480"


def test_the_live_rows_accept_exactly_one_type_and_switch_that_type_to_entropy() -> None:
    """The measured rows (docs/evidence), refit: the gate keeps only what survives the holdout.

    `score` is the type whose held-out ECE improves — and only under the `entropy` statistic, which
    is why the mode selection exists at all; `choice` and `noul` are rejected (the fit chose the
    identity there, and `margin`'s +0.003 improvement on `choice` is under the margin a switch
    needs). Pinned so a change to the gate has to face the published data.
    """
    if not EVIDENCE_ROWS.exists():            # a checkout without the evidence artifacts
        pytest.skip("docs/evidence/e2p5_rows_qwen08.json is not present")
    table = calibrate.fit_table(calibrate.load_rows(EVIDENCE_ROWS), model_key=QWEN_KEY)
    assert table.accepted_types == ("score",)
    assert table.types["score"].mode == "entropy"
    assert table.types["score"].temperature == pytest.approx(1.6475, abs=0.002)
    assert table.types["score"].holdout_before["ece"] == pytest.approx(0.4672, abs=0.001)
    assert table.types["score"].holdout_after["ece"] == pytest.approx(0.4479, abs=0.001)
    assert table.types["choice"].accepted is False
    assert table.types["choice"].modes["margin"]["holdout_ece_improvement"] == pytest.approx(
        0.0031, abs=0.001)
    assert table.types["noul"].accepted is False
    assert table.params_hash.startswith("sha256:")
    assert table.to_json()["accepted"] is True


def test_the_table_applies_only_to_accepted_types() -> None:
    accepted = calibrate.fit_table(_fit_rows(("choice",), per_type=18, temperature=2.0),
                                   model_key="sha256:test")
    rejected = calibrate.fit_table(_flat_rows(per_type=18), model_key="sha256:test")
    probabilities = [0.6, 0.3, 0.1]
    applied = accepted.apply(probabilities, "choice")
    untouched = rejected.apply(probabilities, "choice")
    assert applied["calibrated"] is True
    assert applied["probabilities"] != pytest.approx(probabilities)
    assert applied["temperature"] == accepted.types["choice"].temperature
    assert untouched["calibrated"] is False
    assert untouched["probabilities"] == pytest.approx(probabilities)
    assert untouched["confidence"] == pytest.approx(readout.confidence(probabilities))


def test_applying_the_table_never_changes_the_decision() -> None:
    """Calibration rescales the reported probabilities; the answer stays the answer."""
    table = calibrate.fit_table(_fit_rows(("choice",), per_type=18, temperature=2.0),
                                model_key="sha256:test")
    for probabilities in ([0.6, 0.3, 0.1], [0.34, 0.33, 0.33], [0.4, 0.4, 0.2], [0.2, 0.5, 0.3]):
        applied = table.apply(probabilities, "choice")
        assert readout.argmax_first(applied["probabilities"]) == \
            readout.argmax_first(probabilities)
        assert sum(applied["probabilities"]) == pytest.approx(1.0)


def test_applying_an_unknown_question_type_is_a_no_op() -> None:
    table = calibrate.fit_table(_fit_rows(("choice",), per_type=18, temperature=2.0),
                                model_key="sha256:test")
    applied = table.apply([0.5, 0.5], "noul")
    assert applied["calibrated"] is False and applied["probabilities"] == [0.5, 0.5]


def test_the_table_round_trips_through_json() -> None:
    table = calibrate.fit_table(_fit_rows(("choice", "noul"), per_type=18, temperature=2.0),
                                model_key="sha256:test", model_path="/models/x.gguf",
                                alias="spark")
    payload = table.to_json()
    again = calibrate.Table.from_json(payload)
    assert again.params_hash == table.params_hash
    assert again.to_json() == payload
    assert again.apply([0.6, 0.3, 0.1], "choice")["probabilities"] == pytest.approx(
        table.apply([0.6, 0.3, 0.1], "choice")["probabilities"])
    assert again.model_path == "/models/x.gguf" and again.alias == "spark"


def test_a_re_run_on_the_same_rows_hashes_identically() -> None:
    """A-E2p5-6: same dev set + same model => identical params hash (any row order)."""
    rows = _fit_rows(("choice", "noul"), per_type=18, temperature=2.0)
    first = calibrate.fit_table(rows, model_key="sha256:test")
    second = calibrate.fit_table(list(reversed(rows)), model_key="sha256:test")
    assert first.params_hash == second.params_hash
    assert first.params_hash.startswith("sha256:")
    assert first.to_json()["params"]["types"] == second.to_json()["params"]["types"]


def test_the_hash_covers_the_parameters_the_model_and_the_dev_set() -> None:
    rows = _fit_rows(("choice",), per_type=18, temperature=2.0)
    base = calibrate.fit_table(rows, model_key="sha256:test")
    other_model = calibrate.fit_table(rows, model_key="sha256:other")
    other_set = calibrate.fit_table(rows[:12], model_key="sha256:test")
    assert base.params_hash != other_model.params_hash
    assert base.params_hash != other_set.params_hash


def test_the_hash_ignores_the_fields_the_fit_never_reads() -> None:
    """A-E2p5-6: a re-measurement that moves only coverage/reliability keeps the same parameters.

    The digest covers exactly what `_index_of`/`fit_temperature` consume (id, type, expected,
    correct, probabilities) — a live re-run whose `coverage` float or `reliability` label moved
    would otherwise produce a different hash for byte-identical *fit input*, which is the one
    thing "same set + same model ⇒ identical params hash" must not do.
    """
    import dataclasses
    rows = _fit_rows(("choice",), per_type=18, temperature=2.0)
    drifted = [dataclasses.replace(row, coverage=(row.coverage or 0.0) + 1e-9,
                                   reliability="low_mass" if index % 3 else "ok",
                                   confidence=row.confidence + 1e-6)
               for index, row in enumerate(rows)]
    base = calibrate.fit_table(rows, model_key="sha256:test")
    same = calibrate.fit_table(drifted, model_key="sha256:test")
    assert same.devset_digest == base.devset_digest
    assert same.params_hash == base.params_hash
    moved = calibrate.fit_table([dataclasses.replace(rows[0], probabilities=(0.5, 0.5, 0.0))]
                                + rows[1:], model_key="sha256:test")
    assert moved.params_hash != base.params_hash


def test_the_hash_does_not_depend_on_when_the_fit_ran() -> None:
    import dataclasses
    table = calibrate.fit_table(_fit_rows(("choice",), per_type=18, temperature=2.0),
                                model_key="sha256:test")
    later = dataclasses.replace(table, created_at="2001-01-01T00:00:00Z")
    assert later.params_hash == table.params_hash
    assert later.to_json()["params_hash"] == table.params_hash


def test_the_store_round_trips_one_table(tmp_path) -> None:
    table = calibrate.fit_table(_fit_rows(("choice",), per_type=18, temperature=2.0),
                                model_key="sha256:test")
    path = tmp_path / "calibration.json"
    assert calibrate.save_table(path, table) == path
    loaded = calibrate.load_table(path, "sha256:test")
    assert loaded is not None and loaded.params_hash == table.params_hash
    assert calibrate.load_table(path, "sha256:unknown") is None
    assert calibrate.load_table(tmp_path / "missing.json", "sha256:test") is None


def test_nothing_is_stored_when_no_type_was_accepted(tmp_path) -> None:
    table = calibrate.fit_table(_flat_rows(per_type=18), model_key="sha256:test")
    path = tmp_path / "calibration.json"
    assert calibrate.save_table(path, table) is None
    assert not path.exists()


def test_a_rejected_fit_retires_a_previously_stored_table(tmp_path) -> None:
    """A measurement that no longer supports the parameters must not leave them in force."""
    path = tmp_path / "calibration.json"
    good = calibrate.fit_table(_fit_rows(("choice",), per_type=18, temperature=2.0),
                               model_key="sha256:test")
    other = calibrate.fit_table(_fit_rows(("choice",), per_type=18, temperature=2.0),
                                model_key="sha256:other")
    assert calibrate.save_table(path, good) == path
    assert calibrate.save_table(path, other) == path
    assert calibrate.load_table(path, "sha256:test") is not None
    rejected = calibrate.fit_table(_flat_rows(per_type=18), model_key="sha256:test")
    assert calibrate.save_table(path, rejected) is None
    assert calibrate.load_table(path, "sha256:test") is None
    assert calibrate.load_table(path, "sha256:other") is not None


def test_a_corrupt_store_is_reported_not_ignored(tmp_path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="calibration"):
        calibrate.load_table(path, "sha256:test")


def test_the_dry_run_table_names_every_type_and_the_verdict() -> None:
    table = calibrate.fit_table(_fit_rows(("choice", "noul"), per_type=18, temperature=2.0),
                                model_key="sha256:test")
    text = calibrate.render_table(table)
    assert "choice" in text and "noul" in text
    assert "temperature" in text and "holdout" in text
    assert "applied" in text
    assert table.params_hash in text


# ------------------------------------------- the `calibrate` command (A-E2p5-1)
def _report_file(tmp_path, items: list[dict], name: str = "e2_calibration.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"schema": "ggufone.bench/v1", "suite": "calibration",
                                "items": items}), encoding="utf-8")
    return path


def _model_file(tmp_path, name: str = "tiny.gguf"):
    path = tmp_path / name
    path.write_bytes(b"GGUF" + b"\0" * 32)
    return path


def test_the_calibrate_command_fits_a_report_and_stores_it(tmp_path, monkeypatch, capsys) -> None:
    from ggufone import cli
    from ggufone.registry import store
    home = tmp_path / "home"
    monkeypatch.setenv("GGUFONE_HOME", str(home))
    report = _report_file(tmp_path, _report_rows(("choice", "noul"), per_type=18,
                                                 temperature=2.0))
    model = _model_file(tmp_path)
    code = cli.main(["calibrate", "--model", str(model), "--from-report", str(report),
                     "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["accepted"] is True
    assert payload["params_hash"].startswith("sha256:")
    stored = calibrate.load_table(store.calibration_path(home), payload["model"])
    assert stored is not None and stored.params_hash == payload["params_hash"]


def test_the_calibrate_dry_run_prints_the_table_and_stores_nothing(tmp_path, monkeypatch,
                                                                  capsys) -> None:
    from ggufone import cli
    from ggufone.registry import store
    home = tmp_path / "home"
    monkeypatch.setenv("GGUFONE_HOME", str(home))
    report = _report_file(tmp_path, _report_rows(("choice",), per_type=18, temperature=2.0))
    model = _model_file(tmp_path)
    code = cli.main(["calibrate", "--model", str(model), "--from-report", str(report),
                     "--dry-run"])
    assert code == 0
    out = capsys.readouterr().out
    assert "temperature" in out and "choice" in out and "applied" in out
    assert not store.calibration_path(home).exists()


def test_the_calibrate_command_reports_when_nothing_was_stored(tmp_path, monkeypatch,
                                                               capsys) -> None:
    from ggufone import cli
    from ggufone.registry import store
    home = tmp_path / "home"
    monkeypatch.setenv("GGUFONE_HOME", str(home))
    items = [row.to_json() for row in _flat_rows(per_type=18)]
    report = _report_file(tmp_path, items)
    code = cli.main(["calibrate", "--model", str(_model_file(tmp_path)),
                     "--from-report", str(report)])
    assert code == 1
    assert "no calibration applied" in capsys.readouterr().out
    assert not store.calibration_path(home).exists()


def test_the_calibrate_command_needs_a_model(tmp_path, monkeypatch, capsys) -> None:
    from ggufone import cli
    monkeypatch.setenv("GGUFONE_HOME", str(tmp_path / "empty-home"))
    assert cli.main(["calibrate"]) == 2
    assert "E_MODEL_NOT_FOUND" in capsys.readouterr().err


def _fake_serving_path(monkeypatch, tmp_path) -> None:
    """Patch the model out of `calibrate`'s live path: the same code, a deterministic session."""
    import contextlib

    from ggufone import cli
    from tests.fake_engine import FakeSession, biased_row

    plan = type("Plan", (), {"n_ctx": 4096, "to_dict": lambda self: {}})()
    monkeypatch.setattr(cli, "fit_plan_for", lambda *a, **k: plan)
    session = FakeSession(n_vocab=8192)
    session.row_fn = lambda ctx, session=session: biased_row(session.n_vocab, {})

    @contextlib.contextmanager
    def fake_open_model(*args, **kwargs):
        yield session

    @contextlib.contextmanager
    def fake_session(handle, plan_, **kwargs):
        yield handle

    monkeypatch.setattr(cli.session_module, "open_model", fake_open_model)
    monkeypatch.setattr(cli.session_module, "ModelSession", fake_session)
    monkeypatch.setattr(cli.session_module, "runtime_backend", lambda home=None: "cpu")
    monkeypatch.setenv("GGUFONE_HOME", str(tmp_path / "home"))


def test_the_calibrate_command_writes_the_measured_rows_and_refits_them_identically(
        tmp_path, monkeypatch, capsys) -> None:
    """`--out` is the raw artifact: the same rows refit from the file give the same params hash."""
    from ggufone import cli
    from ggufone.registry import store
    _fake_serving_path(monkeypatch, tmp_path)
    rows_out = tmp_path / "rows.json"
    model = _model_file(tmp_path)
    code = cli.main(["calibrate", "--model", str(model), "--items", "18",
                     "--out", str(rows_out), "--json"])
    first = json.loads(capsys.readouterr().out)
    assert rows_out.exists()
    rows = json.loads(rows_out.read_text(encoding="utf-8"))
    assert rows["suite"] == "calibration" and len(rows["items"]) == 18
    assert code == (0 if first["accepted"] else 1)
    # refit the very same rows from disk: the stored artifact must reproduce the parameters
    code = cli.main(["calibrate", "--model", str(model), "--from-report", str(rows_out),
                     "--json"])
    second = json.loads(capsys.readouterr().out)
    assert code == (0 if second["accepted"] else 1)
    assert second["params_hash"] == first["params_hash"]
    assert second["params"]["types"] == first["params"]["types"]
    assert store.calibration_path(tmp_path / "home").exists() == first["accepted"]


def test_the_calibrate_command_measures_the_committed_dev_set_without_a_report(
        tmp_path, monkeypatch, capsys) -> None:
    """Without `--from-report` the dev set is re-measured through the serving path."""
    from ggufone import cli
    _fake_serving_path(monkeypatch, tmp_path)
    code = cli.main(["calibrate", "--model", str(_model_file(tmp_path)), "--items", "18",
                     "--dry-run", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["devset"]["items"] == 18
    assert payload["model"].endswith("tiny.gguf:36")
    assert payload["schema"] == "ggufone.calibration/v1"


# ------------------------------------------------- the readout hook (A-E2p5-1)
def _choice_request(options: dict | None = None):
    from ggufone import schema
    payload = {"state": "The checkout page returns HTTP 500 for every customer.",
               "questions": {"area": {"type": "choice",
                                      "criteria": {"billing": None, "api": None, "sales": None}}}}
    if options:
        payload["options"] = options
    return schema.parse_request(payload)


def _biased_session(labels: tuple[str, ...] = ("billing", "api", "sales")):
    from tests.fake_engine import FakeSession, biased_row
    session = FakeSession(n_vocab=512)
    biases = {}
    for index, label in enumerate(labels):
        biases[session.tokenize(label)[0]] = 6.0 - 0.7 * index
    session.row_fn = lambda ctx, biases=biases, session=session: biased_row(session.n_vocab,
                                                                           biases)
    return session


def _table_for_fake_model(temperature: float = 2.0):
    return calibrate.fit_table(_fit_rows(("choice",), per_type=18, temperature=temperature),
                               model_key="file:tiny.gguf:36")


def test_a_calibrated_table_scales_the_readout_and_marks_the_response() -> None:
    from ggufone.engine import decide
    table = _table_for_fake_model()
    baseline = decide.DecisionEngine(_biased_session()).decide(_choice_request())
    engine = decide.DecisionEngine(_biased_session(), calibration=table)
    result = engine.decide(_choice_request())
    payload = result.payload()
    assert payload["calibrated"] is True
    assert payload["calibration"]["applied"] is True
    assert payload["calibration"]["temperatures"]["choice"] == table.types["choice"].temperature
    assert payload["calibration"]["params_hash"] == table.params_hash
    raw = payload["answers"]["area"]["probabilities"]
    before = baseline.payload()["answers"]["area"]["probabilities"]
    expected = table.apply(list(before.values()), "choice")["probabilities"]
    assert list(raw) == list(before)
    for key, value in zip(raw, expected, strict=True):
        assert raw[key] == pytest.approx(value)
    assert raw["billing"] > before["billing"]           # a soft run gets sharpened
    assert payload["answers"]["area"]["choice"] == baseline.payload()["answers"]["area"]["choice"]
    assert baseline.payload()["calibrated"] is False
    assert baseline.payload()["calibration"]["applied"] is False


def test_the_engine_ignores_a_table_that_was_never_accepted() -> None:
    from ggufone.engine import decide
    table = calibrate.fit_table(_flat_rows(per_type=18), model_key="file:tiny.gguf:36")
    payload = decide.DecisionEngine(_biased_session(), calibration=table).decide(
        _choice_request()).payload()
    assert table.accepted is False
    assert payload["calibrated"] is False
    assert payload["calibration"]["source"] == ""
    assert payload["calibration"]["applied"] is False


def _score_request(confidence_mode: str | None = None):
    from ggufone import schema
    payload = {"state": "The checkout page returns HTTP 500 for every customer.",
               "questions": {"sev": {"type": "score",
                                     "criteria": ["cosmetic", "degraded", "blocking"]}}}
    if confidence_mode is not None:
        payload["options"] = {"confidence_mode": confidence_mode}
    return schema.parse_request(payload)


def test_a_promoted_mode_is_the_reported_statistic_unless_the_request_names_one() -> None:
    """A-E2p5-3 with the live table: `score` was accepted under `entropy`, so that is its
    confidence — and an explicit `confidence_mode` in the request still wins."""
    from ggufone.engine import decide
    table = calibrate.fit_table(calibrate.load_rows(EVIDENCE_ROWS), model_key=QWEN_KEY)
    assert table.mode_for("score") == "entropy"
    payload = decide.DecisionEngine(_biased_session(("0", "1", "2")),
                                    calibration=table).decide(_score_request()).payload()
    probabilities = list(payload["answers"]["sev"]["probabilities"].values())
    assert payload["calibration"]["confidence_modes"] == {"score": "entropy"}
    assert payload["calibration"]["temperatures"]["score"] == table.types["score"].temperature
    assert payload["answers"]["sev"]["confidence"] == pytest.approx(
        readout.confidence(probabilities, "entropy"))
    assert payload["answers"]["sev"]["confidence"] != pytest.approx(
        readout.confidence(probabilities, "normalized_peak"))
    named = decide.DecisionEngine(_biased_session(("0", "1", "2")),
                                  calibration=table).decide(
        _score_request("normalized_peak")).payload()
    assert named["calibration"]["confidence_modes"] == {"score": "normalized_peak"}
    assert named["answers"]["sev"]["confidence"] == pytest.approx(
        readout.confidence(list(named["answers"]["sev"]["probabilities"].values()),
                           "normalized_peak"))
