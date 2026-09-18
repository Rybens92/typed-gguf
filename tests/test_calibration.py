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
    """The gate's honesty floor: `temperature = 1.0` is always in the grid (and is the reference)."""
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
