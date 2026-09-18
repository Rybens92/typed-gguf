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


# ---------------------------------------------- the fit, the gate, the table (A-E2p5-1/2/3/6)
def _calibrated_rows(types: tuple[str, ...] = ("choice",), *, per_type: int = 12,
                     confidence: float = 0.5) -> list:
    """Rows whose confidence *is* their accuracy: ECE is exactly zero at temperature 1.0.

    Half the rows hit and half miss, and every row reports the same confidence, so any move away
    from the identity makes the ECE worse — the fit has to choose "no calibration".
    """
    items = []
    for offset, qtype in enumerate(types):
        for index in range(per_type):
            hit = index % 2 == 0
            if qtype == "choice":
                peak = 1 / 3 + (2 / 3) * confidence
                probabilities = {"billing": peak, "api": (1 - peak) * 2 / 3,
                                 "sales": (1 - peak) / 3}
                expected = "billing"
                got = "billing" if hit else "api"
                if not hit:                       # the miss must be the argmax failure, not the peak
                    probabilities = {"billing": (1 - peak) * 2 / 3, "api": peak,
                                     "sales": (1 - peak) / 3}
            else:
                peak = 0.5 + 0.5 * confidence
                probabilities = {"yes": peak if hit else 1 - peak,
                                 "no": (1 - peak) if hit else peak}
                expected, got = "yes", ("yes" if hit else "no")
            items.append({
                "id": f"cal-{qtype}-{offset}-{index:02d}", "type": qtype, "expected": expected,
                "got": got, "correct": hit, "confidence": confidence, "coverage": 0.9,
                "reliability": "ok", "probabilities": probabilities,
            })
    return [calibrate.Row.from_item(item) for item in items]


def test_the_fit_produces_one_temperature_per_question_type() -> None:
    table = calibrate.fit_table(_fit_rows(("choice", "noul"), per_type=12),
                                model_key="sha256:test")
    assert set(table.types) == {"choice", "noul"}
    assert all(entry.temperature > 0 for entry in table.types.values())
    assert table.mode == readout.DEFAULT_CONFIDENCE_MODE == "normalized_peak"


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
    table = calibrate.fit_table(_calibrated_rows(per_type=18), model_key="sha256:test")
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

    The fit half is miscalibrated (so a temperature other than 1.0 is chosen) while the held-out
    half is already honest — applying the fitted temperature there makes the ECE worse, which is
    exactly the case "record the score and apply nothing" exists for.
    """
    rows = (_renamed(_fit_rows(("choice",), per_type=12, temperature=2.0), "a")
            + _renamed(_calibrated_rows(per_type=6), "z"))
    table = calibrate.fit_table(rows, model_key="sha256:test")
    entry = table.types["choice"]
    assert entry.temperature != 1.0
    assert entry.accepted is False
    assert "held-out split did not improve" in entry.reason
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
                "agreement_holdout_before", "agreement_holdout_after"} <= set(report)
    assert table.mode == "normalized_peak"
    assert table.to_json()["mode"] == "normalized_peak"


def test_the_table_applies_only_to_accepted_types() -> None:
    accepted = calibrate.fit_table(_fit_rows(("choice",), per_type=18, temperature=2.0),
                                   model_key="sha256:test")
    rejected = calibrate.fit_table(_calibrated_rows(per_type=18), model_key="sha256:test")
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
    table = calibrate.fit_table(_calibrated_rows(per_type=18), model_key="sha256:test")
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
    rejected = calibrate.fit_table(_calibrated_rows(per_type=18), model_key="sha256:test")
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
