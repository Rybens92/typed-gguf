"""A-E1b-1 / A-E1b-7 / A-E1b-12: readout math — invariants + the oracle mirror.

Every number in `REFERENCE` is executed by `docs/verify_runtime_contract.py` section C;
section D re-checks three of the functions by importing `ggufone.engine.readout`.
The property tests use synthetic logits only (no model, no runtime).
"""
from __future__ import annotations

import math
import random

import pytest

from ggufone.engine import readout

# (name, candidate logits, full-vocab row, temperature) -> executed reference values
SOFTMAX_REFERENCE = ([2.0, 1.0, 0.0], [0.665241, 0.244728, 0.090031])


def test_restricted_softmax_reproduces_the_oracle_reference_values() -> None:
    p = readout.restricted_softmax([2.0, 1.0, 0.0])
    assert [round(v, 6) for v in p] == SOFTMAX_REFERENCE[1]
    assert abs(sum(p) - 1.0) < 1e-12


def test_restricted_softmax_is_flat_for_flat_logits_and_accepts_temperature() -> None:
    assert readout.restricted_softmax([0.0, 0.0]) == [0.5, 0.5]
    hot = readout.restricted_softmax([1.0, 0.0], temperature=2.0)
    assert hot[0] > 0.5 > hot[1]
    # temperature only rescales the gap: order and sum are preserved
    assert abs(sum(readout.restricted_softmax([3.0, 1.0, -2.0], temperature=0.5)) - 1.0) < 1e-12


def test_confidence_normalized_peak_reference_values() -> None:
    peak = readout.confidence_normalized_peak
    assert peak([1.0, 0.0]) == 1.0
    assert peak([0.5, 0.5]) == 0.0
    assert abs(peak([0.0, 0.7, 0.3]) - 0.55) < 1e-12
    assert peak([1.0]) == 1.0                     # K <= 1 -> 1.0 (SPEC 2.4)
    # the documented doc-capture table (oracle TS_CONFIDENCE_EXAMPLES, tolerance 0.02)
    for label, probs, documented in (
        ("choice/returns-ticket", [0.02, 0.38, 0.6], 0.39),
        ("choice/return_reason", [1.0, 0.0, 0.0, 0.0, 0.0], 1.0),
        ("choice/shipping_issue", [0.63, 0.37, 0.0, 0.0, 0.0], 0.53),
        ("choice/requested_resolution", [0.37, 0.24, 0.29, 0.1], 0.16),
        ("choice/tone", [0.92, 0.08, 0.0], 0.88),
        ("score/bug_severity", [0.0, 0.7, 0.3], 0.54),
        ("score/spinner+examples", [0.0, 0.94, 0.06], 0.91),
    ):
        assert abs(peak(probs) - documented) <= 0.02, label


def test_confidence_modes_exist_and_are_bounded() -> None:
    probs = [0.6, 0.3, 0.1]
    assert readout.CONFIDENCE_MODES["normalized_peak"](probs) == readout.confidence_normalized_peak(probs)
    assert readout.CONFIDENCE_MODES["entropy"](probs) == readout.confidence_entropy(probs)
    assert readout.CONFIDENCE_MODES["margin"](probs) == readout.confidence_margin(probs)
    assert readout.confidence(probs, "entropy") == readout.confidence_entropy(probs)
    for values in ([1.0, 0.0], [0.25] * 4, [0.5, 0.5], [0.34, 0.33, 0.33], [1.0]):
        for fn in readout.CONFIDENCE_MODES.values():
            assert 0.0 <= fn(values) <= 1.0, (fn.__name__, values)
    # entropy: uniform -> 0, point mass -> ~1
    assert readout.confidence_entropy([0.25] * 4) == pytest.approx(0.0, abs=1e-12)
    assert readout.confidence_entropy([1.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0, abs=1e-12)
    # margin: no second candidate -> 0 by definition; the runner-up at 0.0 -> full lead
    assert readout.confidence_margin([1.0]) == 0.0
    assert readout.confidence_margin([1.0, 0.0]) == 1.0
    assert readout.confidence_margin([0.0, 1.0]) == 1.0        # order-independent
    assert readout.confidence_margin([0.5, 0.5]) == 0.0


def test_score_weighted_mean_reference_values() -> None:
    assert readout.score_weighted_mean([0.0, 0.7, 0.3]) == 1.30         # exact: wire precision
    assert readout.score_weighted_mean([0.0, 0.94, 0.06]) == 1.06
    assert readout.score_weighted_mean([1.0, 0.0, 0.0]) == 0.0
    assert readout.round_sig(0.6652409557748219) == 0.665241


def test_coverage_matches_the_full_vocab_softmax_mass() -> None:
    full = [2.0, 1.0, 0.0, -1.0]
    cov = readout.coverage([2.0, 1.0], full)
    assert abs(cov - 0.880797) < 1e-6
    assert abs(cov - sum(readout.softmax(full)[:2])) < 1e-12
    assert 0.0 <= cov <= 1.0


def test_coverage_from_row_uses_token_ids_not_positions() -> None:
    row = [2.0, 1.0, 0.0, -1.0]
    assert readout.coverage_from_row(row, [0, 1]) == pytest.approx(readout.coverage([2.0, 1.0], row))
    # ids out of the prefix select their own slots (engine path: arbitrary vocab ids)
    assert readout.coverage_from_row(row, [3]) == pytest.approx(readout.softmax(row)[3])
    assert readout.coverage_from_row(row, [3, 3]) == pytest.approx(2 * readout.softmax(row)[3])
    assert readout.coverage_from_row(row, []) == 0.0


def test_candidate_sequence_score_reference_values() -> None:
    z = readout.candidate_sequence_score([-1.0, -1.0, -1.0], length_norm=1.0)
    assert abs(z + 1.0) < 1e-12                       # length-neutral at norm 1.0
    assert readout.candidate_sequence_score([-2.0], 0.0) == -2.0   # norm 0 -> plain sum
    assert readout.candidate_sequence_score([-2.0, -4.0], 1.0) == -3.0
    assert readout.candidate_sequence_score([-1.0, -1.0, -1.0], 0.0) == -3.0
    assert readout.candidate_sequence_score([-3.0], 1.0) == -3.0     # single token: length-neutral


def test_logprob_is_the_row_minus_the_full_row_logsumexp() -> None:
    row = [2.0, 1.0, 0.0, -1.0]
    total = sum(math.exp(v) for v in row)
    assert readout.logprob(row, 0) == pytest.approx(math.log(math.exp(2.0) / total))
    for token in range(4):
        p = readout.logprob(row, token)
        assert p <= 0.0
        assert math.exp(p) == pytest.approx(readout.softmax(row)[token])
    # a restricted softmax over per-token logprobs equals the restricted softmax over the
    # subset's raw logits (the full-row logsumexp cancels inside the restricted softmax)
    z = [readout.logprob(row, 0), readout.logprob(row, 1)]
    assert readout.restricted_softmax(z) == pytest.approx(readout.softmax(row[:2]))


def test_argmax_tie_break_is_the_lowest_index() -> None:
    assert readout.argmax_first([0.5, 0.5, 0.5]) == 0
    assert readout.argmax_first([0.1, 0.9, 0.9]) == 1
    assert readout.argmax_first([1.0]) == 0


@pytest.mark.parametrize("seed", range(25))
def test_invariants_over_random_synthetic_logits(seed: int) -> None:
    """A-E1b-1: sum(p) = 1 +/-1e-6, confidence in [0,1], score in [0, K-1], noul in [0,1]."""
    rng = random.Random(seed)
    k = rng.randint(1, 12)
    logits = [rng.uniform(-12.0, 12.0) for _ in range(k)]
    probs = readout.restricted_softmax(logits, temperature=rng.choice([0.5, 1.0, 2.0]))
    assert abs(sum(probs) - 1.0) <= 1e-6
    assert all(p >= 0.0 for p in probs)
    assert readout.argmax_first(probs) == max(range(k), key=lambda i: probs[i])
    conf = readout.confidence_normalized_peak(probs)
    assert 0.0 <= conf <= 1.0
    score = readout.score_weighted_mean(probs)
    assert 0.0 <= score <= k - 1 + 1e-9
    noul = probs[0]                        # noul = P(yes) is a probability, so in [0,1]
    assert 0.0 <= noul <= 1.0
    full = [rng.uniform(-12.0, 12.0) for _ in range(32)]
    cov = readout.coverage_from_row(full, [rng.randrange(32) for _ in range(k)])
    assert 0.0 <= cov <= 1.0


def test_reliability_flags_low_mass_and_never_renormalizes() -> None:
    """A-E1b-7: below the coverage floor -> low_mass (+ W_LOW_MASS at the engine layer)."""
    assert readout.reliability(0.9, coverage_floor=0.10) == "ok"
    assert readout.reliability(0.05, coverage_floor=0.10) == "low_mass"
    assert readout.reliability(0.9, coverage_floor=0.10, low_confidence=True) == "low_confidence"
    # the floor is inclusive: exactly at the floor is not below it
    assert readout.reliability(0.10, coverage_floor=0.10) == "ok"
