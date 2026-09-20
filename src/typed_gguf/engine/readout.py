"""Readout math — the part of the engine that is pure arithmetic (SPEC 2.3 steps 5-7, 2.4).

Transplanted from the oracle mirror (`docs/verify_runtime_contract.py` sections C/D execute
these values). Everything here is a pure function over synthetic or real logits, so the whole
decision math is testable without a model, a runtime or a GPU.

Contract (FROZEN):
  restricted_softmax(logits, T)      p_c = exp(z_c/T) / sum_j exp(z_j/T)          sum(p) = 1
  confidence_normalized_peak(probs)  (max(p) - 1/K) / (1 - 1/K), clamped to [0,1]
  confidence_entropy(probs)          1 - H / log K, clamped to [0,1]              (E2.5 mode)
  confidence_margin(probs)           (p1 - p2) / (1 - p2), clamped to [0,1]       (E2.5 mode)
  score_weighted_mean(probs)         sum(i * p_i)                                 score in [0, K-1]
  coverage(cands, full_row)          sum of the FULL-vocab softmax mass of the candidate tokens
  candidate_sequence_score(logps, n) sum(logps) / len(logps) ** n
  argmax_first(values)               argmax with the FROZEN tie-break: lowest index

The three names pinned by `docs/verify_runtime_contract.py` section D
(`restricted_softmax`, `confidence_normalized_peak`, `score_weighted_mean`) match the mirror
exactly — the oracle re-derives them from this module.

`coverage` is computed from the un-renormalized full-vocab row (SPEC 2.3 step 6): a low-mass
answer is reported (`reliability = "low_mass"` + `W_LOW_MASS`), never hidden by renormalizing
the restricted softmax. There is no code path here that rescales `probabilities` to hide it.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence


def confidence_normalized_peak(probs: Sequence[float]) -> float:
    """Excess of the peak over uniform, rescaled to [0, 1] (SPEC 2.4)."""
    k = len(probs)
    if k <= 1:
        return 1.0
    pmax = max(probs)
    return _clamp01((pmax - 1.0 / k) / (1.0 - 1.0 / k))


def confidence_entropy(probs: Sequence[float]) -> float:
    """1 - H/log K: 1.0 for a point mass, 0.0 for a uniform distribution (SPEC 2.10)."""
    k = len(probs)
    if k <= 1:
        return 1.0
    h = -sum(p * math.log(p) for p in probs if p > 0.0)
    return _clamp01(1.0 - h / math.log(k))


def confidence_margin(probs: Sequence[float]) -> float:
    """(p1 - p2) / (1 - p2): the lead of the top candidate over the runner-up (SPEC 2.10)."""
    if len(probs) < 2:
        return 0.0
    ordered = sorted(probs, reverse=True)
    p1, p2 = ordered[0], ordered[1]
    if p2 >= 1.0:
        return 0.0
    return _clamp01((p1 - p2) / (1.0 - p2))


CONFIDENCE_MODES: dict[str, Callable[[Sequence[float]], float]] = {
    "normalized_peak": confidence_normalized_peak,
    "entropy": confidence_entropy,
    "margin": confidence_margin,
}
DEFAULT_CONFIDENCE_MODE = "normalized_peak"


def confidence(probs: Sequence[float], mode: str = DEFAULT_CONFIDENCE_MODE) -> float:
    """Dispatch to the configured confidence statistic; unknown modes raise KeyError."""
    return CONFIDENCE_MODES[mode](probs)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def softmax(values: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Numerically stable softmax over the given values (no renormalization of anything else)."""
    if not values:
        raise ValueError("softmax needs at least one value")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    m = max(values)
    exps = [math.exp((v - m) / temperature) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def restricted_softmax(values: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Softmax restricted to one question's candidates (SPEC 2.3 step 5)."""
    return softmax(values, temperature)


def logsumexp(values: Sequence[float]) -> float:
    """log(sum(exp(v))) without overflow; -inf for an empty sequence."""
    if not values:
        return -math.inf
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def logprob(row: Sequence[float], token_id: int) -> float:
    """log p(token_id) under the FULL vocab softmax of `row` (SPEC 2.3 step 6)."""
    return float(row[token_id]) - logsumexp(row)


def coverage(candidate_logits: Sequence[float], full_logits: Sequence[float]) -> float:
    """Oracle-mirror signature: mass of the first len(candidate_logits) full-vocab slots.

    Kept for parity with `docs/verify_runtime_contract.py` section C; the engine uses
    `coverage_from_row` (arbitrary vocab ids).
    """
    p_full = softmax(full_logits)
    return sum(p_full[i] for i in range(len(candidate_logits)))


def coverage_from_row(full_row: Sequence[float], token_ids: Sequence[int]) -> float:
    """Answer mass at the *decision row*: sum of softmax_full(row)[token_id] (SPEC 2.3 step 6).

    Computed from the full-vocab row before any renormalization, so a candidate with low mass
    reports low coverage instead of a healthy-looking renormalized probability. The value is
    capped at 1.0 (a caller that passes the same token twice would otherwise exceed it).
    """
    if not token_ids:
        return 0.0
    p_full = softmax(full_row)
    return min(1.0, sum(p_full[t] for t in token_ids))


def candidate_sequence_score(logprobs: Sequence[float], length_norm: float = 1.0) -> float:
    """z_c = sum(log p(tok_i | prefix, tok_<i)) / L ** length_norm (SPEC 2.3 step 4)."""
    if not logprobs:
        raise ValueError("a candidate must score at least one token")
    return sum(logprobs) / (len(logprobs) ** length_norm)


def score_weighted_mean(probs: Sequence[float]) -> float:
    """Score = sum(level_index * p_level) — the documented TypeSafe rule (SPEC 2.4, 2.6).

    Returned at the WIRE precision (6 significant decimals, SPEC 2.5: "numbers are rounded to
    6 significant decimals in JSON to keep runs byte-comparable"). This is not cosmetic: the
    oracle pins `score([0, .7, .3]) == 1.30` exactly, and the raw IEEE-754 sum of the levels is
    1.2999999999999998 (0.7 + 0.6 rounds to the even mantissa). Rounding here makes the frozen
    reference value reproducible instead of "1.30 within 1e-16".
    """
    return round_sig(sum(i * p for i, p in enumerate(probs)), 6)


def round_sig(value: float, digits: int = 6) -> float:
    """Round to `digits` significant decimals — the frozen JSON precision (SPEC 2.5)."""
    return float(f"{value:.{digits}g}")


def argmax_first(values: Sequence[float]) -> int:
    """argmax with the FROZEN tie-break: the lowest candidate index wins (SPEC 2.3 step 7)."""
    if not values:
        raise ValueError("argmax needs at least one value")
    best, best_value = 0, values[0]
    for index in range(1, len(values)):
        if values[index] > best_value:
            best, best_value = index, values[index]
    return best


def logprob_from_scale(row: Sequence[float], token_id: int, scale: float) -> float:
    """`logprob` with the row's logsumexp computed once by the caller (engine hot path)."""
    return float(row[token_id]) - scale


def coverage_from_scale(row: Sequence[float], token_ids: Sequence[int], scale: float) -> float:
    """`coverage_from_row` with the row's logsumexp computed once by the caller."""
    if not token_ids:
        return 0.0
    total = 0.0
    for token_id in token_ids:
        total += math.exp(float(row[token_id]) - scale)
    return min(1.0, total)


def reliability(coverage_value: float, *, coverage_floor: float = 0.10,
                low_confidence: bool = False, confidence_floor: float | None = None,
                confidence_value: float | None = None) -> str:
    """`ok | low_mass | low_confidence` (SPEC 2.5 response field).

    `low_mass` wins over `low_confidence`: missing answer mass is the failure the caller can
    still fix by changing the criteria; a flat distribution is the model's own opinion.
    """
    if coverage_value < coverage_floor:
        return "low_mass"
    if low_confidence:
        return "low_confidence"
    if confidence_floor is not None and confidence_value is not None \
            and confidence_value < confidence_floor:
        return "low_confidence"
    return "ok"
