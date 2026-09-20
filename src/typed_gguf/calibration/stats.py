"""Agreement, Wilson CI, ECE, reliability bins — the calibration math (SPEC 2.10).

Milestone: E2.5.

Everything here is a pure function over stored distributions, so the whole calibration surface
(A-E2p5-1/2/3/6) is testable and reproducible without a model, a runtime or a GPU.

Three decisions worth naming before editing:

* **One definition of ECE.** `bins`/`ece` reproduce `bench.harness.reliability_bins`/`ece`
  exactly (a test cross-checks the two implementations): the published calibration table and the
  acceptance gate must never disagree about what "calibrated" means.
* **The fit can only improve, never silently degrade.** `DEFAULT_GRID` always contains
  `temperature == 1.0`, and a grid point is only eligible while its NLL does not get worse than
  the identity's. A fit therefore never returns a parameter set that is worse than doing nothing
  — which is exactly what `A-E2p5-2` ("otherwise report *no calibration applied*") needs.
* **Calibration is a monotone map on one question's candidates** (`power_scale`). It re-scales
  the reported probabilities and the confidence, never the ranking: the answer cannot change.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from typed_gguf.engine import readout

SCHEMA = "typed_gguf.calibration/v1"
N_BINS = 10
#: 61 log-spaced temperatures in [0.05, 20] centred on the identity (index 30 == 1.0 exactly).
DEFAULT_GRID: tuple[float, ...] = tuple(
    math.exp(math.log(0.05) + (math.log(20.0) - math.log(0.05)) * index / 60.0)
    for index in range(61))
#: the "agreement at a fixed threshold" alternative of A-E2p5-2
FIXED_THRESHOLD = 0.5
#: floats are rounded to this many significant digits before hashing: the last bits of an IEEE
#: sum are not data, but a real parameter change must still move the hash.
HASH_DIGITS = 12
NLL_TOLERANCE = 1e-12


# ------------------------------------------------------------------ bins / ECE
def bins(confidences: Sequence[float], correct: Sequence[bool], *,
         n_bins: int = N_BINS) -> list[dict[str, Any]]:
    """Equal-width confidence bins with accuracy, mean confidence and their gap.

    Same shape and same arithmetic as `bench.harness.reliability_bins` (empty bins keep their
    place so a table keeps its shape); `gap = accuracy - mean_confidence`, positive = the model
    is under-confident in that bin.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    buckets: list[dict[str, Any]] = [
        {"lo": index / n_bins, "hi": (index + 1) / n_bins, "n": 0, "correct": 0,
         "mean_confidence": None, "accuracy": None, "gap": None}
        for index in range(n_bins)]
    sums = [0.0] * n_bins
    for confidence, hit in zip(confidences, correct, strict=True):
        value = _clamp01(float(confidence))
        index = min(int(value * n_bins), n_bins - 1)
        bucket = buckets[index]
        bucket["n"] += 1
        bucket["correct"] += int(bool(hit))
        sums[index] += value
    for bucket, total in zip(buckets, sums, strict=True):
        if bucket["n"]:
            bucket["mean_confidence"] = total / bucket["n"]
            bucket["accuracy"] = bucket["correct"] / bucket["n"]
            bucket["gap"] = bucket["accuracy"] - bucket["mean_confidence"]
    return buckets


def ece(bins: Sequence[Mapping[str, Any]]) -> float:
    """Expected calibration error: the sample-weighted mean of `|accuracy - confidence|`."""
    total = sum(int(entry["n"]) for entry in bins)
    if total == 0:
        return 0.0
    return sum(int(entry["n"]) * abs(float(entry["accuracy"]) - float(entry["mean_confidence"]))
               for entry in bins if entry["n"]) / total


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The 95 % Wilson score interval for a binomial proportion (0/0 -> the full range)."""
    if n <= 0:
        return (0.0, 1.0)
    phat = successes / n
    denominator = 1.0 + (z * z) / n
    centre = (phat + (z * z) / (2 * n)) / denominator
    margin = (z * math.sqrt((phat * (1 - phat) + (z * z) / (4 * n)) / n)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def agreement_at(confidences: Sequence[float], correct: Sequence[bool], threshold: float, *,
                 z: float = 1.96) -> dict[str, Any]:
    """Agreement restricted to the answers whose confidence is at or above `threshold`.

    This is the second acceptance signal of A-E2p5-2 ("ECE *or* agreement at a fixed threshold"):
    a calibrated run that keeps every decision and reports fewer confident-but-wrong answers
    improves the threshold agreement even when the binned ECE is unchanged.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    selected = [hit for confidence, hit in zip(confidences, correct, strict=True)
                if float(confidence) >= float(threshold)]
    hits = sum(1 for hit in selected if hit)
    low, high = wilson(hits, len(selected), z=z)
    return {"threshold": float(threshold), "n": len(selected), "correct": hits,
            "agreement": (hits / len(selected)) if selected else 0.0, "ci": [low, high]}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


# ------------------------------------------------------------- temperature math
def power_scale(probabilities: Sequence[float], temperature: float) -> list[float]:
    """`softmax(log(p) / T)`: temperature scaling on a restricted distribution.

    `T > 1` flattens, `T < 1` sharpens, `T = 1` is the identity. Computed in log space with the
    max subtracted, so a tiny probability cannot underflow the whole row; a zero probability
    stays zero (the log form drops it instead of inventing mass for it).
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if not probabilities:
        raise ValueError("power_scale needs at least one probability")
    logs = [math.log(float(p)) / temperature if float(p) > 0.0 else -math.inf
            for p in probabilities]
    top = max(logs)
    if not math.isfinite(top):
        raise ValueError("power_scale needs at least one positive probability")
    exps = [math.exp(value - top) if value > -math.inf else 0.0 for value in logs]
    total = sum(exps)
    return [value / total for value in exps]


def nll(probabilities: Sequence[float], index: int) -> float:
    """`-log p[index]` — the negative log-likelihood of the correct candidate."""
    value = float(probabilities[index])
    return -math.log(value) if value > 0.0 else math.inf


def evaluate(rows: Sequence[Sequence[float]], correct: Sequence[int], temperature: float, *,
             mode: str = readout.DEFAULT_CONFIDENCE_MODE,
             n_bins: int = N_BINS) -> dict[str, Any]:
    """Every statistic the report publishes for ONE temperature and ONE confidence mode."""
    confidences: list[float] = []
    hits: list[bool] = []
    losses: list[float] = []
    wins = 0
    for probabilities, index in zip(rows, correct, strict=True):
        scaled = power_scale(probabilities, temperature)
        confidences.append(readout.confidence(scaled, mode))
        wins += int(readout.argmax_first(scaled) == index)
        hits.append(readout.argmax_first(scaled) == index)
        losses.append(nll(scaled, index))
    n = len(confidences)
    return {
        "temperature": float(temperature),
        "mode": mode,
        "n": n,
        "ece": ece(bins(confidences, hits, n_bins=n_bins)),
        "nll": (sum(losses) / n) if n else 0.0,
        "mean_confidence": (sum(confidences) / n) if n else None,
        "agreement": (wins / n) if n else 0.0,
        "agreement_at_0.5": agreement_at(confidences, hits, FIXED_THRESHOLD),
    }


def fit_temperature(rows: Sequence[Sequence[float]], correct: Sequence[int], *,
                    grid: Sequence[float] = DEFAULT_GRID,
                    mode: str = readout.DEFAULT_CONFIDENCE_MODE,
                    n_bins: int = N_BINS) -> dict[str, Any]:
    """Fit one temperature by minimising ECE under an NLL guard (A-E2p5-1/2/6).

    Eligible grid points are those whose NLL is not worse than the identity's; among them the
    lowest ECE wins, ties (and near-ties under `NLL_TOLERANCE`) break towards `T = 1.0`. The
    result carries the before/after statistics the report needs, so a caller never has to
    re-derive them from the rows.
    """
    if not rows or len(rows) != len(correct):
        raise ValueError("fit_temperature needs the same non-zero number of rows and answers")
    if mode not in readout.CONFIDENCE_MODES:
        raise KeyError(f"unknown confidence mode {mode!r}")
    candidates = [float(value) for value in grid]
    if not candidates or any(value <= 0.0 for value in candidates):
        raise ValueError("the temperature grid must be a non-empty list of positive values")
    baseline = evaluate(rows, correct, 1.0, mode=mode, n_bins=n_bins)
    best: dict[str, Any] | None = None
    best_key: tuple[float, float, float] | None = None
    for temperature in candidates:
        evaluated = evaluate(rows, correct, temperature, mode=mode, n_bins=n_bins)
        if evaluated["nll"] > baseline["nll"] + NLL_TOLERANCE:
            continue                                  # the NLL guard: never trade NLL for ECE
        key = (evaluated["ece"], abs(math.log(temperature)), evaluated["nll"])
        if best_key is None or key < best_key:
            best, best_key = evaluated, key
    if best is None:                                  # only possible with an empty eligible set
        best = baseline
    return {
        "temperature": best["temperature"],
        "mode": mode,
        "n": best["n"],
        "grid_points": len(candidates),
        "ece": best["ece"],
        "ece_before": baseline["ece"],
        "nll": best["nll"],
        "nll_before": baseline["nll"],
        "mean_confidence": best["mean_confidence"],
        "agreement": best["agreement"],
        "agreement_before": baseline["agreement"],
        "agreement_at_0.5": best["agreement_at_0.5"],
        "agreement_at_0.5_before": baseline["agreement_at_0.5"],
    }


# ------------------------------------------------------------- reproducibility
def _canonical_value(value: Any) -> Any:
    if isinstance(value, float):
        return readout.round_sig(value, HASH_DIGITS)
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


def canonical(payload: Any) -> str:
    """The canonical JSON of `payload`: sorted keys, no whitespace, floats rounded to 12 digits."""
    return json.dumps(_canonical_value(payload), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def params_hash(payload: Any) -> str:
    """`sha256:<hex>` of the canonical parameters — A-E2p5-6's reproducibility receipt.

    Hashes the *parameters*, not the run: the same dev set on the same model fits the same
    temperature and therefore produces the same digest, on any machine and in any process
    (`PYTHONHASHSEED` does not reach `sort_keys`).
    """
    return "sha256:" + hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()
