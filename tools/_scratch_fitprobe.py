"""Scratch: what the designed test data gives the fit for each true temperature."""
from __future__ import annotations

import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from ggufone.bench import harness                      # noqa: E402
from ggufone.calibration import stats                  # noqa: E402
from ggufone.engine import readout                     # noqa: E402


def distorted_rows(temperature: float, *, n: int = 200, k: int = 4, seed: int = 7):
    state = seed

    def draw() -> float:
        nonlocal state
        state = (1103515245 * state + 12345) % (2 ** 31)
        return state / 2 ** 31

    rows, correct = [], []
    for _ in range(n):
        confidence = 0.5 + 0.5 * draw()
        hit = draw() <= confidence
        peak = 1.0 / k + (1.0 - 1.0 / k) * confidence
        rest_total = 1.0 - peak
        others = [draw() + 0.05 for _ in range(k - 1)]
        total = sum(others)
        probs = [peak] + [rest_total * value / total for value in others]
        rows.append(readout.softmax([__import__("math").log(p) for p in probs], temperature))
        correct.append(0 if hit else 1)
    return rows, correct


for true_temperature in (0.25, 0.5, 1.0, 2.0, 4.0):
    rows, correct = distorted_rows(true_temperature)
    fit = stats.fit_temperature(rows, correct)
    base = stats.evaluate(rows, correct, 1.0)
    print(f"T_true={true_temperature:<5} -> T_fit={fit['temperature']:.4f} "
          f"ece {fit['ece_before']:.4f}->{fit['ece']:.4f} "
          f"nll {fit['nll_before']:.4f}->{fit['nll']:.4f} acc={base['agreement']:.3f}")
    for n in (400, 60):
        rows_n, correct_n = distorted_rows(true_temperature, n=n)
        fit_n = stats.fit_temperature(rows_n, correct_n)
        print(f"    n={n:<4} T_fit={fit_n['temperature']:.4f} "
              f"ece {fit_n['ece_before']:.4f}->{fit_n['ece']:.4f}")
    # cross-check the two ECE implementations on the same rows
    base_conf = [readout.confidence(row, "normalized_peak") for row in rows]
    hits = [readout.argmax_first(row) == index for row, index in zip(rows, correct, strict=True)]
    assert stats.bins(base_conf, hits) == harness.reliability_bins(base_conf, hits)
print("cross-check ok")
