"""Scratch: per-mode holdout ECE at each mode's own fitted temperature (from the live rows)."""
from __future__ import annotations

import sys

sys.path.insert(0, "/work/e2p5repo2/src")

from ggufone.calibration import calibrate, stats  # noqa: E402
from ggufone.engine import readout  # noqa: E402

ROWS = "/work/e2p5-evidence/e2p5_rows_qwen08.json"
rows = calibrate.load_rows(ROWS)
split = calibrate.split_rows(rows)
print(f"{len(rows)} rows -> fit {len(split.fit)} / holdout {len(split.holdout)}")
for qtype in sorted({row.type for row in rows}):
    fit_rows = [row for row in split.fit if row.type == qtype]
    hold_rows = [row for row in split.holdout if row.type == qtype]
    print(f"\n== {qtype}: fit {len(fit_rows)} holdout {len(hold_rows)}")
    for mode in sorted(readout.CONFIDENCE_MODES):
        fitted = stats.fit_temperature([r.probabilities for r in fit_rows],
                                       [int(r.correct_index) for r in fit_rows], mode=mode)
        t = fitted["temperature"]
        hold_before = calibrate._evaluate(hold_rows, 1.0, mode=mode, n_bins=10)
        hold_after = calibrate._evaluate(hold_rows, t, mode=mode, n_bins=10)
        print(f"   {mode:<16} T={t:8.4f}  fit ECE {fitted['ece_before']:.4f}->{fitted['ece']:.4f}"
              f"  holdout ECE {hold_before['ece']:.4f}->{hold_after['ece']:.4f}"
              f"  delta {hold_before['ece'] - hold_after['ece']:+.4f}")
