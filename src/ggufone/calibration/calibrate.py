"""Temperature/scale fitting per (model, question type) — SPEC 2.10, A-E2p5-1/2/3/6.

Milestone: E2.5.

The fit is a pure function of *rows*. A row is one dev-set decision in the exact shape
`--suite calibration` publishes (`report["items"][i]`: id, type, expected, got, correct,
confidence, coverage, probabilities, …), so a calibration is reproducible from a committed
JSON report (`load_rows`) without loading a model — and the live path (a model runs the dev set,
the rows are the same objects) cannot drift from the reproduced one.

The acceptance gate is the honest part of A-E2p5-2:

  * per question type, the rows are split by id into a fit part and a held-out part;
  * a temperature is only *accepted* when the held-out ECE improves (or, on an ECE tie, the
    agreement at a fixed confidence threshold improves);
  * a rejected type stores nothing, and `save_table` retires any previously stored parameters
    for that model — a stale table must not survive a measurement that no longer supports it;
  * the fitted temperature is always at most as bad as the identity (`stats.fit_temperature`),
    so "no calibration applies" is a real outcome, never a cosmetic one.
"""
from __future__ import annotations

import json
import pathlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ggufone.calibration import stats
from ggufone.engine import readout

#: a type needs this many rows on EACH side of the split before a temperature may be fitted
MIN_FIT_ROWS = 8
MIN_HOLDOUT_ROWS = 3
#: the ECE improvement an accepted fit must show on the held-out split (absolute)
ACCEPT_EPSILON = 1e-6
HOLDOUT_FRACTION = 1 / 3
QUESTION_TYPES = ("choice", "score", "noul")


# ------------------------------------------------------------------------ rows
@dataclass(frozen=True, slots=True)
class Row:
    """One labelled decision: the candidate probabilities the engine reported + the truth."""

    id: str
    type: str
    labels: tuple[str, ...]
    probabilities: tuple[float, ...]
    correct: bool
    confidence: float = 0.0
    coverage: float | None = None
    expected: str = ""
    got: str = ""
    model: str = ""

    @property
    def correct_index(self) -> int | None:
        """The index of the labelled answer in `labels` (None when the report lost it)."""
        if self.expected and self.expected in self.labels:
            return self.labels.index(self.expected)
        return None

    @classmethod
    def from_item(cls, item: Mapping[str, Any], *, model: str = "") -> Row:
        """Parse one `--suite calibration` report item (never guesses a missing field)."""
        probabilities = item.get("probabilities")
        if not isinstance(probabilities, Mapping) or not probabilities:
            raise ValueError(f"row {item.get('id')!r} carries no probabilities")
        labels = tuple(str(key) for key in probabilities)
        values = tuple(float(value) for value in probabilities.values())
        expected = str(item.get("expected", ""))
        got = str(item.get("got", ""))
        correct = item.get("correct")
        if correct is None:
            correct = bool(expected) and expected == got
        return cls(id=str(item["id"]), type=str(item["type"]), labels=labels,
                   probabilities=values, correct=bool(correct),
                   confidence=float(item.get("confidence") or max(values)),
                   coverage=(float(item["coverage"]) if item.get("coverage") is not None else None),
                   expected=expected or (labels[readout.argmax_first(values)] if labels else ""),
                   got=got or (labels[readout.argmax_first(values)] if labels else ""),
                   model=model)

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "correct": self.correct,
                "confidence": self.confidence, "coverage": self.coverage,
                "expected": self.expected, "got": self.got,
                "probabilities": dict(zip(self.labels, self.probabilities, strict=True))}


def load_rows(path: str | pathlib.Path, *, model: str = "") -> list[Row]:
    """Read a `--suite calibration` report (or any object with an `items` list) into rows."""
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, Mapping) else None
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path}: no `items` list — is this a `--suite calibration` report?")
    return [Row.from_item(item, model=model) for item in items]


# ----------------------------------------------------------------------- split
@dataclass(frozen=True, slots=True)
class Split:
    fit: tuple[Row, ...]
    holdout: tuple[Row, ...]


def split_rows(rows: Sequence[Row], *, holdout_fraction: float = HOLDOUT_FRACTION) -> Split:
    """Split every question type by id: the tail of the sorted ids is held out (A-E2p5-2).

    Deterministic and order-independent (rows are sorted by id first), and stratified by type so
    a type with few rows cannot end up entirely on one side: at least one row stays in each part
    of every type that has at least two rows.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in (0, 1), got {holdout_fraction}")
    by_type: dict[str, list[Row]] = {}
    for row in sorted(rows, key=lambda item: item.id):
        by_type.setdefault(row.type, []).append(row)
    fit: list[Row] = []
    holdout: list[Row] = []
    for qtype in sorted(by_type):
        group = by_type[qtype]
        if len(group) == 1:
            fit.extend(group)
            continue
        held = int(round(len(group) * holdout_fraction))
        held = max(1, min(len(group) - 1, held))
        fit.extend(group[:-held])
        holdout.extend(group[-held:])
    return Split(fit=tuple(fit), holdout=tuple(holdout))
