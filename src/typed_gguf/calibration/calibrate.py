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
from dataclasses import dataclass
from typing import Any

from typed_gguf.calibration import stats
from typed_gguf.engine import readout

#: a type needs this many rows on EACH side of the split before a temperature may be fitted
MIN_FIT_ROWS = 8
MIN_HOLDOUT_ROWS = 3
#: the ECE improvement an accepted fit must show on the held-out split (absolute)
ACCEPT_EPSILON = 1e-6
#: how much better a non-default confidence mode must be on the held-out split to be selected
#: at all (A-E2p5-3: "the default stays normalized_peak unless a mode wins by a documented
#: margin" — a switch is a user-visible change and must be worth more than noise)
MODE_MARGIN = 0.005
#: "fit every documented mode, then pick one with the rule above"
MODE_AUTO = "auto"
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
    #: the engine's own `reliability` (ok | low_mass | low_confidence) — the escalation policy
    #: reads it, so the row keeps it instead of dropping it at the fit boundary
    reliability: str = ""

    @property
    def correct_index(self) -> int | None:
        """The index of the labelled answer in `labels` (None when the report lost it)."""
        if self.expected and self.expected in self.labels:
            return self.labels.index(self.expected)
        return None

    def as_answer(self) -> dict[str, Any]:
        """The row as an `answers`-shaped object (what `routing.escalation_candidates` reads).

        The type-specific decision field travels with it (`noul`/`choice`), because the shipped
        decision rule (`routing.decision_of`) reads exactly those fields: a row stripped down to
        its probabilities would make every `noul` answer look like a "no".
        """
        probabilities = dict(zip(self.labels, self.probabilities, strict=True))
        answer: dict[str, Any] = {"type": self.type, "probabilities": probabilities,
                                  "confidence": self.confidence, "reliability": self.reliability}
        if self.type == "noul":
            answer["noul"] = float(probabilities.get("yes", max(self.probabilities, default=0.0)))
        elif self.type == "choice" and self.labels:
            answer["choice"] = self.labels[readout.argmax_first(self.probabilities)]
        return answer

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
                   model=model, reliability=str(item.get("reliability") or ""))

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "correct": self.correct,
                "confidence": self.confidence, "coverage": self.coverage,
                "reliability": self.reliability, "expected": self.expected, "got": self.got,
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


# ------------------------------------------------------------------- the fit
def _index_of(row: Row) -> int | None:
    """The candidate index the NLL is read at (None when the report lost the label)."""
    if row.correct_index is not None:
        return row.correct_index
    return readout.argmax_first(row.probabilities) if row.probabilities else None


def _evaluate(rows: Sequence[Row], temperature: float, *, mode: str,
              n_bins: int) -> dict[str, Any]:
    usable = [(row, _index_of(row)) for row in rows]
    usable = [(row, index) for row, index in usable if index is not None]
    return stats.evaluate([row.probabilities for row, _ in usable],
                          [int(index) for _, index in usable], temperature,
                          mode=mode, n_bins=n_bins)


@dataclass(frozen=True, slots=True)
class TypeFit:
    """What one question type measured: the parameter, the verdict and the receipts."""

    qtype: str
    temperature: float
    accepted: bool
    reason: str
    mode: str
    n_fit: int
    n_holdout: int
    fit_before: dict[str, Any]
    fit_after: dict[str, Any]
    holdout_before: dict[str, Any]
    holdout_after: dict[str, Any]
    modes: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"qtype": self.qtype, "temperature": self.temperature, "accepted": self.accepted,
                "reason": self.reason, "mode": self.mode,
                "n": {"fit": self.n_fit, "holdout": self.n_holdout},
                "ece": {"fit": {"before": self.fit_before["ece"], "after": self.fit_after["ece"]},
                        "holdout": {"before": self.holdout_before["ece"],
                                    "after": self.holdout_after["ece"]}},
                "nll": {"fit": {"before": self.fit_before["nll"], "after": self.fit_after["nll"]},
                        "holdout": {"before": self.holdout_before["nll"],
                                    "after": self.holdout_after["nll"]}},
                "agreement": {"holdout": {"before": self.holdout_before["agreement"],
                                          "after": self.holdout_after["agreement"]}},
                "agreement_at_0.5": {
                    "holdout": {"before": self.holdout_before["agreement_at_0.5"]["agreement"],
                                "after": self.holdout_after["agreement_at_0.5"]["agreement"]}},
                "modes": self.modes,
                # the four full evaluations verbatim: the report keeps every number it used, and
                # a stored table round-trips without re-running anything
                "evaluate": {"fit_before": self.fit_before, "fit_after": self.fit_after,
                             "holdout_before": self.holdout_before,
                             "holdout_after": self.holdout_after}}

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> TypeFit:
        evaluated = dict(payload.get("evaluate") or {})
        fallback = dict(evaluated.get("holdout_after") or evaluated.get("fit_after") or {})
        return cls(qtype=str(payload["qtype"]), temperature=float(payload["temperature"]),
                   accepted=bool(payload["accepted"]), reason=str(payload["reason"]),
                   mode=str(payload["mode"]), n_fit=int(payload["n"]["fit"]),
                   n_holdout=int(payload["n"]["holdout"]),
                   fit_before=dict(evaluated.get("fit_before", fallback)),
                   fit_after=dict(evaluated.get("fit_after", fallback)),
                   holdout_before=dict(evaluated.get("holdout_before", fallback)),
                   holdout_after=dict(evaluated.get("holdout_after", fallback)),
                   modes=dict(payload["modes"]))

    @property
    def holdout_delta(self) -> float:
        return float(self.holdout_before["ece"]) - float(self.holdout_after["ece"])


def _select_mode(per_mode: Mapping[str, Mapping[str, Any]], *, default: str,
                 margin: float = MODE_MARGIN) -> tuple[str | None, str]:
    """Which confidence mode the parameter was accepted with — `None` when none was (A-E2p5-3).

    `per_mode[mode]` carries `holdout_before`, `holdout_after` and `temperature` for that mode's
    *own* fitted parameter. The rule, in order:

    1. a mode is **eligible** when its own held-out ECE improves (by more than `ACCEPT_EPSILON`);
    2. the default mode wins whenever it is eligible (it is the documented default);
    3. otherwise the best eligible mode is selected only when it beats the identity by at least
       `margin` — a switch is a user-visible change and does not happen on noise;
    4. when the fit chose the identity everywhere, the answer is "no calibration applied" with
       that stated as the reason.
    """
    improvements = {
        name: float(report["holdout_before"]["ece"]) - float(report["holdout_after"]["ece"])
        for name, report in per_mode.items()}
    identity_everywhere = all(
        float(report["temperature"]) == 1.0 for report in per_mode.values())
    eligible = {name: value for name, value in improvements.items() if value > ACCEPT_EPSILON}
    if default in eligible:
        return default, ""
    best = max(eligible.items(), key=lambda item: (item[1], item[0] == default,
                                                   item[0])) if eligible else None
    if best is not None and best[1] >= margin:
        return best[0], ""
    if identity_everywhere:
        return None, "no calibration applied: the fit chose the identity (temperature 1.0)"
    if eligible:
        winner = best[0] if best else default
        return None, (f"no calibration applied: {winner} improved the held-out ECE by only "
                      f"{improvements[winner]:.4f} (< the {margin:.4f} margin a mode switch needs)")
    return None, "no calibration applied: the held-out split did not improve"


def _fit_type(qtype: str, fit_rows: Sequence[Row], holdout_rows: Sequence[Row], *,
              mode: str, n_bins: int, grid: Sequence[float],
              min_fit: int, min_holdout: int) -> TypeFit:
    """One type: fit every requested mode, accept on the held-out split (A-E2p5-2/3)."""
    identity_fit = _evaluate(fit_rows, 1.0, mode=_default_mode(mode), n_bins=n_bins)
    identity_holdout = _evaluate(holdout_rows, 1.0, mode=_default_mode(mode), n_bins=n_bins)
    too_thin = ""
    if len(fit_rows) < min_fit:
        too_thin = f"too few fit rows ({len(fit_rows)} < {min_fit})"
    elif len(holdout_rows) < min_holdout:
        too_thin = f"too few held-out rows ({len(holdout_rows)} < {min_holdout})"
    if too_thin:
        return TypeFit(qtype=qtype, temperature=1.0, accepted=False,
                       reason=f"no calibration applied: {too_thin}",
                       mode=_default_mode(mode), n_fit=len(fit_rows), n_holdout=len(holdout_rows),
                       fit_before=identity_fit, fit_after=identity_fit,
                       holdout_before=identity_holdout, holdout_after=identity_holdout,
                       modes=_mode_report(fit_rows, holdout_rows, 1.0, n_bins=n_bins))

    usable = [row for row in fit_rows if _index_of(row) is not None]
    per_mode: dict[str, dict[str, Any]] = {}
    for name in _modes_to_fit(mode):
        fitted = stats.fit_temperature([row.probabilities for row in usable],
                                       [int(_index_of(row)) for row in usable],
                                       grid=grid, mode=name, n_bins=n_bins)
        temperature = float(fitted["temperature"])
        per_mode[name] = {
            "mode": name, "temperature": temperature, "fitted": fitted,
            "fit_before": _evaluate(fit_rows, 1.0, mode=name, n_bins=n_bins),
            "fit_after": _evaluate(fit_rows, temperature, mode=name, n_bins=n_bins),
            "holdout_before": _evaluate(holdout_rows, 1.0, mode=name, n_bins=n_bins),
            "holdout_after": _evaluate(holdout_rows, temperature, mode=name, n_bins=n_bins),
        }
    selected, reason = _select_mode(per_mode, default=_default_mode(mode))
    if selected is None:
        return TypeFit(qtype=qtype, temperature=1.0, accepted=False, reason=reason,
                       mode=_default_mode(mode), n_fit=len(fit_rows), n_holdout=len(holdout_rows),
                       fit_before=identity_fit, fit_after=identity_fit,
                       holdout_before=identity_holdout, holdout_after=identity_holdout,
                       modes=_mode_report(fit_rows, holdout_rows, 1.0, n_bins=n_bins,
                                          per_mode=per_mode, selected=None))
    chosen = per_mode[selected]
    temperature = float(chosen["temperature"])
    # `_select_mode` only returns a mode whose own held-out ECE improved by more than
    # ACCEPT_EPSILON, which is exactly what `_accept` re-checks — the call is here for the
    # *reason* string, and it must agree (a test pins that one implies the other).
    accepted, reason = _accept(temperature, chosen["holdout_before"], chosen["holdout_after"])
    return TypeFit(qtype=qtype, temperature=temperature, accepted=accepted, reason=reason,
                   mode=selected, n_fit=len(fit_rows), n_holdout=len(holdout_rows),
                   fit_before=chosen["fit_before"], fit_after=chosen["fit_after"],
                   holdout_before=chosen["holdout_before"], holdout_after=chosen["holdout_after"],
                   modes=_mode_report(fit_rows, holdout_rows, temperature, n_bins=n_bins,
                                      per_mode=per_mode, selected=selected))


def _default_mode(mode: str) -> str:
    return readout.DEFAULT_CONFIDENCE_MODE if mode == MODE_AUTO else mode


def _modes_to_fit(mode: str) -> tuple[str, ...]:
    if mode == MODE_AUTO:
        return tuple(sorted(readout.CONFIDENCE_MODES))
    if mode not in readout.CONFIDENCE_MODES:
        raise KeyError(f"unknown confidence mode {mode!r}")
    return (mode,)


def _accept(temperature: float, before: Mapping[str, Any], after: Mapping[str, Any]
            ) -> tuple[bool, str]:
    """The gate: the held-out ECE improves, or the threshold agreement does on an ECE tie."""
    ece_before, ece_after = float(before["ece"]), float(after["ece"])
    if temperature == 1.0:
        return False, "no calibration applied: the fit chose the identity (temperature 1.0)"
    if ece_after < ece_before - ACCEPT_EPSILON:
        return True, (f"applied: held-out ECE {ece_before:.4f} -> {ece_after:.4f} "
                      f"at temperature {temperature:.4f}")
    if abs(ece_after - ece_before) <= ACCEPT_EPSILON:
        agreement_before = float(before["agreement_at_0.5"]["agreement"])
        agreement_after = float(after["agreement_at_0.5"]["agreement"])
        if agreement_after > agreement_before + ACCEPT_EPSILON:
            return True, (f"applied: held-out ECE unchanged ({ece_before:.4f}) and the agreement "
                          f"at {stats.FIXED_THRESHOLD} improved {agreement_before:.3f} -> "
                          f"{agreement_after:.3f}")
    return False, "no calibration applied: the held-out split did not improve"


def _mode_report(fit_rows: Sequence[Row], holdout_rows: Sequence[Row], temperature: float, *,
                 n_bins: int, per_mode: Mapping[str, Mapping[str, Any]] | None = None,
                 selected: str | None = None) -> dict[str, Any]:
    """A-E2p5-3: every confidence mode measured in one report, with its own fitted parameter.

    Each mode is evaluated at its **own** fitted temperature (that is the number the selection
    rule compares) and the report also says whether that mode was eligible and which one was
    selected. When no per-mode fits exist (too few rows), every mode is measured at the given
    temperature — the identity there — so the report keeps its three columns.
    """
    names = sorted(per_mode) if per_mode else sorted(readout.CONFIDENCE_MODES)
    report: dict[str, Any] = {}
    for name in names:
        measured = per_mode.get(name) if per_mode else None
        mode_temperature = float(measured["temperature"]) if measured else temperature
        fit_before = _evaluate(fit_rows, 1.0, mode=name, n_bins=n_bins)
        fit_after = _evaluate(fit_rows, mode_temperature, mode=name, n_bins=n_bins)
        holdout_before = _evaluate(holdout_rows, 1.0, mode=name, n_bins=n_bins)
        holdout_after = _evaluate(holdout_rows, mode_temperature, mode=name, n_bins=n_bins)
        improvement = float(holdout_before["ece"]) - float(holdout_after["ece"])
        report[name] = {
            "mode": name,
            "temperature": mode_temperature,
            "selected": name == selected,
            "eligible": improvement > ACCEPT_EPSILON,
            "holdout_ece_improvement": improvement,
            "ece_fit_before": float(fit_before["ece"]),
            "ece_fit_after": float(fit_after["ece"]),
            "ece_holdout_before": float(holdout_before["ece"]),
            "ece_holdout_after": float(holdout_after["ece"]),
            "agreement_holdout_before": float(holdout_before["agreement"]),
            "agreement_holdout_after": float(holdout_after["agreement"]),
            "agreement_at_0.5_holdout_before": float(
                holdout_before["agreement_at_0.5"]["agreement"]),
            "agreement_at_0.5_holdout_after": float(
                holdout_after["agreement_at_0.5"]["agreement"]),
            "mean_confidence_before": holdout_before["mean_confidence"],
            "mean_confidence_after": holdout_after["mean_confidence"],
        }
    return report


# ----------------------------------------------------------------- the table
def _devset_digest(rows: Sequence[Row]) -> str:
    """A digest of the labelled data the fit actually reads: same fit input -> same digest.

    Only the fields `_index_of` and `fit_temperature` consume take part. A re-measurement that
    moves a field the fit never looks at — a coverage float, a reliability label, this run's
    confidence — must not move the parameters hash (A-E2p5-6: *same set + same model* ⇒ identical
    hash), while any change to the probabilities or the labels does.
    """
    fields = ("id", "type", "expected", "correct", "probabilities")
    payload = sorted(({key: row.to_json()[key] for key in fields} for row in rows),
                     key=lambda item: str(item["id"]))
    return stats.params_hash({"rows": payload})


@dataclass(frozen=True, slots=True)
class Table:
    """The stored parameters for one model + everything the report needs to justify them."""

    model: str
    types: dict[str, TypeFit]
    mode: str = readout.DEFAULT_CONFIDENCE_MODE
    model_path: str = ""
    model_sha256: str = ""
    alias: str | None = None
    devset_path: str = ""
    devset_items: int = 0
    devset_digest: str = ""
    holdout_fraction: float = HOLDOUT_FRACTION
    n_bins: int = stats.N_BINS
    created_at: str = ""
    source: str = ""                      # where it was loaded from ("" = never stored)
    #: the confidence modes this fit was asked for (MODE_AUTO = all of them)
    modes: tuple[str, ...] = ()

    @property
    def accepted_types(self) -> tuple[str, ...]:
        return tuple(sorted(name for name, entry in self.types.items() if entry.accepted))

    @property
    def accepted(self) -> bool:
        return bool(self.accepted_types)

    @property
    def params(self) -> dict[str, Any]:
        """The parameters themselves — exactly what `params_hash` covers (A-E2p5-6)."""
        return {
            "model": self.model,
            "mode": self.mode,
            "modes": list(self.modes),
            "holdout_fraction": self.holdout_fraction,
            "n_bins": self.n_bins,
            "devset_digest": self.devset_digest,
            "types": {name: {"temperature": entry.temperature, "accepted": entry.accepted}
                      for name, entry in sorted(self.types.items())},
        }

    @property
    def params_hash(self) -> str:
        return stats.params_hash(self.params)

    def mode_for(self, qtype: str) -> str | None:
        """The confidence statistic this type's parameter was accepted with, or None.

        A-E2p5-3: the default stays `normalized_peak` unless another mode won by the documented
        margin — when one did, that statistic *is* the readout's confidence for the type (a
        parameter that improves a statistic nobody reports would be pointless).
        """
        entry = self.types.get(qtype)
        if entry is None or not entry.accepted:
            return None
        return entry.mode

    def temperature_for(self, qtype: str) -> float | None:
        entry = self.types.get(qtype)
        if entry is None or not entry.accepted:
            return None
        return entry.temperature

    def apply(self, probabilities: Sequence[float], qtype: str, *,
              mode: str | None = None) -> dict[str, Any]:
        """The readout hook: scale one question's probabilities (never its ranking).

        `mode` is the statistic the *confidence* is computed with: the caller's explicit choice
        wins, otherwise the statistic this type was accepted with, otherwise the documented
        default (`normalized_peak`).
        """
        confidence_mode = mode or self.mode_for(qtype) or readout.DEFAULT_CONFIDENCE_MODE
        values = [float(value) for value in probabilities]
        temperature = self.temperature_for(qtype)
        if temperature is None:
            return {"probabilities": values, "temperature": 1.0, "calibrated": False,
                    "confidence": readout.confidence(values, confidence_mode),
                    "mode": confidence_mode, "source": self.source,
                    "params_hash": self.params_hash if self.accepted else ""}
        scaled = stats.power_scale(values, temperature)
        return {"probabilities": scaled, "temperature": temperature, "calibrated": True,
                "confidence": readout.confidence(scaled, confidence_mode),
                "mode": confidence_mode, "source": self.source, "params_hash": self.params_hash}

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": stats.SCHEMA,
            "created_at": self.created_at,
            "model": self.model,
            "model_path": self.model_path,
            "model_sha256": self.model_sha256,
            "alias": self.alias,
            "mode": self.mode,
            "modes": list(self.modes),
            "accepted": self.accepted,
            "accepted_types": list(self.accepted_types),
            "holdout_fraction": self.holdout_fraction,
            "n_bins": self.n_bins,
            "devset": {"path": self.devset_path, "items": self.devset_items,
                       "digest": self.devset_digest},
            "params": self.params,
            "params_hash": self.params_hash,
            "types": {name: entry.to_json() for name, entry in sorted(self.types.items())},
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any], *, source: str = "") -> Table:
        return cls(
            model=str(payload["model"]),
            types={name: TypeFit.from_json(entry)
                   for name, entry in dict(payload.get("types") or {}).items()},
            mode=str(payload.get("mode", readout.DEFAULT_CONFIDENCE_MODE)),
            model_path=str(payload.get("model_path", "")),
            model_sha256=str(payload.get("model_sha256", "")),
            alias=payload.get("alias"),
            devset_path=str((payload.get("devset") or {}).get("path", "")),
            devset_items=int((payload.get("devset") or {}).get("items", 0)),
            devset_digest=str((payload.get("devset") or {}).get("digest", "")),
            holdout_fraction=float(payload.get("holdout_fraction", HOLDOUT_FRACTION)),
            n_bins=int(payload.get("n_bins", stats.N_BINS)),
            modes=tuple(payload.get("modes") or ()),
            created_at=str(payload.get("created_at", "")),
            source=source)

    def response_fields(self, applied: Mapping[str, str] | Sequence[str] = ()) -> dict[str, Any]:
        """The response's `calibration` block (A-E2p5-1): what was applied, and its source.

        `applied` maps the question types that were really rescaled to the confidence statistic
        that was *actually used* in this response (which is the caller's explicit choice when the
        request named one, not necessarily the promoted one) — so a reader can tell what the
        numbers in `answers` mean without guessing. `source` is the store the table was read from.
        """
        modes = dict(applied) if isinstance(applied, Mapping) else {
            name: self.types[name].mode for name in applied if name in self.types}
        names = sorted(modes)
        return {
            "source": self.source if self.accepted else "",
            "applied": bool(names),
            "model": self.model if self.accepted else None,
            "params_hash": self.params_hash if self.accepted else "",
            "temperatures": {name: self.types[name].temperature for name in names
                             if name in self.types},
            "confidence_modes": {name: modes[name] for name in names},
            "accepted_types": list(self.accepted_types),
        }

    def render(self) -> str:
        """The `--dry-run` table (A-E2p5-1): one row per type, one verdict per row."""
        lines = [
            f"calibration {stats.SCHEMA}  model {self.model}",
            f"  devset {self.devset_items} item(s) {self.devset_path or '<none>'} "
            f"digest {self.devset_digest[:26]}",
            f"  mode {self.mode}  holdout {self.holdout_fraction:.3f}  params {self.params_hash}",
            f"  {'type':<8} {'temperature':>11}  {'holdout ECE':>19}  {'fit ECE':>19}  verdict",
        ]
        for name, entry in sorted(self.types.items()):
            holdout = f"{entry.holdout_before['ece']:.4f} -> {entry.holdout_after['ece']:.4f}"
            fit = f"{entry.fit_before['ece']:.4f} -> {entry.fit_after['ece']:.4f}"
            verdict = "applied" if entry.accepted else entry.reason
            lines.append(f"  {name:<8} {entry.temperature:>11.4f}  {holdout:>19}  {fit:>19}  "
                         f"{verdict}")
        if not self.accepted:
            lines.append("  no calibration applied (no question type improved on the held-out "
                         "split)")
        return "\n".join(lines)


def fit_table(rows: Sequence[Row], *, model_key: str, model_path: str = "",
              model_sha256: str = "", alias: str | None = None, devset_path: str = "",
              mode: str = MODE_AUTO,
              n_bins: int = stats.N_BINS, grid: Sequence[float] = stats.DEFAULT_GRID,
              holdout_fraction: float = HOLDOUT_FRACTION, min_fit: int = MIN_FIT_ROWS,
              min_holdout: int = MIN_HOLDOUT_ROWS) -> Table:
    """Fit every question type present in `rows` and build the table (A-E2p5-1/2/3/6).

    `mode` is a confidence statistic (`normalized_peak` / `entropy` / `margin`) or `"auto"`, the
    default: fit each of the three, then keep the documented default unless another statistic wins
    by `MODE_MARGIN` on the held-out split (A-E2p5-3).
    """
    if not rows:
        raise ValueError("fit_table needs at least one row")
    modes = _modes_to_fit(mode)
    split = split_rows(rows, holdout_fraction=holdout_fraction)
    types: dict[str, TypeFit] = {}
    for qtype in sorted({row.type for row in rows}):
        types[qtype] = _fit_type(qtype,
                                 [row for row in split.fit if row.type == qtype],
                                 [row for row in split.holdout if row.type == qtype],
                                 mode=mode, n_bins=n_bins, grid=grid,
                                 min_fit=min_fit, min_holdout=min_holdout)
    return Table(model=model_key, types=types, mode=mode, model_path=model_path,
                 model_sha256=model_sha256, alias=alias, devset_path=devset_path,
                 devset_items=len(rows), devset_digest=_devset_digest(rows),
                 holdout_fraction=holdout_fraction, n_bins=n_bins,
                 created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 modes=tuple(modes))


# ------------------------------------------------------------------ the store
STORE_SCHEMA = "typed_gguf.calibration-store/v1"


def model_key_for(path: str | pathlib.Path, *, sha256: str | None = None) -> str:
    """The store key of one model: its SHA-256 when known, else the file name and size.

    Name + size rather than the full path: the same GGUF moved to another directory (or pulled
    under another alias) is the same model, so its parameters stay valid — A-E2p5-6's "same set +
    same model ⇒ identical params hash" must not depend on where the file happens to sit.
    """
    if sha256:
        return f"sha256:{sha256}"
    target = pathlib.Path(path)
    size = target.stat().st_size if target.exists() else 0
    return f"file:{target.name}:{size}"


def _read_store(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not readable JSON ({exc}); delete it to re-calibrate "
                         f"from scratch") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("models"), Mapping):
        raise ValueError(f"{path} is not a calibration store (no `models` object)")
    return dict(payload["models"])


def _write_store(path: pathlib.Path, models: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": STORE_SCHEMA, "models": dict(models)}
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_table(path: str | pathlib.Path, model_key: str) -> Table | None:
    """The stored table for one model, or None (a missing store is not an error)."""
    target = pathlib.Path(path)
    models = _read_store(target)
    entry = models.get(model_key)
    if not isinstance(entry, Mapping):
        return None
    return Table.from_json(entry, source=str(target))


def save_table(path: str | pathlib.Path, table: Table) -> pathlib.Path | None:
    """Store an accepted table; a rejected one retires whatever was stored for that model.

    Returns the path written, or None when nothing was stored (A-E2p5-2: "otherwise the tool
    reports *no calibration applied* and stores nothing"). A previously accepted table for the
    same model is removed, because the newest measurement is the one that must stand.
    """
    target = pathlib.Path(path)
    models = _read_store(target)
    if table.accepted:
        models[table.model] = table.to_json()
        _write_store(target, models)
        return target
    if table.model in models:
        del models[table.model]
        if models:
            _write_store(target, models)
        else:
            target.unlink(missing_ok=True)
    return None


def render_table(table: Table) -> str:
    """`--dry-run` output: the same table `Table.render()` prints, without any side effect."""
    return table.render()
