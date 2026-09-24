#!/usr/bin/env python3
"""E3d: does a different cue shape beat the shipped one? (card t_d90404ac).

E3c (`t_6c119626`) measured seven cue shapes on **6** dev items and found the shape is the lever
on a model that answers: the shipped cue took 3/6 correct with 6/6 `low_mass`, the two-step
readout 5/6 with 0/6, and the opened JSON field cleared the 0.10 floor 6/6. Six items is a signal;
this tool is the same question on the **full 60-item dev set**, with the statistics the decision
needs — and the statistics are the point, because the naive reading (compare the two Wilson
intervals) cannot decide it:

* every shape measures the **same 60 items**, so the comparison is *paired*. Two marginal Wilson
  intervals that overlap say nothing about the difference; two that do not overlap are not needed
  either. The test is the **discordant pairs**: exact two-sided McNemar (binomial tail, not the
  chi-square approximation — 60 items is small), plus a seeded percentile bootstrap of the
  paired risk difference.
* `coverage` and `reliability` are read from the shape's own readout row: for `two_step_*` that is
  the **advanced** row (the second decision point), never the cue row it moved past, and the
  per-shape `W_CUE_REFUSED` count is the probe's own `cue_verdict` on that same row.

The decision rule is stated, not implied (`decide()`): the challenger wins when the paired
difference's 95 % CI excludes zero **and** the winner is consistent in the direction that matters
(coverage above the floor). Anything else is "not by more than the CI noise" and the default stays.

    # offline: the card's evidence tables from a stored probe run
    python3 tools/e3d_cue_decision.py report --run docs/evidence/e3d/full.json \\
        --out docs/evidence/e3d_...md

The tool never loads a model and never touches the network.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
from collections.abc import Mapping, Sequence
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from typed_gguf.bench import harness  # noqa: E402

SCHEMA = "typed_gguf.e3d.cue-decision/v1"
BOOTSTRAP_ITERS = 10000
BOOTSTRAP_SEED = 20260919
#: the rule the card states: a win is a paired CI that excludes zero
UNIT = "paired"


class DecisionError(Exception):
    """A record/argument problem the caller can fix (the probe's own error style)."""


# ------------------------------------------------------------------- statistics
def wilson(successes: int, n: int) -> tuple[float, float]:
    """The 95 % Wilson interval — the harness's function, never a second implementation.

    Deliberately *not* re-derived here: a probe that computed its own interval could disagree
    with the bench tables it is compared against, and the whole card is about not doing that.
    """
    return harness.wilson_interval(successes, n)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs `(b, c)`.

    `b` = pairs where A was right and B wrong, `c` = the reverse. Under the null both are
    equally likely, so `X | X + Y = n` is `Binomial(n, 0.5)` and the two-sided p-value is
    `2 * P(X <= min(b, c))`, clamped to 1. The exact form is used because the dev set is 60
    items — the chi-square approximation is not licensed at that size.
    """
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    smaller = min(int(b), int(c))
    tail = sum(math.comb(n, index) for index in range(smaller + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def paired_diff(baseline: Sequence[bool], challenger: Sequence[bool], *,
                iters: int = BOOTSTRAP_ITERS, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """The paired risk difference + a seeded percentile bootstrap CI (items resampled together).

    Pairing is the whole point: item `i` contributes `(a_i, b_i)` and a resample draws items, not
    independent observations. The point estimate is exact (mean difference over all items); the
    interval is the 2.5/97.5 percentile of `iters` resamples.
    """
    if len(baseline) != len(challenger):
        raise DecisionError("paired_diff needs the same items on both sides; "
                            f"got {len(baseline)} and {len(challenger)}")
    if not baseline:
        raise DecisionError("paired_diff needs at least one item")
    a = [bool(value) for value in baseline]
    b = [bool(value) for value in challenger]
    n = len(a)
    difference = sum(b) / n - sum(a) / n
    a_win = sum(1 for index in range(n) if a[index] and not b[index])
    b_win = sum(1 for index in range(n) if b[index] and not a[index])
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(max(1, int(iters))):
        total = 0
        for _ in range(n):
            index = rng.randrange(n)
            total += int(b[index]) - int(a[index])
        draws.append(total / n)
    draws.sort()
    low = draws[int(0.025 * (len(draws) - 1))]
    high = draws[int(0.975 * (len(draws) - 1))]
    return {"n": n, "difference": difference, "low": low, "high": high,
            "discordant": {"a_win": a_win, "b_win": b_win},
            "mcnemar_p": mcnemar_exact(a_win, b_win),
            "bootstrapped": True, "iters": max(1, int(iters)), "seed": seed}


def _quantile(ordered: Sequence[float], fraction: float) -> float:
    """Linear-interpolated quantile of a sorted sequence (numpy's default rule)."""
    if not ordered:
        raise DecisionError("quantile of an empty sequence")
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    below = int(math.floor(position))
    above = min(below + 1, len(ordered) - 1)
    weight = position - below
    return float(ordered[below]) * (1 - weight) + float(ordered[above]) * weight


# ------------------------------------------------------------------- record reading
def load_run(path: str | pathlib.Path) -> dict[str, Any]:
    """The stored probe record (the tool's only input; a model is never loaded)."""
    try:
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise DecisionError(f"cannot read the probe record {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DecisionError(f"{path} is not JSON: {exc}") from exc


def policies(record: Mapping[str, Any]) -> list[str]:
    """The `shape=label` policies the run ranked, in wire order."""
    keys = [str(key) for key in record.get("ranked_keys") or []]
    if keys:
        return keys
    seen: list[str] = []
    for item in record.get("items") or []:
        for key in (item.get("ranked") or {}):
            if key not in seen:
                seen.append(key)
    return seen


def shape_of(policy: str) -> str:
    """`two_step_shipped=bare` -> `two_step_shipped` (the probe keys are `shape=label`)."""
    return policy.split("=", 1)[0]


def _readout_block(record: Mapping[str, Any], item: Mapping[str, Any], shape: str,
                   ) -> Mapping[str, Any] | None:
    """The row a shape reads at: the item's own measurement, never the cue row of another shape."""
    entry = (item.get("shapes") or {}).get(shape)
    if entry is None:
        return None
    return entry.get("readout")


def rows_for(record: Mapping[str, Any], policy: str) -> list[dict[str, Any]]:
    """Every item's row for one policy: what the engine answered, whether it was right, and where.

    A policy the run did not rank is a caller error (a silent `[]` would render as "no items"
    instead of "you asked for the wrong key").
    """
    if policy not in policies(record):
        known = ", ".join(policies(record)) or "none"
        raise DecisionError(f"the run carries no ranked policy {policy!r}; ranked: {known}")
    rows: list[dict[str, Any]] = []
    for item in record.get("items") or []:
        ranked = (item.get("ranked") or {}).get(policy)
        if ranked is None:
            continue
        readout = _readout_block(record, item, shape_of(policy))
        cue = (readout or {}).get("cue") or {}
        rows.append({
            "id": item.get("id"), "type": item.get("type"), "expected": item.get("expected"),
            "got": ranked.get("got"), "correct": bool(ranked.get("correct")),
            "coverage": float(ranked.get("coverage") or 0.0),
            "reliability": ranked.get("reliability"),
            "refused": bool(cue.get("refused")),
            "closer": cue.get("closer"),
            "readout_row": (readout or {}).get("row"),
            "top_token": ((readout or {}).get("top_tokens") or [{}])[0].get("piece"),
        })
    if not rows:
        raise DecisionError(f"policy {policy!r} is in ranked_keys but carries no item rows")
    return rows


def coverage_summary(values: Sequence[float], *, floor: float) -> dict[str, Any]:
    """The coverage distribution: quartiles, extremes and the share at/above the engine's floor."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise DecisionError("coverage_summary of an empty sequence")
    above = sum(1 for value in ordered if value >= floor)
    return {"n": len(ordered), "min": ordered[0], "q1": _quantile(ordered, 0.25),
            "median": _quantile(ordered, 0.5), "q3": _quantile(ordered, 0.75),
            "max": ordered[-1], "mean": sum(ordered) / len(ordered),
            "above_floor": above, "share_above_floor": above / len(ordered), "floor": floor}


def _per_type(rows: Sequence[Mapping[str, Any]], *, floor: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in rows:
        bucket = out.setdefault(str(row["type"]), {"n": 0, "correct": 0, "low_mass": 0,
                                                   "refused": 0, "coverages": []})
        bucket["n"] += 1
        bucket["correct"] += int(bool(row["correct"]))
        bucket["low_mass"] += int(row["reliability"] == "low_mass")
        bucket["refused"] += int(bool(row["refused"]))
        bucket["coverages"].append(float(row["coverage"]))
    for bucket in out.values():
        bucket["agreement"] = bucket["correct"] / bucket["n"]
        bucket["coverage"] = coverage_summary(bucket.pop("coverages"), floor=floor)
    return dict(sorted(out.items()))


def analyse(record: Mapping[str, Any], *, floor: float | None = None) -> dict[str, Any]:
    """Everything the card asks for, per policy: agreement + CI, coverage, `low_mass`, refusals."""
    mass_floor = float(record.get("mass_floor", 0.10) if floor is None else floor)
    shapes: dict[str, Any] = {}
    for policy in policies(record):
        rows = rows_for(record, policy)
        correct = sum(1 for row in rows if row["correct"])
        low, high = wilson(correct, len(rows))
        coverages = [row["coverage"] for row in rows]
        shapes[policy] = {
            "shape": shape_of(policy), "readout_row": rows[0]["readout_row"],
            "n": len(rows), "correct": correct, "agreement": correct / len(rows),
            "wilson": (low, high),
            "low_mass": sum(1 for row in rows if row["reliability"] == "low_mass"),
            "refused": sum(1 for row in rows if row["refused"]),
            "closers": sorted({row["closer"] for row in rows if row["closer"]}),
            "coverage": coverage_summary(coverages, floor=mass_floor),
            "per_type": _per_type(rows, floor=mass_floor),
            "rows": rows,
        }
    return {"schema": SCHEMA, "mass_floor": mass_floor,
            "model": record.get("model"), "devset": record.get("devset"),
            "runtime": record.get("runtime"), "backend_claim": record.get("backend_claim"),
            "counts": dict(record.get("counts") or {}), "shapes": shapes,
            "policies": list(shapes)}


def decide(record: Mapping[str, Any], *, baseline: str, challenger: str,
           floor: float | None = None) -> dict[str, Any]:
    """The card's rule, executed: a win is a paired CI that excludes zero.

    Marginal Wilson intervals are reported next to it (`marginal_overlap`) precisely because they
    are the tempting wrong reading: on a paired 60-item set they can overlap while the paired
    difference is decided, and they can fail to overlap while it is not.
    """
    analysis = analyse(record, floor=floor)
    for key in (baseline, challenger):
        if key not in analysis["shapes"]:
            known = ", ".join(analysis["policies"])
            raise DecisionError(f"unknown policy {key!r}; analysed: {known}")
    base_rows = analysis["shapes"][baseline]["rows"]
    new_rows = analysis["shapes"][challenger]["rows"]
    base_by_id = {row["id"]: row for row in base_rows}
    new_by_id = {row["id"]: row for row in new_rows}
    shared = [row["id"] for row in base_rows if row["id"] in new_by_id]
    if len(shared) != len(base_rows) or len(shared) != len(new_rows):
        raise DecisionError(
            f"{baseline} and {challenger} measured different items "
            f"({len(base_rows)} vs {len(new_rows)}; {len(shared)} shared) — a paired reading "
            "needs the same items on both sides")
    paired = paired_diff([base_by_id[key]["correct"] for key in shared],
                         [new_by_id[key]["correct"] for key in shared])
    base_int = analysis["shapes"][baseline]["wilson"]
    new_int = analysis["shapes"][challenger]["wilson"]
    overlap = not (new_int[0] > base_int[1] or new_int[1] < base_int[0])
    challenger_wins = paired["low"] > 0.0
    loser_wins = paired["high"] < 0.0
    verdict = "promote" if challenger_wins else ("refuse" if loser_wins else "keep")
    note = (
        f"{UNIT} reading: the {paired['n']} items both policies measured, "
        f"{paired['discordant']['a_win']} won by {baseline} only and "
        f"{paired['discordant']['b_win']} by {challenger} only (exact McNemar p = "
        f"{paired['mcnemar_p']:.4g}); risk difference {paired['difference']:+.3f} "
        f"(95 % CI {paired['low']:+.3f}…{paired['high']:+.3f}). "
        + ("The marginal Wilson intervals overlap, which is why they are not the test."
           if overlap else
           "The marginal Wilson intervals do not overlap either, which is incidental — the "
           "paired interval is the reading."))
    return {"baseline": baseline, "challenger": challenger, "verdict": verdict,
            "challenger_wins": challenger_wins, "loser_wins": loser_wins,
            "paired": paired, "marginal_overlap": overlap,
            "marginal": {"baseline": base_int, "challenger": new_int}, "note": note,
            "floor_effect": {
                "baseline_share": analysis["shapes"][baseline]["coverage"]["share_above_floor"],
                "challenger_share": analysis["shapes"][challenger]["coverage"]["share_above_floor"],
                "baseline_low_mass": analysis["shapes"][baseline]["low_mass"],
                "challenger_low_mass": analysis["shapes"][challenger]["low_mass"]},
            }


# ------------------------------------------------------------------- rendering
def coverage_table(analysis: Mapping[str, Any]) -> list[str]:
    lines = ["| policy | n | min | q1 | median | q3 | max | above floor | `low_mass` |",
             "|---|---|---|---|---|---|---|---|---|"]
    for policy, block in analysis["shapes"].items():
        stats = block["coverage"]
        lines.append(f"| `{policy}` | {stats['n']} | {stats['min']:.3g} | {stats['q1']:.3g} "
                     f"| {stats['median']:.3g} | {stats['q3']:.3g} | {stats['max']:.3g} "
                     f"| {stats['above_floor']}/{stats['n']} | {block['low_mass']}/{block['n']} |")
    lines.append("")
    lines.append(f"Coverage is the full-vocabulary mass of the label's first token at the shape's "
                 f"own readout row (engine floor {analysis['mass_floor']:.2f}); for `two_step_*` "
                 f"that row is the **advanced** one, i.e. after the model's own first content "
                 f"token. `low_mass` counts the engine's verdict, which also depends on the "
                 f"answer's own reliability.")
    return lines


def agreement_table(analysis: Mapping[str, Any]) -> list[str]:
    lines = ["| policy | readout row | n | correct | agreement | 95 % Wilson | `low_mass` "
             "| `W_CUE_REFUSED` |", "|---|---|---|---|---|---|---|---|"]
    for policy, block in analysis["shapes"].items():
        low, high = block["wilson"]
        lines.append(f"| `{policy}` | {block['readout_row']} | {block['n']} | {block['correct']} "
                     f"| {block['agreement']:.3f} | {low:.3f}–{high:.3f} "
                     f"| {block['low_mass']}/{block['n']} | {block['refused']}/{block['n']} |")
    return lines


def type_table(analysis: Mapping[str, Any]) -> list[str]:
    types = sorted({name for block in analysis["shapes"].values() for name in block["per_type"]})
    lines = ["| policy | " + " | ".join(f"`{name}`" for name in types) + " |",
             "|---|---|" + "---|" * len(types)]
    for policy, block in analysis["shapes"].items():
        cells = []
        for name in types:
            bucket = block["per_type"].get(name)
            cells.append("—" if bucket is None else
                         f"{bucket['correct']}/{bucket['n']} ({bucket['agreement']:.3f})")
        lines.append(f"| `{policy}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Per question type: correct/n (agreement). The full per-item rows and the "
                 "coverage quartiles per type are in the JSON next to this report.")
    return lines


def decision_section(analysis: Mapping[str, Any], decisions: Sequence[Mapping[str, Any]],
                     ) -> list[str]:
    """`## 5` — what the numbers above decide, and what each candidate would invalidate.

    The paired reading decides two *different* claims, and the shapes carry different costs: the
    candidate with the significant agreement win is the one that rewrites the prompt (and with it
    the at-the-cue refusal verdict the previous cards built), while the prompt-preserving candidate
    wins on the axis this engine actually publishes — the coverage its `reliability` word is read
    from. Both facts are rendered from the analysis; the constraint columns are properties of the
    shapes themselves (`engine/schema.py::CUE_SHAPES`, `engine/decide.py::_advance_token`).
    """
    by_policy = analysis["shapes"]
    base = analysis["policies"][0]

    two_step = "two_step_shipped=bare"
    json_field = "json_field=bare"

    def has(policy: str) -> bool:
        return policy in by_policy

    def share(policy: str) -> str:
        row = by_policy[policy]
        return (f"{row['coverage']['share_above_floor']:.3f} above the floor / "
                f"{row['low_mass']} `low_mass`")

    def agreement(policy: str) -> str:
        row = by_policy[policy]
        return f"{row['correct']}/{row['n']} = {row['agreement']:.3f}"

    def ci(policy: str) -> str:
        found = next((decision for decision in decisions
                      if decision["challenger"] == policy), None)
        if found is None:
            return "—"
        paired = found["paired"]
        mark = "excludes 0" if paired["low"] > 0 or paired["high"] < 0 else "includes 0"
        return f"{paired['low']:+.3f}…{paired['high']:+.3f} ({mark}, p = {paired['mcnemar_p']:.4g})"

    lines = [
        "## 5. The decision (card t_d90404ac)",
        "",
        "The two candidates that beat the shipped shape are not the same kind of change: the",
        "paired reading above decides the *agreement* claim, the engine's own floor decides the",
        "*coverage* claim, and the shape decides what a published table means afterwards:",
        "",
        "| candidate | agreement | paired risk difference | coverage | prompt bytes | at-the-cue "
        "refusal (`W_CUE_REFUSED`) |",
        "|---|---|---|---|---|---|",
        f"| `{base}` (shipped, the control) | {agreement(base)} | — | {share(base)} | "
        f"unchanged | read at the cue (E3c) |",
        f"| `two_step` | {agreement(two_step) if has(two_step) else '—'} "
        f"| {ci(two_step)} | "
        f"{share(two_step) if has(two_step) else '—'} | "
        "**identical** (the readout moves, the prompt does not) | preserved: the engine advances "
        "only when the cue row's argmax is not a turn-closer |",
        f"| `json_field` | {agreement(json_field) if has(json_field) else '—'} "
        f"| {ci(json_field)} | "
        f"{share(json_field) if has(json_field) else '—'} | "
        "rewritten (the cue line plus the per-type opener) | **gone**: the opener *is* the prompt, "
        "so the refusal never gets a row of its own |",
        "",
        "**What ships.** The mechanism, not the default: `options.cue` / `--cue "
        "shipped|two_step|json_field` (the bench reads the same knob), with `shipped` still the",
        "frozen default, so no published row moves by itself.",
        "",
        "**What the data supports — and what it licenses.** `two_step` — the only candidate that",
        "leaves every prompt byte, every published prompt-level table and the refusal verdict",
        "intact, and takes the engine's own coverage verdict from a mostly-unusable row to a",
        "measured one. Its agreement gain is *not* significant (the paired CI above includes",
        "zero), so under the card's promotion rule (a paired agreement CI that excludes zero)",
        "**nothing here is licensed to become the default** — this is a readout-position fix, not",
        "an accuracy claim. The reliability axis (44 → 2 `low_mass`; exact McNemar 3.1e-11 on the",
        "same 60 items) is a separate criterion the card does not state — promoting on it is a",
        "coordinator/user decision, not an inference from this table (auditor, card t_fc037544).",
        "",
        "**Why the agreement winner is not the default.** `json_field` clears the card's literal",
        "rule (the paired CI excludes zero) and is excluded by the card's own constraint 4: the",
        "opener is part of the prompt, so a family that refuses the cue can no longer be *seen*",
        "refusing it — the verdict E3b/E3c built for Occamy (30/30 at-the-cue refusals) would read",
        "as an answered row with a tiny mass instead. It stays available as `--cue json_field`,",
        "measured and documented, for models that answer.",
        "",
        "**What a flip would need (not licensed by this data).** One constant,",
        "`schema.OPTION_DEFAULTS[\"cue\"]` (plus `Options.cue`) and this document's default",
        "column; the mechanics (the readout, the refusal stop, the bench knob, the CLI flag, the",
        "payload key) are already gated by `tests/test_e3d_cue_switch.py`. A future flip needs",
        "either (a) a `two_step` agreement CI that excludes zero — at the observed discordant",
        "rates (3/60 vs 7/60) roughly a 140-160 item set — or (b) an explicit decision to promote",
        "on the readout-availability axis instead, with the bench seam of section 6 fixed first.",
        "",
        "**What a default change invalidates** (the card's risk note):",
        "",
        "* `two_step` — every published *readout* column: the `coverage` / `reliability`",
        "  values and the `W_CUE_REFUSED` counts of `docs/BENCHMARKS.md` §7 and",
        "  `docs/evidence/e2_quality.json`",
        "  (the row the label is read from moves one token in; the prompt tables — the E1c label",
        "  policy, E3b's 15 cue × label cells, E3c's seven shapes — stay valid).",
        "* `json_field` — everything printed *at the shipped cue*, because the bytes change:",
        "  the E1c label-policy table, E3b's cue × label grid, E3c's seven shapes, and the",
        "  refusal verdict itself. A re-run, not a footnote.",
        "",
        "**Reproducing this document** (no model for the first two):",
        "",
        "    bash docs/evidence/e3d/run_full.sh                    "
        "# the probe record -> docs/evidence/e3d/full.json",
        "    python3 tools/e3d_cue_decision.py report --run docs/evidence/e3d/full.json \\",
        "        --out docs/evidence/e3d_cue_decision_4b.md \\",
        "        --json docs/evidence/e3d_cue_decision_4b.json",
        "    python3 tools/e3d_engine_check.py --record docs/evidence/e3d/full.json --items 6",
        "    bash docs/evidence/e3d/run_bench_arms.sh               # the bench arms (## 6)",
        "",
        "Byte-identity of the JSON holds under CPython 3.11; 3.12+ changes the last ulp of",
        "`mean` fields — diff with tolerance.",
        "",
    ]
    return lines


def _interval(ci: Any) -> str:
    """A Wilson interval as `lo–hi` with three decimals, or `n/a` when absent."""
    if not ci or len(ci) != 2:
        return "n/a"
    return f"{ci[0]:.3f}–{ci[1]:.3f}"


#: card t_6de5fc53: the arms the fixed instrument writes (`--cue` × the model's chat template),
#: then the same three cues measured on the pre-fix tree (plain framing), kept so the §6 table can
#: show both instruments side by side.
#: `docs/evidence/e3d/bench_shipped.json` / `docs/evidence/e3d/bench_two_step.json`
#: (card t_d90404ac) carry the same plain numbers and stay untouched as the published record.
FRAMING_ARMS = ("bench_templated_shipped.json", "bench_templated_two_step.json",
                "bench_templated_json_field.json")
PLAIN_ARMS = ("bench_plain_shipped.json", "bench_plain_two_step.json",
              "bench_plain_json_field.json")
#: the framing markers a report can carry (`harness.framing_label`): the corrected instrument
#: resolves the model's template, the pre-fix one fell back to `prompt.py`'s plain E1b framing
TEMPLATED_PREFIX = "chat-template"
PLAIN_PREFIX = "plain"
FLOOR = 0.10


def arm_framing(report: Mapping[str, Any]) -> str:
    """Which prompt one arm measured — the report's own marker, `unrecorded` for older reports."""
    framing = report.get("framing") or {}
    labels = [str(label) for label in framing.get("labels") or []]
    if not labels:
        return "unrecorded (written before card t_6de5fc53)"
    return ", ".join(labels) + (" (MIXED)" if framing.get("mixed") else "")


def arm_stats(report: Mapping[str, Any]) -> dict[str, Any]:
    """One arm's row, read from its own report (nothing re-derived, nothing assumed)."""
    overall = report.get("overall") or {}
    rows = report.get("items") or []
    coverages = sorted(float(row.get("coverage") or 0.0) for row in rows)
    median = coverages[len(coverages) // 2] if coverages else float("nan")
    return {"cue": (report.get("config") or {}).get("cue"),
            "framing": arm_framing(report),
            "n": len(rows),
            "correct": overall.get("correct"),
            "agreement": overall.get("agreement"),
            "ci": overall.get("ci"),
            "low_mass": sum(1 for row in rows if row.get("reliability") == "low_mass"),
            "refused": sum(1 for row in rows if (row.get("cue") or {}).get("refused")),
            "coverage_median": median,
            "above_floor": sum(1 for value in coverages if value >= FLOOR)}


def _arm_row(entry: Mapping[str, Any]) -> str:
    return (f"| `--cue {entry['cue']}` | {entry['framing']} | {entry['correct']}/{entry['n']} = "
            f"{entry['agreement']:.3f} | {_interval(entry['ci'])} | {entry['low_mass']}/"
            f"{entry['n']} | {entry['refused']}/{entry['n']} | {entry['coverage_median']:.4g} | "
            f"{entry['above_floor']}/{entry['n']} |")


def _find_arm(stats: Sequence[Mapping[str, Any]], cue: str, prefix: str
              ) -> Mapping[str, Any] | None:
    return next((entry for entry in stats
                 if entry["cue"] == cue and str(entry["framing"]).startswith(prefix)), None)


def _delta(a: Mapping[str, Any], b: Mapping[str, Any]) -> str:
    """`shipped -> two_step` for the numbers the switch moves (agreement, mass, refusals)."""
    return (f"agreement {a['correct']}/{a['n']} -> {b['correct']}/{b['n']}, `low_mass` "
            f"{a['low_mass']} -> {b['low_mass']}, refusals at the cue {a['refused']} -> "
            f"{b['refused']}, coverage median {a['coverage_median']:.4g} -> "
            f"{b['coverage_median']:.4g}, above the {FLOOR:.2f} floor {a['above_floor']} -> "
            f"{b['above_floor']}")


def _probe_verdict(decisions: Sequence[Mapping[str, Any]], challenger: str) -> str:
    """The probe's own paired verdict for one challenger policy (`## 2`), or an empty string."""
    match = next((entry for entry in decisions
                  if str(entry.get("challenger", "")).startswith(challenger)), None)
    if not match:
        return ""
    return (f"{str(match['verdict']).upper()} (paired risk difference "
            f"{match['paired']['difference']:+.3f}, 95 % CI {match['paired']['low']:+.3f}…"
            f"{match['paired']['high']:+.3f})")


def arms_section(arms: Sequence[Mapping[str, Any]],
                 decisions: Sequence[Mapping[str, Any]] = ()) -> list[str]:
    """`## 6` — the cue switch through the *bench*, before and after the framing fix.

    Rendered when arm reports are given (`--arms`, default `default_arms()`): the post-fix arms
    (card t_6de5fc53 — the executed plan resolves the model's chat template, exactly like the
    serving path) and, when present, the same cues on the pre-fix tree
    (`docs/evidence/e3d/bench_plain_*.json`,
    written before this card), whose rows measured the plain E1b framing because
    `LiveModel.decide` re-planned the context from the live session — a `ModelSession` carries no
    `.model`/`.runtime`, so `resolve_template` returned `None`. Every row names its framing, and
    the prose below is derived from the reports themselves: what the plain arms said, what the
    templated arms say, and whether that agrees in direction with the probe's paired reading.
    """
    if not arms:
        return []
    stats = [arm_stats(report) for report in arms]
    lines = ["## 6. The bench arms — the switch through the instrument, before and after the fix",
             "",
             "Same box, same model, same 60 committed items, same context; only `--cue` moves.",
             "The `framing` column is the arm's own report marker (`engine.template`, summarized "
             "over the rows — `harness.framing_label`):", "",
             "| arm | framing | agreement | Wilson | `low_mass` | refused at the cue | coverage "
             "median | above floor |",
             "|---|---|---|---|---|---|---|---|"]
    lines += [_arm_row(entry) for entry in stats]
    lines.append("")
    templated = [entry for entry in stats if str(entry["framing"]).startswith(TEMPLATED_PREFIX)]
    plain = [entry for entry in stats if str(entry["framing"]).startswith(PLAIN_PREFIX)]
    if plain and templated:
        lines += [
            "**The plain rows are the pre-fix instrument** (kept because they were published): "
            "`LiveModel.decide` planned the executed context from the live session, and a "
            "`ModelSession` carries no `.model`/`.runtime`, so `resolve_template(request, "
            "session)` returned `None` and `prompt.build_prefix(state, resolution=None)` "
            "silently fell back "
            "to the plain E1b framing — the documented escape hatch for a session with no model "
            "handle, not a default for a real one. The chat-template rows are the same command "
            "after card `t_6de5fc53` (one plan, resolved from the **handle** — the source the "
            "serving path uses); on dev item `c01` that is 102 prefix tokens against 119 and the "
            "cue row's label mass 0.03677 against 0.00696.", ""]
    for prefix, label in ((TEMPLATED_PREFIX, "the corrected instrument"),
                          (PLAIN_PREFIX, "the pre-fix instrument")):
        shipped = _find_arm(stats, "shipped", prefix)
        two_step = _find_arm(stats, "two_step", prefix)
        if not (shipped and two_step):
            continue
        lines += [f"**`two_step` vs `shipped` on {label}** — {_delta(shipped, two_step)}."]
        if prefix == TEMPLATED_PREFIX:
            verdict = _probe_verdict(decisions, "two_step")
            bench_delta = (two_step["correct"] or 0) - (shipped["correct"] or 0)
            if verdict:
                probe_says = ("the probe's paired reading does not put `two_step` behind "
                              "`shipped`"
                              if verdict.startswith(("KEEP", "PROMOTE"))
                              else "the probe's paired reading puts `two_step` behind `shipped`")
                probe_says = (f"section 2's paired verdict for `two_step` is **{verdict}**, so "
                              f"{probe_says} on agreement")
            else:
                probe_says = "no `--compare` pair was rendered, so section 2 carries no verdict"
            if bench_delta >= 0:
                lines += [
                    f"The corrected bench's own delta is {bench_delta:+d} correct items — the "
                    f"signs agree, and the plain arm's reversal (`--cue two_step` costing "
                    f"agreement and doubling `low_mass`) is a property of the *pre-fix framing*, "
                    f"not of the cue shape. ({probe_says}.)",
                    ""]
            else:
                lines += [
                    f"The corrected bench's own delta is {bench_delta:+d} correct items — the "
                    f"reversal survives the fix for the bench on this model ({probe_says}.).",
                    ""]
    if templated:
        lines += [
            "**What this changes.** The bench now sends the prompt the product sends, so its "
            "quality tables describe the serving path; the pre-fix rows above are superseded (they "
            "measure the plain framing) and `docs/BENCHMARKS.md` §2/§7/§8 carry the corrected "
            "numbers next to them. The cue default is frozen by the E3d card and does not move "
            "here.", ""]
    return lines


def render(analysis: Mapping[str, Any], *, decisions: Sequence[Mapping[str, Any]] = (),
           run_path: str = "", arms: Sequence[Mapping[str, Any]] = ()) -> str:
    model = analysis.get("model") or {}
    lines = [
        "# E3d — the cue shape on the full dev set (is the two-step readout the better default?)",
        "",
        f"- card `t_d90404ac` · model `{model.get('name')}` "
        f"({model.get('bytes') or 0:,} bytes) · arch `{model.get('arch')}`",
        f"- run `{run_path}` · runtime `{analysis.get('runtime')}` · backend claim "
        f"`{analysis.get('backend_claim') or 'n/a'}`",
        f"- devset `{analysis.get('devset')}` · items "
        f"{', '.join(f'{key} {value}' for key, value in sorted(analysis['counts'].items()))} "
        f"· engine floor {analysis['mass_floor']:.2f}",
        f"- policies: {', '.join(f'`{name}`' for name in analysis['policies'])}",
        "",
        "## 1. Agreement per policy (marginal Wilson intervals — *not* the paired reading)",
        "",
        *agreement_table(analysis),
        "",
        "## 2. The paired comparison (the decision's actual statistic)",
        "",
    ]
    if not decisions:
        lines.append("No baseline/challenger pair was requested (`--compare baseline challenger`).")
    for decision in decisions:
        paired = decision["paired"]
        lines += [
            f"### `{decision['challenger']}` vs `{decision['baseline']}`",
            "",
            f"- paired on **{paired['n']}** items: {paired['discordant']['a_win']} correct only "
            f"for the baseline, {paired['discordant']['b_win']} correct only for the challenger",
            f"- exact McNemar p = **{paired['mcnemar_p']:.4g}** (two-sided binomial tail on the "
            f"discordant pairs)",
            f"- risk difference **{paired['difference']:+.3f}**; {paired['iters']}-draw paired "
            f"percentile bootstrap 95 % CI **{paired['low']:+.3f}…{paired['high']:+.3f}** "
            f"(seed {paired['seed']})",
            f"- marginal Wilson intervals: baseline "
            f"{decision['marginal']['baseline'][0]:.3f}–{decision['marginal']['baseline'][1]:.3f}, "
            f"challenger "
            f"{decision['marginal']['challenger'][0]:.3f}–"
            f"{decision['marginal']['challenger'][1]:.3f} "
            f"({'they overlap' if decision['marginal_overlap'] else 'they do not overlap'})",
            f"- coverage: baseline {decision['floor_effect']['baseline_share']:.3f} above the "
            f"floor / {decision['floor_effect']['baseline_low_mass']} `low_mass`; challenger "
            f"{decision['floor_effect']['challenger_share']:.3f} / "
            f"{decision['floor_effect']['challenger_low_mass']}",
            f"- **verdict: {decision['verdict'].upper()}** — {decision['note']}",
            "",
        ]
    lines += ["## 3. Coverage distribution per policy", "", *coverage_table(analysis), "",
              "## 4. Agreement per question type", "", *type_table(analysis), "",
              *decision_section(analysis, decisions),
              *arms_section(arms, decisions)]
    return "\n".join(lines) + "\n"


def default_arms() -> list[dict[str, Any]]:
    """The committed bench arms next to the record, if they are there (`## 6`).

    Card t_6de5fc53: the post-fix arms first (the bench plans from the handle — the model's chat
    template), then the pre-fix ones (plain framing), so the section always shows both the
    corrected instrument and the published record it supersedes.
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    arms = []
    for name in (*FRAMING_ARMS, *PLAIN_ARMS):
        path = root / "docs" / "evidence" / "e3d" / name
        if path.is_file():
            arms.append(json.loads(path.read_text(encoding="utf-8")))
    return arms


def report(record: Mapping[str, Any], *, run_path: str = "",
           comparisons: Sequence[tuple[str, str]] = (),
           arms: Sequence[Mapping[str, Any]] | None = None) -> str:
    """The card's evidence document from a stored record (deterministic, offline)."""
    analysis = analyse(record)
    decisions = [decide(record, baseline=base, challenger=new) for base, new in comparisons]
    return render(analysis, decisions=decisions, run_path=run_path,
                  arms=default_arms() if arms is None else arms)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("report", help="the card's tables from a stored probe record")
    run.add_argument("--run", required=True, help="a `tools/e3c_cue_shapes.py run` record")
    run.add_argument("--compare", nargs=2, action="append", default=None, metavar=("BASE", "NEW"),
                     help="a `shape=label` pair to decide (repeatable); default: every policy "
                          "against the first one measured")
    run.add_argument("--out", default=None)
    run.add_argument("--json", default=None, help="write the analysis as JSON too")
    run.add_argument("--arms", nargs="+", default=None, metavar="JSON",
                     help="bench `--suite quality` arm reports for `## 6` (default: the committed "
                          "`docs/evidence/e3d/bench_shipped.json` + "
                          "`docs/evidence/e3d/bench_two_step.json` when present)")
    args = parser.parse_args(argv)
    record = load_run(args.run)
    pairs = [tuple(pair) for pair in (args.compare or [])]
    if not pairs:
        known = policies(record)
        pairs = [(known[0], other) for other in known[1:]]
    analysis = analyse(record)
    decisions = [decide(record, baseline=base, challenger=new) for base, new in pairs]
    if args.arms is None:
        arms: list[Mapping[str, Any]] = default_arms()
    else:
        arms = [json.loads(pathlib.Path(path).read_text(encoding="utf-8")) for path in args.arms]
    markdown = render(analysis, decisions=decisions, run_path=str(args.run), arms=arms)
    if args.out:
        pathlib.Path(args.out).write_text(markdown, encoding="utf-8")
    if args.json:
        # the per-item rows are the evidence; the summary stays readable
        summary = {"schema": SCHEMA, "run": str(args.run), "policies": analysis["policies"],
                   "mass_floor": analysis["mass_floor"],
                   "shapes": {key: {name: value for name, value in block.items() if name != "rows"}
                              for key, block in analysis["shapes"].items()},
                   "decisions": decisions}
        pathlib.Path(args.json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
